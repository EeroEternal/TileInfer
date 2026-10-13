"""Pure-torch MoE oracle (same role as ``reference.py`` for attention).

It is the correctness reference for the fused grouped-GEMM kernels in
:mod:`tileinfer.kernels.moe` and the executable specification of the *slot
layout* the kernels consume.

Shapes (Qwen3.6-35B-A3B, 1×Ascend950, TP=1)::

    x        [num_tokens, hidden]          bf16
    w13      [num_experts, 2*inter, hidden] bf16   (rows 0..inter-1 = gate, inter..2*inter-1 = up)
    w2       [num_experts, hidden, inter]   bf16
    topk_ids     [num_tokens, top_k]  int   expert index per (token, slot)
    topk_weights [num_tokens, top_k]  float

The *slot layout* expands tokens to ``num_tokens * top_k`` rows, sorts them by
expert and pads every expert's tile group to ``BM`` rows (``sorted_token_ids``
holds the expanded row index per slot, a sentinel ``num_expanded`` for padding;
``expert_ids`` holds one expert per padded tile).  Nothing here is
Ascend-specific and nothing is meant to be fast; everything runs in fp32.
"""

from __future__ import annotations

import torch

__all__ = [
    "reference_moe",
    "reference_gate_up_swiglu_slots",
    "reference_down_slots",
    "build_slot_layout",
]


def reference_moe(
    x: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    """Straight-line oracle: routed experts with SwiGLU, fp32, weighted combine.

    ``out[t] = Σ_k weight[t,k] · ( silu(x[t]@w13[e,0:i].T) * (x[t]@w13[e,i:2i].T) ) @ w2[e].T``
    with ``e = topk_ids[t,k]``.  Matches ``npu_grouped_matmul`` + ``npu_swiglu`` semantics
    (gate = first half of the w13 output, up = second half).
    """
    tokens, hidden = x.shape
    experts, two_i, _ = w13.shape
    inter = two_i // 2
    top_k = topk_ids.shape[1]
    xf = x.float()
    out = torch.zeros((tokens, hidden), dtype=torch.float32, device=x.device)
    gate_w, up_w = w13[:, :inter, :].float(), w13[:, inter:, :].float()
    for t in range(tokens):
        for k in range(top_k):
            e = int(topk_ids[t, k])
            h = xf[t]
            g = h @ gate_w[e].T
            u = h @ up_w[e].T
            act = g * torch.sigmoid(g) * u
            out[t] = out[t] + float(topk_weights[t, k]) * (act @ w2[e].float().T)
    return out


def build_slot_layout(
    topk_ids: torch.Tensor,
    block_m: int,
    num_experts: int | None = None,
) -> dict:
    """Expand ``topk_ids`` into the padded slot layout the MoE kernels consume.

    Returns a dict with:
      ``sorted_ids``   [EM*BM] int32 — expanded row index per slot, ``num_expanded`` for pad
      ``expert_ids``   [EM]    int32 — expert of each tile (pad tiles carry expert 0)
      ``x_slots``      [EM*BM, hidden] — gathered input rows (pad slots point at row 0's data
                       but are never written back, so their content is irrelevant)
      ``counts``       [E] int64 — rows per expert (host view, for compact-layout baselines)
      ``num_expanded`` int — R = num_tokens * top_k
    """
    num_tokens, top_k = topk_ids.shape
    num_expanded = num_tokens * top_k
    flat = topk_ids.reshape(-1).to(torch.int64)
    counts = torch.bincount(flat, minlength=num_experts or 0)
    order = torch.argsort(flat, stable=True)
    sorted_experts = flat[order]
    tiles_per_expert = (counts + block_m - 1) // block_m
    em = int(tiles_per_expert.sum())
    sorted_ids = torch.full((em * block_m,), num_expanded, dtype=torch.int32, device=topk_ids.device)
    expert_ids = torch.zeros((em,), dtype=torch.int32, device=topk_ids.device)
    cursor = 0
    counts_list = counts.tolist()
    for e in range(int(counts.shape[0])):
        n = counts_list[e]
        n_tiles = (n + block_m - 1) // block_m
        if n_tiles == 0:
            continue
        mask = sorted_experts == e
        rows = order[mask]
        sorted_ids[cursor * block_m : cursor * block_m + n] = rows.to(torch.int32)
        expert_ids[cursor : cursor + n_tiles] = e
        cursor += n_tiles
    x_slots = None  # built by the caller from sorted_ids (clamp + index_select)
    return {
        "sorted_ids": sorted_ids,
        "expert_ids": expert_ids,
        "num_tiles": em,
        "num_expanded": num_expanded,
        "counts": counts,
        "x_slots": x_slots,
    }


def gather_slots(x_expanded: torch.Tensor, sorted_ids: torch.Tensor, num_expanded: int) -> torch.Tensor:
    """Gather expanded rows into slot order; pad slots point at row 0 (never read for output)."""
    clamped = torch.where(
        sorted_ids >= num_expanded, torch.zeros_like(sorted_ids), sorted_ids
    ).to(torch.int64)
    return x_expanded.index_select(0, clamped)


def reference_gate_up_swiglu_slots(
    x_slots: torch.Tensor,
    w13: torch.Tensor,
    expert_ids: torch.Tensor,
    sorted_ids: torch.Tensor,
    num_expanded: int,
    block_m: int,
) -> torch.Tensor:
    """Oracle for the fused gate+up+SwiGLU kernel, slot-for-slot in fp32."""
    rows, hidden = x_slots.shape
    inter = w13.shape[1] // 2
    y = torch.zeros((rows, inter), dtype=torch.float32, device=x_slots.device)
    for tile in range(int(expert_ids.shape[0])):
        e = int(expert_ids[tile])
        w_g = w13[e, :inter, :].float()
        w_u = w13[e, inter:, :].float()
        for r in range(tile * block_m, min((tile + 1) * block_m, rows)):
            if int(sorted_ids[r]) >= num_expanded:
                continue
            h = x_slots[r].float()
            g = h @ w_g.T
            u = h @ w_u.T
            y[r] = g * torch.sigmoid(g) * u
    return y


def reference_down_slots(
    y_slots: torch.Tensor,
    w2: torch.Tensor,
    expert_ids: torch.Tensor,
    sorted_ids: torch.Tensor,
    num_expanded: int,
    block_m: int,
) -> torch.Tensor:
    """Oracle for the down-projection kernel, slot-for-slot in fp32."""
    rows, inter = y_slots.shape
    hidden = w2.shape[1]
    out = torch.zeros((rows, hidden), dtype=torch.float32, device=y_slots.device)
    for tile in range(int(expert_ids.shape[0])):
        e = int(expert_ids[tile])
        w = w2[e].float()
        for r in range(tile * block_m, min((tile + 1) * block_m, rows)):
            if int(sorted_ids[r]) >= num_expanded:
                continue
            out[r] = y_slots[r].float() @ w.T
    return out
