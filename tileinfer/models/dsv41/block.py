"""Block / MoE / Hyper-Connection / full-stack forward for DeepSeek-V4.1.

Mirrors ``inference/model.py``'s ``Block``, ``MoE``, ``Gate``, ``Expert``, ``Engram`` and
``Transformer`` step by step, on top of :mod:`tileinfer.models.dsv41.layer` (attention) and the torch
kernel set (:mod:`tools.dsv41.ref_cpu`).  The point of this module is S3: a whole-stack forward that
can be compared against the official implementation token by token.

Fidelity notes:

* ``y = torch.zeros_like(x, dtype=torch.float32)`` then ``y += expert(...)``: the reference accumulates
  MoE output in fp32 and casts once at the end, so a fp32 accumulator is part of the semantics.
* Routing follows ``Gate.forward`` exactly: ``sqrtsoftplus(scores)``, the correction bias only steers
  *selection* (``scores + bias``), the weights come from the unbiased scores, ``norm_topk_prob``
  normalizes by ``sum + 1e-20`` and ``route_scale`` multiplies afterwards.
* Hyper-Connections keep the reference's cross-sublayer timing: the mix this sublayer computes is
  consumed by the *next* one, and the block returns ``ffn_pre`` for the next block's attention.
* ``pre_mix`` starts as a one-hot on copy 0 (``make_identity_pre_mix``).
"""

from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .config import Dsv41Config
from .layer import (
    LayerRuntime,
    SharedAttentionState,
    attention_forward,
    ops,
    rms_norm,
)
from .modules import (
    Dsv41Block,
    Dsv41Engram,
    Dsv41Model,
    Dsv41MoE,
    Dsv41RMSNorm,
)

#: Diagnostic only (`VLLM_DSV41_MOE_HITSTATS=1`): how many of a step's experts were also used by the
#: previous step at the same layer.  This decides whether a dequantised-expert cache can pay off.
_PREV_USED: Dict[int, set] = {}
_HIT_ACC = {"used": 0, "hit": 0}
_MOE_ACC: Dict[str, float] = {}


def _moe_sync() -> None:
    try:
        torch.npu.synchronize()
    except Exception:  # noqa: BLE001
        pass


def _moe_mark(key: str, t0: float) -> float:
    import time as _t

    _moe_sync()
    now = _t.perf_counter()
    _MOE_ACC[key] = _MOE_ACC.get(key, 0.0) + (now - t0)
    return now


def _hitstats(layer_id: int, used) -> None:
    prev = _PREV_USED.get(layer_id)
    if prev is not None:
        _HIT_ACC["used"] += len(used)
        _HIT_ACC["hit"] += len(set(used) & prev)
    _PREV_USED[layer_id] = set(used)
    if layer_id == 39:
        u, h = _HIT_ACC["used"], _HIT_ACC["hit"]
        print(
            "DSV41_MOE_HITSTATS used=%d hit=%d rate=%.3f" % (u, h, h / max(u, 1)),
            flush=True,
        )
        _HIT_ACC["used"] = 0
        _HIT_ACC["hit"] = 0

__all__ = [
    "ModelState",
    "hc_mixes",
    "hc_pre",
    "hc_post",
    "make_identity_pre_mix",
    "expert_forward",
    "moe_forward",
    "engram_forward",
    "block_forward",
    "transformer_forward",
    "logits_from_hidden",
]


# =============================================================================================
# Hyper-Connections
# =============================================================================================


def make_identity_pre_mix(x: torch.Tensor, hc_mult: int) -> torch.Tensor:
    """``model.py:make_identity_pre_mix`` -- one-hot on copy 0."""
    pre_mix = x.new_zeros(x.size(0), x.size(1), hc_mult, dtype=torch.float32)
    pre_mix[:, :, 0] = 1.0
    return pre_mix


def hc_mixes(block: Dsv41Block, x: torch.Tensor, hc_fn, hc_scale, hc_base):
    """``Block.hc_mixes`` -- pre / post / comb coefficients from the flattened residual stream."""
    o = ops()
    flat = x.flatten(2).float()
    rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + block.norm_eps)
    mixes = F.linear(flat, hc_fn.float()) * rsqrt
    pre, post, comb = o.hc_split_sinkhorn(
        mixes,
        hc_scale.float(),
        hc_base.float(),
        block.hc_mult,
        block.hc_sinkhorn_iters,
        block.hc_eps,
    )
    return pre, post, comb


def hc_pre(x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
    """``Block.hc_pre``: ``sum_c pre_mix[c] * x[c]`` -- ``[b,s,hc,d] -> [b,s,d]``."""
    y = torch.sum(pre_mix.unsqueeze(-1) * x.float(), dim=2)
    return y.to(x.dtype)


def hc_post(
    x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor
) -> torch.Tensor:
    """``Block.hc_post``: ``post * x + comb @ residual`` -- ``[b,s,d] -> [b,s,hc,d]``."""
    y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(
        comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2
    )
    return y.type_as(x)


# =============================================================================================
# MoE
# =============================================================================================


def expert_forward(expert, x: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``model.py:Expert.forward`` -- SwiGLU with the training clamps."""
    o = ops()
    dtype = x.dtype
    limit = expert.swiglu_limit
    gate = o.linear(x, expert.w1.weight, expert.w1.scale).float()
    up = o.linear(x, expert.w3.weight, expert.w3.scale).float()
    if limit > 0:
        up = torch.clamp(up, min=-limit, max=limit)
        gate = torch.clamp(gate, max=limit)
    out = F.silu(gate) * up
    if weights is not None:
        out = weights * out
    return o.linear(out.to(dtype), expert.w2.weight, expert.w2.scale)


def gate_forward(gate, x: torch.Tensor, image_mask: Optional[torch.Tensor] = None):
    """``model.py:Gate.forward`` -- bias picks experts, scores give the weights."""
    scores = F.linear(x.float(), gate.weight.float()) / getattr(gate, "gate_temp", 1.0)
    if gate.score_func == "softmax":
        scores = scores.softmax(dim=-1)
    elif gate.score_func == "sigmoid":
        scores = scores.sigmoid()
    else:
        scores = F.softplus(scores).sqrt()
    bias = gate.bias
    if image_mask is not None and gate.bias_vl is not None:
        bias = torch.where(image_mask.unsqueeze(-1), gate.bias_vl, bias)
    indices = (scores + bias).topk(gate.topk, dim=-1)[1]
    weights = scores.gather(1, indices)
    if gate.norm_topk_prob and gate.topk > 1:
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    weights = weights * gate.route_scale
    return weights, indices


def moe_forward(
    moe: Dsv41MoE, x: torch.Tensor, image_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """``model.py:MoE.forward`` -- per-expert gather plus the always-on shared expert."""
    shape = x.size()
    flat = x.view(-1, moe.gate.weight.size(-1))
    mask = None if image_mask is None else image_mask.flatten()
    weights, indices = gate_forward(moe.gate, flat, mask)
    y = torch.zeros_like(flat, dtype=torch.float32)
    counts = torch.bincount(indices.flatten(), minlength=moe.n_routed_experts).tolist()
    for i in range(moe.n_routed_experts):
        if counts[i] == 0:
            continue
        expert = moe.experts[i]
        idx, top = torch.where(indices == i)
        y[idx] += expert_forward(expert, flat[idx], weights[idx, top, None]).float()
    y = y + expert_forward(moe.shared_experts, flat).float().cpu()
    return y.type_as(x).to(x.device).view(shape)


def moe_forward_streamed_grouped(
    moe: Dsv41MoE,
    x: torch.Tensor,
    layer_id: int,
    arena,
    device: str = "npu:0",
    image_mask: Optional[torch.Tensor] = None,
    swiglu_limit: float = 0.0,
) -> torch.Tensor:
    """Routed MoE with the experts of one arena group folded into a **single batched GEMM**.

    The per-expert path calls ``ref_cpu.linear`` three times per expert (w1, w3, w2), i.e. one
    unpack + one block-scale broadcast + one matmul each.  Here the group's experts are stacked so
    that ``w1`` and ``w3`` for *all* of them are dequantized by one ``_fp4_linear_batched`` call:

    * inputs: one row-block per expert, zero-padded to ``max_n`` (padding rows quantize to exact
      zeros, so they contribute nothing);
    * weights: ``[E, 2*out, in/2]`` by stacking the ``uint8`` views (torch_npu cannot ``stack`` fp4
      or e8m0 dtypes directly);
    * scales: ``[E, 2*out, in/32]`` the same way.

    ``w2`` stays per expert because its input width (``moe_intermediate_size``) differs.  Numerics
    are identical to the per-expert path for the same reason the batched decode arm is: every row
    is an independent GEMM against the same dequantized weights.

    Opt-in via ``DSV41_MOE_GROUPED=1``; the default path is untouched.
    """
    from tools.dsv41 import ref_cpu

    import time as _time

    _tm = os.environ.get("VLLM_DSV41_MOE_TIMING", "") not in ("", "0")
    _t0 = _time.perf_counter() if _tm else 0.0

    shape = x.size()
    flat = x.view(-1, moe.gate.weight.size(-1))
    mask = None if image_mask is None else image_mask.flatten()
    weights, indices = gate_forward(moe.gate, flat, mask)
    if _tm:
        _t0 = _moe_mark("gate", _t0)

    xd = flat.to(device)
    in_f = int(xd.shape[-1])
    out_f = int(moe.experts[0].w1.weight.shape[0])
    y = torch.zeros(flat.shape, dtype=torch.float32, device=device)

    # Route bookkeeping on the host: a per-expert `torch.where` on the device re-syncs for every
    # expert (the gather's output shape is data-dependent).  At 4 rows x 6 experts this is ~10 ms per
    # layer of pure launch/sync overhead; the 24 routing pairs are tiny, so move them once.
    n_route = int(indices.numel())
    packed = torch.cat([indices.reshape(-1), weights.reshape(-1).float()]).cpu()
    indices_h = packed[:n_route].reshape(indices.shape).tolist()
    weights_h = packed[n_route:].reshape(weights.shape).tolist()
    per_expert: Dict[int, List[Tuple[int, int]]] = {}
    for r, row in enumerate(indices_h):
        for t, e in enumerate(row):
            per_expert.setdefault(int(e), []).append((r, t))
    used = sorted(per_expert)
    groups = [used[i : i + arena.n_slots] for i in range(0, len(used), arena.n_slots)]

    if _tm:
        _t0 = _moe_mark("route", _t0)
    if os.environ.get("VLLM_DSV41_MOE_HITSTATS", "") not in ("", "0"):
        _hitstats(layer_id, used)

    for group in groups:
        slots, fresh = {}, []
        for e in group:
            slot, hit = arena.bind(layer_id, e)
            slots[e] = slot
            if not hit:
                fresh.append(slot)
            arena.release(layer_id, e)
        for e in group:
            if slots[e] in fresh:
                arena.load_slot(slots[e], moe.experts[e], device)
        if _tm:
            _t0 = _moe_mark("load", _t0)

        nmax = max(len(per_expert[e]) for e in group)
        views = {e: arena.views(slots[e], on_device=True) for e in group}
        xpad = torch.zeros((len(group), nmax, in_f), dtype=xd.dtype, device=device)
        rows_cpu: List[int] = []
        spans = {}
        for e in group:
            spans[e] = (len(rows_cpu), len(rows_cpu) + len(per_expert[e]))
            rows_cpu.extend(r for r, _t in per_expert[e])
        rows_flat = torch.as_tensor(rows_cpu, dtype=torch.long, device=device)
        rows_by_expert = {e: rows_flat[a:b] for e, (a, b) in spans.items()}
        for i, e in enumerate(group):
            rows = rows_by_expert[e]
            xpad[i, : rows.numel()] = xd.index_select(0, rows)

        w13 = torch.cat(
            [
                torch.stack([views[e]["w1"].view(torch.uint8) for e in group]),
                torch.stack([views[e]["w3"].view(torch.uint8) for e in group]),
            ],
            dim=1,
        ).view(views[group[0]]["w1"].dtype)
        s13 = torch.cat(
            [
                torch.stack([views[e]["w1.scale"].view(torch.uint8) for e in group]),
                torch.stack([views[e]["w3.scale"].view(torch.uint8) for e in group]),
            ],
            dim=1,
        ).view(views[group[0]]["w1.scale"].dtype)

        if _tm:
            _t0 = _moe_mark("stack", _t0)
        both = _fp4_linear_batched(xpad, w13, s13, 2 * out_f, in_f)   # [E, nmax, 2*out]
        if _tm:
            _t0 = _moe_mark("gemm13", _t0)
        gate_all, up_all = both[..., :out_f].float(), both[..., out_f:].float()
        wpad_cpu = [[0.0] * nmax for _ in group]
        for i, e in enumerate(group):
            for j, (r, t) in enumerate(per_expert[e]):
                wpad_cpu[i][j] = weights_h[r][t]
            limit = moe.experts[e].swiglu_limit
            if limit > 0:
                up_all[i].clamp_(min=-limit, max=limit)
                gate_all[i].clamp_(max=limit)
        wpad = torch.as_tensor(wpad_cpu, dtype=torch.float32, device=device)
        act = (torch.nn.functional.silu(gate_all) * up_all * wpad.unsqueeze(-1)).to(xd.dtype)

        # w2 for all experts of the group in one batched GEMM: the per-expert version launched (and
        # unpacked) once per expert, which is the remaining fixed cost at 4 rows x ~12 experts.
        w2o, w2i = int(views[group[0]]["w2"].shape[0]), int(views[group[0]]["w2"].shape[1]) * 2
        w2 = torch.stack([views[e]["w2"].view(torch.uint8) for e in group]).view(
            views[group[0]]["w2"].dtype
        )
        s2 = torch.stack([views[e]["w2.scale"].view(torch.uint8) for e in group]).view(
            views[group[0]]["w2.scale"].dtype
        )
        out_all = _fp4_linear_batched(act, w2, s2, w2o, w2i)
        if _tm:
            _t0 = _moe_mark("gemm2", _t0)
        for i, e in enumerate(group):
            n = len(per_expert[e])
            y.index_add_(0, rows_by_expert[e], out_all[i, :n].float())
        if _tm:
            _t0 = _moe_mark("scatter", _t0)

        if len(groups) > 1:
            for e in group:
                arena.cache.drop((layer_id, e))
    y = y + expert_forward(moe.shared_experts, flat).float().to(device)
    if _tm:
        _t0 = _moe_mark("shared", _t0)
        if layer_id == 39:
            print(
                "DSV41_MOE_TIMING "
                + " ".join(f"{k}={v * 1e3:.0f}" for k, v in sorted(_MOE_ACC.items())),
                flush=True,
            )
            for k in list(_MOE_ACC):
                _MOE_ACC[k] = 0.0
    return y.type_as(x).to(x.device).view(shape)


def _moe_forward_streamed_device_accum(
    moe: Dsv41MoE,
    x: torch.Tensor,
    layer_id: int,
    arena,
    device: str = "npu:0",
    image_mask: Optional[torch.Tensor] = None,
    swiglu_limit: float = 0.0,
) -> torch.Tensor:
    """Same maths as :func:`moe_forward_streamed`, but everything stays on the device.

    The default path keeps the accumulator in host memory (``y[idx] += out.float().cpu()``) and scans
    the routing table with ``torch.where`` on the CPU.  For a decode step that is 6 experts, but a
    prefill can touch all 384 -- i.e. 384 device-to-host copies plus 384 host-side scatter-adds per
    layer.  This variant instead

    * keeps the accumulator on the device and folds with ``index_add_`` (one D2H at the very end),
    * runs ``torch.where`` on the device so the gather indices never round-trip,
    * reads the routing weights from a device copy (the gate already produced them on the device).

    Accumulation order differs from the default path, so results agree to fp32 rounding (~1e-7),
    not bit-for-bit; the acceptance bar for numerics is the first token (`11111`) and the S2 block
    parity (<= 1e-2), both far above that.
    """
    import os as _os

    from tools.dsv41 import ref_cpu

    # w1 (gate) and w3 (up) dequantized in one call instead of two; opt-in so the arms can be A/B'd
    merge_w13 = _os.environ.get("DSV41_MOE_W13", "") not in ("", "0")
    shape = x.size()
    flat = x.view(-1, moe.gate.weight.size(-1))
    mask = None if image_mask is None else image_mask.flatten()
    weights, indices = gate_forward(moe.gate, flat, mask)

    used = sorted({int(e) for e in indices.flatten().tolist()})
    groups = [used[i : i + arena.n_slots] for i in range(0, len(used), arena.n_slots)]

    xd = flat.to(device)
    indices_d = indices.to(device)
    weights_d = weights.to(device).float()
    y = torch.zeros(flat.shape, dtype=torch.float32, device=device)
    for group in groups:
        slots, fresh = {}, []
        for e in group:
            slot, hit = arena.bind(layer_id, e)
            slots[e] = slot
            if not hit:
                fresh.append(slot)
            arena.release(layer_id, e)
        for e in group:
            if slots[e] in fresh:
                arena.load_slot(slots[e], moe.experts[e], device)
        for e in group:
            idx, top = torch.where(indices_d == e)
            expert = moe.experts[e]
            limit = expert.swiglu_limit
            xsel = xd[idx]
            views = arena.views(slots[e], on_device=True)
            if merge_w13:
                # w1 (gate) and w3 (up) share the same input and are only used side by side, so their
                # packed fp4 blocks are concatenated along the output dim and dequantized once.  The
                # concatenation is done on the uint8 view: repeat/cat on the fp4 storage dtype is not
                # supported by torch_npu (same reason ``repeat`` is avoided elsewhere).
                w1, w3 = views["w1"], views["w3"]
                cat = torch.cat(
                    [w1.view(torch.uint8), w3.view(torch.uint8)], dim=0
                ).view(w1.dtype)
                scat = torch.cat(
                    [views["w1.scale"].view(torch.uint8), views["w3.scale"].view(torch.uint8)], dim=0
                ).view(views["w1.scale"].dtype)
                both = ref_cpu.linear(xsel, cat, scat)
                gate_out, up = both.split([w1.shape[0], w3.shape[0]], dim=-1)
                gate_out, up = gate_out.float(), up.float()
            else:
                gate_out = ref_cpu.linear(xsel, views["w1"], views["w1.scale"]).float()
                up = ref_cpu.linear(xsel, views["w3"], views["w3.scale"]).float()
            if limit > 0:
                up = torch.clamp(up, min=-limit, max=limit)
                gate_out = torch.clamp(gate_out, max=limit)
            out = torch.nn.functional.silu(gate_out) * up
            out = out * weights_d[idx, top, None]
            out = ref_cpu.linear(out.to(xd.dtype), views["w2"], views["w2.scale"])
            y.index_add_(0, idx, out.float())
        if len(groups) > 1:
            for e in group:
                arena.cache.drop((layer_id, e))
    y = y + expert_forward(moe.shared_experts, flat).float().to(device)
    return y.type_as(x).to(x.device).view(shape)


def moe_forward_streamed(
    moe: Dsv41MoE,
    x: torch.Tensor,
    layer_id: int,
    arena,
    device: str = "npu:0",
    image_mask: Optional[torch.Tensor] = None,
    swiglu_limit: float = 0.0,
) -> torch.Tensor:
    """``moe_forward`` with the routed experts *paged* in from the host arena (S4).

    Same maths as :func:`moe_forward`, but the six experts a token routes to are copied into
    pre-allocated device slots as **one** flat block (``ExpertArena``) instead of living on the card
    permanently.  The shared expert stays device-resident: it is used by every token, so paging it
    would only add traffic.

    ``s4_layer_moe_stream.py`` shows this is bit-identical to keeping the weights on the device, and
    that one layer's worth of paging costs ~4 ms at 44 GB/s.

    ``DSV41_MOE_FAST=1`` switches to :func:`_moe_forward_streamed_device_accum`, which keeps the
    accumulator and the index scans on the device (see the docstring there).  The default path is
    byte-for-byte unchanged, so both arms can be A/B'd with only an env var.
    """
    import os as _os

    _grouped = _os.environ.get("DSV41_MOE_GROUPED", "")
    if _grouped == "auto":
        # measured: grouped wins on multi-row (prefill) and loses on 1-row (decode) -- see
        # docs/notes/dsv41-perf-moe-ab.md §6
        _grouped = "1" if (x.numel() // x.shape[-1]) > 1 else ""
    if _grouped and _grouped != "0":
        # P4: 长 prefill 的路由倾斜会让单专家 nmax 很大，grouped 一次性构造
        # [E, nmax, ...] 会瞬时分配 30+ GiB（16K 实测 OOM）。按 seq 维切块后每块重新分组，
        # 把峰值压到可控范围；默认关闭，P4 服务显式设置 DSV41_MOE_PREFILL_CHUNK。
        _chunk = int(_os.environ.get("DSV41_MOE_PREFILL_CHUNK", "0"))
        if _chunk > 0 and x.dim() == 3 and x.shape[0] == 1 and x.shape[1] > _chunk:
            parts = [
                moe_forward_streamed_grouped(
                    moe, x[:, i : i + _chunk], layer_id, arena, device=device,
                    image_mask=None if image_mask is None else image_mask[:, i : i + _chunk],
                    swiglu_limit=swiglu_limit,
                )
                for i in range(0, x.shape[1], _chunk)
            ]
            return torch.cat(parts, dim=1)
        return moe_forward_streamed_grouped(
            moe, x, layer_id, arena, device=device, image_mask=image_mask,
            swiglu_limit=swiglu_limit,
        )
    _fast = _os.environ.get("DSV41_MOE_FAST", "")
    # Measured A/B (one layer, same CPU reference, docs/notes/dsv41-perf-moe-ab.md):
    #   rows=64 : default 912 ms -> fast 538 ms  (-41%), +W13 515 ms (-43%)
    #   rows=1  : default  60 ms -> fast  63 ms  (+4%),  +W13  68 ms (+12%)
    # The device-accumulator variant wins when a layer touches many experts (prefill) and loses
    # slightly when it touches six (decode), so "auto" picks per call.
    if _fast == "auto":
        _fast = "1" if (x.numel() // x.shape[-1]) > 1 else ""
    if _fast and _fast != "0":
        return _moe_forward_streamed_device_accum(
            moe, x, layer_id, arena, device=device, image_mask=image_mask,
            swiglu_limit=swiglu_limit,
        )
    from tools.dsv41 import ref_cpu

    shape = x.size()
    flat = x.view(-1, moe.gate.weight.size(-1))
    mask = None if image_mask is None else image_mask.flatten()
    weights, indices = gate_forward(moe.gate, flat, mask)

    used = sorted({int(e) for e in indices.flatten().tolist()})
    # a prefill can touch all 384 experts while the arena holds far fewer slots, so page in groups:
    # every group is packed, copied and *dropped* before the next one starts, which keeps the working
    # set inside the watermark without ever reading a slot that was silently reused.
    groups = [used[i : i + arena.n_slots] for i in range(0, len(used), arena.n_slots)]

    xd = flat.to(device)
    # the gate may have run on the device; the scatter and the weight lookup stay on the host
    indices_h = indices.cpu()
    weights_h = weights.cpu()
    y = torch.zeros(flat.shape, dtype=torch.float32)
    for group in groups:
        slots, fresh = {}, []
        for e in group:
            slot, hit = arena.bind(layer_id, e)
            slots[e] = slot
            if not hit:
                fresh.append(slot)
            arena.release(layer_id, e)
        for slot_and_expert in [(slots[e], moe.experts[e]) for e in group if slots[e] in fresh]:
            arena.load_slot(slot_and_expert[0], slot_and_expert[1], device)
        for e in group:
            idx, top = torch.where(indices_h == e)
            expert = moe.experts[e]
            limit = expert.swiglu_limit
            xsel = xd[idx.to(xd.device)]  # NPU needs the index on the device too
            views = arena.views(slots[e], on_device=True)
            # exactly the order and dtypes of expert_forward()
            gate_out = ref_cpu.linear(xsel, views["w1"], views["w1.scale"]).float()
            up = ref_cpu.linear(xsel, views["w3"], views["w3.scale"]).float()
            if limit > 0:
                up = torch.clamp(up, min=-limit, max=limit)
                gate_out = torch.clamp(gate_out, max=limit)
            out = torch.nn.functional.silu(gate_out) * up
            out = out * weights_h[idx, top, None].float().to(xd.device)
            out = ref_cpu.linear(out.to(xd.dtype), views["w2"], views["w2.scale"])
            y[idx] += out.float().cpu()
        if len(groups) > 1:
            # only a fan-out that does not fit needs its slots back; a whole layer that fits stays
            # resident so the next token's experts are cache hits (no H2D, no unpack again)
            for e in group:
                arena.cache.drop((layer_id, e))
    y = y + expert_forward(moe.shared_experts, flat).float().cpu()
    return y.type_as(x).to(x.device).view(shape)


def _fp4_linear_batched(x, packed, scale, out_f, in_f, return_w=False):
    """``ref_cpu.fp4_gemm`` with a batch dim: x [k, m, in], packed [k, out, in//2], scale [k, out, in//32]."""
    from tools.dsv41 import ref_cpu

    # fp4 grid values and their e8m0 block scales are powers of two, so both weights and the
    # quantised activations are exactly representable in bf16.  A bf16 GEMM therefore computes the
    # same products (fp32 accumulate) at half the unpack/copy traffic -- measured decode win on the
    # NPU; opt-in until the 100-prompt greedy gate has run (N5).
    bf16 = os.environ.get("DSV41_MOE_BF16_GEMM", "") not in ("", "0")
    xq, xs = ref_cpu.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu, False)
    k = packed.shape[0]
    nblocks = in_f // 32
    w = ref_cpu.unpack_fp4_lut(
        packed, k * out_f, in_f, dtype=torch.bfloat16 if bf16 else None
    )
    w = w.reshape(k, out_f, nblocks, 32)
    # broadcast the block scale instead of expanding it: repeat_interleave allocates a
    # [k, out, in] fp32 tensor (280 MB at k=6) that then has to be read back by the mul
    s = ref_cpu.scale_to_float(scale)
    if bf16:
        s = s.to(torch.bfloat16)
    w = w * s.unsqueeze(-1)
    wf = w.reshape(k, out_f, in_f)
    xf = xq.float() * ref_cpu.scale_to_float(xs).repeat_interleave(32, dim=-1)
    if bf16:
        xf = xf.to(torch.bfloat16)
    c = torch.bmm(xf, wf.transpose(1, 2))
    if return_w:
        return c.to(torch.get_default_dtype()), wf
    return c.to(torch.get_default_dtype())


def _bf16_linear_batched(x, w, out_f, in_f):
    """Same maths as ``_fp4_linear_batched`` but ``w`` is already dequantised bf16."""
    from tools.dsv41 import ref_cpu

    xq, xs = ref_cpu.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu, False)
    xf = (xq.float() * ref_cpu.scale_to_float(xs).repeat_interleave(32, dim=-1)).to(torch.bfloat16)
    return torch.bmm(xf, w.transpose(1, 2))


#: ``(layer_id, expert_id) -> (w1, w3, w2)`` dequantised bf16 expert weights (P1 decode cache).
_DEQ_CACHE: "OrderedDict[tuple, tuple]" = OrderedDict()
_DEQ_BYTES = 0
#: ``(layer_id, ordered expert ids) -> (w13 bf16 [k,2h,dim], w2 bf16 [k,dim,h])`` decode memo.
_SET_CACHE: "OrderedDict[tuple, tuple]" = OrderedDict()
_SET_BYTES = 0


def _deq_expert(layer_id: int, expert_id: int, views, hid: int, dim: int, cap_bytes: int):
    """Return cached bf16 weights for one expert, dequantising on miss."""
    global _DEQ_BYTES
    from tools.dsv41 import ref_cpu

    key = (layer_id, expert_id)
    hit = _DEQ_CACHE.get(key)
    if hit is not None:
        _DEQ_CACHE.move_to_end(key)
        return hit

    def deq(packed, scale, out_f, in_f):
        w = ref_cpu.unpack_fp4_lut(packed, out_f, in_f, dtype=torch.bfloat16)
        s = ref_cpu.scale_to_float(scale).to(torch.bfloat16)
        return (w.reshape(out_f, in_f // 32, 32) * s.unsqueeze(-1)).reshape(out_f, in_f)

    w1 = deq(views["w1"], views["w1.scale"], hid, dim)
    w3 = deq(views["w3"], views["w3.scale"], hid, dim)
    w2 = deq(views["w2"], views["w2.scale"], dim, hid)
    nbytes = sum(t.numel() * 2 for t in (w1, w3, w2))
    if nbytes <= cap_bytes:
        while _DEQ_BYTES + nbytes > cap_bytes and _DEQ_CACHE:
            _old_key, old = _DEQ_CACHE.popitem(last=False)
            _DEQ_BYTES -= sum(t.numel() * 2 for t in old)
        _DEQ_CACHE[key] = (w1, w3, w2)
        _DEQ_BYTES += nbytes
    return w1, w3, w2


def moe_forward_batched(moe, x, layer_id, arena, device="npu:0", image_mask=None):
    """Decode fast path: one batched GEMM over the token's topk experts instead of six."""
    from tools.dsv41 import ref_cpu

    import time as _bt

    _tm = os.environ.get("VLLM_DSV41_BATCHED_MOE_TIMING", "") not in ("", "0")
    _marks = {}

    def _mark(name):
        if _tm:
            torch.npu.synchronize()
            _marks[name] = _bt.perf_counter()

    _mark("t0")

    shape = x.size()
    flat = x.view(-1, moe.gate.weight.size(-1))
    mask = None if image_mask is None else image_mask.flatten()
    weights, indices = gate_forward(moe.gate, flat, mask)
    if _tm:
        _mark("gate")
    if flat.shape[0] != 1:
        raise ValueError("moe_forward_batched is the decode fast path (one token per step)")
    ids = [int(e) for e in indices[0].tolist()]

    dim = moe.gate.weight.size(-1)
    hid = moe.experts[0].w1.weight.shape[0]
    _scap_gb = float(os.environ.get("DSV41_DECODE_SET_CACHE_GB", "0") or 0)
    _skey = (int(layer_id), tuple(ids))
    if _scap_gb > 0:
        _ent = _SET_CACHE.get(_skey)
        if _ent is not None:
            _SET_CACHE.move_to_end(_skey)
            w13, w2s = _ent
            xd = flat.to(device)
            xk = xd.expand(len(ids), -1, -1)
            both = _bf16_linear_batched(xk, w13, 2 * hid, dim).float()
            gate_out, up = both[:, 0, :hid], both[:, 0, hid:]
            limit = moe.experts[ids[0]].swiglu_limit
            if limit > 0:
                up = torch.clamp(up, min=-limit, max=limit)
                gate_out = torch.clamp(gate_out, max=limit)
            act = torch.nn.functional.silu(gate_out) * up
            act = act * weights[0, :, None].float().to(act.device)
            out = _bf16_linear_batched(act[:, None, :].to(xd.dtype), w2s, dim, hid).float()[:, 0]
            y = out.sum(dim=0, keepdim=True).cpu()
            y = y + expert_forward(moe.shared_experts, flat).float().cpu()
            return y.type_as(x).to(x.device).view(shape)

    slots = []
    for e in ids:
        slot, hit = arena.bind(layer_id, e)
        if not hit:
            arena.load_slot(slot, moe.experts[e], device)
        arena.release(layer_id, e)
        slots.append(slot)
    if _tm:
        _mark("load")

    dev_flat = arena._device
    S = arena.slot_bytes
    specs = {sp.name: sp for sp in arena.specs}

    def gather(name, out_f, in_f, width):
        sp = specs[name]
        parts = [dev_flat[sl * S + sp.offset : sl * S + sp.end] for sl in slots]
        return torch.stack(parts).view(torch.uint8).view(width).reshape(len(slots), out_f, in_f)

    dim = moe.gate.weight.size(-1)
    hid = moe.experts[0].w1.weight.shape[0]

    _cap_gb = float(os.environ.get("DSV41_DECODE_DEQ_CACHE_GB", "0") or 0)
    if _cap_gb > 0:
        cap_bytes = int(_cap_gb * 2**30)
        cached = []
        for e, slot in zip(ids, slots):
            w1, w3, w2 = _deq_expert(
                layer_id, e, arena.views(slot, on_device=True), hid, dim, cap_bytes
            )
            cached.append((w1, w3, w2))
        w13 = torch.stack([torch.cat([w1, w3], dim=0) for w1, w3, _ in cached])
        w2s = torch.stack([w2 for _, _, w2 in cached])

        xd = flat.to(device)
        xk = xd.expand(len(slots), -1, -1)
        both = _bf16_linear_batched(xk, w13, 2 * hid, dim).float()
        gate_out, up = both[:, 0, :hid], both[:, 0, hid:]
        limit = moe.experts[ids[0]].swiglu_limit
        if limit > 0:
            up = torch.clamp(up, min=-limit, max=limit)
            gate_out = torch.clamp(gate_out, max=limit)
        act = torch.nn.functional.silu(gate_out) * up
        act = act * weights[0, :, None].float().to(act.device)
        out = _bf16_linear_batched(act[:, None, :].to(xd.dtype), w2s, dim, hid).float()[:, 0]
        y = out.sum(dim=0, keepdim=True).cpu()
        y = y + expert_forward(moe.shared_experts, flat).float().cpu()
        return y.type_as(x).to(x.device).view(shape)

    w1 = gather("w1", hid, dim // 2, torch.float4_e2m1fn_x2)
    w3 = gather("w3", hid, dim // 2, torch.float4_e2m1fn_x2)
    w2 = gather("w2", dim, hid // 2, torch.float4_e2m1fn_x2)
    s1 = gather("w1.scale", hid, dim // 32, torch.float8_e8m0fnu)
    s3 = gather("w3.scale", hid, dim // 32, torch.float8_e8m0fnu)
    s2 = gather("w2.scale", dim, hid // 32, torch.float8_e8m0fnu)
    if _tm:
        _mark("gather")

    xd = flat.to(device)
    xk = xd.expand(len(slots), -1, -1)
    _want = _scap_gb > 0
    if _want:
        gate_c, gate_w = _fp4_linear_batched(xk, w1, s1, hid, dim, return_w=True)
        up_c, up_w = _fp4_linear_batched(xk, w3, s3, hid, dim, return_w=True)
        gate_out, up = gate_c.float()[:, 0], up_c.float()[:, 0]
    else:
        gate_out = _fp4_linear_batched(xk, w1, s1, hid, dim).float()[:, 0]
        up = _fp4_linear_batched(xk, w3, s3, hid, dim).float()[:, 0]
    if _tm:
        _mark("w13")
    limit = moe.experts[ids[0]].swiglu_limit
    if limit > 0:
        up = torch.clamp(up, min=-limit, max=limit)
        gate_out = torch.clamp(gate_out, max=limit)
    act = torch.nn.functional.silu(gate_out) * up
    act = act * weights[0, :, None].float().to(act.device)
    if _want:
        out_c, w2_w = _fp4_linear_batched(
            act[:, None, :].to(xd.dtype), w2, s2, dim, hid, return_w=True
        )
        out = out_c.float()[:, 0]
    else:
        out = _fp4_linear_batched(act[:, None, :].to(xd.dtype), w2, s2, dim, hid).float()[:, 0]
    if _want:
        global _SET_BYTES

        w13s = torch.stack([torch.cat([gate_w[i], up_w[i]], dim=0) for i in range(len(ids))])
        w2ss = torch.stack([w2_w[i] for i in range(len(ids))])
        _nb = sum(t.numel() * t.element_size() for t in (w13s, w2ss))
        _capb = int(_scap_gb * 2**30)
        while _SET_BYTES + _nb > _capb and _SET_CACHE:
            _k, _old = _SET_CACHE.popitem(last=False)
            _SET_BYTES -= sum(t.numel() * t.element_size() for t in _old)
        _SET_CACHE[_skey] = (w13s, w2ss)
        _SET_BYTES += _nb
    if _tm:
        _mark("w2")
        _d = {k: (_marks[k] - _marks["t0"]) * 1e3 for k in _marks if k != "t0"}
        print(
            "DSV41_BMOE_TIMING layer=%d gate=%.2f load=%.2f gather=%.2f w13=%.2f w2=%.2f"
            % (
                int(layer_id),
                _d.get("gate", 0.0),
                _d.get("load", 0.0) - _d.get("gate", 0.0),
                _d.get("gather", 0.0) - _d.get("load", 0.0),
                _d.get("w13", 0.0) - _d.get("gather", 0.0),
                _d.get("w2", 0.0) - _d.get("w13", 0.0),
            ),
            flush=True,
        )
    y = out.sum(dim=0, keepdim=True).cpu()
    y = y + expert_forward(moe.shared_experts, flat).float().cpu()
    return y.type_as(x).to(x.device).view(shape)


# =============================================================================================
# Engram
# =============================================================================================


def engram_forward(
    engram: Dsv41Engram,
    x: torch.Tensor,
    hash_ids: torch.Tensor,
    token_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``model.py:Engram.forward`` -- n-gram lookup gated by how well it matches the stream."""
    o = ops()
    values = o.dequant_engram_rows(engram.embed, hash_ids)  # [b,s,24,256] -> [b,s,6144] bf16
    flat_vals = values.flatten(-2)
    target = engram.wkv.weight.device
    if flat_vals.device != target:  # the table is host-resident, the projection may be on the card
        flat_vals = flat_vals.to(target)
    kv = o.linear(flat_vals, engram.wkv.weight, engram.wkv.scale)
    key, value = kv.split([engram.q_weight.size(0) * x.size(-1), x.size(-1)], dim=-1)
    key = key.float().unflatten(-1, (engram.q_weight.size(0), x.size(-1)))
    weight = engram.q_weight.float() * engram.k_weight.float()
    h, eps = x.float(), engram.eps
    rstd = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(key.square().mean(-1) + eps)
    dot = (h * weight * key).sum(-1) * rstd * x.size(-1) ** -0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(engram.clamp_value).sqrt(), dot))
    if token_mask is not None:
        gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
    return (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)


# =============================================================================================
# Block
# =============================================================================================


@dataclass
class ModelState:
    """Everything one whole-stack forward needs: per-layer runtimes plus the shared stream slot."""

    cfg: Dsv41Config
    shared: SharedAttentionState = field(default_factory=SharedAttentionState)
    layers: Dict[int, LayerRuntime] = field(default_factory=dict)
    engram_hash: object = None
    max_seq_len: int = 4096
    bsz: int = 1
    device: str = "cpu"

    def runtime(self, layer_id: int, n_layers: int) -> LayerRuntime:
        if layer_id not in self.layers:
            self.layers[layer_id] = LayerRuntime(
                cfg=self.cfg,
                layer_id=layer_id,
                shared=self.shared,
                bsz=self.bsz,
                max_seq_len=self.max_seq_len,
                device=self.device,
            ).build()
        return self.layers[layer_id]


def block_forward(
    block: Dsv41Block,
    x: torch.Tensor,
    start_pos: int,
    pre_mix: torch.Tensor,
    state: ModelState,
    image_mask: Optional[torch.Tensor] = None,
    engram_hashes: Optional[torch.Tensor] = None,
    moe_fn: Optional[Callable] = None,
    timings: Optional[dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``model.py:Block.forward`` -- attention then MoE, each between hc_pre and hc_post."""
    residual = x
    attn_pre, attn_post, attn_comb = hc_mixes(
        block, x, block.hc_attn_fn, block.hc_attn_scale, block.hc_attn_base
    )
    x = hc_pre(x, pre_mix)
    x = rms_norm(block.attn_norm, x)
    if os.environ.get("DSV41_TRACE") and block.layer_id == 0:
        _wo = getattr(block.attn, "wo_a", None)
        print(
            "TRACE blk0 attn_in[:4]",
            [round(float(v), 5) for v in x.flatten()[:4].tolist()],
            "rms",
            round(float(x.float().pow(2).mean().sqrt()), 6),
            "wo_a_rms",
            None if _wo is None else round(float(_wo.weight.float().pow(2).mean().sqrt()), 6),
            "wo_a_absmean",
            None if _wo is None else round(float(_wo.weight.float().abs().mean()), 6),
            flush=True,
        )
    rt = state.runtime(block.layer_id, state.cfg.n_layers)
    x = attention_forward(block.attn, x, start_pos, rt, state.cfg)
    if os.environ.get("DSV41_TRACE") and block.layer_id == 0:
        print(
            "TRACE blk0 attn_out[:4]",
            [round(float(v), 5) for v in x.flatten()[:4].tolist()],
            "rms",
            round(float(x.float().pow(2).mean().sqrt()), 6),
            flush=True,
        )
    x = hc_post(x, residual, attn_post, attn_comb)

    residual = x
    ffn_pre, ffn_post, ffn_comb = hc_mixes(
        block, x, block.hc_ffn_fn, block.hc_ffn_scale, block.hc_ffn_base
    )
    x = hc_pre(x, attn_pre)
    x = rms_norm(block.ffn_norm, x)
    if timings is None:
        x = (moe_fn or moe_forward)(block.ffn, x, image_mask)
    else:
        import time as _t

        _t0 = _t.perf_counter()
        x = (moe_fn or moe_forward)(block.ffn, x, image_mask)
        try:
            torch.npu.synchronize()
        except Exception:
            pass
        timings["moe"] = timings.get("moe", 0.0) + (_t.perf_counter() - _t0)
    x = hc_post(x, residual, ffn_post, ffn_comb)
    return x, ffn_pre


# =============================================================================================
# Whole stack
# =============================================================================================


def logits_from_hidden(model: Dsv41Model, h: torch.Tensor) -> torch.Tensor:
    """Final norm + head on the last position (``ParallelHead.forward`` with ``full_logits=False``)."""
    o = ops()
    x = rms_norm(model.norm, h)
    # the head is 129280x5120 bf16; casting it to fp32 **every step** costs ~85 ms on the NPU.
    # The values are unchanged by caching, so parity is unaffected.
    if os.environ.get("DSV41_HEAD_BF16", "") not in ("", "0"):
        # opt-in: half the head traffic, cube-native bf16 GEMM; check the greedy gate before default
        return F.linear(x[:, -1].to(model.head.weight.dtype), model.head.weight)
    wf = getattr(model, "_head_weight_f32", None)
    if wf is None:
        wf = model.head.weight.float()
        model._head_weight_f32 = wf
    return F.linear(x[:, -1].float(), wf)


def logits_all_positions(model: Dsv41Model, h: torch.Tensor) -> torch.Tensor:
    """Final norm + head on **every** position → ``[b, s, vocab]``。

    投机解码的验证需要"一次前向、逐位置取 argmax"：把 k 个候选 token 一起喂进去，
    每个位置都要有自己的 logits。:func:`logits_from_hidden` 只给最后一个位置，所以单列一个函数
    （对应 ``ParallelHead.forward`` 的 ``full_logits=True``）。

    与 :func:`logits_from_hidden` 保持一致：在最后一维上，两者的最后位置必须逐位相同。
    """
    o = ops()
    x = rms_norm(model.norm, h)
    return F.linear(x.float(), model.head.weight.float())


def transformer_forward(
    model: Dsv41Model,
    input_ids: torch.Tensor,
    state: ModelState,
    start_pos: int = 0,
    engram_hashes: Optional[torch.Tensor] = None,
    image_mask: Optional[torch.Tensor] = None,
    capture: Optional[dict] = None,
    moe_fn: Optional[Callable] = None,
    timings: Optional[dict] = None,
    capture_layers: Optional[dict] = None,
    skip_logits: bool = False,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """``model.py:Transformer.forward`` -- embed, expand, blocks, collapse, logits.

    Returns ``(logits, extras)`` where ``extras["main_hiddens"]`` is the concatenation the DSpark
    stages consume (the *attention input* of the target layers, i.e. before that layer runs).
    """
    o = ops()
    engram_mask = None if image_mask is None else ~image_mask
    if engram_hashes is None and getattr(state, "engram_hash", None) is not None:
        if any(block.engram is not None for block in model.layers):
            engram_hashes = state.engram_hash.forward(input_ids, start_pos, engram_mask)
    input_ids = input_ids.to(model.embed.weight.device)
    h = o.embedding(input_ids, model.embed.weight)
    if os.environ.get("DSV41_TRACE"):
        # vLLM bisect (docs/notes/dsv41-vllm-integration-plan.md §23): the first block already differs
        # by ~670x, so the embedding output is the first thing to compare between the two runners
        _rms = float(h.float().pow(2).mean().sqrt())
        print(
            "TRACE embed[:4]",
            [round(float(v), 6) for v in h.flatten()[:4].tolist()],
            "rms",
            round(_rms, 6),
            "ids",
            input_ids.flatten()[:4].tolist(),
            flush=True,
        )
    h = h.unsqueeze(2).repeat(1, 1, state.cfg.hc_mult, 1)
    pre_mix = make_identity_pre_mix(h, state.cfg.hc_mult)
    main_hiddens: List[torch.Tensor] = []
    target = set(int(x) for x in state.cfg.dspark_target_layer_ids)
    for layer_id, block in enumerate(model.layers):
        if block.engram is not None and engram_hashes is not None:
            h = engram_forward(
                block.engram,
                h,
                engram_hashes[:, :, block.engram.layer_hash_index, :],
                None if image_mask is None else ~image_mask,
            )
        if layer_id in target:
            main_hiddens.append(h.mean(dim=2))
        h, pre_mix = block_forward(
            block, h, start_pos, pre_mix, state, image_mask, engram_hashes=None, moe_fn=moe_fn,
            timings=timings,
        )
        if os.environ.get("DSV41_TRACE_MEM"):
            try:
                _alloc = torch.npu.memory_allocated() / 2**30
                _peak = torch.npu.max_memory_allocated() / 2**30
            except Exception:  # noqa: BLE001 - CPU
                _alloc = _peak = 0.0
            print(
                f"DSV41_MEM layer={layer_id} start={start_pos} seq={input_ids.shape[-1]} "
                f"alloc={_alloc:.1f}GiB peak={_peak:.1f}GiB",
                flush=True,
            )
        if capture_layers is not None:
            capture_layers[layer_id] = h.detach().mean(dim=2)
    h = hc_pre(h, pre_mix)
    if capture is not None:
        capture["h"] = h.detach()
        # 投机解码验证需要"全部位置的 hidden"（一次前向吃 k 个候选、逐位置取 argmax）。
        # 走 capture 旁路，不改 (logits, extras) 的返回契约，现有调用点全部不受影响。
        capture["h_all"] = h.detach()
    logits = None if skip_logits else logits_from_hidden(model, h)
    extras = {}
    if main_hiddens:
        extras["main_hiddens"] = torch.cat(main_hiddens, dim=-1)
    return logits, extras
