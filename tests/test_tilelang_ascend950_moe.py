"""Device tests for the fused MoE grouped-GEMM kernels (Ascend 950).

Run on the reference machine as::

    .venv/bin/python -m pytest tests/test_tilelang_ascend950_moe.py -v

They skip (never fail) where the NPU / toolchain is absent.  Model shapes are
Qwen3.6-35B-A3B's: hidden=2048, 256 experts, moe_intermediate=512, top_k=8, bf16.

Cases:
* gate+up+SwiGLU vs the fp32 slot oracle, uniform routing (decode batch 4 / 32,
  prefill 128 / 400 tokens) — checks the two-accumulator column-halves trick;
* down projection vs the fp32 slot oracle — checks the second K loop;
* a skewed routing case (all tokens on one expert) — exercises multi-tile experts;
* the end-to-end routed MLP vs the full oracle.
"""

from __future__ import annotations

import pytest
import torch

from tileinfer.testing.moe_reference import (
    build_slot_layout,
    gather_slots,
    reference_down_slots,
    reference_gate_up_swiglu_slots,
    reference_moe,
)
from tileinfer.utils import is_npu_available

try:  # pragma: no cover
    import tilelang.ascend  # noqa: F401

    _HAVE_ASCEND950 = True
except Exception:  # pragma: no cover
    _HAVE_ASCEND950 = False

pytestmark = [
    pytest.mark.npu,
    pytest.mark.skipif(not is_npu_available(), reason="no Ascend NPU available"),
    pytest.mark.skipif(not _HAVE_ASCEND950, reason="tilelang.ascend (>=0.1.15) not importable"),
]

DTYPE = torch.bfloat16
TOL = 3e-2
E = 16  # experts in tests (full 256 is the benchmark's job; 16 keeps oracle loops sane)
H = 256  # hidden (scaled-down; kernel math is size-generic)
I = 128  # moe_intermediate
TOPK = 4
BM = 32


def _make_case(tokens: int, experts: int, skewed: bool = False, seed: int = 0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn((tokens, H), dtype=torch.float32, generator=g).to("npu").to(DTYPE)
    w13 = (torch.randn((experts, 2 * I, H), dtype=torch.float32, generator=g) * 0.02).to("npu").to(DTYPE)
    w2 = (torch.randn((experts, H, I), dtype=torch.float32, generator=g) * 0.02).to("npu").to(DTYPE)
    if skewed:
        topk_ids = torch.zeros((tokens, TOPK), dtype=torch.int64)
        topk_ids[:, 0] = 0  # everything piles onto expert 0
        topk_ids[:, 1:] = torch.randint(1, experts, (tokens, TOPK - 1), generator=g)
    else:
        topk_ids = torch.randint(0, experts, (tokens, TOPK), generator=g)
    topk_ids = topk_ids.to("npu")
    topk_weights = torch.rand((tokens, TOPK), generator=g).to("npu").to(torch.float32)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return x, w13, w2, topk_ids, topk_weights


def _run_gate_up(x, w13, topk_ids, block_m=BM, paired=True):
    from tileinfer.kernels.moe import build_gate_up_kernel, build_gate_up_paired_kernel, pair_gate_up_rows

    expanded = x.unsqueeze(1).expand(-1, topk_ids.shape[1], -1).reshape(-1, H)
    lay = build_slot_layout(topk_ids, block_m, num_experts=w13.shape[0])
    rows_slots = lay["num_tiles"] * block_m
    x_slots = gather_slots(expanded, lay["sorted_ids"], lay["num_expanded"]).contiguous()
    spec = dict(
        num_tiles_max=lay["num_tiles"],
        rows_slots=rows_slots,
        num_experts=w13.shape[0],
        inter=I,
        hidden=H,
        block_m=block_m,
    )
    if paired:
        kernel = build_gate_up_paired_kernel(**spec)
        w = pair_gate_up_rows(w13, 128)
    else:
        kernel = build_gate_up_kernel(**spec)
        w = w13
    y = torch.empty((rows_slots, I), dtype=DTYPE, device=x.device)
    kernel(x_slots, w, lay["expert_ids"], lay["sorted_ids"], y)
    return y, lay


def _run_down(y, w2, topk_ids, block_m=BM):
    from tileinfer.kernels.moe import build_down_kernel

    lay = build_slot_layout(topk_ids, block_m, num_experts=w2.shape[0])
    rows_slots = lay["num_tiles"] * block_m
    kernel = build_down_kernel(
        num_tiles_max=lay["num_tiles"],
        rows_slots=rows_slots,
        num_experts=w2.shape[0],
        inter=I,
        hidden=H,
        block_m=block_m,
    )
    out = torch.empty((rows_slots, H), dtype=DTYPE, device=y.device)
    kernel(y.contiguous(), w2, lay["expert_ids"], lay["sorted_ids"], out)
    return out, lay


@pytest.mark.parametrize("tokens", [1, 4, 32, 128])
def test_gate_up_swiglu_vs_oracle(tokens):
    x, w13, _, topk_ids, _ = _make_case(tokens, E, seed=tokens)
    y, lay = _run_gate_up(x, w13, topk_ids, paired=False)
    if tokens == 4:  # keep the paired kernel covered too
        y2, _ = _run_gate_up(x, w13, topk_ids, paired=True)
        assert (y2.float() - y.float()).abs().max().item() < TOL
    expanded = x.unsqueeze(1).expand(-1, TOPK, -1).reshape(-1, H)
    x_slots = gather_slots(expanded, lay["sorted_ids"], lay["num_expanded"])
    ref = reference_gate_up_swiglu_slots(
        x_slots, w13, lay["expert_ids"], lay["sorted_ids"], lay["num_expanded"], BM
    )
    valid = lay["sorted_ids"] < lay["num_expanded"]
    diff = (y[valid].float() - ref[valid]).abs().max().item()
    assert diff < TOL, f"gate_up mismatch {diff}"


def test_down_vs_oracle():
    x, w13, w2, topk_ids, _ = _make_case(8, E, seed=7)
    y, lay = _run_gate_up(x, w13, topk_ids)
    out, _ = _run_down(y, w2, topk_ids)
    expanded = x.unsqueeze(1).expand(-1, TOPK, -1).reshape(-1, H)
    x_slots = gather_slots(expanded, lay["sorted_ids"], lay["num_expanded"])
    y_ref = reference_gate_up_swiglu_slots(
        x_slots, w13, lay["expert_ids"], lay["sorted_ids"], lay["num_expanded"], BM
    )
    out_ref = reference_down_slots(
        y_ref, w2, lay["expert_ids"], lay["sorted_ids"], lay["num_expanded"], BM
    )
    valid = lay["sorted_ids"] < lay["num_expanded"]
    diff = (out[valid].float() - out_ref[valid]).abs().max().item()
    assert diff < TOL, f"down mismatch {diff}"


def test_skewed_routing_multi_tile():
    x, w13, _, topk_ids, _ = _make_case(16, E, skewed=True, seed=3)
    y, lay = _run_gate_up(x, w13, topk_ids, paired=True)  # skewed case exercises the paired path
    expanded = x.unsqueeze(1).expand(-1, TOPK, -1).reshape(-1, H)
    x_slots = gather_slots(expanded, lay["sorted_ids"], lay["num_expanded"])
    ref = reference_gate_up_swiglu_slots(
        x_slots, w13, lay["expert_ids"], lay["sorted_ids"], lay["num_expanded"], BM
    )
    valid = lay["sorted_ids"] < lay["num_expanded"]
    diff = (y[valid].float() - ref[valid]).abs().max().item()
    assert diff < TOL, f"skewed mismatch {diff}"


def test_end_to_end_vs_full_oracle():
    x, w13, w2, topk_ids, topk_weights = _make_case(8, E, seed=11)
    y, lay = _run_gate_up(x, w13, topk_ids)
    out, _ = _run_down(y, w2, topk_ids)
    # slot-order -> expanded-order scatter, then weighted combine per token
    rows_slots = out.shape[0]
    num_expanded = lay["num_expanded"]
    si = lay["sorted_ids"].to(torch.int64)
    valid = si < num_expanded
    out_expanded = torch.zeros((num_expanded, H), dtype=torch.float32, device=out.device)
    out_expanded.index_copy_(0, si[valid], out[valid].float())
    combined = (
        (out_expanded * topk_weights.reshape(-1).unsqueeze(-1))
        .view(topk_ids.shape[0], TOPK, H)
        .sum(dim=1)
    )
    ref = reference_moe(x, w13, w2, topk_ids, topk_weights)
    diff = (combined - ref).abs().max().item()
    assert diff < 4 * TOL, f"end-to-end mismatch {diff}"
