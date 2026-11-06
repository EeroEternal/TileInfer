"""Batched fp4 GEMM (the decode fast path): semantics must match ``ref_cpu.linear`` exactly.

``moe_forward_batched`` replaces six per-expert GEMMs with three ``bmm`` calls over a stacked weight.
The stacking is the only new thing -- unpack, block-scale expansion and activation quantisation are the
same -- so this test pins exactly that: for ``k`` experts the batched result must equal the per-expert
``linear`` result, row by row.

No device and no checkpoint: weights are quantised here with the same routine the loader uses, which
keeps the test runnable on a CPU-only box (the shared machine is often busy with someone else's work).
"""

from __future__ import annotations

import torch

from tileinfer.models.dsv41.block import _fp4_linear_batched
from tools.dsv41 import ref_cpu

OUT_F = 64
IN_F = 128


def make_weight(seed: int):
    """A quantised weight exactly as the checkpoint stores it: fp4 view + e8m0 block scales."""
    gen = torch.Generator().manual_seed(seed)
    w = torch.randn(OUT_F, IN_F, generator=gen)
    packed, scale = ref_cpu.quantize_weight_fp4_block32(w)
    return packed.view(torch.float4_e2m1fn_x2), scale


def test_batched_equals_per_expert():
    packed, scale = make_weight(0)
    k = 3
    packed_k = packed.unsqueeze(0).repeat(k, 1, 1)
    scale_k = scale.unsqueeze(0).repeat(k, 1, 1)

    gen = torch.Generator().manual_seed(7)
    x = torch.randn(k, 1, IN_F, generator=gen).to(torch.bfloat16)

    got = _fp4_linear_batched(x, packed_k, scale_k, OUT_F, IN_F)
    assert got.shape == (k, 1, OUT_F)
    for i in range(k):
        want = ref_cpu.linear(x[i], packed, scale)
        assert torch.equal(got[i], want), f"expert {i} differs from the per-expert GEMM"


def test_each_row_uses_its_own_weight():
    """Two different experts: each row must be its own (weight, scale) pair, and swapping must swap."""
    p0, s0 = make_weight(1)
    p1, s1 = make_weight(2)
    x = torch.randn(2, 1, IN_F).to(torch.bfloat16)

    got = _fp4_linear_batched(
        x, torch.stack([p0, p1]), torch.stack([s0, s1]), OUT_F, IN_F
    )
    assert torch.equal(got[0], ref_cpu.linear(x[0], p0, s0))
    assert torch.equal(got[1], ref_cpu.linear(x[1], p1, s1))

    swapped = _fp4_linear_batched(
        x, torch.stack([p1, p0]), torch.stack([s1, s0]), OUT_F, IN_F
    )
    # swapping only the weights must swap the rows, each still paired with its own input row\n    assert torch.equal(swapped[0], ref_cpu.linear(x[0], p1, s1))\n    assert torch.equal(swapped[1], ref_cpu.linear(x[1], p0, s0))


def test_single_expert_matches_reference():
    packed, scale = make_weight(3)
    x = torch.randn(1, 1, IN_F).to(torch.bfloat16)
    got = _fp4_linear_batched(x, packed.unsqueeze(0), scale.unsqueeze(0), OUT_F, IN_F)
    assert torch.equal(got[0], ref_cpu.linear(x[0], packed, scale))


def test_bf16_gemm_variant_is_numerically_equivalent(monkeypatch):
    """The opt-in bf16 GEMM (`DSV41_MOE_BF16_GEMM`) must agree with the fp32 variant.

    fp4 values and their power-of-two block scales are exact in bf16, so only the accumulation order
    differs; the bar is the S2 parity bar, not bit identity.
    """
    packed, scale = make_weight(4)
    k = 2
    x = torch.randn(k, 2, IN_F).to(torch.bfloat16)
    packed_k = packed.unsqueeze(0).repeat(k, 1, 1)
    scale_k = scale.unsqueeze(0).repeat(k, 1, 1)

    monkeypatch.delenv("DSV41_MOE_BF16_GEMM", raising=False)
    want = _fp4_linear_batched(x, packed_k, scale_k, OUT_F, IN_F)
    monkeypatch.setenv("DSV41_MOE_BF16_GEMM", "1")
    got = _fp4_linear_batched(x, packed_k, scale_k, OUT_F, IN_F)

    diff = (got.float() - want.float()).abs()
    rel = diff / (want.float().abs() + 1e-2)
    assert rel.max().item() <= 2e-2, f"max rel={rel.max().item():.3e} (abs={diff.max().item():.3e})"
    assert torch.allclose(got.float(), want.float(), rtol=2e-2, atol=2e-2)
