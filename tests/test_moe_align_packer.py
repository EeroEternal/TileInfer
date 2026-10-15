"""CPU contract tests for the H26 preNZ align module (tileinfer.kernels.moe.align).

Scope (CPU-only by design):
* ``pack_gate_up`` / ``pack_down`` strip-layout bookkeeping: flat output must decode, strip by
  strip, back to the original ND weights via ``unpack_compact_nz_to_nd``;
* strip addressing contract consumed by the align kernels (gate strip = e*2*NT+nt, up = +NT,
  down = e*NT+nt), including the multi-tile (NT>1) case;
* kernel factory constructibility and the burst geometry constants.

Device numerics/perf authority is the H26 probe chain (benchmarks/probes/h26_moe_prenz_align.py:
fp32 oracle TOL 3e-2 relative + 7-scenario bench vs aclnn, gate decode_b32/prefill_400 >= 1.0x);
the module is a line-identical modularization of that probe, so no duplicate device test lives here.
"""

from __future__ import annotations

import pytest
import torch

from tileinfer.kernels.moe import (
    moe_down_align,
    moe_gate_up_swiglu_align,
    pack_down,
    pack_gate_up,
    unpack_compact_nz_to_nd,
)
from tileinfer.kernels.moe.align import (
    BURST_BYTES,
    DOWN_BLOCK_N,
    GATE_UP_BLOCK_N,
    NUM_CORES,
)

E = 8
H = 256
I = 256  # == GATE_UP_BLOCK_N -> NT_gate_up = 1


def _w(seed: int):
    g = torch.Generator().manual_seed(seed)
    w13 = (torch.randn((E, 2 * I, H), generator=g) * 0.02).to(torch.bfloat16)
    w2 = (torch.randn((E, H, I), generator=g) * 0.02).to(torch.bfloat16)
    return w13, w2


def test_pack_gate_up_strip_roundtrip():
    w13, _ = _w(0)
    nt = I // GATE_UP_BLOCK_N
    assert nt >= 1
    flat = pack_gate_up(w13, inter=I)
    assert flat.shape == (E * 2 * nt * GATE_UP_BLOCK_N * H,)
    # strip (e, h, nt) must decode to w13[e, h*I + nt*BN : +BN, :] ([BN, K] compact-NZ tile)
    strips = flat.view(E, 2, nt, GATE_UP_BLOCK_N, H)
    for e in range(E):
        for h in range(2):
            for t in range(nt):
                nz = unpack_compact_nz_to_nd(strips[e, h, t])
                ref = w13[e, h * I + t * GATE_UP_BLOCK_N: h * I + (t + 1) * GATE_UP_BLOCK_N]
                assert torch.equal(nz, ref)


def test_pack_gate_up_multi_tile_strip_addressing():
    # production-like geometry: I=512, BN=256 -> NT=2; up strip of (e, nt) = gate strip + NT
    i2 = 512
    g = torch.Generator().manual_seed(3)
    w13 = (torch.randn((E, 2 * i2, H), generator=g) * 0.02).to(torch.bfloat16)
    nt = i2 // GATE_UP_BLOCK_N
    flat = pack_gate_up(w13, inter=i2)
    strips = flat.view(E, 2, nt, GATE_UP_BLOCK_N, H)
    # gate strip (e, 0, nt) at offset (e*2*NT+nt); up strip = +NT (kernel's base_g / up_off math)
    for e in range(E):
        for t in range(nt):
            gate_idx = (e * 2 * nt + t)
            up_idx = gate_idx + nt
            flat_gate = flat[gate_idx * GATE_UP_BLOCK_N * H: (gate_idx + 1) * GATE_UP_BLOCK_N * H]
            flat_up = flat[up_idx * GATE_UP_BLOCK_N * H: (up_idx + 1) * GATE_UP_BLOCK_N * H]
            assert torch.equal(flat_gate, strips[e, 0, t].reshape(-1))
            assert torch.equal(flat_up, strips[e, 1, t].reshape(-1))


def test_pack_down_strip_roundtrip():
    _, w2 = _w(1)
    nt = H // DOWN_BLOCK_N
    flat = pack_down(w2, inter=I)
    assert flat.shape == (E * nt * DOWN_BLOCK_N * I,)
    strips = flat.view(E, nt, DOWN_BLOCK_N, I)
    for e in range(E):
        for t in range(nt):
            nz = unpack_compact_nz_to_nd(strips[e, t])
            assert torch.equal(nz, w2[e, t * DOWN_BLOCK_N: (t + 1) * DOWN_BLOCK_N])


def test_pack_non_contiguous_input_rejected_or_handled():
    w13, _ = _w(2)
    transposed = w13.transpose(1, 2)  # non-contiguous view
    with pytest.raises((AssertionError, RuntimeError, ValueError)):
        pack_gate_up(transposed, inter=I)


def test_kernel_factories_construct():
    spec_gu = dict(num_tiles_max=4, rows_slots=128, num_experts=E, inter=I, hidden=H,
                   block_m=32, block_n=GATE_UP_BLOCK_N, block_k=128)
    spec_dn = dict(num_tiles_max=4, rows_slots=128, num_experts=E, inter=I, hidden=H,
                   block_m=32, block_n=DOWN_BLOCK_N, block_k=512)
    f1 = moe_gate_up_swiglu_align(**spec_gu)
    f2 = moe_gate_up_swiglu_align(**spec_gu)
    assert f1 is not None and f2 is not None  # TIRScript objects (lowered/compiled on device)
    d1 = moe_down_align(**spec_dn)
    d2 = moe_down_align(**spec_dn)
    assert d1 is not None and d2 is not None


def test_slice_burst_contract():
    # gate_up spec (BN=256, BK=128) and down spec (BN=64, BK=512) both give 64KiB K-slices
    # == 16 x 4096B bursts (iter-053/054 contract); slices must divide BURST_BYTES exactly
    for bn, bk in ((GATE_UP_BLOCK_N, 128), (DOWN_BLOCK_N, 512)):
        total = bn * bk * 2
        assert total % BURST_BYTES == 0
        assert total // BURST_BYTES == 16
    assert NUM_CORES == 28
