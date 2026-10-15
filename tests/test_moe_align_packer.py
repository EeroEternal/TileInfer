"""CPU contract tests for the H26 preNZ align module (tileinfer.kernels.moe.align).

Scope (CPU-only by design):
* ``pack_gate_up`` / ``pack_down`` transposed-continuous layout bookkeeping (H27/E76: the
  measured nd2nz L1 physical order is column-major-continuous, NOT compact-NZ; align loads
  GM flat verbatim, so each [BN,K] slice must be transposed before flattening);
* strip addressing contract consumed by the align kernels (gate strip = e*2*NT+nt, up = +NT,
  down = e*NT+nt), including the multi-tile (NT>1) case;
* D73 flat-memory in-place pack/unpack (equal-numel pure permutation on the weight's own
  storage, temp peak = one layer; reversible ND restore for fallback) and the takeover
  state's acquire/release registry semantics;
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
    pack_down_inplace,
    pack_gate_up,
    pack_gate_up_inplace,
    unpack_down_inplace,
    unpack_gate_up_inplace,
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
    # transposed-continuous: flat.view(..., K, BN) == w13.view(..., BN, K).transpose(-2, -1)
    strips = flat.view(E, 2, nt, H, GATE_UP_BLOCK_N)
    ref = w13.view(E, 2, nt, GATE_UP_BLOCK_N, H).transpose(-2, -1)
    assert torch.equal(strips, ref.contiguous())


def test_pack_gate_up_multi_tile_strip_addressing():
    # production-like geometry: I=512, BN=256 -> NT=2; up strip of (e, nt) = gate strip + NT
    i2 = 512
    g = torch.Generator().manual_seed(3)
    w13 = (torch.randn((E, 2 * i2, H), generator=g) * 0.02).to(torch.bfloat16)
    nt = i2 // GATE_UP_BLOCK_N
    flat = pack_gate_up(w13, inter=i2)
    strips = flat.view(E, 2, nt, H, GATE_UP_BLOCK_N)
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
    strips = flat.view(E, nt, I, DOWN_BLOCK_N)
    ref = w2.view(E, nt, DOWN_BLOCK_N, I).transpose(-2, -1)
    assert torch.equal(strips, ref.contiguous())


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


def test_pack_gate_up_inplace_roundtrip_bitwise():
    """D73 原地 pack:与拷贝版输出逐位一致;unpack 逆置换逐位还原 ND。"""
    w13, _ = _w(5)
    orig = w13.clone()
    ref = pack_gate_up(orig, inter=I)
    pack_gate_up_inplace(w13, inter=I)
    assert torch.equal(w13.view(-1), ref)
    unpack_gate_up_inplace(w13, inter=I)
    assert torch.equal(w13, orig)


def test_pack_down_inplace_roundtrip_bitwise():
    _, w2 = _w(6)
    orig = w2.clone()
    ref = pack_down(orig, inter=I)
    pack_down_inplace(w2, inter=I)
    assert torch.equal(w2.view(-1), ref)
    unpack_down_inplace(w2, inter=I)
    assert torch.equal(w2, orig)


def test_pack_inplace_multi_tile_gate_up():
    """原地版多 tile(NT>1)几何:转置连续序与拷贝版一致。"""
    i2 = 512
    g = torch.Generator().manual_seed(7)
    w13 = (torch.randn((E, 2 * i2, H), generator=g) * 0.02).to(torch.bfloat16)
    ref = pack_gate_up(w13.clone(), inter=i2)
    pack_gate_up_inplace(w13, inter=i2)
    assert torch.equal(w13.view(-1), ref)
    unpack_gate_up_inplace(w13, inter=i2)


def test_prenz_state_inplace_flat_memory_multilayer():
    """D73 平内存契约:打包=权重自身存储(data_ptr 不变,零第二副本,E78),
    回退可逆逐位还原 ND,多层注册并存互不干扰,回退后重入重 pack 一致。"""
    from tileinfer.integrations.moe_prenz import _PrenzMoeState

    st = _PrenzMoeState(inter=I, hidden=H, num_experts=E)
    w13a, w2a = _w(10)
    w13b, w2b = _w(11)
    ref_a = (pack_gate_up(w13a.clone(), inter=I), pack_down(w2a.clone(), inter=I))
    ref_b = (pack_gate_up(w13b.clone(), inter=I), pack_down(w2b.clone(), inter=I))
    orig_a = (w13a.clone(), w2a.clone())
    orig_b = (w13b.clone(), w2b.clone())

    pa = st.acquire_packed(w13a, w2a)
    assert pa[0].data_ptr() == w13a.data_ptr()
    assert pa[1].data_ptr() == w2a.data_ptr()
    assert torch.equal(pa[0], ref_a[0]) and torch.equal(pa[1], ref_a[1])
    assert len(st._packed) == 2

    pb = st.acquire_packed(w13b, w2b)  # 层 B 不得干扰层 A(多层并存)
    assert len(st._packed) == 4
    assert torch.equal(pb[0], ref_b[0]) and torch.equal(pb[1], ref_b[1])

    pa2 = st.acquire_packed(w13a, w2a)  # 打包态重入:零重 pack 快路径
    assert pa2[0].data_ptr() == pa[0].data_ptr()
    assert torch.equal(pa2[0], ref_a[0]) and torch.equal(pa2[1], ref_a[1])

    st.release_packed(w13a, w2a)  # 回退可逆:逐位还原 ND
    assert torch.equal(w13a, orig_a[0]) and torch.equal(w2a, orig_a[1])
    assert len(st._packed) == 2

    pa3 = st.acquire_packed(w13a, w2a)  # 回退后重入:重 pack 结果一致
    assert pa3[0].data_ptr() == w13a.data_ptr()
    assert torch.equal(pa3[0], ref_a[0]) and torch.equal(pa3[1], ref_a[1])

    st.release_packed(w13a, w2a)
    st.release_packed(w13b, w2b)
    assert torch.equal(w13a, orig_a[0]) and torch.equal(w2a, orig_a[1])
    assert torch.equal(w13b, orig_b[0]) and torch.equal(w2b, orig_b[1])
    assert len(st._packed) == 0


def test_prenz_state_repack_after_external_nd_mutation():
    """回退后权重被外部就地改写(ND 语义)再接入:按新 ND 内容重 pack。"""
    from tileinfer.integrations.moe_prenz import _PrenzMoeState

    st = _PrenzMoeState(inter=I, hidden=H, num_experts=E)
    w13, w2 = _w(12)
    st.acquire_packed(w13, w2)
    st.release_packed(w13, w2)
    w13.mul_(1.5)  # ND 语义下的外部更新(版本漂移)
    ref = pack_gate_up(w13.clone(), inter=I)
    pa = st.acquire_packed(w13, w2)
    assert torch.equal(pa[0], ref)
    st.release_packed(w13, w2)
