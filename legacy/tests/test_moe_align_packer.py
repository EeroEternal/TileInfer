"""冻结快照:主树 ``tests/test_moe_align_packer.py`` 里两个 ``_PrenzMoeState`` 用例。

搬到 ``legacy/`` 的原因:这两个用例依赖 ``tileinfer.integrations.moe_prenz``(preNZ 原地打包的
接管补丁),该模块已随 legacy 适配器一起冻结。主树剩下的 align 内核用例(``pack_gate_up`` /
``pack_down`` 布局、burst 几何、kernel 构造)不依赖 legacy,继续留在 ``tests/`` 里跑。

不参与 CI;只在需要复现现网 preNZ 行为时手动执行。
"""

from __future__ import annotations

import torch

from tileinfer.kernels.moe import pack_down, pack_gate_up

E = 8
H = 256
I = 256  # == GATE_UP_BLOCK_N -> NT_gate_up = 1


def _w(seed: int):
    g = torch.Generator().manual_seed(seed)
    w13 = (torch.randn((E, 2 * I, H), generator=g) * 0.02).to(torch.bfloat16)
    w2 = (torch.randn((E, H, I), generator=g) * 0.02).to(torch.bfloat16)
    return w13, w2


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
