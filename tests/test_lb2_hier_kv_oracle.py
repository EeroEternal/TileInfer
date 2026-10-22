"""L-B2 分层 KV(池化余页)oracle 合成契约(H89,iter-253)。

真实 trace 判据以 benchmarks/probes/lb2_hier_kv_oracle.py --load 执行(设备 dump 域);
本文件固化 CPU 合成域的数学正确性与选择器/门控行为,防止判据实现缺陷伪证。
"""

import math

import pytest
import torch

from benchmarks.probes.lb2_hier_kv_oracle import (
    HQ,
    HKV,
    PAGE,
    exact_attention,
    hier_attention,
    pad_to_pages,
    rel_err,
    run_dump,
    select_centroid,
    select_mass,
)


def _case(pages: int, tail: int = 0):
    g = torch.Generator().manual_seed(11)
    s = pages * PAGE + tail
    k = torch.randn(HKV, s, D_DIM := 256, generator=g)
    v = torch.randn(HKV, s, D_DIM, generator=g)
    q = torch.randn(HQ, D_DIM, generator=g)
    return q, k, v


def test_full_selection_matches_exact():
    """全页精确选择 ⇒ 分层输出 ≡ exact(混合 softmax 路径数学正确性)。"""
    q, k, v = _case(4)
    valid = torch.full((4,), PAGE, dtype=torch.long)
    masks = torch.ones(HKV, 4, dtype=torch.bool)
    e = exact_attention(q, k, v)
    h = hier_attention(q, k, v, masks, valid)
    assert rel_err(h, e) < 1e-5


def test_f0_landmark_matches_manual():
    """f=0(全池化)单页场景:输出=ṽ(单 landmark softmax 权重=1)。"""
    q, k, v = _case(1)
    valid = torch.full((1,), PAGE, dtype=torch.long)
    m0 = torch.zeros(HKV, 1, dtype=torch.bool)
    h = hier_attention(q, k, v, m0, valid)
    ref = torch.empty(HQ, 256)
    gsz = HQ // HKV
    for hk in range(HKV):
        for qi in range(gsz):
            ref[hk * gsz + qi] = v[hk].mean(dim=0)
    assert rel_err(h, ref) < 1e-6


def test_sharp_distribution_error_monotone():
    """尖锐分布(单页强信号)下,mass 选择器误差随 f 上升而下降。"""
    g = torch.Generator().manual_seed(3)
    gsz = HQ // HKV
    k = torch.randn(HKV, 4 * PAGE, 256, generator=g) * 0.1
    v = torch.randn(HKV, 4 * PAGE, 256, generator=g)
    q = torch.randn(HQ, 256, generator=g) * 0.5
    for hk in range(HKV):
        k[hk, :PAGE] = q[hk * gsz] * 6.0 + torch.randn(PAGE, 256, generator=g) * 0.05
    valid = torch.full((4,), PAGE, dtype=torch.long)
    e = exact_attention(q, k, v)
    errs = {}
    for frac in (0.0, 0.25, 0.5):
        b = int(round(4 * frac))
        m = (torch.ones(HKV, 4, dtype=torch.bool) if b == 4
             else (torch.zeros(HKV, 4, dtype=torch.bool) if b == 0
                   else select_mass(q, k, b, valid)))
        errs[frac] = rel_err(hier_attention(q, k, v, m, valid), e)
    assert errs[0.5] < errs[0.0]
    assert errs[0.25] < errs[0.0]


def test_partial_tail_page_consistent():
    """非页对齐尾页(pad 0)在全页精确选择下与 exact 一致且有限。"""
    q, k, v = _case(2, tail=37)
    kp, vp = pad_to_pages(k), pad_to_pages(v)
    valid = torch.tensor([PAGE, PAGE, 37], dtype=torch.long)
    masks = torch.ones(HKV, 3, dtype=torch.bool)
    e = exact_attention(q, kp, vp)
    h = hier_attention(q, kp, vp, masks, valid)
    assert torch.isfinite(h).all()
    assert rel_err(h, e) < 1e-4


def test_selectors_budget_and_disjoint():
    """双选择器:预算精确、页集布尔、mass 选择器对强信号页必选中。"""
    g = torch.Generator().manual_seed(5)
    gsz = HQ // HKV
    k = torch.randn(HKV, 8 * PAGE, 256, generator=g) * 0.1
    q = torch.randn(HQ, 256, generator=g)
    k[:, :PAGE] = q[0] * 8.0  # page 0 强信号
    valid = torch.full((8,), PAGE, dtype=torch.long)
    for sel in (select_mass, select_centroid):
        m = sel(q, k, 2, valid)
        assert m.shape == (HKV, 8) and m.dtype == torch.bool
        assert int(m.sum()) == 2 * HKV
    m = select_mass(q, k, 2, valid)
    assert bool(m[:, 0].all()), "oracle-mass 选择器必须命中强信号页"


def test_run_dump_gate_shape():
    """run_dump 合成小形状:summary/gate 结构完整,gate 语义与 rel_max 一致。"""
    g = torch.Generator().manual_seed(9)
    layers = []
    for _ in range(2):
        layers.append({
            "q": torch.randn(HQ, 256, generator=g),
            "k": torch.randn(HKV, 2 * PAGE, 256, generator=g),
            "v": torch.randn(HKV, 2 * PAGE, 256, generator=g),
        })
    import benchmarks.probes.lb2_hier_kv_oracle as m
    orig = m.load_dump
    m.load_dump = lambda path: layers
    try:
        res = m.run_dump("synthetic")
    finally:
        m.load_dump = orig
    assert res["n_layers"] == 2 and res["n_pages"] == 2
    assert res["gate"]["frac"] == 0.5 and res["gate"]["tol"] == pytest.approx(3e-2)
    assert res["gate"]["pass"] == (res["gate"]["rel_max"] <= 3e-2)
    assert "f0.5_mass" in res["summary"]
