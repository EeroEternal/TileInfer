"""H74 设备数值验证:编译器页流水各阶段深度 vs 串行版**逐位等价**。

与 h74_sw_pipeline_bench 的区别:本文件走 pytest,覆盖边界与错误路径,
且显式断言 `err == 0`(逐位),而不是"小于容差"。

覆盖:
  1. 各 num_stages ∈ {0,2,3,4} × 多形状(batch/kv_heads/group/dim/page)逐位一致;
  2. 单页请求(num_pages=1,流水退化为单次迭代);
  3. ragged batch(不同请求页数不同);
  4. tail 页(seq_len 不是 page_size 整数倍);
  5. `num_stages=0` 与不传参走同一 cache key(默认路径零变化)。
"""

from __future__ import annotations

import pytest
import torch

from tileinfer.utils import is_npu_available

try:  # pragma: no cover
    import tilelang.ascend  # noqa: F401

    _HAVE_ASCEND950 = True
except Exception:  # pragma: no cover
    _HAVE_ASCEND950 = False

pytestmark = [
    pytest.mark.npu,
    pytest.mark.skipif(not is_npu_available(), reason="no Ascend NPU available"),
    pytest.mark.skipif(not _HAVE_ASCEND950, reason="tilelang.ascend not importable"),
]

DT = torch.bfloat16
PAGE = 128
STAGES = (2, 3, 4)


def _build_case(batch, kv_heads, group, dim, lens, seed=0):
    from tileinfer.kernels.attention.paged_decode_ascend950 import declared_pool_cap

    dev = torch.device("npu", 0)
    torch.manual_seed(seed)
    indptr, lpl = [0], []
    for L in lens:
        indptr.append(indptr[-1] + -(-L // PAGE))
        lpl.append(PAGE if L % PAGE == 0 else L % PAGE)
    total_pages = indptr[-1]
    pool = declared_pool_cap(total_pages + 8)
    k = torch.randn(pool, PAGE, kv_heads, dim, dtype=DT, device=dev)
    v = torch.randn(pool, PAGE, kv_heads, dim, dtype=DT, device=dev)
    idx = torch.arange(pool, dtype=torch.int32, device=dev)
    BR = max(32, ((group + 31) // 32) * 32)
    q = torch.randn(batch, kv_heads * BR, dim, dtype=DT, device=dev) * 0.2
    # 只保留每个 KV 头的 group 行有效,其余置零(ABI 约定)
    qv = q.view(batch, kv_heads, BR, dim)
    qv[:, :, group:] = 0
    out = torch.zeros(batch, kv_heads * BR, dim, dtype=DT, device=dev)
    meta = (
        torch.tensor(indptr, dtype=torch.int32, device=dev),
        idx,
        torch.tensor(lpl, dtype=torch.int32, device=dev),
    )
    return q, k, v, out, meta, pool


def _run(batch, kv_heads, group, dim, lens, stages, seed=0):
    from tileinfer.kernels.attention.paged_decode_ascend950 import build_decode_kernel

    q, k, v, out, (indptr, idx, lpl), pool = _build_case(
        batch, kv_heads, group, dim, lens, seed
    )
    kern = build_decode_kernel(
        batch=batch, kv_heads=kv_heads, group=group, dim=dim, page_size=PAGE,
        num_pages_cap=pool, dtype="bfloat16", out_manual=True, sw_pipeline=stages,
    )
    out.zero_()
    kern(q, k, v, indptr, idx, lpl, out)
    torch.npu.synchronize()
    return out.clone()


SHAPES = [
    # (batch, kv_heads, group, dim, lens)
    (1, 2, 8, 128, [128]),          # 单页
    (1, 2, 8, 128, [8192]),         # c1 长上下文
    (1, 2, 8, 256, [2048]),         # head_dim 256(D-split)
    (2, 2, 8, 128, [200, 512]),     # ragged + tail 页
    (8, 2, 8, 128, [128] * 8),      # c8 短
]


@pytest.mark.parametrize("shape", SHAPES, ids=[f"s{i}" for i in range(len(SHAPES))])
@pytest.mark.parametrize("stages", STAGES)
def test_sw_pipeline_bit_identical(shape, stages):
    """各流水深度必须与串行版**逐位**一致(不是"小于容差")。"""
    batch, kvh, group, dim, lens = shape
    ref = _run(batch, kvh, group, dim, lens, 0)
    got = _run(batch, kvh, group, dim, lens, stages)
    assert torch.equal(ref, got), (
        f"stages={stages} shape={shape}: 与串行版不逐位一致 "
        f"(max abs diff {(ref.float() - got.float()).abs().max().item():.6f})"
    )


def test_sw_pipeline_zero_matches_default():
    """`sw_pipeline=0` 与完全不传该参数必须产生同一 cache key(默认零变化)。"""
    from tileinfer.kernels.attention.paged_decode_ascend950 import build_decode_kernel, _KERNEL_CACHE
    import tilelang

    _KERNEL_CACHE.clear()
    orig = tilelang.compile
    tilelang.compile = lambda *a, **kw: object()
    try:
        base = dict(
            batch=1, kv_heads=2, group=8, dim=128, page_size=PAGE,
            num_pages_cap=512, dtype="bfloat16", out_manual=True,
        )
        build_decode_kernel(**base)
        k_none = next(iter(_KERNEL_CACHE))
        _KERNEL_CACHE.clear()
        build_decode_kernel(**base, sw_pipeline=0)
        k_zero = next(iter(_KERNEL_CACHE))
    finally:
        tilelang.compile = orig
        _KERNEL_CACHE.clear()
    assert k_none == k_zero


def test_sw_pipeline_positive_differs_from_default():
    """非 0 必须是不同产物(否则流水版与串行版共享一份编译结果)。"""
    from tileinfer.kernels.attention.paged_decode_ascend950 import build_decode_kernel, _KERNEL_CACHE
    import tilelang

    base = dict(
        batch=1, kv_heads=2, group=8, dim=128, page_size=PAGE,
        num_pages_cap=512, dtype="bfloat16", out_manual=True,
    )
    _KERNEL_CACHE.clear()
    orig = tilelang.compile
    tilelang.compile = lambda *a, **kw: object()
    try:
        build_decode_kernel(**base)
        k0 = next(iter(_KERNEL_CACHE))
        _KERNEL_CACHE.clear()
        build_decode_kernel(**base, sw_pipeline=3)
        k3 = next(iter(_KERNEL_CACHE))
    finally:
        tilelang.compile = orig
        _KERNEL_CACHE.clear()
    assert k0 != k3
