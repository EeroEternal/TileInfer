"""H83 契约测试:PV 单 pass 变体 + NZ K/V 布局参考实现。

CPU 域(任何环境必跑):
* NZ pack/unpack 往返一致;
* NZ 展开语义 == K^T/V^T 切片(「L1→L0B 直拷」的布局数学证明)。

设备域(等窗,skipif 无 NPU/tilelang.ascend):
* pv_single_pass=True 编译冒烟(L0 恰满是显式风险,编译拒 = H83-① 该轴判停);
* 单 pass 变体 vs 现役 D2 双 pass 数值对照(5e-3 容差:GEMM 求和顺序不同,
  非 bitwise,但远紧于 attention 精度口径 3e-2)。
"""

from __future__ import annotations

import pytest
import torch

from tileinfer.kernels.attention.kv_nz_layout import (
    NZ_BLOCK,
    nz_shape,
    nz_slice_as_transposed,
    pack_nd_to_nz,
    unpack_nz_to_nd,
)
from tileinfer.utils import is_npu_available

try:  # pragma: no cover
    import tilelang.ascend  # noqa: F401

    _HAVE_ASCEND950 = True
except Exception:  # pragma: no cover
    _HAVE_ASCEND950 = False

def _device_test(fn):
    """设备域标记:npu 标记 + 无 NPU/tilelang.ascend 时 skip(与现役设备测试同款)."""
    for mark in reversed([
        pytest.mark.npu,
        pytest.mark.skipif(not is_npu_available(), reason="no Ascend NPU available"),
        pytest.mark.skipif(not _HAVE_ASCEND950, reason="tilelang.ascend (>=0.1.15) not importable"),
    ]):
        fn = mark(fn)
    return fn

# ─────────────────────────── CPU 域:布局数学 ───────────────────────────

@pytest.mark.parametrize("pages,page_size,kv_heads,dim", [
    (3, 128, 2, 256),
    (1, 128, 2, 128),
    (5, 64, 4, 128),
])
def test_nz_roundtrip(pages, page_size, kv_heads, dim):
    g = torch.Generator().manual_seed(7)
    nd = torch.randn(pages, page_size, kv_heads, dim, generator=g)
    nz = pack_nd_to_nz(nd)
    assert nz.shape == nz_shape(pages, kv_heads, page_size, dim)
    back = unpack_nz_to_nd(nz)
    assert torch.equal(back, nd), "NZ pack/unpack roundtrip must be exact"

@pytest.mark.parametrize("dim", [128, 256])
def test_nz_slice_is_transposed_dense(dim):
    """NZ 直拷语义:fractal 展开 (d_stop, page_size) == cache^T 的对应切片。

    pv_b/qk_b 进 gemm(transpose_B=True) 需要 (DPV, BC) = V^T/K^T 行切片;
    NZ 布局若能在 L1→L0B 直拷,其逻辑内容必须逐元素等于该切片。
    """
    g = torch.Generator().manual_seed(11)
    pages, page_size, kv_heads = 2, 128, 2
    nd = torch.randn(pages, page_size, kv_heads, dim, generator=g)
    nz = pack_nd_to_nz(nd)
    for d_stop in (16, dim // 2, dim):
        got = nz_slice_as_transposed(nz, d_stop)  # (P, H, d_stop, BC)
        want = nd.permute(0, 2, 3, 1)[..., :d_stop, :]  # (P, H, d_stop, BC) of K^T
        assert torch.equal(got, want), f"NZ slice semantics broken at d_stop={d_stop}"

def test_nz_block_size_contract():
    assert NZ_BLOCK == 16
    pages, page_size, kv_heads, dim = 1, 128, 1, 32
    nd = torch.arange(page_size * dim, dtype=torch.float32).reshape(1, page_size, 1, dim)
    nz = pack_nd_to_nz(nd)
    # 首个 fractal 块 = K^T 的前 16 行(dim 方向)× 前 16 列(BC 方向),行优先
    first_block = nz[0, 0, 0, 0]
    assert torch.equal(first_block, nd[0, :16, 0, :16].T.reshape(16, 16))

# ─────────────────────────── 设备域:编译冒烟 + 数值契约 ───────────────────────────

@_device_test
@pytest.mark.parametrize("dim", [128, 256])
def test_pv_single_pass_build_smoke(dim):
    """L0 容量门存证(H83-① 判决,iter-244):

    dim=256 单 pass 的 pv_b(256,128)bf16=64KB 与 qk_b(128,128)bf16=32KB 在 L0B
    并存 = 96KB > 物理容量 64KB,Z3 调度编译拒(iter-244 实测
    ``capacity 65536 insufficient for minimum buffer usage 98304``)——PV 单 pass 轴
    按预登记红线判停(disproven)。本测试反向断言编译拒,**守卫语义**:未来若容量
    扩展或调度器支持生命周期复用使编译通过,此测试失败 = 重开信号(判据零改动)。
    dim=128 无 split(D=D2=128),退化为现役形态,正向冒烟。
    """
    from tileinfer.kernels.attention.paged_decode_ascend950 import build_prefill_kernel

    spec = dict(
        batch=1,
        kv_heads=2,
        group=8,
        dim=dim,
        page_size=128,
        num_pages_cap=128,
        num_tiles=2,
        pv_single_pass=True,
    )
    if dim == 128:
        assert build_prefill_kernel(**spec) is not None
        return
    with pytest.raises(Exception, match="l0b|capacity|insufficient"):
        build_prefill_kernel(**spec)

@_device_test
def test_pv_single_pass_numeric_contract():
    """单 pass 数值契约存证(dim=256 编译拒,H83-① 判死):

    现役 D2 双 pass 正向跑通(ref);单 pass 断言编译拒(l0b 容量),守卫语义同
    build smoke——编译一旦通过即测试失败 = 重开信号,届时本测试恢复数值对照口径
    (5e-3,GEMM 求和顺序不同非 bitwise)。
    """

    from tileinfer.kernels.attention.paged_decode_ascend950 import forward_prefill
    from tileinfer.metadata import RaggedMetadata
    from tileinfer.plan import plan_query_tiles

    device = torch.device("npu", 0)
    torch.manual_seed(3)
    batch, kv_heads, group, page_size, dim = 2, 2, 8, 128, 256
    kv_lens, qo_lens = [5 * page_size + 64, page_size + 7], [97, 33]

    indptr, last = [0], []
    for L in kv_lens:
        indptr.append(indptr[-1] + -(-L // page_size))
        last.append(page_size if L % page_size == 0 else L % page_size)
    total = indptr[-1]
    k_cache = torch.randn(total, page_size, kv_heads, dim, dtype=torch.bfloat16, device=device)
    v_cache = torch.randn(total, page_size, kv_heads, dim, dtype=torch.bfloat16, device=device)
    indices = torch.arange(total, dtype=torch.int32, device=device)
    q = 0.2 * torch.randn(
        batch, kv_heads * group, max(qo_lens), dim, dtype=torch.bfloat16, device=device
    )
    qo = torch.tensor(qo_lens, dtype=torch.int32, device=device)
    qo_indptr = torch.zeros(batch + 1, dtype=torch.int32, device=device)
    qo_indptr[1:] = torch.cumsum(qo, dim=0).to(torch.int32)
    meta = RaggedMetadata(
        kv_indptr=torch.tensor(indptr, dtype=torch.int32, device=device),
        kv_indices=indices,
        kv_last_page_len=torch.tensor(last, dtype=torch.int32, device=device),
        page_size=page_size,
        qo_indptr=qo_indptr,
    )
    page_counts = (meta.kv_indptr[1:] - meta.kv_indptr[:-1]).to(torch.int64)
    sched = plan_query_tiles(
        (qo.to(torch.int64) * group), page_counts, block_q=128, load_balance=False
    )

    ref = forward_prefill(q, k_cache, v_cache, meta, sched, qo, group=group)
    with pytest.raises(Exception, match="l0b|capacity|insufficient"):
        forward_prefill(
            q, k_cache, v_cache, meta, sched, qo, group=group, pv_single_pass=True
        )
