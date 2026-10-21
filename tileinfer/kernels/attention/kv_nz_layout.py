"""NZ(fractal)K/V 布局的 host 侧参考实现(H83-③ 布局轴)。

布局轴动机(E342/H83 定谳面):现役 prefill 内核每页迭代对 ND 布局的 K/V 做
GM→L1→L0B 的分段搬运——K 走 k_l1 ND 中转后再两次切片拷贝(0:D2 / D2:D),
V 走 ``transpose=True`` 转置 DMA + 两次 (D2,BC) 二段拷贝。若 K/V cache 以
NZ fractal 布局存储,L1→L0B 可直通:

* K 以 ``(D, BC)``(即 K^T)的 fractal ``[D/16][BC/16][16][16]`` 存储,
  与 ``qk_b`` 进 ``gemm(transpose_B=True)`` 的 L0B fractal 顺序一致;
* V 以 ``(D, BC)``(即 V^T)的同一 fractal 存储,与 ``pv_b`` 的 L0B
  fractal 顺序一致(消除 transpose DMA 税 + 二段拷贝)。

本模块只做布局数学(纯 torch,无 tilelang 依赖),供:
* CPU 域往返/语义测试(契约测试,任何环境可跑);
* 设备窗内核数值对照的 pack 输入(微基准/契约)。

注意:第一版 H83 实验载体是 PV 单 pass(H83-①);NZ 直通布局轴是 H80 重写
框架的第二步,达到微基准门槛前不接入任何在役路径。
"""

from __future__ import annotations

import torch

NZ_BLOCK = 16
_FRACTAL_ND = (NZ_BLOCK, NZ_BLOCK)


def nz_shape(pages: int, kv_heads: int, page_size: int, dim: int) -> tuple:
    """NZ cache 的张量形状:(pages, kv_heads, dim/16, page_size/16, 16, 16)."""
    assert dim % NZ_BLOCK == 0, f"dim={dim} must be a multiple of {NZ_BLOCK}"
    assert page_size % NZ_BLOCK == 0, f"page_size={page_size} must be a multiple of {NZ_BLOCK}"
    return (pages, kv_heads, dim // NZ_BLOCK, page_size // NZ_BLOCK, NZ_BLOCK, NZ_BLOCK)


def pack_nd_to_nz(cache_nd: torch.Tensor) -> torch.Tensor:
    """ND cache ``(pages, page_size, kv_heads, dim)`` → NZ fractal。

    语义:每页每头存 (dim, page_size) 矩阵(即 K^T / V^T)的 fractal
    ``[dim/16][page_size/16][16][16]``,内块行优先(dim 方向为行)。
    """
    pages, page_size, kv_heads, dim = cache_nd.shape
    out = cache_nd.permute(0, 2, 3, 1)  # (P, H, D, BC)
    out = out.reshape(pages, kv_heads, dim // NZ_BLOCK, NZ_BLOCK, page_size // NZ_BLOCK, NZ_BLOCK)
    out = out.permute(0, 1, 2, 4, 3, 5)  # (P, H, D/16, BC/16, 16, 16)
    return out.contiguous()


def unpack_nz_to_nd(cache_nz: torch.Tensor) -> torch.Tensor:
    """:func:`pack_nd_to_nz` 的精确逆变换,回到 ``(pages, page_size, kv_heads, dim)``."""
    pages, kv_heads, dn, bn = cache_nz.shape[:4]
    dim, page_size = dn * NZ_BLOCK, bn * NZ_BLOCK
    # pack 的最后一步 permute 是 (0,1,2,4,3,5):(P,H,d16,di,b16,bi)→(P,H,d16,b16,di,bi);
    # 逆序先还原 (P,H,d16,di,b16,bi),reshape 回 (P,H,D,BC),再 permute 回 (P,BC,H,D)。
    out = cache_nz.permute(0, 1, 2, 4, 3, 5)  # (P, H, D/16, 16, BC/16, 16)
    out = out.reshape(pages, kv_heads, dim, page_size)  # (P, H, D, BC)
    out = out.permute(0, 3, 1, 2)  # (P, BC, H, D)
    return out.contiguous()


def nz_slice_as_transposed(cache_nz: torch.Tensor, d_stop: int) -> torch.Tensor:
    """把 NZ 布局的 ``[:d_stop, :]`` 切片展开成 (d_stop, page_size) 稠密矩阵。

    这就是 L1→L0B 直拷后 ``pv_b``/``qk_b`` 应有的逻辑内容(K^T / V^T 的
    前 d_stop 行)——CPU 域用它验证「fractal 直拷 == 转置切片」的语义。
    """
    pages, kv_heads, dn, bn = cache_nz.shape[:4]
    page_size = bn * NZ_BLOCK
    # (P, H, d_stop, page_size):按 fractal 顺序展开
    out = cache_nz[:, :, : d_stop // NZ_BLOCK]
    out = out.permute(0, 1, 2, 4, 3, 5).reshape(pages, kv_heads, d_stop, page_size)
    return out
