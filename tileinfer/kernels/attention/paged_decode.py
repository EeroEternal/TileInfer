"""Paged decode attention kernel (GQA/MQA), written in TileLang for Ascend NPU.

Layout contract
---------------
``Q``        ``[batch, kv_heads, group, dim]`` fp16, where ``group = num_qo_heads // num_kv_heads``
             is the GQA multiplicity.  The view is free (no repack): for a
             ``[batch, num_qo_heads, 1, dim]`` query tensor, query head ``kv_head * group + i``
             belongs to KV head ``kv_head``, which is the standard GQA convention.
``KCache``/``VCache``  ``[num_pages, block_size, kv_heads, dim]`` fp16 — the ``NHD`` layout used
             by vLLM-Ascend / MindIE, so no conversion is needed at the engine boundary.
``kv_indptr``/``kv_indices``  the engine-agnostic page list: logical page ``p`` of request ``b``
             lives at ``kv_indices[kv_indptr[b] + p]``.  Pages may be permuted or shared.
``kv_last_page_len``  valid tokens in the last page of each request, as **fp32**.
``tok_idx``  ``[block_size]`` fp32 constant ``0..block_size-1``, used to build the tail mask on
             the device so a partially filled last page costs no host work per step.

Design notes
------------
* One kernel block per ``(request, kv_head)``: the first GEMM has M = GQA group, N = KV page,
  K = ``head_dim``, so all query heads of a KV head are served by one block and each KV page is
  read once per request.
* Single pass over the request's pages with **online softmax** (running max/sum), one read of K
  and V.  Splitting a long request along the KV axis (``kv_tile_pages`` in the plan) is the next
  step and needs the merge kernel.
* Everything after the QK^T GEMM runs on the vector core; cube↔vector hand-off goes through the
  workspace tensors, as the Ascend memory hierarchy requires.
* The tail page is masked *before* ``exp`` with a device-computed bitmask, so padding in the
  unused page slots cannot leak into the softmax denominator — the classic silent bug in paged
  attention kernels, and the reason ``kv_last_page_len`` is a first-class input.

Implementation note (important)
-------------------------------
Two things about this file are load-bearing and non-obvious:

1. It must **not** use ``from __future__ import annotations``.  PEP 563 turns the ``T.Tensor(...)``
   annotations into source strings, which the TVM script parser then cannot evaluate — the failure
   mode is a confusing ``expected Object but got str`` from ``script.ir_builder.tir.Arg``.
2. ``T`` is imported at module scope and every shape constant used in the signature is computed
   *inside* the jit function, because the parser resolves annotation names through module globals
   plus the locals of the defining frame.  Putting them in an enclosing scope degrades the same
   way.
"""

from typing import Any, Dict

from ...utils import have_tilelang

try:  # module-scope import: see "Implementation note" above
    import tilelang
    from tilelang import DataType
    from tilelang import language as T

    _IMPORT_ERROR: Any = None
except Exception as _exc:  # pragma: no cover - only on machines without TileLang
    tilelang = None  # type: ignore[assignment]
    DataType = None  # type: ignore[assignment]
    T = None  # type: ignore[assignment]
    _IMPORT_ERROR = _exc

__all__ = ["paged_decode_gqa", "build_decode_kernel", "DecodeKernelSpec", "is_available", "clear_kernel_cache"]

_KERNEL_CACHE: Dict[tuple, Any] = {}


def is_available() -> bool:
    """True when a TileLang toolchain with an Ascend backend is importable."""
    return have_tilelang() and tilelang is not None


class DecodeKernelSpec:
    """Compile-time description of one decode kernel variant.

    Every field becomes a constant inside the generated kernel, so a change of any of them means a
    different kernel — which is exactly why it is an object with a ``key()`` rather than a bag of
    kwargs passed around.
    """

    __slots__ = ("batch", "kv_heads", "group", "dim", "block_size", "num_pages_cap")

    def __init__(
        self,
        batch: int,
        kv_heads: int,
        group: int,
        dim: int,
        block_size: int,
        num_pages_cap: int,
    ) -> None:
        self.batch = int(batch)
        self.kv_heads = int(kv_heads)
        self.group = int(group)
        self.dim = int(dim)
        self.block_size = int(block_size)
        self.num_pages_cap = int(num_pages_cap)

    def key(self) -> tuple:
        return (
            self.batch,
            self.kv_heads,
            self.group,
            self.dim,
            self.block_size,
            self.num_pages_cap,
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"DecodeKernelSpec(batch={self.batch}, kv_heads={self.kv_heads}, group={self.group}, "
            f"dim={self.dim}, block_size={self.block_size}, num_pages_cap={self.num_pages_cap})"
        )


def paged_decode_gqa(
    batch: int,
    kv_heads: int,
    group: int,
    dim: int,
    block_size: int,
    num_pages_cap: int,
):
    """Build (and JIT-compile) the decode kernel.

    The returned object is callable as
    ``kernel(q, k_cache, v_cache, kv_indptr, kv_indices, tok_idx, kv_last_page_len) -> out``;
    TileLang allocates the workspace tensors (``workspace_idx``) and returns the output
    (``out_idx``).
    """
    if tilelang is None:  # pragma: no cover
        raise RuntimeError(f"TileLang is not importable: {_IMPORT_ERROR}")
    if group < 2:
        raise ValueError(
            "the TileLang decode kernel needs at least two query heads per KV head (GQA/MQA); "
            f"got group={group}.  Use backend='reference' for pure MHA."
        )

    VEC_NUM = 2  # the two Ascend vector units the M (= GQA group) dimension is split across
    pass_configs = {
        tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
        tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    }

    @tilelang.jit(out_idx=[7], workspace_idx=[8, 9, 10], pass_configs=pass_configs)
    def _decode(
        batch: int,
        kv_heads: int,
        group: int,
        dim: int,
        block_size: int,
        num_pages_cap: int,
    ):
        # --- everything below is a *local* of the frame that defines ``main``: that is what the
        # script parser can resolve (see the module docstring) ---
        DTYPE = "float16"
        ACCUM = "float32"
        INDICES = "int32"
        block_M = group
        block_M_2 = (block_M + VEC_NUM - 1) // VEC_NUM
        L0AB_MAX_SIZE = 64 * 1024
        bytes_per_elem = DataType(DTYPE).bits // 8
        # keep one L0B operand within 64 KiB by chunking the page dimension when needed
        n_num = max(-(-(block_size * dim * bytes_per_elem) // L0AB_MAX_SIZE), 1)
        block_N = -(-block_size // n_num)
        block_DIM = -(-dim // n_num)
        kernel_blocks = batch * kv_heads
        sm_scale = float(dim) ** -0.5
        q_shape = [batch, kv_heads, group, dim]
        cache_shape = [num_pages_cap, block_size, kv_heads, dim]
        out_shape = [batch, kv_heads, group, dim]

        @T.prim_func
        def main(
            Q: T.Tensor(q_shape, DTYPE),  # type: ignore
            KCache: T.Tensor(cache_shape, DTYPE),  # type: ignore
            VCache: T.Tensor(cache_shape, DTYPE),  # type: ignore
            kv_indptr: T.Tensor([batch + 1], INDICES),  # type: ignore
            kv_indices: T.Tensor([num_pages_cap], INDICES),  # type: ignore
            tok_idx: T.Tensor([block_size], ACCUM),  # type: ignore
            kv_last_page_len: T.Tensor([batch], ACCUM),  # type: ignore
            Output: T.Tensor(out_shape, DTYPE),  # type: ignore
            workspace_1: T.Tensor([kernel_blocks, block_M, block_size], ACCUM),  # type: ignore
            workspace_2: T.Tensor([kernel_blocks, block_M, block_size], DTYPE),  # type: ignore
            workspace_3: T.Tensor([kernel_blocks, block_M, dim], ACCUM),  # type: ignore
        ):
            with T.Kernel(kernel_blocks, is_npu=True) as (cid, vid):
                bh = cid % kv_heads
                bz = cid // kv_heads

                q_l1 = T.alloc_L1([block_M, dim], DTYPE)
                k_l1 = T.alloc_L1([block_N, dim], DTYPE)
                v_l1 = T.alloc_L1([block_size, block_DIM], DTYPE)
                p_l1 = T.alloc_L1([block_M, block_size], DTYPE)

                acc_s_l0c = T.alloc_L0C([block_M, block_N], ACCUM)
                acc_o_l0c = T.alloc_L0C([block_M, block_DIM], ACCUM)

                acc_o = T.alloc_ub([block_M_2, dim], ACCUM)
                acc_o_ub = T.alloc_ub([block_M_2, dim], ACCUM)
                acc_o_half = T.alloc_ub([block_M_2, dim], DTYPE)
                acc_s_ub = T.alloc_ub([block_M_2, block_size], ACCUM)
                acc_s_half = T.alloc_ub([block_M_2, block_size], DTYPE)
                m_i = T.alloc_ub([block_M_2, 1], ACCUM)
                m_i_prev = T.alloc_ub([block_M_2, 1], ACCUM)
                m_i_2d = T.alloc_ub([block_M_2, block_size], ACCUM)
                m_i_prev_2d = T.alloc_ub([block_M_2, dim], ACCUM)
                sumexp = T.alloc_ub([block_M_2, 1], ACCUM)
                sumexp_i = T.alloc_ub([block_M_2, 1], ACCUM)
                sumexp_2d = T.alloc_ub([block_M_2, dim], ACCUM)
                tok_ub = T.alloc_ub([block_size], ACCUM)
                tail_mask = T.alloc_ub([block_size // 8], "uint8")

                num_pages = kv_indptr[bz + 1] - kv_indptr[bz]
                page_base = kv_indptr[bz]
                last_len = kv_last_page_len[bz]

                T.tile.fill(acc_o, 0.0)
                T.tile.fill(sumexp, 0.0)
                T.tile.fill(m_i, -T.infinity(ACCUM))

                T.copy(Q[bz, bh, 0:block_M, 0:dim], q_l1)
                T.copy(tok_idx[0:block_size], tok_ub)

                for page in T.serial(num_pages):
                    pid = kv_indices[page_base + page]

                    for n_i in T.serial(n_num):
                        T.copy(
                            KCache[pid, n_i * block_N : (n_i + 1) * block_N, bh, 0:block_DIM],
                            k_l1,
                        )
                        T.gemm_v0(q_l1, k_l1, acc_s_l0c, transpose_B=True, init=True)
                        T.copy(
                            acc_s_l0c,
                            workspace_1[cid, 0:block_M, n_i * block_N : (n_i + 1) * block_N],
                        )

                    # --- vector core: masked online softmax for this page ---
                    T.copy(m_i, m_i_prev)
                    T.copy(
                        workspace_1[cid, vid * block_M_2 : (vid + 1) * block_M_2, 0:block_size],
                        acc_s_ub,
                    )
                    T.tile.mul(acc_s_ub, acc_s_ub, sm_scale)

                    # on the last page only the first ``last_len`` slots hold real tokens
                    limit = T.if_then_else(page == num_pages - 1, last_len, T.float32(block_size))
                    T.tile.compare(tail_mask, tok_ub, limit, "LT")
                    for row in T.serial(block_M_2):
                        T.tile.select(
                            acc_s_ub[row, :],
                            tail_mask,
                            acc_s_ub[row, :],
                            -T.infinity(ACCUM),
                            "VSEL_TENSOR_SCALAR_MODE",
                        )

                    T.reduce_max(acc_s_ub, m_i, dim=-1)
                    T.tile.max(m_i, m_i, m_i_prev)
                    T.tile.sub(m_i_prev, m_i_prev, m_i)
                    T.tile.exp(m_i_prev, m_i_prev)
                    T.tile.broadcast(m_i_2d, m_i)
                    T.tile.sub(acc_s_ub, acc_s_ub, m_i_2d)
                    T.tile.exp(acc_s_ub, acc_s_ub)
                    T.reduce_sum(acc_s_ub, sumexp_i, dim=-1)
                    T.tile.mul(sumexp, sumexp, m_i_prev)
                    T.tile.add(sumexp, sumexp, sumexp_i)

                    T.tile.cast(acc_s_half, acc_s_ub, "CAST_NONE", block_M_2 * block_size)
                    T.copy(
                        acc_s_half,
                        workspace_2[cid, vid * block_M_2 : (vid + 1) * block_M_2, 0:block_size],
                    )

                    # --- cube: P @ V ---
                    T.copy(workspace_2[cid, 0:block_M, 0:block_size], p_l1)
                    for n_i in T.serial(n_num):
                        T.copy(
                            VCache[pid, 0:block_size, bh, n_i * block_DIM : (n_i + 1) * block_DIM],
                            v_l1,
                        )
                        T.gemm_v0(p_l1, v_l1, acc_o_l0c, init=True)
                        T.copy(
                            acc_o_l0c,
                            workspace_3[cid, 0:block_M, n_i * block_DIM : (n_i + 1) * block_DIM],
                        )

                    T.tile.broadcast(m_i_prev_2d, m_i_prev)
                    T.tile.mul(acc_o, acc_o, m_i_prev_2d)
                    T.copy(
                        workspace_3[cid, vid * block_M_2 : (vid + 1) * block_M_2, 0:dim], acc_o_ub
                    )
                    T.tile.add(acc_o, acc_o, acc_o_ub)

                T.tile.broadcast(sumexp_2d, sumexp)
                T.tile.div(acc_o, acc_o, sumexp_2d)
                T.tile.cast(acc_o_half, acc_o, "CAST_NONE", block_M_2 * dim)
                T.copy(
                    acc_o_half,
                    Output[bz, bh, vid * block_M_2 : (vid + 1) * block_M_2, 0:dim],
                )

        return main

    return _decode(batch, kv_heads, group, dim, block_size, num_pages_cap)


def build_decode_kernel(spec: DecodeKernelSpec):
    """Cached entry point: compile at most once per :class:`DecodeKernelSpec`."""
    key = spec.key()
    kernel = _KERNEL_CACHE.get(key)
    if kernel is None:
        kernel = paged_decode_gqa(
            spec.batch, spec.kv_heads, spec.group, spec.dim, spec.block_size, spec.num_pages_cap
        )
        _KERNEL_CACHE[key] = kernel
    return kernel


def clear_kernel_cache() -> None:
    _KERNEL_CACHE.clear()
