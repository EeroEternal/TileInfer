"""Paged decode attention for Ascend 950, written in the official ``tilelang.ascend`` dialect.

This is the port of ``paged_decode.py`` (which targets the ``tile-ai/tilelang-ascend`` fork) to the
backend that actually runs on the reference machine: the Ascend 950 support that ships inside the
PyPI ``tilelang`` wheel (``target="ascend"``, arch ``dav-3510``).

Design notes
------------
* One kernel block per ``(request, kv_head)``.  The M dimension of the QK^T GEMM is the GQA group,
  i.e. all query heads that share a KV head, so each KV page is read once per request.
* ``dual_copy`` splits the M dimension across the two AIVs, so every UB buffer that receives cube
  output holds ``ROWS = BR // 2`` rows — the same convention as upstream's
  ``examples/ascend/flash_attention/core.py``.  L1/L0 buffers keep the full ``BR`` rows.
* The softmax is written in **SIMT** (``T.SimtVF`` + ``T.alloc_reducer``) rather than with the
  hand-tuned ``T.simd.*`` micro-API: it is a few lines instead of a hundred, which is the right
  trade for the first correct version.  Vector time is not the bottleneck for decode anyway.
* The tail page is masked **before** ``exp`` by clamping the visible column count to
  ``kv_last_page_len`` — the same rule the fork kernel and the reference implementation follow.
* The M dimension is **padded to a multiple of 32**: the ND→NZ copy template asserts
  ``ROWS % 16 == 0`` and ``ROWS = BR // 2``, so a bare GQA group of 8 (a very common shape) does
  not compile.  The caller passes zero-padded ``Q``/``Out`` of ``[batch, kv_heads * BR, dim]``
  and slices the real ``group`` rows back out; padded rows cost cube time but no correctness.

Note: this module must **not** use ``from __future__ import annotations``.  The eager builder calls
``get_type_hints`` on the kernel, and PEP 563 would turn the ``T.Tensor(...)`` annotations into
strings that can no longer resolve the enclosing function's locals.
"""

import math
import os
from typing import List

import torch
import tilelang
import tilelang.ascend.language as T

__all__ = [
    "paged_decode_ascend950",
    "paged_decode_split",
    "paged_decode_merge",
    "build_decode_kernel",
    "paged_prefill_ascend950",
    "build_prefill_kernel",
    "paged_prefill_split_ascend950",
    "build_prefill_split_kernel",
    "plan_prefill_split_ctx",
    "forward_prefill_split",
    "build_split_kernel",
    "build_merge_kernel",
    "build_tile_slots",
    "padded_rows",
    "declared_pool_cap",
    "forward",
    "forward_split",
    "forward_prefill",
]


# ---------------------------------------------------------------------------
# T5: opt-in SIMD softmax.
#
# ``TILEINFER_SIMD_SOFTMAX=1`` swaps the per-page online softmax in the attention kernels from the
# SIMT reducer form to the packed ``T.simd.*`` form.  Off by default: the SIMD path is numerically
# equivalent (device-verified, ``benchmarks/probes/simd_softmax_*_device.py``) and ~2.6x faster on
# the decode shape, but it keeps vector time off the critical path only on the Ascend 950 shapes the
# probes cover, so it is opt-in until the A/B serves it.
#
# Backend constraints the SIMD path must respect (found on device):
#   * a ``T.SimdVF`` region cannot read global memory — ``kv_last_page_len`` is staged into UB first;
#   * bisheng cannot lower f32 ``select``/``sitofp`` — the tail mask compares an **int32** lane ramp
#     against an int scalar (``vci(int32)`` + the ``vcmps`` int overload), never a float ternary;
#   * a ``vsts`` of a whole vector register needs a full-register destination; per-row running state
#     is written with ``ONEPT_B32`` (one element) and read back with ``BRC_B32`` (broadcast);
#   * the per-row ``p`` is emitted as **f32** and cast to the compute dtype by the existing SIMT
#     epilogue, because a SIMD f32->bf16 store interleaves lanes.
# ---------------------------------------------------------------------------


def _simd_softmax_enabled() -> bool:
    """True when the opt-in SIMD softmax path is requested (``TILEINFER_SIMD_SOFTMAX=1``)."""
    return os.environ.get("TILEINFER_SIMD_SOFTMAX", "").strip() not in ("", "0", "false", "False")


def _page_pipeline_enabled() -> bool:
    """True when the opt-in double-buffered page pipeline is requested.

    ``TILEINFER_PAGE_PIPELINE=1`` selects :func:`paged_decode_pipeline` as the plain-path decode
    kernel.  Unlike the SIMD softmax switch this is *not* read at trace time inside the kernel
    body: the backend passes it as an explicit spec argument, so it lands in the kernel cache key
    and can never silently mix traces built under different settings (E116-era lesson).
    """
    return os.environ.get("TILEINFER_PAGE_PIPELINE", "").strip() not in ("", "0", "false", "False")


def _sw_pipeline_stages() -> int:
    """H74: compiler-scheduled page-loop stages (``TILEINFER_SW_PIPELINE=N``, 0 = off).

    Same cache-key discipline as :func:`_page_pipeline_enabled`: the value is passed as an
    explicit spec argument so traces built under different depths can never mix.
    """
    try:
        return max(0, int(os.environ.get("TILEINFER_SW_PIPELINE", "0") or 0))
    except ValueError:
        return 0


def paged_decode_pipeline(
    batch: int,
    kv_heads: int,
    group: int,
    dim: int,
    page_size: int,
    num_pages_cap: int,
    dtype: str = "bfloat16",
    threads: int = 128,
):
    """Double-buffered variant of :func:`paged_decode_ascend950` (H42-③).

    Math is bit-identical to the plain kernel; the only structural change is the page loop.  The
    plain kernel serialises, per page, ``K copy → QK → softmax → P copy → V copy → PV`` — every
    global→L1 copy stalls the Cube/Vector chain (~30µs/page at 8K ctx, E116, with a ~12x
    bandwidth headroom).  This variant keeps **two** K/V L1 buffer pairs and issues the page
    ``p+1`` copies *before* the page-``p`` compute chain into the spare pair, so the MTE2 copy
    latency of the next page hides under the Cube+Vector work of the current one.

    Semantics: ``T.copy`` is issued in source order; the copies target a buffer pair the current
    iteration never reads, so no data dependency is reordered — correctness does not rely on any
    scheduling promise (worst case the copies degenerate to synchronous and the kernel matches
    the plain form).  A zero-length request copies page 0 once into scratch (page 0 always exists
    in the declared pool) and never consumes it — safe padding semantics, same as the plain
    kernel's zero-trip loop.
    """
    BR = max(32, ((group + 31) // 32) * 32)
    BC = page_size
    D = dim
    ROWS = BR // 2
    accum = "float32"
    scale = 1.0 / math.sqrt(dim)
    num_blocks = batch * kv_heads

    assert BR % 2 == 0, "dual_copy splits the M dimension across the two AIVs, so group must be even"
    assert dim in (128, 256), "the Ascend 950 attention path is tuned for head_dim 128 or 256"
    assert page_size % 2 == 0

    @T.prim_func
    def main(
        Q: T.Tensor((batch, kv_heads * BR, dim), dtype),
        KCache: T.Tensor((num_pages_cap, page_size, kv_heads, dim), dtype),
        VCache: T.Tensor((num_pages_cap, page_size, kv_heads, dim), dtype),
        kv_indptr: T.Tensor((batch + 1,), "int32"),
        kv_indices: T.Tensor((num_pages_cap,), "int32"),
        kv_last_page_len: T.Tensor((batch,), "int32"),
        Out: T.Tensor((batch, kv_heads * BR, dim), dtype),
    ):
        with T.Kernel(num_blocks) as bx:
            b = bx // kv_heads
            bh = bx % kv_heads

            q_l1 = T.alloc_l1((BR, D), dtype)
            k_l1 = T.alloc_l1((2, BC, D), dtype)
            v_l1 = T.alloc_l1((2, D, BC), dtype)  # V stored transposed, as in the upstream FA
            p_l1 = T.alloc_l1((BR, BC), dtype)

            D2 = min(D, 128)
            qk_a = T.alloc_l0a((BR, D2), dtype)
            qk_b = T.alloc_l0b((BC, D2), dtype)
            pv_a = T.alloc_l0a((BR, BC), dtype)
            pv_b = T.alloc_l0b((D2, BC), dtype)
            qk_acc = T.alloc_l0c((BR, BC), accum)
            pv_acc0 = T.alloc_l0c((BR, D2), accum)
            if D > D2:
                pv_acc1 = T.alloc_l0c((BR, D2), accum)

            s_ub = T.alloc_shared((ROWS, BC), accum)
            p_ub = T.alloc_shared((ROWS, BC), dtype)
            o_ub0 = T.alloc_shared((ROWS, D2), accum)
            o_tmp0 = T.alloc_shared((ROWS, D2), accum)
            out_ub0 = T.alloc_shared((ROWS, D2), dtype)
            if D > D2:
                o_ub1 = T.alloc_shared((ROWS, D2), accum)
                o_tmp1 = T.alloc_shared((ROWS, D2), accum)
                out_ub1 = T.alloc_shared((ROWS, D2), dtype)
            m_ub = T.alloc_shared((ROWS,), accum)
            l_ub = T.alloc_shared((ROWS,), accum)
            if _simd_softmax_enabled():
                a_ub = T.alloc_shared((ROWS,), accum)
                ll_ub = T.alloc_shared((1,), "int32")

            num_pages = kv_indptr[b + 1] - kv_indptr[b]

            T.copy(Q[b, bh * BR : (bh + 1) * BR, 0:D], q_l1)
            T.fill(m_ub, T.float32(-1e30))
            T.fill(l_ub, T.float32(0))
            T.fill(o_ub0, T.float32(0))
            if D > D2:
                T.fill(o_ub1, T.float32(0))
            if _simd_softmax_enabled():
                T.copy(kv_last_page_len[b : b + 1], ll_ub)

            # ---- prologue: stage page 0 into buffer 0 ----
            pid0 = kv_indices[kv_indptr[b]]
            T.copy(KCache[pid0, 0:BC, bh, 0:D], k_l1[0, 0:BC, 0:D])
            T.copy(VCache[pid0, 0:BC, bh, 0:D], v_l1[0, 0:D, 0:BC], transpose=True)

            for p in T.serial(num_pages):
                # ---- prefetch page p+1 into the spare pair (issued before page-p compute) ----
                if p + 1 < num_pages:
                    pid_n = kv_indices[kv_indptr[b] + p + 1]
                    T.copy(KCache[pid_n, 0:BC, bh, 0:D], k_l1[(p + 1) % 2, 0:BC, 0:D])
                    T.copy(VCache[pid_n, 0:BC, bh, 0:D], v_l1[(p + 1) % 2, 0:D, 0:BC], transpose=True)

                # ---- scores = Q @ K^T from buffer (p % 2) ----
                T.copy(q_l1[0:BR, 0:D2], qk_a)
                T.copy(k_l1[p % 2, 0:BC, 0:D2], qk_b)
                T.gemm(qk_a, qk_b, qk_acc, transpose_B=True, clear_accum=True)
                if D > D2:
                    T.copy(q_l1[0:BR, D2:D], qk_a)
                    T.copy(k_l1[p % 2, 0:BC, D2:D], qk_b)
                    T.gemm(qk_a, qk_b, qk_acc, transpose_B=True)
                T.dual_copy(qk_acc, s_ub)

                # ---- online softmax (identical to the plain kernel) ----
                if _simd_softmax_enabled():
                    limit = T.if_then_else(p + 1 < num_pages, BC, ll_ub[0])
                    with T.SimdVF():
                        for r in T.serial(ROWS):
                            idx0 = T.simd.vci(0, "int32", order="INC_ORDER")
                            idx1 = T.simd.vci(64, "int32", order="INC_ORDER")
                            sc = T.simd.vdup(T.float32(scale), "float32")
                            a0 = T.simd.vmul(T.simd.vld(s_ub[r, 0], dist="NORM"), sc)
                            a1 = T.simd.vmul(T.simd.vld(s_ub[r, 64], dist="NORM"), sc)
                            neg = T.simd.vdup(T.float32(-1e30), "float32")
                            a0 = T.simd.vsel(a0, neg, T.simd.vcmps(idx0, limit, op="lt"))
                            a1 = T.simd.vsel(a1, neg, T.simd.vcmps(idx1, limit, op="lt"))
                            pm = T.simd.vdupv(T.simd.vld(m_ub[r], dist="BRC_B32"), pos="POS_LOWEST")
                            nm = T.simd.vmax(pm, T.simd.vmax(T.simd.vcmax(a0), T.simd.vcmax(a1)))
                            nml = T.simd.vdupv(nm, pos="POS_LOWEST")
                            al = T.simd.vexp(T.simd.vsub(pm, nml))
                            e0 = T.simd.vexp(T.simd.vsub(a0, nml))
                            e1 = T.simd.vexp(T.simd.vsub(a1, nml))
                            lp = T.simd.vdupv(T.simd.vld(l_ub[r], dist="BRC_B32"), pos="POS_LOWEST")
                            ln = T.simd.vadd(
                                T.simd.vmul(lp, al),
                                T.simd.vadd(T.simd.vcadd(e0), T.simd.vcadd(e1)),
                            )
                            T.simd.vsts(m_ub[r], nml, dist="ONEPT_B32")
                            T.simd.vsts(l_ub[r], T.simd.vdupv(ln, pos="POS_LOWEST"), dist="ONEPT_B32")
                            T.simd.vsts(a_ub[r], T.simd.vdupv(al, pos="POS_LOWEST"), dist="ONEPT_B32")
                            T.simd.vsts(s_ub[r, 0], e0)
                            T.simd.vsts(s_ub[r, 64], e1)
                    with T.SimtVF(threads=threads):
                        for r, d in T.Parallel(ROWS, D2):
                            o_ub0[r, d] = o_ub0[r, d] * a_ub[r]
                        if D > D2:
                            for r, d in T.Parallel(ROWS, D2):
                                o_ub1[r, d] = o_ub1[r, d] * a_ub[r]
                        for r, c in T.Parallel(ROWS, BC):
                            p_ub[r, c] = T.Cast(dtype, s_ub[r, c])
                else:
                    limit = T.if_then_else(p + 1 < num_pages, BC, kv_last_page_len[b])
                    with T.SimtVF(threads=threads):
                        s_frag = T.alloc_fragment((ROWS, BC), accum)
                        for r, c in T.Parallel(ROWS, BC):
                            s_frag[r, c] = T.if_then_else(
                                c < limit, s_ub[r, c] * T.float32(scale), T.float32(-1e30)
                            )

                        row_max = T.alloc_reducer((ROWS,), accum, op="max")
                        T.reducer_init(row_max)
                        for r, c in T.Parallel(ROWS, BC):
                            T.reducer_update(row_max[r], s_frag[r, c])
                        new_m = T.alloc_fragment((ROWS,), accum)
                        T.finalize_reducer(row_max, new_m)

                        alpha = T.alloc_fragment((ROWS,), accum)
                        for r in T.Parallel(ROWS):
                            alpha[r] = T.exp(m_ub[r] - T.max(m_ub[r], new_m[r]))
                            m_ub[r] = T.max(m_ub[r], new_m[r])

                        row_sum = T.alloc_reducer((ROWS,), accum, op="sum")
                        T.reducer_init(row_sum)
                        for r, c in T.Parallel(ROWS, BC):
                            s_frag[r, c] = T.exp(s_frag[r, c] - m_ub[r])
                            T.reducer_update(row_sum[r], s_frag[r, c])
                        partial = T.alloc_fragment((ROWS,), accum)
                        T.finalize_reducer(row_sum, partial)

                        for r in T.Parallel(ROWS):
                            l_ub[r] = l_ub[r] * alpha[r] + partial[r]
                        for r, d in T.Parallel(ROWS, D2):
                            o_ub0[r, d] = o_ub0[r, d] * alpha[r]
                        if D > D2:
                            for r, d in T.Parallel(ROWS, D2):
                                o_ub1[r, d] = o_ub1[r, d] * alpha[r]
                        for r, c in T.Parallel(ROWS, BC):
                            p_ub[r, c] = T.Cast(dtype, s_frag[r, c])

                # ---- o += P @ V from buffer (p % 2) ----
                T.dual_copy(p_ub[0:ROWS, 0:BC], p_l1[0:BR, 0:BC])
                T.copy(p_l1[0:BR, 0:BC], pv_a)
                T.copy(v_l1[p % 2, 0:D2, 0:BC], pv_b)
                T.gemm(pv_a, pv_b, pv_acc0, transpose_B=True, clear_accum=True)
                if D > D2:
                    T.copy(v_l1[p % 2, D2:D, 0:BC], pv_b)
                    T.gemm(pv_a, pv_b, pv_acc1, transpose_B=True, clear_accum=True)
                T.dual_copy(pv_acc0, o_tmp0)
                if D > D2:
                    T.dual_copy(pv_acc1, o_tmp1)
                with T.SimtVF(threads=threads):
                    for r, d in T.Parallel(ROWS, D2):
                        o_ub0[r, d] = o_ub0[r, d] + o_tmp0[r, d]
                    if D > D2:
                        for r, d in T.Parallel(ROWS, D2):
                            o_ub1[r, d] = o_ub1[r, d] + o_tmp1[r, d]

            # ---- normalise and store ----
            with T.SimtVF(threads=threads):
                for r, d in T.Parallel(ROWS, D2):
                    out_ub0[r, d] = T.Cast(dtype, o_ub0[r, d] / l_ub[r])
                if D > D2:
                    for r, d in T.Parallel(ROWS, D2):
                        out_ub1[r, d] = T.Cast(dtype, o_ub1[r, d] / l_ub[r])
            T.dual_copy(out_ub0[0:ROWS, 0:D2], Out[b, bh * BR : (bh + 1) * BR, 0:D2])
            if D > D2:
                T.dual_copy(out_ub1[0:ROWS, 0:D2], Out[b, bh * BR : (bh + 1) * BR, D2:D])

    return main


def paged_decode_ascend950(
    batch: int,
    kv_heads: int,
    group: int,
    dim: int,
    page_size: int,
    num_pages_cap: int,
    dtype: str = "bfloat16",
    threads: int = 128,
    pipeline: bool = False,
    sw_pipeline: int = 0,
):
    """Build the paged decode attention ``T.prim_func``.

    ``num_pages_cap`` is the *declared* capacity (:func:`declared_pool_cap`): it sizes the
    ``kv_indices`` ABI slot and the K/V page extents, but never equals the real pool — the
    caller presents the pool through :func:`pool_view` and no page outside the real pool is
    ever dereferenced.

    Layouts (chosen to match what serving engines already allocate, see ``docs/architecture.md``):

    ``Q``        ``[batch, kv_heads * BR, dim]`` — **padded**: query head ``kv_head * group + i``
                 belongs to KV head ``kv_head``, and rows ``group..BR-1`` of each KV head must be
                 zero (the caller pads and slices).
    ``Out``      same padded shape; only the first ``group`` rows of each KV head are meaningful.
    ``KCache`` / ``VCache``  ``[num_pages_cap, page_size, kv_heads, dim]`` (NHD) — declared at
                 the *declared* capacity (:func:`declared_pool_cap`), which covers but does not
                 equal the real pool (L2 pool-agnostic ABI).  The caller presents the pool
                 through :func:`pool_view`; pages beyond the real pool are never dereferenced
                 (every access is bounded by ``kv_indptr``/``kv_indices``).
    ``kv_indptr`` / ``kv_indices`` / ``kv_last_page_len``  the engine-agnostic page list.

    ``pipeline=True`` (H42-③, opt-in via ``TILEINFER_PAGE_PIPELINE=1``) returns the
    double-buffered page-pipeline variant :func:`paged_decode_pipeline` instead — identical
    math, K/V L1 double-buffering so the next page's global→L1 copies overlap the current
    page's compute.  The flag is part of the caller's build spec, hence of the kernel cache key.

    ``sw_pipeline=N`` (H74, opt-in via ``TILEINFER_SW_PIPELINE=N``) keeps this plain kernel but
    turns its page loop into ``T.Pipelined(num_stages=N)`` so **the compiler** stages the
    global→L1 K/V copies across iterations.  This is the approach DeepInfer's Ascend kernels
    all use (``T.Persistent``/``T.Pipelined`` with ``num_stages``); the hand-written
    double-buffer above (``pipeline=True``) was measured as a regression (E296), whereas the
    compiler-scheduled form is not the same transformation.  ``0`` keeps the old serial loop
    bit-for-bit.
    """
    if pipeline:
        return paged_decode_pipeline(
            batch=batch,
            kv_heads=kv_heads,
            group=group,
            dim=dim,
            page_size=page_size,
            num_pages_cap=num_pages_cap,
            dtype=dtype,
            threads=threads,
        )
    # M of the QK^T GEMM.  Padded so that the per-AIV row count is a multiple of 16: the ND->NZ
    # copy template asserts `ROWS % 16 == 0` (see the module docstring).
    BR = max(32, ((group + 31) // 32) * 32)
    BC = page_size  # N of the QK^T GEMM (one KV page)
    D = dim
    ROWS = BR // 2  # per-AIV rows: `dual_copy` splits the M dimension across two AIVs
    accum = "float32"
    scale = 1.0 / math.sqrt(dim)
    num_blocks = batch * kv_heads

    assert BR % 2 == 0, "dual_copy splits the M dimension across the two AIVs, so group must be even"
    assert dim in (128, 256), "the Ascend 950 attention path is tuned for head_dim 128 or 256"
    assert page_size % 2 == 0

    @T.prim_func
    def main(
        Q: T.Tensor((batch, kv_heads * BR, dim), dtype),
        KCache: T.Tensor((num_pages_cap, page_size, kv_heads, dim), dtype),
        VCache: T.Tensor((num_pages_cap, page_size, kv_heads, dim), dtype),
        kv_indptr: T.Tensor((batch + 1,), "int32"),
        kv_indices: T.Tensor((num_pages_cap,), "int32"),
        kv_last_page_len: T.Tensor((batch,), "int32"),
        Out: T.Tensor((batch, kv_heads * BR, dim), dtype),
    ):
        with T.Kernel(num_blocks) as bx:
            b = bx // kv_heads
            bh = bx % kv_heads

            q_l1 = T.alloc_l1((BR, D), dtype)
            k_l1 = T.alloc_l1((BC, D), dtype)
            v_l1 = T.alloc_l1((D, BC), dtype)  # V stored transposed, as in the upstream FA
            p_l1 = T.alloc_l1((BR, BC), dtype)

            # D-split: head_dim 256 exceeds the 64 KB L0A/L0B capacity with single GEMMs (Z3
            # scheduling refuses 98304/131072 B), so both products are split along D into D2=128
            # halves: QK^T accumulates two GEMMs into one accumulator, P@V keeps two accumulators.
            # For dim <= 128 every `if D > D2` branch is compile-time dead and the buffers below
            # are exactly the pre-split ones.
            D2 = min(D, 128)
            qk_a = T.alloc_l0a((BR, D2), dtype)
            qk_b = T.alloc_l0b((BC, D2), dtype)
            pv_a = T.alloc_l0a((BR, BC), dtype)
            pv_b = T.alloc_l0b((D2, BC), dtype)
            qk_acc = T.alloc_l0c((BR, BC), accum)
            pv_acc0 = T.alloc_l0c((BR, D2), accum)
            if D > D2:
                pv_acc1 = T.alloc_l0c((BR, D2), accum)

            s_ub = T.alloc_shared((ROWS, BC), accum)
            p_ub = T.alloc_shared((ROWS, BC), dtype)
            o_ub0 = T.alloc_shared((ROWS, D2), accum)
            o_tmp0 = T.alloc_shared((ROWS, D2), accum)
            out_ub0 = T.alloc_shared((ROWS, D2), dtype)
            if D > D2:
                o_ub1 = T.alloc_shared((ROWS, D2), accum)
                o_tmp1 = T.alloc_shared((ROWS, D2), accum)
                out_ub1 = T.alloc_shared((ROWS, D2), dtype)
            m_ub = T.alloc_shared((ROWS,), accum)
            l_ub = T.alloc_shared((ROWS,), accum)
            if _simd_softmax_enabled():
                # T5: per-row rescale factor and staged per-request last-page length (SIMD_VF cannot
                # read global memory); the fp32 weights reuse `s_ub` in place.
                a_ub = T.alloc_shared((ROWS,), accum)
                ll_ub = T.alloc_shared((1,), "int32")

            num_pages = kv_indptr[b + 1] - kv_indptr[b]

            T.copy(Q[b, bh * BR : (bh + 1) * BR, 0:D], q_l1)
            T.fill(m_ub, T.float32(-1e30))
            T.fill(l_ub, T.float32(0))
            T.fill(o_ub0, T.float32(0))
            if D > D2:
                T.fill(o_ub1, T.float32(0))
            if _simd_softmax_enabled():
                T.copy(kv_last_page_len[b : b + 1], ll_ub)  # stage global -> UB for SimdVF

            # H74: deepseek/DeepInfer 的 Ascend 算子全部把页/任务循环写成
            # `T.Persistent`/`T.Pipelined(..., num_stages=N)`,由**编译器**把 GM→L1 的
            # K/V 拷贝排到上一轮计算之下。这里用同一手法替换纯串行页循环。
            #
            # `num_stages=0` 时 tilelang 明确不启用流水(docstring:"pipeline will not be
            # enabled"),因此这一种写法同时覆盖"默认关闭"与"开启 N 阶段"两种情况,
            # 不需要在 for 语句处写 Python 分支(eager builder 不允许把 frame 先赋给
            # 变量再迭代,且局部函数体在重放时不可见)。
            for p in T.Pipelined(num_pages, num_stages=sw_pipeline):
                pid = kv_indices[kv_indptr[b] + p]

                # ---- scores = Q @ K^T, then cube -> UB ----
                T.copy(KCache[pid, 0:BC, bh, 0:D], k_l1)
                T.copy(q_l1[0:BR, 0:D2], qk_a)
                T.copy(k_l1[0:BC, 0:D2], qk_b)
                T.gemm(qk_a, qk_b, qk_acc, transpose_B=True, clear_accum=True)
                if D > D2:
                    T.copy(q_l1[0:BR, D2:D], qk_a)
                    T.copy(k_l1[0:BC, D2:D], qk_b)
                    T.gemm(qk_a, qk_b, qk_acc, transpose_B=True)
                T.dual_copy(qk_acc, s_ub)

                # ---- online softmax ----
                # Only the last page can be partially filled; for every other page all BC columns
                # are real tokens.  Masking before `exp` is what keeps padding out of the
                # denominator.
                if _simd_softmax_enabled():
                    # T5 packed-SIMD online softmax: row max/sum via `vcmax`/`vcadd` (+`vmax`/`vadd`
                    # to merge the two registers of a 128-wide page), tail mask via an int32 lane
                    # ramp, running state in `(ROWS,)` read with `BRC_B32`/written with `ONEPT_B32`.
                    # Emits fp32 weights into `s_ub`; the cast to `dtype` and the output rescale
                    # stay below (unchanged).
                    limit = T.if_then_else(p + 1 < num_pages, BC, ll_ub[0])
                    with T.SimdVF():
                        for r in T.serial(ROWS):
                            idx0 = T.simd.vci(0, "int32", order="INC_ORDER")
                            idx1 = T.simd.vci(64, "int32", order="INC_ORDER")
                            sc = T.simd.vdup(T.float32(scale), "float32")
                            a0 = T.simd.vmul(T.simd.vld(s_ub[r, 0], dist="NORM"), sc)
                            a1 = T.simd.vmul(T.simd.vld(s_ub[r, 64], dist="NORM"), sc)
                            neg = T.simd.vdup(T.float32(-1e30), "float32")
                            a0 = T.simd.vsel(a0, neg, T.simd.vcmps(idx0, limit, op="lt"))
                            a1 = T.simd.vsel(a1, neg, T.simd.vcmps(idx1, limit, op="lt"))
                            pm = T.simd.vdupv(T.simd.vld(m_ub[r], dist="BRC_B32"), pos="POS_LOWEST")
                            nm = T.simd.vmax(pm, T.simd.vmax(T.simd.vcmax(a0), T.simd.vcmax(a1)))
                            nml = T.simd.vdupv(nm, pos="POS_LOWEST")
                            al = T.simd.vexp(T.simd.vsub(pm, nml))
                            e0 = T.simd.vexp(T.simd.vsub(a0, nml))
                            e1 = T.simd.vexp(T.simd.vsub(a1, nml))
                            lp = T.simd.vdupv(T.simd.vld(l_ub[r], dist="BRC_B32"), pos="POS_LOWEST")
                            ln = T.simd.vadd(
                                T.simd.vmul(lp, al),
                                T.simd.vadd(T.simd.vcadd(e0), T.simd.vcadd(e1)),
                            )
                            T.simd.vsts(m_ub[r], nml, dist="ONEPT_B32")
                            T.simd.vsts(l_ub[r], T.simd.vdupv(ln, pos="POS_LOWEST"), dist="ONEPT_B32")
                            T.simd.vsts(a_ub[r], T.simd.vdupv(al, pos="POS_LOWEST"), dist="ONEPT_B32")
                            T.simd.vsts(s_ub[r, 0], e0)
                            T.simd.vsts(s_ub[r, 64], e1)
                    with T.SimtVF(threads=threads):
                        for r, d in T.Parallel(ROWS, D2):
                            o_ub0[r, d] = o_ub0[r, d] * a_ub[r]
                        if D > D2:
                            for r, d in T.Parallel(ROWS, D2):
                                o_ub1[r, d] = o_ub1[r, d] * a_ub[r]
                        for r, c in T.Parallel(ROWS, BC):
                            p_ub[r, c] = T.Cast(dtype, s_ub[r, c])
                else:
                    limit = T.if_then_else(p + 1 < num_pages, BC, kv_last_page_len[b])
                    with T.SimtVF(threads=threads):
                        s_frag = T.alloc_fragment((ROWS, BC), accum)
                        for r, c in T.Parallel(ROWS, BC):
                            s_frag[r, c] = T.if_then_else(
                                c < limit, s_ub[r, c] * T.float32(scale), T.float32(-1e30)
                            )

                        row_max = T.alloc_reducer((ROWS,), accum, op="max")
                        T.reducer_init(row_max)
                        for r, c in T.Parallel(ROWS, BC):
                            T.reducer_update(row_max[r], s_frag[r, c])
                        new_m = T.alloc_fragment((ROWS,), accum)
                        T.finalize_reducer(row_max, new_m)

                        alpha = T.alloc_fragment((ROWS,), accum)
                        for r in T.Parallel(ROWS):
                            alpha[r] = T.exp(m_ub[r] - T.max(m_ub[r], new_m[r]))
                            m_ub[r] = T.max(m_ub[r], new_m[r])

                        row_sum = T.alloc_reducer((ROWS,), accum, op="sum")
                        T.reducer_init(row_sum)
                        for r, c in T.Parallel(ROWS, BC):
                            s_frag[r, c] = T.exp(s_frag[r, c] - m_ub[r])
                            T.reducer_update(row_sum[r], s_frag[r, c])
                        partial = T.alloc_fragment((ROWS,), accum)
                        T.finalize_reducer(row_sum, partial)

                        for r in T.Parallel(ROWS):
                            l_ub[r] = l_ub[r] * alpha[r] + partial[r]
                        for r, d in T.Parallel(ROWS, D2):
                            o_ub0[r, d] = o_ub0[r, d] * alpha[r]
                        if D > D2:
                            for r, d in T.Parallel(ROWS, D2):
                                o_ub1[r, d] = o_ub1[r, d] * alpha[r]
                        for r, c in T.Parallel(ROWS, BC):
                            p_ub[r, c] = T.Cast(dtype, s_frag[r, c])

                # ---- o += P @ V ----
                T.dual_copy(p_ub[0:ROWS, 0:BC], p_l1[0:BR, 0:BC])
                T.copy(VCache[pid, 0:BC, bh, 0:D], v_l1[0:D, 0:BC], transpose=True)
                T.copy(p_l1[0:BR, 0:BC], pv_a)
                T.copy(v_l1[0:D2, 0:BC], pv_b)
                T.gemm(pv_a, pv_b, pv_acc0, transpose_B=True, clear_accum=True)
                if D > D2:
                    T.copy(v_l1[D2:D, 0:BC], pv_b)
                    T.gemm(pv_a, pv_b, pv_acc1, transpose_B=True, clear_accum=True)
                T.dual_copy(pv_acc0, o_tmp0)
                if D > D2:
                    T.dual_copy(pv_acc1, o_tmp1)
                with T.SimtVF(threads=threads):
                    for r, d in T.Parallel(ROWS, D2):
                        o_ub0[r, d] = o_ub0[r, d] + o_tmp0[r, d]
                    if D > D2:
                        for r, d in T.Parallel(ROWS, D2):
                            o_ub1[r, d] = o_ub1[r, d] + o_tmp1[r, d]

            # ---- normalise and store ----
            with T.SimtVF(threads=threads):
                for r, d in T.Parallel(ROWS, D2):
                    out_ub0[r, d] = T.Cast(dtype, o_ub0[r, d] / l_ub[r])
                if D > D2:
                    for r, d in T.Parallel(ROWS, D2):
                        out_ub1[r, d] = T.Cast(dtype, o_ub1[r, d] / l_ub[r])
            T.dual_copy(out_ub0[0:ROWS, 0:D2], Out[b, bh * BR : (bh + 1) * BR, 0:D2])
            if D > D2:
                T.dual_copy(out_ub1[0:ROWS, 0:D2], Out[b, bh * BR : (bh + 1) * BR, D2:D])

    return main


def paged_decode_split(
    batch: int,
    kv_heads: int,
    group: int,
    dim: int,
    page_size: int,
    num_pages_cap: int,
    num_tiles: int,
    max_splits: int,
    dtype: str = "bfloat16",
    threads: int = 128,
):
    """Split-KV decode: one block per ``(tile, kv_head)``, writing *partial* results.

    Same arithmetic as :func:`paged_decode_ascend950`, but the page range comes from the
    scheduler's tile arrays instead of "all pages of the request", and the epilogue writes the
    tile's **normalised output plus its log-sum-exp**.

    The partials are laid out **densely by slot**: request ``b``'s split ``s`` of KV head ``bh``
    lives at index ``((b * max_splits + s) * kv_heads + bh)``.  ``tile_slots[tile]`` says which
    slot a tile must write, which is what makes the merge kernel free of runtime conditionals: the
    slots a request does not use are *pre-filled* with ``lse = -1e30`` once per plan, so they
    contribute exactly zero weight (see :func:`paged_decode_merge`).

    Parallelism: the grid is ``num_tiles * kv_heads`` instead of ``batch * kv_heads``, so a single
    long request can occupy every core instead of eight of them (the 51 GB/s case in
    ``docs/performance.md`` measures 153 GB/s with 16 splits).

    Like the plain kernel, ``KCache``/``VCache`` are declared at the declared capacity, not the
    real pool (L2 pool-agnostic ABI, see :func:`paged_decode_ascend950`).
    """
    BR = padded_rows(group)
    BC = page_size
    D = dim
    ROWS = BR // 2
    accum = "float32"
    scale = 1.0 / math.sqrt(dim)
    num_slots = batch * max_splits

    assert BR % 2 == 0 and group % 2 == 0 and group <= BR
    assert dim in (128, 256), "the Ascend 950 attention path is tuned for head_dim 128 or 256"
    assert page_size % 2 == 0

    @T.prim_func
    def main(
        Q: T.Tensor((batch, kv_heads * BR, dim), dtype),
        KCache: T.Tensor((num_pages_cap, page_size, kv_heads, dim), dtype),
        VCache: T.Tensor((num_pages_cap, page_size, kv_heads, dim), dtype),
        kv_indptr: T.Tensor((batch + 1,), "int32"),
        kv_indices: T.Tensor((num_pages_cap,), "int32"),
        kv_last_page_len: T.Tensor((batch,), "int32"),
        seq_ids: T.Tensor((num_tiles,), "int32"),
        page_starts: T.Tensor((num_tiles,), "int32"),
        page_lens: T.Tensor((num_tiles,), "int32"),
        tile_slots: T.Tensor((num_tiles,), "int32"),
        token_starts: T.Tensor((num_tiles,), "int32"),
        token_lens: T.Tensor((num_tiles,), "int32"),
        PartOut: T.Tensor((num_slots * kv_heads, BR, dim), dtype),
        PartLse: T.Tensor((num_slots * kv_heads * BR,), accum),
    ):
        with T.Kernel(num_tiles * kv_heads) as bx:
            tile = bx // kv_heads
            bh = bx % kv_heads
            b = seq_ids[tile]
            start = page_starts[tile]
            plen = page_lens[tile]
            req_pages = kv_indptr[b + 1] - kv_indptr[b]
            # H67 token 级切分:``token_starts/token_lens`` 只对**首尾两个 split** 非平凡。
            # 页内偏移让同一页可以被多个 split 分摊——短上下文(1 页)因此也能切出
            # 多个 tile,把 grid 从 2 blocks 提到可用核数量级。
            tok_start = token_starts[tile]
            tok_len = token_lens[tile]
            # One *full* plan serves any page-count distribution (graph capture pins the plan
            # while replay-time sequences grow): clamp this tile's page loop to the pages that
            # actually exist.  An emptied tile (start >= req_pages) skips the loop entirely and
            # writes l=0 -> the l_safe below turns its slot into (out=0, lse=-1e30), exactly the
            # pre-filled values the merge weights at zero; a partially covered tile stops at
            # req_pages and the `limit` inside the loop still pins the request's real last page.
            eff = req_pages - start
            eff = T.if_then_else(eff < 0, 0, eff)
            plen = T.if_then_else(plen < eff, plen, eff)
            slot = tile_slots[tile]

            q_l1 = T.alloc_l1((BR, D), dtype)
            k_l1 = T.alloc_l1((BC, D), dtype)
            v_l1 = T.alloc_l1((D, BC), dtype)
            p_l1 = T.alloc_l1((BR, BC), dtype)

            # D-split, see paged_decode_ascend950: keeps L0A/L0B within capacity for dim=256.
            D2 = min(D, 128)
            qk_a = T.alloc_l0a((BR, D2), dtype)
            qk_b = T.alloc_l0b((BC, D2), dtype)
            pv_a = T.alloc_l0a((BR, BC), dtype)
            pv_b = T.alloc_l0b((D2, BC), dtype)
            qk_acc = T.alloc_l0c((BR, BC), accum)
            pv_acc0 = T.alloc_l0c((BR, D2), accum)
            if D > D2:
                pv_acc1 = T.alloc_l0c((BR, D2), accum)

            s_ub = T.alloc_shared((ROWS, BC), accum)
            p_ub = T.alloc_shared((ROWS, BC), dtype)
            o_ub0 = T.alloc_shared((ROWS, D2), accum)
            o_tmp0 = T.alloc_shared((ROWS, D2), accum)
            out_ub0 = T.alloc_shared((ROWS, D2), dtype)
            if D > D2:
                o_ub1 = T.alloc_shared((ROWS, D2), accum)
                o_tmp1 = T.alloc_shared((ROWS, D2), accum)
                out_ub1 = T.alloc_shared((ROWS, D2), dtype)
            lse_ub = T.alloc_shared((ROWS,), accum)
            m_ub = T.alloc_shared((ROWS,), accum)
            l_ub = T.alloc_shared((ROWS,), accum)
            if _simd_softmax_enabled():
                a_ub = T.alloc_shared((ROWS,), accum)
                ll_ub = T.alloc_shared((1,), "int32")

            T.copy(Q[b, bh * BR : (bh + 1) * BR, 0:D], q_l1)
            T.fill(m_ub, T.float32(-1e30))
            T.fill(l_ub, T.float32(0))
            T.fill(o_ub0, T.float32(0))
            if D > D2:
                T.fill(o_ub1, T.float32(0))
            if _simd_softmax_enabled():
                T.copy(kv_last_page_len[b : b + 1], ll_ub)

            for p in T.serial(plen):
                pid = kv_indices[kv_indptr[b] + start + p]

                T.copy(KCache[pid, 0:BC, bh, 0:D], k_l1)
                T.copy(q_l1[0:BR, 0:D2], qk_a)
                T.copy(k_l1[0:BC, 0:D2], qk_b)
                T.gemm(qk_a, qk_b, qk_acc, transpose_B=True, clear_accum=True)
                if D > D2:
                    T.copy(q_l1[0:BR, D2:D], qk_a)
                    T.copy(k_l1[0:BC, D2:D], qk_b)
                    T.gemm(qk_a, qk_b, qk_acc, transpose_B=True)
                T.dual_copy(qk_acc, s_ub)

                # H67:本 split 在**首页**可能从 tok_start 开始、在**尾页**可能提前结束。
                # 两者都由 token_starts/token_lens 决定,只对本 tile 覆盖的边界页生效:
                #   - 首页:p == 0 且 tok_start > 0 → 前面的 token 置 -1e30;
                #   - 尾页:p == plen-1 且 tok_start+tok_len < 该页有效长度 → 后面置 -1e30。
                # 页内 token 索引 c 的全局偏移是 start*BC + p*BC + c。
                # 对未做 token 级切分的 tile(tok_start=0 且 tok_len 覆盖整段),
                # 下面的下界恒为 0、上界恒为 BC,与旧行为逐位一致。
                # only the request's *last* page can be partially filled, wherever the split puts it
                if _simd_softmax_enabled():
                    limit = T.if_then_else(start + p + 1 < req_pages, BC, ll_ub[0])
                    # token 窗口换算到本页内的 [lo, hi) 区间
                    base = (start + p) * BC
                    lo = tok_start - base
                    lo = T.if_then_else(lo > 0, lo, 0)
                    hi = tok_start + tok_len - base
                    hi = T.if_then_else(hi < limit, hi, limit)
                    hi = T.if_then_else(hi < 0, 0, hi)
                    with T.SimdVF():
                        for r in T.serial(ROWS):
                            idx0 = T.simd.vci(0, "int32", order="INC_ORDER")
                            idx1 = T.simd.vci(64, "int32", order="INC_ORDER")
                            sc = T.simd.vdup(T.float32(scale), "float32")
                            a0 = T.simd.vmul(T.simd.vld(s_ub[r, 0], dist="NORM"), sc)
                            a1 = T.simd.vmul(T.simd.vld(s_ub[r, 64], dist="NORM"), sc)
                            neg = T.simd.vdup(T.float32(-1e30), "float32")
                            # H67 token 窗口(仅 split 内核):lo 屏蔽本 split 之前的 token;上界取
                            # min(limit, hi)。无 token 切分时 lo=0、hi=BC,与旧行为逐位一致。
                            a0 = T.simd.vsel(a0, neg, T.simd.vcmps(idx0, lo, op="ge"))
                            a1 = T.simd.vsel(a1, neg, T.simd.vcmps(idx1, lo, op="ge"))
                            a0 = T.simd.vsel(a0, neg, T.simd.vcmps(idx0, limit, op="lt"))
                            a1 = T.simd.vsel(a1, neg, T.simd.vcmps(idx1, limit, op="lt"))
                            a0 = T.simd.vsel(a0, neg, T.simd.vcmps(idx0, hi, op="lt"))
                            a1 = T.simd.vsel(a1, neg, T.simd.vcmps(idx1, hi, op="lt"))
                            pm = T.simd.vdupv(T.simd.vld(m_ub[r], dist="BRC_B32"), pos="POS_LOWEST")
                            nm = T.simd.vmax(pm, T.simd.vmax(T.simd.vcmax(a0), T.simd.vcmax(a1)))
                            nml = T.simd.vdupv(nm, pos="POS_LOWEST")
                            al = T.simd.vexp(T.simd.vsub(pm, nml))
                            e0 = T.simd.vexp(T.simd.vsub(a0, nml))
                            e1 = T.simd.vexp(T.simd.vsub(a1, nml))
                            lp = T.simd.vdupv(T.simd.vld(l_ub[r], dist="BRC_B32"), pos="POS_LOWEST")
                            ln = T.simd.vadd(
                                T.simd.vmul(lp, al),
                                T.simd.vadd(T.simd.vcadd(e0), T.simd.vcadd(e1)),
                            )
                            T.simd.vsts(m_ub[r], nml, dist="ONEPT_B32")
                            T.simd.vsts(l_ub[r], T.simd.vdupv(ln, pos="POS_LOWEST"), dist="ONEPT_B32")
                            T.simd.vsts(a_ub[r], T.simd.vdupv(al, pos="POS_LOWEST"), dist="ONEPT_B32")
                            T.simd.vsts(s_ub[r, 0], e0)
                            T.simd.vsts(s_ub[r, 64], e1)
                    with T.SimtVF(threads=threads):
                        for r, d in T.Parallel(ROWS, D2):
                            o_ub0[r, d] = o_ub0[r, d] * a_ub[r]
                        if D > D2:
                            for r, d in T.Parallel(ROWS, D2):
                                o_ub1[r, d] = o_ub1[r, d] * a_ub[r]
                        for r, c in T.Parallel(ROWS, BC):
                            p_ub[r, c] = T.Cast(dtype, s_ub[r, c])
                else:
                    limit = T.if_then_else(start + p + 1 < req_pages, BC, kv_last_page_len[b])
                    # H67 token 窗口(非 SIMD 分支同样必须生效,否则开 token_split 而不开
                    # SIMD softmax 会算出错误结果):与上面 SIMD 分支同一套 lo/hi 换算,
                    # 且 hi 已 clamp 到 limit,故 `c < hi` 覆盖原 `c < limit`。
                    base = (start + p) * BC
                    lo = tok_start - base
                    lo = T.if_then_else(lo > 0, lo, 0)
                    hi = tok_start + tok_len - base
                    hi = T.if_then_else(hi < limit, hi, limit)
                    hi = T.if_then_else(hi < 0, 0, hi)
                    with T.SimtVF(threads=threads):
                        s_frag = T.alloc_fragment((ROWS, BC), accum)
                        for r, c in T.Parallel(ROWS, BC):
                            s_frag[r, c] = T.if_then_else(
                                c < hi,
                                T.if_then_else(
                                    c >= lo,
                                    s_ub[r, c] * T.float32(scale),
                                    T.float32(-1e30),
                                ),
                                T.float32(-1e30),
                            )
                        row_max = T.alloc_reducer((ROWS,), accum, op="max")
                        T.reducer_init(row_max)
                        for r, c in T.Parallel(ROWS, BC):
                            T.reducer_update(row_max[r], s_frag[r, c])
                        new_m = T.alloc_fragment((ROWS,), accum)
                        T.finalize_reducer(row_max, new_m)

                        alpha = T.alloc_fragment((ROWS,), accum)
                        for r in T.Parallel(ROWS):
                            alpha[r] = T.exp(m_ub[r] - T.max(m_ub[r], new_m[r]))
                            m_ub[r] = T.max(m_ub[r], new_m[r])

                        row_sum = T.alloc_reducer((ROWS,), accum, op="sum")
                        T.reducer_init(row_sum)
                        for r, c in T.Parallel(ROWS, BC):
                            s_frag[r, c] = T.exp(s_frag[r, c] - m_ub[r])
                            T.reducer_update(row_sum[r], s_frag[r, c])
                        partial = T.alloc_fragment((ROWS,), accum)
                        T.finalize_reducer(row_sum, partial)

                        for r in T.Parallel(ROWS):
                            l_ub[r] = l_ub[r] * alpha[r] + partial[r]
                        for r, d in T.Parallel(ROWS, D2):
                            o_ub0[r, d] = o_ub0[r, d] * alpha[r]
                        if D > D2:
                            for r, d in T.Parallel(ROWS, D2):
                                o_ub1[r, d] = o_ub1[r, d] * alpha[r]
                        for r, c in T.Parallel(ROWS, BC):
                            p_ub[r, c] = T.Cast(dtype, s_frag[r, c])

                T.dual_copy(p_ub[0:ROWS, 0:BC], p_l1[0:BR, 0:BC])
                T.copy(VCache[pid, 0:BC, bh, 0:D], v_l1[0:D, 0:BC], transpose=True)
                T.copy(p_l1[0:BR, 0:BC], pv_a)
                T.copy(v_l1[0:D2, 0:BC], pv_b)
                T.gemm(pv_a, pv_b, pv_acc0, transpose_B=True, clear_accum=True)
                if D > D2:
                    T.copy(v_l1[D2:D, 0:BC], pv_b)
                    T.gemm(pv_a, pv_b, pv_acc1, transpose_B=True, clear_accum=True)
                T.dual_copy(pv_acc0, o_tmp0)
                if D > D2:
                    T.dual_copy(pv_acc1, o_tmp1)
                with T.SimtVF(threads=threads):
                    for r, d in T.Parallel(ROWS, D2):
                        o_ub0[r, d] = o_ub0[r, d] + o_tmp0[r, d]
                    if D > D2:
                        for r, d in T.Parallel(ROWS, D2):
                            o_ub1[r, d] = o_ub1[r, d] + o_tmp1[r, d]

            with T.SimtVF(threads=threads):
                # l_safe: an emptied tile carries l=0 — divide by 1 instead, so the slot gets
                # (out=0, lse=-1e30 + log(1)) = the merge's zero-weight filler, never a NaN.
                for r, d in T.Parallel(ROWS, D2):
                    l_safe = T.if_then_else(l_ub[r] > 0, l_ub[r], T.float32(1))
                    out_ub0[r, d] = T.Cast(dtype, o_ub0[r, d] / l_safe)
                if D > D2:
                    for r, d in T.Parallel(ROWS, D2):
                        l_safe = T.if_then_else(l_ub[r] > 0, l_ub[r], T.float32(1))
                        out_ub1[r, d] = T.Cast(dtype, o_ub1[r, d] / l_safe)
                for r in T.Parallel(ROWS):
                    l_safe = T.if_then_else(l_ub[r] > 0, l_ub[r], T.float32(1))
                    lse_ub[r] = m_ub[r] + T.log(l_safe)

            T.dual_copy(out_ub0[0:ROWS, 0:D2], PartOut[slot * kv_heads + bh, 0:BR, 0:D2])
            if D > D2:
                T.dual_copy(out_ub1[0:ROWS, 0:D2], PartOut[slot * kv_heads + bh, 0:BR, D2:D])
            T.dual_copy(
                lse_ub[0:ROWS],
                PartLse[(slot * kv_heads + bh) * BR : (slot * kv_heads + bh + 1) * BR],
            )

    return main


def paged_decode_merge(
    batch: int,
    kv_heads: int,
    group: int,
    dim: int,
    max_splits: int,
    dtype: str = "bfloat16",
    threads: int = 128,
):
    """Merge the split tiles of every request: a weighted average over the split axis.

    Contract (the one the torch reference implements and ``benchmarks/probes/split_kv_decode.py``
    validated with a host-side merge):

        lse_all[r] = logsumexp_s(lse[s, r])
        out[r, :]  = sum_s exp(lse[s, r] - lse_all[r]) * partial[s, r, :]

    The weights sum to 1 by construction (``exp(lse_all) = sum_s exp(lse_s)``), so there is no
    second division.

    No runtime conditionals anywhere: slots a request does not use are pre-filled with
    ``lse = -1e30``, so ``exp(-1e30 - lse_all)`` is 0 and the max/sum reductions ignore them.  The
    split axis is *unrolled in Python* (``max_splits`` is a compile-time constant), which keeps the
    per-split GM→UB staging static and the arithmetic race-free.

    The GM→UB staging copies whole **BR** lse rows, never bare ``group`` rows (KI-1 root cause):
    the copy engine faults with a vector-core exception on sub-32-byte descriptors, and with the
    serving GQA group of 4 a ``PartLse`` row is only 16 bytes (``group=8`` happened to be exactly
    one 32 B sector, which is why every early probe - all built with group 8 - passed).  ``BR`` is
    the slot stride of ``PartLse``, so the extra rows are real memory: finite values for slots the
    split kernel wrote (padding rows of a real slot carry ``-inf``, which the reductions below
    never read) and the pre-filled ``-inf`` of unused slots.  The ``PartOut`` copy keeps ``group``
    rows: each row is 256 B and rows scale with ``max_splits``, so padding it to ``BR`` would
    overflow UB at the long-context split counts (32 splits × 32 rows × dim).
    """
    BR = padded_rows(group)
    accum = "float32"
    num_slots = batch * max_splits

    assert group <= BR

    @T.prim_func
    def main(
        PartOut: T.Tensor((num_slots * kv_heads, BR, dim), dtype),
        PartLse: T.Tensor((num_slots * kv_heads * BR,), accum),
        Out: T.Tensor((batch, kv_heads * BR, dim), dtype),
    ):
        with T.Kernel(batch * kv_heads) as bx:
            b = bx // kv_heads
            bh = bx % kv_heads

            lse_g = T.alloc_shared((max_splits, BR), accum)
            out_g = T.alloc_shared((max_splits, group, dim), dtype)
            w_g = T.alloc_shared((max_splits, group), accum)
            acc_g = T.alloc_shared((group, dim), accum)
            # BR rows, not `group`: the UB->GM copy is row-granular (16/32), so storing only the
            # live rows would let the DMA write past the end of `Out` for the last KV-head block of
            # a request (observed as a deterministic 8 KB overrun).  Storing the whole padded tile
            # keeps the write exactly in-bounds, and matches what the unsplit kernel does.
            res_g = T.alloc_shared((BR, dim), dtype)

            for s in T.serial(max_splits):
                base = (b * max_splits + s) * kv_heads + bh
                T.copy(PartLse[base * BR : (base + 1) * BR], lse_g[s, 0:BR])
                T.copy(PartOut[base, 0:group, 0:dim], out_g[s, 0:group, 0:dim])

            with T.SimtVF(threads=threads):
                lse_all = T.alloc_fragment((group,), accum)
                red_max = T.alloc_reducer((group,), accum, op="max")
                T.reducer_init(red_max)
                for r, s in T.Parallel(group, max_splits):
                    T.reducer_update(red_max[r], lse_g[s, r])
                T.finalize_reducer(red_max, lse_all)

                red_sum = T.alloc_reducer((group,), accum, op="sum")
                T.reducer_init(red_sum)
                for r, s in T.Parallel(group, max_splits):
                    T.reducer_update(red_sum[r], T.exp(lse_g[s, r] - lse_all[r]))
                norm = T.alloc_fragment((group,), accum)
                T.finalize_reducer(red_sum, norm)

                for s in range(max_splits):  # python unroll: static, race-free
                    for r in T.Parallel(group):
                        w_g[s, r] = T.exp(lse_g[s, r] - lse_all[r]) / norm[r]
                for r, d in T.Parallel(group, dim):
                    acc_g[r, d] = T.float32(0)
                for s in range(max_splits):  # python unroll
                    for r, d in T.Parallel(group, dim):
                        acc_g[r, d] = acc_g[r, d] + w_g[s, r] * T.Cast(accum, out_g[s, r, d])
                for r, d in T.Parallel(group, dim):
                    res_g[r, d] = T.Cast(dtype, acc_g[r, d])
                for r, d in T.Parallel(BR - group, dim):
                    res_g[group + r, d] = T.Cast(dtype, T.float32(0))

            T.copy(res_g[0:BR, 0:dim], Out[b, bh * BR : (bh + 1) * BR, 0:dim])

    return main


def _cache_key(tag: str, spec: dict) -> tuple:
    """Kernel cache key.

    SIMD-softmax is read at *trace* time inside several kernel bodies (E116-era lesson:
    anything that changes the emitted program must land in the cache key), so the opt-in
    env flag is folded in here instead of leaving two env settings to silently share one
    compiled artifact.
    """
    return (tag,) + tuple(sorted({**spec, "simd_softmax": _simd_softmax_enabled()}.items()))


def build_merge_kernel(**spec):
    """Compile (and cache) the split-merge kernel."""
    key = _cache_key("merge", spec)
    kernel = _KERNEL_CACHE.get(key)
    if kernel is None:
        kernel = tilelang.compile(
            paged_decode_merge(**spec),
            out_idx=[],
            pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        )
        _KERNEL_CACHE[key] = kernel
    return kernel


_KERNEL_CACHE: dict = {}

def build_split_kernel(**spec):
    """Compile (and cache) a split-KV decode kernel.

    The pool capacity is quantized to the ABI's declared value (see :func:`declared_pool_cap`),
    so the cache key does not depend on the real pool size.
    """
    if spec.get("num_pages_cap") is not None:
        spec["num_pages_cap"] = declared_pool_cap(spec["num_pages_cap"])
    key = _cache_key("split", spec)
    kernel = _KERNEL_CACHE.get(key)
    if kernel is None:
        kernel = tilelang.compile(
            paged_decode_split(**spec),
            out_idx=[],
            pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        )
        _KERNEL_CACHE[key] = kernel
    return kernel



def build_decode_kernel(**spec):
    """Compile (and cache) the kernel for one shape combination.

    ``num_pages_cap`` is quantized to the ABI's *declared* capacity (see
    :func:`declared_pool_cap`) before it enters the cache key: the kernel declares the K/V cache
    at that extent, and the caller presents the real pool through :func:`pool_view` (a zero-copy
    capacity view re-binding the pool's own storage).  Pages beyond the real pool are never
    dereferenced — every access is bounded by ``kv_indptr``/``kv_indices`` — so one artifact
    serves every pool in the declared-capacity bucket, which decouples the kernel key from
    vLLM's snapshot-derived pool size.

    ``out_manual=True`` compiles the same kernel with ``out_idx=[]`` so the caller hands in the
    output tensor (``Out`` becomes the 7th positional argument).  That is what serving wants: the
    output buffer is owned by the plan and reused every step, so a captured step never allocates.
    """
    out_manual = bool(spec.pop("out_manual", False))
    pipeline = bool(spec.pop("pipeline", False))
    if pipeline:
        spec["pipeline"] = True
    # H74: `sw_pipeline=0` 是默认(逐位等同旧版),不进 cache key——避免默认路径
    # 因为多一个键而重复编译;非 0 值必须进 key,否则与串行版共享同一份产物。
    sw_pipeline = int(spec.pop("sw_pipeline", 0) or 0)
    if sw_pipeline > 0:
        spec["sw_pipeline"] = sw_pipeline
    if spec.get("num_pages_cap") is not None:
        spec["num_pages_cap"] = declared_pool_cap(spec["num_pages_cap"])
    key = _cache_key("decode", spec) + ("manual" if out_manual else "auto",)
    kernel = _KERNEL_CACHE.get(key)
    if kernel is None:
        kernel = tilelang.compile(
            paged_decode_ascend950(**spec),
            out_idx=[] if out_manual else -1,
            pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        )
        _KERNEL_CACHE[key] = kernel
    return kernel


def paged_prefill_ascend950(
    batch: int,
    kv_heads: int,
    group: int,
    dim: int,
    page_size: int,
    num_pages_cap: int,
    num_tiles: int,
    block_q: int = 128,
    dtype: str = "bfloat16",
    threads: int = 128,
    pv_single_pass: bool = False,
):
    """Paged **prefill / append** attention: causal, ragged, GQA, over a paged KV cache.

    Same building blocks as :func:`paged_decode_ascend950`, with three differences:

    * the M tile is a block of *query rows* (``block_q`` rows, padded to a multiple of 32), not the
      GQA group, so the grid is ``num_tiles * kv_heads`` and tiles are disjoint along the query
      axis — which means **no merge step**;
    * rows are in *flattened* ``(q_pos, group)`` order, i.e. row ``r`` of a tile belongs to query
      position ``(row_offset + r) // group`` and head ``(row_offset + r) % group``.  The caller packs
      ``Q`` into that layout (one permutation + reshape, see :func:`forward_prefill`), which keeps the
      kernel's DMA trivially contiguous;
    * causal (and append) masking: a query at token position ``p_q`` may attend to KV positions
      ``<= p_q + causal_shift``, where ``causal_shift = kv_len - qo_len``.  That is the offset that
      makes chunked prefill and speculative decoding correct: row 0 of a chunk sees the cached
      prefix, not position 0.

    **Causal mask and the two AIVs.**  The mask is row-dependent, and the row index inside the vector
    region is *per-AIV*: `dual_copy` splits the M tile over the two AIVs and each one sees
    `ROWS = block_q // 2` rows starting at 0.  The global row a local ``r`` belongs to is therefore
    ``sid * ROWS + r``, and ``sid`` is not exposed to a ``T.SimtVF`` region.  Instead of deriving the
    mask from the row index, the *caller* precomputes each row's causal threshold ``q_pos + shift``
    into ``RowThr`` (rows past the live count get -1), and the kernel ``dual_copy``s that 1-D array
    exactly like ``s_ub`` -- so each AIV half receives its own half of the thresholds and the mask is
    correct on both.  Callers may use the full ``block_q`` tile; there is no tile-size restriction.

    Like the decode kernels, ``KCache``/``VCache`` are declared at the declared capacity, not
    the real pool (L2 pool-agnostic ABI, see :func:`paged_decode_ascend950`).
    """
    BR = block_q
    BC = page_size
    D = dim
    ROWS = BR // 2
    accum = "float32"
    scale = 1.0 / math.sqrt(dim)

    assert BR % 32 == 0, "the ND->NZ copy template wants ROWS = BR // 2 to be a multiple of 16"
    assert BR <= 128, "one L0C tile holds BR x BC floats; keep block_q <= 128 at page_size 128"
    assert dim in (128, 256), "the Ascend 950 attention path is tuned for head_dim 128 or 256"
    assert page_size % 2 == 0

    # H83-① PV single-pass (a compile-time constant: the branches below are resolved at build
    # time, zero runtime cost).  ``D2`` stays the *QK* split (its single-pass is dead on L0
    # capacity: qk_a + qk_b both exactly full); ``DPV`` is the PV split — ``D`` when the caller
    # opts in on dim=256, which halves the per-page PV-side command count (gemm 2→1, pv_b copy
    # 2→1, dual_copy 2→1, rescale/accumulate loop 2→1) at the cost of an exactly-full L0B
    # (DPV·BC bf16 = 64KB) and L0C (BR·DPV fp32 = 128KB).  If the compiler rejects the
    # exactly-full tiles, this axis is disproven and the rewrite falls back to the NZ K/V
    # layout axis only.
    D2 = min(D, 128)
    DPV = D if (pv_single_pass and D > D2) else D2

    @T.prim_func
    def main(
        QPacked: T.Tensor((num_tiles * kv_heads, BR, dim), dtype),
        KCache: T.Tensor((num_pages_cap, page_size, kv_heads, dim), dtype),
        VCache: T.Tensor((num_pages_cap, page_size, kv_heads, dim), dtype),
        kv_indptr: T.Tensor((batch + 1,), "int32"),
        kv_indices: T.Tensor((num_pages_cap,), "int32"),
        kv_last_page_len: T.Tensor((batch,), "int32"),
        seq_ids: T.Tensor((num_tiles,), "int32"),
        q_offsets: T.Tensor((num_tiles,), "int32"),
        tile_rows: T.Tensor((num_tiles,), "int32"),
        causal_shift: T.Tensor((batch,), "int32"),
        RowThr: T.Tensor((num_tiles * BR,), "int32"),
        TilePageLens: T.Tensor((num_tiles,), "int32"),
        OutPacked: T.Tensor((num_tiles * kv_heads, BR, dim), dtype),
    ):
        with T.Kernel(num_tiles * kv_heads) as bx:
            tile = bx // kv_heads
            bh = bx % kv_heads
            b = seq_ids[tile]
            q_off = q_offsets[tile]
            rows = tile_rows[tile]
            shift = causal_shift[b]
            req_pages = kv_indptr[b + 1] - kv_indptr[b]
            # KV rescan truncation: the causal mask (RowThr) means this tile can only attend to KV
            # positions ``<= max(thresholds)``, so scanning the request's *whole* page list would
            # re-read (and re-GEMM) pages every row masks out.  ``TilePageLens`` is the host-side
            # count of pages this tile actually needs; clamp it to the real page count so an empty
            # or padded tile runs zero iterations.  The per-page ``limit`` below is unchanged: a
            # truncated tile never visits the request's last page, so `p + 1 < req_pages` holds and
            # its full-page columns stay live (the within-page frontier is still masked by RowThr).
            plen = TilePageLens[tile]
            plen = T.if_then_else(plen < req_pages, plen, req_pages)
            plen = T.if_then_else(plen < 0, 0, plen)

            q_l1 = T.alloc_l1((BR, D), dtype)
            k_l1 = T.alloc_l1((BC, D), dtype)
            v_l1 = T.alloc_l1((D, BC), dtype)
            p_l1 = T.alloc_l1((BR, BC), dtype)

            # D-split, see paged_decode_ascend950: keeps L0A/L0B within capacity for dim=256.
            # QK keeps the D2 split; PV uses DPV (== D under pv_single_pass, H83-①).
            qk_a = T.alloc_l0a((BR, D2), dtype)
            qk_b = T.alloc_l0b((BC, D2), dtype)
            pv_a = T.alloc_l0a((BR, BC), dtype)
            pv_b = T.alloc_l0b((DPV, BC), dtype)
            qk_acc = T.alloc_l0c((BR, BC), accum)
            pv_acc0 = T.alloc_l0c((BR, DPV), accum)
            if D > DPV:
                pv_acc1 = T.alloc_l0c((BR, D2), accum)

            s_ub = T.alloc_shared((ROWS, BC), accum)
            p_ub = T.alloc_shared((ROWS, BC), dtype)
            o_ub0 = T.alloc_shared((ROWS, DPV), accum)
            o_tmp0 = T.alloc_shared((ROWS, DPV), accum)
            out_ub0 = T.alloc_shared((ROWS, DPV), dtype)
            if D > DPV:
                o_ub1 = T.alloc_shared((ROWS, D2), accum)
                o_tmp1 = T.alloc_shared((ROWS, D2), accum)
                out_ub1 = T.alloc_shared((ROWS, D2), dtype)
            m_ub = T.alloc_shared((ROWS,), accum)
            l_ub = T.alloc_shared((ROWS,), accum)
            # Per-row causal threshold for *this AIV's* half of the M tile.  `dual_copy` splits it
            # exactly like `s_ub`, so a local row r holds the threshold of global row
            # `sid * ROWS + r` without the kernel ever needing the AIV index.
            thr_ub = T.alloc_shared((ROWS,), "int32")
            if _simd_softmax_enabled():
                a_ub = T.alloc_shared((ROWS,), accum)
                ll_ub = T.alloc_shared((1,), "int32")

            T.copy(QPacked[bx, 0:BR, 0:D], q_l1)
            T.dual_copy(RowThr[tile * BR : (tile + 1) * BR], thr_ub)
            T.fill(m_ub, T.float32(-1e30))
            T.fill(l_ub, T.float32(0))
            T.fill(o_ub0, T.float32(0))
            if D > DPV:
                T.fill(o_ub1, T.float32(0))
            if _simd_softmax_enabled():
                T.copy(kv_last_page_len[b : b + 1], ll_ub)

            for p in T.serial(plen):
                pid = kv_indices[kv_indptr[b] + p]

                T.copy(KCache[pid, 0:BC, bh, 0:D], k_l1)
                T.copy(q_l1[0:BR, 0:D2], qk_a)
                T.copy(k_l1[0:BC, 0:D2], qk_b)
                T.gemm(qk_a, qk_b, qk_acc, transpose_B=True, clear_accum=True)
                if D > DPV:
                    T.copy(q_l1[0:BR, D2:D], qk_a)
                    T.copy(k_l1[0:BC, D2:D], qk_b)
                    T.gemm(qk_a, qk_b, qk_acc, transpose_B=True)
                T.dual_copy(qk_acc, s_ub)

                if _simd_softmax_enabled():
                    # T5 packed-SIMD online softmax with causal + tail mask, applied as two chained
                    # `vsel`s (a `vand` of two `boolx256` masks fails to infer its default mask
                    # width).  `kv_pos` uses `vadds(vci(0), p*BC)` because `vci` needs a
                    # compile-time start.  fp32 weights -> `s_ub`; cast/rescale stay below.
                    limit = T.if_then_else(p + 1 < req_pages, BC, ll_ub[0])
                    with T.SimdVF():
                        for r in T.serial(ROWS):
                            idx0 = T.simd.vci(0, "int32", order="INC_ORDER")
                            idx1 = T.simd.vci(64, "int32", order="INC_ORDER")
                            kp0 = T.simd.vadds(T.simd.vci(0, "int32", order="INC_ORDER"), p * BC)
                            kp1 = T.simd.vadds(T.simd.vci(64, "int32", order="INC_ORDER"), p * BC)
                            tb = T.simd.vdupv(T.simd.vld(thr_ub[r], dist="BRC_B32"), pos="POS_LOWEST")
                            sc = T.simd.vdup(T.float32(scale), "float32")
                            a0 = T.simd.vmul(T.simd.vld(s_ub[r, 0], dist="NORM"), sc)
                            a1 = T.simd.vmul(T.simd.vld(s_ub[r, 64], dist="NORM"), sc)
                            neg = T.simd.vdup(T.float32(-1e30), "float32")
                            a0 = T.simd.vsel(a0, neg, T.simd.vcmps(idx0, limit, op="lt"))
                            a1 = T.simd.vsel(a1, neg, T.simd.vcmps(idx1, limit, op="lt"))
                            a0 = T.simd.vsel(a0, neg, T.simd.vcmp(kp0, tb, op="le"))
                            a1 = T.simd.vsel(a1, neg, T.simd.vcmp(kp1, tb, op="le"))
                            pm = T.simd.vdupv(T.simd.vld(m_ub[r], dist="BRC_B32"), pos="POS_LOWEST")
                            nm = T.simd.vmax(pm, T.simd.vmax(T.simd.vcmax(a0), T.simd.vcmax(a1)))
                            nml = T.simd.vdupv(nm, pos="POS_LOWEST")
                            al = T.simd.vexp(T.simd.vsub(pm, nml))
                            e0 = T.simd.vexp(T.simd.vsub(a0, nml))
                            e1 = T.simd.vexp(T.simd.vsub(a1, nml))
                            lp = T.simd.vdupv(T.simd.vld(l_ub[r], dist="BRC_B32"), pos="POS_LOWEST")
                            ln = T.simd.vadd(
                                T.simd.vmul(lp, al),
                                T.simd.vadd(T.simd.vcadd(e0), T.simd.vcadd(e1)),
                            )
                            T.simd.vsts(m_ub[r], nml, dist="ONEPT_B32")
                            T.simd.vsts(l_ub[r], T.simd.vdupv(ln, pos="POS_LOWEST"), dist="ONEPT_B32")
                            T.simd.vsts(a_ub[r], T.simd.vdupv(al, pos="POS_LOWEST"), dist="ONEPT_B32")
                            T.simd.vsts(s_ub[r, 0], e0)
                            T.simd.vsts(s_ub[r, 64], e1)
                    with T.SimtVF(threads=threads):
                        for r, d in T.Parallel(ROWS, DPV):
                            o_ub0[r, d] = o_ub0[r, d] * a_ub[r]
                        if D > DPV:
                            for r, d in T.Parallel(ROWS, D2):
                                o_ub1[r, d] = o_ub1[r, d] * a_ub[r]
                        for r, c in T.Parallel(ROWS, BC):
                            p_ub[r, c] = T.Cast(dtype, s_ub[r, c])
                else:
                    limit = T.if_then_else(p + 1 < req_pages, BC, kv_last_page_len[b])
                    with T.SimtVF(threads=threads):
                        s_frag = T.alloc_fragment((ROWS, BC), accum)
                        for r, c in T.Parallel(ROWS, BC):
                            # causal + tail masks, both before exp.  The row's global query position
                            # arrives as a precomputed threshold (`thr_ub`, split like `s_ub`), so the
                            # mask is correct for either AIV half; padded rows carry thr < 0 and are
                            # therefore fully masked.
                            kv_pos = p * BC + c
                            s_frag[r, c] = T.if_then_else(
                                (c < limit) and (kv_pos <= thr_ub[r]),
                                s_ub[r, c] * T.float32(scale),
                                T.float32(-1e30),
                            )
                        row_max = T.alloc_reducer((ROWS,), accum, op="max")
                        T.reducer_init(row_max)
                        for r, c in T.Parallel(ROWS, BC):
                            T.reducer_update(row_max[r], s_frag[r, c])
                        new_m = T.alloc_fragment((ROWS,), accum)
                        T.finalize_reducer(row_max, new_m)

                        alpha = T.alloc_fragment((ROWS,), accum)
                        for r in T.Parallel(ROWS):
                            alpha[r] = T.exp(m_ub[r] - T.max(m_ub[r], new_m[r]))
                            m_ub[r] = T.max(m_ub[r], new_m[r])

                        row_sum = T.alloc_reducer((ROWS,), accum, op="sum")
                        T.reducer_init(row_sum)
                        for r, c in T.Parallel(ROWS, BC):
                            s_frag[r, c] = T.exp(s_frag[r, c] - m_ub[r])
                            T.reducer_update(row_sum[r], s_frag[r, c])
                        partial = T.alloc_fragment((ROWS,), accum)
                        T.finalize_reducer(row_sum, partial)

                        for r in T.Parallel(ROWS):
                            l_ub[r] = l_ub[r] * alpha[r] + partial[r]
                        for r, d in T.Parallel(ROWS, DPV):
                            o_ub0[r, d] = o_ub0[r, d] * alpha[r]
                        if D > DPV:
                            for r, d in T.Parallel(ROWS, D2):
                                o_ub1[r, d] = o_ub1[r, d] * alpha[r]
                        for r, c in T.Parallel(ROWS, BC):
                            p_ub[r, c] = T.Cast(dtype, s_frag[r, c])

                T.dual_copy(p_ub[0:ROWS, 0:BC], p_l1[0:BR, 0:BC])
                T.copy(VCache[pid, 0:BC, bh, 0:D], v_l1[0:D, 0:BC], transpose=True)
                T.copy(p_l1[0:BR, 0:BC], pv_a)
                T.copy(v_l1[0:DPV, 0:BC], pv_b)
                T.gemm(pv_a, pv_b, pv_acc0, transpose_B=True, clear_accum=True)
                if D > DPV:
                    T.copy(v_l1[DPV:D, 0:BC], pv_b)
                    T.gemm(pv_a, pv_b, pv_acc1, transpose_B=True, clear_accum=True)
                T.dual_copy(pv_acc0, o_tmp0)
                if D > DPV:
                    T.dual_copy(pv_acc1, o_tmp1)
                with T.SimtVF(threads=threads):
                    for r, d in T.Parallel(ROWS, DPV):
                        o_ub0[r, d] = o_ub0[r, d] + o_tmp0[r, d]
                    if D > DPV:
                        for r, d in T.Parallel(ROWS, D2):
                            o_ub1[r, d] = o_ub1[r, d] + o_tmp1[r, d]

            with T.SimtVF(threads=threads):
                for r, d in T.Parallel(ROWS, DPV):
                    out_ub0[r, d] = T.Cast(dtype, o_ub0[r, d] / l_ub[r])
                if D > DPV:
                    for r, d in T.Parallel(ROWS, D2):
                        out_ub1[r, d] = T.Cast(dtype, o_ub1[r, d] / l_ub[r])

            T.dual_copy(out_ub0[0:ROWS, 0:DPV], OutPacked[bx, 0:BR, 0:DPV])
            if D > DPV:
                T.dual_copy(out_ub1[0:ROWS, 0:D2], OutPacked[bx, 0:BR, DPV:D])

    return main


def build_prefill_kernel(**spec):
    """Compile (and cache) the prefill kernel.  See `build_decode_kernel` re: `num_pages_cap`
    (quantized to the ABI's declared capacity, so the key is pool-agnostic)."""
    if spec.get("num_pages_cap") is not None:
        spec["num_pages_cap"] = declared_pool_cap(spec["num_pages_cap"])
    key = _cache_key("prefill", spec)
    kernel = _KERNEL_CACHE.get(key)
    if kernel is None:
        kernel = tilelang.compile(
            paged_prefill_ascend950(**spec),
            out_idx=[],
            pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        )
        _KERNEL_CACHE[key] = kernel
    return kernel


def paged_prefill_split_ascend950(
    batch: int,
    kv_heads: int,
    group: int,
    dim: int,
    page_size: int,
    num_pages_cap: int,
    num_tiles: int,
    max_splits: int,
    block_q: int = 128,
    dtype: str = "bfloat16",
    threads: int = 128,
):
    """Split-KV **prefill / append**: the prefill kernel writing *partial* results (H19).

    Identical arithmetic to :func:`paged_prefill_ascend950` — same M tile (``block_q`` rows in
    flattened ``(q_pos, head)`` order), same per-row causal thresholds (``RowThr``), same D-split
    — with two differences:

    * each tile visits only the page range the scheduler gave it (``kv_page_starts[tile]``,
      ``TilePageLens[tile]`` pages, further clamped to the pages that actually exist), so one
      (request, query-block) split ``ceil(pages / kv_tile_pages)`` ways runs its ranges
      **concurrently** instead of one tile paging through the whole prefix serially — the W1-C8
      pathology (E46: per-tile serial page walks, vector chain 96% busy, blocks <10% occupied);
    * the epilogue writes the tile's *normalised* output plus its log-sum-exp into the dense
      slot layout (``(seq_id * max_splits + split_id) * kv_heads + bh``) exactly like the decode
      split kernel, so a merge step combines the partials with softmax weights.  A fully-masked
      or emptied tile writes ``(out=0, lse=-1e30)`` (the ``l_safe`` guard), which the merge
      weights at zero.

    ``RowThr`` thresholds are *global* KV positions, so no per-split mask adjustment is needed:
    a split whose whole range lies past a row's causal frontier simply produces the zero-weight
    ``(0, -1e30)`` partial.  The per-page ``limit`` keeps the request's real last page (wherever
    the split puts it) from reading past ``kv_last_page_len``.

    Like every kernel here, ``KCache``/``VCache`` are declared at the declared capacity, not the
    real pool (L2 pool-agnostic ABI).
    """
    BR = block_q
    BC = page_size
    D = dim
    ROWS = BR // 2
    accum = "float32"
    scale = 1.0 / math.sqrt(dim)
    num_slots = batch * max_splits

    assert BR % 32 == 0, "the ND->NZ copy template wants ROWS = BR // 2 to be a multiple of 16"
    assert BR <= 128, "one L0C tile holds BR x BC floats; keep block_q <= 128 at page_size 128"
    assert dim in (128, 256), "the Ascend 950 attention path is tuned for head_dim 128 or 256"
    assert page_size % 2 == 0

    @T.prim_func
    def main(
        QPacked: T.Tensor((num_tiles * kv_heads, BR, dim), dtype),
        KCache: T.Tensor((num_pages_cap, page_size, kv_heads, dim), dtype),
        VCache: T.Tensor((num_pages_cap, page_size, kv_heads, dim), dtype),
        kv_indptr: T.Tensor((batch + 1,), "int32"),
        kv_indices: T.Tensor((num_pages_cap,), "int32"),
        kv_last_page_len: T.Tensor((batch,), "int32"),
        seq_ids: T.Tensor((num_tiles,), "int32"),
        q_offsets: T.Tensor((num_tiles,), "int32"),
        tile_rows: T.Tensor((num_tiles,), "int32"),
        kv_page_starts: T.Tensor((num_tiles,), "int32"),
        causal_shift: T.Tensor((batch,), "int32"),
        RowThr: T.Tensor((num_tiles * BR,), "int32"),
        TilePageLens: T.Tensor((num_tiles,), "int32"),
        tile_slots: T.Tensor((num_tiles,), "int32"),
        PartOut: T.Tensor((num_slots * kv_heads, BR, dim), dtype),
        PartLse: T.Tensor((num_slots * kv_heads * BR,), accum),
    ):
        with T.Kernel(num_tiles * kv_heads) as bx:
            tile = bx // kv_heads
            bh = bx % kv_heads
            b = seq_ids[tile]
            q_off = q_offsets[tile]
            rows = tile_rows[tile]
            shift = causal_shift[b]
            start = kv_page_starts[tile]
            req_pages = kv_indptr[b + 1] - kv_indptr[b]
            # Same truncation contract as the plain prefill kernel (TilePageLens = the tile's
            # causal frontier) plus the split-KV clamp: a tile may only visit its own page range
            # ``[start, start + plen)`` and never past the pages that exist.  An emptied tile
            # (range past the frontier, or a padded tile) runs zero iterations and the epilogue's
            # ``l_safe`` turns its slot into (out=0, lse=-1e30) — the merge's zero weight.
            plen = TilePageLens[tile]
            eff = req_pages - start
            eff = T.if_then_else(eff < 0, 0, eff)
            plen = T.if_then_else(plen < eff, plen, eff)
            plen = T.if_then_else(plen < 0, 0, plen)
            slot = tile_slots[tile]

            q_l1 = T.alloc_l1((BR, D), dtype)
            k_l1 = T.alloc_l1((BC, D), dtype)
            v_l1 = T.alloc_l1((D, BC), dtype)
            p_l1 = T.alloc_l1((BR, BC), dtype)

            # D-split, see paged_decode_ascend950: keeps L0A/L0B within capacity for dim=256.
            D2 = min(D, 128)
            qk_a = T.alloc_l0a((BR, D2), dtype)
            qk_b = T.alloc_l0b((BC, D2), dtype)
            pv_a = T.alloc_l0a((BR, BC), dtype)
            pv_b = T.alloc_l0b((D2, BC), dtype)
            qk_acc = T.alloc_l0c((BR, BC), accum)
            pv_acc0 = T.alloc_l0c((BR, D2), accum)
            if D > D2:
                pv_acc1 = T.alloc_l0c((BR, D2), accum)

            thr_ub = T.alloc_shared((ROWS,), "int32")
            s_ub = T.alloc_shared((ROWS, BC), accum)
            p_ub = T.alloc_shared((ROWS, BC), dtype)
            o_ub0 = T.alloc_shared((ROWS, D2), accum)
            o_tmp0 = T.alloc_shared((ROWS, D2), accum)
            out_ub0 = T.alloc_shared((ROWS, D2), dtype)
            if D > D2:
                o_ub1 = T.alloc_shared((ROWS, D2), accum)
                o_tmp1 = T.alloc_shared((ROWS, D2), accum)
                out_ub1 = T.alloc_shared((ROWS, D2), dtype)
            m_ub = T.alloc_shared((ROWS,), accum)
            l_ub = T.alloc_shared((ROWS,), accum)
            if _simd_softmax_enabled():
                a_ub = T.alloc_shared((ROWS,), accum)
                ll_ub = T.alloc_shared((1,), "int32")

            T.copy(QPacked[bx, 0:BR, 0:D], q_l1)
            T.dual_copy(RowThr[tile * BR : (tile + 1) * BR], thr_ub)
            T.fill(m_ub, T.float32(-1e30))
            T.fill(l_ub, T.float32(0))
            T.fill(o_ub0, T.float32(0))
            if D > D2:
                T.fill(o_ub1, T.float32(0))
            if _simd_softmax_enabled():
                T.copy(kv_last_page_len[b : b + 1], ll_ub)
            for p in T.serial(plen):
                pid = kv_indices[kv_indptr[b] + start + p]

                T.copy(KCache[pid, 0:BC, bh, 0:D], k_l1)
                T.copy(q_l1[0:BR, 0:D2], qk_a)
                T.copy(k_l1[0:BC, 0:D2], qk_b)
                T.gemm(qk_a, qk_b, qk_acc, transpose_B=True, clear_accum=True)
                if D > D2:
                    T.copy(q_l1[0:BR, D2:D], qk_a)
                    T.copy(k_l1[0:BC, D2:D], qk_b)
                    T.gemm(qk_a, qk_b, qk_acc, transpose_B=True)
                T.dual_copy(qk_acc, s_ub)

                # only the request's *last* page can be partially filled, wherever the split puts it
                if _simd_softmax_enabled():
                    # T5 packed-SIMD online softmax (causal + tail mask, two chained `vsel`s).
                    # `kv_pos` is the *global* column `(start+p)*BC + lane`; `vci` needs a
                    # compile-time start, so the page offset is added with `vadds`.
                    limit = T.if_then_else(start + p + 1 < req_pages, BC, ll_ub[0])
                    with T.SimdVF():
                        for r in T.serial(ROWS):
                            idx0 = T.simd.vci(0, "int32", order="INC_ORDER")
                            idx1 = T.simd.vci(64, "int32", order="INC_ORDER")
                            kp0 = T.simd.vadds(T.simd.vci(0, "int32", order="INC_ORDER"), (start + p) * BC)
                            kp1 = T.simd.vadds(T.simd.vci(64, "int32", order="INC_ORDER"), (start + p) * BC)
                            tb = T.simd.vdupv(T.simd.vld(thr_ub[r], dist="BRC_B32"), pos="POS_LOWEST")
                            sc = T.simd.vdup(T.float32(scale), "float32")
                            a0 = T.simd.vmul(T.simd.vld(s_ub[r, 0], dist="NORM"), sc)
                            a1 = T.simd.vmul(T.simd.vld(s_ub[r, 64], dist="NORM"), sc)
                            neg = T.simd.vdup(T.float32(-1e30), "float32")
                            a0 = T.simd.vsel(a0, neg, T.simd.vcmps(idx0, limit, op="lt"))
                            a1 = T.simd.vsel(a1, neg, T.simd.vcmps(idx1, limit, op="lt"))
                            a0 = T.simd.vsel(a0, neg, T.simd.vcmp(kp0, tb, op="le"))
                            a1 = T.simd.vsel(a1, neg, T.simd.vcmp(kp1, tb, op="le"))
                            pm = T.simd.vdupv(T.simd.vld(m_ub[r], dist="BRC_B32"), pos="POS_LOWEST")
                            nm = T.simd.vmax(pm, T.simd.vmax(T.simd.vcmax(a0), T.simd.vcmax(a1)))
                            nml = T.simd.vdupv(nm, pos="POS_LOWEST")
                            al = T.simd.vexp(T.simd.vsub(pm, nml))
                            e0 = T.simd.vexp(T.simd.vsub(a0, nml))
                            e1 = T.simd.vexp(T.simd.vsub(a1, nml))
                            lp = T.simd.vdupv(T.simd.vld(l_ub[r], dist="BRC_B32"), pos="POS_LOWEST")
                            ln = T.simd.vadd(
                                T.simd.vmul(lp, al),
                                T.simd.vadd(T.simd.vcadd(e0), T.simd.vcadd(e1)),
                            )
                            T.simd.vsts(m_ub[r], nml, dist="ONEPT_B32")
                            T.simd.vsts(l_ub[r], T.simd.vdupv(ln, pos="POS_LOWEST"), dist="ONEPT_B32")
                            T.simd.vsts(a_ub[r], T.simd.vdupv(al, pos="POS_LOWEST"), dist="ONEPT_B32")
                            T.simd.vsts(s_ub[r, 0], e0)
                            T.simd.vsts(s_ub[r, 64], e1)
                    with T.SimtVF(threads=threads):
                        for r, d in T.Parallel(ROWS, D2):
                            o_ub0[r, d] = o_ub0[r, d] * a_ub[r]
                        if D > D2:
                            for r, d in T.Parallel(ROWS, D2):
                                o_ub1[r, d] = o_ub1[r, d] * a_ub[r]
                        for r, c in T.Parallel(ROWS, BC):
                            p_ub[r, c] = T.Cast(dtype, s_ub[r, c])
                else:
                    limit = T.if_then_else(start + p + 1 < req_pages, BC, kv_last_page_len[b])
                    with T.SimtVF(threads=threads):
                        s_frag = T.alloc_fragment((ROWS, BC), accum)
                        for r, c in T.Parallel(ROWS, BC):
                            # causal + tail masks: ``thr_ub`` holds *global* KV thresholds, so a split
                            # range entirely past a row's frontier masks every position of it.
                            # Nested if_then_else (not a boolean `and`): the flattened compound
                            # predicate mis-lowers on multi-block launches — frontier-truncated
                            # tiles ran with the causal mask effectively dropped (h19 forensics).
                            kv_pos = (start + p) * BC + c
                            masked = T.if_then_else(c < limit, s_ub[r, c] * T.float32(scale), T.float32(-1e30))
                            s_frag[r, c] = T.if_then_else(kv_pos <= thr_ub[r], masked, T.float32(-1e30))
                        row_max = T.alloc_reducer((ROWS,), accum, op="max")
                        T.reducer_init(row_max)
                        for r, c in T.Parallel(ROWS, BC):
                            T.reducer_update(row_max[r], s_frag[r, c])
                        new_m = T.alloc_fragment((ROWS,), accum)
                        T.finalize_reducer(row_max, new_m)

                        alpha = T.alloc_fragment((ROWS,), accum)
                        for r in T.Parallel(ROWS):
                            alpha[r] = T.exp(m_ub[r] - T.max(m_ub[r], new_m[r]))
                            m_ub[r] = T.max(m_ub[r], new_m[r])

                        row_sum = T.alloc_reducer((ROWS,), accum, op="sum")
                        T.reducer_init(row_sum)
                        for r, c in T.Parallel(ROWS, BC):
                            s_frag[r, c] = T.exp(s_frag[r, c] - m_ub[r])
                            T.reducer_update(row_sum[r], s_frag[r, c])
                        partial = T.alloc_fragment((ROWS,), accum)
                        T.finalize_reducer(row_sum, partial)

                        for r in T.Parallel(ROWS):
                            l_ub[r] = l_ub[r] * alpha[r] + partial[r]
                        for r, d in T.Parallel(ROWS, D2):
                            o_ub0[r, d] = o_ub0[r, d] * alpha[r]
                        if D > D2:
                            for r, d in T.Parallel(ROWS, D2):
                                o_ub1[r, d] = o_ub1[r, d] * alpha[r]
                        for r, c in T.Parallel(ROWS, BC):
                            p_ub[r, c] = T.Cast(dtype, s_frag[r, c])

                T.dual_copy(p_ub[0:ROWS, 0:BC], p_l1[0:BR, 0:BC])
                T.copy(VCache[pid, 0:BC, bh, 0:D], v_l1[0:D, 0:BC], transpose=True)
                T.copy(p_l1[0:BR, 0:BC], pv_a)
                T.copy(v_l1[0:D2, 0:BC], pv_b)
                T.gemm(pv_a, pv_b, pv_acc0, transpose_B=True, clear_accum=True)
                if D > D2:
                    T.copy(v_l1[D2:D, 0:BC], pv_b)
                    T.gemm(pv_a, pv_b, pv_acc1, transpose_B=True, clear_accum=True)
                T.dual_copy(pv_acc0, o_tmp0)
                if D > D2:
                    T.dual_copy(pv_acc1, o_tmp1)
                with T.SimtVF(threads=threads):
                    for r, d in T.Parallel(ROWS, D2):
                        o_ub0[r, d] = o_ub0[r, d] + o_tmp0[r, d]
                    if D > D2:
                        for r, d in T.Parallel(ROWS, D2):
                            o_ub1[r, d] = o_ub1[r, d] + o_tmp1[r, d]

            with T.SimtVF(threads=threads):
                # l_safe: an emptied/fully-masked tile carries l=0 — divide by 1 instead, so the
                # slot gets (out=0, lse=-1e30 + log(1)) = the merge's zero-weight filler (the
                # decode split kernel's exact contract).
                for r, d in T.Parallel(ROWS, D2):
                    l_safe = T.if_then_else(l_ub[r] > 0, l_ub[r], T.float32(1))
                    out_ub0[r, d] = T.Cast(dtype, o_ub0[r, d] / l_safe)
                if D > D2:
                    for r, d in T.Parallel(ROWS, D2):
                        l_safe = T.if_then_else(l_ub[r] > 0, l_ub[r], T.float32(1))
                        out_ub1[r, d] = T.Cast(dtype, o_ub1[r, d] / l_safe)
                # the LSE overwrites ``m_ub`` in place: m/l/o_ub are dead from here on, and one
                # fewer 1-D shared buffer keeps this kernel in the 3-buffer layout the plain
                # prefill and decode-split kernels use — with four, the compiler has been
                # observed to overlap their UB ranges (thr_ub/m_ub/l_ub colliding), which
                # corrupted the causal mask on frontier-truncated tiles
                for r in T.Parallel(ROWS):
                    l_safe = T.if_then_else(l_ub[r] > 0, l_ub[r], T.float32(1))
                    m_ub[r] = m_ub[r] + T.log(l_safe)

            T.dual_copy(out_ub0[0:ROWS, 0:D2], PartOut[slot * kv_heads + bh, 0:BR, 0:D2])
            if D > D2:
                T.dual_copy(out_ub1[0:ROWS, 0:D2], PartOut[slot * kv_heads + bh, 0:BR, D2:D])
            T.dual_copy(
                m_ub[0:ROWS],
                PartLse[(slot * kv_heads + bh) * BR : (slot * kv_heads + bh + 1) * BR],
            )

    return main


def paged_prefill_split_head_ascend950(
    batch: int,
    kv_heads: int,
    group: int,
    dim: int,
    page_size: int,
    num_pages_cap: int,
    num_tiles: int,
    max_splits: int,
    block_q: int = 128,
    dtype: str = "bfloat16",
    threads: int = 128,
    num_slots: int | None = None,
):
    """Head-split prefill: the **fully-visible** page chunks of a q-block (H20).

    ``num_slots`` overrides the partial-slot capacity (dense per-tile slot layout: one slot
    per (q-block, split) pair — the default ``batch * max_splits`` only names the decode-path
    shape, which has no q-block axis).

    Arithmetic is identical to :func:`paged_prefill_split_ascend950` minus the per-row causal
    predicate: the scheduler only ever gives a head tile pages entirely before every row's
    causal frontier (``frontier_page_splits``), so the mask reduces to the scalar tail check
    ``c < limit`` and the kernel takes **no ``RowThr`` input at all**.  That is the point: the
    E48 forensics pinned the tilelang failure on the SimtVF per-row ``thr_ub`` predicate under
    multi-block launches, and every kernel here without per-row predicates (decode, decode
    split, plain prefill) is multi-block-clean.  The epilogue writes the same normalised
    partial + LSE slots, so the merge is unchanged.
    """
    BR = block_q
    BC = page_size
    D = dim
    ROWS = BR // 2
    accum = "float32"
    scale = 1.0 / math.sqrt(dim)
    num_slots = num_slots if num_slots is not None else batch * max_splits

    assert BR % 32 == 0
    assert BR <= 128
    assert dim in (128, 256)
    assert page_size % 2 == 0

    @T.prim_func
    def main(
        QPacked: T.Tensor((num_tiles * kv_heads, BR, dim), dtype),
        KCache: T.Tensor((num_pages_cap, page_size, kv_heads, dim), dtype),
        VCache: T.Tensor((num_pages_cap, page_size, kv_heads, dim), dtype),
        kv_indptr: T.Tensor((batch + 1,), "int32"),
        kv_indices: T.Tensor((num_pages_cap,), "int32"),
        kv_last_page_len: T.Tensor((batch,), "int32"),
        seq_ids: T.Tensor((num_tiles,), "int32"),
        tile_rows: T.Tensor((num_tiles,), "int32"),
        kv_page_starts: T.Tensor((num_tiles,), "int32"),
        TilePageLens: T.Tensor((num_tiles,), "int32"),
        tile_slots: T.Tensor((num_tiles,), "int32"),
        PartOut: T.Tensor((num_slots * kv_heads, BR, dim), dtype),
        PartLse: T.Tensor((num_slots * kv_heads * BR,), accum),
    ):
        with T.Kernel(num_tiles * kv_heads) as bx:
            tile = bx // kv_heads
            bh = bx % kv_heads
            b = seq_ids[tile]
            start = kv_page_starts[tile]
            req_pages = kv_indptr[b + 1] - kv_indptr[b]
            # Head tiles may include the request's last page only when the frontier split is
            # absent (P_vis == pages); keep the same truncation contract as the split kernel.
            plen = TilePageLens[tile]
            eff = req_pages - start
            eff = T.if_then_else(eff < 0, 0, eff)
            plen = T.if_then_else(plen < eff, plen, eff)
            plen = T.if_then_else(plen < 0, 0, plen)
            slot = tile_slots[tile]

            q_l1 = T.alloc_l1((BR, D), dtype)
            k_l1 = T.alloc_l1((BC, D), dtype)
            v_l1 = T.alloc_l1((D, BC), dtype)
            p_l1 = T.alloc_l1((BR, BC), dtype)

            D2 = min(D, 128)
            qk_a = T.alloc_l0a((BR, D2), dtype)
            qk_b = T.alloc_l0b((BC, D2), dtype)
            pv_a = T.alloc_l0a((BR, BC), dtype)
            pv_b = T.alloc_l0b((D2, BC), dtype)
            qk_acc = T.alloc_l0c((BR, BC), accum)
            pv_acc0 = T.alloc_l0c((BR, D2), accum)
            if D > D2:
                pv_acc1 = T.alloc_l0c((BR, D2), accum)

            s_ub = T.alloc_shared((ROWS, BC), accum)
            p_ub = T.alloc_shared((ROWS, BC), dtype)
            o_ub0 = T.alloc_shared((ROWS, D2), accum)
            o_tmp0 = T.alloc_shared((ROWS, D2), accum)
            out_ub0 = T.alloc_shared((ROWS, D2), dtype)
            if D > D2:
                o_ub1 = T.alloc_shared((ROWS, D2), accum)
                o_tmp1 = T.alloc_shared((ROWS, D2), accum)
                out_ub1 = T.alloc_shared((ROWS, D2), dtype)
            m_ub = T.alloc_shared((ROWS,), accum)
            l_ub = T.alloc_shared((ROWS,), accum)
            if _simd_softmax_enabled():
                a_ub = T.alloc_shared((ROWS,), accum)
                ll_ub = T.alloc_shared((1,), "int32")

            T.copy(QPacked[bx, 0:BR, 0:D], q_l1)
            T.fill(m_ub, T.float32(-1e30))
            T.fill(l_ub, T.float32(0))
            T.fill(o_ub0, T.float32(0))
            if D > D2:
                T.fill(o_ub1, T.float32(0))
            if _simd_softmax_enabled():
                T.copy(kv_last_page_len[b : b + 1], ll_ub)
            for p in T.serial(plen):
                pid = kv_indices[kv_indptr[b] + start + p]

                T.copy(KCache[pid, 0:BC, bh, 0:D], k_l1)
                T.copy(q_l1[0:BR, 0:D2], qk_a)
                T.copy(k_l1[0:BC, 0:D2], qk_b)
                T.gemm(qk_a, qk_b, qk_acc, transpose_B=True, clear_accum=True)
                if D > D2:
                    T.copy(q_l1[0:BR, D2:D], qk_a)
                    T.copy(k_l1[0:BC, D2:D], qk_b)
                    T.gemm(qk_a, qk_b, qk_acc, transpose_B=True)
                T.dual_copy(qk_acc, s_ub)

                if _simd_softmax_enabled():
                    # T5 packed-SIMD online softmax; head tiles carry only the scalar tail mask (no
                    # per-row causal predicate, H20/E48).
                    limit = T.if_then_else(start + p + 1 < req_pages, BC, ll_ub[0])
                    with T.SimdVF():
                        for r in T.serial(ROWS):
                            idx0 = T.simd.vci(0, "int32", order="INC_ORDER")
                            idx1 = T.simd.vci(64, "int32", order="INC_ORDER")
                            sc = T.simd.vdup(T.float32(scale), "float32")
                            a0 = T.simd.vmul(T.simd.vld(s_ub[r, 0], dist="NORM"), sc)
                            a1 = T.simd.vmul(T.simd.vld(s_ub[r, 64], dist="NORM"), sc)
                            neg = T.simd.vdup(T.float32(-1e30), "float32")
                            a0 = T.simd.vsel(a0, neg, T.simd.vcmps(idx0, limit, op="lt"))
                            a1 = T.simd.vsel(a1, neg, T.simd.vcmps(idx1, limit, op="lt"))
                            pm = T.simd.vdupv(T.simd.vld(m_ub[r], dist="BRC_B32"), pos="POS_LOWEST")
                            nm = T.simd.vmax(pm, T.simd.vmax(T.simd.vcmax(a0), T.simd.vcmax(a1)))
                            nml = T.simd.vdupv(nm, pos="POS_LOWEST")
                            al = T.simd.vexp(T.simd.vsub(pm, nml))
                            e0 = T.simd.vexp(T.simd.vsub(a0, nml))
                            e1 = T.simd.vexp(T.simd.vsub(a1, nml))
                            lp = T.simd.vdupv(T.simd.vld(l_ub[r], dist="BRC_B32"), pos="POS_LOWEST")
                            ln = T.simd.vadd(
                                T.simd.vmul(lp, al),
                                T.simd.vadd(T.simd.vcadd(e0), T.simd.vcadd(e1)),
                            )
                            T.simd.vsts(m_ub[r], nml, dist="ONEPT_B32")
                            T.simd.vsts(l_ub[r], T.simd.vdupv(ln, pos="POS_LOWEST"), dist="ONEPT_B32")
                            T.simd.vsts(a_ub[r], T.simd.vdupv(al, pos="POS_LOWEST"), dist="ONEPT_B32")
                            T.simd.vsts(s_ub[r, 0], e0)
                            T.simd.vsts(s_ub[r, 64], e1)
                    with T.SimtVF(threads=threads):
                        for r, d in T.Parallel(ROWS, D2):
                            o_ub0[r, d] = o_ub0[r, d] * a_ub[r]
                        if D > D2:
                            for r, d in T.Parallel(ROWS, D2):
                                o_ub1[r, d] = o_ub1[r, d] * a_ub[r]
                        for r, c in T.Parallel(ROWS, BC):
                            p_ub[r, c] = T.Cast(dtype, s_ub[r, c])
                else:
                    limit = T.if_then_else(start + p + 1 < req_pages, BC, kv_last_page_len[b])
                    with T.SimtVF(threads=threads):
                        s_frag = T.alloc_fragment((ROWS, BC), accum)
                        # Scalar tail mask only — deliberately no per-row predicate (H20/E48).
                        for r, c in T.Parallel(ROWS, BC):
                            s_frag[r, c] = T.if_then_else(
                                c < limit, s_ub[r, c] * T.float32(scale), T.float32(-1e30)
                            )
                        row_max = T.alloc_reducer((ROWS,), accum, op="max")
                        T.reducer_init(row_max)
                        for r, c in T.Parallel(ROWS, BC):
                            T.reducer_update(row_max[r], s_frag[r, c])
                        new_m = T.alloc_fragment((ROWS,), accum)
                        T.finalize_reducer(row_max, new_m)

                        alpha = T.alloc_fragment((ROWS,), accum)
                        for r in T.Parallel(ROWS):
                            alpha[r] = T.exp(m_ub[r] - T.max(m_ub[r], new_m[r]))
                            m_ub[r] = T.max(m_ub[r], new_m[r])

                        row_sum = T.alloc_reducer((ROWS,), accum, op="sum")
                        T.reducer_init(row_sum)
                        for r, c in T.Parallel(ROWS, BC):
                            s_frag[r, c] = T.exp(s_frag[r, c] - m_ub[r])
                            T.reducer_update(row_sum[r], s_frag[r, c])
                        partial = T.alloc_fragment((ROWS,), accum)
                        T.finalize_reducer(row_sum, partial)

                        for r in T.Parallel(ROWS):
                            l_ub[r] = l_ub[r] * alpha[r] + partial[r]
                        for r, d in T.Parallel(ROWS, D2):
                            o_ub0[r, d] = o_ub0[r, d] * alpha[r]
                        if D > D2:
                            for r, d in T.Parallel(ROWS, D2):
                                o_ub1[r, d] = o_ub1[r, d] * alpha[r]
                        for r, c in T.Parallel(ROWS, BC):
                            p_ub[r, c] = T.Cast(dtype, s_frag[r, c])

                T.dual_copy(p_ub[0:ROWS, 0:BC], p_l1[0:BR, 0:BC])
                T.copy(VCache[pid, 0:BC, bh, 0:D], v_l1[0:D, 0:BC], transpose=True)
                T.copy(p_l1[0:BR, 0:BC], pv_a)
                T.copy(v_l1[0:D2, 0:BC], pv_b)
                T.gemm(pv_a, pv_b, pv_acc0, transpose_B=True, clear_accum=True)
                if D > D2:
                    T.copy(v_l1[D2:D, 0:BC], pv_b)
                    T.gemm(pv_a, pv_b, pv_acc1, transpose_B=True, clear_accum=True)
                T.dual_copy(pv_acc0, o_tmp0)
                if D > D2:
                    T.dual_copy(pv_acc1, o_tmp1)
                with T.SimtVF(threads=threads):
                    for r, d in T.Parallel(ROWS, D2):
                        o_ub0[r, d] = o_ub0[r, d] + o_tmp0[r, d]
                    if D > D2:
                        for r, d in T.Parallel(ROWS, D2):
                            o_ub1[r, d] = o_ub1[r, d] + o_tmp1[r, d]

            with T.SimtVF(threads=threads):
                # l_safe: an emptied tile carries l=0 -> (out=0, lse=-1e30 + log(1)), the
                # merge's zero-weight filler (same contract as the decode split kernel).
                for r, d in T.Parallel(ROWS, D2):
                    l_safe = T.if_then_else(l_ub[r] > 0, l_ub[r], T.float32(1))
                    out_ub0[r, d] = T.Cast(dtype, o_ub0[r, d] / l_safe)
                if D > D2:
                    for r, d in T.Parallel(ROWS, D2):
                        l_safe = T.if_then_else(l_ub[r] > 0, l_ub[r], T.float32(1))
                        out_ub1[r, d] = T.Cast(dtype, o_ub1[r, d] / l_safe)
                for r in T.Parallel(ROWS):
                    l_safe = T.if_then_else(l_ub[r] > 0, l_ub[r], T.float32(1))
                    m_ub[r] = m_ub[r] + T.log(l_safe)

            T.dual_copy(out_ub0[0:ROWS, 0:D2], PartOut[slot * kv_heads + bh, 0:BR, 0:D2])
            if D > D2:
                T.dual_copy(out_ub1[0:ROWS, 0:D2], PartOut[slot * kv_heads + bh, 0:BR, D2:D])
            T.dual_copy(
                m_ub[0:ROWS],
                PartLse[(slot * kv_heads + bh) * BR : (slot * kv_heads + bh + 1) * BR],
            )

    return main


def paged_prefill_split_frontier_ascend950(
    batch: int,
    kv_heads: int,
    group: int,
    dim: int,
    page_size: int,
    num_pages_cap: int,
    num_tiles: int,
    max_splits: int,
    block_q: int = 128,
    dtype: str = "bfloat16",
    threads: int = 128,
    num_slots: int | None = None,
):
    """Frontier-split prefill: the causal-frontier page chunk (H20, 3e mask form).

    ``num_slots`` overrides the partial-slot capacity (dense per-tile slot layout, see the
    head kernel).

    The exact :func:`paged_prefill_ascend950` body — including the per-row ``RowThr`` causal
    mask — with three deltas taken from the split kernel: the tile starts at
    ``kv_page_starts[tile]`` instead of page 0, ``plen`` is clamped to the pages that actually
    exist, and the epilogue writes the normalised partial plus LSE into the dense slot layout
    (instead of a final output), so the unchanged merge combines it with the head splits'
    partials.  The scheduler gives each q-block exactly one frontier tile covering
    ``[P_vis, pages_needed)``; every page it visits may be partially masked for some rows,
    which the ``RowThr`` predicate handles.

    Mask-form history (the tilelang 0.1.15 failure domain, E48/E54-E57): the plain kernel's
    *vectorized* compound predicate (literal syntactic parity, E50/D52) compiles for plain but
    sends this kernel's pass pipeline into a >60 min blowup (E54; the head kernel — identical
    minus the per-row term — compiles in 2m14s).  Scalarizing the predicate straight into the
    distributed fragment (3b) compiles (29.3s) but mis-numbers (E56/E57: flat ~0.25 error,
    single tile already wrong) — serial-in-SimtVF fragment-write ownership is the suspect.
    The working form (3e, iter-047) computes the mask element-by-element into **shared**
    (``s_ub`` in place) and feeds the fragment through the plain kernels' proven Parallel
    shared→fragment copy; every other statement of the softmax body is byte-identical to the
    green head kernel (E56).
    """
    BR = block_q
    BC = page_size
    D = dim
    ROWS = BR // 2
    accum = "float32"
    scale = 1.0 / math.sqrt(dim)
    num_slots = num_slots if num_slots is not None else batch * max_splits

    assert BR % 32 == 0
    assert BR <= 128
    assert dim in (128, 256)
    assert page_size % 2 == 0

    @T.prim_func
    def main(
        QPacked: T.Tensor((num_tiles * kv_heads, BR, dim), dtype),
        KCache: T.Tensor((num_pages_cap, page_size, kv_heads, dim), dtype),
        VCache: T.Tensor((num_pages_cap, page_size, kv_heads, dim), dtype),
        kv_indptr: T.Tensor((batch + 1,), "int32"),
        kv_indices: T.Tensor((num_pages_cap,), "int32"),
        kv_last_page_len: T.Tensor((batch,), "int32"),
        seq_ids: T.Tensor((num_tiles,), "int32"),
        tile_rows: T.Tensor((num_tiles,), "int32"),
        kv_page_starts: T.Tensor((num_tiles,), "int32"),
        causal_shift: T.Tensor((batch,), "int32"),
        RowThr: T.Tensor((num_tiles * BR,), "int32"),
        TilePageLens: T.Tensor((num_tiles,), "int32"),
        tile_slots: T.Tensor((num_tiles,), "int32"),
        PartOut: T.Tensor((num_slots * kv_heads, BR, dim), dtype),
        PartLse: T.Tensor((num_slots * kv_heads * BR,), accum),
    ):
        with T.Kernel(num_tiles * kv_heads) as bx:
            tile = bx // kv_heads
            bh = bx % kv_heads
            b = seq_ids[tile]
            start = kv_page_starts[tile]
            req_pages = kv_indptr[b + 1] - kv_indptr[b]
            plen = TilePageLens[tile]
            eff = req_pages - start
            eff = T.if_then_else(eff < 0, 0, eff)
            plen = T.if_then_else(plen < eff, plen, eff)
            plen = T.if_then_else(plen < 0, 0, plen)
            slot = tile_slots[tile]

            q_l1 = T.alloc_l1((BR, D), dtype)
            k_l1 = T.alloc_l1((BC, D), dtype)
            v_l1 = T.alloc_l1((D, BC), dtype)
            p_l1 = T.alloc_l1((BR, BC), dtype)

            D2 = min(D, 128)
            qk_a = T.alloc_l0a((BR, D2), dtype)
            qk_b = T.alloc_l0b((BC, D2), dtype)
            pv_a = T.alloc_l0a((BR, BC), dtype)
            pv_b = T.alloc_l0b((D2, BC), dtype)
            qk_acc = T.alloc_l0c((BR, BC), accum)
            pv_acc0 = T.alloc_l0c((BR, D2), accum)
            if D > D2:
                pv_acc1 = T.alloc_l0c((BR, D2), accum)

            s_ub = T.alloc_shared((ROWS, BC), accum)
            p_ub = T.alloc_shared((ROWS, BC), dtype)
            o_ub0 = T.alloc_shared((ROWS, D2), accum)
            o_tmp0 = T.alloc_shared((ROWS, D2), accum)
            out_ub0 = T.alloc_shared((ROWS, D2), dtype)
            if D > D2:
                o_ub1 = T.alloc_shared((ROWS, D2), accum)
                o_tmp1 = T.alloc_shared((ROWS, D2), accum)
                out_ub1 = T.alloc_shared((ROWS, D2), dtype)
            m_ub = T.alloc_shared((ROWS,), accum)
            l_ub = T.alloc_shared((ROWS,), accum)
            # Per-row causal threshold for this AIV's half of the M tile (dual_copy splits it
            # exactly like s_ub, see paged_prefill_ascend950).
            thr_ub = T.alloc_shared((ROWS,), "int32")
            if _simd_softmax_enabled():
                a_ub = T.alloc_shared((ROWS,), accum)
                ll_ub = T.alloc_shared((1,), "int32")

            T.copy(QPacked[bx, 0:BR, 0:D], q_l1)
            T.dual_copy(RowThr[tile * BR : (tile + 1) * BR], thr_ub)
            T.fill(m_ub, T.float32(-1e30))
            T.fill(l_ub, T.float32(0))
            T.fill(o_ub0, T.float32(0))
            if D > D2:
                T.fill(o_ub1, T.float32(0))
            if _simd_softmax_enabled():
                T.copy(kv_last_page_len[b : b + 1], ll_ub)
            for p in T.serial(plen):
                pid = kv_indices[kv_indptr[b] + start + p]

                T.copy(KCache[pid, 0:BC, bh, 0:D], k_l1)
                T.copy(q_l1[0:BR, 0:D2], qk_a)
                T.copy(k_l1[0:BC, 0:D2], qk_b)
                T.gemm(qk_a, qk_b, qk_acc, transpose_B=True, clear_accum=True)
                if D > D2:
                    T.copy(q_l1[0:BR, D2:D], qk_a)
                    T.copy(k_l1[0:BC, D2:D], qk_b)
                    T.gemm(qk_a, qk_b, qk_acc, transpose_B=True)
                T.dual_copy(qk_acc, s_ub)

                # Page offset enters only through host-side values: RowThr arrives pre-relativised
                # (thr − start·BC, plan_prefill_split_ctx) and eff is a kernel scalar — p is the
                # only loop variable in the mask (E50/D52 relativisation, kept in 3e).
                if _simd_softmax_enabled():
                    # T5 packed-SIMD online softmax (causal + tail mask, two chained `vsel`s).
                    # RowThr is pre-relativised, so `kv_pos` is the in-request column `p*BC + lane`.
                    limit = T.if_then_else(p + 1 < eff, BC, ll_ub[0])
                    with T.SimdVF():
                        for r in T.serial(ROWS):
                            idx0 = T.simd.vci(0, "int32", order="INC_ORDER")
                            idx1 = T.simd.vci(64, "int32", order="INC_ORDER")
                            kp0 = T.simd.vadds(T.simd.vci(0, "int32", order="INC_ORDER"), p * BC)
                            kp1 = T.simd.vadds(T.simd.vci(64, "int32", order="INC_ORDER"), p * BC)
                            tb = T.simd.vdupv(T.simd.vld(thr_ub[r], dist="BRC_B32"), pos="POS_LOWEST")
                            sc = T.simd.vdup(T.float32(scale), "float32")
                            a0 = T.simd.vmul(T.simd.vld(s_ub[r, 0], dist="NORM"), sc)
                            a1 = T.simd.vmul(T.simd.vld(s_ub[r, 64], dist="NORM"), sc)
                            neg = T.simd.vdup(T.float32(-1e30), "float32")
                            a0 = T.simd.vsel(a0, neg, T.simd.vcmps(idx0, limit, op="lt"))
                            a1 = T.simd.vsel(a1, neg, T.simd.vcmps(idx1, limit, op="lt"))
                            a0 = T.simd.vsel(a0, neg, T.simd.vcmp(kp0, tb, op="le"))
                            a1 = T.simd.vsel(a1, neg, T.simd.vcmp(kp1, tb, op="le"))
                            pm = T.simd.vdupv(T.simd.vld(m_ub[r], dist="BRC_B32"), pos="POS_LOWEST")
                            nm = T.simd.vmax(pm, T.simd.vmax(T.simd.vcmax(a0), T.simd.vcmax(a1)))
                            nml = T.simd.vdupv(nm, pos="POS_LOWEST")
                            al = T.simd.vexp(T.simd.vsub(pm, nml))
                            e0 = T.simd.vexp(T.simd.vsub(a0, nml))
                            e1 = T.simd.vexp(T.simd.vsub(a1, nml))
                            lp = T.simd.vdupv(T.simd.vld(l_ub[r], dist="BRC_B32"), pos="POS_LOWEST")
                            ln = T.simd.vadd(
                                T.simd.vmul(lp, al),
                                T.simd.vadd(T.simd.vcadd(e0), T.simd.vcadd(e1)),
                            )
                            T.simd.vsts(m_ub[r], nml, dist="ONEPT_B32")
                            T.simd.vsts(l_ub[r], T.simd.vdupv(ln, pos="POS_LOWEST"), dist="ONEPT_B32")
                            T.simd.vsts(a_ub[r], T.simd.vdupv(al, pos="POS_LOWEST"), dist="ONEPT_B32")
                            T.simd.vsts(s_ub[r, 0], e0)
                            T.simd.vsts(s_ub[r, 64], e1)
                    with T.SimtVF(threads=threads):
                        for r, d in T.Parallel(ROWS, D2):
                            o_ub0[r, d] = o_ub0[r, d] * a_ub[r]
                        if D > D2:
                            for r, d in T.Parallel(ROWS, D2):
                                o_ub1[r, d] = o_ub1[r, d] * a_ub[r]
                        for r, c in T.Parallel(ROWS, BC):
                            p_ub[r, c] = T.Cast(dtype, s_ub[r, c])
                else:
                    limit = T.if_then_else(p + 1 < eff, BC, kv_last_page_len[b])
                    with T.SimtVF(threads=threads):
                        # 3e (iter-047, E57 结论 3): 逐元素标量谓词写 **shared**（s_ub 就地），
                        # fragment 由与 plain/head 内核同构的 Parallel 拷贝填充。E57 候选①：
                        # 3b 的 serial-in-SimtVF 直接写分布式 fragment（s_frag）时 owned-element
                        # 判定失准/写序无屏障 → 单 tile 即错、误差平坦（0.24-0.28）；shared 写在
                        # SimtVF 内有在役先例（m_ub/l_ub/o_ub*），serial 仅决定执行序、无归属歧义，
                        # 而 Parallel 拷贝（shared 读 → fragment 写）是 plain/head 内核已验证形态。
                        # 其余 softmax 体与 head 内核（E56 数值已绿 0.0005）逐字节一致。
                        # （历史：向量化复合谓词 = plain 字面同构，但编译 >60min blowup，E54/E55。）
                        for r in T.serial(ROWS):
                            thr_r = thr_ub[r]
                            for c in T.serial(BC):
                                kv_pos = p * BC + c
                                s_ub[r, c] = T.if_then_else(
                                    (c < limit) and (kv_pos <= thr_r),
                                    s_ub[r, c] * T.float32(scale),
                                    T.float32(-1e30),
                                )
                        s_frag = T.alloc_fragment((ROWS, BC), accum)
                        for r, c in T.Parallel(ROWS, BC):
                            s_frag[r, c] = s_ub[r, c]
                        row_max = T.alloc_reducer((ROWS,), accum, op="max")
                        T.reducer_init(row_max)
                        for r, c in T.Parallel(ROWS, BC):
                            T.reducer_update(row_max[r], s_frag[r, c])
                        new_m = T.alloc_fragment((ROWS,), accum)
                        T.finalize_reducer(row_max, new_m)

                        alpha = T.alloc_fragment((ROWS,), accum)
                        for r in T.Parallel(ROWS):
                            alpha[r] = T.exp(m_ub[r] - T.max(m_ub[r], new_m[r]))
                            m_ub[r] = T.max(m_ub[r], new_m[r])

                        row_sum = T.alloc_reducer((ROWS,), accum, op="sum")
                        T.reducer_init(row_sum)
                        for r, c in T.Parallel(ROWS, BC):
                            s_frag[r, c] = T.exp(s_frag[r, c] - m_ub[r])
                            T.reducer_update(row_sum[r], s_frag[r, c])
                        partial = T.alloc_fragment((ROWS,), accum)
                        T.finalize_reducer(row_sum, partial)

                        for r in T.Parallel(ROWS):
                            l_ub[r] = l_ub[r] * alpha[r] + partial[r]
                        for r, d in T.Parallel(ROWS, D2):
                            o_ub0[r, d] = o_ub0[r, d] * alpha[r]
                        if D > D2:
                            for r, d in T.Parallel(ROWS, D2):
                                o_ub1[r, d] = o_ub1[r, d] * alpha[r]
                        for r, c in T.Parallel(ROWS, BC):
                            p_ub[r, c] = T.Cast(dtype, s_frag[r, c])

                T.dual_copy(p_ub[0:ROWS, 0:BC], p_l1[0:BR, 0:BC])
                T.copy(VCache[pid, 0:BC, bh, 0:D], v_l1[0:D, 0:BC], transpose=True)
                T.copy(p_l1[0:BR, 0:BC], pv_a)
                T.copy(v_l1[0:D2, 0:BC], pv_b)
                T.gemm(pv_a, pv_b, pv_acc0, transpose_B=True, clear_accum=True)
                if D > D2:
                    T.copy(v_l1[D2:D, 0:BC], pv_b)
                    T.gemm(pv_a, pv_b, pv_acc1, transpose_B=True, clear_accum=True)
                T.dual_copy(pv_acc0, o_tmp0)
                if D > D2:
                    T.dual_copy(pv_acc1, o_tmp1)
                with T.SimtVF(threads=threads):
                    for r, d in T.Parallel(ROWS, D2):
                        o_ub0[r, d] = o_ub0[r, d] + o_tmp0[r, d]
                    if D > D2:
                        for r, d in T.Parallel(ROWS, D2):
                            o_ub1[r, d] = o_ub1[r, d] + o_tmp1[r, d]

            with T.SimtVF(threads=threads):
                for r, d in T.Parallel(ROWS, D2):
                    l_safe = T.if_then_else(l_ub[r] > 0, l_ub[r], T.float32(1))
                    out_ub0[r, d] = T.Cast(dtype, o_ub0[r, d] / l_safe)
                if D > D2:
                    for r, d in T.Parallel(ROWS, D2):
                        l_safe = T.if_then_else(l_ub[r] > 0, l_ub[r], T.float32(1))
                        out_ub1[r, d] = T.Cast(dtype, o_ub1[r, d] / l_safe)
                for r in T.Parallel(ROWS):
                    l_safe = T.if_then_else(l_ub[r] > 0, l_ub[r], T.float32(1))
                    m_ub[r] = m_ub[r] + T.log(l_safe)

            T.dual_copy(out_ub0[0:ROWS, 0:D2], PartOut[slot * kv_heads + bh, 0:BR, 0:D2])
            if D > D2:
                T.dual_copy(out_ub1[0:ROWS, 0:D2], PartOut[slot * kv_heads + bh, 0:BR, D2:D])
            T.dual_copy(
                m_ub[0:ROWS],
                PartLse[(slot * kv_heads + bh) * BR : (slot * kv_heads + bh + 1) * BR],
            )

    return main


def build_prefill_split_kernel(**spec):
    """Compile (and cache) the split-KV prefill kernel (H19).  See `build_prefill_kernel`
    re: `num_pages_cap` (quantized to the ABI's declared capacity, key stays pool-agnostic)."""
    if spec.get("num_pages_cap") is not None:
        spec["num_pages_cap"] = declared_pool_cap(spec["num_pages_cap"])
    key = _cache_key("prefill_split", spec)
    kernel = _KERNEL_CACHE.get(key)
    if kernel is None:
        kernel = tilelang.compile(
            paged_prefill_split_ascend950(**spec),
            out_idx=[],
            pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        )
        _KERNEL_CACHE[key] = kernel
    return kernel


def build_prefill_split_head_kernel(**spec):
    """Compile (and cache) the head-split prefill kernel (H20).  Same capacity/key contract
    as :func:`build_prefill_split_kernel`."""
    if spec.get("num_pages_cap") is not None:
        spec["num_pages_cap"] = declared_pool_cap(spec["num_pages_cap"])
    key = _cache_key("prefill_split_head", spec)
    kernel = _KERNEL_CACHE.get(key)
    if kernel is None:
        kernel = tilelang.compile(
            paged_prefill_split_head_ascend950(**spec),
            out_idx=[],
            pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        )
        _KERNEL_CACHE[key] = kernel
    return kernel


def build_prefill_split_frontier_kernel(**spec):
    """Compile (and cache) the frontier-split prefill kernel (H20).  Same capacity/key
    contract as :func:`build_prefill_split_kernel`."""
    if spec.get("num_pages_cap") is not None:
        spec["num_pages_cap"] = declared_pool_cap(spec["num_pages_cap"])
    key = _cache_key("prefill_split_frontier", spec)
    kernel = _KERNEL_CACHE.get(key)
    if kernel is None:
        kernel = tilelang.compile(
            paged_prefill_split_frontier_ascend950(**spec),
            out_idx=[],
            pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        )
        _KERNEL_CACHE[key] = kernel
    return kernel


def build_tile_slots(schedule, max_splits: int) -> torch.Tensor:
    """Dense slot index for every tile: ``slot = request * max_splits + split_id``.

    Built once per plan (the schedule is fixed for a shape), which is what lets the split kernel
    write and the merge kernel read without any runtime conditionals.
    """
    return (schedule.seq_ids.to(torch.int32) * max_splits + schedule.split_ids.to(torch.int32)).contiguous()


def forward_split(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    meta,
    schedule,
    *,
    group: int,
    max_splits: int,
    tile_slots: torch.Tensor,
    part_out: torch.Tensor,
    part_lse: torch.Tensor,
    out_pad: torch.Tensor,
    q_pad: torch.Tensor,
    dtype: str | None = None,
    kv_indptr: torch.Tensor | None = None,
    kv_indices: torch.Tensor | None = None,
    kv_last_page_len: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Two-launch split-KV decode: partial attention, then the weighted merge.

    Every buffer is owned by the plan (see the backend's ``finalize_plan``): the run stage only
    copies the step's query into ``q_pad`` and launches, so it stays allocation-free and
    capturable.  ``part_lse``'s padding slots are pre-filled with ``-1e30`` once per plan.
    """
    import torch

    kv_heads = k_cache.shape[2]
    page_size = int(k_cache.shape[1])
    br = padded_rows(group)
    num_tiles = schedule.num_tiles
    dtype = dtype or _TORCH_TO_TILELANG_DTYPE[q.dtype]
    # Declared ABI capacity (covers the real pool; see declared_pool_cap): sizes the split
    # kernel's kv_indices slot and its K/V page extents (the pools are capacity views).
    cap = declared_pool_cap(int(k_cache.shape[0]))

    if q.dim() == 3:
        # Integration path: q is the caller's *already padded* staging buffer
        # ([batch, kv_heads*BR, dim] — what the vLLM integration fills before run()), passed
        # in place of the `q_pad` argument, which is then ignored.  The result stays 3-D
        # (the integration copies it into the engine's output buffer itself).
        batch, hbr, dim = q.shape
        assert hbr == kv_heads * br, (hbr, kv_heads, br)
        q_pad = q
    else:
        batch, num_qo_heads, qo_len, dim = q.shape
        assert qo_len == 1
        q_pad.view(batch, kv_heads, br, dim)[:, :, :group] = q[:, :, 0, :].view(
            batch, kv_heads, group, dim
        )

    split = build_split_kernel(
        batch=batch,
        kv_heads=kv_heads,
        group=group,
        dim=dim,
        page_size=page_size,
        num_pages_cap=cap,
        num_tiles=num_tiles,
        max_splits=max_splits,
        dtype=dtype,
    )
    merge = build_merge_kernel(
        batch=batch, kv_heads=kv_heads, group=group, dim=dim, max_splits=max_splits, dtype=dtype
    )

    # Prefer the plan-owned metadata buffers (stable addresses, caller tensors untouched).
    if kv_indices is not None:
        indices = kv_indices
        indptr = kv_indptr
        last_page_len = kv_last_page_len
    else:
        indices = meta.kv_indices.to(torch.int32)
        if indices.numel() < cap:
            indices = torch.cat([indices, indices.new_zeros(cap - indices.numel())])
        indptr = meta.kv_indptr.to(torch.int32)
        last_page_len = meta.kv_last_page_len.to(torch.int32)

    import os as _os

    _watch = meta.kv_indices.cpu().clone() if _os.environ.get("TILEINFER_SPLIT_DEBUG") else None

    def _hook(tag):
        if _watch is not None:
            torch.npu.synchronize()
            caller_ok = bool((meta.kv_indices.cpu() == _watch).all())
            plan_ok = bool((indices.cpu() == _watch[: indices.numel()]).all())
            print(
                f"    [split-debug] {tag}: caller_kv_indices_ok={caller_ok} "
                f"plan_copy_ok={plan_ok}",
                flush=True,
            )

    _hook("before split")
    # H67:token 级窗口。未启用时传全 0 起点 + 全量长度,内核走与原版逐位一致的路径。
    if getattr(schedule, "token_split", False) and schedule.token_starts is not None:
        tok_starts = schedule.token_starts.to(torch.int32)
        tok_lens = schedule.token_lens.to(torch.int32)
    else:
        tok_starts = torch.zeros(num_tiles, dtype=torch.int32, device=q_pad.device)
        tok_lens = schedule.kv_page_lens.to(torch.int32) * page_size
    split(
        q_pad,
        pool_view(k_cache, cap),
        pool_view(v_cache, cap),
        indptr,
        indices,
        last_page_len,
        schedule.seq_ids.to(torch.int32),
        schedule.kv_page_starts.to(torch.int32),
        schedule.kv_page_lens.to(torch.int32),
        tile_slots,
        tok_starts,
        tok_lens,
        part_out,
        part_lse,
    )
    _hook("after split")
    merge(part_out, part_lse, out_pad)
    _hook("after merge")

    if q.dim() == 3:
        # integration path: the engine integration reads out_pad through its own plan-owned
        # view — hand back the padded buffer untouched
        return out_pad

    result = out_pad.view(batch, kv_heads, br, dim)[:, :, :group].reshape(
        batch, num_qo_heads, dim
    )
    result = result.unsqueeze(2)
    if out is not None:
        out.copy_(result)
        return out
    return result


def forward_prefill(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    meta,
    schedule,
    qo_lens: torch.Tensor,
    *,
    group: int,
    block_q: int = 128,
    dtype: str | None = None,
    out: torch.Tensor | None = None,
    kernel_batch: int | None = None,
    pv_single_pass: bool = False,
) -> torch.Tensor:
    """Run paged prefill / append: packs ``Q`` into the kernel's tile layout, unpacks the result.

    ``q``         ``[batch, kv_heads * group, max_qo_len, dim]``; rows past ``qo_lens[b]`` are ignored
    ``schedule``  from :func:`tileinfer.plan.plan_query_tiles` over the *flattened* row counts
                  ``qo_lens * group`` — a tile's rows are contiguous in ``(q_pos, head)`` order;
                  may be *padded* past the live tiles (P3): padded tiles carry ``q_len == 0`` and
                  ``seq_ids`` pointing at a trash row, and are skipped by the pack/unpack loops
    ``qo_lens``   ``[batch]``, used for the append offset (``kv_len - qo_len``)
    ``kernel_batch``
                  compile/call the kernel for a *padded* batch (P3 canonical-variant quantization):
                  ``kv_indptr``/``kv_last_page_len``/``causal_shift`` grow by trash rows (0 pages,
                  never referenced by a live tile) so one compiled variant serves every real batch
                  up to the bucket.  ``None`` = use ``q.shape[0]`` (exact batch, the old behaviour).

    Two host-side steps, both plain tensor ops: the kernel wants rows ordered ``(q_pos, head)``
    inside one KV head (the caller's tensor is ``(head, q_pos)``), and tiles are addressed per
    ``(request-tile, kv_head)``.  Packing costs one pass over Q, which is nothing next to the K/V
    traffic a prefill reads.
    """
    import torch

    batch, num_qo_heads, max_qo_len, dim = q.shape
    kv_heads = k_cache.shape[2]
    page_size = int(k_cache.shape[1])
    dtype = dtype or _TORCH_TO_TILELANG_DTYPE[q.dtype]
    br = block_q
    num_tiles = schedule.num_tiles
    assert num_qo_heads == kv_heads * group

    # [batch, kv_heads, group, q_pos, dim] -> [batch, kv_heads, q_pos * group, dim]
    q_flat = (
        q.view(batch, kv_heads, group, max_qo_len, dim)
        .permute(0, 1, 3, 2, 4)
        .reshape(batch, kv_heads, max_qo_len * group, dim)
        .contiguous()
    )

    # P3 padded tiles (q_len == 0, seq_id -> trash row) carry no query rows; they stay zero in
    # ``q_packed`` and are skipped on unpack.  Everything the kernel sees stays shape-static.
    #
    # Pack/unpack are index-driven instead of per-tile Python loops: the old form did
    # ``nonzero().tolist()`` + 3 ``.item()`` per tile (a D2H sync each) + one slice launch per
    # (tile, kv_head).  At L=8192 that is ~256 tiles → ~1.5k syncs and ~2k launches on every
    # prefill call, which the TTFT budget cannot afford.  The schedule tensors are already on
    # the device, so both directions become one vectorized gather/scatter with **zero** D2H.
    num_slots = num_tiles * kv_heads
    rows_per_req = max_qo_len * group
    dev = q.device
    # Flat row index of each packed slot's first row inside ``q_flat`` ([batch, kv_heads, rows, dim]).
    seq_ids64 = schedule.seq_ids.to(device=dev, dtype=torch.int64)
    q_off64 = schedule.q_offsets.to(device=dev, dtype=torch.int64)
    q_len64 = schedule.q_lens.to(device=dev, dtype=torch.int64)
    head_ids = torch.arange(kv_heads, dtype=torch.int64, device=dev)
    src_row = (seq_ids64 * kv_heads).unsqueeze(1) * rows_per_req + q_off64.unsqueeze(1)
    src_row = src_row + head_ids.unsqueeze(0) * rows_per_req  # [num_tiles, kv_heads]
    row_ar = torch.arange(br, dtype=torch.int64, device=dev)
    src = src_row.unsqueeze(2) + row_ar  # [num_tiles, kv_heads, br]
    valid = (
        row_ar.view(1, 1, br) < q_len64.view(num_tiles, 1, 1)
    ).expand(num_tiles, kv_heads, br)  # [num_tiles, kv_heads, br]

    total_rows = q_flat.numel() // max(dim, 1)
    # Rows past a request's live count (and every row of a padded tile) may address beyond
    # ``q_flat``; clamp so the gather stays in bounds and mask them to zero right after.
    src_flat = src.clamp(0, max(total_rows - 1, 0)).reshape(-1)
    valid_flat = valid.reshape(-1)
    gathered = q_flat.view(-1, dim).index_select(0, src_flat).reshape(num_slots, br, dim)
    # Padded rows read an arbitrary neighbour row; mask them back to zero (the kernel's RowThr
    # already masks them, this keeps the buffer contract "padding is zero").
    q_packed = gathered.masked_fill(~valid_flat.view(num_slots, br, 1), 0).contiguous()
    out_packed = torch.empty_like(q_packed)

    indices = meta.kv_indices.to(torch.int32)
    # Declared ABI capacity (covers the real pool; see declared_pool_cap): kv_indices padding,
    # the compiled variant and the pool views all use it, so the kernel key is pool-agnostic.
    cap = declared_pool_cap(int(k_cache.shape[0]))
    if indices.numel() < cap:
        indices = torch.cat([indices, indices.new_zeros(cap - indices.numel())])
    # P3 canonical-variant batch padding: grow the per-request metadata to ``kernel_batch`` rows
    # with trash rows (0 pages, last_page_len = a full page so the derived kv_len is exactly 0).
    # A padded tile references only the trash row, whose page loop runs zero iterations.
    kb = batch if kernel_batch is None else int(kernel_batch)
    if kb < batch:
        raise ValueError(f"kernel_batch={kb} is smaller than the real batch={batch}")
    if kb > batch:
        pad = kb - batch
        indptr = torch.cat(
            [meta.kv_indptr.to(torch.int32), meta.kv_indptr[-1:].to(torch.int32).expand(pad)]
        ).contiguous()
        lpl = torch.cat(
            [
                meta.kv_last_page_len.to(torch.int32),
                torch.full((pad,), page_size, dtype=torch.int32, device=q.device),
            ]
        ).contiguous()
    else:
        indptr = meta.kv_indptr.to(torch.int32)
        lpl = meta.kv_last_page_len.to(torch.int32)
    # causal shift per request: kv_len - qo_len.  Zero for a plain prefill, > 0 for append /
    # chunked prefill, which is what makes row 0 of a chunk see the cached prefix.  Derived from
    # the *padded* indptr/lpl so it covers the trash rows too (their kv_len is exactly 0).
    qo_lens = qo_lens.to(torch.int32)
    if qo_lens.numel() < kb:
        qo_lens = torch.cat([qo_lens, qo_lens.new_zeros(kb - qo_lens.numel())])
    pages = indptr[1:] - indptr[:-1]
    kv_lens = ((pages - 1) * page_size + lpl).to(torch.int32)
    causal_shift = (kv_lens - qo_lens).contiguous()

    # Per-row causal threshold, one entry per (tile, M row).  Global row ``q_off + r`` carries query
    # position ``(q_off + r) // group`` and may attend to ``kv_pos <= q_pos + shift``; rows past the
    # live count get -1 so they mask everything.  The kernel ``dual_copy``s this exactly like
    # ``s_ub``, so each AIV half receives its own half of the tile -- the mask is correct on both
    # without the kernel ever needing the AIV index (the old block_q//2 host restriction is gone).
    row_thr = torch.full((num_tiles * br,), -1, dtype=torch.int32, device=q.device)
    if num_tiles:
        ar = torch.arange(br, dtype=torch.int64, device=q.device).unsqueeze(0)
        q_off64 = schedule.q_offsets.to(torch.int64).unsqueeze(1)
        rows64 = schedule.q_lens.to(torch.int64).unsqueeze(1)
        shift = causal_shift.to(torch.int64)[schedule.seq_ids.to(torch.int64)].unsqueeze(1)
        thr = ((q_off64 + ar) // group) + shift
        row_thr = torch.where(ar < rows64, thr, torch.full_like(thr, -1)).reshape(-1).to(torch.int32)

    # Per-tile page count (KV-rescan truncation): the tile's causal frontier -- the largest row
    # threshold, i.e. the last live row's query position plus the append shift -- is the furthest
    # KV position any of its rows may attend to, so its page loop only needs the pages covering
    # ``[0, frontier]``.  A padded / empty tile (no live rows) gets 0 and runs zero iterations.
    # The kernel clamps this to the request's real page count, and the per-page ``limit`` plus the
    # row thresholds keep the within-page frontier masked, so skipping the trailing pages is
    # numerically identical to scanning them (they were masked to -1e30 before exp).
    tile_page_lens = torch.zeros((num_tiles,), dtype=torch.int32, device=q.device)
    if num_tiles:
        last_row = schedule.q_offsets.to(torch.int64) + schedule.q_lens.to(torch.int64) - 1
        frontier = (last_row // group) + causal_shift.to(torch.int64)[
            schedule.seq_ids.to(torch.int64)
        ]
        needed = frontier // page_size + 1
        tile_page_lens = torch.where(
            schedule.q_lens.to(torch.int64) > 0, needed, torch.zeros_like(needed)
        ).clamp(min=0).to(torch.int32)
        if os.environ.get("TILEINFER_PREFILL_NOSCAN_OFF"):
            # Debug-only: disable the truncation (scan every page) so the pre-truncation baseline can
            # be reproduced back to back.  Off by default; never set in serving.
            tile_page_lens = torch.full_like(tile_page_lens, 1 << 30)

    kernel = build_prefill_kernel(
        batch=kb,
        kv_heads=kv_heads,
        group=group,
        dim=dim,
        page_size=page_size,
        num_pages_cap=cap,
        num_tiles=num_tiles,
        block_q=br,
        dtype=dtype,
        pv_single_pass=pv_single_pass,
    )
    kernel(
        q_packed,
        pool_view(k_cache, cap),
        pool_view(v_cache, cap),
        indptr,
        indices,
        lpl,
        schedule.seq_ids.to(torch.int32),
        schedule.q_offsets.to(torch.int32),
        schedule.q_lens.to(torch.int32),
        causal_shift,
        row_thr,
        tile_page_lens,
        out_packed,
    )

    # Unpack is the exact inverse of the pack above: one masked scatter per prefill call instead
    # of ``nonzero`` + per-tile ``.item()`` + one slice launch per (tile, kv_head).
    #
    # ``out_flat`` must start zeroed: rows past a request's ``q_len`` (and all rows of padded
    # requests) are never written by the scatter below, and the caller treats unwritten rows as
    # padding.  Allocating with ``zeros_like`` is the cheapest way to guarantee that contract.
    out_flat = torch.zeros_like(q_flat)
    flat_out = out_packed.reshape(num_slots, br, dim)
    dst_flat = out_flat.view(-1, dim)
    dst_flat.index_copy_(
        0,
        src_flat[valid_flat],
        flat_out.reshape(-1, dim)[valid_flat],
    )

    # ``out_flat`` is [batch, kv_heads, max_qo_len*group, dim] with (q_pos, head) packed per KV
    # head; un-permute to [batch, qo_heads, max_qo_len, dim].  The permute is a strided view, so
    # the final ``contiguous()`` is a real copy -- but it writes straight into the caller's
    # ``out`` buffer when one is supplied, avoiding a second full-size temporary.
    result = out_flat.view(batch, kv_heads, max_qo_len, group, dim).permute(0, 1, 3, 2, 4)
    if out is not None:
        out.copy_(
            result.reshape(batch, num_qo_heads, max_qo_len, dim)
        )
        return out
    return result.reshape(batch, num_qo_heads, max_qo_len, dim).contiguous()


def plan_prefill_split_ctx(
    qo_tokens_host: List[int],
    pages_host: List[int],
    lpl_host: List[int],
    *,
    group: int,
    kv_tile_pages: int,
    kernel_batch: int,
    kernel_tiles: int,
    kv_heads: int,
    block_q: int,
    max_qo_tokens: int,
    page_size: int,
    indices,
    device,
):
    """Build the split-prefill step context entirely on the host (H19/H20, zero device syncs).

    Tile enumeration is the H20 two-kernel layout: every (request, q-block) is partitioned by
    :func:`tileinfer.plan.frontier_page_splits` into fully-visible **head splits** (no per-row
    causal predicate needed — the tilelang SimtVF multi-block mask bug's trigger surface) and
    one **frontier split** (plain-kernel ``RowThr`` form, multi-block-verified).  All head
    tiles enumerate first (request/q-block order), then all frontier tiles, so the runner can
    slice the tile axis in two contiguous halves and launch one kernel per half; the merge
    reads the dense ``(seq * ms + split)`` slots exactly as before.  Derived per step:

    * ``schedule`` — the :class:`TileSchedule`, padded to ``kernel_tiles`` dead tiles pointing
      at the trash row ``kernel_batch - 1`` (P3 canonical-variant quantization);
    * ``row_thr`` — **frontier tiles only**, relativised to each tile's page offset
      (``thr − kv_start * page_size``) so the kernel's predicate is syntactically identical
      to the plain kernel's (the head kernel takes no ``RowThr`` input; E50/D52);
    * ``tile_page_lens`` / ``causal_shift`` / ``indptr`` / ``lpl`` / ``indices`` — the kernels'
      truncation/page-table inputs (``indices`` padded to the declared capacity here);
    * ``pack_idx`` — flat gather index materialising ``q_packed`` from ``q_flat`` with one
      ``index_select`` (head rows first, frontier rows after — matching the tile halves);
    * ``gather_idx`` / ``scatter_idx`` — the merge's slot gather (per q-block × split; missing
      entries point at a dedicated zero-weight trash slot) and the unpack scatter into
      ``out_flat``.

    Split counts may differ per q-block (``P_vis`` moves with the block's first row);
    ``ms_compile`` is the power-of-two ceiling and the slot layout's ``max_splits`` — unused
    splits gather the trash slot, which carries ``(out=0, lse=-1e30)`` and merges at exactly
    zero weight.
    """
    from tileinfer.plan import TileSchedule, frontier_page_splits

    BR = block_q
    # ---- enumerate the per-(request, q-block) structures once (host ints) ----
    blocks = []  # (b, q_start, rows, splits, thr_rows)
    for b, qo_tokens in enumerate(qo_tokens_host):
        q_rows = qo_tokens * group
        pages = pages_host[b] if b < len(pages_host) else 0
        if q_rows <= 0:
            continue
        lpl = lpl_host[b] if b < len(lpl_host) else page_size
        kv_tokens = (pages - 1) * page_size + lpl if pages > 0 else 0
        shift = kv_tokens - qo_tokens
        for q_start in range(0, q_rows, BR):
            rows = min(BR, q_rows - q_start)
            if kv_tile_pages > 0:
                splits = frontier_page_splits(
                    q_start,
                    qo_tokens,
                    pages,
                    lpl,
                    group=group,
                    page_size=page_size,
                    kv_tile_pages=kv_tile_pages,
                    block_q=BR,
                )
            else:
                # ktp=0 degenerate layout (one frontier-form tile per q-block): same
                # thin-window truncation as the primitive — pages past the block's highest
                # live-row threshold are zero-weight for every row.
                live = min(BR, q_rows - q_start)
                pages_needed = ((q_start + live - 1) // group + shift) // page_size + 1
                splits = [(0, min(pages, max(1, pages_needed)), True)]
            thr_rows = [((q_start + i) // group + shift) if i < rows else -1 for i in range(BR)]
            blocks.append((b, q_start, rows, splits, thr_rows))
    ms_compile = 1
    for _b, _q, _r, splits, _t in blocks:
        while ms_compile < len(splits):
            ms_compile *= 2

    num_qblocks = 0
    head_tiles = []  # (b, q_start, rows, kv_start, kv_len, split_id, slot_id)
    frontier_tiles = []  # (b, q_start, rows, kv_start, kv_len, split_id, thr_rows, slot_id)
    pack_h: List[int] = []
    pack_f: List[int] = []
    gather: List[int] = []
    scatter: List[int] = []
    live_pairs = sum(len(splits) for _b, _q, _r, splits, _t in blocks)
    # Dense per-(q-block, split) slot ids (iter-047 fix): the previous (request * ms + split)
    # layout had **no q-block axis**, so every q-block of a request wrote its split-<sid>
    # partial into the SAME slot — the last-written tile won and the merge combined one
    # block's partials into every block (device-confirmed: all frontier slots byte-identical
    # to the last q-block's).  Decode never saw this (one q-block per request); prefill split
    # has one partial per (q-block, split), so slots are allocated densely in enumeration
    # order and carried per tile (the tilelang ``tile_slots`` input, same as the decode
    # path's ``build_tile_slots``).  Capacity covers the pow2 tile quantisation; slot
    # ``kernel_slots`` itself is the dedicated trash slot (dead tiles write it; it is
    # pre-filled (0, -1e30) and never gathered for a live split).
    kernel_slots = max(kernel_tiles, live_pairs)
    trash_part = kernel_slots * kv_heads  # flat row of the dedicated trash slot
    slot_counter = 0
    for b, q_start, rows, splits, thr_rows in blocks:
        num_qblocks += 1
        for split_id, (kv_start, kv_len, is_frontier) in enumerate(splits):
            slot_id = slot_counter
            slot_counter += 1
            if is_frontier:
                frontier_tiles.append((b, q_start, rows, kv_start, kv_len, split_id, thr_rows, slot_id))
            else:
                head_tiles.append((b, q_start, rows, kv_start, kv_len, split_id, slot_id))
            for bh in range(kv_heads):
                gather.append(slot_id * kv_heads + bh)
                dst = pack_f if is_frontier else pack_h
                for i in range(BR):
                    dst.append(
                        (b * kv_heads + bh) * (max_qo_tokens * group)
                        + (q_start + i if i < rows else 0)
                    )
        # unpack scatter: source rows are merged.view(-1, dim), i.e. (g, bh, i)-major with
        # ``i`` fastest within a head — keep the exact same order.  index_copy_ needs one
        # index per source row, so padding rows point at out_flat's dedicated zero dummy
        # row (allocated by the runner, never read: it sits past every live output row).
        for bh in range(kv_heads):
            for i in range(BR):
                if i < rows:
                    scatter.append(
                        (b * kv_heads + bh) * (max_qo_tokens * group) + q_start + i
                    )
                else:
                    scatter.append(-1)  # resolved to the dummy row by the runner
        # splits this q-block does not have (ragged split counts under one power-of-two
        # ``ms_compile``) gather the trash slot: (out=0, lse=-1e30) merges at zero weight
        for _s in range(len(splits), ms_compile):
            for _bh in range(kv_heads):
                gather.append(trash_part)

    # ---- head-half schedule/pack arrays (live tiles first, then dead padding at the tail
    # of the frontier half) ----
    seq_h = [t[0] for t in head_tiles]
    qoff_h = [t[1] for t in head_tiles]
    qrows_h = [t[2] for t in head_tiles]
    kvs_h = [t[3] for t in head_tiles]
    kvl_h = [t[4] for t in head_tiles]
    sid_h = [t[5] for t in head_tiles]
    slot_h = [t[6] for t in head_tiles]
    pack = list(pack_h)
    num_head_tiles = len(head_tiles)
    # ---- frontier-half arrays: RowThr lives only here ----
    seq_f = [t[0] for t in frontier_tiles]
    qoff_f = [t[1] for t in frontier_tiles]
    qrows_f = [t[2] for t in frontier_tiles]
    kvs_f = [t[3] for t in frontier_tiles]
    kvl_f = [t[4] for t in frontier_tiles]
    sid_f = [t[5] for t in frontier_tiles]
    slot_f = [t[7] for t in frontier_tiles]

    # Dead (padding) tiles: zero page iterations, thr = -1, pack rows point at real memory.
    # Each half is padded to its own width so the two launch slices have *stable, quantised*
    # sizes: the integration passes kernel_tiles = next_pow2(live), giving
    # ``[live head | dead head → kernel_tiles | live frontier | dead frontier → kernel_tiles]``.
    # The runtime then always launches ``(num_tiles=kernel_tiles, num_slots=kernel_tiles+1)``
    # for both kernels — exactly the keys the prewarm ladder compiles (the D56-era A/B showed
    # per-step exact counts racing a data-dependent compile on the serving path).  Dead tiles
    # write the dedicated trash slot.  The max() keeps the CPU tests' smaller kernel_tiles
    # well-formed (widths collapse to the live counts).
    dead_head = max(0, kernel_tiles - num_head_tiles) if num_head_tiles > 0 else 0
    dead_frontier = max(0, kernel_tiles - len(frontier_tiles)) if frontier_tiles else 0
    seq_h.extend([kernel_batch - 1] * dead_head)
    qoff_h.extend([0] * dead_head)
    qrows_h.extend([0] * dead_head)
    kvs_h.extend([0] * dead_head)
    kvl_h.extend([0] * dead_head)
    sid_h.extend([0] * dead_head)
    slot_h.extend([kernel_slots] * dead_head)
    seq_f.extend([kernel_batch - 1] * dead_frontier)
    qoff_f.extend([0] * dead_frontier)
    qrows_f.extend([0] * dead_frontier)
    kvs_f.extend([0] * dead_frontier)
    kvl_f.extend([0] * dead_frontier)
    sid_f.extend([0] * dead_frontier)
    slot_f.extend([kernel_slots] * dead_frontier)
    head_half = num_head_tiles + dead_head
    # Per-tile axes follow the schedule order: page_lens/pack are full-schedule (both halves),
    # row_thr is frontier-half-local (the head kernel takes no RowThr input).
    page_lens_host = (
        [t[4] for t in head_tiles]
        + [0] * dead_head
        + [t[4] for t in frontier_tiles]
        + [0] * dead_frontier
    )
    thr_host = []
    for t in frontier_tiles:
        # RowThr arrives relativised to the tile's page offset (thr − kv_start*BC): the
        # frontier kernel's predicate keeps the plain kernel's `p * BC + c` kv_pos form
        # (syntactic parity, E50/D52).  Padding rows stay negative → fully masked.
        thr_host.extend(thr - t[3] * page_size for thr in t[6])
    thr_host += [-1] * (BR * dead_frontier)
    pack = (
        list(pack_h)
        + [0] * (dead_head * BR * kv_heads)
        + list(pack_f)
        + [0] * (dead_frontier * BR * kv_heads)
    )

    num_tiles_total = head_half + len(frontier_tiles) + dead_frontier
    schedule = TileSchedule(
        seq_ids=torch.tensor(seq_h + seq_f, dtype=torch.int32, device=device),
        q_offsets=torch.tensor(qoff_h + qoff_f, dtype=torch.int32, device=device),
        q_lens=torch.tensor(qrows_h + qrows_f, dtype=torch.int32, device=device),
        kv_page_starts=torch.tensor(kvs_h + kvs_f, dtype=torch.int32, device=device),
        kv_page_lens=torch.tensor(kvl_h + kvl_f, dtype=torch.int32, device=device),
        split_ids=torch.tensor(sid_h + sid_f, dtype=torch.int32, device=device),
        # split_counts is unused by the split-prefill runner (slots come from split_ids);
        # keep it well-formed for the dataclass anyway.
        split_counts=torch.ones((num_tiles_total,), dtype=torch.int32, device=device),
        order=torch.arange(num_tiles_total, dtype=torch.int32, device=device),
    )

    # P3 canonical-variant batch padding: trash rows (0 pages, full-page lpl) so the derived
    # kv_len is exactly 0; a padded tile only ever references the trash row.
    causal_shift_host = [
        ((pages_host[b] - 1) * page_size + lpl_host[b] - qo_tokens_host[b])
        if b < len(qo_tokens_host) and pages_host[b] > 0
        else 0
        for b in range(kernel_batch)
    ]
    indptr_host: List[int] = [0]
    for b in range(len(qo_tokens_host)):
        indptr_host.append(indptr_host[-1] + pages_host[b])
    while len(indptr_host) < kernel_batch + 1:
        indptr_host.append(indptr_host[-1])
    lpl_padded = [lpl_host[b] if b < len(lpl_host) else page_size for b in range(kernel_batch)]

    cap = int(indices.numel())  # caller passes the already capacity-padded page table
    # out_flat gets one extra all-zero dummy row; scatter entries marked -1 (padding rows of a
    # q-block) are resolved to it so index_copy_ keeps one index per merged source row.
    dummy_row = len(qo_tokens_host) * kv_heads * max_qo_tokens * group
    ctx = {
        "schedule": schedule,
        "ms_compile": ms_compile,
        "kernel_slots": kernel_slots,
        "kernel_batch": kernel_batch,
        # Dense per-tile slot ids (head half first, then frontier, then dead → trash slot):
        # what the two split kernels write through ``tile_slots`` and the merge gathers
        # through ``gather_idx`` (the (request*ms+split) layout had no q-block axis and
        # collided across a request's q-blocks — iter-047 root cause).
        "tile_slots": torch.tensor(slot_h + slot_f, dtype=torch.int32, device=device),
        "num_qblocks": num_qblocks,
        "num_tiles": schedule.num_tiles,
        # Launch slice widths (each half INCLUDING its dead padding): the two kernels are
        # always built/launched with exactly these tile counts, so the compiled-variant keys
        # are data-independent (quantised by the caller's kernel_tiles).
        "num_head_tiles": head_half,
        "num_frontier_tiles": schedule.num_tiles - head_half,
        "kv_heads": kv_heads,
        "block_q": BR,
        "tile_page_lens": torch.tensor(page_lens_host, dtype=torch.int32, device=device),
        "row_thr": torch.tensor(thr_host, dtype=torch.int32, device=device),
        "causal_shift": torch.tensor(causal_shift_host, dtype=torch.int32, device=device),
        "indptr": torch.tensor(indptr_host, dtype=torch.int32, device=device),
        "lpl": torch.tensor(lpl_padded, dtype=torch.int32, device=device),
        "indices": indices,
        "pack_idx": torch.tensor(pack, dtype=torch.int64, device=device),
        "gather_idx": (
            torch.tensor(gather, dtype=torch.int64, device=device).view(
                num_qblocks, ms_compile, kv_heads
            )
            if num_qblocks
            else torch.zeros((0, ms_compile, kv_heads), dtype=torch.int64, device=device)
        ),
        "scatter_idx": torch.tensor(
            [dummy_row if s < 0 else s for s in scatter], dtype=torch.int64, device=device
        ),
        "dummy_row": dummy_row,
        "cap": cap,
    }
    return ctx


def _merge_prefill_partials(
    part_out: torch.Tensor,
    part_lse: torch.Tensor,
    ctx: dict,
    *,
    dim: int,
    o_dtype: torch.dtype,
) -> torch.Tensor:
    """Merge split partials into per-q-block outputs (the torch half of :func:`forward_prefill_split`).

    Reads normalized partials plus per-slot LSE (the decode split kernel's exact contract):
    ``merged[g, bh] = Σ_s softmax_s(lse) ⊙ partial_s``.  Slots a request does not use gather the
    trash slot ``(0, -1e30)`` and contribute exactly zero.  fp32 math; chunked over q-blocks to
    keep the fp32 partial upcast bounded.  Returns ``[num_qblocks, kv_heads, block_q, dim]`` in
    ``o_dtype``.

    Exposed at module level so tests can drive it with synthetic partials (CPU).
    """
    g = ctx["num_qblocks"]
    kv_heads = ctx["kv_heads"]
    br = ctx["block_q"]
    ms = ctx["ms_compile"]
    merged = torch.empty((g, kv_heads, br, dim), dtype=o_dtype, device=part_out.device)
    if not g:
        return merged
    lse_rows = part_lse.view(-1, br)
    lse_parts = []
    for s in range(ms):
        idx_s = ctx["gather_idx"][:, s, :].reshape(-1)
        lse_parts.append(lse_rows.index_select(0, idx_s))
    lse_stack = torch.stack(lse_parts, dim=1).view(g, kv_heads, ms, br)
    m = lse_stack.max(dim=2, keepdim=True).values
    lse_all = m.squeeze(2) + torch.log(torch.exp(lse_stack - m).sum(dim=2))  # [g, kvh, br]
    # Explicit normalization (the decode merge kernel's ``/norm``): fp32 collapses ``lse_all``
    # to the shared value when every split carries the -1e30 filler (all-masked rows), and the
    # raw weights would come out as 1 each; their partials are exactly 0, but normalize anyway
    # so the contract (weights sum to 1) holds bit-for-bit in every case.
    w = torch.exp(lse_stack - lse_all.unsqueeze(2))
    w = w / w.sum(dim=2, keepdim=True)
    CH = 64
    for g0 in range(0, g, CH):
        g1 = min(g0 + CH, g)
        acc = None
        for s in range(ms):
            idx_gs = ctx["gather_idx"][g0:g1, s, :].reshape(-1)  # [(g1-g0)*kvh]
            outs = part_out.index_select(0, idx_gs).view(g1 - g0, kv_heads, br, dim)
            term = w[g0:g1, :, s, :].unsqueeze(-1) * outs.to(torch.float32)
            acc = term if acc is None else acc + term
        merged[g0:g1] = acc.to(o_dtype)
    return merged


def forward_prefill_split(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    ctx: dict,
    *,
    group: int,
    dtype: str | None = None,
) -> torch.Tensor:
    """Run split-KV prefill from a prebuilt :func:`plan_prefill_split_ctx` (H19).

    ``q`` is ``[batch, kv_heads * group, max_qo, dim]`` (the integration's padded staging).  The
    step runs as: one ``index_select`` pack → the split kernel (partial, per-slot outputs) → a
    log-sum-exp weighted merge of the splits (plain tensor ops; the prefill path is eager, so no
    capture constraints) → one ``index_copy_`` unpack.  Every index was built once per step on
    the host, so the per-layer cost is a dozen device ops regardless of the split factor.

    The merge reads normalized partials plus per-slot LSE (the decode split kernel's exact
    contract): ``merged = Σ_s softmax_s(lse) ⊙ partial_s``.  Slots a request does not use gather
    the trash slot ``(0, -1e30)`` and contribute exactly zero.
    """
    import torch

    batch, num_qo_heads, max_qo, dim = q.shape
    kv_heads = k_cache.shape[2]
    page_size = int(k_cache.shape[1])
    dtype = dtype or _TORCH_TO_TILELANG_DTYPE[q.dtype]
    br = ctx["block_q"]
    schedule = ctx["schedule"]
    num_tiles = ctx["num_tiles"]
    ms = ctx["ms_compile"]
    kernel_slots = ctx["kernel_slots"]
    num_head_tiles = ctx["num_head_tiles"]
    assert num_qo_heads == kv_heads * group

    # [batch, kv_heads, group, q_pos, dim] -> [batch, kv_heads, q_pos * group, dim]
    q_flat = (
        q.view(batch, kv_heads, group, max_qo, dim)
        .permute(0, 1, 3, 2, 4)
        .reshape(batch, kv_heads, max_qo * group, dim)
        .contiguous()
    )
    q_packed = q_flat.view(-1, dim).index_select(0, ctx["pack_idx"]).view(
        num_tiles * kv_heads, br, dim
    )

    # Partial buffers + the dedicated trash slot (out=0, lse=-1e30): merge rows a request does
    # not use gather it, dead padding tiles write it (their per-tile slot is kernel_slots).
    part_out = torch.empty(
        ((kernel_slots + 1) * kv_heads, br, dim), dtype=q.dtype, device=q.device
    )
    part_lse = torch.empty(((kernel_slots + 1) * kv_heads * br,), dtype=torch.float32, device=q.device)
    part_out[-kv_heads:].zero_()
    part_lse[-kv_heads * br :].fill_(-1e30)

    # H20 two-kernel launch: the schedule/pack layout puts every head tile before every
    # frontier tile, so the two halves slice off the same tensors contiguously.  Each tile
    # writes its dense slot (ctx["tile_slots"]); the merge gathers the same ids.
    head_rows = num_head_tiles * kv_heads
    frontier_rows = (num_tiles - num_head_tiles) * kv_heads
    slots = ctx["tile_slots"]
    part_out_view = part_out.narrow(0, 0, (kernel_slots + 1) * kv_heads)
    part_lse_view = part_lse.narrow(0, 0, (kernel_slots + 1) * kv_heads * br)
    head_spec = dict(
        # the launch passes the *padded* indptr/lpl (kernel_batch rows), so the kernel's
        # batch declaration must be kernel_batch — not q.shape[0] (the padding rows are
        # trash: seq_ids point at row kernel_batch-1 whose indptr repeats the last entry).
        batch=ctx["kernel_batch"],
        kv_heads=kv_heads,
        group=group,
        dim=dim,
        page_size=page_size,
        num_pages_cap=ctx["cap"],
        max_splits=SPLIT_MS_COMPILE,
        num_slots=kernel_slots + 1,
        block_q=br,
        dtype=dtype,
    )
    if head_rows > 0:
        k_head = build_prefill_split_head_kernel(num_tiles=num_head_tiles, **head_spec)
        k_head(
            q_packed.narrow(0, 0, head_rows),
            pool_view(k_cache, ctx["cap"]),
            pool_view(v_cache, ctx["cap"]),
            ctx["indptr"],
            ctx["indices"],
            ctx["lpl"],
            schedule.seq_ids.to(torch.int32).narrow(0, 0, num_head_tiles),
            # H20 head kernel has no q_offsets input (rows are pre-packed host-side); the
            # 8th slot is ``tile_rows`` (live-row count, unused by the body but kept for
            # signature parity with the plain kernel) — feed it the per-tile q_lens.
            schedule.q_lens.to(torch.int32).narrow(0, 0, num_head_tiles),
            schedule.kv_page_starts.to(torch.int32).narrow(0, 0, num_head_tiles),
            ctx["tile_page_lens"].narrow(0, 0, num_head_tiles),
            slots.narrow(0, 0, num_head_tiles),
            part_out_view,
            part_lse_view,
        )
    if frontier_rows > 0:
        k_frontier = build_prefill_split_frontier_kernel(
            num_tiles=num_tiles - num_head_tiles, **head_spec
        )
        k_frontier(
            q_packed.narrow(0, head_rows, frontier_rows),
            pool_view(k_cache, ctx["cap"]),
            pool_view(v_cache, ctx["cap"]),
            ctx["indptr"],
            ctx["indices"],
            ctx["lpl"],
            schedule.seq_ids.to(torch.int32).narrow(0, num_head_tiles, num_tiles - num_head_tiles),
            schedule.q_lens.to(torch.int32).narrow(0, num_head_tiles, num_tiles - num_head_tiles),
            schedule.kv_page_starts.to(torch.int32).narrow(0, num_head_tiles, num_tiles - num_head_tiles),
            ctx["causal_shift"],
            ctx["row_thr"],
            ctx["tile_page_lens"].narrow(0, num_head_tiles, num_tiles - num_head_tiles),
            slots.narrow(0, num_head_tiles, num_tiles - num_head_tiles),
            part_out_view,
            part_lse_view,
        )

    # Merge: log-sum-exp weights over the split axis (see _merge_prefill_partials), then one
    # scatter unpack into the flattened output rows.  The extra dummy row (all zeros, resolved
    # from the ctx's -1 markers) absorbs the padding rows of a q-block; it is never read.
    out_flat = torch.zeros(
        (batch * kv_heads * max_qo * group + 1, dim), dtype=q.dtype, device=q.device
    )
    if ctx["num_qblocks"]:
        merged = _merge_prefill_partials(part_out, part_lse, ctx, dim=dim, o_dtype=q.dtype)
        out_flat.view(-1, dim).index_copy_(0, ctx["scatter_idx"], merged.view(-1, dim))

    result = (
        out_flat[: batch * kv_heads * max_qo * group]
        .view(batch, kv_heads, max_qo, group, dim)
        .permute(0, 1, 3, 2, 4)
        .reshape(batch, num_qo_heads, max_qo, dim)
        .contiguous()
    )
    return result


def padded_rows(group: int) -> int:
    """M-tile size used for a given GQA group (see the padding note in the module docstring)."""
    return max(32, ((group + 31) // 32) * 32)


# Fixed compile-time ``max_splits`` for the split-prefill kernels: inert in the kernel body
# (the explicit ``num_slots`` carries the real capacity — ``max_splits`` only feeds the
# default), it exists purely to keep the compiled-variant keys data-independent.  Deriving it
# per step as next_pow2(actual splits) made every first-seen context race a ~2.5 min compile
# against the serving path (the D56-era engine A/B: 4400/4400 prefill steps fell back to FIA).
SPLIT_MS_COMPILE = max(1, int(os.environ.get("TILEINFER_PREFILL_SPLIT_MS", "16") or 16))

_TORCH_TO_TILELANG_DTYPE = {
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
    torch.float32: "float32",
}


def declared_pool_cap(pool_pages: int) -> int:
    """The kernel ABI's *declared* page capacity for a real KV pool (L2, pool-agnostic ABI).

    Pages outside ``kv_indptr``/``kv_indices`` spans are never dereferenced, so the declared
    capacity of ``KCache``/``VCache``/``kv_indices`` only has to *cover* the real pool — its
    exact value is free.  Rounding up to a power of two (or pinning it with
    ``TILEINFER_POOL_CAP``) removes the pool size from the kernel/plan/pre-warm keys: vLLM
    derives the pool from a memory snapshot, so it drifts run to run (9425 vs 9899 pages) and
    used to silently invalidate an out-of-process pre-warm.  ``cap >= pool`` always holds; the
    caller presents the pool through :func:`pool_view`, which is a zero-copy capacity view.
    """
    pool_pages = int(pool_pages)
    pinned = os.environ.get("TILEINFER_POOL_CAP", "").strip()
    if pinned:
        cap = int(pinned)
        if cap < pool_pages:
            raise ValueError(
                f"TILEINFER_POOL_CAP={cap} is below the real pool ({pool_pages} pages); "
                "the declared capacity must cover every live page"
            )
        return cap
    return 1 if pool_pages <= 1 else 1 << (pool_pages - 1).bit_length()


def pool_view(k_cache: torch.Tensor, cap: int) -> torch.Tensor:
    """Present the KV pool with the kernel's declared page capacity, without copying.

    L2 ABI: kernels are compiled for the *declared* capacity (see :func:`declared_pool_cap`),
    which covers but does not equal the real pool, so one compiled artifact serves every pool in
    its bucket and the pool size stays out of the kernel/plan/pre-warm keys.  The FFI binder
    wants a runtime tensor of the declared shape, and ``as_strided`` refuses to create a view
    whose shape exceeds the storage — but ``Tensor.set_`` does not validate, so the capacity view
    is built by re-binding the pool's own storage at the declared extent (same base pointer, same
    strides, contiguous).  Pages beyond the real pool are never dereferenced: every access is
    bounded by ``kv_indptr``/``kv_indices`` (the page-table compaction parks invalid columns in a
    trash slot and the spans only ever name real pages), and the pool *must not* be padded — the
    engine writes new KV into it every step.

    ``cap < pool`` would silently hide live pages and is a caller bug (asserted).
    """
    pool = int(k_cache.shape[0])
    if pool == cap:
        return k_cache.reshape(-1, *k_cache.shape[1:])
    if pool > cap:
        raise ValueError(
            f"pool grew past the kernel's compiled capacity ({pool} > {cap}); "
            "the declared capacity must cover every live page"
        )
    flat = k_cache if k_cache.is_contiguous() else k_cache.contiguous()
    view = torch.empty(0, dtype=flat.dtype, device=flat.device)
    view.set_(flat.untyped_storage(), 0, (cap, *flat.shape[1:]), flat.stride())
    return view
    pool = int(k_cache.shape[0])
    if pool == cap:
        return k_cache.reshape(-1, *k_cache.shape[1:])
    assert pool < cap, f"pool grew past the kernel's compiled capacity ({pool} > {cap})"
    flat = k_cache if k_cache.is_contiguous() else k_cache.contiguous()
    return flat.as_strided((cap, *flat.shape[1:]), flat.stride())


def forward(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    meta,
    *,
    group: int,
    dtype: str | None = None,
    kv_indptr: torch.Tensor | None = None,
    kv_indices: torch.Tensor | None = None,
    kv_last_page_len: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run paged decode for one step: pads ``q``, calls the kernel, slices the padded rows away.

    ``q``       ``[batch, num_qo_heads, 1, dim]`` (the engine's layout, head order
                ``kv_head * group + i``)
    ``k_cache`` / ``v_cache``  ``[num_pages, page_size, num_kv_heads, dim]``
    Returns ``[batch, num_qo_heads, 1, dim]``.

    Kept deliberately free of policy (no planning, no workspace): the backend owns that.
    """
    import torch

    batch, num_qo_heads, qo_len, dim = q.shape
    assert qo_len == 1, "paged decode handles one query row per request"
    kv_heads = k_cache.shape[2]
    assert k_cache.shape == v_cache.shape
    assert num_qo_heads == kv_heads * group

    br = padded_rows(group)
    page_size = int(k_cache.shape[1])
    # Derive the kernel dtype from the tensors: a wrapper that silently disagrees with its inputs
    # is a debugging session nobody wants.
    dtype = dtype or _TORCH_TO_TILELANG_DTYPE[q.dtype]
    # Declared ABI capacity (covers the real pool; see declared_pool_cap): sizes the kv_indices
    # slot the kernel reads page ids from and the K/V page extents (capacity views).
    cap = declared_pool_cap(int(k_cache.shape[0]))
    kernel = build_decode_kernel(
        batch=batch,
        kv_heads=kv_heads,
        group=group,
        dim=dim,
        page_size=page_size,
        num_pages_cap=cap,
        dtype=dtype,
    )

    # pad: zero rows so the padded half of the tile cannot produce NaN
    q_pad = q.new_zeros((batch, kv_heads * br, dim))
    q_pad.view(batch, kv_heads, br, dim)[:, :, :group] = q[:, :, 0, :].view(batch, kv_heads, group, dim)

    k_flat = pool_view(k_cache, cap)
    v_flat = pool_view(v_cache, cap)

    # The kernel is compiled for the declared capacity, which covers but does not equal the
    # real pool, while one step usually references a handful of pages still.  The ABI wants the
    # declared length, and the unused tail is never dereferenced (every access is bounded by
    # kv_indptr), so pad rather than recompile.
    # Prefer the plan-owned metadata buffers (stable addresses, caller tensors untouched).
    if kv_indices is not None:
        indices = kv_indices
        indptr = kv_indptr
        last_page_len = kv_last_page_len
    else:
        indices = meta.kv_indices.to(torch.int32)
        if indices.numel() < cap:
            indices = torch.cat([indices, indices.new_zeros(cap - indices.numel())])
        indptr = meta.kv_indptr.to(torch.int32)
        last_page_len = meta.kv_last_page_len.to(torch.int32)

    out_pad = kernel(q_pad, k_flat, v_flat, indptr, indices, last_page_len)
    result = out_pad.view(batch, kv_heads, br, dim)[:, :, :group].reshape(batch, num_qo_heads, dim)
    result = result.unsqueeze(2)
    if out is not None:
        out.copy_(result)
        return out
    return result
