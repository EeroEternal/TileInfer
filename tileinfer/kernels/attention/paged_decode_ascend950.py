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
    "build_split_kernel",
    "build_merge_kernel",
    "build_tile_slots",
    "padded_rows",
    "declared_pool_cap",
    "forward",
    "forward_split",
    "forward_prefill",
]


def paged_decode_ascend950(
    batch: int,
    kv_heads: int,
    group: int,
    dim: int,
    page_size: int,
    num_pages_cap: int,
    dtype: str = "bfloat16",
    threads: int = 128,
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
    """
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

            num_pages = kv_indptr[b + 1] - kv_indptr[b]

            T.copy(Q[b, bh * BR : (bh + 1) * BR, 0:D], q_l1)
            T.fill(m_ub, T.float32(-1e30))
            T.fill(l_ub, T.float32(0))
            T.fill(o_ub0, T.float32(0))
            if D > D2:
                T.fill(o_ub1, T.float32(0))

            for p in T.serial(num_pages):
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

                # ---- online softmax (SIMT) ----
                # Only the last page can be partially filled; for every other page all BC columns
                # are real tokens.  Masking before `exp` is what keeps padding out of the
                # denominator.
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


_KERNEL_CACHE: dict = {}


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

            T.copy(Q[b, bh * BR : (bh + 1) * BR, 0:D], q_l1)
            T.fill(m_ub, T.float32(-1e30))
            T.fill(l_ub, T.float32(0))
            T.fill(o_ub0, T.float32(0))
            if D > D2:
                T.fill(o_ub1, T.float32(0))

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
                limit = T.if_then_else(start + p + 1 < req_pages, BC, kv_last_page_len[b])
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


def build_merge_kernel(**spec):
    """Compile (and cache) the split-merge kernel."""
    key = ("merge",) + tuple(sorted(spec.items()))
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
    key = ("split",) + tuple(sorted(spec.items()))
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
    if spec.get("num_pages_cap") is not None:
        spec["num_pages_cap"] = declared_pool_cap(spec["num_pages_cap"])
    key = ("decode", "manual" if out_manual else "auto") + tuple(sorted(spec.items()))
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
            # Per-row causal threshold for *this AIV's* half of the M tile.  `dual_copy` splits it
            # exactly like `s_ub`, so a local row r holds the threshold of global row
            # `sid * ROWS + r` without the kernel ever needing the AIV index.
            thr_ub = T.alloc_shared((ROWS,), "int32")

            T.copy(QPacked[bx, 0:BR, 0:D], q_l1)
            T.dual_copy(RowThr[tile * BR : (tile + 1) * BR], thr_ub)
            T.fill(m_ub, T.float32(-1e30))
            T.fill(l_ub, T.float32(0))
            T.fill(o_ub0, T.float32(0))
            if D > D2:
                T.fill(o_ub1, T.float32(0))

            for p in T.serial(plen):
                pid = kv_indices[kv_indptr[b] + p]

                T.copy(KCache[pid, 0:BC, bh, 0:D], k_l1)
                T.copy(q_l1[0:BR, 0:D2], qk_a)
                T.copy(k_l1[0:BC, 0:D2], qk_b)
                T.gemm(qk_a, qk_b, qk_acc, transpose_B=True, clear_accum=True)
                if D > D2:
                    T.copy(q_l1[0:BR, D2:D], qk_a)
                    T.copy(k_l1[0:BC, D2:D], qk_b)
                    T.gemm(qk_a, qk_b, qk_acc, transpose_B=True)
                T.dual_copy(qk_acc, s_ub)

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
                    out_ub0[r, d] = T.Cast(dtype, o_ub0[r, d] / l_ub[r])
                if D > D2:
                    for r, d in T.Parallel(ROWS, D2):
                        out_ub1[r, d] = T.Cast(dtype, o_ub1[r, d] / l_ub[r])

            T.dual_copy(out_ub0[0:ROWS, 0:D2], OutPacked[bx, 0:BR, 0:D2])
            if D > D2:
                T.dual_copy(out_ub1[0:ROWS, 0:D2], OutPacked[bx, 0:BR, D2:D])

    return main


def build_prefill_kernel(**spec):
    """Compile (and cache) the prefill kernel.  See `build_decode_kernel` re: `num_pages_cap`
    (quantized to the ABI's declared capacity, so the key is pool-agnostic)."""
    if spec.get("num_pages_cap") is not None:
        spec["num_pages_cap"] = declared_pool_cap(spec["num_pages_cap"])
    key = ("prefill",) + tuple(sorted(spec.items()))
    kernel = _KERNEL_CACHE.get(key)
    if kernel is None:
        kernel = tilelang.compile(
            paged_prefill_ascend950(**spec),
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

    q_packed = torch.zeros((num_tiles * kv_heads, br, dim), dtype=q.dtype, device=q.device)
    out_packed = torch.empty_like(q_packed)
    seq_ids_host = schedule.seq_ids.to(torch.int64)
    # P3 padded tiles (q_len == 0, seq_id -> trash row) carry no query rows: skip them here and
    # in the unpack loop below.  Everything the kernel sees stays shape-static.
    live = schedule.q_lens.to(torch.int64) > 0
    for tile in torch.nonzero(live).flatten().tolist():
        b = int(seq_ids_host[tile].item())
        q_off = int(schedule.q_offsets[tile].item())
        rows = int(schedule.q_lens[tile].item())
        for bh in range(kv_heads):
            q_packed[tile * kv_heads + bh, :rows] = q_flat[b, bh, q_off : q_off + rows]

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

    out_flat = torch.zeros_like(q_flat)
    for tile in torch.nonzero(live).flatten().tolist():
        b = int(seq_ids_host[tile].item())
        q_off = int(schedule.q_offsets[tile].item())
        rows = int(schedule.q_lens[tile].item())
        for bh in range(kv_heads):
            out_flat[b, bh, q_off : q_off + rows] = out_packed[tile * kv_heads + bh, :rows]

    result = (
        out_flat.view(batch, kv_heads, max_qo_len, group, dim)
        .permute(0, 1, 3, 2, 4)
        .reshape(batch, num_qo_heads, max_qo_len, dim)
        .contiguous()
    )
    if out is not None:
        out.copy_(result)
        return out
    return result


def padded_rows(group: int) -> int:
    """M-tile size used for a given GQA group (see the padding note in the module docstring)."""
    return max(32, ((group + 31) // 32) * 32)


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
