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

import torch
import tilelang
import tilelang.ascend.language as T

__all__ = ["paged_decode_ascend950", "build_decode_kernel", "padded_rows", "forward"]


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

    Layouts (chosen to match what serving engines already allocate, see ``docs/architecture.md``):

    ``Q``        ``[batch, kv_heads * BR, dim]`` — **padded**: query head ``kv_head * group + i``
                 belongs to KV head ``kv_head``, and rows ``group..BR-1`` of each KV head must be
                 zero (the caller pads and slices).
    ``Out``      same padded shape; only the first ``group`` rows of each KV head are meaningful.
    ``KCache`` / ``VCache``  ``[num_pages, page_size, kv_heads, dim]`` (NHD).
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
    assert dim == 128, "the Ascend 950 attention path is tuned for head_dim == 128"
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

            qk_a = T.alloc_l0a((BR, D), dtype)
            qk_b = T.alloc_l0b((BC, D), dtype)
            pv_a = T.alloc_l0a((BR, BC), dtype)
            pv_b = T.alloc_l0b((D, BC), dtype)
            qk_acc = T.alloc_l0c((BR, BC), accum)
            pv_acc = T.alloc_l0c((BR, D), accum)

            s_ub = T.alloc_shared((ROWS, BC), accum)
            p_ub = T.alloc_shared((ROWS, BC), dtype)
            o_ub = T.alloc_shared((ROWS, D), accum)
            o_tmp_ub = T.alloc_shared((ROWS, D), accum)
            out_ub = T.alloc_shared((ROWS, D), dtype)
            m_ub = T.alloc_shared((ROWS,), accum)
            l_ub = T.alloc_shared((ROWS,), accum)

            num_pages = kv_indptr[b + 1] - kv_indptr[b]

            T.copy(Q[b, bh * BR : (bh + 1) * BR, 0:D], q_l1)
            T.fill(m_ub, T.float32(-1e30))
            T.fill(l_ub, T.float32(0))
            T.fill(o_ub, T.float32(0))

            for p in T.serial(num_pages):
                pid = kv_indices[kv_indptr[b] + p]

                # ---- scores = Q @ K^T, then cube -> UB ----
                T.copy(KCache[pid, 0:BC, bh, 0:D], k_l1)
                T.copy(q_l1[0:BR, 0:D], qk_a)
                T.copy(k_l1[0:BC, 0:D], qk_b)
                T.gemm(qk_a, qk_b, qk_acc, transpose_B=True, clear_accum=True)
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
                    for r, d in T.Parallel(ROWS, D):
                        o_ub[r, d] = o_ub[r, d] * alpha[r]
                    for r, c in T.Parallel(ROWS, BC):
                        p_ub[r, c] = T.Cast(dtype, s_frag[r, c])

                # ---- o += P @ V ----
                T.dual_copy(p_ub[0:ROWS, 0:BC], p_l1[0:BR, 0:BC])
                T.copy(VCache[pid, 0:BC, bh, 0:D], v_l1[0:D, 0:BC], transpose=True)
                T.copy(p_l1[0:BR, 0:BC], pv_a)
                T.copy(v_l1[0:D, 0:BC], pv_b)
                T.gemm(pv_a, pv_b, pv_acc, transpose_B=True, clear_accum=True)
                T.dual_copy(pv_acc, o_tmp_ub)
                with T.SimtVF(threads=threads):
                    for r, d in T.Parallel(ROWS, D):
                        o_ub[r, d] = o_ub[r, d] + o_tmp_ub[r, d]

            # ---- normalise and store ----
            with T.SimtVF(threads=threads):
                for r, d in T.Parallel(ROWS, D):
                    out_ub[r, d] = T.Cast(dtype, o_ub[r, d] / l_ub[r])
            T.dual_copy(out_ub[0:ROWS, 0:D], Out[b, bh * BR : (bh + 1) * BR, 0:D])

    return main


_KERNEL_CACHE: dict = {}


def build_decode_kernel(**spec):
    """Compile (and cache) the kernel for one shape combination."""
    key = tuple(sorted(spec.items()))
    kernel = _KERNEL_CACHE.get(key)
    if kernel is None:
        kernel = tilelang.compile(
            paged_decode_ascend950(**spec),
            out_idx=-1,
            pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        )
        _KERNEL_CACHE[key] = kernel
    return kernel


def padded_rows(group: int) -> int:
    """M-tile size used for a given GQA group (see the padding note in the module docstring)."""
    return max(32, ((group + 31) // 32) * 32)


def forward(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    meta,
    *,
    group: int,
    dtype: str = "bfloat16",
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
    kernel = build_decode_kernel(
        batch=batch,
        kv_heads=kv_heads,
        group=group,
        dim=dim,
        page_size=page_size,
        num_pages_cap=int(k_cache.shape[0]),
    )

    # pad: zero rows so the padded half of the tile cannot produce NaN
    q_pad = q.new_zeros((batch, kv_heads * br, dim))
    q_pad.view(batch, kv_heads, br, dim)[:, :, :group] = q[:, :, 0, :].view(batch, kv_heads, group, dim)

    k_flat = k_cache.reshape(-1, page_size, kv_heads, dim)
    v_flat = v_cache.reshape(-1, page_size, kv_heads, dim)

    # The kernel is compiled for the whole pool (`k_cache.shape[0]`), while one step usually
    # references fewer pages.  The ABI wants the declared length, and the unused tail is never
    # dereferenced (every access is bounded by kv_indptr), so pad rather than recompile.
    indices = meta.kv_indices.to(torch.int32)
    cap = int(k_cache.shape[0])
    if indices.numel() < cap:
        indices = torch.cat([indices, indices.new_zeros(cap - indices.numel())])
    elif indices.numel() > cap:
        raise ValueError(
            f"step references {indices.numel()} pages but the caches only hold {cap}; "
            "rebuild the plan (pool growth needs a new plan)"
        )

    out_pad = kernel(
        q_pad,
        k_flat,
        v_flat,
        meta.kv_indptr.to(torch.int32),
        indices,
        meta.kv_last_page_len.to(torch.int32),
    )
    result = out_pad.view(batch, kv_heads, br, dim)[:, :, :group].reshape(batch, num_qo_heads, dim)
    result = result.unsqueeze(2)
    if out is not None:
        out.copy_(result)
        return out
    return result
