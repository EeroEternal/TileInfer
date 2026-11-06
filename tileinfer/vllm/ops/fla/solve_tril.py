"""Torch implementation of FLA's ``solve_tril``: the blocked inverse of ``I + A``.

Why this exists
---------------
vLLM 0.23 runs the GDN chunked prefill chain through
``vllm/model_executor/layers/fla/ops/solve_tril.py``, whose only code path is Triton
(``solve_tril_16x16_kernel`` plus the ``merge_16x16_to_32x32_inverse_kernel`` /
``merge_16x16_to_64x64_inverse_kernel`` mergers).  There is no torch fallback, so a backend without
Triton has to supply the operator itself.  This module is that fallback, written for exactness and
readability; the TileInfer kernel can replace it later once a profile says it pays.

Semantics, as established from the source
-----------------------------------------
Source quoted below is ``runtime/vllm-api/fla_solve_tril.py`` (vLLM 0.23 copy of FLA).

1. **What is inverted.**  The host docstring (lines 513-530) states it outright::

       Compute the inverse of the matrix I + A
       A should be strictly lower triangular, i.e., A.triu() == 0.
       ...
       Returns:
           (I + A)^-1 with the same shape as A

   and the kernel agrees.  The diagonal and the upper triangle of the *input* are discarded and the
   result is given a unit diagonal (lines 64-66 define ``m_A`` strictly-lower and ``m_I`` identity)::

       o_i = tl.arange(0, 16)
       m_A = o_i[:, None] > o_i[None, :]          # strictly lower (no diagonal)
       m_I = o_i[:, None] == o_i[None, :]        # identity
       ...
       b_A = -tl.where(m_A, b_A, 0)              # line 82: keep the strict lower part, negated
       b_A += m_I                                # line 89: add I

   The forward substitution in between (lines 84-88) is the textbook unit-triangular solve::

       for i in range(2, min(16, T - i_t * 16)):
           b_a = -tl.load(A + (i_t * 16 + i) * H * BT + o_i + offset)
           b_a = b_a + tl.sum(b_a[:, None] * b_A, 0)
           b_A = tl.where((o_i == i)[:, None], b_a, b_A)

   Rows 0 and 1 come for free from the initialisation (row 0 of a strictly-lower matrix is 0, row 1
   already holds ``-A[1, 0]``), so the loop starts at 2.  Writing ``R_i`` for the final row ``i``
   (``rec_i + e_i``), the body is ``rec_i = -A_i - sum_{k<i} A_ik rec_k``, i.e.
   ``R_i = e_i - sum_{j<i} A_ij R_j`` - exactly ``(I + A)^{-1}`` and *not* ``(I - A)^{-1}``.
   The off-by-one that would flip the sign is a real trap (``docs/notes/vllm-api-0.23.md`` line 103
   records "``(I - A)``"; that note is wrong, the kernels and ``tileinfer/testing/gdn_reference.py``
   line 94 both say ``(I + A)``).

2. **Layout.**  ``A`` is ``[B, T, H, BT]`` (host docstring line 519) and the kernel treats it as a
   flat ``(T, BT)`` matrix per ``(batch, head)`` with row stride ``H * BT`` and column stride 1
   (lines 68-75)::

       A = A + (bos * H + i_h) * BT
       Ai = Ai + (bos * H + i_h) * 16
       offset = (i_t * 16) % BT
       p_A = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * 16, offset), (16, 16), (1, 0))

   So the token axis is the *row* axis (``T`` is a token count) and the last axis is the
   within-chunk column index (``BT`` of them).  The matrix being inverted never spans tokens across
   chunk boundaries: chunk ``c`` of a sequence occupies rows ``[c*BT, (c+1)*BT)`` of the sequence and
   *all* ``BT`` columns.  ``BT`` (last dim) must be 16, 32 or 64 (host assert, line 531).

3. **The unit of work is the whole BT x BT block, not the 16 x 16 diagonal blocks.**  The 16 x 16
   kernel only serves ``BT == 16``; for ``BT == 32/64`` the merge kernels recompute the off-diagonal
   blocks from the raw ``A`` blocks with the block-triangular inverse identity (lines 185-189,
   356-390)::

       b_Ai_21 = -tl.dot(tl.dot(b_Ai_22, b_A_21, input_precision=DOT_PRECISION), b_Ai_11, ...)
       ...
       b_Ai_31 = -tl.dot(b_Ai_33, tl.dot(b_A_31, b_Ai_11, ...) + tl.dot(b_A_32, b_Ai_21, ...), ...)
       b_Ai_41 = -tl.dot(b_Ai_44, tl.dot(b_A_41, b_Ai_11, ...) + tl.dot(b_A_42, b_Ai_21, ...)
                                    + tl.dot(b_A_43, b_Ai_31, ...), ...)

   An implementation that inverts the four 16 x 16 diagonal blocks and stops is wrong by O(1) on any
   real input (the test suite pins this down).

4. **Varlen.**  ``IS_VARLEN`` is keyed off ``cu_seqlens`` alone (line 28, ``triton.heuristics``)::

       if IS_VARLEN:
           i_n, i_t = (tl.load(chunk_indices + i_t * 2).to(tl.int32),
                       tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32))
           bos, eos = (tl.load(cu_seqlens + i_n).to(tl.int32),
                       tl.load(cu_seqlens + i_n + 1).to(tl.int32))
           T = eos - bos
       else:
           bos, eos = i_b * T, i_b * T + T

   ``chunk_indices`` is ``[NT, 2]`` of ``(i_n, i_t)``: sequence index and *sequence-local* chunk
   index.  The grid is ``[NT, B * H]`` where (line 537) ``NT = len(chunk_indices) if cu_seqlens is
   not None else triton.cdiv(T, BT)``.  Because ``T`` becomes the sequence length, the trailing
   partial chunk of a sequence only writes rows ``< eos - bos`` (``boundary_check=(0, 1)`` on every
   store); the remainder of that block keeps the ``torch.zeros_like`` initial value.

5. **Output.**  ``Ai = torch.zeros_like(A, dtype=output_dtype)`` (line 539) with
   ``output_dtype = A.dtype if output_dtype is None else output_dtype`` (line 532), default
   ``torch.float``.  The kernels load and accumulate in fp32 and cast on store
   (``.to(p_Ai.dtype.element_ty, fp_downcast_rounding="rtne")``).  Therefore, for a partial trailing
   chunk, rows at/after the sequence end are **zero**, not identity, and columns at/after the
   sequence end of *valid* rows are zero too (they are strictly above the diagonal).

Verified vs assumed
-------------------
Verified against the source (quoted above, and re-derived numerically before writing this file):
the inverse is ``(I + A)^{-1}`` taken on the strictly-lower part of the input (its diagonal and
upper triangle are masked away - see below); the result is unit lower triangular; the layout is
``[B, T, H, BT]`` with rows = tokens and columns = within-chunk index; per-chunk blocks are
independent and the *whole* ``BT x BT`` block is inverted (16 x 16 sub-inverses are merged, not used
as the answer); varlen is driven by ``cu_seqlens`` + ``(i_n, i_t)`` chunk indices and only rows
``< eos - bos`` are written; the output dtype is ``output_dtype`` (``None`` -> input dtype) and the
arithmetic happens in fp32 before the cast (``FLA_TRIL_PRECISION`` defaults to ``"ieee"``, line 21,
so the merge dots are fp32 as well, not tf32).

Reconstructed / assumed (the source of ``fla/ops/utils/index.py`` was not downloaded, so
``prepare_chunk_indices`` below is rebuilt from its use sites):
* The short run of ``(i_n, i_t)`` pairs, sequence-major with ``i_t`` increasing, ``cdiv(len, BT)``
  per sequence.  Its exact *ordering* cannot matter: each program is independent, so only the set of
  pairs affects the output, and the implementation here iterates exactly the pairs it is given.
* The dtype of ``prepare_chunk_indices``' output (the kernel casts with ``.to(tl.int32)``, so the
  generated indices are int64 here, matching a plain ``torch.arange`` build).
* That callers write only the strictly-lower part, i.e. ``A.triu() == 0`` holds in memory (the
  host docstring requires it, and ``chunk_scaled_dot_kkt`` masks with ``tril(..., -1)``, so it does).
  See the masking bullet under "Deliberate deviations" for what happens if it does not.

Deliberate deviations from the Triton path
-----------------------------------------
* **Precision.**  The kernels force fp32 accumulation regardless of the input dtype.  Here the
  arithmetic runs in fp32 for fp16/bf16/fp32 input and in fp64 for fp64 input (so an fp64 call is
  exact against an fp64 oracle); the result is cast to ``output_dtype`` exactly like the store.
  This only differs from upstream for an fp64 input, which Triton would downcast to fp32.
* **Aliasing / contiguity.**  The kernels compute addresses from constants (``H * BT``, ``1``), so a
  non-contiguous ``A`` is silently misread.  This version indexes logically, so a non-contiguous
  ``[B, T, H, BT]`` tensor gives the same answer as its ``.contiguous()`` copy.
* **Input masking.**  The substitution loop reads a whole 16-wide *row* of the block
  (``b_a = -tl.load(A + (i_t * 16 + i) * H * BT + o_i + offset)``) and only truncates the sum
  through the strictly-lower ``m_A`` mask on the *other* operand, so a non-zero diagonal or
  upper-triangle entry inside a 16 x 16 diagonal block does leak into the Triton result.  This port
  applies ``tril(..., -1)`` and therefore ignores everything on or above the diagonal.  For the
  documented domain (``A.triu() == 0``) the two agree exactly; on malformed input this port is the
  forgiving one.  The test suite asserts the masking, and says so explicitly.
* **``@input_guard``** (line 506) is a Triton autotune helper with no math; it is dropped.
* ``torch.linalg.solve_triangular`` on the unit-lower system is the same forward substitution the
  kernels perform, chosen over ``torch.linalg.inv`` so the strictly-upper part of the result is
  exactly zero (no LU pivoting residue).
* The module is inference-only, like the kernel: no autograd support is claimed.

The block loop is over ``(sequence, chunk)`` pairs and the linear solve is batched over heads;
there is no per-token loop (see ``docs/performance.md`` and ``docs/plan-vllm-native-backend.md``
line 301: a per-token Python loop is a 14x regression).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

__all__ = ["solve_tril", "prepare_chunk_indices"]


def _batched_solve_enabled() -> bool:
    """Invert all full chunks at once, by Newton doubling, instead of one triangular solve per chunk.

    **Default on, measured** (§59, prefill shape T=512/BT=64/H=16):

    | | serial substitution | Newton doubling |
    |---|---|---|
    | ``solve_tril`` | 11.91 ms | **0.261 ms** (45.6x) |
    | ``chunk_gated_delta_rule`` | 16.10 ms | **8.43 ms** (1.91x) |

    The first attempt at this (a batched ``linalg.inv``) measured *worse* and is what §58 recorded:
    ``inv`` is an LU factorization, not the BLAS-3 batched solve.  ``X <- X(2I - M X)`` is: from
    ``X0 = I`` it gives ``X1 = I - A`` exactly, and each step doubles the number of correct terms, so
    ``ceil(log2(BT))`` steps are exact for a nilpotent-index-``BT`` strictly-lower part.  Six batched
    matmuls replace a 64-step dependency chain.

    ``TILEINFER_GDN_SOLVE_BATCH=0`` restores the per-block forward substitution; every oracle test
    passes on both paths.
    """
    from tileinfer.serving import flags  # noqa: PLC0415

    return flags.flag("gdn_solve_batch", "TILEINFER_GDN_SOLVE_BATCH", True)


def prepare_chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Build the ``[NT, 2]`` ``(i_n, i_t)`` table the FLA kernels expect (reconstructed).

    ``i_n`` is the sequence index inside ``cu_seqlens`` and ``i_t`` the sequence-local chunk index,
    with ``cdiv(eos - bos, chunk_size)`` chunks per sequence.  The source of the upstream function
    (``fla/ops/utils/index.py``) was not available, so this rebuilds it from its use sites; only the
    *set* of pairs is load-bearing for :func:`solve_tril`, and any order is acceptable because every
    program writes a disjoint block.
    """
    if cu_seqlens is None:  # upstream returns None for the dense form
        return None  # type: ignore[return-value]
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    from .hostcache import int_list, pairs_int_list  # noqa: PLC0415 - cached batch-meta reads

    bounds = int_list(cu_seqlens)
    if len(bounds) < 2:
        raise ValueError(f"cu_seqlens needs at least two entries, got {bounds}")
    pairs: List[Tuple[int, int]] = []
    for i_n, (bos, eos) in enumerate(zip(bounds[:-1], bounds[1:])):
        length = eos - bos
        if length <= 0:
            continue
        n_chunks = -(-length // chunk_size)
        pairs.extend((i_n, i_t) for i_t in range(n_chunks))
    if not pairs:
        return torch.zeros((0, 2), dtype=torch.int64, device=cu_seqlens.device)
    return torch.tensor(pairs, dtype=torch.int64, device=cu_seqlens.device)


def solve_tril(
    A: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor] = None,
    chunk_indices: Optional[torch.Tensor] = None,
    output_dtype: Optional[torch.dtype] = torch.float,
) -> torch.Tensor:
    """Blockwise ``(I + A)^{-1}`` on the last two axes of ``A``, one ``BT x BT`` block per chunk.

    Args:
        A: ``[B, T, H, BT]``, ``BT`` in ``{16, 32, 64}``, strictly lower triangular inside every
            chunk (``A.triu() == 0``).  Its diagonal and upper triangle are ignored, as in the
            kernels.
        cu_seqlens: ``[N + 1]`` cumulative sequence lengths over the packed ``T`` axis.  Passing it
            switches on the ragged form: chunk boundaries restart at every sequence start and the
            trailing partial chunk of a sequence only writes its valid rows.
        chunk_indices: optional precomputed ``[NT, 2]`` ``(i_n, i_t)`` table.  Only consulted when
            ``cu_seqlens`` is given (upstream keys ``IS_VARLEN`` off ``cu_seqlens`` alone); when
            omitted it is derived with :func:`prepare_chunk_indices`.
        output_dtype: dtype of the returned tensor; ``None`` means "same as ``A``".

    Returns:
        ``(I + A)^{-1}`` with the shape of ``A``: unit lower triangular inside each chunk, and zero
        in the rows a chunk does not cover (the kernel's store masks them out of a
        ``zeros_like`` buffer).
    """
    if A.dim() != 4:
        raise ValueError(f"solve_tril expects A of shape [B, T, H, BT], got {tuple(A.shape)}")
    BT = A.shape[-1]
    assert BT in (16, 32, 64), f"BT must be 16, 32 or 64, got {BT}"  # upstream: A.shape[-1] in [...]

    out_dtype = A.dtype if output_dtype is None else output_dtype
    B, T, H, _ = A.shape
    # fp32 accumulation mirrors the kernels' `.to(tl.float32)`; fp64 input stays fp64 (see module
    # docstring, "Deliberate deviations").
    work_dtype = torch.float64 if A.dtype == torch.float64 else torch.float32

    out = torch.zeros_like(A, dtype=out_dtype)
    if T == 0 or B == 0 or H == 0 or A.numel() == 0:
        return out

    # ---- batched fast path -------------------------------------------------------------------
    # The loop below is correct but serial in two ways: one Python iteration per chunk, and inside
    # each `length x length` block a unit-triangular solve, which is a 64-step dependency chain with
    # only H=16 matrices to hide it.  Measured at prefill shape (T=512, BT=64, H=16): **11.9 ms of
    # the chain's 16.1 ms, 74%** (§58).
    #
    # When the layout is dense and every chunk is full, all chunks can be inverted as one batch:
    # a general inverse is a parallel algorithm, and `tril` on the result keeps the same answer
    # (the inverse of a lower-triangular matrix is lower triangular).  Partial trailing chunks and
    # ragged batches keep the exact per-block path.
    if _batched_solve_enabled() and cu_seqlens is None and T % BT == 0:
        n = T // BT
        # ``A`` is [B, T, H, BT]: the last axis holds each chunk's *own* columns (chunk-local), not a
        # global [T, T] block, so splitting T into (n, BT) is the only reshape needed.
        blocks = A.view(B, n, BT, H, BT).permute(0, 1, 3, 2, 4).to(work_dtype)
        eye = torch.eye(BT, dtype=work_dtype, device=A.device)
        system = torch.tril(blocks, diagonal=-1) + eye
        # Newton doubling: X <- X (2I - M X) doubles the number of correct terms each step, so
        # ceil(log2(BT)) steps are exact for a unit-lower-triangular M whose strictly-lower part is
        # nilpotent with index BT.  Starting from X = I gives X1 = I - A exactly.
        # All steps are batched matmuls over [B*n, H, BT, BT]: no Python loop over chunks, and no
        # serial 64-step substitution (§58 measured that at 11.9 ms).
        inverse = eye.expand_as(system).contiguous()
        for _ in range(max(1, BT.bit_length() - 1)):
            inverse = inverse @ (2.0 * eye - system @ inverse)
        out.view(B, n, BT, H, BT).copy_(
            torch.tril(inverse).permute(0, 1, 3, 2, 4).to(out_dtype)
        )
        return out

    if cu_seqlens is None:
        # IS_VARLEN == False: batch b owns rows [b*T, (b+1)*T), chunks of BT inside it.
        for i_b in range(B):
            for t0 in range(0, T, BT):
                _invert_block(A, out, i_b, t0, min(BT, T - t0), work_dtype)
        return out

    from .hostcache import int_list, pairs_int_list  # noqa: PLC0415 - cached batch-meta reads

    bounds = int_list(cu_seqlens)
    if len(bounds) < 2 or bounds[0] != 0 or bounds[-1] > T:
        raise ValueError(f"cu_seqlens must start at 0 and span <= {T}, got {bounds[:4]}...{bounds[-1:]}")
    if any(later < earlier for earlier, later in zip(bounds[:-1], bounds[1:])):
        raise ValueError(f"cu_seqlens must be non-decreasing, got {bounds}")
    if chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    for i_n, i_t in pairs_int_list(chunk_indices):
        i_n, i_t = int(i_n), int(i_t)  # tolerate float/int32/int64 tables
        if not 0 <= i_n < len(bounds) - 1:
            raise ValueError(f"chunk_indices references sequence {i_n}, but cu_seqlens has {len(bounds) - 1}")
        bos, eos = bounds[i_n], bounds[i_n + 1]
        t0 = bos + i_t * BT
        length = min(BT, eos - t0)
        if length <= 0:  # pair outside the sequence: the kernel's grid would have skipped it too
            continue
        for i_b in range(B):
            _invert_block(A, out, i_b, t0, length, work_dtype)
    return out


def _invert_block(
    A: torch.Tensor,
    out: torch.Tensor,
    i_b: int,
    t0: int,
    length: int,
    work_dtype: torch.dtype,
) -> None:
    """Invert one chunk's ``length x length`` leading block for every head of batch ``i_b``.

    ``length`` is ``BT`` except for a sequence's trailing partial chunk, where the kernel's store
    masks everything past it; those rows keep the ``zeros_like`` value here as well.
    """
    if length <= 0:
        return
    # [length, H, length] -> [H, length, length]; only the strictly-lower part is read (m_A).
    block = A[i_b, t0 : t0 + length, :, :length].to(work_dtype).permute(1, 0, 2)
    lower = torch.tril(block, diagonal=-1).contiguous()
    eye = torch.eye(length, dtype=work_dtype, device=A.device)
    system = lower + eye  # I + A  (the input diagonal was discarded, as in the kernel)
    inverse = torch.linalg.solve_triangular(
        system, eye.expand(system.shape[0], length, length), upper=False, unitriangular=True
    )
    out[i_b, t0 : t0 + length, :, :length] = inverse.permute(1, 0, 2).to(out.dtype)
