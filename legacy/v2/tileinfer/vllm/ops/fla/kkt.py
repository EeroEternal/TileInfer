"""Torch fallback for FLA's ``chunk_scaled_dot_kkt_fwd`` - GDN chunk-chain step 1.

Role in the chain
-----------------
``chunk_gated_delta_rule_fwd`` runs these steps in order [vLLM chunk.py:37-86]::

    g = chunk_local_cumsum(g, chunk_size=FLA_CHUNK_SIZE, ...)        # [vLLM] chunk.py:37-39
    A = chunk_scaled_dot_kkt_fwd(k, beta, g, ..., output_dtype=fp32) # [vLLM] chunk.py:41-48
    A = solve_tril(A, ..., output_dtype=k.dtype)                     # [vLLM] chunk.py:49-51
    w, u = recompute_w_u_fwd(k, v, beta, A=A, g_cumsum=g, ...)       # [vLLM] chunk.py:52-60

This module is the second line only.  It produces the strictly-lower transition matrix that
``solve_tril`` (``tileinfer/vllm/ops/fla/solve_tril.py``) inverts into ``(I + A)^{-1}``.

Signature provenance
--------------------
* The gating argument is named ``g`` in vLLM, and the caller passes ``g=g`` [vLLM chunk.py:44].
  Its value is nonetheless ``chunk_local_cumsum``'s output, i.e. the quantity the sibling kernel
  calls ``g_cumsum`` [sglang chunk_scaled_dot_kkt.py:78-79].  Two names, one tensor; the other
  operator in this pair names it ``g_cumsum`` [vLLM chunk.py:57].
* ``chunk_indices`` is accepted by vLLM's version and passed at [vLLM chunk.py:46].  The vendored
  sibling revision has no such parameter [sglang chunk_scaled_dot_kkt.py:89-96]; the parameter list
  here follows vLLM (contract §0 item 1).
* ``chunk_size=64`` and ``output_dtype=torch.float32`` are the sibling's defaults
  [sglang chunk_scaled_dot_kkt.py:94-95]; that vLLM 0.23 still has the ``chunk_size`` parameter is
  [推断] - ``chunk.py`` never passes it, which is only consistent with "the default (64) is what
  production uses".  ``output_dtype=torch.float32`` *is* pinned by the vLLM call site
  [vLLM chunk.py:47].
* ``BT`` is a constant 64 in vLLM 0.23, never shrunk for short sequences
  [vLLM qwen_gdn_linear_attn.py:1085-1087].  This port therefore uses ``chunk_size`` as given and
  never derives a smaller BT from ``T`` (the sibling ``chunk_o`` does shrink it; contract §0 item 4).
  ``T < BT`` is a legal single partial chunk.

Tensor contract
---------------
============  ==========================  =============================  =========================
name          shape                       dtype                          source
============  ==========================  =============================  =========================
``k``         ``[B, T, Hg, K]``           ``k.dtype`` (bf16 in prod.)    [sglang] :120
``beta``      ``[B, T, H]`` (3-D)         ``beta.dtype``                 [sglang] :122, [vLLM] chunk.py:216
``g``         ``[B, T, H]``               fp32 (cumsum's default)        [vLLM] chunk.py:37-39 + contract §3.2
``A`` (out)   ``[B, T, H, BT]``           ``output_dtype`` = fp32        [sglang] :128, [vLLM] chunk.py:47
============  ==========================  =============================  =========================

``H`` is taken from ``beta.shape[-1]``, *not* from ``k`` [sglang chunk_scaled_dot_kkt.py:122], and
``A = torch.empty(B, T, H, BT)`` uses that ``H`` [sglang :128].  With GQA (``HK = Hg < H``), value
head ``i_h`` reads key head ``i_h // (H // Hg)`` [sglang :63]:
``k_head = i_h // (HV // HK)``.

Layout: ``A`` is a flat ``(T, BT)`` matrix per ``(batch, head)`` - row = token (global), column =
*within-chunk* index, row stride ``H * BT``, column stride 1 [sglang :84].  So "lower/upper
triangular" is a per-chunk statement: chunk ``c``'s rows occupy global rows
``[c*BT, c*BT + len)`` and its own ``BT`` columns, one independent triangle per chunk (contract §3.1).

The math (all from the sibling kernel body, not from vLLM 0.23)
--------------------------------------------------------------
For token indices ``i > j`` inside one chunk and one value head ``h``::

    A[i, j, h] = beta[i, h] * safe_exp(g[i, h] - g[j, h]) * (k_i . k_j)

built in this exact order by the kernel::

    b_A      = dot(b_k, trans(b_k))                          # [sglang] :71   k_i . k_j (fp32 acc)
    b_A     *= safe_exp(b_g[:, None] - b_g[None, :])         # [sglang] :78-79
    b_A     *= b_beta[:, None]                               # [sglang] :81   ROW scale (query i)
    b_A      = tl.where(o_t[:, None] > o_t[None, :], b_A, 0) # [sglang] :82   STRICT lower triangle
    tl.store(p_A, b_A.to(p_A.dtype.element_ty), boundary_check=(0, 1))       # [sglang] :83-86

Three properties are load-bearing and each is a silent-failure trap (contract §5):

1. **Strictly lower triangular, diagonal dropped.**  The mask is ``>``, not ``>=``
   [sglang chunk_scaled_dot_kkt.py:82], matching ``solve_tril``'s host requirement
   "``A.triu() == 0``" [vLLM fla_solve_tril.py:515].  Adding the diagonal is invisible downstream
   because ``solve_tril`` discards the diagonal silently (contract §5.2), so a wrong triangle here
   only shows up in a test like the negative control in ``tests/test_gdn_kkt_wu.py``.
2. **Smallest-exp clamp on the *difference*.**  ``safe_exp(x) = exp(where(x <= 0, x, -inf))``
   [sglang op.py:26-27].  ``g`` is a chunk-local cumulative log-decay and is non-increasing, so
   ``g_i - g_j`` is ``<= 0`` only in the lower triangle and *positive* in the upper triangle, where
   its magnitude reaches ``BT * |decay|`` (hundreds to thousands).  Plain ``exp`` there overflows
   fp32/fp64 to ``inf``; the clamp makes it ``0.0`` exactly.  ``safe_exp`` is used only on
   differences, never on an absolute ``g`` (which is always ``<= 0`` and gets plain ``exp`` in the
   other operator; contract §5.1).  The clamp and the ``where`` below are two independent guards for
   this one overflow: the kernel applies both [sglang :79 and :82] and so does this port, and
   ``tests/test_gdn_kkt_wu.py`` shows that the plausible shortcut (plain ``exp`` + ``* mask``)
   yields NaN on these inputs.  (Dropping *only* the clamp is unobservable while the ``where`` stays:
   the overflow lands in a lane that is discarded anyway - which is precisely why the two must be
   reviewed together.)
3. **``torch.where`` must *discard*, not multiply.**  Masking with ``value * mask`` evaluates
   ``inf * 0 = nan`` in the upper triangle, and the NaN travels into ``solve_tril`` -> ``w``/``u``
   -> the whole output.  The kernel uses ``tl.where`` [sglang :82]; this port uses ``torch.where``.
   ``tests/test_gdn_kkt_wu.py`` contains a negative control that shows the multiply form produces
   NaN on exactly the inputs where the ``where`` form stays finite.
   ``beta`` is a *row* (query-index) scale, ``beta[:, None]`` [sglang :81]; a column scale would be
   a different operator and is rejected by a dedicated test.

Precision
---------
The Triton kernel accumulates ``dot`` in fp32 and loads ``beta``/``g`` in their own dtypes; the store
casts to the output element type [sglang :86].  This port follows: fp32 work for fp16/bf16/fp32
inputs, fp64 work when any of ``k``/``beta``/``g`` is fp64 (so an fp64 call is exact against an fp64
oracle; Triton would have downcast it - the same deviation the ``solve_tril`` port documents).

Deliberate deviations from the Triton path
------------------------------------------
* **Trailing rows are zero, not uninitialised.**  Upstream allocates ``A`` with ``torch.empty``
  [sglang :128] and only stores rows ``< end of sequence`` (``boundary_check=(0, 1)``
  [sglang :86]), so rows/columns a partial chunk does not cover are garbage upstream.  They are
  never read: ``solve_tril`` starts from ``zeros_like`` and its stores are boundary-masked
  (contract §3.5).  This port allocates zeros, which is deterministic and still inside the
  documented contract ("unwritten rows may be anything", contract §6.1 item 6).
* **Contiguity.**  The kernel computes addresses from constants (``H * BT``, ``1``), so a
  non-contiguous ``k``/``beta``/``g`` is silently misread unless ``@input_guard`` made it
  contiguous [sglang utils.py:145-179].  This port indexes logically, so a non-contiguous tensor
  gives the same answer as its ``.contiguous()`` copy (contract §5.4).
* **No autograd.**  The module is inference-only, like the rest of the ported chain.
* **No ``@triton.autotune`` / ``input_guard`` / ``do_not_specialize`` machinery** - no math.

Verified vs assumed
-------------------
Verified against the real vLLM 0.23 caller: the call order, every keyword argument, ``g``'s name,
``chunk_indices``, ``output_dtype=torch.float32``, ``beta`` being 3-D, BT being pinned to 64
(vLLM ``chunk.py`` and ``qwen_gdn_linear_attn.py`` line numbers above).

Reconstructed from the sibling revision (the vLLM 0.23 kernel body was not available offline):
the formula and its factor order, the strict-lower ``>`` mask, the ``safe_exp`` clamp direction, the
``H = beta.shape[-1]`` head count, the GQA head mapping, the ``(T, BT)`` layout, ``empty`` vs
``zeros``, and the fp32 accumulation.  Every such statement is labelled ``[sglang]`` above with a
line number into
``E:\\python_test\\.venv\\Lib\\site-packages\\sglang\\srt\\layers\\attention\\fla\\chunk_scaled_dot_kkt.py``.

Assumed (labelled ``[推断]``): that vLLM 0.23 still exposes ``chunk_size`` with default 64.

Not claimed: bit-exactness against Triton.  ``exp`` is torch's, not Triton's; the autotuner's
splits and ``FLA_USE_FAST_OPS`` are not modelled (contract §5.4).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from .solve_tril import prepare_chunk_indices

__all__ = ["chunk_scaled_dot_kkt_fwd"]


def _safe_exp(x: torch.Tensor) -> torch.Tensor:
    """``exp(where(x <= 0, x, -inf))`` - the one-sided clamp of [sglang op.py:26-27].

    Positive arguments map to ``exp(-inf) == 0`` exactly; they are *not* evaluated.  This is used
    only on ``g`` *differences*, never on an absolute (always non-positive) ``g``.
    """
    return torch.exp(torch.where(x <= 0, x, torch.full_like(x, float("-inf"))))


def _chunk_blocks(
    T: int,
    cu_seqlens: Optional[torch.Tensor],
    chunk_indices: Optional[torch.Tensor],
    chunk_size: int,
) -> List[Tuple[int, int]]:
    """``(t0, length)`` of every chunk, dense or ragged.

    Dense (``cu_seqlens is None``): chunks of ``chunk_size`` over the whole ``T`` axis, which is what
    the kernel's grid ``NT = cdiv(T, BT)`` does [sglang chunk_scaled_dot_kkt.py:127].

    Ragged: one block per ``(i_n, i_t)`` pair of ``chunk_indices``, where ``i_n`` indexes
    ``cu_seqlens`` and ``i_t`` is the *sequence-local* chunk index
    [sglang :44-49, index.py:17-26].  ``chunk_indices`` is only consulted when ``cu_seqlens`` is
    given, because ``IS_VARLEN`` is keyed off ``cu_seqlens`` alone [sglang :142].
    """
    if cu_seqlens is None:
        return [(t0, min(chunk_size, T - t0)) for t0 in range(0, T, chunk_size)]

    from .hostcache import int_list, pairs_int_list  # noqa: PLC0415 - cached batch-meta reads

    bounds = int_list(cu_seqlens)
    if len(bounds) < 2 or bounds[0] != 0 or bounds[-1] > T:
        raise ValueError(
            f"cu_seqlens must start at 0 and span <= {T}, got {bounds[:4]}...{bounds[-1:]}"
        )
    if any(later < earlier for earlier, later in zip(bounds[:-1], bounds[1:])):
        raise ValueError(f"cu_seqlens must be non-decreasing, got {bounds}")
    if chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)

    blocks: List[Tuple[int, int]] = []
    for i_n, i_t in pairs_int_list(chunk_indices):
        i_n, i_t = int(i_n), int(i_t)  # tolerate float/int32/int64 tables
        if not 0 <= i_n < len(bounds) - 1:
            raise ValueError(
                f"chunk_indices references sequence {i_n}, but cu_seqlens has {len(bounds) - 1}"
            )
        bos, eos = bounds[i_n], bounds[i_n + 1]
        t0 = bos + i_t * chunk_size
        length = min(chunk_size, eos - t0)
        if length <= 0:  # pair outside the sequence: the kernel's grid would have skipped it
            continue
        blocks.append((t0, length))
    return blocks


def chunk_scaled_dot_kkt_fwd(
    k: torch.Tensor,
    beta: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    chunk_indices: Optional[torch.Tensor] = None,
    chunk_size: int = 64,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """``A = beta_i * safe_exp(g_i - g_j) * (k_i . k_j)`` for ``i > j`` inside each chunk, else 0.

    Args:
        k: ``[B, T, Hg, K]`` keys.  Value head ``i_h`` reads key head ``i_h // (H // Hg)``.
        beta: ``[B, T, H]`` delta-rule step sizes; a *row* (query-index) scale on ``A``.
        g: ``[B, T, H]`` fp32 chunk-local cumulative log-decay (``chunk_local_cumsum`` output).
            ``None`` disables gating (``USE_G=False`` in the kernel [sglang :143]).
        cu_seqlens: ``[N + 1]`` cumulative sequence lengths of a packed ``[1, T, ...]`` batch.
        chunk_indices: optional ``[NT, 2]`` ``(i_n, i_t)`` table; derived when omitted, ignored when
            ``cu_seqlens`` is ``None``.
        chunk_size: chunk width ``BT``; vLLM 0.23 always uses 64 [vLLM
            qwen_gdn_linear_attn.py:1085-1087].
        output_dtype: output dtype; the vLLM call site passes ``torch.float32``
            [vLLM chunk.py:47].

    Returns:
        ``[B, T, H, BT]`` strictly lower triangular inside every chunk; ``H = beta.shape[-1]``.
        Rows/columns a partial trailing chunk does not cover are zero (see module docstring).
    """
    if k.dim() != 4:
        raise ValueError(f"k must be [B, T, Hg, K], got {tuple(k.shape)}")
    if beta.dim() != 3:  # [vLLM] chunk.py:216 asserts len(beta.shape) == 3
        raise ValueError(f"beta must be [B, T, H], got {tuple(beta.shape)}")
    B, T, Hg, K = k.shape
    H = beta.shape[-1]
    if beta.shape != (B, T, H):
        raise ValueError(f"beta must be [B, T, H] with H = {H}, got {tuple(beta.shape)}")
    if g is not None and g.dim() != 3:
        raise ValueError(f"g must be [B, T, H], got {tuple(g.shape)}")
    if g is not None and g.shape != (B, T, H):
        raise ValueError(f"g must be [B, T, H] = {(B, T, H)}, got {tuple(g.shape)}")
    if Hg < 1 or H % Hg != 0 or H < Hg:
        raise ValueError(f"value heads H={H} must be a positive multiple of key heads Hg={Hg}")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    if output_dtype is None:
        raise ValueError("output_dtype must be a torch.dtype (upstream would pass it to torch.empty)")

    A = torch.zeros(B, T, H, chunk_size, dtype=output_dtype, device=k.device)
    if B == 0 or T == 0 or H == 0 or K == 0:
        return A

    # fp32 accumulation as in the kernel; fp64 is passed through so an fp64 call is exact.
    work = (
        torch.float64
        if k.dtype == torch.float64
        or beta.dtype == torch.float64
        or (g is not None and g.dtype == torch.float64)
        else torch.float32
    )
    neg_zero = torch.zeros((), dtype=work, device=k.device)
    # value head -> key head: i_h // (H // Hg) [sglang chunk_scaled_dot_kkt.py:63]
    head_map = (
        (torch.arange(H, device=k.device) // (H // Hg)) if Hg != H else None
    )
    blocks = _chunk_blocks(T, cu_seqlens, chunk_indices, chunk_size)

    for t0, length in blocks:
        idx = torch.arange(t0, t0 + length, device=k.device)
        g_blk = g[:, idx].to(work) if g is not None else None  # [B, l, H]
        beta_blk = beta[:, idx].to(work)  # [B, l, H]
        gd = None
        if g_blk is not None:
            gd = g_blk[:, :, None, :] - g_blk[:, None, :, :]  # [B, l, l, H]: g_i - g_j
            gd = gd.permute(0, 3, 1, 2)  # [B, H, l, l]
        mask = (
            torch.arange(length, device=k.device)[:, None]
            > torch.arange(length, device=k.device)[None, :]
        )  # [l, l] strictly lower: `>` not `>=` [sglang :82]
        for i_b in range(B):
            kb = k[i_b, idx]  # logical indexing: correct for non-contiguous input
            if head_map is not None:
                kb = kb[:, head_map]
            kb = kb.to(work)  # [l, H, K]
            a = torch.einsum("ihd,jhd->hij", kb, kb)  # [H, l, l]
            if gd is not None:
                # safe_exp on the DIFFERENCE; upper-triangle positives become exactly 0.
                a = a * _safe_exp(gd[i_b])
            # beta scales rows = the query index i: b_A *= b_beta[:, None] [sglang :81]
            a = a * beta_blk[i_b].permute(1, 0).unsqueeze(-1)  # [H, l, 1]
            # discard, do not multiply: `* mask` would turn inf into NaN [sglang :82]
            a = torch.where(mask.unsqueeze(0), a, neg_zero)
            A[i_b, t0 : t0 + length, :, :length] = a.permute(1, 0, 2).to(output_dtype)
    return A
