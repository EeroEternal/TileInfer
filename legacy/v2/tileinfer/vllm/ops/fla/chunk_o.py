"""Torch fallback for FLA's ``chunk_fwd_o`` - GDN chunk-chain step 5 (the final one).

Role in the chain
-----------------
``chunk_gated_delta_rule_fwd`` runs these steps in order [vLLM chunk.py:37-86]::

    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(...)          # [vLLM] :61-71
    o = chunk_fwd_o(q=q, k=k, v=v_new, h=h, g=g, scale=scale,          # [vLLM] :72-82
                    cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
                    core_attn_out=core_attn_out)

This module is that last line: the intra-chunk "attention" plus the inter-chunk state term, per
64-token chunk, per value head.  ``h`` is the chunk's *entry* state from
``tileinfer/vllm/ops/fla/delta_h.py`` and ``v`` is **``v_new``**, not the caller's original values
[vLLM chunk.py:75] - the single most dangerous name collision in this chain (contract §1.1, §5.4).

Signature provenance
--------------------
* Every parameter name and position comes from the vLLM 0.23 call site, which passes everything by
  keyword: ``q``, ``k``, ``v``, ``h``, ``g``, ``scale``, ``cu_seqlens``, ``chunk_indices``,
  ``core_attn_out`` [vLLM chunk.py:72-82].  There is **no** ``chunk_size`` parameter: the sibling
  launcher has one [sglang chunk_o.py:134], vLLM replaced it with ``chunk_indices``
  [vLLM chunk.py:80] (contract §0 item 3).
* ``BT`` is the constant 64.  The sibling derives ``BT = min(chunk_size, max(16,
  next_power_of_2(T)))`` [sglang chunk_o.py:138], i.e. it shrinks on short sequences; vLLM 0.23 must
  **not**: "All kernels including ``chunk_fwd_kernel_o`` now use a fixed ``BT = chunk_size`` (64)"
  [vLLM qwen_gdn_linear_attn.py:1085-1087].  This port therefore always chunks at 64 and treats
  ``T < 64`` as a single partial chunk (contract §3.4).
* ``scale`` defaults to ``k.shape[-1] ** -0.5`` (sibling [sglang :143-144]; the same default is
  computed at the outer vLLM entry [vLLM chunk.py:228-229]).  vLLM always passes it explicitly.

Tensor contract
---------------
============  ==========================  ===============================  ========================
name          shape                       dtype                            source
============  ==========================  ===============================  ========================
``q``         ``[B, T, Hg, K]``           ``q.dtype`` (bf16 in prod.)      [sglang] :136
``k``         ``[B, T, Hg, K]``           ``k.dtype``                      [sglang] :136
``v``         ``[B, T, H, V]``            ``v.dtype`` = ``v_new``'s dtype  [sglang] :136-137
``h``         ``[B, NT, H, V, K]``        ``k.dtype`` (bf16)               [sglang] :74
``g``         ``[B, T, H]``               fp32 chunk-local cumsum          [vLLM] chunk.py:37-39
``scale``     Python ``float``            -                                [sglang] :39
``o`` (out)   ``[B, T, H, V]``            **``v.dtype``**                  [sglang] :146
============  ==========================  ===============================  ========================

``H`` is ``v.shape[-2]`` (the *value* head count) [sglang :137]; ``q``/``k`` have ``Hg`` key heads
and value head ``i_h`` reads key head ``i_h // (H // Hg)`` [sglang :70-71].  ``NT`` is the number of
``h`` slots: ``cdiv(T, BT)`` per batch row (dense) or ``chunk_offsets[-1]`` (ragged)
[contract §2.3].  ``h`` is addressed by the **global chunk ordinal** ``i_tg``: ``i_b * NT + i_t``
dense or the row of ``chunk_indices`` ragged [sglang :65-66, :55], which for a canonical
``chunk_indices`` equals ``chunk_offsets[i_n] + i_t`` - the index ``delta_h`` wrote.  This port
re-derives the canonical ordinal (see "Deliberate deviations") so the two operators cannot drift
apart.

The math (sibling kernel body; the vLLM 0.23 body was not available offline)
----------------------------------------------------------------------------
For one chunk of ``len`` tokens, one value head, with ``h_c = h[i_tg]`` the chunk's entry state
[sglang chunk_o.py:76-123]::

    b_o  = q @ h_c^T                                   # [sglang] :97   [BT, BV]
    b_A  = q @ k^T                                     # [sglang] :99   [BT, BT]
    if gated:
        b_o = b_o * exp(g)[:, None]                    # [sglang] :105  PLAIN exp on absolute g
        b_A = b_A * safe_exp(g[:, None] - g[None, :])  # [sglang] :106  one-sided clamp on DIFF
    m_A  = o_i[:, None] >= o_i[None, :]                # [sglang] :109  INCLUSIVE diagonal
    b_A  = tl.where(m_A, b_A, 0)                       # [sglang] :110  discard, not multiply
    b_o  = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale   # [sglang] :122  BOTH scaled
    store o = b_o                                      # [sglang] :123  tail rows untouched

i.e. per token ``t`` of the chunk::

    o_t = scale * ( exp(g_t) * (q_t . h_c) + sum_{j<=t} (q_t . k_j) * exp(g_t - g_j) * v_j )

For a single chunk with ``h_c = 0`` this is the closed form used as the test oracle, and it
simultaneously pins the ``>=`` mask, the ``safe_exp`` clamp and the double ``scale``
(contract §6.4).  Composed with ``delta_h`` it is exactly ``scale * q_t S_t`` of the per-token
gated delta rule.

Traps (all silent - contract §5)
---------------------------------
1. **``v`` is ``v_new``.**  ``chunk_gated_delta_rule_fwd`` passes ``v=v_new`` [vLLM chunk.py:75].
   Passing the original values produces a perfectly shaped, wrong answer; the interface test
   contains the negative control (contract §5.4).
2. **The triangle is ``>=``, not ``>``.**  The diagonal is the token's own contribution
   ``(q_i . k_i) * v_i``; dropping it loses an O(1) term while the output still looks plausible
   [sglang :109].  ``kkt``'s triangle is the opposite (strict ``>``), and the two are adjacent in
   the chain (contract §5.2).
3. **``torch.where`` must discard the upper triangle, never multiply by a 0/1 mask.**  For ``i<j``
   the exponent ``g_i - g_j`` is positive and can exceed fp32's ``exp`` range; ``inf * 0`` is NaN
   while ``tl.where`` drops the ``inf`` [sglang :110] (contract §5.1).
4. **``exp`` vs ``safe_exp`` split.**  ``exp(g)`` acts on the absolute gate (always ``<= 0``);
   ``safe_exp(g_i - g_j)`` acts on the difference (can be positive) [sglang :105-106].
5. **``scale`` multiplies both terms**, and it is applied *after* the gate multiply of ``b_o``
   [sglang :122].  Scaling only the intra-chunk term is a plausible-looking bug.
6. **``b_A`` is rounded to ``v``'s dtype before the second dot** [sglang :122]
   ``b_A.to(b_v.dtype)``; the first dot's ``h`` operand is used as loaded (``h`` is bf16 in the
   chain) [sglang :97, :122 + contract §2.4].

Precision
---------
The kernel accumulates both dots in fp32 and stores with a cast to ``v.dtype`` [sglang :123].
This port computes in fp32 for fp16/bf16/fp32 inputs and in fp64 when any input is fp64 (so an
fp64 call is exact against an fp64 oracle; Triton would downcast - the same deviation the other
ports in this package document).  The two operand roundings the kernel performs are reproduced:
``h`` is cast to ``q.dtype`` (the dtype of the other operand of that dot; in production both are
bf16) and ``b_A`` is cast to ``v.dtype`` before the intra-chunk dot.

``core_attn_out``
-----------------
vLLM accepts a pre-allocated output buffer and writes it in place; the buffer may only be reused
for inference and must share ``o``'s dtype [vLLM chunk.py:129-133].  The vLLM kernel body was not
available offline, so the exact write range is unestablished (contract §7.6).  This port writes the
**whole** buffer (``copy_``) and requires the exact ``[B, T, H, V]`` shape, returning the buffer
itself so a caller that does ``o = chunk_fwd_o(..., core_attn_out=buf)`` observes the alias; the
inference-only assertion stays where vLLM puts it, in the outer ``forward`` [vLLM chunk.py:129-133]
[推断].

Deliberate deviations from the Triton path
------------------------------------------
* **``BT`` never shrinks** (the sibling would use 16 for ``T <= 16``, 32 for ``T <= 32``); vLLM
  pins 64 [vLLM qwen_gdn_linear_attn.py:1085-1087] (contract §0 item 4).
* **Canonical ``h`` slot instead of the row of ``chunk_indices``.**  In the ragged form the kernel
  uses ``i_tg = i_t`` (the program's row in ``chunk_indices``) [sglang :54-58]; this port computes
  ``chunk_offsets[i_n] + i_t`` from ``cu_seqlens``.  For every canonical ``chunk_indices``
  (sequence-major, ``i_t`` increasing - what ``prepare_chunk_indices`` builds,
  ``solve_tril._indices``) the two are identical, so the kernel's behaviour is reproduced; for a
  permuted table this port stays consistent with ``delta_h`` (which always indexes by
  ``chunk_offsets``) instead of silently reading another chunk's state.
* **Unwritten rows are zero.**  ``o = torch.zeros_like(v)`` [sglang :146]; the kernel only stores
  rows covered by a sequence (``boundary_check`` [sglang :123]), so a packed tail not covered by
  ``cu_seqlens`` keeps the zero initial value here as well.
* **Logical indexing, not address arithmetic.**  The kernel derives addresses from ``Hg * K``,
  ``H * V`` and ``1`` [sglang :70-73], so non-contiguous inputs are silently misread unless
  ``@input_guard`` made them contiguous [sglang utils.py:145-179]; this port keeps the logical
  answer for a non-contiguous tensor (contract §5.4).
* **No autograd, no tile-size / autotune machinery** (``BK``/``BV`` are invisible in the maths),
  and no ``g_gamma`` / per-key gate (vLLM passes neither [vLLM chunk.py:72-82]).

Verified vs assumed
-------------------
Verified against the real vLLM 0.23 caller: parameter names/order, ``v=v_new``, the absence of
``chunk_size``, ``chunk_indices`` and ``core_attn_out`` being explicit arguments, the
``[B, T, H, V]`` / ``v.dtype`` output contract, the ``scale`` default and BT = 64.

Reconstructed from the sibling revision (vLLM 0.23's kernel body was not available offline): the
two dots, the gate split, the ``>=`` mask, the ``where`` discard, the double ``scale``, the
``b_A`` downcast, the ``h`` addressing, and ``zeros_like`` for ``o``.  Every such statement
carries a ``[sglang]`` line number into
``E:\\project\\python_test\\.venv\\Lib\\site-packages\\sglang\\srt\\layers\\attention\\fla\\chunk_o.py``.

Assumed (labelled ``[推断]``): that vLLM's ``BT`` is 64 here (pinned by the caller's comment rather
than this kernel's source), that ``h`` is cast to ``q.dtype`` for the inter-chunk dot (Triton
requires both ``tl.dot`` operands to share a dtype and ``h`` is bf16 in the chain), and the
``core_attn_out`` write range/alias behaviour.

Not claimed: bit-exactness against Triton (``exp`` and matmul ordering differ; ``FLA_USE_FAST_OPS``
and the tile sizes are not modelled - contract §5.4).
"""

from __future__ import annotations

from typing import Optional

import torch

from .delta_h import _iter_chunks, _n_slots, _prepare_chunk_offsets, _sequence_rows
from .gating import _working_dtype
from .kkt import _safe_exp  # exp(where(x <= 0, x, -inf)) [sglang op.py:25-27]

__all__ = ["chunk_fwd_o"]

#: vLLM 0.23 fixes ``BT = 64`` for every chunk kernel, ``chunk_fwd_kernel_o`` included; the sibling
#: launcher's ``min(chunk_size, max(16, next_pow2(T)))`` must NOT be copied
#: [vLLM qwen_gdn_linear_attn.py:1085-1087 vs sglang chunk_o.py:138] (contract §0 item 4).
_BT = 64


def chunk_fwd_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: Optional[torch.Tensor],
    scale: Optional[float] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    chunk_indices: Optional[torch.Tensor] = None,
    core_attn_out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Intra-chunk attention plus the inter-chunk state term, per 64-token chunk.

    Args:
        q: ``[B, T, Hg, K]`` queries (already scaled by the caller? no - ``scale`` is applied here).
        k: ``[B, T, Hg, K]`` keys.
        v: ``[B, T, H, V]`` **``v_new`` from** ``chunk_gated_delta_rule_fwd_h``, not the original
            values [vLLM chunk.py:75].
        h: ``[B, NT, H, V, K]`` per-chunk *entry* states in the layout ``delta_h`` produced.
        g: ``[B, T, H]`` fp32 chunk-local cumulative log-decay, or ``None`` to disable gating.
        scale: softmax-like scale; ``None`` -> ``K ** -0.5`` [sglang chunk_o.py:143-144].
        cu_seqlens: ``[N+1]`` cumulative token lengths of a flattened ``[1, T, ...]`` batch; ``B``
            must then be 1 [vLLM chunk.py:217-222].
        chunk_indices: optional ``[NT, 2]`` ``(i_n, i_t)`` table.  Accepted for ABI compatibility;
            this port enumerates chunks from ``cu_seqlens`` and computes the same canonical ``h``
            slots, so only the *set* of chunks matters (contract §4.2).  Ignored when
            ``cu_seqlens is None``.
        core_attn_out: optional pre-allocated ``[B, T, H, V]`` output buffer, written in full and
            returned (see the module docstring).

    Returns:
        ``[B, T, H, V]`` in ``v.dtype``: zero on rows no sequence covers.
    """
    if q.dim() != 4 or k.dim() != 4:
        raise ValueError(
            f"q and k must be [B, T, Hg, K], got {tuple(q.shape)} and {tuple(k.shape)}"
        )
    if v.dim() != 4:
        raise ValueError(f"v must be [B, T, H, V], got {tuple(v.shape)}")
    if h.dim() != 5:
        raise ValueError(f"h must be [B, NT, H, V, K], got {tuple(h.shape)}")
    B, T, Hg, K = q.shape
    if tuple(k.shape) != (B, T, Hg, K):
        raise ValueError(f"k must match q {(B, T, Hg, K)}, got {tuple(k.shape)}")
    H, V = v.shape[-2], v.shape[-1]
    if tuple(v.shape[:2]) != (B, T):
        raise ValueError(f"v must have the same [B, T] as q {(B, T)}, got {tuple(v.shape[:2])}")
    if tuple(h.shape[2:]) != (H, V, K):
        raise ValueError(f"h must be [B, NT, H, V, K] with (H, V, K) = {(H, V, K)}, got {tuple(h.shape)}")
    if Hg < 1 or H < 1 or H % Hg:
        raise ValueError(f"value heads H={H} must be a positive multiple of key heads Hg={Hg}")
    if g is not None:
        if g.dim() != 3:
            raise ValueError(f"g must be [B, T, H], got {tuple(g.shape)}")
        if tuple(g.shape) != (B, T, H):
            raise ValueError(f"g must be [B, T, H] = {(B, T, H)}, got {tuple(g.shape)}")
    if cu_seqlens is not None and B != 1:
        # [vLLM] chunk.py:217-222, same wording.
        raise ValueError(
            f"The batch size is expected to be 1 rather than {B} when using `cu_seqlens`."
            f"Please flatten variable-length inputs before processing."
        )
    if scale is None:
        scale = K**-0.5  # [sglang] chunk_o.py:143-144; same default at [vLLM] chunk.py:228-229

    rows = _sequence_rows(B, T, cu_seqlens)
    offsets = None if cu_seqlens is None else _prepare_chunk_offsets(cu_seqlens, _BT)
    nt_dense, nt_total = _n_slots(B, T, _BT, offsets)
    if cu_seqlens is None:
        expected_h = (B, nt_dense, H, V, K)
    else:
        expected_h = (1, nt_total, H, V, K)
    if tuple(h.shape) != expected_h:
        raise ValueError(
            f"h must be {expected_h} to match the chunk plan of this operator; got {tuple(h.shape)}"
        )

    o = torch.zeros_like(v)  # [sglang] chunk_o.py:146: dtype = v.dtype, unfilled rows stay 0
    if T == 0 or B == 0 or H == 0 or V == 0 or K == 0:
        if core_attn_out is not None:
            return _write_core_attn_out(core_attn_out, o)
        return o

    work = _working_dtype(q, k, v, h, g)
    # value head -> key head: i_h // (H // Hg) [sglang chunk_o.py:70-71]
    head_map = (torch.arange(H, device=q.device) // (H // Hg)) if Hg != H else None
    neg_zero = torch.zeros((), dtype=work, device=q.device)

    for chunk in _iter_chunks(rows, _BT, nt_dense, offsets):
        i_b, t0, length = chunk.i_b, chunk.t0, chunk.length
        idx = torch.arange(t0, t0 + length, device=q.device)
        # Global chunk ordinal: i_b*NT + i_t dense, chunk_offsets[i_n] + i_t ragged
        # [sglang] :65-66 / :55, and the very index delta_h wrote [sglang] :67-83.
        h_blk = h[i_b, chunk.i_t] if offsets is None else h[0, chunk.slot]

        q_blk = q[i_b, idx]  # logical indexing: correct for non-contiguous input
        k_blk = k[i_b, idx]
        if head_map is not None:
            q_blk = q_blk[:, head_map]  # [length, H, K]
            k_blk = k_blk[:, head_map]
        v_blk = v[i_b, idx]  # [length, H, V]
        q_w = q_blk.to(work)
        k_w = k_blk.to(work)

        # b_o += tl.dot(b_q, tl.trans(b_h)) [sglang :97].  Triton needs both operands in one dtype;
        # h is bf16 in the chain and q is bf16, so cast h to q.dtype (then accumulate in work).
        inter = torch.einsum("thk,hvk->thv", q_w, h_blk.to(q.dtype).to(work))
        a = torch.einsum("thk,shk->tsh", q_w, k_w)  # [length, length, H] = q_t . k_s

        if g is not None:
            g_blk = g[i_b, idx].to(work)  # [length, H]
            # PLAIN exp on the absolute gate (always <= 0) [sglang :105]
            inter = inter * torch.exp(g_blk).unsqueeze(-1)
            # safe_exp on the DIFFERENCE (can be positive) [sglang :106]
            a = a * _safe_exp(g_blk[:, None, :] - g_blk[None, :, :])

        # m_A = o_i[:, None] >= o_i[None, :] - the diagonal is KEPT [sglang :109].
        pos = torch.arange(length, device=q.device)
        keep = pos[:, None] >= pos[None, :]  # [length, length]
        # tl.where discards the positive-exponent upper triangle; `a * keep` would turn inf into NaN
        # [sglang :110] (contract §5.1).
        a = torch.where(keep.unsqueeze(-1), a, neg_zero)
        # b_A.to(b_v.dtype) before the second dot [sglang :122].
        a = a.to(v.dtype).to(work)
        intra = torch.einsum("tsh,shv->thv", a, v_blk.to(work))

        # b_o * scale + tl.dot(...) * scale [sglang :122]: BOTH terms are scaled.
        o_blk = inter * scale + intra * scale
        o[i_b, idx] = o_blk.to(o.dtype)  # store cast to p_o.dtype [sglang :123]

    if core_attn_out is not None:
        return _write_core_attn_out(core_attn_out, o)
    return o


def _write_core_attn_out(core_attn_out: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
    """Write ``o`` into the caller's buffer in full and return the buffer (``[推断]``, §7.6)."""
    if tuple(core_attn_out.shape) != tuple(o.shape):
        raise ValueError(
            f"core_attn_out must have o's shape {tuple(o.shape)}, got {tuple(core_attn_out.shape)}"
        )
    core_attn_out.copy_(o)  # store semantics: the kernel casts to the buffer's element type
    return core_attn_out
