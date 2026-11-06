"""Torch fallback for FLA's ``chunk_gated_delta_rule_fwd_h`` - GDN chunk-chain step 4.

Role in the chain
-----------------
``chunk_gated_delta_rule_fwd`` runs these steps in order [vLLM chunk.py:37-86]::

    A = chunk_scaled_dot_kkt_fwd(k, beta, g, ..., output_dtype=torch.float32)  # [vLLM] :41-48
    A = solve_tril(A, ..., output_dtype=k.dtype)                               # [vLLM] :49-51
    w, u = recompute_w_u_fwd(k, v, beta, A=A, g_cumsum=g, ...)                 # [vLLM] :52-60
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(                      # [vLLM] :61-71
        k=k, w=w, u=u, g=g, initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, chunk_offsets=chunk_offsets)
    o = chunk_fwd_o(q=q, k=k, v=v_new, h=h, g=g, scale=scale, ...)             # [vLLM] :72-82

This module is the fourth line.  It is the only producer of the two intermediates the last step
consumes: ``h`` (one in-chunk *entry* state per chunk, ``[B, NT, H, V, K]``) and ``v_new`` (the
"new v", ``[B, T, H, V]``) which the caller passes as ``chunk_fwd_o``'s **``v``** argument
[vLLM chunk.py:75] - not the original ``v`` (contract §1.1, §5.4).

Signature provenance
--------------------
* Every parameter name and its position come from the vLLM 0.23 call site, which passes everything
  by keyword: ``k``, ``w``, ``u``, ``g``, ``initial_state``, ``output_final_state``, ``cu_seqlens``,
  ``chunk_indices``, ``chunk_offsets`` [vLLM chunk.py:61-71].
* vLLM returns **three** values ``(h, v_new, final_state)`` [vLLM chunk.py:61 + :86].  The vendored
  sibling revision returns two and has ``gk`` / ``initial_state_indices`` / ``save_new_value``
  instead [sglang chunk_delta_h.py:274-284, :340]; contract §0 item 2 pins the vLLM shape of the
  ABI, so those three sibling-only parameters do not exist here.
* ``chunk_indices`` is accepted but **never read**: the sibling kernel's varlen branch reads only
  ``cu_seqlens`` and ``chunk_offsets`` [sglang :61-67]; the parameter list of the sibling launcher
  does not even contain ``chunk_indices`` [sglang :274-284].  This port keeps the parameter for ABI
  compatibility and ignores it, exactly as the kernel would [vLLM chunk.py:69].
* ``chunk_offsets`` is a vLLM-side lift of something the sibling computes internally
  [sglang :289-302 vs vLLM chunk.py:70]; it is **cumulative chunk counts**, ``[N+1]``, built as
  ``cat([0], cdiv(lens, BT)).cumsum(-1)`` (contract §4.3).  When omitted (only ever for tests here:
  vLLM always passes it) this port derives it from ``cu_seqlens`` and raises if a supplied one
  disagrees, because a wrong offset silently reads/writes the wrong ``h`` slot.
* ``BT`` is the constant 64: ``CHUNK_SIZE = 64`` in the sibling [sglang :19] and pinned to
  ``FLA_CHUNK_SIZE = 64`` for vLLM 0.23 [vLLM qwen_gdn_linear_attn.py:1085-1087]; there is no
  ``chunk_size`` parameter in this ABI.

Tensor contract
---------------
============  ==========================  ==============================  ==========================
name          shape                       dtype                           source
============  ==========================  ==============================  ==========================
``k``         ``[B, T, Hg, K]``           ``k.dtype`` (bf16 in prod.)     [sglang] :85, :285
``w``         ``[B, T, H, K]``            ``k.dtype`` (wy_fast output)    [sglang] :86
``u``         ``[B, T, H, V]``            ``v.dtype`` (wy_fast output)    [sglang] :84
``g``         ``[B, T, H]``               fp32 chunk-local cumsum         [vLLM] chunk.py:37-39
``initial_state`` ``[N, H, V, K]``       state dtype                     [vLLM] chunk.py:169
``h`` (out)   ``[B, NT, H, V, K]``        **``k.dtype`` (bf16)**          [sglang] :305
``v_new``     ``[B, T, H, V]``            ``u.dtype``                     [sglang] :307
``final_state`` ``[N, H, V, K]``          ``initial_state.dtype``         [推断] see below
============  ==========================  ==============================  ==========================

``H`` is the *value* head count ``u.shape[-2]``; ``Hg`` is the key head count and value head ``i_h``
reads key head ``i_h // (H // Hg)`` [sglang :59-60, :85].  The last axis order is ``[V, K]``, i.e.
``h[v, k]`` and **not** ``h[k, v]`` [sglang :104 ``make_block_ptr(h0, (V, K), (K, 1), ...)`` and
:305 ``k.new_empty(B, NT, H, V, K)``].  ``NT`` is ``cdiv(T, BT)`` per batch row in the dense form
and ``len(chunk_indices) == chunk_offsets[-1]`` in the ragged form [sglang :66, :296-302].

``h`` is laid out by the **global chunk ordinal**: the kernel advances its pointer by
``boh * H * V * K`` where ``boh = chunk_offsets[i_n]`` (ragged) or ``i_n * NT`` (dense)
[sglang :67, :71, :83].  That is the only reason ``chunk_offsets`` exists: ``chunk_fwd_o``
re-derives the same flat index to read ``h`` back (contract §4.3).  A chunk's ``h`` slot is the
state *before* that chunk, because the store is the first statement of the token-chunk loop body
and the update is the last [sglang :123-127 vs :229-233] (contract §5.4).

The math (sibling kernel body; the vLLM 0.23 body was not available offline)
----------------------------------------------------------------------------
For one sequence, one chunk ``c`` (tokens ``t0 .. t0+len-1``), one value head, with ``S_c = h[c]``
the entry state [sglang :123-251]::

    # store the ENTRY state first
    h[c]          = S_c                                             # [sglang] :127
    # prediction of the chunk's values from the entry state
    pred[t]       = w[t] @ S_c^T                                    # [sglang] :144-148  (K-tiled)
    v_new[t]      = u[t] - pred[t]                                  # [sglang] :167-170
    # saved BEFORE the gate multiply (see "the v_new store" below)
    v_new_store[t] = v_new[t]                                       # [sglang] :172-176
    if gated:
        g_last    = g[bos + min((c+1)*BT, T) - 1]                   # [sglang] :178-180
        v_new[t] *= safe_exp(g_last - g[t])                         # [sglang] :185
        S_c      *= exp(g_last)                                     # [sglang] :186-187
    v_new         = v_new.to(k.dtype)                               # [sglang] :227
    S_{c+1}       = S_c + sum_t k[t] (x) v_new[t]                   # [sglang] :229-233  outer product

i.e. ``S_{c+1} = S_c * exp(g_last) + sum_t exp(g_last - g_t) v_new[t] k_t^T`` with
``h[c] = S_c``.  Iterating the chunk map is exactly the per-token gated delta-rule recurrence
``S_t = exp(d_t) S_{t-1} + beta_t (v_t - exp(d_t) S_{t-1} k_t) k_t^T`` (``d_t`` = the raw log decay,
``g`` = its chunk-local cumsum), with ``v_new[t] = beta_t (v_t - exp(d_t) S_{t-1} k_t)``
[推断: algebra; it is what makes ``chunk_fwd_o`` reproduce ``q_t S_t``.  The tests verify it against
a per-token fp64 recurrence].

The ``v_new`` store: the sibling source stores the *ungated* ``u - w S_c^T``
[sglang chunk_delta_h.py:172-176], then gates a copy for the state update [:185].  This port follows
the source.  Note that contract §6.3 item 2 summarises ``v_new`` as
``(u - w h^T) * safe_exp(g_last - g)`` - i.e. the gated value; that summary is **not** what the
sibling code does, and the gated variant does not compose: ``chunk_fwd_o`` already applies
``exp(g_i - g_j)`` to its ``v`` argument [sglang chunk_o.py:106], so a gated ``v_new`` would be
decayed twice and the composed output would no longer equal ``q_t S_t``.  The interface test in
``tests/test_gdn_delta_h_chunk_o.py`` pins the ungated choice against the per-token recurrence;
the vLLM 0.23 body itself was unavailable offline (§7.1), so the store's position in vLLM is
[推断] (inherited from the sibling).

Traps (all silent - contract §5)
---------------------------------
1. **Plain ``exp`` on the absolute gate, ``safe_exp`` on the difference.**  ``exp(g_last)`` is
   ``<= 1`` because a log-decay cumsum is non-positive [sglang :186]; ``safe_exp(g_last - g)``
   clamps every positive argument to ``exp(-inf) = 0`` [sglang :185 + op.py:25-27].  Swapping the
   two (or using plain ``exp`` on the difference) overflows fp32 for a large decay and poisons the
   state with ``inf`` (contract §5.1).
2. **The state is rounded on the way into the prediction dot.**  ``tl.dot(b_w,
   tl.trans(b_h1).to(b_w.dtype))`` rounds the fp32 accumulator to ``w``'s dtype for the dot
   [sglang :148]; the accumulator itself stays fp32 across chunks.  Keeping the prediction in fp32
   changes ``v_new`` at the bf16-quantum level (the ``test_delta_h_rounds_state_before_the_dot``
   probe amplifies it to O(1)).
3. **The ``h`` snapshot is bf16** (``k.new_empty`` [sglang :305]) while the accumulator is fp32:
   each chunk boundary publishes a rounded state but does not consume it [推断: the loop carries
   ``b_h1`` in registers; verified numerically by the round-trip probe].
4. **The last-token gate uses the chunk's last *valid* token**, ``min((i_t+1)*BT, T) - 1``
   [sglang :178] - not ``g[BT-1]``, which is a different (padding) token for a partial chunk.
5. **``v_new`` is stored before the gate** (see above) and **rounded to ``k.dtype`` before the
   ``k^T v_new`` dot** [sglang :227], while ``v_new`` itself is written in ``u.dtype``
   [sglang :307]: two different casts when ``u.dtype != k.dtype``.

Precision
---------
The kernel keeps ``b_h1`` in fp32 and casts at the three documented points.  This port computes in
fp32 for fp16/bf16/fp32 inputs and in fp64 when any input is fp64 (so an fp64 call is exact against
an fp64 oracle; Triton would downcast it - the same deviation the other ports in this package
document).  The three round-trips are reproduced explicitly (``S.to(w.dtype)`` before the
prediction dot, the ``h`` store cast, ``v_new.to(k.dtype)`` before the state update), so a bf16
call is comparable to Triton modulo torch-vs-Triton ``exp``/matmul ordering.

``output_final_state`` and ``initial_state``
--------------------------------------------
* ``final_state`` is an **independent** buffer; ``initial_state`` is never written.  vLLM's version
  differs from the sibling here: the sibling updates ``initial_state`` in place (``INPLACE_UPDATE=
  True``) and returns no separate tensor [sglang :94-100, :254-271], while vLLM's ABI returns
  ``final_state`` alongside ``initial_state`` and no longer passes ``initial_state_indices``
  [vLLM chunk.py:61-71].  That it therefore allocates instead of aliasing is [推断] (contract §7.5);
  vLLM's consumer writes ``ssm_state[...] = last_recurrent_state`` explicitly
  [vLLM qwen_gdn_linear_attn.py:1532], which is consistent with an independent buffer.
* ``output_final_state=False`` -> ``final_state is None``, per the vLLM docstring
  "Final state of shape ``[N, H, V, K]`` if ``output_final_state=True`` else ``None``"
  [vLLM chunk.py:180-181].  The sibling has no such flag, so the *other* behaviour the contract
  flags as unknown (§7.5: whether a final_state is still written) is not reproduced here.
* ``initial_state=None`` disables the initial state (the kernel's ``USE_INITIAL_STATE=False``
  [sglang :97-105]).  In vLLM's prefill path it is always a gathered tensor
  [vLLM qwen_gdn_linear_attn.py:1513-1514], so this is only a test convenience; the resulting
  ``final_state`` then uses ``u.dtype`` [推断].

Deliberate deviations from the Triton path
------------------------------------------
* **Unwritten rows are zero, not uninitialised.**  ``v_new = torch.empty_like(u)`` [sglang :307]
  is only stored for tokens ``< eos`` of a sequence [sglang :172-176, ``boundary_check``], so a
  packed tail not covered by ``cu_seqlens`` is garbage upstream; this port zero-fills.  ``h`` is
  allocated with ``k.new_empty`` and *every* chunk slot is written before it is read, so its
  allocation is faithful and deterministic.
* **Logical indexing, not address arithmetic.**  The kernel derives addresses from ``H * V * K``,
  so a non-contiguous ``k``/``w``/``u``/``initial_state`` is silently misread unless
  ``@input_guard`` made it contiguous [sglang utils.py:145-179].  This port indexes logically: a
  non-contiguous tensor gives the same answer as its contiguous copy (contract §5.4).
* **No autograd, no Triton autotune / ``input_guard`` / ``do_not_specialize`` machinery, no
  ``gk`` per-key gate** (vLLM does not pass one [vLLM chunk.py:61-71], contract §7.8).

Verified vs assumed
-------------------
Verified against the real vLLM 0.23 caller: parameter names/order, the 3-tuple return, ``u``'s
identity, ``output_final_state``, ``chunk_offsets`` being an explicit argument, ``initial_state``'s
``[N, H, V, K]`` layout, ``final_state``'s None-when-disabled docstring, ``B == 1`` under
``cu_seqlens``, and BT = 64 (vLLM ``chunk.py`` / ``qwen_gdn_linear_attn.py`` line numbers above).

Reconstructed from the sibling revision (vLLM 0.23's kernel body was not available offline): the
formula and its order, the entry-state snapshot, ``V``-before-``K`` layout, the flat chunk-ordinal
addressing, the ``safe_exp``/``exp`` split, the ``g_last`` tail rule, the three dtype round-trips,
``empty_like``/``new_empty``, and the head mapping.  Every such statement carries a ``[sglang]``
line number into
``E:\\project\\python_test\\.venv\\Lib\\site-packages\\sglang\\srt\\layers\\attention\\fla\\chunk_delta_h.py``.

Assumed (labelled ``[推断]``): the ungated ``v_new`` store in vLLM (sibling-verified, contract
text says otherwise - see above), ``final_state`` being an independent buffer, its dtype being
``initial_state.dtype``, and the ``u.dtype`` fallback when ``initial_state is None``.

Not claimed: bit-exactness against Triton.  ``exp`` is torch's, the tile splits (``BV``, the
``K > 64`` unrolling [sglang :74-80]) are invisible in the maths, and ``FLA_USE_FAST_OPS`` is not
modelled (contract §5.4).
"""

from __future__ import annotations

from typing import Iterator, List, NamedTuple, Optional, Sequence, Tuple

import torch

from .gating import _working_dtype

# Re-use the sibling ports' one-sided clamp instead of defining a third dialect of `safe_exp`.
# It is `exp(where(x <= 0, x, -inf))` [sglang op.py:25-27]; see kkt.py for the full rationale.
from .kkt import _safe_exp

__all__ = ["chunk_gated_delta_rule_fwd_h"]

#: ``BT`` is pinned to 64 for vLLM 0.23 [vLLM qwen_gdn_linear_attn.py:1085-1087], matching the
#: sibling's ``CHUNK_SIZE`` [sglang chunk_delta_h.py:19].  Nothing here shrinks it for short inputs.
_BT = 64


class _Chunk(NamedTuple):
    """One ``(batch row, sequence, chunk)`` unit of work.

    ``i_b`` is the index into the ``[B, T, ...]`` token tensors; ``i_n`` the row of the state
    tensors (``initial_state``/``final_state``); ``i_t`` the *sequence-local* chunk index;
    ``t0``/``length`` the global token range; ``seq_len`` the owning sequence's length (needed for
    the ``g_last`` tail rule); ``slot`` the flat chunk ordinal into ``h``.
    """

    i_b: int
    i_n: int
    i_t: int
    t0: int
    length: int
    seq_len: int
    slot: int


def _sequence_rows(
    B: int, T: int, cu_seqlens: Optional[torch.Tensor]
) -> List[Tuple[int, int, int, int]]:
    """``(i_b, i_n, bos, eos)`` for every batch row / packed sequence.

    Dense: one row per batch element over the whole token axis.  Ragged: ``B`` must be 1 [vLLM
    chunk.py:217-222] and each interval of ``cu_seqlens`` is one row; the state row ``i_n`` is the
    ``cu_seqlens`` index, the token row is always 0.  Trailing tokens not covered by
    ``cu_seqlens[-1]`` belong to no row (they are never written; same convention as kkt.py/wu.py).
    """
    if cu_seqlens is None:
        return [(b, b, 0, T) for b in range(B)]

    from .hostcache import int_list  # noqa: PLC0415 - cached batch-meta reads

    bounds = int_list(cu_seqlens)
    if len(bounds) < 2 or bounds[0] != 0 or bounds[-1] > T:
        raise ValueError(
            f"cu_seqlens must start at 0 and span <= {T}, got {bounds[:4]}...{bounds[-1:]}"
        )
    if any(later < earlier for earlier, later in zip(bounds[:-1], bounds[1:])):
        raise ValueError(f"cu_seqlens must be non-decreasing, got {bounds}")
    return [(0, i_n, bos, eos) for i_n, (bos, eos) in enumerate(zip(bounds[:-1], bounds[1:]))]


def _prepare_chunk_offsets(cu_seqlens: torch.Tensor, chunk_size: int) -> List[int]:
    """``[N+1]`` cumulative **chunk** counts (not token counts) - ``prepare_chunk_offsets``.

    Rebuilt from its use sites (contract §4.3): ``cat([0], cdiv(lens, BT)).cumsum(-1)``
    [sglang index.py:29-35, used at chunk_delta_h.py:67].  This is how both this module and
    ``chunk_o`` address the flat ``h`` slots, so it is shared rather than derived twice.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    from .hostcache import int_list  # noqa: PLC0415 - cached batch-meta reads

    bounds = int_list(cu_seqlens)
    if len(bounds) < 2:
        raise ValueError(f"cu_seqlens needs at least two entries, got {bounds}")
    if bounds[0] != 0:
        raise ValueError(f"cu_seqlens must start at 0, got {bounds[:4]}")
    if any(later < earlier for earlier, later in zip(bounds[:-1], bounds[1:])):
        raise ValueError(f"cu_seqlens must be non-decreasing, got {bounds}")
    offsets = [0]
    for bos, eos in zip(bounds[:-1], bounds[1:]):
        length = eos - bos
        offsets.append(offsets[-1] + (-(-length // chunk_size)))
    return offsets


def _iter_chunks(
    rows: Sequence[Tuple[int, int, int, int]],
    chunk_size: int,
    nt_dense: Optional[int],
    offsets: Optional[Sequence[int]],
) -> Iterator[_Chunk]:
    """Yield one :class:`_Chunk` per chunk, row-major (all chunks of a row, in order)."""
    for i_b, i_n, bos, eos in rows:
        seq_len = eos - bos
        n_chunks = -(-seq_len // chunk_size)
        for i_t in range(n_chunks):
            t0 = bos + i_t * chunk_size
            length = min(chunk_size, seq_len - i_t * chunk_size)
            slot = i_b * nt_dense + i_t if offsets is None else offsets[i_n] + i_t
            yield _Chunk(i_b, i_n, i_t, int(t0), int(length), int(seq_len), int(slot))


def _n_slots(
    B: int,
    T: int,
    chunk_size: int,
    offsets: Optional[Sequence[int]],
) -> Tuple[int, int]:
    """``(nt_dense, nt_total)``: the per-row chunk count and the total number of ``h`` slots.

    Dense ``h`` is ``[B, nt_dense, ...]``; ragged ``h`` is ``[1, nt_total, ...]`` with
    ``nt_total == chunk_offsets[-1]`` [sglang chunk_delta_h.py:305, contract §2.3].
    """
    if offsets is None:
        nt_dense = -(-T // chunk_size)
        return nt_dense, B * nt_dense
    return 0, int(offsets[-1])


def chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: Optional[torch.Tensor],
    initial_state: Optional[torch.Tensor],
    output_final_state: bool,
    cu_seqlens: Optional[torch.Tensor] = None,
    chunk_indices: Optional[torch.Tensor] = None,
    chunk_offsets: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Chunked gated delta-rule state pass: per-chunk entry states, ``v_new`` and the final state.

    Args:
        k: ``[B, T, Hg, K]`` keys; value head ``i_h`` reads key head ``i_h // (H // Hg)``.
        w: ``[B, T, H, K]`` WY keys from ``recompute_w_u_fwd`` (``H`` = value heads).
        u: ``[B, T, H, V]`` WY values.  The sibling kernel receives this very tensor under the
            parameter name ``v`` [sglang chunk_delta_h.py:314] - see the module docstring.
        g: ``[B, T, H]`` fp32 chunk-local cumulative log-decay (``chunk_local_cumsum`` output), or
            ``None`` to disable gating (``USE_G=False`` [sglang :331]).
        initial_state: ``[N, H, V, K]`` entry state per sequence, ``N = B`` (dense) or
            ``len(cu_seqlens) - 1`` (ragged) [vLLM chunk.py:169-171]; ``None`` means zero state.
        output_final_state: return the ``[N, H, V, K]`` final state, else ``None``
            [vLLM chunk.py:180-181].
        cu_seqlens: ``[N+1]`` cumulative token lengths of a flattened ``[1, T, ...]`` batch.
        chunk_indices: accepted for ABI compatibility, **ignored** - the kernel does not read it
            [sglang :61-67, :274-284], contract §4.2.
        chunk_offsets: ``[N+1]`` cumulative chunk counts [vLLM chunk.py:70]; derived when omitted
            and validated against the derivation when supplied.

    Returns:
        ``(h, v_new, final_state)``: ``h`` is ``[B, NT, H, V, K]`` in ``k.dtype`` holding each
        chunk's *entry* state, ``v_new`` is ``[B, T, H, V]`` in ``u.dtype`` = the ungated
        ``u - w @ h[c]^T``, and ``final_state`` is ``[N, H, V, K]`` or ``None``.
    """
    if k.dim() != 4:
        raise ValueError(f"k must be [B, T, Hg, K], got {tuple(k.shape)}")
    if w.dim() != 4:
        raise ValueError(f"w must be [B, T, H, K], got {tuple(w.shape)}")
    if u.dim() != 4:
        raise ValueError(f"u must be [B, T, H, V], got {tuple(u.shape)}")
    B, T, Hg, K = k.shape
    H, V = u.shape[-2], u.shape[-1]
    if tuple(w.shape) != (B, T, H, K):
        raise ValueError(f"w must be [B, T, H, K] = {(B, T, H, K)}, got {tuple(w.shape)}")
    if tuple(u.shape[:2]) != (B, T):
        raise ValueError(f"u must have the same [B, T] as k {(B, T)}, got {tuple(u.shape[:2])}")
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

    rows = _sequence_rows(B, T, cu_seqlens)
    N = len(rows)
    offsets = None if cu_seqlens is None else _prepare_chunk_offsets(cu_seqlens, _BT)
    if cu_seqlens is not None and chunk_offsets is not None:
        from .hostcache import int_list  # noqa: PLC0415 - cached batch-meta reads

        given = (
            int_list(chunk_offsets)
            if hasattr(chunk_offsets, "tolist")
            else [int(v) for v in chunk_offsets]
        )
        if given != offsets:
            raise ValueError(
                f"chunk_offsets must be the cumulative chunk counts {offsets} (prepare_chunk_offsets,"
                f" contract §4.3); got {given}.  A wrong offset silently indexes the wrong h slot"
            )

    if initial_state is not None:
        if initial_state.dim() != 4:
            raise ValueError(
                f"initial_state must be [N, H, V, K], got {tuple(initial_state.shape)}"
            )
        if tuple(initial_state.shape) != (N, H, V, K):
            raise ValueError(
                f"initial_state must be [N, H, V, K] = {(N, H, V, K)}, got "
                f"{tuple(initial_state.shape)}"
            )

    nt_dense, nt_total = _n_slots(B, T, _BT, offsets)
    if cu_seqlens is None:
        h = k.new_empty(B, nt_dense, H, V, K)  # [sglang] :305: dtype = k.dtype (bf16 in prod.)
    else:
        h = k.new_empty(1, nt_total, H, V, K)
    v_new = torch.zeros_like(u)  # [sglang] :307 `empty_like(u)`; zeros see module docstring

    if output_final_state:
        state_dtype = initial_state.dtype if initial_state is not None else u.dtype
        final_state: Optional[torch.Tensor] = torch.zeros(
            N, H, V, K, dtype=state_dtype, device=k.device
        )
    else:
        final_state = None

    work = _working_dtype(k, w, u, g, initial_state)
    # value head -> key head: i_h // (H // Hg) [sglang chunk_delta_h.py:85]
    head_map = (torch.arange(H, device=k.device) // (H // Hg)) if Hg != H else None
    zero_state = torch.zeros(H, V, K, dtype=work, device=k.device)

    for i_b, i_n, bos, eos in rows:
        seq_len = eos - bos
        if initial_state is not None:
            # Initial state row: vLLM gathers it before the call (no `initial_state_indices`
            # anymore) [vLLM chunk.py:61-71, contract §2.3].
            state = initial_state[i_n].to(work)
        else:
            state = zero_state.clone()
        for i_t in range(-(-seq_len // _BT)):
            t0 = bos + i_t * _BT
            length = min(_BT, seq_len - i_t * _BT)
            idx = torch.arange(t0, t0 + length, device=k.device)
            slot = i_b * nt_dense + i_t if offsets is None else offsets[i_n] + i_t

            # The snapshot is the ENTRY state, stored before any update [sglang :123-127].
            if offsets is None:
                h[i_b, i_t] = state.to(h.dtype)
            else:
                h[0, slot] = state.to(h.dtype)

            # b_v = dot(b_w, trans(b_h1).to(b_w.dtype)) [sglang :144-148]: the fp32 accumulator is
            # rounded to w's dtype for this dot, then the dot accumulates in fp32.
            pred = torch.einsum(
                "thk,hvk->thv", w[i_b, idx].to(work), state.to(w.dtype).to(work)
            )
            v_blk = u[i_b, idx].to(work) - pred  # [sglang] :167-170
            # SAVE_NEW_VALUE happens BEFORE the gate multiply [sglang :172-176 vs :185].
            v_new[i_b, idx] = v_blk.to(v_new.dtype)

            if g is not None:
                g_blk = g[i_b, idx].to(work)  # [length, H]
                # last valid token of THIS chunk, not g[BT-1] [sglang :178-180]
                g_last = g[i_b, t0 + length - 1].to(work)  # [H]
                # safe_exp on the DIFFERENCE (can be positive for out-of-order gates)
                v_blk = v_blk * _safe_exp(g_last.unsqueeze(0) - g_blk).unsqueeze(-1)
                # plain exp on the absolute gate (always <= 0) [sglang :186]
                state = state * torch.exp(g_last)[:, None, None]

            # b_v = b_v.to(k.dtype.element_ty) [sglang :227]: round before the k^T v dot.
            v_k = v_blk.to(k.dtype).to(work)
            k_blk = k[i_b, idx]  # logical indexing: correct for non-contiguous input
            if head_map is not None:
                k_blk = k_blk[:, head_map]
            # b_h1 += tl.trans(tl.dot(b_k, b_v)) [sglang :229-233]
            state = state + torch.einsum("thk,thv->hvk", k_blk.to(work), v_k)

        if final_state is not None:
            # Epilogue store to the state dtype [sglang :254-271].
            final_state[i_n] = state.to(final_state.dtype)

    return h, v_new, final_state
