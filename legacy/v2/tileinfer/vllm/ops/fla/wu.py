"""Torch fallback for FLA's ``recompute_w_u_fwd`` (``wy_fast``) - GDN chunk-chain step 3.

Role in the chain
-----------------
``chunk_gated_delta_rule_fwd`` runs these steps in order [vLLM chunk.py:37-86]::

    A = chunk_scaled_dot_kkt_fwd(..., output_dtype=torch.float32)    # [vLLM] chunk.py:41-48
    A = solve_tril(A, ..., output_dtype=k.dtype)                     # [vLLM] chunk.py:49-51
    w, u = recompute_w_u_fwd(k, v, beta, A=A, g_cumsum=g, ...)       # [vLLM] chunk.py:52-60
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(k, w, u, g, ...)  # [vLLM] chunk.py:61-71

This module is the third line: the WY representation of the chunk's delta-rule update.  ``A`` is the
*already inverted* ``(I + A)^{-1}`` produced by ``solve_tril`` - this operator never inverts anything.
The returned ``u`` is the delta-rule ``u`` (the "new v" consumed by the delta_h kernel, where the
kernel parameter is confusingly called ``v`` [vLLM chunk.py:64 + contract §1.1]).

Signature provenance
--------------------
* Parameter names come from the vLLM 0.23 call site, which passes every argument by keyword
  [vLLM chunk.py:52-60]: ``k``, ``v``, ``beta``, ``A``, ``g_cumsum``, ``cu_seqlens``,
  ``chunk_indices``.  The vendored sibling revision has the *same* names in the same order
  [sglang wy_fast.py:111-119], so its line numbers are safe to quote here (contract §0).
* Note the gate is named **``g_cumsum``** here but **``g``** in ``chunk_scaled_dot_kkt_fwd``
  [vLLM chunk.py:44 vs :57] - the same tensor, two names.
* There is no ``chunk_size`` parameter: ``BT`` is read back from ``A.shape[-1]``
  [sglang wy_fast.py:122].  vLLM 0.23 pins that to 64 [vLLM qwen_gdn_linear_attn.py:1085-1087].

Tensor contract
---------------
============  ==========================  =============================  =========================
name          shape                       dtype                          source
============  ==========================  =============================  =========================
``k``         ``[B, T, Hg, K]``           ``k.dtype`` (bf16 in prod.)    [sglang] :120
``v``         ``[B, T, H, V]``            ``v.dtype``                    [sglang] :120
``beta``      ``[B, T, H]``               ``beta.dtype``                 [sglang] :121, [vLLM] chunk.py:216
``g_cumsum``  ``[B, T, H]``               fp32 (cumsum's default)        [vLLM] chunk.py:37-39
``A``         ``[B, T, H, BT]``           **bf16** = ``k.dtype``         [vLLM] chunk.py:50 :56
``w`` (out)   ``[B, T, H, K]``            ``k.dtype``                    [sglang] :130
``u`` (out)   ``[B, T, H, V]``            ``v.dtype``                    [sglang] :129
============  ==========================  =============================  =========================

``H`` is ``v.shape[-2]`` (the *value* head count) [sglang :121] and ``BT`` is ``A.shape[-1]``
[sglang :122].  So ``w`` has value-head count but key width: GQA-wise ``w`` is ``[B, T, HV, K]``,
not ``[B, T, HK, K]`` [sglang :130].  ``u`` is ``empty_like(v)`` [sglang :129] and therefore
``V``-wide with ``v``'s dtype.  ``A`` being bf16 is not an implementation detail but the chain
contract: the caller rewrites it with ``solve_tril(..., output_dtype=k.dtype)``
[vLLM chunk.py:49-51].

The math (sibling kernel body; vLLM 0.23 body was not available offline)
-----------------------------------------------------------------------
Per chunk, per value head ``h``, with ``A_inv = (I + A)^{-1}``::

    u = A_inv @ (v * beta[:, None])                  # [sglang] wy_fast.py:84-85
    w = A_inv @ (k * beta[:, None] * exp(g)[:, None]) # [sglang] wy_fast.py:64, :106-107

exactly as the kernel writes it::

    b_g  = tl.exp(tl.load(p_g))                          # [sglang] :64   PLAIN exp (g <= 0)
    b_vb = (b_v * b_beta[:, None]).to(b_v.dtype)         # [sglang] :84   row scale, then DOWNCAST
    b_u  = tl.dot(b_A, b_vb, allow_tf32=False)           # [sglang] :85   fp32 accumulator
    b_kb = (b_k * b_beta[:, None] * b_g[:, None]).to(b_k.dtype)  # [sglang] :106  DOWNCAST again
    b_w  = tl.dot(b_A, b_kb)                             # [sglang] :107

Four traps, all silent (contract §5):

1. **``exp``, never ``safe_exp``, on the absolute gate.**  ``g_cumsum`` is a chunk-local cumulative
   log-decay, so every entry is ``<= 0`` and ``exp(g)`` is in ``(0, 1]``.  The one-sided clamp
   ``safe_exp`` is for ``g`` *differences* (the kkt operator) and must not appear here
   [sglang :64 vs :84 of chunk_scaled_dot_kkt.py:79]; contract §5.1.
2. **``beta`` is a row scale (query index) in both products** [sglang :84, :106].  It is never a
   column scale and never elementwise against ``A``.
3. **The operands are cast *down* to ``v.dtype`` / ``k.dtype`` before each dot, and ``A_inv`` is
   already bf16.**  ``beta`` / ``exp(g)`` are applied in fp32 and the product is rounded to bf16
   before it enters the dot [sglang :84, :106]; the two dots are then fp32 accumulations of bf16
   operands [推断: Triton ``tl.dot`` accumulates fp32 for bf16 inputs - contract §5.3].  The result
   is stored with a cast to the destination dtype [sglang :86, :108].  This port reproduces the
   "round the operand, then accumulate in fp32" order; it does not keep ``A_inv`` in fp32 either -
   it uses the values it is given, which are bf16 in the real chain.
4. **Head mapping is the GQA map.**  Value head ``i_h`` reads key head ``i_h // (H // Hg)``
   [sglang :90]; ``u`` uses ``v`` directly.

Precision
---------
fp32 accumulation for fp16/bf16/fp32 inputs, fp64 when any input is fp64 (so an fp64 call is exact
against an fp64 oracle; Triton would downcast - the deviation the ``solve_tril`` port documents).
A strict reading of Triton's type promotion would round the ``b_k * b_beta`` intermediate to bf16
*before* multiplying by ``b_g`` (both factors bf16 -> bf16), whereas this port multiplies in fp32
and rounds once, following contract §5.3 item 2; the two differ by at most one bf16 ulp and only
for non-fp32 inputs.

Deliberate deviations from the Triton path
------------------------------------------
* **Trailing rows are zero, not uninitialised.**  Upstream uses ``u = torch.empty_like(v)``
  [sglang :129] and ``w = k.new_empty(...)`` [sglang :130] and stores only valid rows
  (``boundary_check``, [sglang :86, :108]), so rows a partial chunk / an uncovered packed tail does
  not cover are garbage there; downstream only reads the valid rows (contract §3.5).  This port
  allocates zeros, keeping the "unwritten means ignored" contract while being deterministic.
* **Contiguity.**  The kernel addresses memory from constants (``H * V``, ``H * K``, ``1``), so a
  non-contiguous ``k``/``v``/``beta``/``A`` is silently misread unless ``@input_guard`` fixed it
  [sglang utils.py:145-179].  This port indexes logically and gives the same answer for a
  non-contiguous tensor as for its ``.contiguous()`` copy (contract §5.4).
* **No autograd**, no ``@triton.autotune``/``input_guard`` machinery (no math in either).

Verified vs assumed
-------------------
Verified against the real vLLM 0.23 caller: every parameter name, the ``g_cumsum`` vs ``g`` naming,
``A`` being the bf16 ``solve_tril`` output, the argument order, and BT = 64 (line numbers above).

Reconstructed from the sibling revision: the formulas, the factor order, plain ``exp`` on the
absolute gate, the row-scaled ``beta``, the pre-dot downcasts, the output dtypes/shapes, the head
count taken from ``v``, ``empty_like``/``new_empty`` (see every ``[sglang]`` line number, pointing at
``E:\\python_test\\.venv\\Lib\\site-packages\\sglang\\srt\\layers\\attention\\fla\\wy_fast.py``).

Assumed (labelled ``[推断]``): fp32 dot accumulation for bf16 operands (Triton semantics), and the
single-rounding variant of the ``k * beta * exp(g)`` product described under "Precision".

Not claimed: bit-exactness against Triton (torch's ``exp`` and matmul differ; ``allow_tf32`` has no
torch analogue here; ``FLA_USE_FAST_OPS`` is not modelled - contract §5.4).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from .solve_tril import prepare_chunk_indices

__all__ = ["recompute_w_u_fwd"]


def _chunk_blocks(
    T: int,
    cu_seqlens: Optional[torch.Tensor],
    chunk_indices: Optional[torch.Tensor],
    chunk_size: int,
) -> List[Tuple[int, int]]:
    """``(t0, length)`` of every chunk, dense or ragged.

    Dense: ``NT = cdiv(T, BT)`` chunks over the token axis [sglang wy_fast.py:126].  Ragged: one
    block per ``(i_n, i_t)`` pair (sequence index, sequence-local chunk index), keyed off
    ``cu_seqlens`` alone [sglang :124-126, index.py:17-26].
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


def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor,
    A: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor],
    chunk_indices: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``u = A_inv @ (v * beta)`` and ``w = A_inv @ (k * beta * exp(g))``, per chunk and head.

    Args:
        k: ``[B, T, Hg, K]`` keys.
        v: ``[B, T, H, V]`` values; ``H`` (its head count) is the operator's head count.
        beta: ``[B, T, H]`` delta-rule step sizes, applied as a row scale in both products.
        g_cumsum: ``[B, T, H]`` chunk-local cumulative log-decay (``exp``, not ``safe_exp``).
        A: ``[B, T, H, BT]`` the inverted transition matrix ``(I + A)^{-1}`` from ``solve_tril``,
            bf16 in the real chain [vLLM chunk.py:49-51].
        cu_seqlens: ``[N + 1]`` cumulative sequence lengths of a packed ``[1, T, ...]`` batch.
        chunk_indices: optional ``[NT, 2]`` ``(i_n, i_t)`` table; derived when omitted, ignored when
            ``cu_seqlens`` is ``None``.

    Returns:
        ``(w, u)``: ``w`` is ``[B, T, H, K]`` with ``k.dtype`` (value-head count, key width),
        ``u`` is ``[B, T, H, V]`` with ``v.dtype`` [sglang wy_fast.py:129-130].  Rows a partial
        chunk / uncovered packed tail does not cover are zero (see module docstring).
    """
    if k.dim() != 4:
        raise ValueError(f"k must be [B, T, Hg, K], got {tuple(k.shape)}")
    if v.dim() != 4:
        raise ValueError(f"v must be [B, T, H, V], got {tuple(v.shape)}")
    if beta.dim() != 3:  # [vLLM] chunk.py:216 asserts len(beta.shape) == 3
        raise ValueError(f"beta must be [B, T, H], got {tuple(beta.shape)}")
    if A.dim() != 4:
        raise ValueError(f"A must be [B, T, H, BT], got {tuple(A.shape)}")
    B, T, Hg, K = k.shape
    H, V = v.shape[-2], v.shape[-1]
    BT = A.shape[-1]
    if v.shape[:2] != (B, T):
        raise ValueError(f"v must have the same [B, T] as k {(B, T)}, got {tuple(v.shape[:2])}")
    if beta.shape != (B, T, H):
        raise ValueError(f"beta must be [B, T, H] = {(B, T, H)}, got {tuple(beta.shape)}")
    if g_cumsum.shape != (B, T, H):
        raise ValueError(f"g_cumsum must be [B, T, H] = {(B, T, H)}, got {tuple(g_cumsum.shape)}")
    if A.shape[:2] != (B, T) or A.shape[2] != H:
        raise ValueError(f"A must be [B, T, H, BT] = {(B, T, H, BT)}, got {tuple(A.shape)}")
    if Hg < 1 or H % Hg != 0 or H < Hg:
        raise ValueError(f"value heads H={H} must be a positive multiple of key heads Hg={Hg}")
    if BT < 1:
        raise ValueError(f"BT = A.shape[-1] must be >= 1, got {BT}")

    w = k.new_zeros(B, T, H, K)  # [sglang] :130 `k.new_empty` -> zeros here, see module docstring
    u = v.new_zeros(B, T, H, V)  # [sglang] :129 `torch.empty_like(v)` -> zeros here
    if B == 0 or T == 0 or H == 0 or K == 0 or V == 0:
        return w, u

    work = (
        torch.float64
        if k.dtype == torch.float64
        or v.dtype == torch.float64
        or beta.dtype == torch.float64
        or g_cumsum.dtype == torch.float64
        or A.dtype == torch.float64
        else torch.float32
    )
    # value head -> key head: i_h // (H // Hg) [sglang wy_fast.py:90]
    head_map = (torch.arange(H, device=k.device) // (H // Hg)) if Hg != H else None
    blocks = _chunk_blocks(T, cu_seqlens, chunk_indices, BT)

    for t0, length in blocks:
        idx = torch.arange(t0, t0 + length, device=k.device)
        beta_blk = beta[:, idx].to(work)  # [B, l, H]
        # PLAIN exp on the absolute gate (g <= 0), never safe_exp [sglang wy_fast.py:64]
        exp_g = torch.exp(g_cumsum[:, idx].to(work))  # [B, l, H]
        for i_b in range(B):
            a = A[i_b, idx, :, :length].to(work).permute(1, 0, 2)  # [H, l, l]
            vb = v[i_b, idx].to(work)  # [l, H, V]
            # row scale, then DOWNCAST to v.dtype before the dot [sglang :84]
            vb = (vb * beta_blk[i_b].unsqueeze(-1)).to(v.dtype).to(work)
            u_blk = torch.einsum("hij,jhd->ihd", a, vb)  # [l, H, V], fp32 accumulation
            u[i_b, idx] = u_blk.to(v.dtype)  # [sglang] :86 store cast

            kb = k[i_b, idx]  # logical indexing: correct for non-contiguous input
            if head_map is not None:
                kb = kb[:, head_map]
            kb = kb.to(work)  # [l, H, K]
            # row scale by beta and exp(g), then DOWNCAST to k.dtype before the dot [sglang :106]
            kb = (
                kb * beta_blk[i_b].unsqueeze(-1) * exp_g[i_b].unsqueeze(-1)
            ).to(k.dtype).to(work)
            w_blk = torch.einsum("hij,jhd->ihd", a, kb)  # [l, H, K]
            w[i_b, idx] = w_blk.to(k.dtype)  # [sglang] :108 store cast
    return w, u
