"""Composed GDN chunked-prefill chain - the six-step FLA pipeline wired into one call.

Role
----
``chunk_gated_delta_rule`` is what a GDN layer's prefill path calls
[vLLM ``qwen_gdn_linear_attn.py:1529`` passes ``use_qk_l2norm_in_kernel=False`` and calls it];
``chunk_gated_delta_rule_fwd`` is the function that actually wires the six operators together
[vLLM ``chunk.py:23-86``].  Every operator in that sequence is already ported into this package
(``gdn.py``, ``kkt.py``, ``solve_tril.py``, ``wu.py``, ``delta_h.py``, ``chunk_o.py``); **this module
adds no math**.  It reproduces the call order, the dtype each binding must carry, the rebinding of
two tensor names (``A``, ``v``) that is the chain's main silent-failure surface, and the validation
the real vLLM caller performs before entering the chain.

The chain (exact call sequence, [vLLM] ``chunk.py:37-86``)
---------------------------------------------------------
====================  ==============  ====================================================  ============================
step                  [vLLM] lines    call                                                  output it introduces
====================  ==============  ====================================================  ============================
0 (not counted)       ``:37-39``      ``g = chunk_local_cumsum(g, chunk_size=64, ...)``     ``g``: chunk-local cumulative
1                     ``:41-48``      ``A = chunk_scaled_dot_kkt_fwd(k, beta, g, ...)``      ``A``: strict-lower KKT, fp32
2                     ``:49-51``      ``A = solve_tril(A, ..., output_dtype=k.dtype)``       ``A``: ``(I + A)^{-1}``, bf16
3                     ``:52-60``      ``w, u = recompute_w_u_fwd(k, v, beta, A, g_cumsum=g)`` ``w``, ``u`` (WY form)
4                     ``:61-71``      ``h, v_new, final_state = ..._fwd_h(k, w, u, g, ...)`` ``h``, ``v_new``, ``final_state``
5                     ``:72-82``      ``o = chunk_fwd_o(q, k, v=v_new, h=h, g, scale, ...)``  ``o``
====================  ==============  ====================================================  ============================

Steps 0-5 in one arrow::

    chunk_local_cumsum -> chunk_scaled_dot_kkt_fwd -> solve_tril -> recompute_w_u_fwd
                       -> chunk_gated_delta_rule_fwd_h -> chunk_fwd_o

[vLLM] (the step order, the keyword arguments and the two ``output_dtype`` rewrites are all read
from ``chunk.py:37-82``; that file is the real vLLM 0.23 caller and is quoted, not reconstructed).

Chain wiring: which binding is rebound where
--------------------------------------------
The table below is the point of this module.  ``A`` and ``v`` are *rebound* mid-chain and the value
that survives is not the value the name suggests.

=========================  ==========================================================  =====================  ===========================
binding                    value                                                       dtype                  consumed by
=========================  ==========================================================  =====================  ===========================
``g``                      chunk-local cumsum of the caller's raw log-decay            fp32                   4 sites: kkt ``:44``,
                                                                                                              wu ``:57``, delta_h
                                                                                                              ``:65``, chunk_o ``:76``
``A`` after step 1         ``beta_i * safe_exp(g_i-g_j) * (k_i.k_j)``, ``i>j``          fp32 (pinned by         ``solve_tril`` only
                           inside each 64-token chunk                                  [vLLM] ``:47``)         [vLLM] ``:50``
``A`` after step 2         ``(I + A_kkt)^{-1}``, same name, **different tensor**        **rebound to k.dtype**  wu only [vLLM] ``:56``
                                                                                       = bf16 [vLLM] ``:50``
``w`` / ``u``              ``A_inv @ (k*beta*exp(g))`` / ``A_inv @ (v*beta)``           ``k.dtype`` /           delta_h [vLLM] ``:63-64``
                                                                                       ``v.dtype``
``h``                      per-chunk **entry** states ``[B, NT, H, V, K]``              ``k.dtype`` (bf16)     chunk_o [vLLM] ``:76``
``v_new``                  **ungated** ``u - w @ h[c]^T``                                ``v.dtype``            chunk_o's **``v``** arg
                                                                                                              [vLLM] ``:75``
``v`` -> ``v_new``         the name ``v`` inside chunk_o is a REBINDING of the           ``v.dtype``            chunk_o's intra-chunk dot
                           caller's ``v``; the caller's values never reach chunk_o      = u.dtype              [sglang] chunk_o:122
``o``                      ``scale * (exp(g)*q@h^T + tril(q@k^T * safe_exp(dg)) @ v_new)`` ``v.dtype`` then      returned as ``o.to(q.dtype)``
                                                                                       ``.to(q.dtype)``       [vLLM] ``:134``
=========================  ==========================================================  =====================  ===========================

``A`` **fp32 -> bf16** is the load-bearing dtype rewrite: ``solve_tril`` is told
``output_dtype=k.dtype`` [vLLM] ``chunk.py:50``, so ``w``/``u`` are computed from a bf16 ``A_inv``
even when the KKT matrix was fp32 [vLLM] ``chunk.py:47``.  ``v -> v_new`` is the load-bearing value
rewrite: ``chunk_fwd_o``'s ``v`` argument is ``v_new`` [vLLM] ``chunk.py:75``; ``v_new`` is stored
**ungated** by ``delta_h`` [sglang] ``chunk_delta_h.py:172-176`` (the gate ``safe_exp(g_last - g)``
only multiplies the copy used for the state update [sglang] ``chunk_delta_h.py:185``), because
``chunk_fwd_o`` applies ``exp(g_i - g_j)`` itself [sglang] ``chunk_o.py:106`` and a gated ``v_new``
would be decayed twice.

Return value
------------
``chunk_gated_delta_rule_fwd`` returns the **debugging** form of the vLLM tuple,
``(g, o, A, final_state, w, h, v_new)`` [vLLM] ``chunk.py:83-86`` (the ``SUPPRESS_LEVEL >= 3``
branch).  The production vLLM default returns ``(g, o, A, final_state, None, None, None)`` [vLLM]
``chunk.py:83-84``; the only difference is visibility of the intermediates, not math.  The public
``chunk_gated_delta_rule`` returns ``(o, final_state)`` exactly like vLLM [vLLM] ``chunk.py:134,245``.

Validation and guards
---------------------
* ``q.dtype == k.dtype == v.dtype`` [vLLM] ``chunk.py:212``; ``beta`` must be 3-D [vLLM]
  ``chunk.py:216`` (both re-checked by the leaf ports).
* ``cu_seqlens`` non-empty -> ``B == 1`` [vLLM] ``chunk.py:217-222`` and
  ``initial_state.shape[0] == len(cu_seqlens) - 1`` [vLLM] ``chunk.py:223-227``.  The error strings
  are the vLLM ones.
* ``scale is None`` -> ``k.shape[-1] ** -0.5`` [vLLM] ``chunk.py:228-229`` (``k``, not ``q``).
* ``use_qk_l2norm_in_kernel=True`` -> ``q, k = l2norm_fwd(q), l2norm_fwd(k)`` **before** the chain
  [vLLM] ``chunk.py:109-111``; it is a rebinding and never mutates the caller's tensors [vLLM].
* **``A.triu() == 0``**: ``solve_tril``'s host contract requires the within-chunk transition matrix
  to be strictly lower triangular ("A should be strictly lower triangular, i.e., ``A.triu() == 0``",
  [vLLM] ``fla_solve_tril.py:515``).  vLLM only *documents* it; the Triton kernel silently discards
  the diagonal (contract §5.2), so a violation there is invisible.  This composed entry checks it
  explicitly with :func:`assert_strict_lower_chunks` between steps 1 and 2 - a broken ``kkt`` (say
  a ``>=`` instead of ``>`` mask) then fails loudly instead of poisoning ``w``/``u``/``h``/``o``.
  The check is one ``triu`` pass over ``A`` and is block-local (per chunk, per sequence), not a
  global ``triu`` of a ``[B, T, H, BT]`` tensor.
* ``chunk_indices`` / ``chunk_offsets``: vLLM passes both explicitly [vLLM] ``chunk.py:46,59,69,80``
  because its launcher computes them once.  This entry omits them from the signature and lets each
  operator derive them from ``cu_seqlens`` - ``solve_tril``/``kkt``/``wu``/``chunk_o`` derive
  ``chunk_indices`` the same way (contract §4.2), and ``delta_h`` derives and validates
  ``chunk_offsets`` (contract §4.3); both use the constant ``BT = 64``, so the derivations agree by
  construction.
* ``core_attn_out`` (vLLM's inference-only output buffer, [vLLM] ``chunk.py:129-133``) is not
  exposed here: pass the buffer directly to ``chunk_fwd_o`` if a caller needs it.  Its write range
  is unestablished (contract §7.6, ``chunk_o.py`` docstring).

Deliberate deviations from the Triton path
------------------------------------------
* **fp64 promotion of the KKT output dtype.**  vLLM pins ``output_dtype=torch.float32`` [vLLM]
  ``chunk.py:47``; that is what this entry uses for fp16/bf16/fp32 inputs (production dtype).
  For an fp64 call the whole chain is promoted to fp64 (``_working_dtype``, like every other port in
  this package), otherwise the fp32 ``A`` would cap the chain at ~1e-7 and an fp64 call could not be
  exact against an fp64 oracle.  [推断: same convention as ``kkt.py``/``solve_tril.py``/``wu.py``.]
* **fp32 is accepted.**  vLLM rejects fp32 inputs because its kernels are bf16 [vLLM]
  ``chunk.py:213-215`` ("ChunkGatedDeltaRuleFunction does not support float32"); the torch port has
  no such limitation and uses the dtype rule above, so fp32 and fp64 calls are legal here.  The
  equal-dtype guard is kept.
* **No autograd, no ``@input_guard``/``torch.compiler.disable``/``custom_fwd``** - the port is
  inference-only like the rest of the chain, and those decorators carry no math.
* **No ``SUPPRESS_LEVEL`` switch.**  The debug tuple is always returned (see "Return value").

Not claimed
-----------
Bit-exactness against Triton (``exp`` and matmul ordering differ, the tile splits are invisible,
``FLA_USE_FAST_OPS`` is not modelled) - the same caveat as the six operator modules.

Verified vs assumed
-------------------
Verified against the real vLLM 0.23 caller: the step order and every keyword argument
(``chunk.py:37-82``), the two ``output_dtype`` rewrites (``:47``, ``:50``), ``v=v_new`` (``:75``),
the 7-tuple debug return (``:83-86``), the outer validation and the ``scale`` default
(``:212-229``), the l2norm rebinding (``:109-111``) and the final ``o.to(q.dtype)`` (``:134``).

Reconstructed / inferred: that the KKT output dtype promotion for fp64 is safe to make (``[推断]``,
justified above), and everything quoted from the sglang sibling (``chunk_delta_h.py``,
``chunk_o.py``) which is the offline kernel-body reference (contract §0).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from ..gdn import chunk_local_cumsum, l2norm_fwd
from .chunk_o import chunk_fwd_o
from .delta_h import _sequence_rows, chunk_gated_delta_rule_fwd_h
from .gating import _working_dtype
from .kkt import chunk_scaled_dot_kkt_fwd
from .solve_tril import solve_tril
from .wu import recompute_w_u_fwd

__all__ = [
    "FLA_CHUNK_SIZE",
    "assert_strict_lower_chunks",
    "chunk_gated_delta_rule",
    "chunk_gated_delta_rule_fwd",
]

#: ``BT`` for every kernel in the chain.  vLLM 0.23 pins it to 64 and never shrinks it for short
#: sequences [vLLM] ``qwen_gdn_linear_attn.py:1085-1087`` ("All kernels including
#: ``chunk_fwd_kernel_o`` now use a fixed ``BT = chunk_size`` (64)"); the sibling's
#: ``CHUNK_SIZE = 64`` [sglang] ``chunk_delta_h.py:19`` and the default ``chunk_size=64`` of
#: [sglang] ``chunk_scaled_dot_kkt.py:94`` agree.  ``T < 64`` is a legal single partial chunk.
FLA_CHUNK_SIZE = 64


def _chunk_cumsum(g: torch.Tensor, cu_seqlens: Optional[torch.Tensor]) -> torch.Tensor:
    """Step 0: ``g = chunk_local_cumsum(g, chunk_size=64, cu_seqlens=cu_seqlens)``.

    [vLLM] ``chunk.py:37-39`` calls ``chunk_local_cumsum`` without ``output_dtype``; the default is
    ``torch.float`` [vLLM] ``fla_cumsum.py:250`` and the kernel loads in fp32 before the cumsum
    (``b_s = tl.load(p_s, ...).to(tl.float32)``, ``b_o = tl.cumsum(b_s, axis=0)`` - [vLLM]
    ``fla_cumsum.py:66-67``), so the caller's raw log-decay (bf16 or fp32) becomes an **fp32**
    chunk-local cumulative sum.  The ported ``chunk_local_cumsum`` in ``gdn.py`` is token-major
    (``dim 0`` is the token axis), hence the per-batch-row loop / the ``cu_seqlens`` branch that
    packs a ``[B, T, H]`` tensor into the port's ``[T, H]`` form.  fp64 input stays fp64 so an fp64
    call is exact against an fp64 oracle [推断: the package-wide dtype convention, deviation from
    Triton which would downcast].
    """
    if g.dim() != 3:
        raise ValueError(f"g must be [B, T, H], got {tuple(g.shape)}")
    work = torch.float64 if g.dtype == torch.float64 else torch.float32
    gw = g.to(work)
    if cu_seqlens is not None:
        if gw.shape[0] != 1:
            # [vLLM] chunk.py:217-222, and fla_cumsum.py:253-256 asserts the same.
            raise ValueError(
                f"The batch size is expected to be 1 rather than {gw.shape[0]} when using"
                f" `cu_seqlens`. Please flatten variable-length inputs before processing."
            )
        return chunk_local_cumsum(gw[0], FLA_CHUNK_SIZE, cu_seqlens=cu_seqlens).unsqueeze(0)
    rows = [chunk_local_cumsum(gw[i_b], FLA_CHUNK_SIZE) for i_b in range(gw.shape[0])]
    return torch.stack(rows, dim=0) if rows else gw.clone()


def assert_strict_lower_chunks(
    A: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor] = None,
    chunk_size: int = FLA_CHUNK_SIZE,
) -> None:
    """Raise ``ValueError`` unless ``A`` is strictly lower triangular *inside every chunk*.

    This is the machine-checkable form of [vLLM] ``fla_solve_tril.py:515`` ("A should be strictly
    lower triangular, i.e., ``A.triu() == 0``").  ``A`` is ``[B, T, H, BT]`` whose last axis is the
    *within-chunk* index, so the invariant is per 64-token block: row ``t`` of a chunk starting at
    ``t0`` may only be non-zero in columns ``< t - t0``.  A global ``A.triu()`` is the wrong test -
    it would inspect rows ``>= BT`` against columns ``< BT`` and reject legal blocks.

    The ported ``kkt`` guarantees this (strict ``>`` mask [sglang] ``chunk_scaled_dot_kkt.py:82``);
    ``solve_tril`` would silently discard a diagonal/upper entry and corrupt everything downstream
    (contract §5.2), which is why the composed chain calls this between steps 1 and 2.

    Args:
        A: ``[B, T, H, BT]`` transition matrix (the output of ``chunk_scaled_dot_kkt_fwd``).
        cu_seqlens: optional ``[N + 1]`` packed-sequence offsets; chunk boundaries restart at every
            sequence start, exactly as in the kernels (contract §4.1).
        chunk_size: ``BT``; must equal ``A.shape[-1]``.

    Returns:
        ``None``.
    """
    if A.dim() != 4:
        raise ValueError(f"A must be [B, T, H, BT], got {tuple(A.shape)}")
    B, T, _H, BT = A.shape
    if BT != chunk_size:
        raise ValueError(f"A.shape[-1] = {BT} must equal chunk_size = {chunk_size}")
    for i_b, _i_n, bos, eos in _sequence_rows(B, T, cu_seqlens):
        length = eos - bos
        for i_t in range(-(-length // BT)):
            t0 = bos + i_t * BT
            block_len = min(BT, length - i_t * BT)
            if block_len <= 0:  # empty sequence segment: nothing to check
                continue
            # [block_len, H, block_len] -> [H, block_len, block_len]; triu keeps the diagonal and
            # above, which must be exactly zero.
            block = A[i_b, t0 : t0 + block_len, :, :block_len].permute(1, 0, 2)
            if int(torch.count_nonzero(torch.triu(block, diagonal=0))):
                raise ValueError(
                    f"A.triu() == 0 violated at batch row {i_b}, chunk {i_t} (tokens "
                    f"{t0}..{t0 + block_len - 1}): the within-chunk transition matrix must be "
                    f"strictly lower triangular [vLLM fla_solve_tril.py:515].  solve_tril would "
                    f"silently discard the offending entries (contract §5.2)."
                )


def chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: Optional[float],
    initial_state: Optional[torch.Tensor],
    output_final_state: bool,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> Tuple[
    torch.Tensor,  # g: fp32 chunk-local cumsum
    torch.Tensor,  # o: [B, T, H, V]
    torch.Tensor,  # A: (I + A_kkt)^{-1}, k.dtype
    Optional[torch.Tensor],  # final_state: [N, H, V, K] or None
    torch.Tensor,  # w: [B, T, H, K], k.dtype
    torch.Tensor,  # h: [B, NT, H, V, K], k.dtype
    torch.Tensor,  # v_new: [B, T, H, V], v.dtype
]:
    """The exact six-step chain of [vLLM] ``chunk.py:37-86``; returns its debug tuple.

    Args:
        q: ``[B, T, Hg, K]`` queries.
        k: ``[B, T, Hg, K]`` keys.
        v: ``[B, T, H, V]`` values (the **original** values; the chain replaces them with
            ``v_new`` only at the ``chunk_o`` call, [vLLM] ``chunk.py:75``).
        g: ``[B, T, H]`` raw (per-token, log-space) forget gate.  It is cumsum'ed inside the chain
            [vLLM] ``chunk.py:37-39``; do not pre-accumulate it.
        beta: ``[B, T, H]`` delta-rule step sizes.
        scale: float; ``None`` means ``k.shape[-1] ** -0.5`` [vLLM] ``chunk.py:228-229``.
        initial_state: ``[N, H, V, K]`` entry states, ``N = B`` (dense) or ``len(cu_seqlens) - 1``
            (ragged), or ``None`` for a zero state [vLLM] ``chunk.py:168-171``.
        output_final_state: return ``final_state``; ``False`` -> ``None`` [vLLM] ``chunk.py:172-173``.
        cu_seqlens: ``[N + 1]`` cumulative token lengths of a packed ``[1, T, ...]`` batch, or
            ``None`` for the dense layout [vLLM] ``chunk.py:174-176``.

    Returns:
        ``(g, o, A, final_state, w, h, v_new)`` [vLLM] ``chunk.py:83-86``:
        ``g`` is the fp32 chunk-local cumsum, ``o`` is ``[B, T, H, V]`` in ``v.dtype``, ``A`` is the
        **bf16** inverse produced by step 2, ``w``/``h`` are in ``k.dtype`` and ``v_new`` in
        ``v.dtype``.  ``final_state`` is ``[N, H, V, K]`` or ``None``.
    """
    # step 0 [vLLM] chunk.py:37-39
    g = _chunk_cumsum(g, cu_seqlens)
    # The KKT matrix is fp32 in production [vLLM] chunk.py:47; fp64 only when the call itself is
    # fp64, so that "fp64" means exact against a fp64 oracle (see module docstring).
    a_dtype = _working_dtype(q, k, v, g, beta, initial_state)
    # step 1 [vLLM] chunk.py:41-48
    A = chunk_scaled_dot_kkt_fwd(
        k=k,
        beta=beta,
        g=g,
        cu_seqlens=cu_seqlens,
        output_dtype=a_dtype,
    )
    # The host contract solve_tril relies on [vLLM] fla_solve_tril.py:515; checked here because the
    # kernels would silently drop a violation (contract §5.2).
    assert_strict_lower_chunks(A, cu_seqlens)
    # step 2 [vLLM] chunk.py:49-51 -- `A` is REBOUND: fp32 KKT -> (I+A)^-1 in k.dtype (bf16).
    A = solve_tril(
        A=A,
        cu_seqlens=cu_seqlens,
        output_dtype=k.dtype,
    )
    # step 3 [vLLM] chunk.py:52-60 -- the gate is named `g_cumsum` here and `g` in kkt: one tensor,
    # two names (contract §2.2).
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g_cumsum=g,
        cu_seqlens=cu_seqlens,
    )
    # step 4 [vLLM] chunk.py:61-71 -- returns 3 values in vLLM 0.23 (the sibling returns 2).
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )
    # step 5 [vLLM] chunk.py:72-82 -- `v=v_new`, NEVER the caller's `v` [vLLM] chunk.py:75.
    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )
    # [vLLM] chunk.py:83-86, SUPPRESS_LEVEL >= 3 branch.
    return g, o, A, final_state, w, h, v_new


def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Public prefill entry point: chunked gated delta rule, returning ``(o, final_state)``.

    This is the torch equivalent of [vLLM] ``chunk.py:137-245``.  It performs the outer validation,
    the optional in-kernel L2 normalisation and the ``scale`` default, then runs the composed chain
    and casts the output back to ``q.dtype`` exactly as vLLM does [vLLM] ``chunk.py:134``.

    Args:
        q: ``[B, T, Hg, K]`` queries.
        k: ``[B, T, Hg, K]`` keys.
        v: ``[B, T, H, V]`` values.  ``q``, ``k`` and ``v`` must share one dtype [vLLM]
            ``chunk.py:212``; bf16 in production, but fp32/fp64 are accepted here (see the module
            docstring: vLLM's fp32 rejection is a Triton limitation, not a math one).
        g: ``[B, T, H]`` raw log-space forget gate (chunk-local cumsum is done inside the chain).
        beta: ``[B, T, H]`` delta-rule step sizes; must be 3-D [vLLM] ``chunk.py:216``.
        scale: optional float, default ``k.shape[-1] ** -0.5`` [vLLM] ``chunk.py:228-229``.
        initial_state: optional ``[N, H, V, K]`` entry state [vLLM] ``chunk.py:168-171``.
        output_final_state: whether to return the final state [vLLM] ``chunk.py:172-173``;
            ``False`` gives ``None``.
        cu_seqlens: optional ``[N + 1]`` packed-sequence offsets [vLLM] ``chunk.py:174-176``.
        use_qk_l2norm_in_kernel: normalise ``q``/``k`` in fp32 with ``l2norm_fwd`` before the chain
            [vLLM] ``chunk.py:109-111``.  It rebinds the locals and never mutates the caller's
            tensors; production prefill passes ``False`` [vLLM] ``qwen_gdn_linear_attn.py:1529``.

    Returns:
        ``(o, final_state)``: ``o`` is ``[B, T, H, V]`` with ``q.dtype`` [vLLM] ``chunk.py:134``;
        ``final_state`` is ``[N, H, V, K]`` or ``None``.
    """
    # [vLLM] chunk.py:212-215: one shared dtype for the three token tensors.  The fp32 rejection
    # of the Triton kernel is deliberately not reproduced (module docstring).
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError(
            f"q, k and v must share one dtype [vLLM chunk.py:212]; got "
            f"{q.dtype}, {k.dtype} and {v.dtype}."
        )
    if len(beta.shape) != 3:  # [vLLM] chunk.py:216
        raise ValueError("beta must be of shape [B, T, H].")
    if cu_seqlens is not None:
        # [vLLM] chunk.py:217-227, same wording as the source.
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using"
                f" `cu_seqlens`. Please flatten variable-length inputs before processing."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input"
                f" sequences, i.e., {len(cu_seqlens) - 1} rather than"
                f" {initial_state.shape[0]}."
            )
    if scale is None:  # [vLLM] chunk.py:228-229: K from `k`, not from q or v.
        scale = k.shape[-1] ** -0.5
    if use_qk_l2norm_in_kernel:  # [vLLM] chunk.py:109-111 (rebinding, not in-place)
        q = l2norm_fwd(q)
        k = l2norm_fwd(k)

    _g, o, _A, final_state, _w, _h, _v_new = chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )
    return o.to(q.dtype), final_state  # [vLLM] chunk.py:134
