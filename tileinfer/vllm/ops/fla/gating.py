"""Torch port of FLA's ``fused_sigmoid_gating_delta_rule_update`` (GDN decode-step gating + state
update).

Upstream: ``vllm/model_executor/layers/fla/ops/fused_sigmoid_gating.py`` at tag ``v0.23.0``, copied
verbatim into ``runtime/vllm-api/fla_fused_sigmoid_gating.py`` (279 lines; every line number below
refers to that local copy).  Upstream is one Triton kernel
(``fused_sigmoid_gating_delta_rule_update_kernel``) plus a launcher; there is no torch branch, so a
backend without Triton has no way to run GDN decode attention.  This module reproduces the
launcher's signature verbatim and the kernel's arithmetic and state semantics in plain torch.

The whole operator, per (sequence, value-head) pair
---------------------------------------------------
The kernel tiles the value dimension (``BV``) and the key dimension (``BK``); the recurrent state
``b_h`` is a ``[BV, BK]`` tile, i.e. ``h[v, k]``.  For every token ``t`` of the sequence:

    x        = a[t, hv] + dt_bias[hv]                                    # line 129
    softplus = where(beta*x <= threshold, (1/beta)*log(1+exp(beta*x)), x)  # lines 130-132
    g        = -exp(A_log[hv]) * softplus                                # line 133
    b_beta   = sigmoid(b[t, hv])                                         # line 136
    q        = q * scale                                                 # line 141, k NOT scaled
    h       *= exp(g)                                                    # line 144, decay FIRST
    v'       = (v - sum_k h[v,k]*k[k]) * b_beta                          # lines 148-149
    h       += v'[v] * k[k]                                              # line 151
    o[v]     = sum_k h_new[v,k] * q[k]                                   # line 153

Two things about that order are load-bearing and are easy to get backwards:

* the decay multiplies the state *before* the delta-rule correction is computed, so the correction
  is taken against the already-decayed state (``b_v -= tl.sum(b_h * b_k[...])`` runs after
  ``b_h *= tl.exp(b_g)``);
* the output reads the state *after* the update (``b_o`` is computed after ``b_h += ...``), so
  ``o_t`` already contains the token's own write.

``a`` also shadows the meaning of ``A_log``/``beta``: ``beta`` (default 1.0) and ``threshold``
(default 20.0) are the *softplus* parameters (lines 189-190, used at 130-132), not the delta-rule
beta.  The delta-rule beta is a *tensor* ``b`` passed through ``sigmoid`` (line 136).

Layouts
-------
* ``q``, ``k``: ``(B, T, H, K)`` — key/value head counts differ (GQA): ``i_h = i_hv // (HV // H)``
  (line 64).
* ``v``: ``(B, T, HV, V)``; ``o`` is allocated from ``v`` (line 225) and returned with the leading
  tile axis squeezed (line 278), i.e. ``(B, T, HV, V)``.
* ``a``, ``b``: **flattened token-major**, ``(B*T, HV)`` (``p_a = a + bos * HV + i_hv``, line 89,
  advanced by ``HV`` per token, line 178).  vLLM passes exactly this: ``a``/``b`` are
  ``(num_tokens, num_v_heads)`` after ``fused_post_conv_prep``.  ``(B, T, HV)`` contiguous aliases
  the same memory.
* ``A_log``, ``dt_bias``: ``(HV,)`` per head (lines 87, 90).
* recurrent state: ``(rows, HV, V, K)``, addressed as
  ``base + i_hv*V*K + v*K + k`` (lines 119, 165) — only the *leading* stride is configurable
  (``stride_init_state_token`` = ``stride(0)``, lines 231-232), the trailing three strides are
  hardcoded, so they must be contiguous.

Which row of the state a sequence starts from, and where the state is written back
----------------------------------------------------------------------------------
``h0``:
  * ``ssm_state_indices`` given ("continuous batching"): row =
    ``ssm_state_indices[i_n * stride_indices_seq + i_t]`` (lines 110-112) with
    ``i_t = num_accepted_tokens[i_n] - 1`` under speculative decoding, else ``i_t = 0`` (lines
    105-108).  ``row <= 0`` means ``NULL_BLOCK_ID``: the kernel **returns** and processes nothing
    for that sequence (lines 113-115).
  * otherwise: offset ``bos * HV * V * K`` (line 118), i.e. row ``bos`` — the token offset, not the
    sequence index.
``ht``, written once per token, so intermediate states are materialised:
  * ``inplace_final_state=True`` + indices: row = ``ssm_state_indices[... + i_t]`` for that token
    (lines 159-166), still skipping ``row <= 0``;
  * ``inplace_final_state=True`` without indices: ``final_state is initial_state`` (line 227) and
    the row is ``bos + i_t`` (line 168);
  * ``inplace_final_state=False``: a fresh ``(T, HV, V, K)`` buffer (line 229), row ``bos + i_t``
    (line 168).

The state is therefore updated **in place** whenever ``inplace_final_state=True`` (the default),
which is what vLLM relies on: the caller keeps ``ssm_state = self.kv_cache[1]`` across steps and
discards the returned tensor (``core_attn_out_decode, _ = ...``).

Verified against the source
---------------------------
V1 signature and defaults — lines 181-199 (copied verbatim below in the function).
V2 gate formula and evaluation order — lines 129-136, plus the identical standalone
   ``fused_gdn_gating_kernel`` in the caller
   (``runtime/vllm-api/vllm__model_executor__layers__mamba__gdn__qwen_gdn_linear_attn.py``, lines
   1781-1789), whose docstring states
   ``g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)`` (line 1805).
V3 decay-before-delta and output-from-updated-state — lines 143-154.
V4 ``scale`` default ``K ** -0.5``, positive-only assertion — lines 220-223.
V5 in-kernel q/k L2 normalisation: ``q * rsqrt(sum(q*q) + 1e-6)`` applied to *both* ``q`` and
   ``k`` before the ``scale`` multiply — lines 138-141.  The eps is added to the squared sum, not
   clamped onto the norm (same convention as ``tileinfer/vllm/ops/gdn.py::l2norm_fwd``).
V6 output allocation/dtype/return shape — lines 225, 154, 278; ``o`` inherits ``q.dtype``
   (``q.new_empty``), not ``v.dtype``.
V7 state layout and per-token writes — lines 116-120, 157-170; writes are cast to the *state*
   dtype (``b_h.to(p_ht.dtype.element_ty)``) while the recurrence itself stays in fp32 registers,
   i.e. an intermediate state written to a fp16 state buffer is **not** read back by the next token
   of the same call.
V8 ``ssm_state_indices`` semantics, incl. ``NULL_BLOCK_ID == 0`` skip and spec-decode
   ``num_accepted_tokens`` — lines 104-115, 157-166.
V9 fla is called by GDN with ``cu_seqlens`` + ``ssm_state_indices`` + ``use_qk_l2norm_in_kernel=
   True`` on the decode path (caller lines 1457-1474, 1484-1499, 1542-1558).

Assumed, not established by the source (flagged, not silently guessed)
----------------------------------------------------------------------
A1 ``a``/``b`` are 2-D token-major.  The kernel only demands "flat index ``tok*HV + hv``", so
   ``(B*T, HV)`` and ``(B, T, HV)`` contiguous both satisfy it; I accept any layout and reshape.
A2 ``HV >= H`` and ``HV % H == 0``.  ``i_h = i_hv // (HV // H)`` (line 64) divides by zero
   otherwise.  Guaranteed by GDN config, not asserted upstream (I raise ``ValueError``).
A3 The state's trailing ``(HV, V, K)`` dims are contiguous (V7/A2 above).  I raise ``ValueError``
   rather than reading wrong elements.
A4 ``ssm_state_indices`` columns are stride-1: the kernel computes ``stride_indices_tok`` (lines
   236-239) but never uses it, only ``+ i_t`` (lines 110, 160) — correct for the contiguous index
   tensors vLLM builds.  I index ``indices[n, i_t]`` and raise if ``stride(1) != 1``.
A5 ``cu_seqlens[-1] == T`` (total tokens); the kernel indexes ``o`` with ``all = T`` for the varlen
   path (line 70) without checking.  I raise ``ValueError`` if violated.
A6 An invalid slot (``row <= 0``) leaves ``o`` **uninitialised** upstream (``o = q.new_empty``,
   line 225).  I return zeros there: deterministic, and no test may depend on upstream's garbage.
A7 ``initial_state`` is mandatory in practice: line 231 dereferences ``initial_state.stride(0)``
   unconditionally, so the ``USE_INITIAL_STATE=False`` heuristic on line 17 is unreachable from
   Python.  I raise a clear ``ValueError`` instead of an ``AttributeError``.
A8 ``inplace_final_state=False`` builds a ``(T, HV, V, K)`` buffer (line 229) but writes rows
   ``bos + i_t`` (line 168), which only stays in bounds for ``B == 1``.  I keep the arithmetic and
   raise ``ValueError`` for ``B > 1`` instead of writing out of bounds.
A9 ``is_kda=True`` advances the ``a`` pointer by ``HV`` (line 178) although the per-token stride is
   ``HV*K`` (line 92): upstream's KDA branch only reads ``a`` correctly at ``T == 1``.  I index
   ``a[tok]`` on the documented ``(B*T, HV, K)`` layout, which agrees at ``T == 1`` and is the only
   reading that can work for ``T > 1``.  This is a deliberate deviation.
A10 dtype: the kernel does ``.to(tl.float32)`` on every load, so it has no fp64 mode.  I compute in
   fp32 for fp16/bf16/fp32 inputs (matching), and in fp64 when any input is fp64 (an extension for
   tests/oracles only).

Deliberate non-features: no Triton, no tiling (the kernel tiles ``V`` into ``BV <= 32`` blocks and
``K`` into ``BK = next_power_of_2(K)`` with ``NK == 1`` asserted on line 210; tiling is invisible in
the results), no CUDA-graph/fake-tensor plumbing.  Correctness first; the per-token python loop is
fine at decode (``T`` is one token per sequence in the serving path).
"""

from __future__ import annotations

from typing import Optional

import torch

__all__ = ["fused_sigmoid_gating_delta_rule_update"]

# Line 139 of the upstream kernel: ``tl.rsqrt(tl.sum(b_q * b_q) + 1e-6)``.
_L2NORM_EPS = 1e-6


def _working_dtype(*tensors: Optional[torch.Tensor]) -> torch.dtype:
    """fp64 only if an input is fp64, else fp32 (the kernel's ``.to(tl.float32)``, A10)."""
    for tensor in tensors:
        if tensor is not None and tensor.dtype == torch.float64:
            return torch.float64
    return torch.float32


def _sigmoid_gating_decay(
    a: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    r"""``g = -exp(A_log) * softplus_beta_threshold(a + dt_bias)`` — upstream lines 129-133.

    Quoted verbatim::

        x = tl.load(p_a).to(tl.float32) + tl.load(p_dt_bias).to(tl.float32)
        softplus_x = tl.where(
            beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x
        )
        b_g = -tl.exp(tl.load(p_A_log).to(tl.float32)) * softplus_x

    ``beta``/``threshold`` are the softplus knobs (``threshold=20`` is the usual "switch to the
    linear branch so exp() cannot overflow" cut).  The ``exp`` is inside ``log1p``; the trailing
    ``-exp(A_log) *`` makes ``g <= 0`` for every finite input, hence ``decay = exp(g) in (0, 1]``.
    """
    working = dtype or _working_dtype(a, A_log, dt_bias)
    x = a.to(working) + dt_bias.to(working)
    beta_x = beta * x
    # ``clamp`` only affects the branch that ``where`` discards, and keeps exp() finite; the
    # selected values are bit-identical to the kernel's formula.
    softplus_x = torch.where(
        beta_x <= threshold,
        torch.log1p(torch.exp(beta_x.clamp(max=threshold))) / beta,
        x,
    )
    return -torch.exp(A_log.to(working)) * softplus_x


def _check_state_layout(state: torch.Tensor, HV: int, V: int, K: int, name: str) -> None:
    if state.dim() != 4 or tuple(state.shape[-3:]) != (HV, V, K):
        raise ValueError(
            f"{name} must have shape (rows, HV, V, K) = (rows, {HV}, {V}, {K}), "
            f"got {tuple(state.shape)}"
        )
    if tuple(state.stride()[-3:]) != (V * K, K, 1):
        raise ValueError(
            f"{name} trailing (HV, V, K) dims must be contiguous: the kernel hardcodes "
            f"i_hv*V*K + v*K + k (lines 119/165) and only makes stride(0) configurable; "
            f"got strides {tuple(state.stride()[-3:])}"
        )


def _state_index(indices: torch.Tensor, n: int, i_t: int) -> int:
    """Row of the state tensor selected for sequence ``n`` at token ``i_t`` (A4)."""
    if indices.dim() == 1:
        return int(indices[n])
    if indices.stride(1) != 1:
        raise ValueError(
            "ssm_state_indices must have stride-1 columns: upstream computes "
            "stride_indices_tok but never applies it (lines 236-239 vs 110/160)"
        )
    return int(indices[n, i_t])


def fused_sigmoid_gating_delta_rule_update(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    inplace_final_state: bool = True,
    cu_seqlens: Optional[torch.Tensor] = None,
    ssm_state_indices: Optional[torch.Tensor] = None,
    num_accepted_tokens: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
    is_kda: bool = False,
):
    """Fused sigmoid gating + gated-delta-rule state update, one decode step (torch port).

    Signature is upstream's, verbatim (lines 181-199)::

        def fused_sigmoid_gating_delta_rule_update(
            A_log: torch.Tensor,
            a: torch.Tensor,
            b: torch.Tensor,
            dt_bias: torch.Tensor,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            beta: float = 1.0,
            threshold: float = 20.0,
            scale: float = None,
            initial_state: torch.Tensor = None,
            inplace_final_state: bool = True,
            cu_seqlens: torch.Tensor | None = None,
            ssm_state_indices: torch.Tensor | None = None,
            num_accepted_tokens: torch.Tensor | None = None,
            use_qk_l2norm_in_kernel: bool = False,
            is_kda: bool = False,
        ):

    Args:
        A_log: ``(HV,)`` log decay magnitude; the gate uses ``-exp(A_log)``.
        a: gate pre-activation, token-major flattened ``(num_tokens, HV)`` (``(num_tokens, HV, K)``
            when ``is_kda``).
        b: delta-rule beta pre-activation, same layout as ``a``; ``sigmoid(b)`` is the step size.
        dt_bias: ``(HV,)`` (``(HV, K)`` for ``is_kda``), added to ``a``.
        q: ``(B, T, H, K)``; unnormalised, unscaled queries.
        k: ``(B, T, H, K)``; unnormalised keys (never scaled).
        v: ``(B, T, HV, V)``; also supplies ``o``'s shape/dtype family.
        beta: softplus slope (NOT the delta-rule beta — that one is ``sigmoid(b)``).
        threshold: softplus linear-branch cut, ``beta*x > threshold`` switches to ``x``.
        scale: query scale; ``None`` -> ``K ** -0.5`` (lines 220-222), otherwise must be ``> 0``.
        initial_state: ``(rows, HV, V, K)`` recurrent state, **mandatory** (A7).  Mutated in place
            when ``inplace_final_state`` is True (the default, and what vLLM relies on).
        inplace_final_state: write the per-token state back into ``initial_state`` and return it
            (line 227), instead of filling a fresh ``(T, HV, V, K)`` buffer (line 229, A8).
        cu_seqlens: ``(N + 1,)`` cumulative token offsets for a flattened ragged batch; requires
            ``B == 1`` (lines 214-219).
        ssm_state_indices: ``(N,)`` or ``(N, T)`` paged-state slot ids; slot ``<= 0`` means
            "no state" and the kernel skips the sequence entirely (lines 113-115, 162-163).
        num_accepted_tokens: ``(N,)`` speculative-decoding accepted counts; the *initial* state is
            read from slot ``num_accepted_tokens[n] - 1`` (lines 105-108).
        use_qk_l2norm_in_kernel: normalise ``q`` and ``k`` per head before scaling (lines 138-141).
        is_kda: per-key-channel gate (``a``/``dt_bias`` carry a ``K`` axis, decay broadcasts over
            ``K``; lines 92-93, 146).  See A9 for the upstream ``T > 1`` pointer bug.

    Returns:
        ``(o, final_state)``: ``o`` is ``(B, T, HV, V)`` in ``q.dtype``; ``final_state`` is
        ``initial_state`` itself when ``inplace_final_state`` is True.
    """
    # Line 205: ``B, T, H, K, V = *k.shape, v.shape[-1]``.
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1

    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError(
            f"The batch size is expected to be 1 rather than {q.shape[0]}"
            f" when using `cu_seqlens`. Please flatten variable-length"
            f" inputs before processing."
        )
    if scale is None:
        scale = K**-0.5
    else:
        assert scale > 0, "scale must be positive"

    if initial_state is None:
        raise ValueError(
            "initial_state is required: upstream dereferences initial_state.stride(0) "
            "unconditionally (line 231), so the USE_INITIAL_STATE=False kernel path is "
            "unreachable from the python launcher"
        )
    _check_state_layout(initial_state, HV, V, K, "initial_state")
    if inplace_final_state:
        # Line 227: ``final_state = initial_state`` — the state is updated in place.
        final_state = initial_state
    else:
        # Line 229.
        final_state = q.new_empty(T, HV, V, K, dtype=initial_state.dtype)
        _check_state_layout(final_state, HV, V, K, "final_state")

    if HV < H or H <= 0 or HV % H:
        raise ValueError(
            f"expected HV % H == 0 and HV >= H (i_h = i_hv // (HV // H), line 64); got H={H}, "
            f"HV={HV}"
        )
    hv_to_h = torch.arange(HV, device=q.device) // (HV // H)

    if is_kda:
        a_tok = a.reshape(-1, HV, K)
        if dt_bias.numel() != HV * K:
            raise ValueError(f"is_kda=True expects dt_bias of shape ({HV}, {K})")
    else:
        a_tok = a.reshape(-1, HV)
        if dt_bias.numel() != HV:
            raise ValueError(f"dt_bias must have {HV} elements (line 90), got {dt_bias.numel()}")
    b_tok = b.reshape(-1, HV)
    total_tokens = B * T  # line 74: ``all = B * T``; with cu_seqlens B == 1 and all == T
    if a_tok.shape[0] != total_tokens or b_tok.shape[0] != total_tokens:
        raise ValueError(
            f"`a`/`b` are addressed as a[tok*HV + hv] (lines 89/95): expected {total_tokens} "
            f"tokens, got a={a_tok.shape[0]}, b={b_tok.shape[0]}"
        )

    if cu_seqlens is not None and int(cu_seqlens[-1]) != total_tokens:
        raise ValueError(
            f"cu_seqlens[-1] must equal the flattened token count ({total_tokens}), "
            f"got {int(cu_seqlens[-1])}"
        )
    if not inplace_final_state and B > 1:
        raise ValueError(
            "inplace_final_state=False writes rows bos + t into a (T, HV, V, K) buffer (lines "
            "229/168), which only stays in bounds for B == 1"
        )
    if ssm_state_indices is None:
        # The kernel reads h0 at row ``bos`` (line 118) and, when in place, writes rows
        # ``bos + i_t`` (line 168) into the same tensor; with cu_seqlens both are token offsets.
        needed = total_tokens if inplace_final_state else (B - 1) * T + 1
        if initial_state.shape[0] < needed:
            raise ValueError(
                f"without ssm_state_indices the state row is the flattened token offset bos + t "
                f"(lines 118/168): initial_state needs >= {needed} rows, got "
                f"{initial_state.shape[0]}; pass ssm_state_indices (the vLLM decode path) or a "
                f"per-token state buffer"
            )

    working = _working_dtype(q, k, v, a, b, A_log, dt_bias, initial_state)
    q_flat = q.to(working).reshape(B * T, H, K)
    k_flat = k.to(working).reshape(B * T, H, K)
    v_flat = v.to(working).reshape(B * T, HV, V)
    A_log_w = A_log.reshape(-1).to(working)
    # Line 90 / 93: ``dt_bias`` is per head, or per (head, key-channel) under KDA.
    dt_bias_w = dt_bias.reshape(HV, K).to(working) if is_kda else dt_bias.reshape(HV).to(working)
    a_w = a_tok.to(working)
    b_w = b_tok.to(working)

    # Lines 225/278: ``o = q.new_empty(NK, *v.shape)`` then ``o.squeeze(0)``; NK == 1 is asserted
    # upstream (line 210).  zeros() so an invalid state slot leaves a defined value (A6).
    o = torch.zeros(B, T, HV, V, dtype=q.dtype, device=q.device)
    o_flat = o.view(B * T, HV, V)

    for n in range(N):
        if cu_seqlens is not None:
            bos, eos = int(cu_seqlens[n]), int(cu_seqlens[n + 1])
        else:
            # Lines 72-73.
            bos, eos = n * T, n * T + T
        if eos < bos:
            raise ValueError(f"sequence {n}: cu_seqlens must be non-decreasing")
        if eos > total_tokens:
            raise ValueError(f"sequence {n}: token range [{bos}, {eos}) exceeds {total_tokens}")
        if eos == bos:
            # Lines 76-78.
            continue

        # ---- initial state (lines 103-120) ------------------------------------------------
        if ssm_state_indices is not None:
            i_t = (
                int(num_accepted_tokens[n]) - 1
                if num_accepted_tokens is not None
                else 0  # lines 105-108
            )
            init_row = _state_index(ssm_state_indices, n, i_t)
            if init_row <= 0:
                # NULL_BLOCK_ID: the kernel returns before writing anything (lines 113-115).
                continue
            h = initial_state[init_row].to(working)
        else:
            # Line 118: ``p_h0 = h0 + bos * HV * V * K`` — a *token* offset (A4/A8).
            init_row = bos
            h = initial_state[init_row].to(working)

        # ---- recurrent update, one token at a time (lines 122-170) ------------------------
        for i_t in range(eos - bos):
            tok = bos + i_t
            # Line 129-133: gate -> log decay.  For KDA the gate is per key channel, so the
            # per-head ``A_log`` scalar is broadcast over the K axis (lines 133 vs 146).
            a_log_term = A_log_w[:, None] if is_kda else A_log_w
            g = _sigmoid_gating_decay(a_w[tok], a_log_term, dt_bias_w, beta, threshold, working)
            decay = torch.exp(g)
            # Line 136.
            beta_t = torch.sigmoid(b_w[tok])
            # Lines 83-85: q/k/v rows for this token; q,k gather the GQA-mapped key head.
            q_t = q_flat[tok].index_select(0, hv_to_h)
            k_t = k_flat[tok].index_select(0, hv_to_h)
            v_t = v_flat[tok]
            if use_qk_l2norm_in_kernel:
                # Lines 138-140.
                q_t = q_t * torch.rsqrt((q_t * q_t).sum(-1, keepdim=True) + _L2NORM_EPS)
                k_t = k_t * torch.rsqrt((k_t * k_t).sum(-1, keepdim=True) + _L2NORM_EPS)
            # Line 141.
            q_t = q_t * scale
            # Lines 143-146: decay the state FIRST.
            if is_kda:
                h = h * decay[:, None, :]
            else:
                h = h * decay[:, None, None]
            # Lines 148-149: delta rule, scaled by sigmoid(b).
            hk = (h * k_t[:, None, :]).sum(-1)
            delta = (v_t - hk) * beta_t[:, None]
            # Line 151.
            h = h + delta[:, :, None] * k_t[:, None, :]
            # Lines 153-154: output from the UPDATED state, q already scaled.
            o_flat[tok] = (h * q_t[:, None, :]).sum(-1).to(o.dtype)

            # ---- state write-back (lines 156-170) ----------------------------------------
            if ssm_state_indices is not None and inplace_final_state:
                final_row = _state_index(ssm_state_indices, n, i_t)
                if final_row > 0:
                    final_state[final_row] = h.to(final_state.dtype)
            else:
                final_state[bos + i_t] = h.to(final_state.dtype)

    return o, final_state
