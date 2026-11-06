"""Torch port of FLA's fused recurrent gated-delta-rule operators (GDN sequence + packed decode).

Upstream: ``vllm/model_executor/layers/fla/ops/fused_recurrent.py`` at tag ``v0.23.0``, copied
verbatim into ``runtime/vllm-api/fla_fused_recurrent.py`` (619 lines; every line number below
refers to that local copy).  Upstream is pure Triton -- one kernel plus a launcher per entry point,
with no torch branch -- so a backend without Triton has no way to run either operator.  This module
reproduces both launchers' signatures verbatim and both kernels' arithmetic and state semantics in
plain torch.

Two *different* entry points, not one operator
----------------------------------------------
* ``fused_recurrent_gated_delta_rule`` (lines 516-619) -- the **sequence form**.  It walks ``T``
  tokens per sequence; the gate ``g`` and the delta-rule step size ``beta`` are *inputs*.
* ``fused_recurrent_gated_delta_rule_packed_decode`` (lines 339-478) -- the **decode form**.  One
  token per sequence, ``q|k|v`` packed into one ``mixed_qkv`` row, and the gate *derived inside the
  kernel* from the per-token pair ``a``/``b`` plus the per-head ``A_log``/``dt_bias``.  Its grid is
  ``(NV, B * HV)`` (line 449) with no token loop anywhere in the kernel, so it is inherently a
  single-step operator: ``T > 1`` means calling it ``T`` times and carrying the state.

They therefore share the recurrence and nothing else; the decode form's ABI (``mixed_qkv``, ``a``,
``b``, ``A_log``, ``dt_bias``, caller-allocated ``out``, 1-D ``ssm_state_indices``) cannot be
expressed by the sequence form without a gate pre-pass and a repack, which is exactly what the test
suite does to prove the two agree (``tests/test_gdn_recurrent.py``).

The recurrence, quoted
----------------------
Sequence form, lines 122-149 (the token loop body)::

    b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
    b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
    b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
    ...
    b_q = b_q * scale                                     # 130: q scaled, k NEVER scaled
    if not IS_KDA:
        b_g = tl.load(p_g).to(tl.float32)
        b_h *= exp(b_g)                                   # 134: decay the state FIRST
    b_v -= tl.sum(b_h * b_k[None, :], 1)                  # 139: delta vs the decayed state
    ...
    b_v *= b_beta                                         # 144
    b_h += b_v[:, None] * b_k[None, :]                    # 146: rank-1 write
    b_o = tl.sum(b_h * b_q[None, :], 1)                   # 148: output AFTER the write

Decode form, lines 327-332, is those same four statements verbatim with ``g_val``/``beta_val``::

    b_h *= exp(g_val)
    b_v -= tl.sum(b_h * b_k[None, :], 1)
    b_v *= beta_val
    b_h += b_v[:, None] * b_k[None, :]
    b_o = tl.sum(b_h * b_q[None, :], 1)

Two orderings are load-bearing and easy to get backwards: the decay multiplies the state *before*
the delta correction is computed, and the output reads the state *after* the write (so ``o_t``
already contains token ``t``'s own rank-1 update).  ``beta`` in the sequence form is the delta-rule
step size, while in the decode form it is ``sigmoid(b)`` -- ``a``/``A_log``/``dt_bias`` are the
*gate*, never the step size.

Decode gate, quoting lines 318-325::

    a_val = tl.load(a + i_n * stride_a_tok + i_hv).to(tl.float32)
    b_val = tl.load(b + i_n * stride_b_tok + i_hv).to(tl.float32)
    A_log_val = tl.load(A_log + i_hv).to(tl.float32)
    dt_bias_val = tl.load(dt_bias + i_hv).to(tl.float32)
    x = a_val + dt_bias_val
    softplus_x = tl.where(x <= SOFTPLUS_THRESHOLD, tl.log(1.0 + tl.exp(x)), x)
    g_val = -tl.exp(A_log_val) * softplus_x
    beta_val = tl.sigmoid(b_val).to(b.dtype.element_ty).to(tl.float32)

i.e. ``softplus(x)`` (linear above 20.0) -> negate -> scale by ``exp(A_log)`` -> ``exp`` inside the
recurrence, hence ``decay in (0, 1]``; the step is ``sigmoid(b)``.  ``SOFTPLUS_THRESHOLD`` is
``20.0`` (line 473).  This is the *same* gate as the GDN prefill/update operator in
``tileinfer/vllm/ops/fla/gating.py`` (its ``_sigmoid_gating_decay``), so it is re-used here rather
than re-derived: one convention, one place to fix it.

Packed layout, quoting lines 305-311::

    p_mixed = mixed_qkv + i_n * stride_mixed_qkv_tok
    q_off = i_h * K + o_k
    k_off = (H * K) + i_h * K + o_k
    v_off = (2 * H * K) + i_hv * V + o_v

so a row is ``[q: H*K][k: H*K][v: HV*V]`` with each block itself head-major ``(heads, head_dim)``;
``H`` is *inferred* from the row width (lines 417-430: ``qk_dim = qkv_dim - HV*V``,
``H = (qk_dim // 2) // K``), and the GQA map is ``i_h = i_hv // (HV // H)`` (line 284).  ``a`` and
``b`` are ``(num_tokens, HV)`` (per-token stride ``stride_a_tok``/``stride_b_tok``, lines 318-319,
441-442), addressed by *token* index -- which in decode equals the sequence index.

State layout and update
-----------------------
Both kernels address the recurrent state as ``h[v, k]`` with a configurable *leading* stride only::

    p_h0 = h0 + state_idx * stride_init_state_token          # 301 (decode) / 116 (sequence)
    p_h0 = p_h0 + i_hv * V * K + o_v[:, None] * K + o_k[None, :]

so the trailing ``(HV, V, K)`` dims must be contiguous (``(V*K, K, 1)``) and the tensor is
``(rows, HV, V, K)``.  The decode launcher passes ``ht=initial_state`` (line 458) and returns
``initial_state`` (line 478): the state is updated **in place** and there is no other option.
The sequence launcher defaults to ``inplace_final_state=True`` (line 524) and then aliases
``final_state = initial_state`` (lines 202-203); with ``False`` it allocates a fresh
``(T, HV, V, K)`` buffer (line 205).  vLLM relies on the in-place form: it keeps
``ssm_state = self.kv_cache[1]`` across steps and discards the returned tensor
(``core_attn_out_decode, _ = ...``, caller lines 1684-1695).

The state carried between tokens lives in fp32 registers (``b_h`` is ``tl.zeros([BV, BK],
dtype=tl.float32)``, line 102), and the per-token snapshot is cast to the *state* dtype on write
(lines 161/165, 336).  An intermediate snapshot is therefore rounded but never read back by the
next token of the same call -- the sequence form is one fp32 recurrence with fp16/bf16 side
effects, not a chain of round-trips.  This port reproduces that exactly.

已验证 / 已核实 (verified against the source)
---------------------------------------------
V1  Signatures and defaults -- sequence: lines 516-529 (``beta``, ``scale``, ``initial_state``
    default to ``None``; ``inplace_final_state=True``; ``cu_seqlens``/``ssm_state_indices``/
    ``num_accepted_tokens`` optional; ``use_qk_l2norm_in_kernel=False``).  Decode: lines 339-350
    (everything except ``use_qk_l2norm_in_kernel`` is required and positional).
V2  Per-token recurrence and its order -- sequence lines 132-149, decode lines 327-332; identical.
    Decay first (134/327), delta against the decayed state (139/328), ``* beta`` (144/329), rank-1
    state write (146/330), output from the *updated* state (148/331).
V3  Decode gate -- lines 318-325 plus ``SOFTPLUS_THRESHOLD=20.0`` on line 473: ``softplus`` with a
    linear branch at ``x > 20``, ``g = -exp(A_log) * softplus``, ``beta = sigmoid(b)`` rounded to
    ``b``'s dtype.  Re-used via ``gating._sigmoid_gating_decay`` (same formula, ``beta=1``).
V4  Packed layout and ``H`` inference -- lines 305-311 (offsets) and 417-430 (host-side inference:
    ``qk_dim = qkv_dim - HV * V``, ``q_dim = qk_dim // 2``, ``H = q_dim // K``, ``HV % H == 0``).
V5  ``a``/``b`` layout and strides -- lines 318-319 read ``a[i_n * stride_a_tok + i_hv]``, strides
    taken from ``a.stride(0)``/``b.stride(0)`` (lines 441-442); host requires 2-D and stride(-1)==1
    (lines 357-362).  One token per row, ``HV`` columns.
V6  State address arithmetic and in-place update -- lines 301-303/334-336 (decode) and 116-119/
    163-165 (sequence); decode passes ``ht=h0`` (line 458); sequence aliases when in place
    (lines 202-203).  Only ``stride(0)`` is configurable; trailing dims are hardcoded.
V7  GQA -- ``i_h = i_hv // (HV // H)`` (line 64 sequence, line 284 decode).  ``q``/``k`` are
    gathered with that map; ``v``/``o``/``beta``/``g`` are indexed by ``i_hv`` directly
    (lines 83-96).
V8  ``q`` is scaled, ``k`` is not -- lines 130 and 316 (``b_q = b_q * scale``), and the delta uses
    the *unscaled* ``b_k`` (lines 139/328).  ``scale`` defaults to ``K ** -0.5`` and must be
    positive (lines 599-602); the decode form has no default (``scale`` is positional, line 345).
V9  In-kernel L2 normalisation -- lines 127-129 / 313-315: ``x / sqrt(sum(x*x) + 1e-6)`` on both
    ``q`` and ``k``, *before* the ``scale`` multiply; eps added to the squared sum (not a norm
    clamp), same convention as ``gating.py`` (V5 there).
V10 ``cu_seqlens`` varlen path -- lines 65-74 (``bos``/``eos``; ``all = T``), requires
    ``q.shape[0] == 1`` (lines 594-598).
V11 ``ssm_state_indices``/``num_accepted_tokens`` -- lines 103-120 read the initial row at
    ``indices[n * stride(0) + i_t]`` with ``i_t = num_accepted_tokens[n] - 1`` under speculative
    decoding else ``0``; ``row <= 0`` (``NULL_BLOCK_ID``) makes the kernel return without touching
    ``o`` or the state (lines 113-115).  Writes go to ``indices[n * stride(0) + t]`` per token
    (lines 157-161), skipping ``<= 0``.  Decode always indexes ``indices[n * stride(0)]`` with no
    ``i_t`` (line 292) and zero-fills ``o`` for an invalid slot (lines 296-299).
V12 ``inplace_final_state=False`` -- fresh ``q.new_empty(T, HV, V, K, dtype=initial_state.dtype)``
    (line 205) written at rows ``bos + t`` (lines 163-165).
V13 ``o`` allocation/dtype/shape -- sequence: ``o = q.new_empty(NK, *v.shape)`` (line 201, so
    **``q``'s** dtype), written as ``b_o.to(p_o.dtype.element_ty)`` (line 149), leading tile axis
    squeezed (line 251) -> ``(B, T, HV, V)``.  Decode: caller-allocated ``out`` ``(B, 1, HV, V)``,
    contiguous (lines 371-372, 412-415), indexed ``(i_n * HV + i_hv) * V`` (line 293).
V14 ``beta=None`` -> ``torch.ones_like(q[..., 0])`` (lines 603-604); ``IS_BETA_HEADWISE`` is
    ``beta.ndim == v.ndim`` (line 244), i.e. ``(B, T, HV, V)`` beta: ``p_beta = beta + (bos*HV +
    i_hv)*V + o_v`` advanced by ``HV*V`` (lines 86-87, 175); otherwise one scalar per
    ``(token, head)`` advanced by ``HV`` (lines 89, 175).
V15 ``T == 0`` returns immediately (lines 76-78).
V16 Input contiguity -- ``FusedRecurrentFunction.forward`` calls ``.contiguous()`` on q/k/v/g/beta
    (lines 499-504) but not on the state or indices, so only the state layout is constrained.

假设 (assumed -- flagged, never silently guessed)
-------------------------------------------------
A1  ``initial_state`` is mandatory in practice.  ``fused_recurrent_gated_delta_rule`` dereferences
    ``initial_state.stride(0)`` unconditionally (line 207), so the
    ``USE_INITIAL_STATE = h0 is not None`` heuristic (line 20) is unreachable from Python.  I raise
    ``ValueError`` instead of the ``AttributeError`` upstream would raise (same call as ``gating``
    A7).  The decode form's state is a required positional argument already.
A2  The state's trailing ``(HV, V, K)`` dims are contiguous.  The kernels hardcode
    ``i_hv*V*K + v*K + k`` and only make ``stride(0)`` configurable (V6); I raise ``ValueError``
    rather than read wrong elements.  Enforced by ``gating._check_state_layout``.
A3  ``ssm_state_indices`` is 1-D (decode requires it, lines 367-370) or 2-D with stride-1 columns
    (sequence).  The sequence kernel computes ``stride_indices_tok`` (lines 213-215) but never
    applies it -- only ``+ i_t`` (lines 110, 160) -- so 2-D index tensors must be row-contiguous.
    A 1-D tensor is indexed with ``indices[n]``, whose PyTorch semantics (``offset + n*stride(0)``)
    are exactly the kernel's ``i_n * stride_indices_seq``.
A4  Without ``ssm_state_indices`` the sequence kernel computes the flat offset
    ``bos * HV * V * K`` (line 118) for the *read* but ``(bos + t) * stride_final_state_token``
    (line 163) for the *write*, which coincide only when the state has dense rows.  I require
    ``initial_state.stride(0) == HV*V*K`` on that path and index row ``bos`` (correct for any
    stride, and identical whenever the kernel is well defined).
A5  ``_sigmoid_gating_decay`` implements the softplus log branch as ``log1p(exp(clamp(x, max=20)))``
    while line 323 writes ``log(1.0 + exp(x))``.  Mathematically identical; the difference is at
    most one ulp of the fp32 ``log`` and only for ``x`` near the cutoff (at ``x == 20`` in fp32 the
    ``+1`` is already below the ulp of ``exp(20)``, so both give ``20.0``).  Deliberate, shared with
    ``gating.py`` rather than forking a second softplus.
A6  dtype: every kernel load does ``.to(tl.float32)``, so there is no fp64 mode upstream.  This
    port computes in fp32 for fp16/bf16/fp32 inputs (matching) and in fp64 when any input is fp64
    (an extension for oracles/tests only) -- same rule as ``gating._working_dtype``.
A7  The decode kernel reads one token per program (grid ``(NV, B*HV)``, line 449, no token loop
    anywhere in lines 282-336) and one state slot per token; a multi-token sequence must call it
    once per token.  That is the decode contract being ported, not a limitation of this port.
A8  ``g``/``beta`` may be ``(B, T, HV)``/``(B, T, HV, V)`` or the flat ``(B*T, HV)``/
    ``(B*T, HV, V)`` alias: the kernel addresses them as ``tok*HV + hv`` with no notion of ``B`` or
    ``T`` (lines 91-92, 175), so the port only requires the element count and reshapes.  The decode
    form's ``a``/``b`` must be 2-D, exactly as its launcher validates (lines 357-362).

偏差 (deliberate deviations, with the reason)
---------------------------------------------
D1  ``beta=None`` with ``HV > H``: upstream's ``torch.ones_like(q[..., 0])`` has ``H`` columns per
    token but the kernel reads it with stride ``HV`` (line 175), so it reads past the end of the
    tensor for every token after the first.  I allocate ``(B, T, HV)`` ones, the only defensible
    reading; with ``HV == H`` the two are identical.
D2  An invalid state slot (``row <= 0``) leaves ``o`` *uninitialised* upstream
    (``o = q.new_empty``, line 201).  I return zeros: deterministic, and no caller may depend on
    upstream garbage.  (The decode form zero-fills by construction, lines 296-299.)
D3  ``inplace_final_state=False`` with ``B > 1`` writes rows ``bos + t`` into a ``(T, HV, V, K)``
    buffer (lines 205/163), which is out of bounds for ``B > 1``.  I raise ``ValueError`` instead
    of writing out of bounds (same call as ``gating`` A8).
D4  No Triton, no tiling, no autograd.  ``BK = next_power_of_2(K)`` / ``BV = min(next_pow2(V), 32)``
    (lines 195, 437) tile the kernel only and are invisible in the results.  Upstream's
    ``FusedRecurrentFunction`` defines a ``forward`` and no ``backward`` (lines 481-513), so the
    sequence operator is inference-only there too; this port returns plain tensors from a plain
    function and does not pretend to be differentiable.  Per-token Python loops are fine at decode
    (``T`` is one token per sequence in the serving path).
D5  Argument validation is a superset of upstream's: the decode entry point mirrors the launcher's
    ``ValueError`` messages (lines 351-430) and additionally rejects an out-of-range state slot and
    a non-contiguous state (A2) where upstream would read/write arbitrary memory.
"""

from __future__ import annotations

from typing import Optional

import os

import torch

# Re-use the sibling module's conventions instead of creating a second dialect: the gate formula
# with its softplus threshold, the fp32/fp64 working-dtype rule, the L2-norm eps, the state-layout
# check and the index addressing are all identical between the prefill/update operator and these.
from .gating import (
    _L2NORM_EPS,
    _check_state_layout,
    _sigmoid_gating_decay,
    _state_index,
    _working_dtype,
)

__all__ = [
    "fused_recurrent_gated_delta_rule",
    "fused_recurrent_gated_delta_rule_packed_decode",
]

#: Line 473 of the launcher: ``SOFTPLUS_THRESHOLD=20.0`` -- the softplus linear-branch cut used on
#: line 323.  (The sequence form's gate arrives pre-computed as ``g``, so it has no threshold.)
_SOFTPLUS_THRESHOLD = 20.0


def _gqa_map(H: int, HV: int, device: torch.device) -> torch.Tensor:
    """``i_h = i_hv // (HV // H)`` -- line 64 (sequence) / line 284 (decode)."""
    if H <= 0 or HV < H or HV % H:
        raise ValueError(
            f"expected HV % H == 0 and HV >= H (i_h = i_hv // (HV // H), lines 64/284); "
            f"got H={H}, HV={HV}"
        )
    return torch.arange(HV, device=device) // (HV // H)


#: Per-(layer, dtype, device) constants of the decode operator, built once instead of every step.
#: The entries hold a **strong reference** to ``A_log``/``dt_bias``, which is what makes the key
#: safe: a cached entry keeps its source tensors alive, so their address cannot be recycled into a
#: different layer.  Decode is dispatch-bound (§22: ~25-45 us per op at batch 1), so removing three
#: allocations + two dtype conversions from a ~15-op function is worth ~20% of it.
_DECODE_CONST_CACHE: dict = {}
_DECODE_CONST_CACHE_MAX = 256


def _decode_constants(A_log, dt_bias, H, HV, device, working):
    """``(hv_to_h, A_w, dtb_w)`` for the decode operator, cached per layer."""
    from tileinfer.serving import flags  # noqa: PLC0415 - measurement channel, off by default

    use_cache = flags.flag("gdn_cache", "TILEINFER_GDN_CACHE", True)
    key = (
        id(A_log),
        A_log.data_ptr(),
        id(dt_bias),
        dt_bias.data_ptr(),
        H,
        HV,
        str(working),
        str(device),
    )
    if use_cache:
        hit = _DECODE_CONST_CACHE.get(key)
        if hit is not None:
            return hit[0], hit[1], hit[2]
    hv_to_h = _gqa_map(H, HV, device)
    A_w = A_log.reshape(-1).to(working)
    dtb_w = dt_bias.reshape(-1).to(working)
    if use_cache:
        if len(_DECODE_CONST_CACHE) >= _DECODE_CONST_CACHE_MAX:
            _DECODE_CONST_CACHE.clear()
        _DECODE_CONST_CACHE[key] = (hv_to_h, A_w, dtb_w, A_log, dt_bias)
    return hv_to_h, A_w, dtb_w


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    """``x / sqrt(sum(x*x) + 1e-6)`` -- lines 128-129 / 314-315, per head over ``K``."""
    return x * torch.rsqrt(x.square().sum(-1, keepdim=True) + _L2NORM_EPS)


def fused_recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    inplace_final_state: bool = True,
    cu_seqlens: Optional[torch.Tensor] = None,
    ssm_state_indices: Optional[torch.Tensor] = None,
    num_accepted_tokens: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused recurrent gated delta rule, whole-sequence form (torch port).

    Signature is upstream's, verbatim (lines 516-529)::

        def fused_recurrent_gated_delta_rule(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            g: torch.Tensor,
            beta: torch.Tensor = None,
            scale: float = None,
            initial_state: torch.Tensor = None,
            inplace_final_state: bool = True,
            cu_seqlens: torch.Tensor | None = None,
            ssm_state_indices: torch.Tensor | None = None,
            num_accepted_tokens: torch.Tensor | None = None,
            use_qk_l2norm_in_kernel: bool = False,
        ) -> tuple[torch.Tensor, torch.Tensor]:

    Args:
        q: ``(B, T, H, K)`` queries; scaled by ``scale``.  Non-contiguous input is accepted
            (upstream calls ``.contiguous()``, line 499).
        k: ``(B, T, H, K)`` keys; **never** scaled (line 130).
        v: ``(B, T, HV, V)`` values; also supplies ``o``'s shape (line 201).
        g: ``(B, T, HV)`` log decay, applied as ``exp(g)`` before the delta (line 134).  This is a
            *pre-computed* gate: for GDN, ``g = -exp(A_log) * softplus(a + dt_bias)``.
        beta: delta-rule step size, ``(B, T, HV)`` per head, or ``(B, T, HV, V)`` headwise when
            ``beta.dim() == v.dim()`` (line 244).  ``None`` -> ones (D1).
        scale: query scale; ``None`` -> ``K ** -0.5``; otherwise must be ``> 0`` (lines 599-602).
        initial_state: ``(rows, HV, V, K)`` recurrent state, **mandatory** (A1).  Mutated in place
            when ``inplace_final_state`` is True (the default, and what vLLM relies on).
        inplace_final_state: alias ``initial_state`` as the output state (lines 202-203) instead of
            filling a fresh ``(T, HV, V, K)`` buffer (line 205, B == 1 only -- D3).
        cu_seqlens: ``(N + 1,)`` cumulative token offsets for a flattened ragged batch; requires
            ``B == 1`` (lines 594-598).
        ssm_state_indices: ``(N,)`` or ``(N, T)`` paged-state slot ids (A3); slot ``<= 0`` means
            "no state" and the sequence is skipped entirely (lines 113-115).
        num_accepted_tokens: ``(N,)`` speculative-decoding accepted counts; the *initial* state is
            read from slot ``num_accepted_tokens[n] - 1`` (lines 105-108).  Ignored without
            ``ssm_state_indices``.
        use_qk_l2norm_in_kernel: normalise ``q`` and ``k`` per head before scaling (lines 127-129).

    Returns:
        ``(o, final_state)``: ``o`` is ``(B, T, HV, V)`` in ``q.dtype`` (V13); ``final_state`` is
        ``initial_state`` itself when ``inplace_final_state`` is True.
    """
    # Line 192: ``B, T, H, K, V = *k.shape, v.shape[-1]``; line 193: ``HV = v.shape[2]``.
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError(
            f"q/k/v must be 4-D (q/k: (B,T,H,K), v: (B,T,HV,V)); got {tuple(q.shape)}, "
            f"{tuple(k.shape)}, {tuple(v.shape)}"
        )
    B, T, H, K = k.shape
    HV, V = v.shape[2], v.shape[3]
    if tuple(q.shape) != tuple(k.shape):
        raise ValueError(
            f"q and k must have the same shape; got {tuple(q.shape)} vs {tuple(k.shape)}"
        )
    if tuple(v.shape[:2]) != (B, T):
        raise ValueError(f"v must be (B, T, HV, V) = ({B}, {T}, {HV}, {V}), got {tuple(v.shape)}")
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
            "unconditionally (line 207), so the USE_INITIAL_STATE=False kernel path is "
            "unreachable from the python launcher (A1)"
        )
    _check_state_layout(initial_state, HV, V, K, "initial_state")
    if inplace_final_state:
        # Lines 202-203: ``final_state = initial_state`` -- the state is updated in place.
        final_state = initial_state
    else:
        # Line 205: ``q.new_empty(T, HV, V, K, dtype=initial_state.dtype)``.
        final_state = q.new_empty(T, HV, V, K, dtype=initial_state.dtype)
        _check_state_layout(final_state, HV, V, K, "final_state")
        if B > 1:
            raise ValueError(
                "inplace_final_state=False writes rows bos + t into a (T, HV, V, K) buffer "
                "(lines 205/163), which only stays in bounds for B == 1 (D3)"
            )

    hv_to_h = _gqa_map(H, HV, q.device)

    # Lines 83-89 / 216-217: q, k, v, g, beta are made contiguous by the launcher
    # (``FusedRecurrentFunction.forward``, lines 499-504); the flat ``tok*HV`` addressing below is
    # only defined on those layouts.
    g = g.contiguous()
    if g.numel() != B * T * HV:
        raise ValueError(
            f"g must be (B, T, HV) = ({B}, {T}, {HV}); the kernel reads g + tok*HV + hv "
            f"(lines 91-92).  Got shape {tuple(g.shape)}"
        )
    if beta is None:
        # Line 603-604.  With HV > H upstream under-allocates (D1); allocate the consistent shape.
        beta = torch.ones(B, T, HV, dtype=q.dtype, device=q.device)
    # Line 244: ``IS_BETA_HEADWISE=beta.ndim == v.ndim``.
    beta_headwise = beta.dim() == v.dim()
    beta = beta.contiguous()
    expected_beta = B * T * HV * (V if beta_headwise else 1)
    if beta.numel() != expected_beta:
        raise ValueError(
            f"beta must be (B, T, HV){' , V' if beta_headwise else ''}: the kernel reads "
            f"beta + (tok*HV + hv){'*V + v' if beta_headwise else ''} "
            f"(lines 86-89, 175).  Got shape {tuple(beta.shape)}"
        )
    v = v.contiguous()

    if cu_seqlens is not None and int(cu_seqlens[-1]) != q.shape[1]:
        raise ValueError(
            f"cu_seqlens[-1] must equal the flattened token count ({q.shape[1]}), "
            f"got {int(cu_seqlens[-1])}"
        )
    if ssm_state_indices is not None and ssm_state_indices.dim() not in (1, 2):
        raise ValueError(
            f"ssm_state_indices must be 1-D or 2-D, got shape {tuple(ssm_state_indices.shape)}"
        )
    if ssm_state_indices is not None and ssm_state_indices.dim() == 2:
        if ssm_state_indices.stride(1) != 1:
            raise ValueError(
                "ssm_state_indices must have stride-1 columns: upstream computes "
                "stride_indices_tok but never applies it (lines 213-215 vs 110/160) (A3)"
            )
    if ssm_state_indices is None and initial_state.stride(0) != HV * V * K:
        # A4: the read is at flat offset ``bos * HV * V * K`` (line 118) while the write uses
        # ``(bos + t) * stride(0)`` (line 163); the two only agree on dense rows.
        raise ValueError(
            f"without ssm_state_indices the state rows must be dense (stride(0) == HV*V*K = "
            f"{HV * V * K}); got stride(0)={initial_state.stride(0)} (line 118 vs 163, A4)"
        )
    if ssm_state_indices is None:
        needed = B * T if inplace_final_state else (B - 1) * T + 1
        if initial_state.shape[0] < needed:
            raise ValueError(
                f"without ssm_state_indices the state row is the flattened token offset bos + t "
                f"(lines 118/163): initial_state needs >= {needed} rows, got "
                f"{initial_state.shape[0]}; pass ssm_state_indices (the vLLM decode path) or a "
                f"per-token state buffer"
            )

    working = _working_dtype(q, k, v, g, beta, initial_state)
    q_flat = q.to(working).reshape(B * T, H, K)
    k_flat = k.to(working).reshape(B * T, H, K)
    v_flat = v.to(working).reshape(B * T, HV, V)
    g_flat = g.to(working).reshape(B * T, HV)
    beta_flat = (
        beta.to(working).reshape(B * T, HV, V)
        if beta_headwise
        else beta.to(working).reshape(B * T, HV)
    )

    # Line 201: ``o = q.new_empty(NK, *v.shape)`` -- the output inherits q's dtype, not v's.
    # zeros() instead of empty() so a skipped sequence leaves a defined value (D2).
    o = torch.zeros(B, T, HV, V, dtype=q.dtype, device=q.device)
    o_flat = o.view(B * T, HV, V)

    # The GQA head map does not depend on the token, so gather once for the whole batch instead of
    # twice per token per layer.  On the device this removes ~60 small-kernel launches per token
    # (30 GDN layers x 2 gathers) - and small-kernel launch latency, not bandwidth, is what the
    # phase table blames for GDN's 0.7 ms/layer (docs/plan-performance.md §12).
    q_g = q_flat.index_select(1, hv_to_h)
    k_g = k_flat.index_select(1, hv_to_h)

    for n in range(N):
        if cu_seqlens is None:
            # Lines 72-73.
            bos, eos = n * T, (n + 1) * T
        else:
            bos, eos = int(cu_seqlens[n]), int(cu_seqlens[n + 1])
        if eos < bos:
            raise ValueError(f"sequence {n}: cu_seqlens must be non-decreasing")
        if eos > B * T:
            raise ValueError(f"sequence {n}: token range [{bos}, {eos}) exceeds {B * T}")
        if eos == bos:
            # Lines 76-78.
            continue

        # ---- initial state (lines 103-120) ----------------------------------------------------
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
            # Line 118: ``p_h0 = h0 + bos * HV * V * K`` -- row ``bos`` (A4).
            init_row = bos
            h = initial_state[init_row].to(working)

        # ---- per-token recurrence (lines 122-170) ---------------------------------------------
        for i_t in range(eos - bos):
            tok = bos + i_t
            # Line 134: ``b_h *= exp(b_g)`` -- the decay comes FIRST.
            decay = torch.exp(g_flat[tok])
            # Lines 141/143 -> 144: per-head or headwise delta-rule step.
            beta_t = beta_flat[tok]
            # Lines 83-85: q/k gather the GQA-mapped key head (hoisted above), v is indexed by
            # value head.
            q_t = q_g[tok]
            k_t = k_g[tok]
            v_t = v_flat[tok]
            if use_qk_l2norm_in_kernel:
                # Lines 127-129.
                q_t = _l2norm(q_t)
                k_t = _l2norm(k_t)
            # Line 130: q only.
            q_t = q_t * scale
            h = h * decay[:, None, None]
            # Line 139: ``b_v -= tl.sum(b_h * b_k[None, :], 1)``.
            delta = v_t - (h * k_t[:, None, :]).sum(-1)
            # Line 144: ``b_v *= b_beta`` -- per head, or per (head, value channel) when headwise.
            delta = delta * beta_t if beta_headwise else delta * beta_t[:, None]
            # Line 146.
            h = h + delta[:, :, None] * k_t[:, None, :]
            # Lines 148-149: output from the UPDATED state, q already scaled.
            o_flat[tok] = (h * q_t[:, None, :]).sum(-1).to(o.dtype)

            # ---- state write-back (lines 152-170) ---------------------------------------------
            if ssm_state_indices is not None and inplace_final_state:
                final_row = _state_index(ssm_state_indices, n, i_t)
                if final_row > 0:
                    final_state[final_row] = h.to(final_state.dtype)
            else:
                final_state[bos + i_t] = h.to(final_state.dtype)

    return o, final_state


def fused_recurrent_gated_delta_rule_packed_decode(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    out: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Packed GDN decode step: one token per sequence (torch port of the Triton decode kernel).

    Signature is upstream's, verbatim (lines 339-350)::

        def fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv: torch.Tensor,
            a: torch.Tensor,
            b: torch.Tensor,
            A_log: torch.Tensor,
            dt_bias: torch.Tensor,
            scale: float,
            initial_state: torch.Tensor,
            out: torch.Tensor,
            ssm_state_indices: torch.Tensor,
            use_qk_l2norm_in_kernel: bool = False,
        ) -> tuple[torch.Tensor, torch.Tensor]:

    The kernel has no token loop: its grid is ``(NV, B * HV)`` (line 449) and program
    ``(i_v, i_nh)``
    handles sequence ``i_n``, value head ``i_hv`` and *the* token of that row (A7).  ``T > 1`` is
    ``T`` calls with the state carried in place.

    Args:
        mixed_qkv: ``(B, 2*H*K + HV*V)``, last dim contiguous, a row is
            ``[q: H*K][k: H*K][v: HV*V]`` (lines 305-311); ``H`` is inferred from the row width
            (lines 417-430).  ``B`` is the number of decoding sequences (one token each).
        a: ``(B, HV)`` gate pre-activation, last dim contiguous (lines 318, 357-362).
        b: ``(B, HV)`` delta-rule beta pre-activation; the step is ``sigmoid(b)`` (line 325).
        A_log: ``(HV,)`` log decay magnitude; the gate uses ``-exp(A_log)`` (line 324).
        dt_bias: ``(HV,)`` added to ``a`` before the softplus (line 322).
        scale: query scale; required, must be ``> 0`` (there is no default upstream, line 345).
        initial_state: ``(rows, HV, V, K)``; supplies ``HV``/``V``/``K`` (line 403) and is updated
            **in place** (``ht=h0``, line 458).
        out: caller-allocated ``(B, 1, HV, V)``, contiguous (lines 371-372, 412-415); written in
            place and returned.
        ssm_state_indices: ``(B,)`` paged-state row per token (line 292).  Row ``<= 0`` writes
            zeros to ``out`` and leaves the state untouched (lines 296-299).
        use_qk_l2norm_in_kernel: normalise ``q`` and ``k`` per head before scaling (lines 313-315).
            vLLM's decode path passes ``True`` (caller line 1694).

    Returns:
        ``(out, initial_state)`` -- the two mutated input tensors (line 478).
    """
    # ---- launcher validation, mirrored from lines 351-430 --------------------------------
    if mixed_qkv.ndim != 2:
        raise ValueError(f"`mixed_qkv` must be a 2D tensor (got ndim={mixed_qkv.ndim}).")
    if mixed_qkv.stride(-1) != 1:
        raise ValueError("`mixed_qkv` must be contiguous in the last dim.")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(
            f"`a` and `b` must be 2D tensors (got a.ndim={a.ndim}, b.ndim={b.ndim})."
        )
    if a.stride(-1) != 1 or b.stride(-1) != 1:
        raise ValueError("`a`/`b` must be contiguous in the last dim.")
    if A_log.ndim != 1 or dt_bias.ndim != 1:
        raise ValueError("`A_log`/`dt_bias` must be 1D tensors.")
    if A_log.stride(0) != 1 or dt_bias.stride(0) != 1:
        raise ValueError("`A_log`/`dt_bias` must be contiguous.")
    if ssm_state_indices.ndim != 1:
        raise ValueError(
            f"`ssm_state_indices` must be 1D for packed decode "
            f"(got ndim={ssm_state_indices.ndim})."
        )
    if not out.is_contiguous():
        raise ValueError("`out` must be contiguous.")

    dev = mixed_qkv.device
    if (
        a.device != dev
        or b.device != dev
        or A_log.device != dev
        or dt_bias.device != dev
        or initial_state.device != dev
        or out.device != dev
        or ssm_state_indices.device != dev
    ):
        raise ValueError("All inputs must be on the same device.")

    B = mixed_qkv.shape[0]
    if a.shape[0] != B or b.shape[0] != B:
        raise ValueError(
            "Mismatched batch sizes: "
            f"mixed_qkv.shape[0]={B}, a.shape[0]={a.shape[0]}, b.shape[0]={b.shape[0]}."
        )
    if ssm_state_indices.shape[0] != B:
        raise ValueError(
            f"`ssm_state_indices` must have shape [B] (got "
            f"{tuple(ssm_state_indices.shape)}; expected ({B},))."
        )

    if initial_state.ndim != 4:
        raise ValueError(
            f"`initial_state` must be a 4D tensor (got ndim={initial_state.ndim})."
        )
    if initial_state.stride(-1) != 1:
        raise ValueError("`initial_state` must be contiguous in the last dim.")
    # Line 403: ``HV, V, K = initial_state.shape[-3:]``.
    HV, V, K = initial_state.shape[-3:]
    _check_state_layout(initial_state, HV, V, K, "initial_state")  # A2, lines 301-303/334-336
    if a.shape[1] != HV or b.shape[1] != HV:
        raise ValueError(
            f"`a`/`b` must have shape [B, HV] with HV={HV} (got a.shape="
            f"{tuple(a.shape)}, b.shape={tuple(b.shape)})."
        )
    if A_log.numel() != HV or dt_bias.numel() != HV:
        raise ValueError(
            f"`A_log` and `dt_bias` must have {HV} elements (got "
            f"A_log.numel()={A_log.numel()}, dt_bias.numel()={dt_bias.numel()})."
        )
    if out.shape != (B, 1, HV, V):
        raise ValueError(
            f"`out` must have shape {(B, 1, HV, V)} (got out.shape={tuple(out.shape)})."
        )

    # Lines 417-430: infer H from the packed row width.
    qkv_dim = mixed_qkv.shape[1]
    qk_dim = qkv_dim - HV * V
    if qk_dim <= 0 or qk_dim % 2 != 0:
        raise ValueError(
            f"Invalid packed `mixed_qkv` last dim={qkv_dim} for HV={HV}, V={V}."
        )
    q_dim = qk_dim // 2
    if q_dim % K != 0:
        raise ValueError(f"Invalid packed Q size {q_dim}: must be divisible by K={K}.")
    H = q_dim // K
    if H <= 0 or HV % H != 0:
        raise ValueError(
            f"Invalid head config inferred from mixed_qkv: H={H}, HV={HV}."
        )
    working = _working_dtype(mixed_qkv, a, b, A_log, dt_bias, initial_state, out)
    hv_to_h, A_w, dtb_w = _decode_constants(A_log, dt_bias, H, HV, dev, working)
    # Lines 318-324: the same gate as ``gating._sigmoid_gating_decay`` (``beta=1``,
    # threshold ``SOFTPLUS_THRESHOLD``); ``g_val = -exp(A_log) * softplus(a + dt_bias)``.
    g = _sigmoid_gating_decay(
        a.to(working), A_w, dtb_w, 1.0, _SOFTPLUS_THRESHOLD, working
    )  # (B, HV)
    decay = torch.exp(g)  # line 327: ``b_h *= exp(g_val)``
    # Line 325: ``beta_val = tl.sigmoid(b_val).to(b.dtype.element_ty).to(tl.float32)`` -- the
    # sigmoid is computed in fp32 and then *rounded to b's dtype* before the multiply.
    beta_t = torch.sigmoid(b.to(working)).to(b.dtype).to(working)

    #: Sync-free decode path: index the state with a *device* tensor instead of ``int(t[n])``.
    #: The host read below is one device->host synchronisation per call, which in serving is once
    #: per GDN layer -- 30 per decode step (§32 measured the same pattern in the conv at ~0.5 ms
    #: each).  It is **opt-in** because the price is the loud out-of-range check: without the host
    #: read an out-of-range slot is clamped, not raised, so the default keeps vLLM's contract and
    #: the fast path has to be asked for (``TILEINFER_GDN_NO_SYNC=1``, or the ``gdn_no_sync`` flag).
    from tileinfer.serving import flags  # noqa: PLC0415

    sync_free = flags.flag("gdn_no_sync", "TILEINFER_GDN_NO_SYNC", False)

    for n in range(B):
        # Line 292: one state row per token (1-D, so ``indices[n]`` honours stride(0) exactly as
        # ``ssm_state_indices + i_n * stride_indices_seq`` does).
        if sync_free:
            # ``idx`` stays a 0-dim device tensor: gather with it, write with it, and fold the
            # "slot <= 0 means skip" rule in with ``where`` instead of a host ``if``.
            idx = ssm_state_indices[n]
            live = (idx > 0).reshape(1).reshape(1, 1, 1)
            safe = idx.clamp(0, initial_state.shape[0] - 1).reshape(1)
            old_slot = initial_state.index_select(0, safe)[0]
            h = old_slot.to(working)
        else:
            state_idx = int(ssm_state_indices[n])
            if state_idx <= 0:
                # Lines 296-299: zero ``o`` and return without touching the state (D2).
                out[n].zero_()
                continue
            if state_idx >= initial_state.shape[0]:
                raise ValueError(
                    f"ssm_state_indices[{n}]={state_idx} is out of range for initial_state with "
                    f"{initial_state.shape[0]} rows (upstream would write out of bounds; D5)"
                )

            h = initial_state[state_idx].to(working)
        # Lines 305-311: decode the packed row.  ``stride(0)`` is honoured by row indexing and the
        # last dim is stride-1 (validated above), so these slices alias the caller's buffer.
        row = mixed_qkv[n]
        q_t = row[: H * K].reshape(H, K).index_select(0, hv_to_h)
        k_t = row[H * K : 2 * H * K].reshape(H, K).index_select(0, hv_to_h)
        v_t = row[2 * H * K :].reshape(HV, V)

        if use_qk_l2norm_in_kernel:
            # Lines 313-315.
            q_t = _l2norm(q_t)
            k_t = _l2norm(k_t)
        # Line 316: q only.
        q_t = q_t * scale

        # Lines 327-332, in order.
        h = h * decay[n][:, None, None]
        delta = (v_t - (h * k_t[:, None, :]).sum(-1)) * beta_t[n][:, None]
        h = h + delta[:, :, None] * k_t[:, None, :]
        # Line 332: ``tl.store(p_o, b_o.to(p_o.dtype.element_ty), ...)``.
        y = (h * q_t[:, None, :]).sum(-1).to(out.dtype)
        if not sync_free:
            out[n, 0] = y
            # Lines 334-336: ``p_ht = ht + state_idx * stride_final_state_token`` (``ht is h0``).
            initial_state[state_idx] = h.to(initial_state.dtype)
            continue
        # A skipped slot must leave ``o`` zero and the state untouched; both are folded in here so
        # the loop never reads an index back to the host.  ``old_slot`` is the pre-update row in the
        # state dtype, which is exactly what "untouched" means.
        out[n, 0] = torch.where(live.reshape(1, 1), y, torch.zeros_like(y))
        initial_state.index_copy_(
            0, safe, torch.where(live, h.to(initial_state.dtype), old_slot).unsqueeze(0)
        )

    return out, initial_state
