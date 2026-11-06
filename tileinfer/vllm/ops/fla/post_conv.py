"""Torch port of vLLM's ``fused_post_conv_prep`` - GDN prefill's post-conv split / normalise / gate.

Where it sits, and why it is on the critical path
------------------------------------------------
[vLLM] ``qwen_gdn_linear_attn.py:1428-1439`` calls it inside ``_forward_core``'s prefill branch,
*after* the causal conv ([vLLM] ``:1365-1375``) and *before* the chunk chain ([vLLM] ``:1518-1530``);
the V1 profiling warmup repeats the same call at [vLLM] ``:1113-1124``.  It is imported by name at
[vLLM] ``:27-31`` (``from ...fla.ops import fused_post_conv_prep``) and re-exported from
``fla/ops/__init__.py:10``, so the prefill path cannot start without it.  Both call sites pass
``apply_l2norm=True`` and ``output_g_exp=False``.

Upstream body: the real source, read line by line
-------------------------------------------------
``vllm/model_executor/layers/fla/ops/fused_gdn_prefill_post_conv.py`` **is** in this checkout, as
``runtime/vllm-api/vllm__model_executor__layers__fla__ops__fused_gdn_prefill_post_conv.py``: one
Triton kernel plus a launcher, and every claim below is now read out of it rather than inferred.

* ``_fused_post_conv_kernel`` (:19-149) with grid ``(cdiv(L, BLOCK_T), H + HV)`` (:214).  Every
  program whose ``program_id(1) = i_head < H`` handles exactly one **key** head: it loads q and k
  for all tokens of that head, optionally L2-normalises both, and stores them (:69-107).  Every
  program with ``i_head >= H`` handles exactly one **value** head: it copies v and computes the two
  gate scalars ``g``/``beta`` for every token of that head (:108-149).
* ``fused_post_conv_prep`` (:152-248) validates the packed width
  (``qkv_dim == 2*H*K + HV*V``, :194-196), allocates the five outputs in their target contiguous
  layout (:199-203) and launches that kernel once (:215-246).

Consequences of that split, which the port preserves: the operator is per-token and stateless (no
cross-token reduction, no state in/out), and q/k/v are written in ``conv_output.dtype`` while
``g``/``beta`` are **always** ``torch.float32`` (:202-203) even though both are computed from
bf16/fp16 inputs - the kernel casts every gate operand with ``.to(tl.float32)`` first (:128-134).

Provenance tags used below:

* **[vLLM]** - a line number in ``fused_gdn_prefill_post_conv.py`` (this operator), or - where the
  claim is about the call site - in ``runtime/vllm-api/`` ``vllm__model_executor__layers__mamba__\\
  gdn__qwen_gdn_linear_attn.py`` (the file is named at the citation).
* **[port]** - a deliberate difference between this torch transcription and that source, always
  with the reason.  There is exactly one; see "Deliberate deviations" at the end.

What the operator is
--------------------
A **per-token, stateless** map from the packed conv output to the five chunk-chain tensors.  There is
no cross-token state, no reduction over the token axis and no scale: everything it does is a split, a
per-head L2 normalisation, a per-value-head gate and a per-value-head sigmoid.

Argument semantics (the signature both the launcher and the call site use)
-------------------------------------------------------------------------
The ten arguments are the **caller's**, so every ``[vLLM]`` line in this table is a line of
``runtime/vllm-api/vllm__model_executor__layers__mamba__gdn__qwen_gdn_linear_attn.py``; the kernel
file's own lines are cited from "Packed layout" onward.  The launcher's signature
(``fused_gdn_prefill_post_conv.py:152-163``) takes exactly the same ten in the same order, with the
same two defaults (:161-162).

===================  ==========================  =================================================
argument             shape / dtype             meaning and evidence
===================  ==========================  =================================================
``conv_output``      ``(T, q|k|v)`` same dtype   the packed causal-conv output [vLLM] ``:1429``;
                     as the layer activations   ``conv_dim = 2*key_dim + value_dim`` [vLLM]
                     (bf16 in production)       ``:466``, laid out q|k|v (see "Packed layout")
``a``                ``(T, num_v_heads)``        gate pre-activation, one per **value** head
                                                [vLLM] ``:1430``, ``:802-806``
``b``                ``(T, num_v_heads)``        delta-rule beta pre-activation [vLLM] ``:1431``
``A_log``            ``(num_v_heads,)`` fp32     per-head log decay magnitude [vLLM] ``:1432``,
                                                ``:519-524``
``dt_bias``          ``(num_v_heads,)`` fp32     per-head time-step bias [vLLM] ``:1433``,
                                                ``:516-518``
``num_k_heads``      int                         **key** head count, TP-sharded: the caller passes
                                                ``self.num_k_heads // self.tp_size`` [vLLM] ``:1434``
``head_k_dim``       int                         key head width [vLLM] ``:1435``
``head_v_dim``       int                         value head width [vLLM] ``:1436``
``apply_l2norm``     bool, call sites True      normalise q and k per head (see below)
``output_g_exp``     bool, call sites False     ``False`` -> **log**-space g (the chain's form);
                                                ``True`` -> the decay ``exp(g)``
                                                [vLLM] ``:142-143``, see "Flags"
===================  ==========================  =================================================

There is **no** ``num_v_heads`` argument and no ``scale`` argument: the value head count is not a
parameter upstream either - the kernel takes it from ``A_log.shape[0]`` [vLLM] ``:190`` and then
asserts the packed width against it (:194-196) - and the chunk chain applies its own ``K ** -0.5``
scale, so q must come back unscaled (:93-107 store q as loaded/normalised, with no factor).

Packed layout: q | k | v, head-major inside each slice
------------------------------------------------------
The kernel's own addressing proves the layout without leaving the operator.  q is read at
``i_h * K + k`` (:76-78), k at ``HK + i_h * K + k`` with ``HK = H*K`` (:64, :80-82), and v at
``V_OFFSET + i_hv * V + v`` with ``V_OFFSET = 2 * H * K`` (:115-120).  So the row is
``[ q: num_k_heads*head_k_dim | k: num_k_heads*head_k_dim | v: num_v_heads*head_v_dim ]`` and each
slice is contiguous per head (head ``h`` occupies columns ``h*head_dim .. (h+1)*head_dim - 1``).

The same buffer is consumed by ``rearrange_mixed_qkv`` on the caller's *decode* branch -
[vLLM] ``:1446-1448`` passes the very tensor that ``:1429`` names ``conv_output`` into that helper -
and that helper splits it the same way (``torch.split(mixed_qkv, [q_dim, k_dim, v_dim], dim=-1)``
then ``q_contig.view(1, seq, -1, head_k_dim)``, caller ``:821-840``).  The q|k|v order is also what
the conv weight loader assumes (caller ``:499-510``, ``query_key_settings, query_key_settings,
value_settings``) and what ``conv_dim`` encodes (caller ``:466``).  Three independent agreements on
the same order.

Note the nonlinearity is *before* this operator: ``causal_conv1d_fn`` is called with
``activation=self.activation`` (caller [vLLM] ``:1369``), so ``conv_output`` is already post-SiLU.
The kernel applies no further activation to q/k/v - it only loads, optionally normalises, stores.

The gate (identical to the decode path's, one convention)
---------------------------------------------------------
[vLLM] ``:128-149``, verbatim::

    A_log_val = tl.load(A_log_ptr + i_hv).to(tl.float32)          # :128
    dt_bias_val = tl.load(dt_bias_ptr + i_hv).to(tl.float32)      # :129
    a_vals = tl.load(a_ptr + a_offsets, ...).to(tl.float32)       # :133
    b_vals = tl.load(b_ptr + b_offsets, ...).to(tl.float32)       # :134
    x  = a_vals + dt_bias_val                                     # :137
    sp = tl.where(x > 0, x + tl.log(1.0 + tl.exp(-x)),
                        tl.log(1.0 + tl.exp(x)))                  # :138
    sp = tl.where(x <= SOFTPLUS_THRESHOLD, sp, x)                 # :139  (=20.0, :240)
    g_vals = -tl.exp(A_log_val) * sp                              # :140  -> g <= 0
    if OUTPUT_G_EXP: g_vals = tl.exp(g_vals)                      # :142-143
    beta_vals = tl.sigmoid(b_vals)                                # :145

which is the same ``g = -A_log.exp() * softplus(a + dt_bias)`` / ``beta = b.sigmoid()`` the decode
kernel documents (caller ``:1782-1791``, docstring at ``:1805``).  ``g`` is per **value** head
(``a``/``b`` are, caller ``:802-806``), the ``A_log``/``dt_bias`` index is the value head, and the
key/value head asymmetry is handled downstream by the chunk chain (q/k have ``num_k_heads`` heads,
v/g/beta have ``num_v_heads``).  Rather than re-deriving this, the port **calls**
``tileinfer.vllm.ops.fla.gating._sigmoid_gating_decay`` - the same helper the decode-step port uses -
so there is exactly one place in the package where the softplus/threshold/``-exp(A_log)`` order
lives, and it is pinned to this kernel's ``beta=1`` / ``threshold=20`` / fp32 form.

The L2 normalisation, and the eps convention
--------------------------------------------
``apply_l2norm=True`` normalises **q and k** (each ``head_k_dim`` row of each key head), never v:
the guard is ``if APPLY_L2NORM`` around the q and k blocks only (:84-91), and the v program never
touches the norm.  The kernel computes ``q_inv = 1.0 / tl.sqrt(tl.sum(q_f32 * q_f32, axis=1) +
L2NORM_EPS)`` (:85-86) with ``L2NORM_EPS = 1e-6`` (:238) - the eps is added to the **squared sum**,
not clamped onto the norm, which matters only for near-zero rows, where ``x/sqrt(sum+eps)`` and
``x/max(||x||, eps)`` differ by orders of magnitude; ``tests/test_gdn_post_conv.py`` pins it with a
deliberately tiny head.

The port calls ``tileinfer.vllm.ops.gdn.l2norm_fwd``, i.e. exactly ``x / sqrt(sum(x^2, -1) + 1e-6)``
with the reduction in fp32 and the result cast back to ``x.dtype`` - which is what the kernel does
when it stores ``q_f32.to(q_ptr.dtype.element_ty)`` (:97, :105).  So the port is not a second
transcription of the norm; it is the same formula through the package's one helper.

Why ``apply_l2norm=True`` here rather than in the chain: the prefill call passes
``use_qk_l2norm_in_kernel=False`` (caller [vLLM] ``:1529``), so the chain does **not** re-normalise;
the normalisation has to happen in this operator, once, on the freshly split buffers.  (The decode
path is the opposite: it calls ``fused_sigmoid_gating_delta_rule_update(...,
use_qk_l2norm_in_kernel=True)`` (caller ``:1498``, ``:1557``) and normalises in-kernel.)

Flags
-----
``apply_l2norm``
    ``True``: q/k are returned normalised.  ``False``: q/k are returned **exactly** as split from
    ``conv_output`` (raw conv output, same values, contiguous).  [vLLM] ``:84-91`` - the flag is the
    ``APPLY_L2NORM`` ``tl.constexpr`` (:47, passed at :237) that guards *only* the two normalisation
    blocks, so ``False`` is "skip them", nothing else.  No local call site passes ``False``
    (both pass ``True``), so the branch has no production witness - it is exercised by
    ``test_apply_l2norm_false_*`` against the caller's own split (``rearrange_mixed_qkv``).
``output_g_exp``
    ``False``: ``g`` is the **log**-space forget gate, which is what the chunk chain wants - its
    step 0 is ``chunk_local_cumsum(g)`` [vLLM] ``chunk.py:37-39`` and every later step uses
    ``exp(g_i - g_j)`` (``chunk_o``), so an already-exponentiated ``g`` would be wrong by a factor
    ``exp(g)`` per token.  ``True``: ``g`` comes back as the decay ``exp(g)``, elementwise and still
    fp32 - [vLLM] ``:142-143`` (``if OUTPUT_G_EXP: g_vals = tl.exp(g_vals)``, the ``tl.constexpr``
    declared at :49 and passed at :239).  That branch exists for a consumer that wants ``exp(g)``
    directly (the kernel's own comment: "for FlashInfer", :176); both local call sites pass
    ``False``, so the chain always gets log space.  The test pins ``True`` as
    ``exp(log-space g)`` elementwise.

Inputs -> outputs
-----------------
=================  ==================  ============================  ==============================
output             shape               dtype                         consumed by
=================  ==================  ============================  ==============================
``q``              ``(T, Hk, Dk)``     ``conv_output.dtype``         ``chunk_gated_delta_rule`` after
                                                                     ``.unsqueeze(0)`` [vLLM]
                                                                     ``:1440`` -> ``[1, T, Hk, Dk]``
``k``              ``(T, Hk, Dk)``     ``conv_output.dtype``         same, [vLLM] ``:1441``
``v``              ``(T, Hv, Dv)``     ``conv_output.dtype``         same, [vLLM] ``:1442``
``g``              ``(T, Hv)``         fp32                          ``g_non_spec`` -> ``[1, T, Hv]``
                                                                     [vLLM] ``:1443``.  fp32 is the
                                                                     kernel's own output dtype
                                                                     [vLLM] ``:202`` (allocated
                                                                     fp32, computed in fp32 from
                                                                     ``.to(tl.float32)`` operands
                                                                     :128-134, stored
                                                                     ``g_ptr + ...`` :148)
``beta``           ``(T, Hv)``         fp32                          ``beta_non_spec`` -> ``[1, T, Hv]``
                                                                     [vLLM] ``:1444``.  fp32 by the
                                                                     same launcher [vLLM] ``:203``:
                                                                     ``beta_vals = tl.sigmoid(
                                                                     b_vals)`` on the fp32 b
                                                                     (:145) stored into a float32
                                                                     buffer (:149) - **no** cast
                                                                     back to ``b.dtype``
=================  ==================  ============================  ==============================

``Hk = num_k_heads``, ``Dk = head_k_dim``, ``Hv =`` derived value head count,
``Dv = head_v_dim``; ``T`` is the flattened token count of the packed prefill batch (the caller
hands over a 2-D ``(num_tokens, conv_dim)`` buffer [vLLM] ``:1414-1420``, not a ``[B, T, ...]``).

The rank-3 / rank-2 split is forced by the caller: every output is ``.unsqueeze(0)``-ed
[vLLM] ``:1440-1444`` and the chain unpacks ``B, T, H, K = *k.shape, v.shape[-1]`` (or asserts
``len(beta.shape) == 3`` [vLLM] ``chunk.py:216``), so ``q``/``k``/``v`` must be rank 3 and
``g``/``beta`` rank 2 before that unsqueeze.

The mixed ``bf16 q/k/v + fp32 g/beta`` signature is exactly what the kernel produces (:199-203), and
it is load-bearing for numerics: ``beta`` feeds ``chunk_scaled_dot_kkt``/``recompute_w_u`` for every
chunk, so rounding it to bf16 would add one rounding per token before the chain's first multiply.

What this operator deliberately does NOT do
-------------------------------------------
* **No scale.** ``chunk_o`` applies ``scale`` to both of its terms itself (``b_o * scale +
  dot(...) * scale``, ``chunk_o.py:122`` of the vendored FLA sibling), with ``scale = K ** -0.5``
  from ``chunk_gated_delta_rule``'s default [vLLM] ``chunk.py:228-229``; the call site passes no
  scale to this operator, and scaling here would double-apply it.  The kernel agrees: it stores
  ``q_f32`` (:94-99) with no factor.
* **No cumsum of g** - that is chunk step 0 [vLLM] ``chunk.py:37-39``; the kernel stores one ``g``
  per (token, value head) and nothing else (:147-148).
* **No activation on q/k/v** - the conv already applied it [vLLM] ``:1369``; the kernel only loads
  (:76-82, :115-121) and stores (:93-107, :123-125).
* **No output gate / z** - ``z`` goes to ``RMSNormGated`` on a different path [vLLM] ``:1063``.
* **No quantisation, no ``core_attn_out`` buffer, no in-place writes**: every output is a fresh
  contiguous tensor (the launcher's five ``torch.empty`` calls, :199-203), matching "the kernel
  writes new buffers" and keeping the caller's ``mixed_qkv_non_spec`` reusable (the caller keeps it
  for the decode/stitch branches).

Deliberate deviations from the kernel (one, and it is not numeric)
------------------------------------------------------------------
[port] **``Hv`` comes from the packed width, not from ``A_log``.**  The launcher takes
``HV = A_log.shape[0]`` (:190) and then asserts ``qkv_dim == 2*H*K + HV*V`` (:194-196); this port
derives ``Hv`` from ``conv_output.shape[1]`` and *then* validates ``A_log``/``dt_bias`` against it
(raises ``ValueError`` naming the expected count).  Both refuse an inconsistent triple; the port
splits first so the error can name the buffer that disagrees (``value slice not a multiple of
head_v_dim``) instead of only the width assert.  No numeric consequence: any input the kernel
accepts is accepted here with the same ``Hv``, and vice versa.

Everything else the kernel does, this port reproduces: the q|k|v head-major split (:64-120), the
eps-in-the-squared-sum normalisation of q and k only (:84-91, :238), the gate order
``-exp(A_log) * softplus(a + dt_bias)`` with the ``<= 20`` linear branch (:136-140), the fp32
container for both gate outputs (:202-203), ``output_g_exp`` as ``exp(g)`` (:142-143), and the
``(q, k, v, g, beta)`` return order (:248).
"""

from __future__ import annotations

from typing import Tuple

import torch

from ..gdn import l2norm_fwd
from .gating import _sigmoid_gating_decay

__all__ = ["fused_post_conv_prep"]

#: [vLLM] ``fused_gdn_prefill_post_conv.py:50``/``:240`` (``SOFTPLUS_THRESHOLD=20.0``), with the
#: softplus ``beta`` of :138.  They are ``tl.constexpr`` launcher arguments, not parameters of
#: ``fused_post_conv_prep``, so the port keeps them as module constants: both call sites get
#: exactly these values and no caller can ask for anything else.
_GATE_SOFTPLUS_BETA = 1.0
_GATE_SOFTPLUS_THRESHOLD = 20.0


def _as_2d(conv_output: torch.Tensor) -> torch.Tensor:
    """``conv_output`` as a contiguous ``(T, conv_dim)`` view.

    The prefill caller builds a flat ``(num_tokens, conv_dim)`` buffer [vLLM] ``:1414-1420``; a
    leading singleton batch axis is accepted as well because ``(1, T, D)`` is the same buffer seen
    through the ``[B, T, ...]`` convention the rest of the layer uses.
    """
    x = conv_output
    if x.dim() == 3:
        if x.shape[0] != 1:
            raise ValueError(
                "conv_output must be a packed (num_tokens, conv_dim) buffer; a rank-3 input is "
                f"accepted only with a leading singleton batch axis, got {tuple(x.shape)}"
            )
        x = x[0]
    if x.dim() != 2:
        raise ValueError(
            f"conv_output must be (num_tokens, conv_dim), got shape {tuple(conv_output.shape)}"
        )
    if x.shape[1] < 1:
        raise ValueError("conv_output must have at least one channel")
    if not x.is_contiguous():
        # The kernel reads through strides; the port materialises the contiguous copy once so the
        # per-head ``reshape`` below is a pure view.
        x = x.contiguous()
    return x


def _split_packed(
    x: torch.Tensor, num_k_heads: int, head_k_dim: int, head_v_dim: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Split the packed row into ``q (T,Hk,Dk)``, ``k (T,Hk,Dk)``, ``v (T,Hv,Dv)`` + ``Hv``.

    Layout per [vLLM] ``:821-840`` (see the module docstring): q | k | v, each slice head-major.
    ``Hv`` is *derived* - the operator is not told it, which is the reason the call passes
    ``num_k_heads``/``head_k_dim``/``head_v_dim`` and not ``num_v_heads`` [vLLM] ``:1434-1436``.
    """
    if num_k_heads < 1 or head_k_dim < 1 or head_v_dim < 1:
        raise ValueError(
            "num_k_heads, head_k_dim and head_v_dim must all be >= 1, got "
            f"({num_k_heads}, {head_k_dim}, {head_v_dim})"
        )
    tokens = x.shape[0]
    conv_dim = x.shape[1]
    key_dim = num_k_heads * head_k_dim
    value_dim = conv_dim - 2 * key_dim
    if value_dim <= 0:
        raise ValueError(
            f"conv_output's last dim ({conv_dim}) must exceed 2 * num_k_heads * head_k_dim "
            f"({2 * key_dim}): the row is q|k|v [vLLM qwen_gdn_linear_attn.py:821-840]"
        )
    if value_dim % head_v_dim:
        raise ValueError(
            f"the value slice ({value_dim} channels) must be a multiple of head_v_dim "
            f"({head_v_dim}); got {value_dim} - check num_k_heads/head_k_dim against the layer's "
            f"conv_dim = 2*key_dim + value_dim [vLLM qwen_gdn_linear_attn.py:466]"
        )
    num_v_heads = value_dim // head_v_dim
    q = x[:, :key_dim].reshape(tokens, num_k_heads, head_k_dim).contiguous()
    k = x[:, key_dim : 2 * key_dim].reshape(tokens, num_k_heads, head_k_dim).contiguous()
    v = x[:, 2 * key_dim :].reshape(tokens, num_v_heads, head_v_dim).contiguous()
    return q, k, v, num_v_heads


def _gate_tokens(x: torch.Tensor, name: str, tokens: int, num_v_heads: int) -> torch.Tensor:
    """``a``/``b`` as ``(T, Hv)``; any flattened/aliased token-major layout is accepted.

    The gate is a flat ``tok*Hv + hv`` load in the decode kernel [vLLM] ``:1775`` (and
    ``gating.py``'s A1 reads the same), so ``(T*Hv,)`` and ``(1, T, Hv)`` are the same buffer.  The
    head count is *checked* by element count: ``a``/``b`` are per **value** head [vLLM] ``:802-806``,
    so a mismatch means the packed conv layout and the ba projection disagree - the exact class of
    bug that would otherwise show up as a shape error deep inside the chain, or (worse) as silently
    misaligned heads when the counts happen to divide.  A numel that happens to coincide with
    ``T*Hv`` while the per-token stride is different is indistinguishable from the flat form the
    kernel reads, so it is accepted (same leniency as ``gating.py``'s A1).
    """
    if x.numel() != tokens * num_v_heads:
        raise ValueError(
            f"{name} must be ({tokens} tokens, {num_v_heads} value heads) = "
            f"{tokens * num_v_heads} values: the gate is indexed as {name}[tok, value_head] "
            f"[vLLM qwen_gdn_linear_attn.py:1775, :802-806]; got shape {tuple(x.shape)}"
        )
    return x.reshape(tokens, num_v_heads)


def fused_post_conv_prep(
    conv_output: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    num_k_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    apply_l2norm: bool = True,
    output_g_exp: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Turn the packed causal-conv output into ``(q, k, v, g, beta)`` for the GDN chunk chain.

    Signature is the call site's, verbatim ([vLLM] ``:1428-1439``; the warmup at ``:1113-1124``
    passes the same ten arguments in the same order)::

        q, k, v, g, beta = fused_post_conv_prep(
            conv_output=conv_output_prefill,      # (T, 2*Hk*Dk + Hv*Dv), q|k|v
            a=a_prefill, b=b_prefill,             # (T, Hv) gate / beta pre-activations
            A_log=self.A_log, dt_bias=self.dt_bias,   # (Hv,) fp32
            num_k_heads=self.num_k_heads // self.tp_size,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            apply_l2norm=True,
            output_g_exp=False,
        )

    Per-token and stateless: split (q|k|v, head-major), L2-normalise q and k per key head, gate
    ``a``/``b``/``A_log``/``dt_bias`` into the log decay and the delta-rule step, reshape v.  No
    scale, no cumsum, no activation, no in-place writes - see the module docstring for each of
    those exclusions and the line references they rest on.

    Args:
        conv_output: packed conv output ``(T, 2*num_k_heads*head_k_dim + num_v_heads*head_v_dim)``
            in the layer's dtype (bf16 in production); already post-activation, because the conv
            kernel applies the SiLU [vLLM] ``:1369``.  A leading singleton batch axis is accepted.
        a: gate pre-activation, ``(T, num_v_heads)`` (per value head [vLLM] ``:802-806``).
        b: delta-rule beta pre-activation, same layout as ``a``; ``sigmoid(b)`` is the step size.
        A_log: ``(num_v_heads,)`` log decay magnitude, fp32 [vLLM] ``:519-524``; the gate uses
            ``-exp(A_log)``.
        dt_bias: ``(num_v_heads,)`` fp32 time-step bias, added to ``a`` [vLLM] ``:516-518``.
        num_k_heads: key head count (TP-sharded, [vLLM] ``:1434``).
        head_k_dim: key head width.
        head_v_dim: value head width; the value head count is derived from the packed width.
        apply_l2norm: normalise q and k per key head with ``l2norm_fwd``
            (``x / sqrt(sum(x^2) + 1e-6)``).  Both call sites pass ``True``; ``False`` returns the
            raw split.
        output_g_exp: ``False`` (both call sites) -> ``g`` in log space, as the chain needs;
            ``True`` -> the decay ``exp(g)`` [vLLM] ``:142-143`` (the kernel's ``OUTPUT_G_EXP``).

    Returns:
        ``(q, k, v, g, beta)``, all fresh contiguous tensors:

        * ``q`` ``(T, num_k_heads, head_k_dim)`` and ``k`` the same, in ``conv_output.dtype``
          (the kernel's ``q_ptr``/``k_ptr``, [vLLM] ``:199-200``);
        * ``v`` ``(T, num_v_heads, head_v_dim)`` in ``conv_output.dtype`` (``v_ptr``, :201);
        * ``g`` ``(T, num_v_heads)`` log-space forget gate, **always fp32** (``g_ptr``, :202);
        * ``beta`` ``(T, num_v_heads)`` **always fp32** (``beta_ptr``, :203) - not ``b.dtype``.

        The caller adds the batch axis (``.unsqueeze(0)``, [vLLM] ``:1440-1444``) before handing
        them to ``chunk_gated_delta_rule``.
    """
    x = _as_2d(conv_output)
    q, k, v, num_v_heads = _split_packed(x, num_k_heads, head_k_dim, head_v_dim)
    tokens = x.shape[0]

    if apply_l2norm:
        # [vLLM] :84-91 - eps inside the squared sum (L2NORM_EPS=1e-6, :238); the port keeps the
        # reduction in fp32 and casts back to x.dtype, exactly like the kernel's store (:97, :105).
        q = l2norm_fwd(q)
        k = l2norm_fwd(k)

    a_tok = _gate_tokens(a, "a", tokens, num_v_heads)
    b_tok = _gate_tokens(b, "b", tokens, num_v_heads)
    A_log_flat = A_log.reshape(-1)
    dt_bias_flat = dt_bias.reshape(-1)
    for name, tensor in (("A_log", A_log_flat), ("dt_bias", dt_bias_flat)):
        if tensor.numel() != num_v_heads:
            raise ValueError(
                f"{name} must have {num_v_heads} elements (one per value head, [vLLM] "
                f"qwen_gdn_linear_attn.py:516-524), got {tensor.numel()}"
            )

    # [vLLM] :131-140, one convention for the whole package: the kernel loads every gate operand
    # with ``.to(tl.float32)`` and this helper is the decode-step port's, not a second transcription
    # of the softplus.  ``dtype=torch.float32`` pins the kernel's fixed fp32 container (:202): even a
    # fp64 ``a`` (never produced by the layer) cannot make ``g`` fp64 here.
    g = _sigmoid_gating_decay(
        a_tok,
        A_log_flat,
        dt_bias_flat,
        beta=_GATE_SOFTPLUS_BETA,
        threshold=_GATE_SOFTPLUS_THRESHOLD,
        dtype=torch.float32,
    )
    if output_g_exp:
        # [vLLM] :142-143: ``if OUTPUT_G_EXP: g_vals = tl.exp(g_vals)`` - elementwise, still fp32.
        g = torch.exp(g)

    # [vLLM] :134/:145/:149: ``b_vals`` is ``.to(tl.float32)``, ``beta_vals = tl.sigmoid(b_vals)``,
    # and the launcher allocated a float32 ``beta`` (:203) which the store does not downcast.  So the
    # value keeps the full fp32 sigmoid - casting it to ``b.dtype`` would add one bf16 rounding per
    # (token, value head) *before* the chunk chain's first multiply.
    beta = torch.sigmoid(b_tok.to(torch.float32))
    return q, k, v, g, beta
