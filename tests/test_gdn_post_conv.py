"""Contract tests for the torch port of vLLM's ``fused_post_conv_prep`` (GDN prefill post-conv prep).

What is under test
------------------
``tileinfer/vllm/ops/fla/post_conv.py``: the operator vLLM's GDN layer calls at
``qwen_gdn_linear_attn.py:1428`` (prefill) / ``:1113`` (profiling warmup) to turn the packed
causal-conv output into ``(q, k, v, g, beta)`` for the chunk chain.

The upstream body is ``reference/vllm-0.23/vllm__model_executor__layers__fla__ops__\\
fused_gdn_prefill_post_conv.py`` - one Triton kernel (:19-149) plus a launcher (:152-248) that
allocates ``q``/``k``/``v`` in ``conv_output.dtype`` and **both** gate outputs in ``torch.float32``
(:199-203).  The oracles below are still written *independently* of that body (a per-token fp64
loop, the caller's own split, and a deliberately wrong normalisation) so a transcription slip in the
port cannot be mirrored by the test - but every *contract* they assert (packed order, eps placement,
gate form, output dtypes) is now read from the real source and cited.

The dtype contract this file pins, straight out of the launcher (:199-203) and the kernel's stores
(:148-149)::

    q, k : (T, Hk, Dk)  conv_output.dtype      # kernel stores ..to(q_ptr.dtype.element_ty) :97
    v    : (T, Hv, Dv)  conv_output.dtype      # no cast, :125
    g    : (T, Hv)      torch.float32          # :202, computed from .to(tl.float32) operands
    beta : (T, Hv)      torch.float32          # :203, sigmoid of the fp32 b (:134, :145)

``beta`` in bf16 was this port's earlier behaviour and is a real numeric deviation: it feeds the
whole chunk chain (``chunk_scaled_dot_kkt`` / ``recompute_w_u``), so one bf16 rounding lands before
the chain's first multiply on every token.  ``test_beta_is_fp32_for_bf16_inputs`` pins the fix.

Three independent oracles, none of which calls the module under test
--------------------------------------------------------------------
* ``_oracle`` - a plain Python loop over (token, head, channel) in fp64, using ``math.sqrt`` /
  ``math.log1p`` / ``math.exp`` / ``1/(1+exp(-x))``.  It reads the packed row itself
  (``q | k | v``, head-major, [vLLM] kernel ``:76-120`` and caller ``:821-840``), does the L2 norm
  with ``eps`` inside the squared sum ([vLLM] ``:85-91``, ``L2NORM_EPS=1e-6`` at ``:238``), applies
  the gate ``-exp(A_log) * softplus(a + dt_bias)`` with the ``<= 20`` linear branch ([vLLM]
  ``:136-140``) and the delta-rule ``sigmoid(b)`` in fp32 ([vLLM] ``:145``).
* ``_caller_rearrange_mixed_qkv`` - a transcription of the caller's own split
  ([vLLM] caller ``:808-842``), used both as the ``apply_l2norm=False`` oracle and as the "the port
  splits the buffer exactly the way vLLM's non-post-conv branch does" check.
* ``_l2norm_max_norm`` - the *wrong* normalisation convention (``x / max(||x||, eps)``), used as a
  negative control so the ``eps``-in-the-squared-sum convention cannot regress silently.

Geometry
--------
``HK, HV, DK, DV = 16, 32, 128, 128`` and ``hidden = 2048`` are the production shapes
(Qwen3-Next/Qwen3.5 GDN at TP=1; the same constants as ``tests/test_gdn_reorder.py``), so the
packed buffer is ``2*16*128 + 32*128 = 8192`` wide - the real conv width-4 output.  The operator
never sees ``hidden`` (it consumes the conv output, not the hidden states); the constant is built
into the buffer construction to keep the production case honest.
"""

from __future__ import annotations

import inspect
import math

import pytest
import torch

from tileinfer.vllm.ops.fla.post_conv import fused_post_conv_prep

# Production GDN shape (Qwen3-Next/Qwen3.5, TP=1).
HK, HV, DK, DV = 16, 32, 128, 128
HIDDEN = 2048
CONV_WIDTH = 4
CONV_DIM = 2 * HK * DK + HV * DV  # 8192 = q|k|v packed width

#: ``(rtol, atol)`` per output dtype, for a comparison against the fp64 oracle.  bf16 carries 8
#: mantissa bits, so one rounding is ~2e-3 relative; fp32 holds ~1e-7.
_TOL = {
    torch.float64: (1e-12, 1e-12),
    torch.float32: (2e-5, 2e-6),
    torch.bfloat16: (8e-3, 1e-4),
}


# ---------------------------------------------------------------------------------------------
# The independent oracle: one loop over (token, head, channel), fp64, no torch vectorisation
# ---------------------------------------------------------------------------------------------


def _oracle(
    conv_output,
    a,
    b,
    A_log,
    dt_bias,
    num_k_heads,
    head_k_dim,
    head_v_dim,
    *,
    apply_l2norm=True,
    output_g_exp=False,
    eps=1e-6,
):
    """Plain-loop fp64 reference for ``fused_post_conv_prep``.

    Returns ``(q, k, v, g, beta)`` as fp64 tensors.  Every scalar is computed with :mod:`math` on
    Python floats, i.e. from the *exact* values of the (possibly bf16) inputs - that is what makes
    the bf16 tolerance meaningful instead of tautological.
    """
    x = conv_output[0] if conv_output.dim() == 3 else conv_output
    rows = x.tolist()
    tokens = len(rows)
    conv_dim = len(rows[0])
    key_dim = num_k_heads * head_k_dim
    num_v_heads = (conv_dim - 2 * key_dim) // head_v_dim

    q = [[[0.0] * head_k_dim for _ in range(num_k_heads)] for _ in range(tokens)]
    k = [[[0.0] * head_k_dim for _ in range(num_k_heads)] for _ in range(tokens)]
    v = [[[0.0] * head_v_dim for _ in range(num_v_heads)] for _ in range(tokens)]
    for t, row in enumerate(rows):
        for h in range(num_k_heads):
            for d in range(head_k_dim):
                q[t][h][d] = row[h * head_k_dim + d]
                k[t][h][d] = row[key_dim + h * head_k_dim + d]
        for h in range(num_v_heads):
            for d in range(head_v_dim):
                v[t][h][d] = row[2 * key_dim + h * head_v_dim + d]

    if apply_l2norm:
        for grid in (q, k):
            for t in range(tokens):
                for h in range(num_k_heads):
                    squared = sum(c * c for c in grid[t][h])
                    rstd = 1.0 / math.sqrt(squared + eps)
                    for d in range(head_k_dim):
                        grid[t][h][d] *= rstd

    a_rows, b_rows = a.reshape(tokens, num_v_heads).tolist(), b.reshape(tokens, num_v_heads).tolist()
    A_list = A_log.reshape(-1).tolist()
    dt_list = dt_bias.reshape(-1).tolist()
    g = [[0.0] * num_v_heads for _ in range(tokens)]
    beta = [[0.0] * num_v_heads for _ in range(tokens)]
    for t in range(tokens):
        for h in range(num_v_heads):
            # -exp(A_log) * softplus(a + dt_bias); softplus switches to its linear branch above 20.
            xv = a_rows[t][h] + dt_list[h]
            softplus = math.log1p(math.exp(xv)) if xv <= 20.0 else xv
            gv = -math.exp(A_list[h]) * softplus
            if output_g_exp:
                gv = math.exp(gv)
            g[t][h] = gv
            beta[t][h] = 1.0 / (1.0 + math.exp(-b_rows[t][h]))

    return (
        torch.tensor(q, dtype=torch.float64),
        torch.tensor(k, dtype=torch.float64),
        torch.tensor(v, dtype=torch.float64),
        torch.tensor(g, dtype=torch.float64),
        torch.tensor(beta, dtype=torch.float64),
    )


def _caller_rearrange_mixed_qkv(mixed_qkv, num_k_heads, head_k_dim, head_v_dim):
    """vLLM's ``rearrange_mixed_qkv`` [vLLM ``:808-842``], transcribed line by line.

    This is the caller's own split of the *same* buffer on the branch that does not go through
    ``fused_post_conv_prep`` [vLLM] ``:1446-1448``, so it is the strongest available statement about
    the packed layout.  Returns ``(q, k, v)`` with the leading batch axis squeezed off.
    """
    seq_len = mixed_qkv.shape[0]
    key_dim = num_k_heads * head_k_dim
    # [vLLM] :821-823 - q_dim, k_dim and v_dim are the three slice widths of the packed row.
    value_dim = mixed_qkv.shape[1] - 2 * key_dim
    query, key, value = torch.split(mixed_qkv, [key_dim, key_dim, value_dim], dim=-1)
    fused = torch.cat([query.reshape(-1), key.reshape(-1), value.reshape(-1)], dim=0)
    q_size = seq_len * key_dim
    k_size = seq_len * key_dim
    q_contig = fused[0:q_size]
    k_contig = fused[q_size : q_size + k_size]
    v_contig = fused[q_size + k_size :]
    return (
        q_contig.view(1, seq_len, -1, head_k_dim).squeeze(0),
        k_contig.view(1, seq_len, -1, head_k_dim).squeeze(0),
        v_contig.view(1, seq_len, -1, head_v_dim).squeeze(0),
    )


def _l2norm_max_norm(x, eps=1e-6):
    """The WRONG convention (``x / max(||x||, eps)``) - negative control for the eps test."""
    norm = x.to(torch.float64).pow(2).sum(-1, keepdim=True).sqrt()
    return x.to(torch.float64) / norm.clamp(min=eps)


def _assert_matches(got, want, label):
    """Compare a port output against the fp64 oracle with the tolerance of ``got.dtype``."""
    rtol, atol = _TOL[got.dtype]
    torch.testing.assert_close(
        got.to(torch.float64), want.to(torch.float64), rtol=rtol, atol=atol, msg=label
    )


# ---------------------------------------------------------------------------------------------
# Input construction: a real width-4 causal conv over a packed pre-conv buffer
# ---------------------------------------------------------------------------------------------


def _causal_conv_width4(preconv, weight, tokens):
    """``out[t] = sum_tap weight[:, tap] * x[t - (W-1-tap)]``, zero before the sequence start.

    The zero past is the implicit initial conv state of a fresh prefill, which is what the caller's
    ``causal_conv1d_fn`` does for a sequence with no cached state [vLLM] ``:1365-1375``.  The
    operator under test must not care how the buffer was produced; building it with a real width-4
    conv keeps the production-geometry case honest.  (The conv kernel also applies the SiLU
    activation [vLLM] ``:1369``; that is upstream of the operator and only changes the buffer's
    statistics, so it is left out here.)  Only the first ``tokens`` outputs are returned.
    """
    total = preconv.shape[0]
    out = torch.zeros_like(preconv)
    for tap in range(CONV_WIDTH):
        lag = CONV_WIDTH - 1 - tap
        if lag == 0:
            out += weight[:, tap] * preconv
        else:
            out[lag:] += weight[:, tap] * preconv[: total - lag]
    return out[:tokens]


def _production_inputs(dtype=torch.bfloat16, tokens=17, seed=0):
    """Packed conv output + per-value-head gate inputs at the production geometry."""
    gen = torch.Generator().manual_seed(seed)
    preconv = torch.randn(tokens, CONV_DIM, generator=gen)
    weight = torch.randn(CONV_DIM, CONV_WIDTH, generator=gen) / CONV_WIDTH
    conv_output = _causal_conv_width4(preconv, weight, tokens).to(dtype)
    a = torch.randn(tokens, HV, generator=gen)
    b = torch.randn(tokens, HV, generator=gen)
    A_log = torch.rand(HV, generator=gen) * 2.0  # spans "tiny decay" .. "fast decay"
    dt_bias = torch.randn(HV, generator=gen)
    return conv_output, a.to(dtype), b.to(dtype), A_log, dt_bias


def _small_inputs(dtype=torch.float32, tokens=5, seed=1):
    """Cheap geometry (4 key heads x 8, 8 value heads x 8) for the exactness-oriented cases."""
    hk, hv, dk, dv = 4, 8, 8, 8
    gen = torch.Generator().manual_seed(seed)
    conv_dim = 2 * hk * dk + hv * dv
    conv_output = torch.randn(tokens, conv_dim, generator=gen)
    a = torch.randn(tokens, hv, generator=gen)
    b = torch.randn(tokens, hv, generator=gen)
    A_log = torch.rand(hv, generator=gen) * 2.0
    dt_bias = torch.randn(hv, generator=gen)
    return conv_output.to(dtype), a.to(dtype), b.to(dtype), A_log, dt_bias, (hk, hv, dk, dv)


# ---------------------------------------------------------------------------------------------
# 1. the signature the caller uses
# ---------------------------------------------------------------------------------------------


def test_signature_is_the_two_call_sites_verbatim():
    """[vLLM] ``:1428-1439`` and ``:1113-1124`` pass these ten names, in this order."""
    parameters = inspect.signature(fused_post_conv_prep).parameters
    assert list(parameters) == [
        "conv_output",
        "a",
        "b",
        "A_log",
        "dt_bias",
        "num_k_heads",
        "head_k_dim",
        "head_v_dim",
        "apply_l2norm",
        "output_g_exp",
    ]
    # Both call sites pass these two explicitly, with these values.
    assert parameters["apply_l2norm"].default is True
    assert parameters["output_g_exp"].default is False


# ---------------------------------------------------------------------------------------------
# 2. production geometry (hidden 2048, 16 k-heads x 128, 32 v-heads x 128, conv width 4)
# ---------------------------------------------------------------------------------------------


def test_production_geometry_shapes_and_dtypes():
    conv_output, a, b, A_log, dt_bias = _production_inputs(torch.bfloat16, tokens=17)
    assert conv_output.shape == (17, 2 * HK * DK + HV * DV)  # 8192, the real conv output width
    q, k, v, g, beta = fused_post_conv_prep(
        conv_output, a, b, A_log, dt_bias, HK, DK, DV, apply_l2norm=True, output_g_exp=False
    )

    assert q.shape == (17, HK, DK) and q.dtype is torch.bfloat16 and q.is_contiguous()
    assert k.shape == (17, HK, DK) and k.dtype is torch.bfloat16 and k.is_contiguous()
    assert v.shape == (17, HV, DV) and v.dtype is torch.bfloat16 and v.is_contiguous()
    # g and beta are BOTH fp32, whatever the inputs are: the launcher allocates both float32
    # ([vLLM] :202-203) because the kernel computes them from `.to(tl.float32)` operands (:128-134)
    # and stores without downcasting (:148-149).
    assert g.shape == (17, HV) and g.dtype is torch.float32
    assert beta.shape == (17, HV) and beta.dtype is torch.float32
    # The chain unpacks [B, T, H, K] after the caller's unsqueeze(0) and asserts 3-D beta.
    assert q.unsqueeze(0).shape == (1, 17, HK, DK)
    assert beta.unsqueeze(0).shape == (1, 17, HV)


def test_production_geometry_multi_token_matches_the_loop_oracle():
    conv_output, a, b, A_log, dt_bias = _production_inputs(torch.bfloat16, tokens=17, seed=3)
    got = fused_post_conv_prep(conv_output, a, b, A_log, dt_bias, HK, DK, DV)
    want = _oracle(conv_output, a, b, A_log, dt_bias, HK, DK, DV)

    for got_tensor, want_tensor, label in zip(
        got, want, ("q", "k", "v", "g", "beta")
    ):
        _assert_matches(got_tensor, want_tensor, f"production T=17 bf16: {label}")

    # Sanity on the oracle itself: the normalised rows have unit-ish norm, and beta is a step size.
    assert torch.allclose(got[0].float().norm(dim=-1), torch.ones(17, HK), atol=2e-2)
    assert bool(((got[4] > 0) & (got[4] < 1)).all())
    assert bool((got[3] <= 0).all())  # g = -exp(A_log) * softplus(...) <= 0, so the decay is in (0,1]


def test_single_token_matches_the_loop_oracle():
    """T = 1: the degenerate prefill (one token, no chunk) must go through the same path."""
    conv_output, a, b, A_log, dt_bias = _production_inputs(torch.bfloat16, tokens=1, seed=5)
    assert conv_output.shape == (1, CONV_DIM)
    got = fused_post_conv_prep(conv_output, a, b, A_log, dt_bias, HK, DK, DV)
    want = _oracle(conv_output, a, b, A_log, dt_bias, HK, DK, DV)

    for got_tensor, want_tensor, label in zip(got, want, ("q", "k", "v", "g", "beta")):
        assert got_tensor.shape[0] == 1
        _assert_matches(got_tensor, want_tensor, f"single token bf16: {label}")


# ---------------------------------------------------------------------------------------------
# 3. dtype: bf16 vs fp32
# ---------------------------------------------------------------------------------------------


def test_fp32_matches_the_loop_oracle():
    conv_output, a, b, A_log, dt_bias, (hk, hv, dk, dv) = _small_inputs(torch.float32, tokens=5)
    got = fused_post_conv_prep(conv_output, a, b, A_log, dt_bias, hk, dk, dv)
    want = _oracle(conv_output, a, b, A_log, dt_bias, hk, dk, dv)

    for got_tensor, want_tensor, label in zip(got, want, ("q", "k", "v", "g", "beta")):
        _assert_matches(got_tensor, want_tensor, f"fp32: {label}")
    # Every output follows the input dtype for q/k/v here; the two gate outputs are always fp32.
    assert got[0].dtype is torch.float32 and got[4].dtype is torch.float32


def test_beta_is_fp32_for_bf16_inputs_and_keeps_the_fp32_sigmoid():
    """The deviation this port had, pinned: ``beta`` is fp32 (the launcher's ``beta_ptr``, :203).

    bf16 ``b`` must be widened to fp32, sigmoided there and **not** cast back: casting back would
    add one bf16 rounding per (token, value head) before the chunk chain multiplies with ``beta``
    (``chunk_scaled_dot_kkt_fwd`` / ``recompute_w_u``).  The two assertions below are different
    failures - dtype, then the extra rounding - and the third proves the extra rounding is real for
    this input, so the check is not vacuous.
    """
    conv_output, a, b, A_log, dt_bias = _production_inputs(torch.bfloat16, tokens=17, seed=31)
    q, k, v, g, beta = fused_post_conv_prep(conv_output, a, b, A_log, dt_bias, HK, DK, DV)

    assert q.dtype is torch.bfloat16 and k.dtype is torch.bfloat16 and v.dtype is torch.bfloat16
    assert beta.dtype is torch.float32
    # The value is exactly sigmoid(b widened to fp32) - no bf16 round trip anywhere.
    assert torch.equal(beta, torch.sigmoid(b.reshape(17, HV).to(torch.float32)))
    # ... and the bf16 round trip is observably different for this input, so "no cast back" matters.
    round_tripped = torch.sigmoid(b.reshape(17, HV).to(torch.float32)).to(torch.bfloat16)
    assert not torch.equal(beta, round_tripped.to(torch.float32))

    # Same call in fp32: identical container and identical value (the cast is a no-op).
    conv32, a32, b32 = conv_output.float(), a.float(), b.float()
    beta32 = fused_post_conv_prep(conv32, a32, b32, A_log, dt_bias, HK, DK, DV)[4]
    assert beta32.dtype is torch.float32
    torch.testing.assert_close(beta, beta32, rtol=1e-2, atol=1e-3)

    # g is fp32 for the same reason (the kernel's own launcher, :202), for bf16 and fp32 inputs.
    assert g.dtype is torch.float32
    assert fused_post_conv_prep(conv32, a32, b32, A_log, dt_bias, HK, DK, DV)[3].dtype is (
        torch.float32
    )


def test_bf16_is_the_fp32_result_rounded_to_bf16():
    """Same values, two precisions: the bf16 error must be the input/storage rounding, nothing else."""
    conv32, a32, b32, A_log, dt_bias, (hk, hv, dk, dv) = _small_inputs(torch.float32, tokens=6)
    out32 = fused_post_conv_prep(conv32, a32, b32, A_log, dt_bias, hk, dk, dv)
    conv16, a16, b16 = conv32.to(torch.bfloat16), a32.to(torch.bfloat16), b32.to(torch.bfloat16)
    out16 = fused_post_conv_prep(conv16, a16, b16, A_log, dt_bias, hk, dk, dv)

    # The bf16 run reads bf16-rounded inputs, so compare against the fp32 run's inputs likewise
    # rounded: q/k/v stay within one bf16 rounding of the fp32 result.
    torch.testing.assert_close(
        out16[0].to(torch.float64), out32[0].to(torch.float64), rtol=8e-3, atol=1e-4,
        msg="q: bf16 vs fp32",
    )
    torch.testing.assert_close(
        out16[2].to(torch.float64), out32[2].to(torch.float64), rtol=8e-3, atol=1e-4,
        msg="v: bf16 vs fp32",
    )
    # g stays fp32 in both calls, so it only inherits the bf16 input rounding of a (<= 2^-9).
    assert out16[3].dtype is torch.float32 and out32[3].dtype is torch.float32
    torch.testing.assert_close(
        out16[3].to(torch.float64), out32[3].to(torch.float64), rtol=1e-2, atol=1e-3,
        msg="g: bf16 inputs vs fp32 inputs",
    )
    # beta is fp32 in BOTH calls: the container is fixed by the kernel's launcher (:203), so the
    # bf16 run only differs through the rounding of `b` itself, never through a bf16 store.
    assert out16[4].dtype is torch.float32 and out32[4].dtype is torch.float32
    torch.testing.assert_close(
        out16[4].to(torch.float64), out32[4].to(torch.float64), rtol=1e-2, atol=1e-3,
        msg="beta: bf16 inputs vs fp32 inputs",
    )


# ---------------------------------------------------------------------------------------------
# 4. the split: identical to the caller's own rearrange_mixed_qkv
# ---------------------------------------------------------------------------------------------


def test_apply_l2norm_false_returns_exactly_the_callers_split():
    """``apply_l2norm=False`` -> raw split, bit-identical to [vLLM] ``:808-842``'s own arithmetic."""
    conv_output, a, b, A_log, dt_bias = _production_inputs(torch.float32, tokens=7, seed=11)
    q, k, v, _g, _beta = fused_post_conv_prep(
        conv_output, a, b, A_log, dt_bias, HK, DK, DV, apply_l2norm=False
    )
    want_q, want_k, want_v = _caller_rearrange_mixed_qkv(conv_output, HK, DK, DV)

    assert torch.equal(q, want_q), "q is not the caller's q slice"
    assert torch.equal(k, want_k), "k is not the caller's k slice"
    assert torch.equal(v, want_v), "v is not the caller's v slice"

    want = _oracle(conv_output, a, b, A_log, dt_bias, HK, DK, DV, apply_l2norm=False)
    for got_tensor, want_tensor, label in zip((q, k, v), want[:3], ("q", "k", "v")):
        _assert_matches(got_tensor, want_tensor, f"apply_l2norm=False: {label}")


def test_apply_l2norm_true_is_l2norm_fwd_on_that_split():
    """The normalisation must be the package's ``l2norm_fwd``, not a second implementation."""
    from tileinfer.vllm.ops.gdn import l2norm_fwd

    conv_output, a, b, A_log, dt_bias = _production_inputs(torch.float32, tokens=7, seed=13)
    q, k, _v, _g, _beta = fused_post_conv_prep(
        conv_output, a, b, A_log, dt_bias, HK, DK, DV, apply_l2norm=True
    )
    want_q, want_k, _want_v = _caller_rearrange_mixed_qkv(conv_output, HK, DK, DV)
    assert torch.equal(q, l2norm_fwd(want_q))
    assert torch.equal(k, l2norm_fwd(want_k))
    # ... and only q/k are normalised: v is the raw slice.
    assert not torch.allclose(_v.float().norm(dim=-1), torch.ones(7, HV), atol=1e-3)


def test_l2norm_eps_sits_inside_the_squared_sum_not_on_the_norm():
    """The convention ``x / sqrt(sum(x^2) + eps)``, pinned by a head whose norm is << sqrt(eps).

    With ``||x|| = 1e-5`` and ``eps = 1e-6`` the two conventions differ by 1000x here:
    ``x/sqrt(1e-10 + 1e-6) ≈ 0.01`` versus ``x/max(1e-5, 1e-6) = 1.0``.
    """
    hk, hv, dk, dv = 2, 2, 4, 4
    tokens = 1
    key_dim = hk * dk
    conv_output = torch.zeros(tokens, 2 * key_dim + hv * dv, dtype=torch.float32)
    conv_output[0, 0:dk] = torch.tensor([1e-5, 0.0, 0.0, 0.0])  # q head 0: norm 1e-5
    conv_output[0, key_dim : key_dim + dk] = torch.tensor([3e-6, 4e-6, 0.0, 0.0])  # k head 0: 5e-6
    a = torch.zeros(tokens, hv)
    b = torch.zeros(tokens, hv)
    A_log = torch.zeros(hv)
    dt_bias = torch.zeros(hv)

    q, k, _v, _g, _beta = fused_post_conv_prep(conv_output, a, b, A_log, dt_bias, hk, dk, dv)

    want_q, want_k = _oracle(conv_output, a, b, A_log, dt_bias, hk, dk, dv)[:2]
    _assert_matches(q, want_q, "tiny-norm q")
    _assert_matches(k, want_k, "tiny-norm k")
    # The value the wrong convention would produce, 100x away - the check above is not vacuous.
    wrong = _l2norm_max_norm(conv_output[:, 0:key_dim].reshape(1, hk, dk))
    assert abs(float(q[0, 0, 0]) - float(wrong[0, 0, 0])) > 0.5
    assert float(q[0, 0, 0]) == pytest.approx(1e-5 / math.sqrt(1e-10 + 1e-6), rel=1e-5)


# ---------------------------------------------------------------------------------------------
# 5. the gate: one convention with the decode port
# ---------------------------------------------------------------------------------------------


def test_g_and_beta_are_the_decode_gating_convention():
    from tileinfer.vllm.ops.fla.gating import _sigmoid_gating_decay

    conv_output, a, b, A_log, dt_bias = _production_inputs(torch.bfloat16, tokens=9, seed=17)
    _q, _k, _v, g, beta = fused_post_conv_prep(conv_output, a, b, A_log, dt_bias, HK, DK, DV)

    # [vLLM] :131-140: the same formula the decode kernel documents, through the package's helper.
    assert torch.equal(g, _sigmoid_gating_decay(a.reshape(9, HV), A_log, dt_bias))
    # [vLLM] :134/:145: sigmoid of the fp32-widened b, kept in fp32 (:203) - no dtype cast back.
    assert torch.equal(beta, torch.sigmoid(b.reshape(9, HV).to(torch.float32)))
    # The gate is per value head: two calls whose `a` differs only in one head must differ only
    # there, i.e. no accidental reduction across heads.
    a2 = a.clone()
    a2[:, HV // 2] += 1.0
    g2 = fused_post_conv_prep(conv_output, a2, b, A_log, dt_bias, HK, DK, DV)[3]
    changed = (g2 != g).any(dim=0)
    assert bool(changed[HV // 2]) and int(changed.sum()) == 1


def test_output_g_exp_true_returns_the_decay_of_the_log_gate():
    """[vLLM] ``:142-143``: ``True`` -> ``exp(g)`` elementwise, still fp32, still in (0, 1]."""
    conv_output, a, b, A_log, dt_bias = _production_inputs(torch.bfloat16, tokens=4, seed=19)
    g_log = fused_post_conv_prep(conv_output, a, b, A_log, dt_bias, HK, DK, DV, output_g_exp=False)[3]
    g_exp = fused_post_conv_prep(conv_output, a, b, A_log, dt_bias, HK, DK, DV, output_g_exp=True)[3]

    assert g_exp.dtype is torch.float32
    torch.testing.assert_close(g_exp, torch.exp(g_log), rtol=1e-6, atol=0)
    assert bool(((g_exp > 0) & (g_exp <= 1)).all())
    want = _oracle(conv_output, a, b, A_log, dt_bias, HK, DK, DV, output_g_exp=True)[3]
    _assert_matches(g_exp, want, "output_g_exp=True: g")


# ---------------------------------------------------------------------------------------------
# 6. the hand-off: what it returns is what the chunk chain consumes
# ---------------------------------------------------------------------------------------------


def test_handoff_to_the_chunk_chain_is_wire_compatible():
    """post_conv -> unsqueeze -> chunk chain == manual split -> l2norm -> gate -> chunk chain.

    This is an ABI/plumbing test (ranks, dtypes, the 3-D beta the chain asserts), not a second
    math test: both sides go through the same chain, and the manual side builds its five tensors
    the way vLLM's *non*-post-conv branch does it ([vLLM] caller ``:1446-1448``) plus the shared gate
    helper, with the dtype contract the real launcher fixes (bf16 q/k/v + fp32 g/beta, :199-203).
    So it also proves the chain accepts a **fp32 beta next to bf16 q/k/v**, which is the production
    combination.
    """
    from tileinfer.vllm.ops.fla import chunk_gated_delta_rule
    from tileinfer.vllm.ops.fla.gating import _sigmoid_gating_decay
    from tileinfer.vllm.ops.gdn import l2norm_fwd

    tokens = 9
    conv_output, a, b, A_log, dt_bias = _production_inputs(torch.bfloat16, tokens=tokens, seed=23)
    q, k, v, g, beta = fused_post_conv_prep(conv_output, a, b, A_log, dt_bias, HK, DK, DV)

    qm, km, vm = _caller_rearrange_mixed_qkv(conv_output, HK, DK, DV)
    qm, km = l2norm_fwd(qm), l2norm_fwd(km)
    gm = _sigmoid_gating_decay(a.reshape(tokens, HV), A_log, dt_bias)
    # fp32, like the kernel's beta buffer ([vLLM] :203) - the old bf16 form is what this test caught.
    bm = torch.sigmoid(b.reshape(tokens, HV).to(torch.float32))
    assert bm.dtype is torch.float32 and gm.dtype is torch.float32

    states = [torch.zeros(1, HV, DV, DK, dtype=torch.float32) for _ in range(2)]
    common = dict(
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,  # [vLLM] :1529 - the chain must not re-normalise
    )
    o_port, s_port = chunk_gated_delta_rule(
        q.unsqueeze(0),
        k.unsqueeze(0),
        v.unsqueeze(0),
        g.unsqueeze(0),
        beta.unsqueeze(0),
        initial_state=states[0],
        **common,
    )
    o_manual, s_manual = chunk_gated_delta_rule(
        qm.unsqueeze(0),
        km.unsqueeze(0),
        vm.unsqueeze(0),
        gm.unsqueeze(0),
        bm.unsqueeze(0),
        initial_state=states[1],
        **common,
    )
    assert o_port.shape == (1, tokens, HV, DV) and o_port.dtype is torch.bfloat16
    assert torch.equal(o_port, o_manual)
    assert torch.equal(s_port, s_manual)


# ---------------------------------------------------------------------------------------------
# 7. guards
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mutation, message",
    [
        # 2*16*128 = 4096 channels of q|k, so 8191 leaves a value slice that is not a whole head.
        (lambda x: x[:, :-1], "multiple of head_v_dim"),
        # Too narrow for q|k at all.
        (lambda x: x[:, : 2 * HK * DK], "must exceed 2 \\* num_k_heads \\* head_k_dim"),
        # Rank 4 is not a packed token-major buffer.
        (lambda x: x.unsqueeze(-1).unsqueeze(-1), "must be \\(num_tokens, conv_dim\\)"),
    ],
)
def test_rejects_a_conv_output_that_is_not_a_packed_qkv_row(mutation, message):
    conv_output, _a, _b, _A, _d = _production_inputs(torch.float32, tokens=3)
    a = torch.zeros(3, HV)
    b = torch.zeros(3, HV)
    A_log = torch.zeros(HV)
    dt_bias = torch.zeros(HV)
    with pytest.raises(ValueError, match=message):
        fused_post_conv_prep(mutation(conv_output), a, b, A_log, dt_bias, HK, DK, DV)


def test_rejects_gate_inputs_with_the_wrong_head_count():
    conv_output, _a, _b, A_log, dt_bias = _production_inputs(torch.float32, tokens=3)
    a = torch.zeros(3, HV)
    b = torch.zeros(3, HV)
    # `a`/`b` are per VALUE head [vLLM :802-806]: 16 heads cannot line up with a 32-head v slice.
    with pytest.raises(ValueError, match="value heads"):
        fused_post_conv_prep(conv_output, a[:, :16], b, A_log, dt_bias, HK, DK, DV)
    # ... and the per-head gate parameters must match that count too.
    with pytest.raises(ValueError, match="A_log must have 32"):
        fused_post_conv_prep(conv_output, a, b, A_log[:8], dt_bias, HK, DK, DV)
    with pytest.raises(ValueError, match="dt_bias must have 32"):
        fused_post_conv_prep(conv_output, a, b, A_log, dt_bias[:8], HK, DK, DV)


def test_does_not_mutate_its_inputs():
    conv_output, a, b, A_log, dt_bias = _production_inputs(torch.float32, tokens=6, seed=29)
    copies = [t.clone() for t in (conv_output, a, b, A_log, dt_bias)]
    fused_post_conv_prep(conv_output, a, b, A_log, dt_bias, HK, DK, DV)
    for original, copy, label in zip((conv_output, a, b, A_log, dt_bias), copies,
                                    ("conv_output", "a", "b", "A_log", "dt_bias")):
        assert torch.equal(original, copy), f"{label} was mutated"
