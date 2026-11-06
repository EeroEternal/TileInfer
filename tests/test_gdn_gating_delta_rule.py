"""Contract tests for the torch port of FLA ``fused_sigmoid_gating_delta_rule_update``.

Why this file is not called ``tests/test_gdn_gating.py``: that path is already taken by a committed
test for a *different* operator (``tileinfer.kernels.gdn.fused_gdn_gating``, the M-GDN gating
fusion, commit 20b9a44).  The task forbids modifying existing files, so the new tests live here.

Everything below is CPU, fp64-first.  Two independent oracles, neither of which calls the module
under test:

* ``_oracle`` — the literal per-token / per-head recurrence transcribed from the kernel
  (``runtime/vllm-api/fla_fused_sigmoid_gating.py`` lines 129-154), written one token at a time;
* ``_prefill_form`` — the *whole-sequence* closed form, using explicit ``K x K`` transfer matrices
  ``P_t = d_t (I - beta_t k_t k_t^T)`` and their prefix/suffix products, with **no state
  recursion at all**::

      h_t = h_0 P_0 ... P_t + sum_{s<=t} beta_s v_s k_s^T (P_{s+1} ... P_t)

The highest-value test is ``test_decode_step_applied_t_times_equals_sequence_form``: applying the
single-token decode step ``T`` times (state carried in place, which is how vLLM drives it) must
reproduce the sequence form, and both must reproduce the independent step-by-step oracle.  That
pins the token order, the decay/delta ordering and the state carry in one assertion.
"""

from __future__ import annotations

import pytest
import torch

from tileinfer.vllm.ops.fla.gating import (
    _sigmoid_gating_decay,
    fused_sigmoid_gating_delta_rule_update as fused_gdn_gating_update,
)

# ---------------------------------------------------------------------------------------------
# Independent oracles
# ---------------------------------------------------------------------------------------------


def _row_at(ssm_state_indices: torch.Tensor, n: int, t: int) -> int:
    """State row selected for sequence ``n`` / token ``t`` (kernel lines 110 and 160)."""
    if ssm_state_indices.dim() == 1:
        return int(ssm_state_indices[n])
    return int(ssm_state_indices[n, t])


def _oracle(
    A_log,
    a,
    b,
    dt_bias,
    q,
    k,
    v,
    *,
    beta=1.0,
    threshold=20.0,
    scale=None,
    initial_state=None,
    inplace_final_state=True,
    cu_seqlens=None,
    ssm_state_indices=None,
    num_accepted_tokens=None,
    use_qk_l2norm_in_kernel=False,
    is_kda=False,
):
    """Step-by-step fp64 reference, transcribed from the kernel body (no call into the port).

    Returns ``(o, states)``; ``states`` is the post-update state tensor (the in-place one when
    ``inplace_final_state``, else the fresh per-token buffer), so callers can compare both the
    returned state and the mutated-in-place tensor to the same expected value.
    """
    B, T, H, K = k.shape
    HV, V = v.shape[2], v.shape[3]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    if scale is None:
        scale = K**-0.5
    hv_to_h = torch.arange(HV) // (HV // H)

    A = A_log.double().reshape(-1)
    dtb = dt_bias.double()
    a_tok = a.double().reshape(-1, HV, K) if is_kda else a.double().reshape(-1, HV)
    b_tok = b.double().reshape(-1, HV)
    qf = q.double().reshape(B * T, H, K)
    kf = k.double().reshape(B * T, H, K)
    vf = v.double().reshape(B * T, HV, V)

    if inplace_final_state:
        states = initial_state.double().clone()
        reads = states
    else:
        states = torch.zeros(T, HV, V, K, dtype=torch.float64)
        reads = initial_state.double()

    o = torch.zeros(B, T, HV, V, dtype=torch.float64)
    o_flat = o.view(B * T, HV, V)

    for n in range(N):
        if cu_seqlens is None:
            bos, eos = n * T, (n + 1) * T
        else:
            bos, eos = int(cu_seqlens[n]), int(cu_seqlens[n + 1])
        if eos == bos:
            continue
        if ssm_state_indices is not None:
            i_t = int(num_accepted_tokens[n]) - 1 if num_accepted_tokens is not None else 0
            row = _row_at(ssm_state_indices, n, i_t)
            if row <= 0:  # NULL_BLOCK_ID -> kernel returns immediately
                continue
        else:
            row = bos
        h = reads[row].clone()  # (HV, V, K)

        for t in range(eos - bos):
            tok = bos + t
            x = a_tok[tok] + dtb
            beta_x = beta * x
            softplus_x = torch.where(
                beta_x <= threshold, torch.log1p(torch.exp(beta_x)) / beta, x
            )
            # exp(g), the gate is per token *first*; KDA broadcasts the per-head A_log over K.
            a_exp = torch.exp(A)[:, None] if is_kda else torch.exp(A)
            decay = torch.exp(-a_exp * softplus_x)
            beta_t = torch.sigmoid(b_tok[tok])

            qe = qf[tok].index_select(0, hv_to_h)
            ke = kf[tok].index_select(0, hv_to_h)
            if use_qk_l2norm_in_kernel:
                qe = qe / torch.sqrt((qe * qe).sum(-1, keepdim=True) + 1e-6)
                ke = ke / torch.sqrt((ke * ke).sum(-1, keepdim=True) + 1e-6)
            qe = qe * scale

            h = h * (decay[:, None, :] if is_kda else decay[:, None, None])
            delta = (vf[tok] - (h * ke[:, None, :]).sum(-1)) * beta_t[:, None]
            h = h + delta[:, :, None] * ke[:, None, :]
            o_flat[tok] = (h * qe[:, None, :]).sum(-1)

            if ssm_state_indices is not None and inplace_final_state:
                r = _row_at(ssm_state_indices, n, t)
                if r > 0:
                    states[r] = h
            else:
                states[bos + t] = h

    return o, states


def _prefill_form(
    A_log,
    a,
    b,
    dt_bias,
    q,
    k,
    v,
    h0,
    *,
    beta=1.0,
    threshold=20.0,
    scale=None,
    use_qk_l2norm_in_kernel=False,
):
    """Whole-sequence closed form, no state recursion (see the module docstring).

    Shapes: ``q``/``k`` ``(T, H, K)``, ``v`` ``(T, HV, V)``, ``a``/``b`` ``(T, HV)``,
    ``h0`` ``(HV, V, K)``.  Returns ``(o, h_final)`` with ``o`` ``(T, HV, V)``.

    With ``P_t = d_t (I - beta_t k_t k_t^T)`` (right-multiplying because ``h`` is ``(V, K)``)::

        h_t   = h_0 P_0...P_t + sum_{s<=t} beta_s v_s k_s^T (P_{s+1}...P_t)
        o_t   = scale * (h_0 P_0...P_t q_t + sum_{s<=t} beta_s (k_s^T P_{s+1}...P_t q_t) v_s)
        h_end = h_0 P_0...P_{T-1} + sum_s beta_s v_s (k_s^T P_{s+1}...P_{T-1})
    """
    T, H, K = k.shape
    HV, V = v.shape[1], v.shape[2]
    if scale is None:
        scale = K**-0.5
    hv_to_h = torch.arange(HV) // (HV // H)
    eye = torch.eye(K, dtype=torch.float64)
    A = A_log.double().reshape(-1)
    dtb = dt_bias.double()
    a6, b6 = a.double(), b.double()
    q6, k6, v6, h06 = q.double(), k.double(), v.double(), h0.double()

    o = torch.zeros(T, HV, V, dtype=torch.float64)
    h_final = torch.zeros(HV, V, K, dtype=torch.float64)

    for hv in range(HV):
        x = a6[:, hv] + dtb[hv]
        beta_x = beta * x
        softplus_x = torch.where(beta_x <= threshold, torch.log1p(torch.exp(beta_x)) / beta, x)
        decay = torch.exp(-torch.exp(A[hv]) * softplus_x)  # (T,)
        beta_t = torch.sigmoid(b6[:, hv])  # (T,)

        qe = q6[:, hv_to_h[hv]]  # (T, K)
        ke = k6[:, hv_to_h[hv]]
        if use_qk_l2norm_in_kernel:
            qe = qe / torch.sqrt((qe * qe).sum(-1, keepdim=True) + 1e-6)
            ke = ke / torch.sqrt((ke * ke).sum(-1, keepdim=True) + 1e-6)
        qe = qe * scale
        ve = v6[:, hv]  # (T, V)

        # Transfer matrices and their prefix products PF[t] = P_0 @ ... @ P_t.
        p = [decay[t] * (eye - beta_t[t] * torch.outer(ke[t], ke[t])) for t in range(T)]
        pf = []
        acc = eye.clone()
        for t in range(T):
            acc = acc @ p[t]
            pf.append(acc)
        # Suffix products PS[t] = P_{t+1} @ ... @ P_{T-1}.
        ps = [None] * T
        acc = eye.clone()
        ps[T - 1] = acc
        for t in range(T - 2, -1, -1):
            acc = p[t + 1] @ acc
            ps[t] = acc

        h0_hv = h06[hv]
        for t in range(T):
            carry = (h0_hv @ pf[t]) @ qe[t]  # (V,)
            acc_out = torch.zeros(V, dtype=torch.float64)
            suffix = eye.clone()
            for s in range(t, -1, -1):
                coef = ke[s] @ suffix @ qe[t]
                acc_out = acc_out + beta_t[s] * coef * ve[s]
                suffix = p[s] @ suffix
            o[t, hv] = carry + acc_out

        acc_h = torch.zeros(V, K, dtype=torch.float64)
        for s in range(T):
            acc_h = acc_h + beta_t[s] * torch.outer(ve[s], ke[s] @ ps[s])
        h_final[hv] = h0_hv @ pf[T - 1] + acc_h

    return o, h_final


# ---------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------


def _rand_inputs(T, H, HV, K, V, *, B=1, seed=0, dtype=torch.float64, kda=False):
    g = torch.Generator().manual_seed(seed)

    def rnd(*shape, spread=1.0):
        return (torch.randn(*shape, generator=g, dtype=torch.float64) * spread).to(dtype)

    q = rnd(B, T, H, K)
    k = rnd(B, T, H, K)
    v = rnd(B, T, HV, V)
    a = rnd(B * T, HV, K) if kda else rnd(B * T, HV, spread=0.7)
    b = rnd(B * T, HV)
    A_log = rnd(HV, spread=0.3) + 0.2  # > 0 so that decay = exp(g) is strictly < 1
    dt_bias = rnd(HV, K) if kda else rnd(HV, spread=0.1)
    return q, k, v, a, b, A_log, dt_bias


def _tolerances(dtype):
    if dtype == torch.float64:
        return dict(rtol=1e-10, atol=1e-12)
    if dtype == torch.float32:
        return dict(rtol=2e-5, atol=2e-6)
    if dtype == torch.bfloat16:
        return dict(rtol=2e-2, atol=2e-2)
    return dict(rtol=4e-3, atol=4e-3)


# One sequence whose recurrent state lives in slot 1 of a paged state tensor -- exactly the vLLM
# decode layout (the serving path always passes ``ssm_state_indices``).  Slot **1**, because slot 0
# is ``NULL_BLOCK_ID`` and the kernel skips it (lines 113-115).  Passing indices is also what makes
# a multi-token call well defined with a small state: the kernel stores rows ``bos + t`` otherwise
# (line 168), so a T-token call needs a slot that every token writes to.
SLOT = torch.tensor([1])


# ---------------------------------------------------------------------------------------------
# 1. The gate itself: softplus branches, beta scaling, -exp(A_log)
# ---------------------------------------------------------------------------------------------


def _naive_gate(a, A_log, dt_bias, beta, threshold):
    """Mask-and-assign softplus, deliberately not the module's ``where``/``clamp`` form."""
    x = a.double() + dt_bias.double()
    out = torch.empty_like(x, dtype=torch.float64)
    log_branch = beta * x <= threshold
    out[log_branch] = (1.0 / beta) * torch.log1p(torch.exp(beta * x[log_branch]))
    out[~log_branch] = x[~log_branch]
    return -torch.exp(A_log.double().reshape(-1)) * out


@pytest.mark.parametrize("beta", [1.0, 2.0, 0.5])
def test_gate_formula_matches_naive_softplus(beta):
    threshold = 20.0
    A_log = torch.log(torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64))
    dt_bias = torch.tensor([0.0, -0.3, 0.25], dtype=torch.float64)
    # Both softplus branches, plus both sides of the exact branch point beta*x == threshold.
    xs = torch.tensor(
        [
            [0.0, -3.5, 25.0],  # log branch, log branch, linear branch
            [-30.0, 12.0, 40.0],
            # beta*x == threshold exactly (threshold / beta is exact for beta in {1, 2, 0.5})
            [threshold / beta, threshold / beta + 1e-3, threshold / beta - 1e-3],
        ],
        dtype=torch.float64,
    )
    a = xs - dt_bias
    got = _sigmoid_gating_decay(a, A_log, dt_bias, beta, threshold)
    torch.testing.assert_close(got, _naive_gate(a, A_log, dt_bias, beta, threshold))

    # The branch point is inclusive on the softplus side: beta*x == threshold -> log1p branch.
    x = a[2, 0] + dt_bias[0]
    assert beta * x == threshold
    log_value = (1.0 / beta) * torch.log1p(torch.exp(beta * x))
    torch.testing.assert_close(got[2, 0], -torch.exp(A_log[0]) * log_value)
    # One ulp above the threshold switches to the linear branch, i.e. softplus(x) == x.
    torch.testing.assert_close(got[2, 1], -torch.exp(A_log[1]) * (a[2, 1] + dt_bias[1]))
    torch.testing.assert_close(
        got[2, 2],
        -torch.exp(A_log[2]) * (1.0 / beta) * torch.log1p(torch.exp(beta * (a[2, 2] + dt_bias[2]))),
    )


def test_gate_value_bounds():
    """``sigmoid(b) in (0, 1)`` and ``decay = exp(g) in (0, 1)`` since ``g = -exp(A_log)*softplus``."""
    A_log = torch.tensor([-2.0, 0.0, 0.5, 1.0, 1.5], dtype=torch.float64)
    dt_bias = torch.zeros(5, dtype=torch.float64)
    a = torch.tensor([[-3.0, -1.0, 0.0, 1.0, 3.0]], dtype=torch.float64)
    g = _sigmoid_gating_decay(a, A_log, dt_bias, 1.0, 20.0)
    decay = torch.exp(g)
    assert torch.all(g <= 0)
    assert torch.all(decay > 0) and torch.all(decay < 1)

    # Extreme gates are still in [0, 1]: decay underflows to exactly 0, never negative/NaN.
    g_ext = _sigmoid_gating_decay(
        torch.tensor([[-1e4, 1e4]], dtype=torch.float64),
        torch.tensor([8.0], dtype=torch.float64),
        torch.zeros(1, dtype=torch.float64),
        1.0,
        20.0,
    )
    decay_ext = torch.exp(g_ext)
    assert torch.all(torch.isfinite(decay_ext))
    assert torch.all(decay_ext >= 0) and torch.all(decay_ext <= 1)
    torch.testing.assert_close(decay_ext[0, 1], torch.zeros((), dtype=torch.float64))

    beta_out = torch.sigmoid(torch.tensor([-30.0, -1.0, 0.0, 1.0, 30.0], dtype=torch.float64))
    assert torch.all(beta_out > 0) and torch.all(beta_out < 1)
    torch.testing.assert_close(beta_out[2], torch.tensor(0.5, dtype=torch.float64))
    # At very large |b| the sigmoid saturates to exactly 0.0/1.0 in binary floating point (fp32,
    # which is what the kernel uses, saturates earlier); that is rounding, not a gate semantics.
    assert torch.sigmoid(torch.tensor(80.0, dtype=torch.float64)) == 1.0


# ---------------------------------------------------------------------------------------------
# 2. THE test: T decode steps == the whole-sequence form
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "T,H,HV,K,V",
    [(1, 2, 2, 3, 4), (4, 2, 2, 3, 4), (7, 2, 4, 4, 3), (5, 2, 4, 2, 2)],
)
def test_decode_step_applied_t_times_equals_sequence_form(T, H, HV, K, V):
    q, k, v, a, b, A_log, dt_bias = _rand_inputs(T, H, HV, K, V, seed=T * 11 + HV)
    h = torch.randn(HV, V, K, dtype=torch.float64)
    # Row 0 is what the no-index path reads/writes (bos == 0); row 1 is the paged slot ``SLOT``.
    state_init = torch.stack([h, h])

    # (a) the module, one token at a time, state carried in place -- the serving path
    state = state_init.clone()
    outs = []
    for t in range(T):
        o_t, state = fused_gdn_gating_update(
            A_log,
            a[t : t + 1],
            b[t : t + 1],
            dt_bias,
            q[:, t : t + 1],
            k[:, t : t + 1],
            v[:, t : t + 1],
            initial_state=state,
        )
        outs.append(o_t)
    decode_o = torch.cat(outs, dim=1)  # (1, T, HV, V)

    # (b) whole-sequence closed form (no state recursion)
    pref_o, pref_h = _prefill_form(A_log, a, b, dt_bias, q[0], k[0], v[0], h)
    # (c) independent step-by-step oracle over the same T tokens in one call
    orc_o, orc_state = _oracle(
        A_log, a, b, dt_bias, q, k, v, initial_state=state_init.clone(), ssm_state_indices=SLOT
    )

    tol = _tolerances(torch.float64)
    torch.testing.assert_close(decode_o[0], pref_o, **tol)
    torch.testing.assert_close(state[0], pref_h, **tol)
    torch.testing.assert_close(decode_o[0], orc_o[0], **tol)
    torch.testing.assert_close(state[0], orc_state[1], **tol)

    # (d) a single multi-token call must agree with T single-token calls
    state2 = state_init.clone()
    o_T, state2 = fused_gdn_gating_update(
        A_log, a, b, dt_bias, q, k, v, initial_state=state2, ssm_state_indices=SLOT
    )
    torch.testing.assert_close(o_T, decode_o, **tol)
    torch.testing.assert_close(state2[1], state[0], **tol)


def test_prefill_form_matches_step_oracle():
    """Sanity-pin the pairwise algebra itself (catches a bug that both callers could share)."""
    T, H, HV, K, V = 6, 2, 2, 3, 3
    q, k, v, a, b, A_log, dt_bias = _rand_inputs(T, H, HV, K, V, seed=404)
    h0 = torch.randn(HV, V, K, dtype=torch.float64)
    pref_o, pref_h = _prefill_form(A_log, a, b, dt_bias, q[0], k[0], v[0], h0)
    # T rows so that the kernel's per-token writes stay in bounds; row 0 is the initial state.
    per_token = h0[None].repeat(T, 1, 1, 1)
    orc_o, orc_state = _oracle(A_log, a, b, dt_bias, q, k, v, initial_state=per_token)
    tol = _tolerances(torch.float64)
    torch.testing.assert_close(pref_o, orc_o[0], **tol)
    torch.testing.assert_close(pref_h, orc_state[-1], **tol)


# ---------------------------------------------------------------------------------------------
# 3. State: updated in place, carried, per-token snapshots
# ---------------------------------------------------------------------------------------------


def test_state_updated_in_place_by_default():
    T, H, HV, K, V = 3, 2, 2, 4, 3
    q, k, v, a, b, A_log, dt_bias = _rand_inputs(T, H, HV, K, V, seed=1)
    # A per-token state buffer: without ssm_state_indices the kernel writes rows bos + t (line 168).
    state = torch.randn(T, HV, V, K, dtype=torch.float64)
    before = state.clone()

    o, final_state = fused_gdn_gating_update(A_log, a, b, dt_bias, q, k, v, initial_state=state)
    assert final_state is state, "inplace_final_state=True must return initial_state itself"
    assert not torch.equal(state, before), "the state tensor must have been mutated"
    orc_o, orc_state = _oracle(A_log, a, b, dt_bias, q, k, v, initial_state=before.clone())
    tol = _tolerances(torch.float64)
    torch.testing.assert_close(o, orc_o, **tol)
    torch.testing.assert_close(state, orc_state, **tol)
    # Row t is the state right after token t; running the decode step T times reproduces row T-1.
    carried = before[0][None]
    for t in range(T):
        _, carried = fused_gdn_gating_update(
            A_log, a[t : t + 1], b[t : t + 1], dt_bias,
            q[:, t : t + 1], k[:, t : t + 1], v[:, t : t + 1], initial_state=carried,
        )
    torch.testing.assert_close(state[-1], carried[0], **tol)


def test_inplace_final_state_false_keeps_input_and_stores_per_token_states():
    T, H, HV, K, V = 4, 2, 2, 3, 3
    q, k, v, a, b, A_log, dt_bias = _rand_inputs(T, H, HV, K, V, seed=2)
    state = torch.randn(2, HV, V, K, dtype=torch.float64)
    before = state.clone()

    o, final_state = fused_gdn_gating_update(
        A_log, a, b, dt_bias, q, k, v, initial_state=state, inplace_final_state=False
    )
    assert torch.equal(state, before), "inplace_final_state=False must not touch the input"
    assert final_state.shape == (T, HV, V, K)
    assert final_state is not state

    orc_o, orc_states = _oracle(
        A_log, a, b, dt_bias, q, k, v, initial_state=before.clone(), inplace_final_state=False
    )
    tol = _tolerances(torch.float64)
    torch.testing.assert_close(o, orc_o, **tol)
    torch.testing.assert_close(final_state, orc_states, **tol)
    # Row t is the state after token t, so the last row equals the state carried by T decode steps.
    carried = before[0][None]
    for t in range(T):
        _, carried = fused_gdn_gating_update(
            A_log, a[t : t + 1], b[t : t + 1], dt_bias,
            q[:, t : t + 1], k[:, t : t + 1], v[:, t : t + 1], initial_state=carried,
        )
    torch.testing.assert_close(final_state[-1], carried[0], **tol)


def test_inplace_final_state_false_rejects_multi_sequence_batch():
    """Upstream indexes a (T, HV, V, K) buffer by the flattened token id (lines 229/168)."""
    T, H, HV, K, V = 2, 2, 2, 2, 2
    q, k, v, a, b, A_log, dt_bias = _rand_inputs(T, H, HV, K, V, B=2, seed=3)
    state = torch.randn(2, HV, V, K, dtype=torch.float64)
    with pytest.raises(ValueError, match="B == 1"):
        fused_gdn_gating_update(
            A_log, a, b, dt_bias, q, k, v, initial_state=state, inplace_final_state=False
        )


# ---------------------------------------------------------------------------------------------
# 4. ssm_state_indices / paged slots / spec decoding
# ---------------------------------------------------------------------------------------------


def test_ssm_state_indices_selects_slots_and_leaves_others_alone():
    T, H, HV, K, V, slots = 1, 2, 2, 3, 3, 5
    q, k, v, a, b, A_log, dt_bias = _rand_inputs(T, H, HV, K, V, B=2, seed=4)
    state = torch.randn(slots, HV, V, K, dtype=torch.float64)
    before = state.clone()
    indices = torch.tensor([3, 1])

    o, final_state = fused_gdn_gating_update(
        A_log, a, b, dt_bias, q, k, v, initial_state=state, ssm_state_indices=indices
    )
    assert final_state is state
    untouched = [s for s in range(slots) if s not in (1, 3)]
    torch.testing.assert_close(state[untouched], before[untouched], rtol=0, atol=0)
    assert not torch.equal(state[1], before[1]) and not torch.equal(state[3], before[3])

    orc_o, orc_state = _oracle(
        A_log, a, b, dt_bias, q, k, v, initial_state=before.clone(), ssm_state_indices=indices
    )
    tol = _tolerances(torch.float64)
    torch.testing.assert_close(o, orc_o, **tol)
    torch.testing.assert_close(state, orc_state, **tol)


def test_null_block_id_zero_skips_sequence():
    """Kernel lines 113-115/162-163: state_idx <= 0 returns without writing o or the state."""
    T, H, HV, K, V, slots = 1, 2, 2, 2, 2, 4
    q, k, v, a, b, A_log, dt_bias = _rand_inputs(T, H, HV, K, V, B=2, seed=5)
    state = torch.randn(slots, HV, V, K, dtype=torch.float64)
    before = state.clone()
    indices = torch.tensor([0, 2])

    o, _ = fused_gdn_gating_update(
        A_log, a, b, dt_bias, q, k, v, initial_state=state, ssm_state_indices=indices
    )
    torch.testing.assert_close(state[0], before[0], rtol=0, atol=0)
    torch.testing.assert_close(state[1], before[1], rtol=0, atol=0)
    torch.testing.assert_close(state[3], before[3], rtol=0, atol=0)
    assert not torch.equal(state[2], before[2])
    # Documented deviation: upstream leaves o uninitialised for a skipped sequence, we zero it.
    assert torch.count_nonzero(o[0]) == 0
    assert torch.count_nonzero(o[1]) > 0


def test_spec_decoding_reads_num_accepted_tokens_and_writes_every_token():
    T, H, HV, K, V, slots = 3, 2, 2, 3, 2, 6
    q, k, v, a, b, A_log, dt_bias = _rand_inputs(T, H, HV, K, V, B=2, seed=6)
    state = torch.randn(slots, HV, V, K, dtype=torch.float64)
    before = state.clone()
    # Per-token slots (spec decoding writes one state per drafted token); slot 0 is NULL.
    indices = torch.tensor([[4, 5, 3], [1, 2, 0]])
    num_accepted = torch.tensor([2, 1])

    o, _ = fused_gdn_gating_update(
        A_log,
        a,
        b,
        dt_bias,
        q,
        k,
        v,
        initial_state=state,
        ssm_state_indices=indices,
        num_accepted_tokens=num_accepted,
    )
    orc_o, orc_state = _oracle(
        A_log,
        a,
        b,
        dt_bias,
        q,
        k,
        v,
        initial_state=before.clone(),
        ssm_state_indices=indices,
        num_accepted_tokens=num_accepted,
    )
    tol = _tolerances(torch.float64)
    torch.testing.assert_close(o, orc_o, **tol)
    torch.testing.assert_close(state, orc_state, **tol)
    # Sequence 0 starts from slot indices[0, 2-1] == 5 and writes 4, 5, 3; sequence 1 starts from
    # indices[1, 1-1] == 1 and writes 1, 2 and skips the NULL slot 0.
    assert not torch.equal(state[4], before[4]) and not torch.equal(state[5], before[5])
    assert not torch.equal(state[3], before[3]) and not torch.equal(state[2], before[2])
    torch.testing.assert_close(state[0], before[0], rtol=0, atol=0)  # NULL slot untouched
    assert not torch.equal(state[1], before[1])
    # The h0 read is governed by num_accepted_tokens, so a different count changes the result.
    o_other, _ = fused_gdn_gating_update(
        A_log, a, b, dt_bias, q, k, v, initial_state=before.clone(),
        ssm_state_indices=indices, num_accepted_tokens=torch.tensor([3, 3]),
    )
    assert not torch.allclose(o, o_other)


# ---------------------------------------------------------------------------------------------
# 5. dtype / shape / scale
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype", [torch.float64, torch.float32, torch.bfloat16, torch.float16]
)
def test_dtype_preservation_and_numerics(dtype):
    T, H, HV, K, V = 3, 2, 2, 4, 4
    q, k, v, a, b, A_log, dt_bias = _rand_inputs(T, H, HV, K, V, seed=7)
    state = torch.randn(2, HV, V, K, dtype=torch.float32)
    before = state.clone()

    qd, kd, vd, ad, bd = (t.to(dtype) for t in (q, k, v, a, b))
    o, final_state = fused_gdn_gating_update(
        A_log, ad, bd, dt_bias, qd, kd, vd, initial_state=state, ssm_state_indices=SLOT
    )
    assert o.dtype == dtype and o.shape == (1, T, HV, V)
    assert final_state is state and final_state.dtype == torch.float32

    # Oracle on the *already rounded* inputs, so only arithmetic precision differs.
    orc_o, orc_state = _oracle(
        A_log, ad, bd, dt_bias, qd, kd, vd, initial_state=before.clone(), ssm_state_indices=SLOT
    )
    torch.testing.assert_close(o.double(), orc_o, **_tolerances(dtype))
    torch.testing.assert_close(state.double(), orc_state, rtol=2e-5, atol=2e-6)


def test_output_dtype_follows_q_not_v():
    """Line 225: ``o = q.new_empty(...)`` -- the output buffer inherits ``q.dtype``."""
    T, H, HV, K, V = 2, 2, 2, 3, 3
    q, k, v, a, b, A_log, dt_bias = _rand_inputs(T, H, HV, K, V, seed=8)
    state = torch.randn(2, HV, V, K, dtype=torch.float32)
    o, _ = fused_gdn_gating_update(
        A_log,
        a.float(),
        b.float(),
        dt_bias,
        q.float(),
        k.float(),
        v.half(),
        initial_state=state,
        ssm_state_indices=SLOT,
    )
    assert o.dtype == torch.float32


def test_scale_default_and_linearity_in_q():
    T, H, HV, K, V = 3, 2, 2, 4, 3
    q, k, v, a, b, A_log, dt_bias = _rand_inputs(T, H, HV, K, V, seed=9)
    state = torch.randn(2, HV, V, K, dtype=torch.float64)
    tol = _tolerances(torch.float64)

    o_default, _ = fused_gdn_gating_update(
        A_log, a, b, dt_bias, q, k, v, initial_state=state.clone(), ssm_state_indices=SLOT
    )
    o_explicit, _ = fused_gdn_gating_update(
        A_log, a, b, dt_bias, q, k, v, scale=K**-0.5, initial_state=state.clone(),
        ssm_state_indices=SLOT,
    )
    torch.testing.assert_close(o_default, o_explicit, **tol)

    o1, h1 = fused_gdn_gating_update(
        A_log, a, b, dt_bias, q, k, v, scale=1.0, initial_state=state.clone(),
        ssm_state_indices=SLOT,
    )
    o5, h5 = fused_gdn_gating_update(
        A_log, a, b, dt_bias, q, k, v, scale=5.0, initial_state=state.clone(),
        ssm_state_indices=SLOT,
    )
    torch.testing.assert_close(o5, 5.0 * o1, **tol)  # scale touches q only, not k or the state
    assert torch.equal(h5, h1)

    with pytest.raises(AssertionError, match="scale must be positive"):
        fused_gdn_gating_update(
            A_log, a, b, dt_bias, q, k, v, scale=0.0, initial_state=state.clone(),
            ssm_state_indices=SLOT,
        )


def test_qk_l2norm_in_kernel_matches_explicit_normalisation():
    T, H, HV, K, V = 3, 2, 2, 3, 3
    q, k, v, a, b, A_log, dt_bias = _rand_inputs(T, H, HV, K, V, seed=10)
    state = torch.randn(2, HV, V, K, dtype=torch.float64)
    tol = _tolerances(torch.float64)

    o_norm, _ = fused_gdn_gating_update(
        A_log, a, b, dt_bias, q, k, v, initial_state=state.clone(),
        use_qk_l2norm_in_kernel=True, ssm_state_indices=SLOT,
    )
    qn = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6)
    kn = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    o_ref, _ = fused_gdn_gating_update(
        A_log, a, b, dt_bias, qn, kn, v, initial_state=state.clone(), ssm_state_indices=SLOT
    )
    torch.testing.assert_close(o_norm, o_ref, **tol)

    o_plain, _ = fused_gdn_gating_update(
        A_log, a, b, dt_bias, q, k, v, initial_state=state.clone(), ssm_state_indices=SLOT
    )
    assert not torch.allclose(o_norm, o_plain), "l2norm must change the result for |q| != 1"
    # The kernel adds eps to the squared sum, it does not clamp the norm: a near-zero row must
    # stay tiny rather than becoming unit length.
    tiny = torch.zeros_like(q)
    tiny[..., 0] = 1e-8
    assert (tiny / torch.sqrt((tiny * tiny).sum(-1, keepdim=True) + 1e-6)).abs().max() < 1e-5


def test_gqa_key_head_mapping():
    """Line 64: ``i_h = i_hv // (HV // H)`` -- two value heads share one key head here."""
    T, H, HV, K, V = 2, 2, 4, 3, 2
    q, k, v, a, b, A_log, dt_bias = _rand_inputs(T, H, HV, K, V, seed=12)
    state = torch.randn(2, HV, V, K, dtype=torch.float64)
    orc_o, _ = _oracle(
        A_log, a, b, dt_bias, q, k, v, initial_state=state.clone(), ssm_state_indices=SLOT
    )
    o, _ = fused_gdn_gating_update(
        A_log, a, b, dt_bias, q, k, v, initial_state=state.clone(), ssm_state_indices=SLOT
    )
    tol = _tolerances(torch.float64)
    torch.testing.assert_close(o, orc_o, **tol)

    # Structural check on the mapping: zeroing key head 0 must change value heads 0 and 1 (which
    # read it) and leave value heads 2 and 3 (which read key head 1) bit-identical.
    k_zero = k.clone()
    k_zero[:, :, 0] = 0.0
    o0, _ = fused_gdn_gating_update(
        A_log, a, b, dt_bias, q, k_zero, v, initial_state=state.clone(), ssm_state_indices=SLOT
    )
    assert not torch.allclose(o0[:, :, 0], o[:, :, 0])
    assert not torch.allclose(o0[:, :, 1], o[:, :, 1])
    assert torch.allclose(o0[:, :, 2], o[:, :, 2])
    assert torch.allclose(o0[:, :, 3], o[:, :, 3])


def test_zero_query_gives_zero_output_for_the_sharing_heads():
    T, H, HV, K, V = 1, 2, 4, 3, 2
    q, k, v, a, b, A_log, dt_bias = _rand_inputs(T, H, HV, K, V, seed=13)
    state = torch.randn(2, HV, V, K, dtype=torch.float64)
    q_zero = q.clone()
    q_zero[:, :, 0] = 0.0  # value heads 0 and 1 read key head 0
    o, _ = fused_gdn_gating_update(
        A_log, a, b, dt_bias, q_zero, k, v, initial_state=state.clone()
    )
    assert torch.count_nonzero(o[:, :, 0]) == 0
    assert torch.count_nonzero(o[:, :, 1]) == 0
    assert torch.count_nonzero(o[:, :, 2]) > 0


# ---------------------------------------------------------------------------------------------
# 6. Variable-length batches and the KDA gate variant
# ---------------------------------------------------------------------------------------------


def test_varlen_cu_seqlens():
    H, HV, K, V, slots = 2, 2, 3, 2, 5
    lens = [2, 1, 3]
    T = sum(lens)
    q, k, v, a, b, A_log, dt_bias = _rand_inputs(T, H, HV, K, V, seed=14)
    cu_seqlens = torch.tensor([0, 2, 3, 6])
    assert int(cu_seqlens[-1]) == T
    state = torch.randn(slots, HV, V, K, dtype=torch.float64)
    before = state.clone()
    indices = torch.tensor([2, 0, 4])  # sequence 1 has no valid state -> skipped entirely

    o, _ = fused_gdn_gating_update(
        A_log,
        a,
        b,
        dt_bias,
        q,
        k,
        v,
        initial_state=state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=indices,
    )
    orc_o, orc_state = _oracle(
        A_log,
        a,
        b,
        dt_bias,
        q,
        k,
        v,
        initial_state=before.clone(),
        cu_seqlens=cu_seqlens,
        ssm_state_indices=indices,
    )
    tol = _tolerances(torch.float64)
    torch.testing.assert_close(o, orc_o, **tol)
    torch.testing.assert_close(state, orc_state, **tol)
    # Sequence 1 (tokens 2..2) is skipped: its output row and slot 0 stay untouched.
    assert torch.count_nonzero(o[0, 2]) == 0
    torch.testing.assert_close(state[0], before[0], rtol=0, atol=0)


def test_cu_seqlens_requires_flat_batch():
    T, H, HV, K, V = 2, 2, 2, 2, 2
    q, k, v, a, b, A_log, dt_bias = _rand_inputs(T, H, HV, K, V, B=2, seed=15)
    state = torch.randn(4, HV, V, K, dtype=torch.float64)
    with pytest.raises(ValueError, match="batch size is expected to be 1"):
        fused_gdn_gating_update(
            A_log,
            a,
            b,
            dt_bias,
            q,
            k,
            v,
            initial_state=state,
            cu_seqlens=torch.tensor([0, 2]),
        )


def test_is_kda_per_channel_gate():
    T, H, HV, K, V = 3, 2, 2, 3, 2
    q, k, v, a, b, A_log, dt_bias = _rand_inputs(T, H, HV, K, V, seed=16, kda=True)
    assert a.shape == (T, HV, K) and dt_bias.shape == (HV, K)
    state = torch.randn(2, HV, V, K, dtype=torch.float64)
    work = state.clone()  # the module mutates in place, so keep the reference it writes into
    o, _ = fused_gdn_gating_update(
        A_log, a, b, dt_bias, q, k, v, initial_state=work, is_kda=True,
        ssm_state_indices=SLOT,
    )
    orc_o, orc_state = _oracle(
        A_log, a, b, dt_bias, q, k, v, initial_state=state.clone(), is_kda=True,
        ssm_state_indices=SLOT,
    )
    tol = _tolerances(torch.float64)
    torch.testing.assert_close(o, orc_o, **tol)
    torch.testing.assert_close(work, orc_state, **tol)

    # A per-channel gate must differ from the per-head one (they only coincide if all K channels
    # share the same gate value).
    o_shared, _ = fused_gdn_gating_update(
        A_log, a[..., 0].contiguous(), b, dt_bias[:, 0].contiguous(), q, k, v,
        initial_state=state.clone(), ssm_state_indices=SLOT,
    )
    assert not torch.allclose(o, o_shared)


# ---------------------------------------------------------------------------------------------
# 7. Argument plumbing
# ---------------------------------------------------------------------------------------------


def test_initial_state_is_mandatory():
    T, H, HV, K, V = 1, 2, 2, 2, 2
    q, k, v, a, b, A_log, dt_bias = _rand_inputs(T, H, HV, K, V, seed=17)
    with pytest.raises(ValueError, match="initial_state is required"):
        fused_gdn_gating_update(A_log, a, b, dt_bias, q, k, v)


def test_state_layout_validation():
    T, H, HV, K, V = 1, 2, 2, 3, 2  # K != V so a (HV, K, V) mix-up is a different shape
    q, k, v, a, b, A_log, dt_bias = _rand_inputs(T, H, HV, K, V, seed=18)
    with pytest.raises(ValueError, match=r"\(rows, HV, V, K\)"):
        fused_gdn_gating_update(
            A_log, a, b, dt_bias, q, k, v,
            initial_state=torch.zeros(1, HV, K, V, dtype=torch.float64),
        )
    with pytest.raises(ValueError, match="contiguous"):
        # Trailing dims permuted: shape is right, strides are not (the kernel hardcodes v*K + k).
        bad = torch.zeros(1, HV, K, V, dtype=torch.float64).transpose(-1, -2)
        fused_gdn_gating_update(A_log, a, b, dt_bias, q, k, v, initial_state=bad)


def test_a_b_token_major_layout_is_accepted_both_ways():
    """Lines 89/95 only require flat index ``tok*HV + hv``; ``(B, T, HV)`` aliases that."""
    T, H, HV, K, V = 2, 2, 2, 2, 2
    q, k, v, a, b, A_log, dt_bias = _rand_inputs(T, H, HV, K, V, seed=19)
    state = torch.randn(2, HV, V, K, dtype=torch.float64)
    o_flat, _ = fused_gdn_gating_update(
        A_log, a, b, dt_bias, q, k, v, initial_state=state.clone(), ssm_state_indices=SLOT
    )
    o_3d, _ = fused_gdn_gating_update(
        A_log, a.view(1, T, HV), b.view(1, T, HV), dt_bias, q, k, v,
        initial_state=state.clone(), ssm_state_indices=SLOT,
    )
    torch.testing.assert_close(o_flat, o_3d, **_tolerances(torch.float64))
