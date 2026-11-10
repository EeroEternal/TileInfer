"""Contract tests for the torch port of FLA's fused recurrent gated-delta-rule family.

Two entry points, two independent oracles, one cross-check that ties them together:

* ``_seq_oracle`` -- the literal per-token / per-head recurrence transcribed from the *sequence*
  kernel (``reference/vllm-0.23/fla_fused_recurrent.py`` lines 122-170), written one token at a time;
* ``_packed_oracle`` -- the literal per-token / per-head recurrence transcribed from the *packed
  decode* kernel (lines 292-336), including its gate (lines 318-325) written as a mask-and-assign
  softplus so that it does not share code with the port's ``gating._sigmoid_gating_decay``.

Neither oracle calls the module under test.

THE property (``test_t_decode_steps_equals_whole_sequence_form``): the packed decode entry point has
no token loop (grid ``(NV, B*HV)``, line 449), so it is a *one-token* operator.  Applying it once
per token with the state carried in place -- which is exactly how vLLM drives it
(``reference/vllm-0.23/vllm__..._qwen_gdn_linear_attn.py`` lines 1684-1695) -- must reproduce, token
for token and state for state, the whole-sequence form run over the same ``T`` tokens once the
sequence gate is supplied as ``g = -exp(A_log) * softplus(a + dt_bias)`` and the step as
``beta = sigmoid(b)``.  Both must also reproduce the two independent oracles, for ``T`` in
``{1, 4, 7}``.  That single assertion pins: the token order, the decay-before-delta order, the
output-from-updated-state rule, the GQA head map, the packed ``q|k|v`` offsets, the gate formula,
the state carry, and the fact that both entry points compute the *same* operator.

Everything below is CPU, fp64-first.
"""

from __future__ import annotations

import os

import pytest
import torch
from tileinfer.vllm.ops.fla.recurrent import (
    fused_recurrent_gated_delta_rule as seq_form,
)
from tileinfer.vllm.ops.fla.recurrent import (
    fused_recurrent_gated_delta_rule_packed_decode as packed_decode,
)

# ---------------------------------------------------------------------------------------------
# Independent oracles
# ---------------------------------------------------------------------------------------------


def _naive_gate(a, A_log, dt_bias, threshold=20.0):
    """``-exp(A_log) * softplus(a + dt_bias)`` -- kernel lines 318-324, mask-and-assign softplus.

    Deliberately *not* the port's ``where``/``clamp`` form: the linear branch takes over for
    ``x > 20`` (``SOFTPLUS_THRESHOLD``, line 473), the log branch is ``log1p(exp(x))``.
    """
    x = a.double() + dt_bias.double().reshape(-1)
    out = torch.empty_like(x, dtype=torch.float64)
    log_branch = x <= threshold
    out[log_branch] = torch.log1p(torch.exp(x[log_branch]))
    out[~log_branch] = x[~log_branch]
    return -torch.exp(A_log.double().reshape(-1)) * out


def _idx(ssm_state_indices, n, t):
    """State row for sequence ``n`` / token ``t`` (kernel lines 110/160/292)."""
    if ssm_state_indices.dim() == 1:
        return int(ssm_state_indices[n])
    return int(ssm_state_indices[n, t])


def _l2norm(x):
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + 1e-6)


def _seq_oracle(
    q,
    k,
    v,
    g,
    beta,
    initial_state,
    *,
    scale=None,
    inplace_final_state=True,
    cu_seqlens=None,
    ssm_state_indices=None,
    num_accepted_tokens=None,
    use_qk_l2norm_in_kernel=False,
):
    """Step-by-step fp64 reference for the sequence kernel (lines 122-170); no call into the port.

    Returns ``(o, states)``: ``o`` is ``(B, T, HV, V)``; ``states`` is the whole state tensor after
    the call (the in-place tensor, or the per-token buffer).  Reads come from a pre-call snapshot,
    because every kernel program loads ``h0`` once before its token loop (lines 103-120).
    """
    B, T, H, K = k.shape
    HV, V = v.shape[2], v.shape[3]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    if scale is None:
        scale = K**-0.5
    hv_to_h = torch.arange(HV) // (HV // H)

    qf = q.double().reshape(B * T, H, K)
    kf = k.double().reshape(B * T, H, K)
    vf = v.double().reshape(B * T, HV, V)
    gf = g.double().reshape(B * T, HV)
    beta_headwise = beta.dim() == v.dim()
    bf = beta.double().reshape(B * T, HV, V) if beta_headwise else beta.double().reshape(B * T, HV)

    snapshot = initial_state.double().clone()
    if inplace_final_state:
        states = snapshot.clone()
    else:
        assert B == 1, "the (T, HV, V, K) buffer is only in bounds for B == 1 (lines 205/163)"
        states = torch.zeros(T, HV, V, K, dtype=torch.float64)

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
            row = _idx(ssm_state_indices, n, i_t)
            if row <= 0:  # NULL_BLOCK_ID -> kernel returns immediately (lines 113-115)
                continue
        else:
            row = bos

        h = snapshot[row].clone()
        for t in range(eos - bos):
            tok = bos + t
            decay = torch.exp(gf[tok])
            qe = qf[tok].index_select(0, hv_to_h)
            ke = kf[tok].index_select(0, hv_to_h)
            if use_qk_l2norm_in_kernel:
                qe = _l2norm(qe)
                ke = _l2norm(ke)
            qe = qe * scale
            # Line 134, then 139/144, then 146, then 148.
            h = h * decay[:, None, None]
            bt = bf[tok]
            delta = vf[tok] - (h * ke[:, None, :]).sum(-1)
            delta = delta * (bt if beta_headwise else bt[:, None])
            h = h + delta[:, :, None] * ke[:, None, :]
            o_flat[tok] = (h * qe[:, None, :]).sum(-1)

            if ssm_state_indices is not None and inplace_final_state:
                r = _idx(ssm_state_indices, n, t)
                if r > 0:
                    states[r] = h
            else:
                states[bos + t] = h

    return o, states


def _packed_oracle(
    mixed_qkv,
    a,
    b,
    A_log,
    dt_bias,
    scale,
    initial_state,
    ssm_state_indices,
    *,
    use_qk_l2norm_in_kernel=False,
):
    """Step-by-step fp64 reference for the packed decode kernel (lines 292-336).

    ``H`` is inferred from the row width exactly as the launcher does (lines 417-430).  One call is
    one token per row (the kernel has no token loop), so a ``T``-token chain is ``T`` calls whose
    returned ``states`` is fed back in; rows must address distinct slots, exactly as the concurrent
    kernel programs require.  Returns ``(o, states)`` with ``o`` ``(B, HV, V)``.
    """
    B = mixed_qkv.shape[0]
    HV, V, K = initial_state.shape[-3:]
    qkv_dim = mixed_qkv.shape[1]
    H = (qkv_dim - HV * V) // 2 // K
    hv_to_h = torch.arange(HV) // (HV // H)

    states = initial_state.double().clone()
    o = torch.zeros(B, HV, V, dtype=torch.float64)

    decay = torch.exp(_naive_gate(a, A_log, dt_bias))  # (B, HV), lines 318-324 + 327
    beta_t = torch.sigmoid(b.double())  # line 325 (the dtype cast is a no-op in fp64)

    for n in range(B):
        idx = int(ssm_state_indices[n])
        if idx <= 0:  # lines 296-299
            continue
        h = initial_state[idx].double().clone()
        row = mixed_qkv[n].double()
        qe = row[: H * K].view(H, K).index_select(0, hv_to_h)
        ke = row[H * K : 2 * H * K].view(H, K).index_select(0, hv_to_h)
        ve = row[2 * H * K :].view(HV, V)
        if use_qk_l2norm_in_kernel:
            qe = _l2norm(qe)
            ke = _l2norm(ke)
        qe = qe * scale

        h = h * decay[n][:, None, None]
        delta = (ve - (h * ke[:, None, :]).sum(-1)) * beta_t[n][:, None]
        h = h + delta[:, :, None] * ke[:, None, :]
        o[n] = (h * qe[:, None, :]).sum(-1)
        states[idx] = h

    return o, states


# ---------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------


def _rand(T, H, HV, K, V, *, B=1, seed=0, dtype=torch.float64):
    gen = torch.Generator().manual_seed(seed)

    def rnd(*shape, spread=1.0):
        return (torch.randn(*shape, generator=gen, dtype=torch.float64) * spread).to(dtype)

    q = rnd(B, T, H, K)
    k = rnd(B, T, H, K)
    v = rnd(B, T, HV, V)
    a = rnd(B * T, HV, spread=0.7)
    b = rnd(B * T, HV)
    A_log = rnd(HV, spread=0.3) + 0.2  # > 0 -> decay = exp(g) strictly < 1
    dt_bias = rnd(HV, spread=0.1)
    return q, k, v, a, b, A_log, dt_bias


def _pack(q_tok, k_tok, v_tok):
    """``(H,K), (H,K), (HV,V)`` -> one ``(1, 2*H*K + HV*V)`` packed row (lines 306-308)."""
    return torch.cat([q_tok.reshape(-1), k_tok.reshape(-1), v_tok.reshape(-1)]).unsqueeze(0)


def _tolerances(dtype):
    if dtype == torch.float64:
        return dict(rtol=1e-10, atol=1e-12)
    if dtype == torch.float32:
        return dict(rtol=2e-5, atol=2e-6)
    if dtype == torch.bfloat16:
        return dict(rtol=2e-2, atol=2e-2)
    return dict(rtol=4e-3, atol=4e-3)


#: A valid paged-state slot.  Slot **1**, never 0: 0 is ``NULL_BLOCK_ID`` (lines 113-115/296-299),
#: and the serving path always passes ``ssm_state_indices``.
SLOT1 = torch.tensor([1])


def _decode_step(mixed, a_tok, b_tok, A_log, dt_bias, scale, state, slot, *, l2norm=False):
    """One packed decode call for one token; returns ``(out, o_row)``."""
    out = torch.zeros(1, 1, state.shape[1], state.shape[2], dtype=mixed.dtype)
    ret_out, ret_state = packed_decode(
        mixed,
        a_tok,
        b_tok,
        A_log,
        dt_bias,
        scale,
        state,
        out,
        torch.tensor([slot]),
        use_qk_l2norm_in_kernel=l2norm,
    )
    assert ret_out is out and ret_state is state
    return out, out.view(1, state.shape[1], state.shape[2])


# ---------------------------------------------------------------------------------------------
# 1. THE property: T decode steps == the whole-sequence form
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "T,H,HV,K,V",
    [
        (1, 2, 2, 3, 4),  # single step: the serving case
        (4, 2, 2, 3, 4),  # equal head counts
        (7, 2, 4, 4, 3),  # GQA, 2 value heads per key head
        (4, 2, 4, 2, 2),  # smallest K/V, GQA
    ],
)
def test_t_decode_steps_equals_whole_sequence_form(T, H, HV, K, V):
    """The key assertion: ``packed_decode`` x T (state carried in place) == sequence form."""
    q, k, v, a, b, A_log, dt_bias = _rand(T, H, HV, K, V, seed=T * 7 + HV)
    scale = K**-0.5
    h0 = torch.randn(HV, V, K, dtype=torch.float64)
    tol = _tolerances(torch.float64)
    # One untouched snapshot per path: both entry points mutate the state they are given, so the
    # oracles must be fed the *pre-call* state, not whatever a previous call left behind.
    state_init = torch.zeros(2, HV, V, K, dtype=torch.float64)
    state_init[1] = h0

    # (a) packed decode: one token per call, one paged slot carried through, fp64 in / out.
    state_dec = state_init.clone()
    dec_outs = []
    for t in range(T):
        mixed = _pack(q[0, t], k[0, t], v[0, t])
        out_t = torch.zeros(1, 1, HV, V, dtype=torch.float64)
        packed_decode(
            mixed, a[t : t + 1], b[t : t + 1], A_log, dt_bias, scale, state_dec, out_t,
            SLOT1,
        )
        dec_outs.append(out_t.view(1, HV, V))
    dec_o = torch.cat(dec_outs, dim=0)  # (T, HV, V)

    # (b) the whole-sequence entry point, same gate inputs expressed as g / beta.
    g = _naive_gate(a, A_log, dt_bias)  # (T, HV)
    beta = torch.sigmoid(b)
    state_seq = state_init.clone()
    seq_o, _ = seq_form(
        q, k, v, g.view(1, T, HV), beta.view(1, T, HV), scale=scale,
        initial_state=state_seq, ssm_state_indices=SLOT1,
    )

    # (c) independent per-token oracles over the same T tokens.
    orc_o, orc_state = _seq_oracle(
        q, k, v, g.view(1, T, HV), beta.view(1, T, HV), state_init.clone(),
        scale=scale, ssm_state_indices=SLOT1,
    )
    # The packed oracle models *one call* (one token per row), so chain it the same way.
    p_state = state_init.clone()
    p_outs = []
    for t in range(T):
        o_t, p_state = _packed_oracle(
            _pack(q[0, t], k[0, t], v[0, t]), a[t : t + 1], b[t : t + 1], A_log, dt_bias,
            scale, p_state, SLOT1,
        )
        p_outs.append(o_t)
    p_orc_o = torch.cat(p_outs, dim=0)
    p_orc_state = p_state

    # decode x T == sequence form == both oracles; the carried state matches too.
    torch.testing.assert_close(dec_o, seq_o[0], **tol)
    torch.testing.assert_close(dec_o, orc_o[0], **tol)
    torch.testing.assert_close(dec_o, p_orc_o, **tol)
    torch.testing.assert_close(state_dec[1], state_seq[1], **tol)
    torch.testing.assert_close(state_dec[1], orc_state[1], **tol)
    torch.testing.assert_close(state_dec[1], p_orc_state[1], **tol)
    # The untouched NULL row stays untouched in every path.
    torch.testing.assert_close(state_dec[0], torch.zeros(HV, V, K, dtype=torch.float64), **tol)

    # Sanity: the T-step chain is not vacuously equal (the state really moves).
    assert not torch.allclose(state_dec[1], h0)


# ---------------------------------------------------------------------------------------------
# 2. The sequence form against its own per-token oracle
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("T,H,HV,K,V", [(1, 2, 2, 3, 4), (4, 2, 2, 3, 4), (7, 2, 4, 4, 3)])
def test_sequence_form_matches_independent_loop(T, H, HV, K, V):
    """Multi-sequence batch, no paged indices: rows ``bos + t`` are the per-token snapshots."""
    B = 2
    q, k, v, a, b, A_log, dt_bias = _rand(T, H, HV, K, V, B=B, seed=101 + T)
    g = _naive_gate(a, A_log, dt_bias).view(B, T, HV)
    beta = torch.sigmoid(b).view(B, T, HV)
    state = torch.randn(B * T, HV, V, K, dtype=torch.float64)
    before = state.clone()

    o, final_state = seq_form(q, k, v, g, beta, initial_state=state)
    assert final_state is state
    orc_o, orc_state = _seq_oracle(q, k, v, g, beta, before)

    tol = _tolerances(torch.float64)
    torch.testing.assert_close(o, orc_o, **tol)
    torch.testing.assert_close(state, orc_state, **tol)
    # Rows are per sequence: sequence 0 writes rows [0, T), sequence 1 writes rows [T, 2T), so the
    # two halves of the output are not the same token stream.
    assert not torch.allclose(o[0], o[1])


def test_sequence_form_row_is_initial_state_of_next_sequence():
    """Sequence 1 must start from row ``T`` (kernel line 118), not from row 0."""
    T, H, HV, K, V, B = 2, 2, 2, 3, 2, 2
    q, k, v, a, b, A_log, dt_bias = _rand(T, H, HV, K, V, B=B, seed=202)
    g = _naive_gate(a, A_log, dt_bias).view(B, T, HV)
    beta = torch.sigmoid(b).view(B, T, HV)
    state = torch.randn(B * T, HV, V, K, dtype=torch.float64)
    before = state.clone()
    o, _ = seq_form(q, k, v, g, beta, initial_state=state)
    orc_o, _ = _seq_oracle(q, k, v, g, beta, before.clone())
    torch.testing.assert_close(o, orc_o, **_tolerances(torch.float64))
    # Changing row T (sequence 1's initial state) must only change sequence 1's output.
    other = before.clone()
    other[T] += 1.0
    o2, _ = seq_form(q, k, v, g, beta, initial_state=other)
    torch.testing.assert_close(o2[0], o[0], rtol=0, atol=0)
    assert not torch.allclose(o2[1], o[1])


# ---------------------------------------------------------------------------------------------
# 3. Packed layout
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("H,HV,K,V,B", [(2, 2, 3, 4, 1), (2, 4, 4, 3, 3), (3, 3, 2, 5, 2)])
def test_packed_layout_matches_separately_passed_qkv(H, HV, K, V, B):
    """Decode the packing by comparing against q/k/v passed separately (sequence form, T=1).

    ``mixed_qkv = [q: H*K][k: H*K][v: HV*V]`` (lines 306-308), each block head-major, and ``H`` is
    inferred as ``((qkv_dim - HV*V) // 2) // K`` (lines 417-430).
    """
    total = HV * V
    gen = torch.Generator().manual_seed(B * 31 + H)
    mixed = torch.randn(B, 2 * H * K + total, generator=gen, dtype=torch.float64)
    a = torch.randn(B, HV, generator=gen, dtype=torch.float64) * 0.7
    b = torch.randn(B, HV, generator=gen, dtype=torch.float64)
    A_log = torch.randn(HV, generator=gen, dtype=torch.float64) * 0.3 + 0.2
    dt_bias = torch.randn(HV, generator=gen, dtype=torch.float64) * 0.1
    slots = torch.arange(1, B + 1)  # distinct slots, no NULL
    state = torch.randn(B + 1, HV, V, K, generator=gen, dtype=torch.float64)
    before = state.clone()
    scale = K**-0.5

    out = torch.zeros(B, 1, HV, V, dtype=torch.float64)
    ret_out, ret_state = packed_decode(mixed, a, b, A_log, dt_bias, scale, state, out, slots)
    assert ret_out is out and ret_state is state

    # Unpack exactly as the kernel offsets say and feed the sequence entry point one token each.
    q = mixed[:, : H * K].reshape(B, 1, H, K)
    k = mixed[:, H * K : 2 * H * K].reshape(B, 1, H, K)
    v = mixed[:, 2 * H * K :].reshape(B, 1, HV, V)
    g = _naive_gate(a, A_log, dt_bias).view(B, 1, HV)
    beta = torch.sigmoid(b).view(B, 1, HV)
    seq_o, _ = seq_form(
        q, k, v, g, beta, scale=scale, initial_state=before.clone(), ssm_state_indices=slots
    )

    orc_o, orc_state = _packed_oracle(mixed, a, b, A_log, dt_bias, scale, before.clone(), slots)

    tol = _tolerances(torch.float64)
    torch.testing.assert_close(out.view(B, HV, V), seq_o.view(B, HV, V), **tol)
    torch.testing.assert_close(out.view(B, HV, V), orc_o, **tol)
    torch.testing.assert_close(state, orc_state, **tol)


def test_packed_layout_is_not_an_interleaved_head_layout():
    """Structural check: element 0 of a row is ``q[head 0, k=0]`` (line 306), nothing else.

    Zeroing it must leave every value head that maps to *another* key head bit-identical, while
    changing the heads that share key head 0 -- the same GQA structure a per-head-interleaved
    packing would also produce, so pair it with ``test_packed_layout_matches_separately_passed_qkv``
    (which pins the exact offsets) for a complete statement.
    """
    H, HV, K, V, B = 2, 4, 3, 2, 1
    gen = torch.Generator().manual_seed(77)
    mixed = torch.randn(B, 2 * H * K + HV * V, generator=gen, dtype=torch.float64)
    a = torch.zeros(B, HV, dtype=torch.float64)
    b = torch.zeros(B, HV, dtype=torch.float64)
    A_log = torch.zeros(HV, dtype=torch.float64)
    dt_bias = torch.zeros(HV, dtype=torch.float64)
    scale = K**-0.5
    st0 = torch.randn(2, HV, V, K, generator=gen, dtype=torch.float64)

    def run(m):
        out = torch.zeros(B, 1, HV, V, dtype=torch.float64)
        packed_decode(m, a, b, A_log, dt_bias, scale, st0.clone(), out, SLOT1)
        return out.view(HV, V)

    base = run(mixed)
    perturbed = mixed.clone()
    perturbed[0, 0] += 1.0  # q[head 0, k=0]
    got = run(perturbed)
    for hv in range(HV):
        if hv // (HV // H) == 0:
            assert not torch.allclose(got[hv], base[hv]), f"head {hv} should read key head 0"
        else:
            torch.testing.assert_close(got[hv], base[hv], rtol=0, atol=0)


# ---------------------------------------------------------------------------------------------
# 4. State semantics: in place, paged slots, NULL slots, multiple sequences
# ---------------------------------------------------------------------------------------------


def test_sequence_state_updated_in_place_by_default():
    T, H, HV, K, V = 3, 2, 2, 4, 3
    q, k, v, a, b, A_log, dt_bias = _rand(T, H, HV, K, V, seed=7)
    g = _naive_gate(a, A_log, dt_bias).view(1, T, HV)
    beta = torch.sigmoid(b).view(1, T, HV)
    state = torch.randn(T, HV, V, K, dtype=torch.float64)
    before = state.clone()

    o, final_state = seq_form(q, k, v, g, beta, initial_state=state)
    assert final_state is state, "inplace_final_state=True must return initial_state itself"
    assert not torch.equal(state, before), "the state tensor must have been mutated"
    orc_o, orc_state = _seq_oracle(q, k, v, g, beta, before)
    tol = _tolerances(torch.float64)
    torch.testing.assert_close(o, orc_o, **tol)
    torch.testing.assert_close(state, orc_state, **tol)
    # Row t is the state right after token t; carrying it through T packed-decode steps must land
    # on the last row.  (The intermediate rows are fp64 here, so this is the same arithmetic path.)
    carried = torch.zeros(2, HV, V, K, dtype=torch.float64)
    carried[1] = before[0]
    for t in range(T):
        mixed = _pack(q[0, t], k[0, t], v[0, t])
        _, _ = _decode_step(
            mixed, a[t : t + 1], b[t : t + 1], A_log, dt_bias, K**-0.5, carried, 1
        )
    torch.testing.assert_close(state[-1], carried[1], **tol)


def test_packed_decode_updates_state_in_place_and_returns_its_inputs():
    H, HV, K, V, B = 2, 2, 3, 2, 2
    gen = torch.Generator().manual_seed(9)
    mixed = torch.randn(B, 2 * H * K + HV * V, generator=gen, dtype=torch.float64)
    a = torch.zeros(B, HV, dtype=torch.float64)
    b = torch.zeros(B, HV, dtype=torch.float64)
    A_log = torch.zeros(HV, dtype=torch.float64)
    dt_bias = torch.zeros(HV, dtype=torch.float64)
    state = torch.randn(4, HV, V, K, generator=gen, dtype=torch.float64)
    before = state.clone()
    out = torch.zeros(B, 1, HV, V, dtype=torch.float64)

    ret_out, ret_state = packed_decode(
        mixed, a, b, A_log, dt_bias, 0.5, state, out, torch.tensor([2, 3])
    )
    assert ret_out is out
    assert ret_state is state
    # Only the addressed rows move.
    torch.testing.assert_close(state[0], before[0], rtol=0, atol=0)
    torch.testing.assert_close(state[1], before[1], rtol=0, atol=0)
    assert not torch.equal(state[2], before[2]) and not torch.equal(state[3], before[3])
    assert torch.count_nonzero(out) > 0


def test_packed_decode_multiple_sequences_and_slots():
    H, HV, K, V, B = 2, 4, 3, 2, 3
    tol = _tolerances(torch.float64)
    gen = torch.Generator().manual_seed(11)
    mixed = torch.randn(B, 2 * H * K + HV * V, generator=gen, dtype=torch.float64)
    a = torch.randn(B, HV, generator=gen, dtype=torch.float64) * 0.5
    b = torch.randn(B, HV, generator=gen, dtype=torch.float64)
    A_log = torch.randn(HV, generator=gen, dtype=torch.float64) * 0.3 + 0.2
    dt_bias = torch.randn(HV, generator=gen, dtype=torch.float64) * 0.1
    state = torch.randn(6, HV, V, K, generator=gen, dtype=torch.float64)
    before = state.clone()
    slots = torch.tensor([3, 1, 5])

    out = torch.zeros(B, 1, HV, V, dtype=torch.float64)
    packed_decode(mixed, a, b, A_log, dt_bias, K**-0.5, state, out, slots)
    orc_o, orc_state = _packed_oracle(mixed, a, b, A_log, dt_bias, K**-0.5, before, slots)
    torch.testing.assert_close(out.view(B, HV, V), orc_o, **tol)
    torch.testing.assert_close(state, orc_state, **tol)
    # Rows are independent: the three addressed slots moved, the three others did not, and the two
    # sequences that read different slots produced different outputs.
    for row in range(6):
        if row in (1, 3, 5):
            assert not torch.equal(state[row], before[row])
        else:
            torch.testing.assert_close(state[row], before[row], rtol=0, atol=0)
    assert not torch.allclose(out[0], out[1]) and not torch.allclose(out[1], out[2])


def test_packed_decode_null_slot_zeroes_out_and_skips_state():
    """Kernel lines 296-299: ``state_idx <= 0`` zeroes ``o`` and returns before touching state."""
    H, HV, K, V, B = 2, 2, 2, 2, 2
    gen = torch.Generator().manual_seed(13)
    mixed = torch.randn(B, 2 * H * K + HV * V, generator=gen, dtype=torch.float64)
    a = torch.zeros(B, HV, dtype=torch.float64)
    b = torch.zeros(B, HV, dtype=torch.float64)
    A_log = torch.zeros(HV, dtype=torch.float64)
    dt_bias = torch.zeros(HV, dtype=torch.float64)
    state = torch.randn(3, HV, V, K, generator=gen, dtype=torch.float64)
    before = state.clone()
    out = torch.full((B, 1, HV, V), float("nan"), dtype=torch.float64)

    packed_decode(mixed, a, b, A_log, dt_bias, 0.5, state, out, torch.tensor([0, 2]))
    torch.testing.assert_close(out[0], torch.zeros(1, HV, V, dtype=torch.float64), rtol=0, atol=0)
    assert torch.count_nonzero(out[1]) > 0
    torch.testing.assert_close(state[0], before[0], rtol=0, atol=0)
    torch.testing.assert_close(state[1], before[1], rtol=0, atol=0)
    assert not torch.equal(state[2], before[2])


def test_sequence_null_slot_skips_sequence_entirely():
    """Kernel lines 113-115: an invalid initial slot makes the program return for that sequence."""
    T, H, HV, K, V, slots = 1, 2, 2, 2, 2, 4
    q, k, v, a, b, A_log, dt_bias = _rand(T, H, HV, K, V, B=2, seed=15)
    g = _naive_gate(a, A_log, dt_bias).view(2, T, HV)
    beta = torch.sigmoid(b).view(2, T, HV)
    state = torch.randn(slots, HV, V, K, dtype=torch.float64)
    before = state.clone()
    indices = torch.tensor([0, 3])

    o, _ = seq_form(q, k, v, g, beta, initial_state=state, ssm_state_indices=indices)
    orc_o, orc_state = _seq_oracle(q, k, v, g, beta, before, ssm_state_indices=indices)
    tol = _tolerances(torch.float64)
    torch.testing.assert_close(o, orc_o, **tol)
    torch.testing.assert_close(state, orc_state, **tol)
    # Documented deviation (D2): upstream leaves o uninitialised for a skipped sequence; we zero it.
    assert torch.count_nonzero(o[0]) == 0
    assert torch.count_nonzero(o[1]) > 0
    torch.testing.assert_close(state[0], before[0], rtol=0, atol=0)


def test_inplace_final_state_false_sequence():
    T, H, HV, K, V = 4, 2, 2, 3, 3
    q, k, v, a, b, A_log, dt_bias = _rand(T, H, HV, K, V, seed=17)
    g = _naive_gate(a, A_log, dt_bias).view(1, T, HV)
    beta = torch.sigmoid(b).view(1, T, HV)
    state = torch.randn(2, HV, V, K, dtype=torch.float64)
    before = state.clone()

    o, final_state = seq_form(q, k, v, g, beta, initial_state=state, inplace_final_state=False)
    assert torch.equal(state, before), "inplace_final_state=False must not touch the input"
    assert final_state.shape == (T, HV, V, K) and final_state is not state

    orc_o, orc_states = _seq_oracle(q, k, v, g, beta, before, inplace_final_state=False)
    tol = _tolerances(torch.float64)
    torch.testing.assert_close(o, orc_o, **tol)
    torch.testing.assert_close(final_state, orc_states, **tol)


def test_inplace_final_state_false_rejects_multi_sequence_batch():
    """Lines 205/163: a (T, HV, V, K) buffer written at rows ``bos + t`` only works for B == 1."""
    T, H, HV, K, V = 2, 2, 2, 2, 2
    q, k, v, a, b, A_log, dt_bias = _rand(T, H, HV, K, V, B=2, seed=19)
    g = _naive_gate(a, A_log, dt_bias).view(2, T, HV)
    with pytest.raises(ValueError, match="B == 1"):
        seq_form(
            q, k, v, g, torch.sigmoid(b).view(2, T, HV),
            initial_state=torch.zeros(2, HV, V, K, dtype=torch.float64),
            inplace_final_state=False,
        )


def test_spec_decoding_num_accepted_tokens():
    T, H, HV, K, V, slots = 3, 2, 2, 2, 2, 6
    q, k, v, a, b, A_log, dt_bias = _rand(T, H, HV, K, V, B=2, seed=21)
    g = _naive_gate(a, A_log, dt_bias).view(2, T, HV)
    beta = torch.sigmoid(b).view(2, T, HV)
    state = torch.randn(slots, HV, V, K, dtype=torch.float64)
    before = state.clone()
    indices = torch.tensor([[4, 5, 3], [1, 2, 0]])
    num_accepted = torch.tensor([2, 1])

    o, _ = seq_form(
        q, k, v, g, beta, initial_state=state, ssm_state_indices=indices,
        num_accepted_tokens=num_accepted,
    )
    orc_o, orc_state = _seq_oracle(
        q, k, v, g, beta, before, ssm_state_indices=indices,
        num_accepted_tokens=num_accepted,
    )
    tol = _tolerances(torch.float64)
    torch.testing.assert_close(o, orc_o, **tol)
    torch.testing.assert_close(state, orc_state, **tol)
    # The initial read is governed by num_accepted_tokens -> a different count changes the result.
    o_other, _ = seq_form(
        q, k, v, g, beta, initial_state=before.clone(), ssm_state_indices=indices,
        num_accepted_tokens=torch.tensor([3, 3]),
    )
    assert not torch.allclose(o, o_other)


# ---------------------------------------------------------------------------------------------
# 5. GQA, gate wiring, headwise beta, varlen
# ---------------------------------------------------------------------------------------------


def test_gqa_maps_value_heads_to_key_heads():
    """Line 64: ``i_h = i_hv // (HV // H)`` -- two value heads share one key head here."""
    T, H, HV, K, V = 3, 2, 4, 3, 2
    q, k, v, a, b, A_log, dt_bias = _rand(T, H, HV, K, V, seed=23)
    g = _naive_gate(a, A_log, dt_bias).view(1, T, HV)
    beta = torch.sigmoid(b).view(1, T, HV)
    state = torch.randn(2, HV, V, K, dtype=torch.float64)

    o, _ = seq_form(q, k, v, g, beta, initial_state=state.clone(), ssm_state_indices=SLOT1)
    orc_o, _ = _seq_oracle(q, k, v, g, beta, state.clone(), ssm_state_indices=SLOT1)
    torch.testing.assert_close(o, orc_o, **_tolerances(torch.float64))

    # Structural check: zeroing key head 0 changes value heads 0,1 (which read it) only.
    k_zero = k.clone()
    k_zero[:, :, 0] = 0.0
    o0, _ = seq_form(q, k_zero, v, g, beta, initial_state=state.clone(), ssm_state_indices=SLOT1)
    assert not torch.allclose(o0[:, :, 0], o[:, :, 0])
    assert not torch.allclose(o0[:, :, 1], o[:, :, 1])
    assert torch.allclose(o0[:, :, 2], o[:, :, 2])
    assert torch.allclose(o0[:, :, 3], o[:, :, 3])


def test_gate_wiring_uses_a_b_A_log_dt_bias_only():
    """The decode gate must be ``-exp(A_log)*softplus(a+dt_bias)`` and ``sigmoid(b)``."""
    H, HV, K, V = 2, 2, 3, 2
    mixed = torch.randn(1, 2 * H * K + HV * V, generator=torch.Generator().manual_seed(25),
                        dtype=torch.float64)
    a = torch.randn(1, HV, dtype=torch.float64)
    b = torch.randn(1, HV, dtype=torch.float64)
    A_log = torch.randn(HV, dtype=torch.float64)
    dt_bias = torch.randn(HV, dtype=torch.float64)
    state = torch.randn(2, HV, V, K, dtype=torch.float64)

    def run(a_, b_, Al_, d_, st):
        out = torch.zeros(1, 1, HV, V, dtype=torch.float64)
        packed_decode(mixed, a_, b_, Al_, d_, 0.7, st, out, SLOT1)
        return out.view(HV, V)

    base = run(a, b, A_log, dt_bias, state.clone())
    # softplus is strictly monotone -> perturbing dt_bias by head changes only that head.
    dt2 = dt_bias.clone()
    dt2[0] += 1.0
    got = run(a, b, A_log, dt2, state.clone())
    assert not torch.allclose(got[0], base[0])
    torch.testing.assert_close(got[1], base[1], rtol=0, atol=0)
    # b feeds only the delta-rule step, and `a` only the gate.
    b2 = b.clone()
    b2[0, 1] += 1.0
    got = run(a, b2, A_log, dt_bias, state.clone())
    assert not torch.allclose(got[1], base[1])
    torch.testing.assert_close(got[0], base[0], rtol=0, atol=0)
    a2 = a.clone()
    a2[0, 0] += 1.0  # larger gate -> smaller decay -> different output for that head
    got = run(a2, b, A_log, dt_bias, state.clone())
    assert not torch.allclose(got[0], base[0])
    torch.testing.assert_close(got[1], base[1], rtol=0, atol=0)


def test_headwise_beta_sequence():
    """Line 244: ``IS_BETA_HEADWISE = beta.ndim == v.ndim`` -- one step per (head, channel)."""
    T, H, HV, K, V = 3, 2, 2, 3, 2
    q, k, v, a, b, A_log, dt_bias = _rand(T, H, HV, K, V, seed=27)
    g = _naive_gate(a, A_log, dt_bias).view(1, T, HV)
    beta = torch.sigmoid(torch.randn(1, T, HV, V, dtype=torch.float64))
    state = torch.randn(2, HV, V, K, dtype=torch.float64)

    o, st = seq_form(
        q, k, v, g, beta, initial_state=state.clone(), ssm_state_indices=SLOT1
    )
    orc_o, orc_state = _seq_oracle(q, k, v, g, beta, state.clone(), ssm_state_indices=SLOT1)
    tol = _tolerances(torch.float64)
    torch.testing.assert_close(o, orc_o, **tol)
    torch.testing.assert_close(st, orc_state, **tol)

    # A per-channel beta must differ from the per-head beta it broadcasts from.
    o_shared, _ = seq_form(
        q, k, v, g, beta[..., 0].contiguous(), initial_state=state.clone(),
        ssm_state_indices=SLOT1,
    )
    assert not torch.allclose(o, o_shared)


def test_beta_none_defaults_to_ones():
    q, k, v, a, b, A_log, dt_bias = _rand(2, 2, 2, 3, 2, seed=29)
    g = _naive_gate(a, A_log, dt_bias).view(1, 2, 2)
    state = torch.randn(2, 2, 2, 3, dtype=torch.float64)
    o_none, _ = seq_form(q, k, v, g, initial_state=state.clone(), ssm_state_indices=SLOT1)
    o_ones, _ = seq_form(
        q, k, v, g, torch.ones(1, 2, 2, dtype=torch.float64), initial_state=state.clone(),
        ssm_state_indices=SLOT1,
    )
    torch.testing.assert_close(o_none, o_ones, rtol=0, atol=0)


def test_beta_none_with_gqa_uses_value_head_count():
    """Documented deviation D1: upstream's ``ones_like(q[..., 0])`` has H columns, not HV."""
    q, k, v, a, b, A_log, dt_bias = _rand(1, 2, 4, 3, 2, seed=31)  # H=2, HV=4
    g = _naive_gate(a, A_log, dt_bias).view(1, 1, 4)
    st = torch.randn(2, 4, 2, 3, dtype=torch.float64)

    # With H=2, HV=4 upstream's ones tensor has only 2 columns per token while the kernel reads it
    # with stride HV=4, i.e. past the end for every token.  The port must equal explicit HV ones.
    o, _ = seq_form(q, k, v, g, initial_state=st.clone(), ssm_state_indices=SLOT1)
    o_explicit, _ = seq_form(
        q, k, v, g, torch.ones(1, 1, 4, dtype=torch.float64),
        initial_state=st.clone(), ssm_state_indices=SLOT1,
    )
    torch.testing.assert_close(o, o_explicit, rtol=0, atol=0)
    assert torch.count_nonzero(o) > 0


def test_varlen_cu_seqlens():
    H, HV, K, V, slots = 2, 2, 3, 2, 5
    lens = [2, 1, 3]
    T = sum(lens)
    q, k, v, a, b, A_log, dt_bias = _rand(T, H, HV, K, V, seed=33)
    g = _naive_gate(a, A_log, dt_bias).view(1, T, HV)
    beta = torch.sigmoid(b).view(1, T, HV)
    cu_seqlens = torch.tensor([0, 2, 3, 6])
    state = torch.randn(slots, HV, V, K, dtype=torch.float64)
    before = state.clone()
    indices = torch.tensor([2, 0, 4])  # sequence 1 has no valid state -> skipped entirely

    o, _ = seq_form(
        q, k, v, g, beta, initial_state=state, cu_seqlens=cu_seqlens,
        ssm_state_indices=indices,
    )
    orc_o, orc_state = _seq_oracle(
        q, k, v, g, beta, before, cu_seqlens=cu_seqlens, ssm_state_indices=indices
    )
    tol = _tolerances(torch.float64)
    torch.testing.assert_close(o, orc_o, **tol)
    torch.testing.assert_close(state, orc_state, **tol)
    assert torch.count_nonzero(o[0, 2]) == 0
    torch.testing.assert_close(state[0], before[0], rtol=0, atol=0)


def test_varlen_requires_flat_batch():
    q, k, v, a, b, A_log, dt_bias = _rand(2, 2, 2, 2, 2, B=2, seed=35)
    g = _naive_gate(a, A_log, dt_bias).view(2, 2, 2)
    with pytest.raises(ValueError, match="batch size is expected to be 1"):
        seq_form(
            q, k, v, g, torch.sigmoid(b).view(2, 2, 2),
            initial_state=torch.zeros(4, 2, 2, 2, dtype=torch.float64),
            cu_seqlens=torch.tensor([0, 2]),
        )


# ---------------------------------------------------------------------------------------------
# 6. dtype / scale / l2norm
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16, torch.float16])
def test_dtype_preservation_sequence(dtype):
    T, H, HV, K, V = 3, 2, 2, 4, 4
    q, k, v, a, b, A_log, dt_bias = _rand(T, H, HV, K, V, seed=37)
    qd, kd, vd, ad, bd = (t.to(dtype) for t in (q, k, v, a, b))
    g = _naive_gate(a, A_log, dt_bias).view(1, T, HV).to(dtype)
    beta = torch.sigmoid(b).view(1, T, HV).to(dtype)
    state = torch.randn(2, HV, V, K, dtype=torch.float32)
    before = state.clone()

    o, final_state = seq_form(
        qd, kd, vd, g, beta, initial_state=state, ssm_state_indices=SLOT1
    )
    assert o.dtype == dtype and o.shape == (1, T, HV, V)
    assert final_state is state and final_state.dtype == torch.float32

    orc_o, orc_state = _seq_oracle(
        qd, kd, vd, g, beta, before, ssm_state_indices=SLOT1
    )
    torch.testing.assert_close(o.double(), orc_o, **_tolerances(dtype))
    torch.testing.assert_close(state.double(), orc_state, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16, torch.float16])
def test_dtype_preservation_packed_decode(dtype):
    H, HV, K, V, B = 2, 2, 3, 2, 2
    gen = torch.Generator().manual_seed(41)
    mixed = torch.randn(B, 2 * H * K + HV * V, generator=gen, dtype=torch.float64).to(dtype)
    a = (torch.randn(B, HV, generator=gen, dtype=torch.float64) * 0.5).to(dtype)
    b = torch.randn(B, HV, generator=gen, dtype=torch.float64).to(dtype)
    A_log = (torch.randn(HV, generator=gen, dtype=torch.float64) * 0.3 + 0.2).to(dtype)
    dt_bias = torch.randn(HV, generator=gen, dtype=torch.float64).to(dtype)
    # A low-precision state too: the decode kernel writes the state in the *state* dtype (line 336).
    state_dtype = torch.float16 if dtype in (torch.float16, torch.bfloat16) else dtype
    state = torch.randn(3, HV, V, K, generator=gen, dtype=torch.float64).to(state_dtype)
    before = state.clone()
    out = torch.zeros(B, 1, HV, V, dtype=dtype)

    ret_out, ret_state = packed_decode(
        mixed, a, b, A_log, dt_bias, K**-0.5, state, out, torch.tensor([1, 2])
    )
    assert ret_out is out and out.dtype == dtype
    assert ret_state is state and state.dtype == state_dtype

    orc_o, orc_state = _packed_oracle(
        mixed, a, b, A_log, dt_bias, K**-0.5, before, torch.tensor([1, 2])
    )
    # ``out`` carries the input dtype, the state carries the state dtype (line 336).
    torch.testing.assert_close(out.view(B, HV, V).double(), orc_o, **_tolerances(dtype))
    torch.testing.assert_close(state.double(), orc_state, **_tolerances(state_dtype))


def test_sequence_output_dtype_follows_q_not_v():
    """Line 201: ``o = q.new_empty(...)`` -- the output buffer inherits ``q.dtype``."""
    T, H, HV, K, V = 2, 2, 2, 3, 3
    q, k, v, a, b, A_log, dt_bias = _rand(T, H, HV, K, V, seed=43)
    g = _naive_gate(a, A_log, dt_bias).view(1, T, HV)
    o, _ = seq_form(
        q.float(), k.float(), v.half(), g.float(), torch.sigmoid(b).view(1, T, HV).float(),
        initial_state=torch.randn(2, HV, V, K, dtype=torch.float32),
        ssm_state_indices=SLOT1,
    )
    assert o.dtype == torch.float32


def test_scale_default_and_linearity_in_q():
    T, H, HV, K, V = 3, 2, 2, 4, 3
    q, k, v, a, b, A_log, dt_bias = _rand(T, H, HV, K, V, seed=45)
    g = _naive_gate(a, A_log, dt_bias).view(1, T, HV)
    beta = torch.sigmoid(b).view(1, T, HV)
    state = torch.randn(2, HV, V, K, dtype=torch.float64)
    tol = _tolerances(torch.float64)

    o_default, _ = seq_form(q, k, v, g, beta, initial_state=state.clone(), ssm_state_indices=SLOT1)
    o_explicit, _ = seq_form(
        q, k, v, g, beta, scale=K**-0.5, initial_state=state.clone(),
        ssm_state_indices=SLOT1,
    )
    torch.testing.assert_close(o_default, o_explicit, **tol)

    o1, h1 = seq_form(
        q, k, v, g, beta, scale=1.0, initial_state=state.clone(), ssm_state_indices=SLOT1
    )
    o5, h5 = seq_form(
        q, k, v, g, beta, scale=5.0, initial_state=state.clone(), ssm_state_indices=SLOT1
    )
    torch.testing.assert_close(o5, 5.0 * o1, **tol)  # scale touches q only, not k or the state
    assert torch.equal(h5, h1)

    with pytest.raises(AssertionError, match="scale must be positive"):
        seq_form(q, k, v, g, beta, scale=0.0, initial_state=state.clone(), ssm_state_indices=SLOT1)


def test_l2norm_both_entry_points():
    T, H, HV, K, V = 2, 2, 2, 3, 3
    q, k, v, a, b, A_log, dt_bias = _rand(T, H, HV, K, V, seed=47)
    g = _naive_gate(a, A_log, dt_bias).view(1, T, HV)
    beta = torch.sigmoid(b).view(1, T, HV)
    state = torch.randn(2, HV, V, K, dtype=torch.float64)
    tol = _tolerances(torch.float64)

    qn = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6)
    kn = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    o_norm, _ = seq_form(
        q, k, v, g, beta, initial_state=state.clone(), ssm_state_indices=SLOT1,
        use_qk_l2norm_in_kernel=True,
    )
    o_ref, _ = seq_form(
        qn, kn, v, g, beta, initial_state=state.clone(), ssm_state_indices=SLOT1
    )
    torch.testing.assert_close(o_norm, o_ref, **tol)
    o_plain, _ = seq_form(q, k, v, g, beta, initial_state=state.clone(), ssm_state_indices=SLOT1)
    assert not torch.allclose(o_norm, o_plain)

    # Packed decode: same flag, same normalisation (lines 313-315), same result.
    mixed = _pack(q[0, 0], k[0, 0], v[0, 0])
    out1 = torch.zeros(1, 1, HV, V, dtype=torch.float64)
    packed_decode(
        mixed, a[:1], b[:1], A_log, dt_bias, K**-0.5, state.clone(), out1,
        SLOT1, use_qk_l2norm_in_kernel=True,
    )
    mixed_n = _pack(qn[0, 0], kn[0, 0], v[0, 0])
    out2 = torch.zeros(1, 1, HV, V, dtype=torch.float64)
    packed_decode(mixed_n, a[:1], b[:1], A_log, dt_bias, K**-0.5, state.clone(), out2, SLOT1)
    torch.testing.assert_close(out1, out2, **tol)


# ---------------------------------------------------------------------------------------------
# 7. Argument plumbing
# ---------------------------------------------------------------------------------------------


def test_noncontiguous_inputs_and_state_row_slice():
    """Upstream calls ``.contiguous()`` on q/k/v/g/beta (lines 499-504); a state row slice with a
    non-zero storage offset is the vLLM paged layout and must be read/written in place."""
    T, H, HV, K, V = 3, 2, 2, 3, 2
    q2, k2, v2, a2, b2, A_log, dt_bias = _rand(2 * T, H, HV, K, V, seed=55)
    q, k, v = q2[0, ::2][None], k2[0, ::2][None], v2[0, ::2][None]  # strided views
    a, b = a2[::2], b2[::2]  # (T, HV) with stride(0) == 2*HV
    assert not q.is_contiguous() and not a.is_contiguous()
    g = _naive_gate(a, A_log, dt_bias).view(1, T, HV)
    beta = torch.sigmoid(b).view(1, T, HV)

    # Rows [T, 2T) of a bigger buffer: storage_offset != 0, dense rows (A4).
    big = torch.randn(2 * T, HV, V, K, dtype=torch.float64)
    init = big[T:].clone()
    view = big[T:]
    o, _ = seq_form(q, k, v, g, beta, initial_state=view)

    orc_o, orc_state = _seq_oracle(q, k, v, g, beta, init)
    tol = _tolerances(torch.float64)
    torch.testing.assert_close(o, orc_o, **tol)
    torch.testing.assert_close(view, orc_state, **tol)


def test_sequence_initial_state_is_mandatory():
    q, k, v, a, b, A_log, dt_bias = _rand(1, 2, 2, 2, 2, seed=49)
    g = _naive_gate(a, A_log, dt_bias).view(1, 1, 2)
    with pytest.raises(ValueError, match="initial_state is required"):
        seq_form(q, k, v, g, torch.sigmoid(b).view(1, 1, 2))


def test_state_layout_validation():
    T, H, HV, K, V = 1, 2, 2, 3, 2  # K != V so a (HV, K, V) mix-up has a different shape
    q, k, v, a, b, A_log, dt_bias = _rand(T, H, HV, K, V, seed=51)
    g = _naive_gate(a, A_log, dt_bias).view(1, T, HV)
    beta = torch.sigmoid(b).view(1, T, HV)
    with pytest.raises(ValueError, match=r"\(rows, HV, V, K\)"):
        seq_form(q, k, v, g, beta, initial_state=torch.zeros(1, HV, K, V, dtype=torch.float64))
    with pytest.raises(ValueError, match="contiguous"):
        bad = torch.zeros(1, HV, K, V, dtype=torch.float64).transpose(-1, -2)
        seq_form(q, k, v, g, beta, initial_state=bad)
    # Packed decode enforces the same layout (lines 301-303 / 334-336).
    mixed = _pack(q[0, 0], k[0, 0], v[0, 0])
    with pytest.raises(ValueError, match="contiguous"):
        packed_decode(
            mixed, a[:1], b[:1], A_log, dt_bias, 0.5,
            torch.zeros(1, HV, K, V, dtype=torch.float64).transpose(-1, -2),
            torch.zeros(1, 1, HV, V, dtype=torch.float64), SLOT1,
        )


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda t: t[0], "must be a 2D tensor"),
        (lambda t: t.t(), "contiguous in the last dim"),
    ],
)
def test_packed_decode_rejects_bad_mixed_qkv(mutate, match):
    H, HV, K, V = 2, 2, 2, 2
    mixed = torch.zeros(1, 2 * H * K + HV * V, dtype=torch.float64)
    state = torch.zeros(2, HV, V, K, dtype=torch.float64)
    out = torch.zeros(1, 1, HV, V, dtype=torch.float64)
    a = torch.zeros(1, HV, dtype=torch.float64)
    b = torch.zeros(1, HV, dtype=torch.float64)
    A_log = torch.zeros(HV, dtype=torch.float64)
    dt_bias = torch.zeros(HV, dtype=torch.float64)
    bad = mutate(mixed)
    with pytest.raises(ValueError, match=match):
        packed_decode(bad, a, b, A_log, dt_bias, 0.5, state, out, SLOT1)


def test_packed_decode_rejects_bad_shapes():
    H, HV, K, V = 2, 2, 2, 2
    mixed = torch.zeros(2, 2 * H * K + HV * V, dtype=torch.float64)
    state = torch.zeros(4, HV, V, K, dtype=torch.float64)
    a = torch.zeros(2, HV, dtype=torch.float64)
    b = torch.zeros(2, HV, dtype=torch.float64)
    A_log = torch.zeros(HV, dtype=torch.float64)
    dt_bias = torch.zeros(HV, dtype=torch.float64)
    out = torch.zeros(2, 1, HV, V, dtype=torch.float64)
    idx = torch.tensor([1, 2])

    with pytest.raises(ValueError, match="Mismatched batch sizes"):
        packed_decode(mixed, a[:1], b, A_log, dt_bias, 0.5, state, out, idx)
    with pytest.raises(ValueError, match="must have shape"):
        packed_decode(mixed, a, b, A_log, dt_bias, 0.5, state, out[:1], idx)
    with pytest.raises(ValueError, match="must be 1D"):
        packed_decode(mixed, a, b, A_log, dt_bias, 0.5, state, out, torch.tensor([[1], [2]]))
    with pytest.raises(ValueError, match="`A_log` and `dt_bias` must have 2 elements"):
        packed_decode(
            mixed, a, b, torch.zeros(3, dtype=torch.float64), dt_bias, 0.5, state, out, idx
        )
    with pytest.raises(ValueError, match="4D tensor"):
        packed_decode(mixed, a, b, A_log, dt_bias, 0.5, state[0, 0], out, idx)
    with pytest.raises(ValueError, match="Invalid packed `mixed_qkv` last dim"):
        packed_decode(mixed[:, :-1], a, b, A_log, dt_bias, 0.5, state, out, idx)
    if os.environ.get("TILEINFER_GDN_NO_SYNC", "").strip().lower() in ("1", "true", "yes"):
        # Documented deviation of the sync-free decode path: it indexes the state with a *device*
        # tensor, so it cannot raise on an out-of-range slot -- it clamps instead.  The loud check
        # is what the default path buys, which is why the fast path has to be opted into.
        packed_decode(mixed, a, b, A_log, dt_bias, 0.5, state, out, torch.tensor([1, 99]))
    else:
        with pytest.raises(ValueError, match="out of range"):
            packed_decode(mixed, a, b, A_log, dt_bias, 0.5, state, out, torch.tensor([1, 99]))


def test_packed_decode_rows_are_independent_single_tokens():
    """One call == one token per row: the grid has no token loop (line 449, A7).

    A ``B``-row call must therefore equal ``B`` independent one-row calls, each reading its state
    row from the snapshot the batch call saw (the concurrent programs never see each other's write).
    """
    H, HV, K, V, B = 2, 2, 3, 2, 2
    gen = torch.Generator().manual_seed(53)
    mixed = torch.randn(B, 2 * H * K + HV * V, generator=gen, dtype=torch.float64)
    a = torch.randn(B, HV, generator=gen, dtype=torch.float64)
    b = torch.randn(B, HV, generator=gen, dtype=torch.float64)
    A_log = torch.zeros(HV, dtype=torch.float64)
    dt_bias = torch.zeros(HV, dtype=torch.float64)
    state = torch.randn(3, HV, V, K, generator=gen, dtype=torch.float64)
    snapshot = state.clone()
    slots = torch.tensor([1, 2])

    out = torch.zeros(B, 1, HV, V, dtype=torch.float64)
    packed_decode(mixed, a, b, A_log, dt_bias, K**-0.5, state, out, slots)
    for n in range(B):
        # Same arithmetic, one row, a two-row state whose row 1 is this row's snapshot.
        single_state = torch.stack([snapshot[0], snapshot[1 + n]])
        single = torch.zeros(1, 1, HV, V, dtype=torch.float64)
        packed_decode(
            mixed[n : n + 1], a[n : n + 1], b[n : n + 1], A_log, dt_bias, K**-0.5,
            single_state, single, SLOT1,
        )
        torch.testing.assert_close(out[n], single[0], rtol=0, atol=0)
        torch.testing.assert_close(state[1 + n], single_state[1], rtol=0, atol=0)
