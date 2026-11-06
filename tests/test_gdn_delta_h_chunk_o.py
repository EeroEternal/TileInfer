"""Tests for the torch ports of ``chunk_gated_delta_rule_fwd_h`` and ``chunk_fwd_o``.

These are the last two operators of the vLLM 0.23 GDN chunk chain and they are adjacent, so the
suite tests them both in isolation and composed [vLLM chunk.py:61-82]::

    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(k, w, u, g, ...)
    o                     = chunk_fwd_o(q, k, v=v_new, h=h, g=g, scale=scale, ...)

What is pinned here (contract ``docs/notes/fla-chunk-contract.md`` §2.3-§2.4, §5, §6.3-§6.4):

* ``h[c]`` is the state **entering** chunk ``c``, laid out ``[B, NT, H, V, K]`` (``V`` before
  ``K``) in ``k.dtype`` (bf16 in production), and indexed by the global chunk ordinal
  ``chunk_offsets[i_n] + i_t``.
* ``v_new = u - w @ h[c]^T`` is stored *ungated*; the gate ``safe_exp(g_last - g)`` multiplies only
  the copy that updates the state, and ``g_last`` is the chunk's last *valid* token.
* The state update is ``S <- S * exp(g_last) + sum_t exp(g_last - g_t) v_new[t] k_t^T``, which is
  the per-token gated delta rule ``S_t = exp(d_t) S_{t-1} + beta_t (v_t - exp(d_t) S_{t-1} k_t)
  k_t^T`` iterated one chunk at a time.
* ``chunk_fwd_o``'s ``v`` argument is ``v_new`` (never the original ``v``), its triangle is ``>=``
  (the diagonal self term is kept, unlike ``kkt``'s strict ``>``), the upper triangle is discarded
  with ``torch.where`` (positive exponents would make ``inf * 0 = NaN``), and ``scale`` multiplies
  **both** the state term and the intra-chunk term.
* bf16 round-trips are contract, not noise: the ``h`` snapshot is cast to ``k.dtype``, the fp32
  state accumulator is cast to ``w.dtype`` on the way into the prediction dot, and ``v_new`` is
  cast to ``k.dtype`` before the ``k^T v_new`` dot.

Oracles
-------
1. A closed form for ``chunk_fwd_o``: for one chunk, per token ``t``,

       o_t = scale * ( exp(g_t) * (q_t . h_c) + sum_{j<=t} (q_t . k_j) exp(g_t - g_j) v_j )

   which for ``h_c = 0`` reduces to ``scale * sum_{j<=t} (q_t.k_j) exp(g_t-g_j) v_j``.  It validates
   the ``>=`` mask, the clamp and the double ``scale`` at once.
2. A per-token fp64 gated delta-rule recurrence, driven by the *original* ``(v, beta, gate)``, that
   yields (a) the state snapshots at chunk boundaries (compared with ``h``), (b) the delta-rule
   ``v_new`` and (c) the outputs ``scale * q_t S_t``.  It is the independent mathematical oracle for
   ``chunk_delta_h`` (which has no pre-existing reference, contract §6.3) and for the interface.
3. A structural fp64 recurrence fed the *same arbitrary* ``(k, w, u, g)`` as the operator, for the
   cases where ``w``/``u`` are not produced by the WY chain.

The production chain (``kkt -> solve_tril -> recompute_w_u_fwd -> fwd_h -> fwd_o``) is what the
interface tests exercise, in fp64 (where it must match the oracle to rounding) and in bf16 (with the
documented uint round-trips, hence a loose tolerance).  Negative controls show that dropping the
diagonal, multiplying by a mask, or swapping ``v_new`` back to the original ``v`` all produce
large errors.

Everything is CPU-only and deterministic (seeded generators).
"""

from __future__ import annotations

import pytest
import torch

from tileinfer.vllm.ops.fla.chunk_o import chunk_fwd_o
from tileinfer.vllm.ops.fla.delta_h import chunk_gated_delta_rule_fwd_h
from tileinfer.vllm.ops.fla.kkt import chunk_scaled_dot_kkt_fwd
from tileinfer.vllm.ops.fla.solve_tril import prepare_chunk_indices, solve_tril
from tileinfer.vllm.ops.fla.wu import recompute_w_u_fwd

BT = 64  # vLLM 0.23 pins BT = 64 for both kernels [vLLM qwen_gdn_linear_attn.py:1085-1087]
BF16_TOL = 2e-2  # 8 mantissa bits + three documented round-trips per 64-token chunk; measured 4e-3..9e-3
FP32_TOL = 2e-5  # measured 1.5e-7 for chunk_o; the chain is not fp32-exact (bg bf16 A_inv)
FP64_TOL = 1e-9  # fp64 chain vs fp64 recurrence: measured 4e-16


# -------------------------------------------------------------------------------------- chunk plans
def _dense_chunks(total: int, chunk_size: int = BT):
    return [(t0, min(chunk_size, total - t0)) for t0 in range(0, total, chunk_size)]


def _varlen_chunks(cu_seqlens, chunk_size: int = BT):
    bounds = [int(v) for v in cu_seqlens]
    chunks = []
    for bos, eos in zip(bounds[:-1], bounds[1:]):
        t0 = bos
        while t0 < eos:
            chunks.append((t0, min(chunk_size, eos - t0)))
            t0 += chunk_size
    return chunks


def _chunks(cu_seqlens, total: int, chunk_size: int = BT):
    return (
        _dense_chunks(total, chunk_size)
        if cu_seqlens is None
        else _varlen_chunks(cu_seqlens, chunk_size)
    )


def _plan(batch: int, total: int, cu_seqlens, chunk_size: int = BT):
    """``(rows, blocks, nt_dense, nt_total, offsets)`` - the oracle-side chunk plan.

    ``rows`` are ``(i_b, i_n, bos, eos)``; ``blocks`` add ``(i_t, t0, length, seq_len, slot)`` with
    ``slot`` the global chunk ordinal into ``h``.
    """
    if cu_seqlens is None:
        rows = [(b, b, 0, total) for b in range(batch)]
        nt_dense = -(-total // chunk_size)
        offsets = None
    else:
        bounds = [int(v) for v in cu_seqlens.tolist()]
        rows = [
            (0, i_n, bos, eos) for i_n, (bos, eos) in enumerate(zip(bounds[:-1], bounds[1:]))
        ]
        nt_dense = 0
        offsets = [0]
        for bos, eos in zip(bounds[:-1], bounds[1:]):
            offsets.append(offsets[-1] + -(-(eos - bos) // chunk_size))

    blocks = []
    for i_b, i_n, bos, eos in rows:
        seq_len = eos - bos
        for i_t in range(-(-seq_len // chunk_size)):
            t0 = bos + i_t * chunk_size
            length = min(chunk_size, seq_len - i_t * chunk_size)
            slot = i_b * nt_dense + i_t if offsets is None else offsets[i_n] + i_t
            blocks.append((i_b, i_n, i_t, t0, length, seq_len, slot))
    nt_total = batch * nt_dense if offsets is None else offsets[-1]
    return rows, blocks, nt_dense, nt_total, offsets


def _h_shape(batch: int, total: int, cu_seqlens, heads: int, v_dim: int, k_dim: int):
    _, _, nt_dense, nt_total, offsets = _plan(batch, total, cu_seqlens)
    if offsets is None:
        return (batch, nt_dense, heads, v_dim, k_dim)
    return (1, nt_total, heads, v_dim, k_dim)


# ----------------------------------------------------------------------------------------- builders
def _make_inputs(
    total,
    hk=2,
    hv=4,
    d=8,
    v_dim=None,
    batch=1,
    dtype=torch.bfloat16,
    seed=0,
):
    """``(q, k, v, beta)``: unit-norm keys (delta-rule-like scale), random queries/values."""
    gen = torch.Generator().manual_seed(seed)
    v_dim = d if v_dim is None else v_dim
    k = torch.nn.functional.normalize(torch.randn(batch, total, hk, d, generator=gen), dim=-1)
    q = torch.randn(batch, total, hk, d, generator=gen)
    v = torch.randn(batch, total, hv, v_dim, generator=gen)
    beta = 0.1 + 0.5 * torch.rand(batch, total, hv, generator=gen)
    return k.to(dtype), q.to(dtype), v.to(dtype), beta.to(dtype)


def _make_gate(total, hv, cu_seqlens, batch=1, decay=0.05, seed=1, dtype=torch.float32):
    """``(inc, g)``: non-positive per-token log decay and its chunk-local cumulative sum.

    Both are built in ``dtype`` so that an fp64 test compares the operator against an fp64 cumsum
    (an fp32 cumsum cast to fp64 would inject a 1e-7 relative difference into ``g``).
    """
    gen = torch.Generator().manual_seed(seed)
    inc = (-torch.rand(batch, total, hv, generator=gen) * decay).to(dtype)
    g = torch.zeros(batch, total, hv, dtype=dtype)
    for b in range(batch):
        for t0, length in _chunks(cu_seqlens, total):
            g[b, t0 : t0 + length] = inc[b, t0 : t0 + length].cumsum(0)
    return inc, g


def _make_wu(total, heads, k_dim, v_dim, batch=1, dtype=torch.bfloat16, seed=2):
    gen = torch.Generator().manual_seed(seed)
    w = torch.randn(batch, total, heads, k_dim, generator=gen)
    u = torch.randn(batch, total, heads, v_dim, generator=gen)
    return w.to(dtype), u.to(dtype)


def _make_state(rows, heads, v_dim, k_dim, dtype=torch.float32, seed=3):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(rows, heads, v_dim, k_dim, generator=gen).to(dtype)


def _gqa(heads, k_heads, device=None):
    return torch.arange(heads, device=device) // (heads // k_heads)


# ------------------------------------------------------------------------------------------ oracles
def _delta_h_structural_oracle(k, w, u, g, initial_state, cu_seqlens):
    """fp64 chunk recurrence fed the same arbitrary ``(k, w, u, g)`` as the operator.

    Returns ``(h, v_new, final_state)`` with ``h``'s slot layout identical to the module's.  The
    clamp on ``g_last - g`` is modelled explicitly (``safe_exp``), because this oracle is also used
    with a deliberately misordered gate.
    """
    k, w, u = k.double(), w.double(), u.double()
    g = None if g is None else g.double()
    B, T, hk, K = k.shape
    H, V = u.shape[-2], u.shape[-1]
    _, _, nt_dense, nt_total, offsets = _plan(B, T, cu_seqlens)
    h = torch.zeros(_h_shape(B, T, cu_seqlens, H, V, K), dtype=torch.float64)
    v_new = torch.zeros(B, T, H, V, dtype=torch.float64)
    final = torch.zeros(len(offsets) - 1 if offsets is not None else B, H, V, K, dtype=torch.float64)
    head_map = _gqa(H, hk)
    rows, blocks, _, _, _ = _plan(B, T, cu_seqlens)

    for i_b, i_n, bos, eos in rows:
        state = (
            initial_state[i_n].double()
            if initial_state is not None
            else torch.zeros(H, V, K, dtype=torch.float64)
        )
        for i_t in range(-(-(eos - bos) // BT)):
            t0 = bos + i_t * BT
            length = min(BT, eos - t0)
            idx = torch.arange(t0, t0 + length)
            slot = i_b * nt_dense + i_t if offsets is None else offsets[i_n] + i_t
            if offsets is None:
                h[i_b, i_t] = state
            else:
                h[0, slot] = state

            pred = torch.einsum("thk,hvk->thv", w[i_b, idx], state)
            val = u[i_b, idx] - pred
            v_new[i_b, idx] = val
            if g is not None:
                glast = g[i_b, t0 + length - 1]  # last VALID token [sglang chunk_delta_h.py:178]
                diff = glast[None, :] - g[i_b, idx]
                val = val * torch.exp(torch.where(diff <= 0, diff, torch.full_like(diff, float("-inf")))).unsqueeze(-1)
                state = state * torch.exp(glast)[:, None, None]
            kb = k[i_b, idx][:, head_map]
            state = state + torch.einsum("thk,thv->hvk", kb, val)
        final[i_n] = state
    return h, v_new, final


def _per_token_delta_rule(q, k, v, beta, inc, initial_state, scale, cu_seqlens):
    """Full per-token fp64 gated delta rule, from the *original* ``(v, beta, gate)``.

    ``S_t = exp(d_t) S_{t-1} + beta_t (v_t - exp(d_t) S_{t-1} k_t) k_t^T`` and ``o_t = scale q_t S_t``
    (the decay applies before the delta correction, and the output reads the updated state - the
    convention of ``fused_recurrent_gated_delta_rule`` and of the chunk chain, see
    ``tileinfer/vllm/ops/fla/recurrent.py``).

    Returns ``(o, v_new, h, final)``: ``v_new_t = beta_t (v_t - exp(d_t) S_{t-1} k_t)`` (ungated),
    ``h`` the chunk-entry snapshots in the module's slot layout.
    """
    q, k, v, beta, inc = q.double(), k.double(), v.double(), beta.double(), inc.double()
    B, T, hk, K = k.shape
    H, V = v.shape[-2], v.shape[-1]
    rows, blocks, nt_dense, nt_total, offsets = _plan(B, T, cu_seqlens)
    o = torch.zeros(B, T, H, V, dtype=torch.float64)
    v_new = torch.zeros(B, T, H, V, dtype=torch.float64)
    h = torch.zeros(_h_shape(B, T, cu_seqlens, H, V, K), dtype=torch.float64)
    final = torch.zeros(len(rows), H, V, K, dtype=torch.float64)
    head_map = _gqa(H, hk)

    chunk_start = {}
    for i_b, i_n, i_t, t0, length, seq_len, slot in blocks:
        chunk_start[(i_b, t0)] = (i_n, i_t, slot)

    for i_b, i_n, bos, eos in rows:
        state = (
            initial_state[i_n].double()
            if initial_state is not None
            else torch.zeros(H, V, K, dtype=torch.float64)
        )
        for t in range(bos, eos):
            if (i_b, t) in chunk_start:
                _, i_t, slot = chunk_start[(i_b, t)]
                if offsets is None:
                    h[i_b, i_t] = state
                else:
                    h[0, slot] = state
            decay = torch.exp(inc[i_b, t])  # [H]
            state = state * decay[:, None, None]
            kt = k[i_b, t][head_map]  # [H, K] (2-D: head_map indexes dim 0)
            vt = v[i_b, t]  # [H, V]
            delta = beta[i_b, t][:, None] * (vt - (state * kt[:, None, :]).sum(-1))
            v_new[i_b, t] = delta
            state = state + delta[:, :, None] * kt[:, None, :]
            qt = q[i_b, t][head_map]  # [H, K]
            o[i_b, t] = scale * (state * qt[:, None, :]).sum(-1)
        final[i_n] = state
    return o, v_new, h, final


def _chunk_o_oracle(q, k, v, h, g, scale, cu_seqlens):
    """fp64 definition of ``chunk_fwd_o``: per chunk and token, ``scale * (exp(g_t) q_t h_c +
    sum_{j<=t} (q_t.k_j) exp(g_t-g_j) v_j)``.

    The mask is applied *before* the exponent so no positive exponent is evaluated (a structurally
    different way of avoiding the overflow than the implementation's clamp + ``where``).
    """
    q, k, v, h = q.double(), k.double(), v.double(), h.double()
    g = None if g is None else g.double()
    B, T, hk, K = q.shape
    H, V = v.shape[-2], v.shape[-1]
    rows, blocks, nt_dense, nt_total, offsets = _plan(B, T, cu_seqlens)
    o = torch.zeros(B, T, H, V, dtype=torch.float64)
    head_map = _gqa(H, hk)

    for i_b, i_n, i_t, t0, length, seq_len, slot in blocks:
        idx = torch.arange(t0, t0 + length)
        h_blk = h[i_b, i_t] if offsets is None else h[0, slot]
        qb = q[i_b, idx][:, head_map]
        kb = k[i_b, idx][:, head_map]
        vb = v[i_b, idx]
        inter = torch.einsum("thk,hvk->thv", qb, h_blk)
        qk = torch.einsum("thk,shk->tsh", qb, kb)
        pos = torch.arange(length)
        keep = pos[:, None] >= pos[None, :]
        a = qk
        if g is not None:
            gb = g[i_b, idx]
            inter = inter * torch.exp(gb).unsqueeze(-1)
            diff = gb[:, None, :] - gb[None, :, :]
            a = qk * torch.exp(torch.where(keep.unsqueeze(-1), diff, torch.zeros_like(diff)))
        a = torch.where(keep.unsqueeze(-1), a, torch.zeros((), dtype=torch.float64))
        o[i_b, idx] = (inter + torch.einsum("tsh,shv->thv", a, vb)) * scale
    return o


def _chain(q, k, v, beta, g, initial_state, scale, cu_seqlens):
    """kkt -> solve_tril -> recompute_w_u -> fwd_h -> fwd_o, the production slice of chunk.py:41-82.

    ``A``'s and ``A_inv``'s dtypes follow whatever the caller set up through ``k.dtype``/``g.dtype``;
    for the fp64 test pass fp64 inputs (kkt defaults to fp32 output, so it is overridden there).
    """
    out_dtype = torch.float64 if k.dtype == torch.float64 else torch.float32
    a_kkt = chunk_scaled_dot_kkt_fwd(
        k, beta, g, cu_seqlens=cu_seqlens, output_dtype=out_dtype
    )
    a_inv = solve_tril(a_kkt, cu_seqlens=cu_seqlens, output_dtype=k.dtype)
    w, u = recompute_w_u_fwd(k, v, beta, g, a_inv, cu_seqlens)
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k, w=w, u=u, g=g, initial_state=initial_state,
        output_final_state=True, cu_seqlens=cu_seqlens,
    )
    o = chunk_fwd_o(q=q, k=k, v=v_new, h=h, g=g, scale=scale, cu_seqlens=cu_seqlens)
    return o, h, v_new, final_state


# --------------------------------------------------------------------------------------- comparison
def _rel_err(actual, expected):
    a, e = actual.to(torch.float64), expected.to(torch.float64)
    assert a.shape == e.shape, (tuple(a.shape), tuple(e.shape))
    return (a - e).abs().max().item() / max(e.abs().max().item(), 1.0)


def _assert_close(actual, expected, tol, what=""):
    err = _rel_err(actual, expected)
    assert err <= tol, f"{what}: normalized max error {err:.3e} exceeds {tol:.3e}"


def _keep_matrix(length, device=None):
    i = torch.arange(length, device=device)
    return i[:, None] >= i[None, :]


# ============================================================================ delta_h: structure
def test_delta_h_shapes_dtypes_and_independent_final_state():
    total, hk, hv, d = 2 * BT + 2, 2, 4, 8
    cu = torch.tensor([0, total], dtype=torch.int32)
    k, _, _, _ = _make_inputs(total, hk=hk, hv=hv, d=d, seed=40)
    w, u = _make_wu(total, hv, d, d, seed=41)
    _, g = _make_gate(total, hv, cu, seed=42)
    init = _make_state(1, hv, d, d, dtype=torch.bfloat16, seed=43)

    h, v_new, final = chunk_gated_delta_rule_fwd_h(
        k, w, u, g, init, True, cu_seqlens=cu
    )
    assert h.shape == (1, 3, hv, d, d)  # [B, NT, H, V, K]: V before K
    assert h.dtype is torch.bfloat16  # k.new_empty -> k.dtype [sglang chunk_delta_h.py:305]
    assert v_new.shape == (1, total, hv, d)
    assert v_new.dtype is torch.bfloat16
    assert final.shape == (1, hv, d, d)
    assert final.dtype is torch.bfloat16
    assert final.data_ptr() != init.data_ptr(), "final_state must not alias initial_state"
    assert torch.isfinite(h).all() and torch.isfinite(v_new).all() and torch.isfinite(final).all()

    # h[0] is the entry state of chunk 0 -> the (rounded) initial state
    torch.testing.assert_close(h[:, 0].float(), init.float(), rtol=2e-2, atol=2e-2)


def test_delta_h_final_state_none_when_not_requested():
    total, hv, d = BT, 4, 8
    cu = torch.tensor([0, total], dtype=torch.int32)
    k, _, _, _ = _make_inputs(total, seed=44)
    w, u = _make_wu(total, hv, d, d, seed=45)
    _, g = _make_gate(total, hv, cu, seed=46)
    init = _make_state(1, hv, d, d, dtype=torch.bfloat16, seed=47)

    h1, vn1, fs1 = chunk_gated_delta_rule_fwd_h(k, w, u, g, init, True, cu_seqlens=cu)
    h0, vn0, fs0 = chunk_gated_delta_rule_fwd_h(k, w, u, g, init, False, cu_seqlens=cu)
    assert fs0 is None
    assert torch.equal(h1, h0) and torch.equal(vn1, vn0)


def test_delta_h_does_not_mutate_initial_state():
    """vLLM returns a separate final_state instead of the sibling's in-place update [推断, §7.5]."""
    total, hv, d = BT, 4, 8
    cu = torch.tensor([0, total], dtype=torch.int32)
    k, _, _, _ = _make_inputs(total, seed=48)
    w, u = _make_wu(total, hv, d, d, seed=49)
    _, g = _make_gate(total, hv, cu, seed=50)
    init = _make_state(1, hv, d, d, dtype=torch.float32, seed=51)
    before = init.clone()
    _, _, final = chunk_gated_delta_rule_fwd_h(k, w, u, g, init, True, cu_seqlens=cu)
    assert torch.equal(init, before), "initial_state was written in place"
    assert not torch.equal(final, init), "the gated chunk update should have moved the state"


# ================================================================== delta_h: per-token recurrence
def test_delta_h_matches_per_token_recurrence_fp64():
    """T = 130 (two full chunks + a 2-token tail): h/v_new/final against the fp64 delta rule."""
    total, hk, hv, d = 2 * BT + 2, 2, 4, 8
    cu = torch.tensor([0, total], dtype=torch.int32)
    k, q, v, beta = _make_inputs(total, hk=hk, hv=hv, d=d, dtype=torch.float64, seed=52)
    inc, g = _make_gate(total, hv, cu, dtype=torch.float64, seed=53)
    init = _make_state(1, hv, d, d, dtype=torch.float64, seed=54)
    scale = d**-0.5

    a_kkt = chunk_scaled_dot_kkt_fwd(k, beta, g, cu_seqlens=cu, output_dtype=torch.float64)
    a_inv = solve_tril(a_kkt, cu_seqlens=cu, output_dtype=k.dtype)
    w, u = recompute_w_u_fwd(k, v, beta, g, a_inv, cu)
    h, v_new, final = chunk_gated_delta_rule_fwd_h(k, w, u, g, init, True, cu_seqlens=cu)

    _, ref_vnew, ref_h, ref_final = _per_token_delta_rule(
        q, k, v, beta, inc, init, scale, cu
    )
    _assert_close(h, ref_h, FP64_TOL, "h")
    _assert_close(v_new, ref_vnew, FP64_TOL, "v_new")
    _assert_close(final, ref_final, FP64_TOL, "final_state")


def test_delta_h_matches_structural_oracle_bf16():
    """Arbitrary (not WY-consistent) w/u, bf16: slot ordering, gate and shape only."""
    total, hk, hv, d = 2 * BT + 7, 2, 4, 8
    cu = torch.tensor([0, total], dtype=torch.int32)
    k, _, _, _ = _make_inputs(total, hk=hk, hv=hv, d=d, seed=55)
    w, u = _make_wu(total, hv, d, d, seed=56)
    _, g = _make_gate(total, hv, cu, seed=57)
    init = _make_state(1, hv, d, d, dtype=torch.bfloat16, seed=58)

    h, v_new, final = chunk_gated_delta_rule_fwd_h(k, w, u, g, init, True, cu_seqlens=cu)
    ref_h, ref_vnew, ref_final = _delta_h_structural_oracle(k, w, u, g, init, cu)
    _assert_close(h, ref_h, BF16_TOL, "h")
    _assert_close(v_new, ref_vnew, BF16_TOL, "v_new")
    _assert_close(final, ref_final, BF16_TOL, "final_state")


def test_delta_h_without_initial_state_fp64():
    total, hv, d = BT + 3, 4, 8
    cu = torch.tensor([0, total], dtype=torch.int32)
    k, _, _, _ = _make_inputs(total, dtype=torch.float64, seed=59)
    w, u = _make_wu(total, hv, d, d, dtype=torch.float64, seed=60)
    _, g = _make_gate(total, hv, cu, dtype=torch.float64, seed=61)

    h, v_new, final = chunk_gated_delta_rule_fwd_h(k, w, u, g, None, True, cu_seqlens=cu)
    ref_h, ref_vnew, ref_final = _delta_h_structural_oracle(k, w, u, g, None, cu)
    _assert_close(h, ref_h, 1e-12, "h")
    _assert_close(v_new, ref_vnew, 1e-12, "v_new")
    _assert_close(final, ref_final, 1e-12, "final_state")
    assert torch.count_nonzero(h[:, 0]) == 0  # every sequence starts from the zero state


def test_delta_h_varlen_tail_g_last_and_chunk_offsets():
    """cu = [0, 37, 130]: the 37-token chunk's ``g_last`` is g[36], not g[63]."""
    total, hk, hv, d = 130, 2, 4, 8
    cu = torch.tensor([0, 37, 130], dtype=torch.int64)
    k, q, v, beta = _make_inputs(total, hk=hk, hv=hv, d=d, dtype=torch.float64, seed=62)
    inc, g = _make_gate(total, hv, cu, dtype=torch.float64, seed=63)
    init = _make_state(2, hv, d, d, dtype=torch.float64, seed=64)
    scale = d**-0.5

    a_kkt = chunk_scaled_dot_kkt_fwd(k, beta, g, cu_seqlens=cu, output_dtype=torch.float64)
    a_inv = solve_tril(a_kkt, cu_seqlens=cu, output_dtype=k.dtype)
    w, u = recompute_w_u_fwd(k, v, beta, g, a_inv, cu)
    h, v_new, final = chunk_gated_delta_rule_fwd_h(k, w, u, g, init, True, cu_seqlens=cu)
    _, ref_vnew, ref_h, ref_final = _per_token_delta_rule(q, k, v, beta, inc, init, scale, cu)
    assert h.shape == (1, 3, hv, d, d)  # 1 chunk for seq 0, 2 for seq 1 (93 tokens)
    _assert_close(h, ref_h, FP64_TOL, "h")
    _assert_close(v_new, ref_vnew, FP64_TOL, "v_new")
    _assert_close(final, ref_final, FP64_TOL, "final_state")


def test_delta_h_chunk_offsets_validated_and_chunk_indices_ignored():
    total, hv, d = 130, 4, 8
    cu = torch.tensor([0, 37, 130], dtype=torch.int32)
    k, _, _, _ = _make_inputs(total, seed=65)
    w, u = _make_wu(total, hv, d, d, seed=66)
    _, g = _make_gate(total, hv, cu, seed=67)
    init = _make_state(2, hv, d, d, dtype=torch.bfloat16, seed=68)

    base = chunk_gated_delta_rule_fwd_h(k, w, u, g, init, True, cu_seqlens=cu)
    good = chunk_gated_delta_rule_fwd_h(
        k, w, u, g, init, True, cu_seqlens=cu, chunk_offsets=torch.tensor([0, 1, 3], dtype=torch.int32)
    )
    assert torch.equal(base[0], good[0]) and torch.equal(base[1], good[1])
    # chunk_indices is accepted and ignored, like the kernel [sglang chunk_delta_h.py:61-67]
    ci = prepare_chunk_indices(cu, BT)
    flipped = chunk_gated_delta_rule_fwd_h(
        k, w, u, g, init, True, cu_seqlens=cu, chunk_indices=ci.flip(0)
    )
    assert torch.equal(base[0], flipped[0]) and torch.equal(base[1], flipped[1])
    with pytest.raises(ValueError, match="cumulative chunk counts"):
        chunk_gated_delta_rule_fwd_h(
            k, w, u, g, init, True, cu_seqlens=cu,
            chunk_offsets=torch.tensor([0, 2, 3], dtype=torch.int32),
        )


def test_delta_h_varlen_two_short_sequences_independent():
    """cu = [0, 37, 64]: each sequence behaves like a dense call on its own slice."""
    total, hv, d = BT, 4, 8
    cu = torch.tensor([0, 37, 64], dtype=torch.int32)
    k, _, _, _ = _make_inputs(total, seed=69)
    w, u = _make_wu(total, hv, d, d, seed=70)
    _, g = _make_gate(total, hv, cu, seed=71)
    init = _make_state(2, hv, d, d, dtype=torch.bfloat16, seed=72)

    h, v_new, final = chunk_gated_delta_rule_fwd_h(k, w, u, g, init, True, cu_seqlens=cu)
    ref_h, ref_vnew, ref_final = _delta_h_structural_oracle(k, w, u, g, init, cu)
    _assert_close(h, ref_h, BF16_TOL, "h")
    _assert_close(v_new, ref_vnew, BF16_TOL, "v_new")
    _assert_close(final, ref_final, BF16_TOL, "final_state")

    # second sequence alone == its slice of the packed result
    h2, vn2, fs2 = chunk_gated_delta_rule_fwd_h(
        k[:, 37:].contiguous(), w[:, 37:].contiguous(), u[:, 37:].contiguous(),
        g[:, 37:].contiguous(), init[1:2].contiguous(), True,
        cu_seqlens=torch.tensor([0, 27], dtype=torch.int32),
    )
    assert torch.equal(h2[0, 0], h[0, 1])
    assert torch.equal(vn2, v_new[:, 37:])
    assert torch.equal(fs2[0], final[1])


def test_delta_h_uncovered_trailing_rows_stay_zero():
    total, cu = 16, torch.tensor([0, 5], dtype=torch.int32)
    k, _, _, _ = _make_inputs(total, seed=73)
    w, u = _make_wu(total, 4, 8, 8, seed=74)
    _, g = _make_gate(total, 4, cu, seed=75)
    init = _make_state(1, 4, 8, 8, seed=76)
    h, v_new, final = chunk_gated_delta_rule_fwd_h(k, w, u, g, init, True, cu_seqlens=cu)
    assert h.shape == (1, 1, 4, 8, 8)
    assert torch.count_nonzero(v_new[0, 5:]) == 0
    assert final.shape == (1, 4, 8, 8)


def test_delta_h_empty_sequence_segment_keeps_its_entry_state():
    """cu = [0, 0, 5]: the empty sequence owns no chunk but still gets its final state back."""
    total, cu = 5, torch.tensor([0, 0, 5], dtype=torch.int32)
    k, _, _, _ = _make_inputs(total, seed=201)
    w, u = _make_wu(total, 4, 8, 8, seed=202)
    _, g = _make_gate(total, 4, cu, seed=203)
    init = _make_state(2, 4, 8, 8, seed=204)

    h, v_new, final = chunk_gated_delta_rule_fwd_h(k, w, u, g, init, True, cu_seqlens=cu)
    assert h.shape == (1, 1, 4, 8, 8)  # only the 5-token sequence has a chunk
    ref_h, ref_vnew, ref_final = _delta_h_structural_oracle(k, w, u, g, init, cu)
    _assert_close(h, ref_h, BF16_TOL, "h")
    _assert_close(v_new, ref_vnew, BF16_TOL, "v_new")
    _assert_close(final, ref_final, BF16_TOL, "final_state")
    # the empty sequence's state is untouched (matching the caller's _compact_empty_segments path)
    torch.testing.assert_close(final[0], init[0], rtol=0, atol=0)


# ==================================================================== delta_h: bf16 round-trips
def test_delta_h_entry_state_snapshot_is_rounded_to_k_dtype():
    """``h[c]`` is stored through ``k.new_empty``: the fp32 accumulator is rounded per chunk."""
    total, hv, d = 1, 1, 1
    cu = torch.tensor([0, total], dtype=torch.int32)
    k = torch.ones(1, total, 1, 1, dtype=torch.bfloat16)
    w = torch.zeros_like(k)
    u = torch.zeros(1, total, hv, d, dtype=torch.bfloat16)
    g = torch.zeros(1, total, hv, dtype=torch.float32)
    init = torch.full((1, hv, d, d), 1.0 + 2.0**-12, dtype=torch.float32)

    h, _, final = chunk_gated_delta_rule_fwd_h(k, w, u, g, init, True, cu_seqlens=cu)
    assert h.dtype is torch.bfloat16
    assert h[0, 0, 0, 0, 0].item() == 1.0, "h snapshot was not rounded to bf16"
    # the accumulator itself is NOT rounded: the final state is written in initial_state's dtype
    assert final.dtype is torch.float32 and final[0, 0, 0, 0].item() == pytest.approx(1.0 + 2.0**-12)


def test_delta_h_rounds_state_before_the_prediction_dot():
    """``dot(b_w, trans(b_h1).to(b_w.dtype))`` [sglang chunk_delta_h.py:148]: amplification probe.

    ``w=u=1000``, ``k=1``, entry state ``1 + 2^-12`` (not bf16-representable).  If the prediction
    dot sees the bf16-rounded state the prediction is exactly 1000 and ``v_new == 0``; if it sees
    the fp32 state, ``v_new ~ -0.244``.
    """
    k = torch.ones(1, 1, 1, 1, dtype=torch.bfloat16)
    w = torch.full((1, 1, 1, 1), 1000.0, dtype=torch.bfloat16)
    u = torch.full((1, 1, 1, 1), 1000.0, dtype=torch.bfloat16)
    g = torch.zeros(1, 1, 1, dtype=torch.float32)
    init = torch.full((1, 1, 1, 1), 1.0 + 2.0**-12, dtype=torch.float32)
    cu = torch.tensor([0, 1], dtype=torch.int32)

    _, v_new, final = chunk_gated_delta_rule_fwd_h(k, w, u, g, init, True, cu_seqlens=cu)
    assert v_new[0, 0, 0, 0].item() == 0.0, "the state was not rounded to w.dtype before the dot"
    assert final[0, 0, 0, 0].item() == pytest.approx(1.0 + 2.0**-12)
    # what the unrounded variant would have produced, to show the probe separates the two
    wrong = (1.0 + 2.0**-12) * 1000.0 - 1000.0
    assert abs(wrong) > 1e-2


def test_delta_h_safe_exp_keeps_misordered_gate_finite():
    """``safe_exp`` on ``g_last - g`` clamps positive differences even on a non-monotone gate.

    Within each chunk ``g_t = -5000 + 30 * (t - t0)`` increases, so ``g_last - g_t`` is +1890 for
    the first token: plain ``exp`` overflows even fp64, while the one-sided clamp makes it exactly
    0 [sglang chunk_delta_h.py:185, op.py:25-27].  The chunk-end decay ``exp(-3110)`` underflows to
    0, keeping every other quantity finite.
    """
    total, hv, d = 3 * BT, 2, 4
    cu = torch.tensor([0, total], dtype=torch.int32)
    k, _, _, _ = _make_inputs(total, hv=hv, d=d, dtype=torch.float64, seed=77)
    w, u = _make_wu(total, hv, d, d, dtype=torch.float64, seed=78)
    g = torch.empty(1, total, hv, dtype=torch.float64)
    for t0, length in _dense_chunks(total):
        steps = torch.arange(length, dtype=torch.float64) * 30.0 - 5000.0
        g[0, t0 : t0 + length] = steps[:, None]
    init = _make_state(1, hv, d, d, dtype=torch.float64, seed=80)

    h, v_new, final = chunk_gated_delta_rule_fwd_h(k, w, u, g, init, True, cu_seqlens=cu)
    assert torch.isfinite(h).all() and torch.isfinite(v_new).all() and torch.isfinite(final).all()
    ref_h, ref_vnew, ref_final = _delta_h_structural_oracle(k, w, u, g, init, cu)
    _assert_close(h, ref_h, 1e-10, "h")
    _assert_close(v_new, ref_vnew, 1e-10, "v_new")
    _assert_close(final, ref_final, 1e-10, "final_state")
    # the raw difference really is in the overflow regime
    diff = g[0, BT - 1].min() - g[0, 0].max()
    assert float(diff) > 700.0


# ============================================================================== delta_h: malformed
def test_delta_h_rejects_malformed_input():
    k = torch.zeros(1, 8, 1, 4)
    w = torch.zeros(1, 8, 4, 4)
    u = torch.zeros(1, 8, 4, 4)
    g = torch.zeros(1, 8, 4)
    init = torch.zeros(1, 4, 4, 4)
    with pytest.raises(ValueError, match=r"k must be \[B, T, Hg, K\]"):
        chunk_gated_delta_rule_fwd_h(k[0], w, u, g, init, True)
    with pytest.raises(ValueError, match="w must be"):
        chunk_gated_delta_rule_fwd_h(k, w[:, :7], u, g, init, True)
    with pytest.raises(ValueError, match="u must have"):
        chunk_gated_delta_rule_fwd_h(k, w, u[:, :7], g, init, True)
    with pytest.raises(ValueError, match="g must be"):
        chunk_gated_delta_rule_fwd_h(k, w, u, g[:, :7], init, True)
    with pytest.raises(ValueError, match="batch size is expected to be 1"):
        chunk_gated_delta_rule_fwd_h(
            k.repeat(2, 1, 1, 1), w.repeat(2, 1, 1, 1), u.repeat(2, 1, 1, 1), g.repeat(2, 1, 1),
            init.repeat(2, 1, 1, 1), True, cu_seqlens=torch.tensor([0, 8]),
        )
    with pytest.raises(ValueError, match="initial_state must be"):
        chunk_gated_delta_rule_fwd_h(k, w, u, g, init[:, :, :3], True)
    with pytest.raises(ValueError, match="positive multiple"):
        chunk_gated_delta_rule_fwd_h(
            torch.zeros(1, 8, 3, 4), torch.zeros(1, 8, 4, 4), u, g, init, True
        )
    with pytest.raises(ValueError, match="start at 0"):
        chunk_gated_delta_rule_fwd_h(k, w, u, g, init, True, cu_seqlens=torch.tensor([1, 8]))


# =============================================================================== chunk_o: closed form
@pytest.mark.parametrize("total", [1, 63, 64])
def test_chunk_o_single_chunk_closed_form(total):
    """T <= 64 with h = 0: ``o_i = scale * sum_{j<=i} (q_i.k_j) exp(g_i-g_j) v_j`` in fp64.

    Validates the ``>=`` mask (the diagonal term is exactly ``scale * (q_i.k_i) * v_i``), the
    ``safe_exp`` clamp and the double ``scale`` at once (contract §6.4).
    """
    hk, hv, d = 2, 4, 8
    cu = torch.tensor([0, total], dtype=torch.int32)
    k, q, v, _ = _make_inputs(total, hk=hk, hv=hv, d=d, dtype=torch.float32, seed=81)
    _, g = _make_gate(total, hv, cu, seed=82)
    h = torch.zeros(_h_shape(1, total, cu, hv, d, d), dtype=torch.float32)
    scale = 0.37  # NOT the default, so a missing "both terms scaled" shows up

    o = chunk_fwd_o(q, k, v, h, g, scale, cu_seqlens=cu)
    assert o.shape == (1, total, hv, d) and o.dtype is torch.float32
    ref = _chunk_o_oracle(q, k, v, h, g, scale, cu)
    _assert_close(o, ref, FP32_TOL, "o")

    if total == 1:
        # the self term is the whole answer for T=1: mask `>` would return 0 here
        hm = _gqa(hv, hk)
        dot = torch.einsum("hk,hk->h", q[0, 0][hm], k[0, 0][hm])
        expected = scale * dot[:, None] * v[0, 0]
        torch.testing.assert_close(o[0, 0], expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("total", [65, 128])
def test_chunk_o_multi_chunk_uses_the_entry_states(total):
    """Two/five chunks with a non-zero initial state: the inter-chunk term must use ``h[c]``."""
    hk, hv, d = 2, 4, 8
    cu = torch.tensor([0, total], dtype=torch.int32)
    k, q, v, _ = _make_inputs(total, hk=hk, hv=hv, d=d, dtype=torch.float32, seed=83)
    _, g = _make_gate(total, hv, cu, seed=84)
    init = _make_state(1, hv, d, d, dtype=torch.float32, seed=85)
    w, u = _make_wu(total, hv, d, d, dtype=torch.float32, seed=86)
    h, _, _ = chunk_gated_delta_rule_fwd_h(k, w, u, g, init, True, cu_seqlens=cu)
    scale = d**-0.5

    o = chunk_fwd_o(q, k, v, h, g, scale, cu_seqlens=cu)
    _assert_close(o, _chunk_o_oracle(q, k, v, h, g, scale, cu), FP32_TOL, "o")
    # chunk 0 of the output must not depend on the entry state of chunk 1
    h2 = h.clone()
    h2[0, 1] = 7.0
    o2 = chunk_fwd_o(q, k, v, h2, g, scale, cu_seqlens=cu)
    torch.testing.assert_close(o2[0, :BT], o[0, :BT], rtol=0, atol=0)


def test_chunk_o_varlen_two_sequences_with_nonzero_states():
    """cu = [0, 37, 64]: sequence 2 reads h slot 1 (its own entry state), not slot 0."""
    total, hk, hv, d = BT, 2, 4, 8
    cu = torch.tensor([0, 37, 64], dtype=torch.int32)
    k, q, v, _ = _make_inputs(total, hk=hk, hv=hv, d=d, dtype=torch.float32, seed=87)
    _, g = _make_gate(total, hv, cu, seed=88)
    init = _make_state(2, hv, d, d, dtype=torch.float32, seed=89)
    w, u = _make_wu(total, hv, d, d, dtype=torch.float32, seed=90)
    h, _, _ = chunk_gated_delta_rule_fwd_h(k, w, u, g, init, True, cu_seqlens=cu)
    assert h.shape == (1, 2, hv, d, d)
    scale = d**-0.5
    o = chunk_fwd_o(q, k, v, h, g, scale, cu_seqlens=cu)
    _assert_close(o, _chunk_o_oracle(q, k, v, h, g, scale, cu), FP32_TOL, "o")
    # each sequence's output equals the same call on its own slice (its own h slot)
    o2 = chunk_fwd_o(
        q[:, 37:].contiguous(), k[:, 37:].contiguous(), v[:, 37:].contiguous(),
        h[:, 1:2].contiguous(), g[:, 37:].contiguous(), scale,
        cu_seqlens=torch.tensor([0, 27], dtype=torch.int32),
    )
    torch.testing.assert_close(o2, o[:, 37:], rtol=1e-6, atol=1e-6)


def test_chunk_o_scale_multiplies_both_terms():
    total, hk, hv, d = BT + 5, 2, 4, 8
    cu = torch.tensor([0, total], dtype=torch.int32)
    k, q, v, _ = _make_inputs(total, hk=hk, hv=hv, d=d, dtype=torch.float64, seed=91)
    _, g = _make_gate(total, hv, cu, dtype=torch.float64, seed=92)
    init = _make_state(1, hv, d, d, dtype=torch.float64, seed=93)
    w, u = _make_wu(total, hv, d, d, dtype=torch.float64, seed=94)
    h, _, _ = chunk_gated_delta_rule_fwd_h(k, w, u, g, init, True, cu_seqlens=cu)

    o1 = chunk_fwd_o(q, k, v, h, g, 0.25, cu_seqlens=cu)
    o2 = chunk_fwd_o(q, k, v, h, g, 0.5, cu_seqlens=cu)
    torch.testing.assert_close(o2, 2.0 * o1, rtol=1e-12, atol=0)
    torch.testing.assert_close(o1, _chunk_o_oracle(q, k, v, h, g, 0.25, cu), rtol=1e-12, atol=0)


def test_chunk_o_huge_decay_stays_finite_and_matches_the_masked_definition():
    """Decay of 40/token puts ``exp(+800)`` in the upper triangle; only clamp/where keep it finite."""
    total, hv, d = BT, 4, 8
    cu = torch.tensor([0, total], dtype=torch.int32)
    k, q, v, _ = _make_inputs(total, seed=95)
    _, g = _make_gate(total, hv, cu, decay=40.0, seed=96)
    gap = float((g[0, :, 0][:, None] - g[0, :, 0][None, :]).max())
    assert gap > 800.0
    h = torch.zeros(_h_shape(1, total, cu, hv, d, d), dtype=k.dtype)
    o = chunk_fwd_o(q, k, v, h, g, 0.11, cu_seqlens=cu)
    assert torch.isfinite(o).all()
    _assert_close(o, _chunk_o_oracle(q, k, v, h, g, 0.11, cu), 5e-2, "o")

    # negative control: the plausible "multiply by a 0/1 triangle" form is NaN on these inputs
    qb = q[0].float()
    kb = k[0].float()
    qk = torch.einsum("thk,shk->tsh", qb[:, _gqa(hv, 2)], kb[:, _gqa(hv, 2)])
    gd = g[0][:, None, :] - g[0][None, :, :]
    wrong = qk * torch.exp(gd)
    wrong = wrong * _keep_matrix(total).unsqueeze(-1)
    assert torch.isnan(wrong).any(), "the multiply-mask variant was expected to produce NaN"


def test_chunk_o_gqa_head_mapping():
    total, hk, hv, d = BT, 1, 4, 8
    cu = torch.tensor([0, total], dtype=torch.int32)
    k, q, v, _ = _make_inputs(total, hk=hk, hv=hv, d=d, dtype=torch.float32, seed=97)
    _, g = _make_gate(total, hv, cu, seed=98)
    # one value head's worth of gate for all heads, identical values: all heads must agree
    g = g[:, :, :1].expand(1, total, hv)
    v_equal = v[:, :, :1].expand(1, total, hv, d)
    h = torch.zeros(_h_shape(1, total, cu, hv, d, d), dtype=torch.float32)
    o = chunk_fwd_o(q, k, v_equal, h, g, 0.2, cu_seqlens=cu)
    for hh in range(1, hv):
        assert torch.equal(o[:, :, 0], o[:, :, hh])
    _assert_close(o, _chunk_o_oracle(q, k, v_equal, h, g, 0.2, cu), FP32_TOL, "o")


def test_chunk_o_g_none_equals_zero_gate():
    total, hv, d = BT + 9, 4, 8
    cu = torch.tensor([0, total], dtype=torch.int32)
    k, q, v, _ = _make_inputs(total, seed=99)
    _, g = _make_gate(total, hv, cu, seed=100)
    h = torch.zeros(_h_shape(1, total, cu, hv, d, d), dtype=k.dtype)
    ungated = chunk_fwd_o(q, k, v, h, None, 0.3, cu_seqlens=cu)
    zero = chunk_fwd_o(q, k, v, h, torch.zeros_like(g), 0.3, cu_seqlens=cu)
    assert torch.equal(ungated, zero)
    _assert_close(ungated, _chunk_o_oracle(q, k, v, h, None, 0.3, cu), BF16_TOL, "o")


# ================================================================================= chunk_o: plumbing
def test_chunk_o_core_attn_out_is_written_in_place():
    total, hv, d = BT + 3, 4, 8
    cu = torch.tensor([0, total], dtype=torch.int32)
    k, q, v, _ = _make_inputs(total, seed=101)
    _, g = _make_gate(total, hv, cu, seed=102)
    h = torch.zeros(_h_shape(1, total, cu, hv, d, d), dtype=k.dtype)
    expected = chunk_fwd_o(q, k, v, h, g, 0.4, cu_seqlens=cu)

    buf = torch.full_like(expected, float("nan"))
    got = chunk_fwd_o(q, k, v, h, g, 0.4, cu_seqlens=cu, core_attn_out=buf)
    assert got is buf
    assert torch.equal(buf, expected)
    with pytest.raises(ValueError, match="core_attn_out must have"):
        chunk_fwd_o(q, k, v, h, g, 0.4, cu_seqlens=cu, core_attn_out=buf[:, : total - 1])


def test_chunk_o_varlen_uncovered_trailing_rows_zero():
    total, cu = 16, torch.tensor([0, 5], dtype=torch.int32)
    k, q, v, _ = _make_inputs(total, seed=103)
    _, g = _make_gate(total, 4, cu, seed=104)
    h = torch.zeros(_h_shape(1, total, cu, 4, 8, 8), dtype=k.dtype)
    o = chunk_fwd_o(q, k, v, h, g, 0.2, cu_seqlens=cu)
    assert torch.count_nonzero(o[0, 5:]) == 0
    _assert_close(o, _chunk_o_oracle(q, k, v, h, g, 0.2, cu), BF16_TOL, "o")


def test_chunk_o_non_contiguous_inputs():
    total, hv, d = BT + 11, 4, 8
    cu = torch.tensor([0, total], dtype=torch.int32)
    k, q, v, _ = _make_inputs(total, seed=105)
    _, g = _make_gate(total, hv, cu, seed=106)
    h = torch.zeros(_h_shape(1, total, cu, hv, d, d), dtype=k.dtype)
    expected = chunk_fwd_o(q, k, v, h, g, 0.3, cu_seqlens=cu)

    qw = torch.zeros(1, total, 2, 2 * d, dtype=q.dtype)
    qw[..., :d] = q
    kw = torch.zeros(1, total, 2, 2 * d, dtype=k.dtype)
    kw[..., :d] = k
    vw = torch.zeros(1, total, hv, 2 * d, dtype=v.dtype)
    vw[..., :d] = v
    gw = torch.zeros(1, total, 2 * hv, dtype=g.dtype)
    gw[..., :hv] = g
    nt = _h_shape(1, total, cu, hv, d, d)[1]
    hw = torch.zeros(1, nt, hv, d, 2 * d, dtype=h.dtype)
    hw[..., :d] = h
    qs, ks, vs, gs, hs = qw[..., :d], kw[..., :d], vw[..., :d], gw[..., :hv], hw[..., :d]
    assert not qs.is_contiguous() and not vs.is_contiguous() and not hs.is_contiguous()
    assert torch.equal(chunk_fwd_o(qs, ks, vs, hs, gs, 0.3, cu_seqlens=cu), expected)


def test_chunk_o_chunk_indices_derived_vs_explicit_and_ignored_when_dense():
    total, hv, d = 130, 4, 8
    cu = torch.tensor([0, 37, 130], dtype=torch.int32)
    k, q, v, _ = _make_inputs(total, seed=107)
    _, g = _make_gate(total, hv, cu, seed=108)
    init = _make_state(2, hv, d, d, dtype=k.dtype, seed=109)
    w, u = _make_wu(total, hv, d, d, seed=110)
    h, _, _ = chunk_gated_delta_rule_fwd_h(k, w, u, g, init, True, cu_seqlens=cu)

    auto = chunk_fwd_o(q, k, v, h, g, 0.25, cu_seqlens=cu)
    ci = prepare_chunk_indices(cu, BT)
    explicit = chunk_fwd_o(q, k, v, h, g, 0.25, cu_seqlens=cu, chunk_indices=ci)
    flipped = chunk_fwd_o(q, k, v, h, g, 0.25, cu_seqlens=cu, chunk_indices=ci.flip(0))
    assert torch.equal(auto, explicit) and torch.equal(auto, flipped)

    dense_h = torch.zeros(_h_shape(1, total, None, hv, d, d), dtype=k.dtype)
    dense = chunk_fwd_o(q, k, v, dense_h, g, 0.25, cu_seqlens=None)
    bogus = torch.zeros((3, 2), dtype=torch.int64)
    stray = chunk_fwd_o(q, k, v, dense_h, g, 0.25, cu_seqlens=None, chunk_indices=bogus)
    assert torch.equal(dense, stray)


def test_chunk_o_rejects_malformed_input():
    q = torch.zeros(1, 8, 1, 4)
    k = torch.zeros(1, 8, 1, 4)
    v = torch.zeros(1, 8, 4, 4)
    h = torch.zeros(1, 1, 4, 4, 4)
    g = torch.zeros(1, 8, 4)
    with pytest.raises(ValueError, match="v must be"):
        chunk_fwd_o(q, k, v[0], h, g, 1.0)
    with pytest.raises(ValueError, match="k must match q"):
        chunk_fwd_o(q, k[:, :7], v, h, g, 1.0)
    with pytest.raises(ValueError, match="h must be"):
        chunk_fwd_o(q, k, v, h[:, :, :, :3], g, 1.0)
    with pytest.raises(ValueError, match="g must be"):
        chunk_fwd_o(q, k, v, h, g[:, :7], 1.0)
    with pytest.raises(ValueError, match="batch size is expected to be 1"):
        chunk_fwd_o(q.repeat(2, 1, 1, 1), k.repeat(2, 1, 1, 1), v.repeat(2, 1, 1, 1),
                    h.repeat(2, 1, 1, 1, 1), g.repeat(2, 1, 1), 1.0,
                    cu_seqlens=torch.tensor([0, 8]))
    with pytest.raises(ValueError, match="positive multiple"):
        chunk_fwd_o(torch.zeros(1, 8, 3, 4), torch.zeros(1, 8, 3, 4), v, h, g, 1.0)


# =============================================================================== the interface
def _interface_case(total, cu, hk=2, hv=4, d=8, dtype=torch.float64, seed=120):
    """Full chain inputs plus the fp64 oracle, for a dense (cu=None) or ragged layout."""
    k, q, v, beta = _make_inputs(total, hk=hk, hv=hv, d=d, dtype=dtype, seed=seed)
    inc, g = _make_gate(total, hv, cu, dtype=dtype, seed=seed + 1)
    n_seq = 1 if cu is None else len(cu) - 1
    init = _make_state(n_seq, hv, d, d, dtype=dtype, seed=seed + 2)
    scale = d**-0.5
    return q, k, v, beta, inc, g, init, scale


@pytest.mark.parametrize("total", [64, 130])
def test_interface_composed_matches_per_token_delta_rule_fp64(total):
    """op1's v_new as op2's v, fp64: o/h/v_new/final must equal the full per-token delta rule."""
    cu = torch.tensor([0, total], dtype=torch.int32)
    q, k, v, beta, inc, g, init, scale = _interface_case(total, cu, seed=130)
    o, h, v_new, final = _chain(q, k, v, beta, g, init, scale, cu)
    ref_o, ref_vnew, ref_h, ref_final = _per_token_delta_rule(
        q, k, v, beta, inc, init, scale, cu
    )
    _assert_close(h, ref_h, FP64_TOL, "h")
    _assert_close(v_new, ref_vnew, FP64_TOL, "v_new")
    _assert_close(final, ref_final, FP64_TOL, "final_state")
    _assert_close(o, ref_o, FP64_TOL, "o")


def test_interface_composed_matches_per_token_delta_rule_bf16():
    """Same as above with the production dtypes (k/q/v/beta bf16, g fp32)."""
    total = 130
    cu = torch.tensor([0, total], dtype=torch.int32)
    q, k, v, beta, inc, g, init, scale = _interface_case(total, cu, dtype=torch.bfloat16, seed=140)
    o, h, v_new, final = _chain(q, k, v, beta, g, init, scale, cu)
    ref_o, ref_vnew, ref_h, ref_final = _per_token_delta_rule(
        q, k, v, beta, inc, init, scale, cu
    )
    assert o.dtype is torch.bfloat16 and h.dtype is torch.bfloat16
    _assert_close(v_new, ref_vnew, BF16_TOL, "v_new")
    _assert_close(final, ref_final, BF16_TOL, "final_state")
    _assert_close(o, ref_o, BF16_TOL, "o")


def test_interface_varlen_matches_per_token_delta_rule_bf16():
    """cu = [0, 37, 130]: a one-chunk tail sequence and a two-chunk sequence, composed."""
    total = 130
    cu = torch.tensor([0, 37, 130], dtype=torch.int32)
    q, k, v, beta, inc, g, init, scale = _interface_case(total, cu, dtype=torch.bfloat16, seed=150)
    o, h, v_new, final = _chain(q, k, v, beta, g, init, scale, cu)
    ref_o, ref_vnew, ref_h, ref_final = _per_token_delta_rule(
        q, k, v, beta, inc, init, scale, cu
    )
    assert h.shape == (1, 3, 4, 8, 8)
    _assert_close(v_new, ref_vnew, BF16_TOL, "v_new")
    _assert_close(o, ref_o, BF16_TOL, "o")
    _assert_close(final, ref_final, BF16_TOL, "final_state")


def test_interface_dense_batch_two_matches_per_token_delta_rule_fp64():
    """The dense (cu_seqlens=None) B = 2 path: h slots are ``i_b * NT + i_t``, not a flat ordinal."""
    total, batch = BT + 6, 2
    k, q, v, beta = _make_inputs(total, batch=batch, dtype=torch.float64, seed=180)
    inc, g = _make_gate(total, 4, None, batch=batch, dtype=torch.float64, seed=181)
    init = _make_state(batch, 4, 8, 8, dtype=torch.float64, seed=182)
    scale = 8**-0.5
    o, h, v_new, final = _chain(q, k, v, beta, g, init, scale, None)
    ref_o, ref_vnew, ref_h, ref_final = _per_token_delta_rule(
        q, k, v, beta, inc, init, scale, None
    )
    assert h.shape == (batch, 2, 4, 8, 8)
    _assert_close(h, ref_h, FP64_TOL, "h")
    _assert_close(v_new, ref_vnew, FP64_TOL, "v_new")
    _assert_close(final, ref_final, FP64_TOL, "final_state")
    _assert_close(o, ref_o, FP64_TOL, "o")


def test_negative_control_original_v_in_chunk_o_would_fail():
    """The interface's ``v`` is ``v_new``: feeding the original ``v`` is a large, silent error."""
    total, cu = 130, torch.tensor([0, 130], dtype=torch.int32)
    q, k, v, beta, inc, g, init, scale = _interface_case(total, cu, dtype=torch.float64, seed=160)
    o, h, v_new, _ = _chain(q, k, v, beta, g, init, scale, cu)
    ref_o, _, _, _ = _per_token_delta_rule(q, k, v, beta, inc, init, scale, cu)
    wrong = chunk_fwd_o(q, k, v, h, g, scale, cu_seqlens=cu)  # original v, not v_new
    correct_err = _rel_err(o, ref_o)
    wrong_err = _rel_err(wrong, ref_o)
    assert correct_err <= FP64_TOL
    assert wrong_err > 1e-2 and wrong_err > 100 * correct_err
    # shapes are identical - the failure is purely numerical, which is the trap
    assert wrong.shape == o.shape


def test_negative_control_strict_triangle_would_fail():
    """A ``>`` mask drops the diagonal self term of ``chunk_fwd_o``: the tests must reject it."""
    total, hk, hv, d = BT, 2, 4, 8
    cu = torch.tensor([0, total], dtype=torch.int32)
    k, q, v, _ = _make_inputs(total, hk=hk, hv=hv, d=d, dtype=torch.float32, seed=170)
    _, g = _make_gate(total, hv, cu, seed=171)
    h = torch.zeros(_h_shape(1, total, cu, hv, d, d), dtype=torch.float32)
    o = chunk_fwd_o(q, k, v, h, g, 0.5, cu_seqlens=cu)
    ref = _chunk_o_oracle(q, k, v, h, g, 0.5, cu)
    _assert_close(o, ref, FP32_TOL, "o")

    strict = ref.clone()
    strict[0] -= 0.5 * torch.einsum(
        "thk,thk->th", q[0][:, _gqa(hv, hk)], k[0][:, _gqa(hv, hk)]
    )[:, :, None] * v[0]  # remove the j == i term exactly
    assert _rel_err(o, strict) > 1e-2, "a strict `>` triangle would have passed the oracle"
