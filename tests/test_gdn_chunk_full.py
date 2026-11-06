"""Full-chain tests for the composed GDN prefill entry point.

`tileinfer/vllm/ops/fla/chunk.py` wires the six already-ported operators into the exact sequence of
[vLLM] `chunk.py:37-86`:

    chunk_local_cumsum -> kkt -> solve_tril -> recompute_w_u_fwd -> delta_h -> chunk_fwd_o

These tests pin the *composition*, not the individual operators (each has its own suite):

1. **The chain against a full oracle.**  For every layout (B=1 dense, B=2 dense, varlen
   `cu_seqlens=[0, 37, 130]`) and for every tail length (`T in {1, 63, 64, 65, 130}`) the composed
   output must equal, in fp64, the per-token gated delta rule that the chain is supposed to
   implement (`S_t = exp(d_t) S_{t-1} + beta_t (v_t - exp(d_t) S_{t-1} k_t) k_t^T`, output
   `scale * q_t S_t`), and the chain's `w`/`u` must equal the independent fp64 reference in
   `tileinfer/testing/gdn_reference.py`.

   What `gdn_reference.py` actually contains is the **WY half** only (`kkt_reference` +
   `chunk_wu_reference`, which returns `w`, `u` and `(I+A)^{-1}`).  It has no `delta_h`/`chunk_o`
   reference, so the state/output half is checked against the per-token recurrence - which is
   exactly the oracle contract §6.3 prescribes for `chunk_delta_h` ("没有现成的 ... 用逐 token 的
   delta-rule 递推在 fp64 里跑").  The two halves meet at `v_new`/`h`, which are compared directly.

2. **Causal masking, end-to-end.**  Token `t`'s output must not depend on tokens `> t`.  This is
   the cheapest test that catches a missing mask anywhere in the chain (a `kkt` that reaches across
   tokens, a `solve_tril` that solves over the whole sequence, a `chunk_o` without its triangle).
   It is enforced bitwise, because every operation on a row depends only on rows at or before it.

3. **The `v` vs `v_new` boundary.**  `chunk_fwd_o` must receive `v_new`, not the caller's `v`
   [vLLM] `chunk.py:75`.  The negative control feeds the original `v` and shows an O(1) error while
   the correct chain is at fp64 rounding; because the composed output is also compared to the
   oracle at 1e-12, a wiring mistake would fail the suite rather than just the control.

4. **`v_new` is stored ungated.**  `delta_h` saves `u - w @ h[c]^T` *before* the state gate
   [sglang] `chunk_delta_h.py:172-176`; `chunk_o` applies `exp(g_i - g_j)` itself [sglang]
   `chunk_o.py:106`.  A gated `v_new` would be decayed twice: the test reconstructs the ungated
   value from the chain's own `w`/`h`/`u`, then shows the gated variant differs by O(1) and that
   composing `chunk_o` with it is O(1) wrong.  (This is the numerical evidence behind the corrected
   contract §6.3 item 2.)

5. **Dtype flow and the public wrapper's guards**: `g` fp32, `A` rebound to `k.dtype`, `w`/`h`
   `k.dtype`, `v_new`/`o` `v.dtype`, `o.to(q.dtype)`; `scale` default from `k.shape[-1]`;
   `output_final_state=False -> None`; the l2norm option as a rebinding (never an in-place write);
   the varlen batch/state-count errors; and the `A.triu() == 0` invariant, which is block-local
   (per chunk, per sequence) and must not be a global `triu`.

Everything is CPU-only and deterministic (seeded generators); fp64 normally measures ~4e-16 and
bf16 ~4e-3 against the oracle.
"""

from __future__ import annotations

import pytest
import torch

from tileinfer.testing.gdn_reference import chunk_wu_reference
from tileinfer.vllm.ops.fla.chunk import (
    FLA_CHUNK_SIZE,
    assert_strict_lower_chunks,
    chunk_gated_delta_rule,
    chunk_gated_delta_rule_fwd,
)
from tileinfer.vllm.ops.fla.chunk_o import chunk_fwd_o
from tileinfer.vllm.ops.fla.kkt import chunk_scaled_dot_kkt_fwd
from tileinfer.vllm.ops.fla.wu import recompute_w_u_fwd
from tileinfer.vllm.ops.gdn import l2norm_fwd

BT = FLA_CHUNK_SIZE  # 64, pinned by [vLLM] qwen_gdn_linear_attn.py:1085-1087
FP64_TOL = 1e-12  # measured ~4e-16 for the whole chain vs the per-token recurrence
BF16_TOL = 2e-2  # bf16 mantissa + the documented round-trips (A_inv, h, v_new, state-dot)
O1_ERR = 1e-2  # "the negative control is broken by an O(1) amount": measured 0.7 .. 1.9


# ------------------------------------------------------------------------------------- chunk plans
def _plan(batch, total, cu_seqlens, chunk_size=BT):
    """``(rows, blocks, nt_dense, offsets)`` - the oracle-side chunk plan.

    ``rows`` are ``(i_b, i_n, bos, eos)``; ``blocks`` are
    ``(i_b, i_n, i_t, t0, length, seq_len, slot)`` with ``slot`` the global chunk ordinal that
    indexes ``h`` (``i_b * NT + i_t`` dense, ``chunk_offsets[i_n] + i_t`` ragged).
    """
    if cu_seqlens is None:
        rows = [(i_b, i_b, 0, total) for i_b in range(batch)]
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
    return rows, blocks, nt_dense, offsets


def _h_shape(batch, total, cu_seqlens, heads, v_dim, k_dim):
    _, _, nt_dense, offsets = _plan(batch, total, cu_seqlens)
    if offsets is None:
        return (batch, nt_dense, heads, v_dim, k_dim)
    return (1, offsets[-1], heads, v_dim, k_dim)


# ----------------------------------------------------------------------------------------- builders
def _make_inputs(total, hk=2, hv=4, d=8, batch=1, dtype=torch.bfloat16, seed=0):
    """``(q, k, v, beta)``: unit-norm keys, random queries/values, beta in (0.1, 0.6)."""
    gen = torch.Generator().manual_seed(seed)
    k = torch.nn.functional.normalize(torch.randn(batch, total, hk, d, generator=gen), dim=-1)
    q = torch.randn(batch, total, hk, d, generator=gen)
    v = torch.randn(batch, total, hv, d, generator=gen)
    beta = 0.1 + 0.5 * torch.rand(batch, total, hv, generator=gen)
    return q.to(dtype), k.to(dtype), v.to(dtype), beta.to(dtype)


def _make_gate(total, hv, cu_seqlens, batch=1, decay=0.05, seed=1, dtype=torch.float32):
    """``(inc, g_ref)``: the non-positive per-token log decay and its chunk-local cumsum.

    The chain is fed ``inc`` and cumsums it itself; ``g_ref`` is the fp64 reference accumulation
    used by the `gdn_reference` oracle (and compared with the chain's ``g``).
    """
    gen = torch.Generator().manual_seed(seed)
    inc = (-torch.rand(batch, total, hv, generator=gen) * decay).to(dtype)
    _, blocks, _, _ = _plan(batch, total, cu_seqlens)
    g_ref = torch.zeros(batch, total, hv, dtype=torch.float64)
    for i_b, _i_n, _i_t, t0, length, _seq_len, _slot in blocks:
        g_ref[i_b, t0 : t0 + length] = inc[i_b, t0 : t0 + length].double().cumsum(0)
    return inc, g_ref


def _make_state(rows, heads, v_dim, k_dim, dtype=torch.float32, seed=3):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(rows, heads, v_dim, k_dim, generator=gen).to(dtype)


def _gqa(heads, k_heads, device=None):
    return torch.arange(heads, device=device) // (heads // k_heads)


# ------------------------------------------------------------------------------------------ oracles
def _full_chain_oracle(q, k, v, beta, inc, initial_state, scale, cu_seqlens):
    """fp64 per-token gated delta rule - the oracle for the whole composed chain.

    ``S_t = exp(d_t) S_{t-1} + beta_t (v_t - exp(d_t) S_{t-1} k_t) k_t^T`` and ``o_t = scale q_t S_t``
    (decay first, then the delta correction, then read the *updated* state - the convention of
    ``fused_recurrent_gated_delta_rule`` and of the chunk chain, see ``recurrent.py`` and contract
    §6.3).  ``d_t = inc_t`` is the raw log decay, so this oracle never uses ``g`` at all: it is
    independent of the cumsum, of the chunking, of ``solve_tril`` and of every mask.

    Returns ``(o, v_new, h, final)`` with ``v_new_t = beta_t (v_t - exp(d_t) S_{t-1} k_t)`` (the
    **ungated** delta) and ``h`` the chunk-entry snapshots in the chain's slot layout.
    """
    q, k, v, beta, inc = (x.to(torch.float64) for x in (q, k, v, beta, inc))
    batch, total, hk, k_dim = k.shape
    heads, v_dim = v.shape[-2], v.shape[-1]
    rows, blocks, nt_dense, offsets = _plan(batch, total, cu_seqlens)
    o = torch.zeros(batch, total, heads, v_dim, dtype=torch.float64, device=q.device)
    v_new = torch.zeros(batch, total, heads, v_dim, dtype=torch.float64, device=q.device)
    h = torch.zeros(
        _h_shape(batch, total, cu_seqlens, heads, v_dim, k_dim),
        dtype=torch.float64,
        device=q.device,
    )
    final = torch.zeros(len(rows), heads, v_dim, k_dim, dtype=torch.float64, device=q.device)
    head_map = _gqa(heads, hk, q.device)

    starts = {(i_b, t0): (i_t, slot) for i_b, _i_n, i_t, t0, _len, _seq, slot in blocks}
    for i_b, i_n, bos, eos in rows:
        if initial_state is not None:
            state = initial_state[i_n].to(torch.float64)
        else:
            state = torch.zeros(heads, v_dim, k_dim, dtype=torch.float64, device=q.device)
        for t in range(bos, eos):
            if (i_b, t) in starts:  # chunk-entry snapshot, taken before the token is processed
                i_t, slot = starts[(i_b, t)]
                if offsets is None:
                    h[i_b, i_t] = state
                else:
                    h[0, slot] = state
            decay = torch.exp(inc[i_b, t])  # [H]
            state = state * decay[:, None, None]
            kt = k[i_b, t][head_map]  # [H, K]
            delta = beta[i_b, t][:, None] * (v[i_b, t] - (state * kt[:, None, :]).sum(-1))
            v_new[i_b, t] = delta
            state = state + delta[:, :, None] * kt[:, None, :]
            qt = q[i_b, t][head_map]  # [H, K]
            o[i_b, t] = scale * (state * qt[:, None, :]).sum(-1)
        final[i_n] = state
    return o, v_new, h, final


def _wu_reference(k, v, beta, g, cu_seqlens):
    """fp64 ``(w, u)`` from ``gdn_reference``; dense ``B > 1`` is fed one row at a time.

    ``chunk_wu_reference``'s ABI is the packed ``[1, T, ...]`` layout, so a dense batch has to be
    split into its rows.  It computes ``(I + A)^{-1}`` with ``torch.linalg.inv`` - independent of
    the port's ``solve_tril`` - and returns ``w``/``u`` in ``v.dtype`` (fp64 here).
    """
    if cu_seqlens is None:
        row_cu = torch.tensor([0, k.shape[1]], dtype=torch.int64)
        ws, us = [], []
        for i_b in range(k.shape[0]):
            w, u, _a_inv = chunk_wu_reference(
                k[i_b : i_b + 1], v[i_b : i_b + 1], beta[i_b : i_b + 1], g[i_b : i_b + 1], row_cu
            )
            ws.append(w)
            us.append(u)
        return torch.cat(ws, 0), torch.cat(us, 0)
    w, u, _a_inv = chunk_wu_reference(k, v, beta, g, cu_seqlens)
    return w, u


# --------------------------------------------------------------------------------------- comparison
def _rel_err(actual, expected):
    a, e = actual.to(torch.float64), expected.to(torch.float64)
    assert tuple(a.shape) == tuple(e.shape), (tuple(a.shape), tuple(e.shape))
    return (a - e).abs().max().item() / max(e.abs().max().item(), 1.0)


def _assert_close(actual, expected, tol, what=""):
    err = _rel_err(actual, expected)
    assert err <= tol, f"{what}: normalized max error {err:.3e} exceeds {tol:.3e}"


# ==================================================================== the composed chain vs oracle
_LAYOUTS = [
    {"batch": 1, "total": 130, "cu": None, "seed": 300},
    {"batch": 2, "total": BT + 6, "cu": None, "seed": 310},
    {"batch": 1, "total": 130, "cu": [0, 37, 130], "seed": 320},
]
_LAYOUT_IDS = ["dense_b1_t130", "dense_b2_t70", "varlen_0_37_130"]


@pytest.mark.parametrize(
    "dtype,tol",
    [(torch.float64, FP64_TOL), (torch.bfloat16, BF16_TOL)],
    ids=["fp64", "bf16"],
)
@pytest.mark.parametrize("layout", _LAYOUTS, ids=_LAYOUT_IDS)
def test_composed_chain_matches_the_full_oracle(dtype, tol, layout):
    """The whole chain, both dtypes, all three layouts: o / final_state / h / v_new / w / u."""
    total, batch, seed = layout["total"], layout["batch"], layout["seed"]
    cu = None if layout["cu"] is None else torch.tensor(layout["cu"], dtype=torch.int32)
    hk, hv, d = 2, 4, 8
    n_seq = batch if cu is None else len(layout["cu"]) - 1
    gate_dtype = torch.float64 if dtype is torch.float64 else torch.float32

    q, k, v, beta = _make_inputs(total, hk=hk, hv=hv, d=d, batch=batch, dtype=dtype, seed=seed)
    inc, g_ref = _make_gate(total, hv, cu, batch=batch, seed=seed + 1, dtype=gate_dtype)
    init = _make_state(n_seq, hv, d, d, dtype=dtype, seed=seed + 2)
    scale = d**-0.5

    g, o, a_inv, final_state, w, h, v_new = chunk_gated_delta_rule_fwd(
        q, k, v, inc, beta, scale, init, True, cu_seqlens=cu
    )

    # --- the public wrapper is the same computation, plus o.to(q.dtype)
    o_pub, final_pub = chunk_gated_delta_rule(
        q, k, v, inc, beta, scale=scale, initial_state=init, output_final_state=True, cu_seqlens=cu
    )
    assert torch.equal(o_pub, o.to(q.dtype)), "wrapper output differs from the composed forward"
    assert torch.equal(final_pub, final_state)

    # --- dtype flow through the chain (contract §1.1)
    assert g.dtype is gate_dtype and g.shape == (batch, total, hv)
    assert a_inv.dtype is k.dtype and a_inv.shape == (batch, total, hv, BT)
    assert w.dtype is k.dtype and w.shape == (batch, total, hv, d)
    assert h.dtype is k.dtype and h.shape == _h_shape(batch, total, cu, hv, d, d)
    assert v_new.dtype is v.dtype and v_new.shape == (batch, total, hv, d)
    assert o.dtype is v.dtype and o.shape == (batch, total, hv, d)
    assert final_state.dtype is init.dtype and final_state.shape == (n_seq, hv, d, d)

    # --- step 0: the chain's cumsum is chunk-local and matches the fp64 reference accumulation
    if dtype is torch.float64:
        assert torch.equal(g, g_ref), "fp64 cumsum must be exact"
    else:
        _assert_close(g, g_ref, 1e-6, "g")

    # --- steps 1-3 vs gdn_reference (kkt + solve_tril + recompute_w_u, all independent)
    w_ref, u_ref = _wu_reference(k, v, beta, g_ref, cu)
    _assert_close(w, w_ref, tol, "w")
    # `u` is not part of the debug tuple; rebuild it from the chain's own A_inv and compare - this
    # pins the A_inv the chain produced (solve_tril) against the reference's independent inverse.
    w_chain, u_chain = recompute_w_u_fwd(
        k=k, v=v, beta=beta, A=a_inv, g_cumsum=g, cu_seqlens=cu
    )
    assert torch.equal(w_chain, w), "recompute_w_u_fwd is not deterministic on the same A"
    _assert_close(u_chain, u_ref, tol, "u")

    # --- steps 4-5 vs the per-token delta rule
    ref_o, ref_v_new, ref_h, ref_final = _full_chain_oracle(
        q, k, v, beta, inc, init, scale, cu
    )
    _assert_close(h, ref_h, tol, "h")
    _assert_close(v_new, ref_v_new, tol, "v_new")
    _assert_close(o, ref_o, tol, "o")
    _assert_close(final_state, ref_final, tol, "final_state")


@pytest.mark.parametrize("total", [1, 63, 64, 65, 130])
def test_every_tail_length_matches_the_fp64_oracle(total):
    """``T in {1, 63, 64, 65, 130}``: a single partial chunk, exact fits and one-token tails.

    ``BT`` is pinned to 64 and never shrinks for short sequences [vLLM]
    ``qwen_gdn_linear_attn.py:1085-1087``, so ``T=1``/``T=63`` exercise the partial-chunk paths of
    every kernel (``g_last``, the store masks, the ``h`` slot count) in the same way ``T=130``
    exercises a 2-token tail.
    """
    hk, hv, d = 2, 4, 8
    q, k, v, beta = _make_inputs(total, hk=hk, hv=hv, d=d, dtype=torch.float64, seed=400 + total)
    inc, _ = _make_gate(total, hv, None, seed=500 + total, dtype=torch.float64)
    init = _make_state(1, hv, d, d, dtype=torch.float64, seed=600 + total)
    scale = d**-0.5

    g, o, a_inv, final_state, w, h, v_new = chunk_gated_delta_rule_fwd(
        q, k, v, inc, beta, scale, init, True
    )
    assert h.shape == (1, -(-total // BT), hv, d, d), "chunk count must be cdiv(T, 64)"
    assert a_inv.shape == (1, total, hv, BT)

    ref_o, ref_v_new, ref_h, ref_final = _full_chain_oracle(q, k, v, beta, inc, init, scale, None)
    _assert_close(h, ref_h, FP64_TOL, "h")
    _assert_close(v_new, ref_v_new, FP64_TOL, "v_new")
    _assert_close(o, ref_o, FP64_TOL, "o")
    _assert_close(final_state, ref_final, FP64_TOL, "final_state")


def test_no_initial_state_starts_from_zero_fp64():
    total, hv, d = BT + 3, 4, 8
    q, k, v, beta = _make_inputs(total, dtype=torch.float64, seed=700)
    inc, _ = _make_gate(total, hv, None, seed=701, dtype=torch.float64)
    g, o, _a, final_state, _w, h, v_new = chunk_gated_delta_rule_fwd(
        q, k, v, inc, beta, 0.3, None, True
    )
    assert torch.count_nonzero(h[:, 0]) == 0, "chunk 0's entry state must be zero"
    ref_o, ref_v_new, ref_h, ref_final = _full_chain_oracle(q, k, v, beta, inc, None, 0.3, None)
    _assert_close(h, ref_h, FP64_TOL, "h")
    _assert_close(v_new, ref_v_new, FP64_TOL, "v_new")
    _assert_close(o, ref_o, FP64_TOL, "o")
    _assert_close(final_state, ref_final, FP64_TOL, "final_state")

    o_pub, final_pub = chunk_gated_delta_rule(q, k, v, inc, beta, 0.3)
    assert torch.equal(o_pub, o) and final_pub is None


# ================================================================== causal masking, end-to-end
def _perturb_later(tensors, cut, seed):
    """Replace every entry at ``tokens > cut`` with a fresh random draw (same shapes/dtypes).

    The last tensor is the raw log gate and is drawn **negative**, so the perturbed gate stays a
    legal log-decay (``g <= 0``) instead of turning ``exp(g)`` into a growth factor.
    """
    gen = torch.Generator().manual_seed(seed)
    out = []
    for pos, x in enumerate(tensors):
        y = x.clone()
        tail = y[:, cut + 1 :]
        if tail.numel():
            draw = torch.rand(tail.shape, generator=gen, dtype=torch.float64).to(y.dtype)
            y[:, cut + 1 :] = -draw if pos == len(tensors) - 1 else draw
        out.append(y)
    return out


def test_causal_masking_dense_token_outputs_do_not_see_later_tokens():
    """Token ``t``'s output must be bit-identical when every token ``> t`` is replaced.

    The perturbation changes ``q``, ``k``, ``v``, ``beta`` and the raw gate for tokens 71..129, i.e.
    a full later chunk plus the 2-token tail.  Nothing at or before token 70 may move: not the
    chunk-local cumsum, not ``A``'s earlier rows, not ``A_inv``'s unit-lower rows, not ``v_new``'s
    rows, not the entry state ``h`` of the chunk holding token 70.  A missing mask anywhere in the
    chain (a kkt that reaches across the chunk, a ``chunk_o`` without its triangle, an ``h`` slot
    read one chunk too late) shows up here as a non-zero difference.
    """
    total, hk, hv, d, cut = 130, 2, 4, 8, 70
    q, k, v, beta = _make_inputs(total, hk=hk, hv=hv, d=d, dtype=torch.float64, seed=800)
    inc, _ = _make_gate(total, hv, None, seed=801, dtype=torch.float64)
    init = _make_state(1, hv, d, d, dtype=torch.float64, seed=802)
    scale = d**-0.5

    o0, _ = chunk_gated_delta_rule(q, k, v, inc, beta, scale, init, False)
    q1, k1, v1, beta1, inc1 = _perturb_later((q, k, v, beta, inc), cut, seed=999)
    o1, _ = chunk_gated_delta_rule(q1, k1, v1, inc1, beta1, scale, init, False)

    assert torch.equal(o1[:, : cut + 1], o0[:, : cut + 1]), (
        "causality violated: tokens <= 70 changed when tokens > 70 were perturbed"
    )
    assert not torch.equal(o1[:, cut + 1 :], o0[:, cut + 1 :]), (
        "the perturbation did not reach the later tokens - the test would be vacuous"
    )


def test_causal_masking_varlen_and_sequence_independence():
    """``cu_seqlens=[0, 37, 130]``: sequences are independent and each token is causal inside one.

    (a) perturbing only the second sequence (tokens 37..129) must leave **all** of sequence 0
        bit-identical - the packed layout must not leak across a ``cu_seqlens`` boundary;
    (b) perturbing tokens 11..129 (the rest of sequence 0's single chunk plus all of sequence 1)
        must leave tokens 0..10 of sequence 0 bit-identical.
    """
    total, hk, hv, d = 130, 2, 4, 8
    cu = torch.tensor([0, 37, 130], dtype=torch.int32)
    q, k, v, beta = _make_inputs(total, hk=hk, hv=hv, d=d, dtype=torch.float64, seed=810)
    inc, _ = _make_gate(total, hv, cu, seed=811, dtype=torch.float64)
    init = _make_state(2, hv, d, d, dtype=torch.float64, seed=812)
    scale = d**-0.5

    o0, _ = chunk_gated_delta_rule(q, k, v, inc, beta, scale, init, False, cu_seqlens=cu)
    assert o0.shape == (1, total, hv, d) and torch.isfinite(o0).all()

    # (a) only sequence 1 is perturbed
    qa, ka, va, ba, ia = _perturb_later((q, k, v, beta, inc), 36, seed=1001)
    oa, _ = chunk_gated_delta_rule(qa, ka, va, ia, ba, scale, init, False, cu_seqlens=cu)
    assert torch.equal(oa[:, :37], o0[:, :37]), "sequence 0 read data from sequence 1"
    assert not torch.equal(oa[:, 37:], o0[:, 37:])

    # (b) sequence 0's own later tokens are perturbed too
    qb, kb, vb, bb, ib = _perturb_later((q, k, v, beta, inc), 10, seed=1002)
    ob, _ = chunk_gated_delta_rule(qb, kb, vb, ib, bb, scale, init, False, cu_seqlens=cu)
    assert torch.equal(ob[:, :11], o0[:, :11]), "causality violated across the varlen chunk"
    assert not torch.equal(ob[:, 11:], o0[:, 11:])


# ============================================================ the v -> v_new boundary and the gate
def test_negative_control_original_v_at_the_chunk_o_boundary():
    """Feed the caller's ``v`` to ``chunk_o`` instead of ``v_new``: an O(1), silent error.

    The composed entry itself must be at fp64 rounding, which it can only be if the boundary passes
    ``v_new`` - so a wiring regression fails this test twice over: the oracle comparison and the
    ``o`` vs ``wrong`` separation.
    """
    total, hk, hv, d, cut = 130, 2, 4, 8, 90
    q, k, v, beta = _make_inputs(total, hk=hk, hv=hv, d=d, dtype=torch.float64, seed=820)
    inc, _ = _make_gate(total, hv, None, seed=821, dtype=torch.float64)
    init = _make_state(1, hv, d, d, dtype=torch.float64, seed=822)
    scale = d**-0.5

    g, o, a_inv, final_state, w, h, v_new = chunk_gated_delta_rule_fwd(
        q, k, v, inc, beta, scale, init, True
    )
    ref_o, _, _, _ = _full_chain_oracle(q, k, v, beta, inc, init, scale, None)
    good_err = _rel_err(o, ref_o)
    assert good_err <= FP64_TOL, f"the composed chain is broken: error {good_err:.3e}"

    wrong = chunk_fwd_o(q=q, k=k, v=v, h=h, g=g, scale=scale)  # ORIGINAL v, not v_new
    wrong_err = _rel_err(wrong, ref_o)
    assert wrong.shape == o.shape  # the trap: a perfectly shaped, wrong answer
    assert wrong_err > max(O1_ERR, 1e3 * good_err), (
        f"feeding the original v should be O(1) wrong, measured {wrong_err:.3e}"
    )
    assert _rel_err(o, wrong) > O1_ERR
    # and the two are different tensors even though v_new and v have the same shape/dtype
    assert not torch.equal(v_new, v)
    # the later tokens (where the state term dominates) are where it hurts, but every row differs
    assert not torch.equal(wrong[:, :cut], o[:, :cut])


def test_v_new_is_stored_ungated_and_a_gated_copy_would_not_compose():
    """`delta_h` saves ``u - w @ h[c]^T`` **before** the gate; gating it again breaks the chain.

    This is the numerical evidence for contract §6.3 item 2: the ungated value is what composes
    (it matches the chain's own ``u``/``w``/``h`` and the per-token oracle), while the "gated
    ``v_new``" variant differs by O(1) and, once fed to ``chunk_o``, produces an O(1)-wrong output.
    """
    total, hk, hv, d = 130, 2, 4, 8
    q, k, v, beta = _make_inputs(total, hk=hk, hv=hv, d=d, dtype=torch.float64, seed=830)
    inc, _ = _make_gate(total, hv, None, seed=831, dtype=torch.float64)
    init = _make_state(1, hv, d, d, dtype=torch.float64, seed=832)
    scale = d**-0.5

    g, o, a_inv, _final, w, h, v_new = chunk_gated_delta_rule_fwd(
        q, k, v, inc, beta, scale, init, True
    )
    # `u` is not part of the public tuple; rebuild it from the chain's own (bf16/fp64) A_inv.
    w2, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=a_inv, g_cumsum=g, cu_seqlens=None)
    assert torch.equal(w2, w), "recompute_w_u_fwd is not deterministic on the same A"

    _rows, blocks, _nt_dense, _offsets = _plan(1, total, None)
    pred = torch.zeros_like(u)
    for i_b, _i_n, _i_t, t0, length, _seq_len, slot in blocks:
        idx = torch.arange(t0, t0 + length)
        pred[i_b, idx] = torch.einsum("thk,hvk->thv", w[i_b, idx], h[i_b, slot])

    ungated = u - pred
    _assert_close(v_new, ungated, FP64_TOL, "v_new must be the UNGATED u - w h^T")

    # The gated variant the (wrong) contract text described.
    gated = ungated.clone()
    for i_b, _i_n, _i_t, t0, length, _seq_len, _slot in blocks:
        idx = torch.arange(t0, t0 + length)
        g_last = g[i_b, t0 + length - 1]  # last VALID token of the chunk
        diff = g_last.unsqueeze(0) - g[i_b, idx]  # <= 0 for a monotone gate
        factor = torch.exp(torch.where(diff <= 0, diff, torch.full_like(diff, float("-inf"))))
        gated[i_b, idx] = gated[i_b, idx] * factor.unsqueeze(-1)

    gated_err = _rel_err(gated, v_new)
    assert gated_err > O1_ERR, (
        f"the gated copy should differ from the stored v_new by O(1), measured {gated_err:.3e}"
    )
    # ... and it does not compose: chunk_o applies exp(g_i - g_j) itself, so the gate lands twice.
    o_gated = chunk_fwd_o(q=q, k=k, v=gated, h=h, g=g, scale=scale)
    composition_err = _rel_err(o_gated, o)
    assert composition_err > O1_ERR, (
        f"a gated v_new should break the composition by O(1), measured {composition_err:.3e}"
    )


# ============================================================================ the A.triu() invariant
def test_strict_lower_invariant_is_enforced_and_block_local():
    """The guard rejects a diagonal/upper entry, accepts legal lower entries, and respects varlen.

    Three separate traps: (i) the diagonal of a chunk is illegal (and the upper triangle too);
    (ii) a global ``A.triu()`` would be the wrong test for ``[B, T, H, BT]`` (with dense chunking,
    token 70 is chunk 1 with local column 6, so column 5 is legal while column 6 is the diagonal);
    (iii) under ``cu_seqlens`` the chunk restarts at the sequence boundary (token 37 is local
    column 0, so *any* column is on or above the diagonal, while token 70 -- 33 tokens into
    sequence 1 -- has 32 legal lower columns).
    """
    total, hv = 130, 4
    cu = torch.tensor([0, 37, 130], dtype=torch.int32)
    q, k, v, beta = _make_inputs(total, hv=hv, seed=840)
    _inc, g_ref = _make_gate(total, hv, cu, seed=841)
    g = g_ref.to(torch.float32)

    def raises(tensor, cu_seqlens):
        with pytest.raises(ValueError, match=r"A\.triu\(\) == 0"):
            assert_strict_lower_chunks(tensor, cu_seqlens)

    # --- dense chunking (cu_seqlens=None): chunk 1 starts at token 64
    dense = chunk_scaled_dot_kkt_fwd(
        k=k, beta=beta, g=g, cu_seqlens=None, output_dtype=torch.float32
    )
    assert int(torch.count_nonzero(dense)) > 0, "the fixture must have non-zero blocks"
    assert_strict_lower_chunks(dense)  # the ported kkt satisfies [vLLM] fla_solve_tril.py:515
    bad = dense.clone()
    bad[0, 5, 1, 5] = 1.0  # token 5, local column 5 = diagonal
    raises(bad, None)
    bad = dense.clone()
    bad[0, 5, 1, 6] = 1.0  # column 6 > local index = upper triangle
    raises(bad, None)
    good = dense.clone()
    good[0, 70, 0, 5] = 1.0  # chunk 1, local index 6 -> column 5 is strictly lower: legal
    assert_strict_lower_chunks(good)
    bad = dense.clone()
    bad[0, 70, 0, 6] = 1.0  # column 6 == local index: the chunk's diagonal
    raises(bad, None)

    # --- packed (cu_seqlens=[0, 37, 130]): sequence 1's chunks start at token 37
    packed = chunk_scaled_dot_kkt_fwd(
        k=k, beta=beta, g=g, cu_seqlens=cu, output_dtype=torch.float32
    )
    assert int(torch.count_nonzero(packed)) > 0
    assert_strict_lower_chunks(packed, cu)
    good = packed.clone()
    good[0, 70, 0, 32] = 1.0  # 70 - 37 = 33 -> column 32 is strictly lower: legal
    assert_strict_lower_chunks(good, cu)
    bad = packed.clone()
    bad[0, 70, 0, 33] = 1.0  # column 33 == local index: the sequence-local chunk diagonal
    raises(bad, cu)
    bad = packed.clone()
    bad[0, 37, 0, 0] = 1.0  # token 37 is local index 0: every column is on/above the diagonal
    raises(bad, cu)
    # reading the packed tensor with dense chunk starts mis-locates the diagonal and is rejected
    raises(packed, None)

    # malformed shapes are rejected too
    with pytest.raises(ValueError, match=r"A must be \[B, T, H, BT\]"):
        assert_strict_lower_chunks(packed[0], cu)
    with pytest.raises(ValueError, match="must equal chunk_size"):
        assert_strict_lower_chunks(packed, cu, chunk_size=32)


def test_composed_chain_rejects_a_kkt_that_violates_the_invariant(monkeypatch):
    """The guard is wired into the chain, between steps 1 and 2 - not just a standalone helper.

    A ``kkt`` that leaks the diagonal (``>=`` instead of ``>``) would be silently swallowed by
    ``solve_tril`` (contract §5.2), so the composed entry must refuse to run it.
    """
    total, hv, d = BT + 1, 4, 8
    q, k, v, beta = _make_inputs(total, hv=hv, d=d, seed=870)
    inc, _ = _make_gate(total, hv, None, seed=871)
    real_kkt = chunk_scaled_dot_kkt_fwd

    def leaky_kkt(*args, **kwargs):
        a = real_kkt(*args, **kwargs)
        a[0, 0, 0, 0] = 1.0  # chunk 0's diagonal entry: exactly what solve_tril discards
        return a

    monkeypatch.setattr("tileinfer.vllm.ops.fla.chunk.chunk_scaled_dot_kkt_fwd", leaky_kkt)
    with pytest.raises(ValueError, match=r"A\.triu\(\) == 0"):
        chunk_gated_delta_rule(q, k, v, inc, beta, 0.5)

    monkeypatch.undo()
    o, _ = chunk_gated_delta_rule(q, k, v, inc, beta, 0.5)  # the unpatched chain is fine
    assert torch.isfinite(o.float()).all()


# ============================================================================= the public wrapper
def test_public_wrapper_contract_and_dtype_flow():
    """``scale`` default, ``output_final_state``, the l2norm rebinding, dtypes and the varlen guards."""
    total, hk, hv, d = BT + 1, 2, 4, 8
    dtype = torch.bfloat16
    q, k, v, beta = _make_inputs(total, hk=hk, hv=hv, d=d, dtype=dtype, seed=850)
    inc, _ = _make_gate(total, hv, None, seed=851, dtype=torch.float32)
    init = _make_state(1, hv, d, d, dtype=dtype, seed=852)

    # --- scale defaults to k.shape[-1] ** -0.5 [vLLM] chunk.py:228-229
    o_default, final_default = chunk_gated_delta_rule(q, k, v, inc, beta, initial_state=init)
    assert final_default is None, "output_final_state=False must give None [vLLM] chunk.py:180-181"
    o_explicit, _ = chunk_gated_delta_rule(
        q, k, v, inc, beta, scale=d**-0.5, initial_state=init
    )
    assert torch.equal(o_default, o_explicit), "the default scale must be K ** -0.5 from k"
    assert o_default.dtype is dtype and o_default.shape == (1, total, hv, d)
    assert torch.isfinite(o_default.float()).all()

    # o is cast to q.dtype [vLLM] chunk.py:134; final_state has [N, H, V, K]
    _o, final_state = chunk_gated_delta_rule(
        q, k, v, inc, beta, scale=0.5, initial_state=init, output_final_state=True
    )
    assert final_state.dtype is dtype and final_state.shape == (1, hv, d, d)

    # --- the chain's dtype flow (contract §1.1): g fp32, A -> k.dtype, w/h k.dtype, v_new v.dtype
    g, o, a_inv, fs, w, h, v_new = chunk_gated_delta_rule_fwd(
        q, k, v, inc, beta, 0.5, init, True
    )
    assert g.dtype is torch.float32 and g.shape == (1, total, hv)
    assert a_inv.dtype is dtype and a_inv.shape == (1, total, hv, BT)
    assert w.dtype is dtype and w.shape == (1, total, hv, d)
    assert h.dtype is dtype and h.shape == (1, -(-total // BT), hv, d, d)
    assert v_new.dtype is dtype and v_new.shape == (1, total, hv, d)
    assert torch.equal(fs, final_state)

    # --- use_qk_l2norm_in_kernel: a rebinding of q/k, identical to normalising before the call
    q_before, k_before = q.clone(), k.clone()
    o_norm, _ = chunk_gated_delta_rule(
        q, k, v, inc, beta, scale=0.5, initial_state=init, use_qk_l2norm_in_kernel=True
    )
    o_manual, _ = chunk_gated_delta_rule(
        l2norm_fwd(q), l2norm_fwd(k), v, inc, beta, scale=0.5, initial_state=init
    )
    assert torch.equal(o_norm, o_manual)
    assert torch.equal(q, q_before) and torch.equal(k, k_before), "l2norm must not write in place"
    o_plain, _ = chunk_gated_delta_rule(q, k, v, inc, beta, scale=0.5, initial_state=init)
    assert not torch.equal(o_norm, o_plain), "the l2norm option should change the result here"

    # --- fp32 is accepted (deliberate deviation: vLLM rejects it for its Triton kernels)
    o_f32, _ = chunk_gated_delta_rule(
        q.float(), k.float(), v.float(), inc.float(), beta.float(), scale=0.5
    )
    assert o_f32.dtype is torch.float32 and torch.isfinite(o_f32).all()

    # --- guards [vLLM] chunk.py:212-227
    with pytest.raises(ValueError, match="share one dtype"):
        chunk_gated_delta_rule(q.float(), k, v, inc, beta)
    with pytest.raises(ValueError, match=r"beta must be of shape"):
        chunk_gated_delta_rule(q, k, v, inc, beta[..., 0])

    total2, cu = 45, torch.tensor([0, 20, 45], dtype=torch.int32)
    q2, k2, v2, beta2 = _make_inputs(total2, hk=hk, hv=hv, d=d, dtype=dtype, seed=860)
    inc2, _ = _make_gate(total2, hv, cu, seed=861, dtype=torch.float32)
    with pytest.raises(ValueError, match="batch size is expected to be 1"):
        chunk_gated_delta_rule(
            q2.repeat(2, 1, 1, 1), k2.repeat(2, 1, 1, 1), v2.repeat(2, 1, 1, 1),
            inc2.repeat(2, 1, 1), beta2.repeat(2, 1, 1), cu_seqlens=cu,
        )
    bad_init = _make_state(3, hv, d, d, dtype=dtype, seed=862)
    with pytest.raises(ValueError, match="number of initial states"):
        chunk_gated_delta_rule(q2, k2, v2, inc2, beta2, initial_state=bad_init, cu_seqlens=cu)

    good_init = _make_state(2, hv, d, d, dtype=dtype, seed=863)
    o_var, fs_var = chunk_gated_delta_rule(
        q2, k2, v2, inc2, beta2, initial_state=good_init, output_final_state=True, cu_seqlens=cu
    )
    assert o_var.shape == (1, total2, hv, d) and fs_var.shape == (2, hv, d, d)
