"""Tests for the torch ports of ``chunk_scaled_dot_kkt_fwd`` and ``recompute_w_u_fwd``.

Semantics pinned here (see ``docs/notes/fla-chunk-contract.md`` and the two module docstrings):

* ``A[i, j, h] = beta[i, h] * safe_exp(g[i, h] - g[j, h]) * (k_i . k_j)`` for ``i > j`` **inside one
  chunk**, exactly zero on the diagonal and in the upper triangle - a ``>` `` mask, and a *discard*
  (``torch.where``) rather than a multiply, because the upper-triangle exponent is positive and
  ``inf * 0`` is NaN.
* ``u = A_inv @ (v * beta)`` and ``w = A_inv @ (k * beta * exp(g))`` per chunk and head, with
  ``A_inv = (I + A_kkt)^{-1}`` supplied by ``solve_tril`` (bf16 in the real chain), plain ``exp`` on
  the (non-positive) cumulative gate, a row-wise ``beta``, and operands rounded down before each dot.
* ``w`` is ``[B, T, H, K]`` (value-head count, key width), ``u`` is ``[B, T, H, V]``.

The oracles are the pre-existing fp64 references in ``tileinfer/testing/gdn_reference.py``:
``kkt_reference`` (``[1, T, HV, BT]`` strict-lower ``A``) and ``chunk_wu_reference`` (``(w, u,
a_inv)``, which independently inverts ``I + A`` with ``torch.linalg.inv``).  They are B=1 only, so
the helpers below apply them per batch row.  No third reference is written; the only additional
formulas in this file are (a) deliberately *wrong* variants used as negative controls and (b) a
closed-form row-0 expectation for an identity ``A_inv``.

The tests are CPU-only and mirror the production chain ``kkt -> solve_tril -> recompute_w_u_fwd``
[vLLM chunk.py:41-60].
"""

from __future__ import annotations

import pytest
import torch

from tileinfer.testing.gdn_reference import chunk_wu_reference, kkt_reference
from tileinfer.vllm.ops.fla.kkt import chunk_scaled_dot_kkt_fwd
from tileinfer.vllm.ops.fla.solve_tril import prepare_chunk_indices, solve_tril
from tileinfer.vllm.ops.fla.wu import recompute_w_u_fwd

BT = 64  # vLLM 0.23 pins BT = chunk_size = 64 (qwen_gdn_linear_attn.py:1085-1087)
KKT_TOL = 1e-5  # A is fp32 and bf16 inputs are exact in fp32: only fp32 rounding separates them
WU_TOL = 1e-2  # w/u are bf16 outputs; measured chain error is <= 4.4e-3
FP32_TOL = 1e-5


# ------------------------------------------------------------------------------------- chunk plans
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
    return _dense_chunks(total, chunk_size) if cu_seqlens is None else _varlen_chunks(cu_seqlens, chunk_size)


def _all_blocks(batch: int, cu_seqlens, total: int, chunk_size: int = BT):
    return [(b, t0, length) for b in range(batch) for t0, length in _chunks(cu_seqlens, total, chunk_size)]


def _strict_lower(length: int, device=None) -> torch.Tensor:
    i = torch.arange(length, device=device)
    return i[:, None] > i[None, :]


# ---------------------------------------------------------------------------------------- builders
def _make_kv(total, hk=2, hv=4, d=8, v_dim=None, batch=1, dtype=torch.bfloat16, seed=0):
    gen = torch.Generator().manual_seed(seed)
    v_dim = d if v_dim is None else v_dim
    k = torch.randn(batch, total, hk, d, generator=gen)
    k = torch.nn.functional.normalize(k, dim=-1).to(dtype)  # unit keys: delta-rule-like scale
    v = torch.randn(batch, total, hv, v_dim, generator=gen).to(dtype)
    beta = (0.1 + 0.5 * torch.rand(batch, total, hv, generator=gen)).to(dtype)
    return k, v, beta


def _make_g(total, hv, cu_seqlens, batch=1, chunk_size=BT, decay=0.05, seed=1, dtype=torch.float32):
    """Chunk-local cumsum of a non-positive log-decay, restarting per chunk and per sequence."""
    gen = torch.Generator().manual_seed(seed)
    g = torch.zeros(batch, total, hv, dtype=torch.float32)
    for b in range(batch):
        for t0, length in _chunks(cu_seqlens, total, chunk_size):
            inc = -torch.rand(length, hv, generator=gen) * decay
            g[b, t0 : t0 + length] = inc.cumsum(0)
    return g.to(dtype)


# ---------------------------------------------------------------------------------------- oracles
def _kkt_oracle(k, beta, g, cu_seqlens, chunk_size=BT):
    """fp64 ``kkt_reference`` per batch row (the oracle only supports B == 1)."""
    rows = [
        kkt_reference(k[b : b + 1], beta[b : b + 1], g[b : b + 1], cu_seqlens, chunk_size)
        for b in range(k.shape[0])
    ]
    return torch.cat(rows, dim=0)


def _wu_oracle(k, v, beta, g, cu_seqlens, a_kkt, chunk_size=BT):
    """fp64 ``chunk_wu_reference`` per batch row; returns ``(w, u, a_inv)``."""
    ws, us, ais = [], [], []
    for b in range(k.shape[0]):
        w, u, ai = chunk_wu_reference(
            k[b : b + 1],
            v[b : b + 1],
            beta[b : b + 1],
            g[b : b + 1],
            cu_seqlens,
            a_kkt=a_kkt[b : b + 1],
            chunk_size=chunk_size,
        )
        ws.append(w)
        us.append(u)
        ais.append(ai)
    return torch.cat(ws), torch.cat(us), torch.cat(ais)


# ------------------------------------------------------------------------------------- comparison
def _rel_err(actual, expected):
    a, e = actual.to(torch.float64), expected.to(torch.float64)
    assert a.shape == e.shape, (tuple(a.shape), tuple(e.shape))
    return (a - e).abs().max().item() / max(e.abs().max().item(), 1.0)


def _assert_close(actual, expected, tol):
    err = _rel_err(actual, expected)
    assert err <= tol, f"normalized max error {err:.3e} exceeds {tol:.3e}"


def _lower_entries(t, chunks):
    """Flatten the strictly-lower entries of every ``(t0, length)`` block of a B=1 tensor."""
    parts = []
    for t0, length in chunks:
        block = t[0, t0 : t0 + length, :, :length].permute(1, 0, 2)  # [H, l, l]
        parts.append(block[:, _strict_lower(length, t.device)])
    return torch.cat(parts)


# ------------------------------------------------------------------------------------- structure
def _above_diagonal_count(block, length):
    """Non-zero entries on/above the diagonal of a ``[l, H, l]`` block (NaN counts as non-zero)."""
    keep = _strict_lower(length, block.device).unsqueeze(1)  # [l, 1, l]
    above = torch.where(keep, torch.zeros((), dtype=block.dtype, device=block.device), block)
    return int(torch.count_nonzero(above).item())


def _assert_kkt_structure(A, batch, cu_seqlens, total, chunk_size=BT):
    """Every chunk block strictly lower, every uncovered column zero, everything finite."""
    assert torch.isfinite(A).all(), "kkt output contains inf/nan"
    for b, t0, length in _all_blocks(batch, cu_seqlens, total, chunk_size):
        block = A[b, t0 : t0 + length, :, :length]
        assert _above_diagonal_count(block, length) == 0, (
            f"chunk at b={b}, t0={t0} is not strictly lower triangular"
        )
        assert torch.count_nonzero(A[b, t0 : t0 + length, :, length:]) == 0, (
            f"chunk at b={b}, t0={t0} wrote past its valid columns"
        )


def _identity_inv(batch, total, heads, chunk_size=BT, dtype=torch.float32):
    """``A_inv`` = identity inside every chunk block: the easiest closed-form ``w``/``u``."""
    A = torch.zeros(batch, total, heads, chunk_size, dtype=dtype)
    for t0, length in _dense_chunks(total, chunk_size):
        A[:, t0 : t0 + length, :, :length] = torch.eye(length, dtype=dtype).view(1, length, 1, length)
    return A


# ==================================================================================== kkt: basics
def test_kkt_shape_dtype_and_head_count():
    total = BT
    k, _, beta = _make_kv(total, hk=2, hv=4, d=8)
    g = _make_g(total, 4, None)
    A = chunk_scaled_dot_kkt_fwd(k, beta, g, cu_seqlens=torch.tensor([0, total], dtype=torch.int32))
    assert A.shape == (1, total, 4, BT)  # H comes from beta, not from k (k has 2 heads)
    assert A.dtype is torch.float32
    assert A.is_contiguous()
    assert torch.isfinite(A).all()


def test_kkt_matches_oracle_single_chunk_bf16():
    total = BT
    k, _, beta = _make_kv(total, seed=2)
    g = _make_g(total, 4, None, seed=3)
    cu = torch.tensor([0, total], dtype=torch.int32)
    A = chunk_scaled_dot_kkt_fwd(k, beta, g, cu_seqlens=cu)
    _assert_close(A, _kkt_oracle(k, beta, g, cu), KKT_TOL)
    _assert_kkt_structure(A, 1, cu, total)
    assert int(torch.count_nonzero(A).item()) > 0


def test_kkt_matches_oracle_fp32_inputs():
    total = BT
    k, _, beta = _make_kv(total, seed=4, dtype=torch.float32)
    g = _make_g(total, 4, None, seed=5)
    cu = torch.tensor([0, total], dtype=torch.int32)
    A = chunk_scaled_dot_kkt_fwd(k, beta, g, cu_seqlens=cu)
    _assert_close(A, _kkt_oracle(k, beta, g, cu), FP32_TOL)
    _assert_kkt_structure(A, 1, cu, total)


def test_kkt_short_sequence_less_than_bt():
    """T < BT must stay one partial chunk: no shrink of BT, only rows/cols < T are written."""
    total = 13
    k, _, beta = _make_kv(total, seed=6, dtype=torch.float32)
    g = _make_g(total, 4, None, seed=7)
    cu = torch.tensor([0, total], dtype=torch.int32)
    A = chunk_scaled_dot_kkt_fwd(k, beta, g, cu_seqlens=cu)
    assert A.shape == (1, total, 4, BT)
    _assert_close(A, _kkt_oracle(k, beta, g, cu), FP32_TOL)
    _assert_kkt_structure(A, 1, cu, total)
    assert torch.count_nonzero(A[:, :, :, total:]) == 0


def test_kkt_tail_of_exactly_one_token():
    """T = BT + 1: the second chunk has a single row, so it has no strictly-lower entry at all."""
    total = BT + 1
    k, _, beta = _make_kv(total, seed=8)
    g = _make_g(total, 4, None, seed=9)
    cu = torch.tensor([0, total], dtype=torch.int32)
    A = chunk_scaled_dot_kkt_fwd(k, beta, g, cu_seqlens=cu)
    _assert_close(A, _kkt_oracle(k, beta, g, cu), KKT_TOL)
    _assert_kkt_structure(A, 1, cu, total)
    assert torch.count_nonzero(A[0, BT]) == 0  # single-token chunk: entire row is zero


def test_kkt_multi_chunk_batch_two():
    total, batch = 3 * BT + 7, 2
    k, _, beta = _make_kv(total, batch=batch, seed=10)
    g = _make_g(total, 4, None, batch=batch, seed=11)
    cu = torch.tensor([0, total], dtype=torch.int32)
    A = chunk_scaled_dot_kkt_fwd(k, beta, g, cu_seqlens=cu)
    assert A.shape == (batch, total, 4, BT)
    _assert_close(A, _kkt_oracle(k, beta, g, cu), KKT_TOL)
    _assert_kkt_structure(A, batch, cu, total)


# ==================================================================================== kkt: varlen
def test_kkt_varlen_two_sequences():
    """cu = [0, 37, 64]: chunk boundaries restart at the second sequence start."""
    total = BT
    cu = torch.tensor([0, 37, 64], dtype=torch.int32)
    k, _, beta = _make_kv(total, seed=12)
    g = _make_g(total, 4, cu, seed=13)
    A = chunk_scaled_dot_kkt_fwd(k, beta, g, cu_seqlens=cu)
    _assert_close(A, _kkt_oracle(k, beta, g, cu), KKT_TOL)
    _assert_kkt_structure(A, 1, cu, total)
    # each sequence must behave exactly like a dense call on its own slice
    assert torch.equal(chunk_scaled_dot_kkt_fwd(k[:, :37].contiguous(), beta[:, :37].contiguous(),
                                                g[:, :37].contiguous(), cu_seqlens=torch.tensor([0, 37])),
                       A[:, :37])
    tail = slice(37, 64)
    assert torch.equal(chunk_scaled_dot_kkt_fwd(k[:, tail].contiguous(), beta[:, tail].contiguous(),
                                                g[:, tail].contiguous(), cu_seqlens=torch.tensor([0, 27])),
                       A[:, tail])


def test_kkt_varlen_multi_chunk_two_sequences():
    total = 164
    cu = torch.tensor([0, 100, 164], dtype=torch.int64)
    k, _, beta = _make_kv(total, seed=14)
    g = _make_g(total, 4, cu, seed=15)
    A = chunk_scaled_dot_kkt_fwd(k, beta, g, cu_seqlens=cu)
    _assert_close(A, _kkt_oracle(k, beta, g, cu), KKT_TOL)
    _assert_kkt_structure(A, 1, cu, total)


def test_kkt_varlen_uncovered_trailing_rows_stay_zero():
    """cu = [0, 5] with T = 16: rows 5..15 belong to no sequence and stay zero."""
    total, cu = 16, torch.tensor([0, 5], dtype=torch.int32)
    k, _, beta = _make_kv(total, seed=16)
    g = _make_g(total, 4, cu, seed=17)
    A = chunk_scaled_dot_kkt_fwd(k, beta, g, cu_seqlens=cu)
    assert torch.count_nonzero(A[0, 5:]) == 0
    _assert_close(A, _kkt_oracle(k, beta, g, cu), KKT_TOL)


# ==================================================================================== kkt: gating
def test_kkt_gqa_head_mapping():
    """HK=1, HV=4: every value head reads key head 0, so with equal beta/g all heads agree."""
    total = BT
    k, _, _ = _make_kv(total, hk=1, hv=4, seed=18)
    beta = torch.rand(1, total, 1).expand(1, total, 4)
    g = _make_g(total, 1, None, seed=19).expand(1, total, 4)
    cu = torch.tensor([0, total], dtype=torch.int32)
    A = chunk_scaled_dot_kkt_fwd(k, beta, g, cu_seqlens=cu)
    for h in range(1, 4):
        assert torch.equal(A[:, :, 0], A[:, :, h]), f"head {h} did not read key head 0"
    _assert_close(A, _kkt_oracle(k, beta, g, cu), KKT_TOL)


def test_kkt_beta_scales_query_rows_not_key_columns():
    """Beta must scale row (query) i, never column (key) j: contract §3.3."""
    total = BT
    k, _, beta = _make_kv(total, seed=20, dtype=torch.float32)
    g = _make_g(total, 4, None, seed=21)
    cu = torch.tensor([0, total], dtype=torch.int32)
    base = chunk_scaled_dot_kkt_fwd(k, beta, g, cu_seqlens=cu)

    beta2 = beta.clone()
    beta2[0, 1] *= 3.0
    scaled = chunk_scaled_dot_kkt_fwd(k, beta2, g, cu_seqlens=cu)
    # row 1 (its only lower entry is column 0) scales by 3 ...
    torch.testing.assert_close(scaled[0, 1], 3.0 * base[0, 1], rtol=1e-6, atol=0)
    # ... and no other row changes: beta_1 must not act as a column scale.
    assert torch.equal(scaled[0, 2], base[0, 2])
    torch.testing.assert_close(scaled[0, 3:], base[0, 3:], rtol=0, atol=0)


def test_kkt_g_none_disables_gating():
    total = BT
    k, _, beta = _make_kv(total, seed=22)
    cu = torch.tensor([0, total], dtype=torch.int32)
    g = _make_g(total, 4, None, seed=23)
    ungated = chunk_scaled_dot_kkt_fwd(k, beta, None, cu_seqlens=cu)
    zero_g = chunk_scaled_dot_kkt_fwd(k, beta, torch.zeros_like(g), cu_seqlens=cu)
    assert torch.equal(ungated, zero_g)
    _assert_close(ungated, _kkt_oracle(k, beta, torch.zeros_like(g), cu), KKT_TOL)


def _max_gate_gap(g, head=0):
    """Largest ``g_i - g_j`` over one head: the upper-triangle exponent that must be clamped."""
    column = g[0, :, head]
    return float((column[:, None] - column[None, :]).max())


def test_kkt_safe_exp_keeps_huge_decay_finite():
    """Decay of 40/token puts exp(+1000) in the upper triangle; only the clamp keeps this finite."""
    total = BT
    k, _, beta = _make_kv(total, seed=24, dtype=torch.float32)
    g = _make_g(total, 4, None, decay=40.0, seed=25)
    assert _max_gate_gap(g) > 800.0  # the input really is in the overflow regime
    cu = torch.tensor([0, total], dtype=torch.int32)
    A = chunk_scaled_dot_kkt_fwd(k, beta, g, cu_seqlens=cu)
    assert torch.isfinite(A).all()
    _assert_kkt_structure(A, 1, cu, total)
    ref = _kkt_oracle(k, beta, g, cu)  # contains inf/nan above the diagonal; compare the lower part
    chunks = _dense_chunks(total)
    _assert_close(_lower_entries(A, chunks), _lower_entries(ref, chunks), FP32_TOL)


# ==================================================================================== kkt: plumbing
def test_kkt_output_dtype_and_fp64_exactness():
    total = BT
    k, _, beta = _make_kv(total, seed=26)
    g = _make_g(total, 4, None, seed=27)
    cu = torch.tensor([0, total], dtype=torch.int32)
    ref = _kkt_oracle(k, beta, g, cu)

    A_bf = chunk_scaled_dot_kkt_fwd(k, beta, g, cu_seqlens=cu, output_dtype=torch.bfloat16)
    assert A_bf.dtype is torch.bfloat16
    _assert_close(A_bf, ref, 5e-2)  # bf16 keeps ~8 mantissa bits

    k64, beta64, g64 = k.double(), beta.double(), g.double()
    A64 = chunk_scaled_dot_kkt_fwd(k64, beta64, g64, cu_seqlens=cu)
    assert A64.dtype is torch.float32  # the default still wins over the input dtype
    _assert_close(A64, _kkt_oracle(k64, beta64, g64, cu), 1e-6)  # fp32 store rounds the fp64 math

    A64d = chunk_scaled_dot_kkt_fwd(k64, beta64, g64, cu_seqlens=cu, output_dtype=torch.float64)
    assert A64d.dtype is torch.float64
    _assert_close(A64d, ref, 1e-12)


def test_kkt_non_contiguous_inputs():
    total, batch = BT + 20, 2
    k, _, beta = _make_kv(total, batch=batch, seed=28)
    g = _make_g(total, 4, None, batch=batch, seed=29)
    cu = torch.tensor([0, total], dtype=torch.int32)
    expected = chunk_scaled_dot_kkt_fwd(k, beta, g, cu_seqlens=cu)

    # last-dim slice of wider buffers (stride on the feature axis is 2*d)
    kw = torch.zeros(batch, total, 2, 16, dtype=k.dtype)
    kw[..., :8] = k
    bw = torch.zeros(batch, total, 8, dtype=beta.dtype)
    bw[..., :4] = beta
    gw = torch.zeros(batch, total, 8, dtype=g.dtype)
    gw[..., :4] = g
    ks, bs, gs = kw[..., :8], bw[..., :4], gw[..., :4]
    assert not ks.is_contiguous() and not bs.is_contiguous() and not gs.is_contiguous()
    assert torch.equal(chunk_scaled_dot_kkt_fwd(ks, bs, gs, cu_seqlens=cu), expected)

    # token-axis slice of a longer buffer (B=2 so the batch stride really is stretched)
    kl = torch.zeros(batch, 2 * total, 2, 8, dtype=k.dtype)
    kl[:, :total] = k
    bl = torch.zeros(batch, 2 * total, 4, dtype=beta.dtype)
    bl[:, :total] = beta
    gl = torch.zeros(batch, 2 * total, 4, dtype=g.dtype)
    gl[:, :total] = g
    kt, bt_, gt = kl[:, :total], bl[:, :total], gl[:, :total]
    assert not kt.is_contiguous() and not bt_.is_contiguous() and not gt.is_contiguous()
    assert torch.equal(chunk_scaled_dot_kkt_fwd(kt, bt_, gt, cu_seqlens=cu), expected)


def test_kkt_chunk_size_parameter_is_honoured():
    """The sibling exposes ``chunk_size`` (default 64); passing 32 must chunk differently."""
    total, cs = 80, 32
    k, _, beta = _make_kv(total, seed=30, dtype=torch.float32)
    g = _make_g(total, 4, None, chunk_size=cs, seed=31)
    cu = torch.tensor([0, total], dtype=torch.int32)
    A = chunk_scaled_dot_kkt_fwd(k, beta, g, cu_seqlens=cu, chunk_size=cs)
    assert A.shape == (1, total, 4, cs)
    _assert_close(A, _kkt_oracle(k, beta, g, cu, chunk_size=cs), FP32_TOL)
    _assert_kkt_structure(A, 1, cu, total, chunk_size=cs)


def test_kkt_chunk_indices_derived_matches_explicit_and_ignored_when_dense():
    total = 70
    cu = torch.tensor([0, 37, 70], dtype=torch.int32)
    k, _, beta = _make_kv(total, seed=32)
    g = _make_g(total, 4, cu, seed=33)
    auto = chunk_scaled_dot_kkt_fwd(k, beta, g, cu_seqlens=cu)
    ci = prepare_chunk_indices(cu, BT)
    assert ci.tolist() == [[0, 0], [1, 0]]
    assert torch.equal(auto, chunk_scaled_dot_kkt_fwd(k, beta, g, cu_seqlens=cu, chunk_indices=ci))
    assert torch.equal(auto, chunk_scaled_dot_kkt_fwd(k, beta, g, cu_seqlens=cu, chunk_indices=ci.flip(0)))

    # chunk_indices is inert without cu_seqlens (IS_VARLEN keys off cu_seqlens alone)
    dense = chunk_scaled_dot_kkt_fwd(k, beta, g, cu_seqlens=None)
    bogus = torch.zeros((3, 2), dtype=torch.int64)
    assert torch.equal(dense, chunk_scaled_dot_kkt_fwd(k, beta, g, cu_seqlens=None, chunk_indices=bogus))


# =============================================================================== kkt: negative controls
def test_negative_control_nonstrict_triangle_would_fail():
    """A ``>=`` mask adds ``beta_i * (k_i . k_i)`` on the diagonal: the tests must reject it."""
    total = BT
    k, _, beta = _make_kv(total, hk=2, hv=4, seed=34, dtype=torch.float32)
    g = _make_g(total, 4, None, seed=35)
    cu = torch.tensor([0, total], dtype=torch.int32)
    A = chunk_scaled_dot_kkt_fwd(k, beta, g, cu_seqlens=cu)
    _assert_kkt_structure(A, 1, cu, total)

    wrong = _kkt_oracle(k, beta, g, cu).clone()
    kb = k.double()[:, :, torch.arange(4) // 2]  # [1, T, H, K]
    diag_terms = (kb * kb).sum(-1)  # [1, T, H] = k_i . k_i
    for h in range(4):
        wrong[0, :, h, :total] += torch.diag(beta[0, :, h].double() * diag_terms[0, :, h])
    assert _rel_err(A, wrong) > 1e-2, "a non-strict triangle would have passed the oracle comparison"


def test_negative_control_multiply_mask_produces_nan():
    """``value * mask`` on the positive upper-triangle exponents gives NaN; ``where`` must not."""
    total = BT
    k, _, beta = _make_kv(total, hk=2, hv=4, seed=36, dtype=torch.float32)
    g = _make_g(total, 4, None, decay=40.0, seed=37)
    assert _max_gate_gap(g) > 800.0
    cu = torch.tensor([0, total], dtype=torch.int32)
    A = chunk_scaled_dot_kkt_fwd(k, beta, g, cu_seqlens=cu)
    assert not torch.isnan(A).any()
    assert torch.isfinite(A).all()

    # The plausible-but-wrong variant: build the full matrix, then multiply by a 0/1 triangle.
    kmap = torch.arange(4) // 2
    kb = k[0][:, kmap].double()
    kk = torch.einsum("ihd,jhd->hij", kb, kb)  # [H, T, T]
    gd = g[0].double()
    gd = gd[:, None, :] - gd[None, :, :]  # [i, j, H]
    val = torch.exp(gd.permute(2, 0, 1)) * kk
    val = val * beta[0].double().permute(1, 0).unsqueeze(-1)
    tri = torch.tril(torch.ones(total, total, dtype=torch.float64), -1)
    wrong = val * tri.unsqueeze(0)  # inf * 0 -> nan
    assert torch.isnan(wrong).any(), "expected the multiply-mask variant to produce NaN"


# =================================================================================== kkt: malformed
def test_kkt_rejects_malformed_input():
    beta4 = torch.zeros(1, 8, 2, 1)
    with pytest.raises(ValueError, match="beta must be"):
        chunk_scaled_dot_kkt_fwd(torch.zeros(1, 8, 2, 4), beta4)
    with pytest.raises(ValueError, match="positive multiple"):
        chunk_scaled_dot_kkt_fwd(torch.zeros(1, 8, 3, 4), torch.zeros(1, 8, 4))
    with pytest.raises(ValueError, match="start at 0"):
        chunk_scaled_dot_kkt_fwd(
            torch.zeros(1, 8, 1, 4), torch.zeros(1, 8, 4), cu_seqlens=torch.tensor([1, 8])
        )
    with pytest.raises(ValueError, match="chunk_size"):
        chunk_scaled_dot_kkt_fwd(torch.zeros(1, 8, 1, 4), torch.zeros(1, 8, 4), chunk_size=0)
    # a chunk_indices entry that points past cu_seqlens is an error, not a silent skip
    with pytest.raises(ValueError, match="references sequence"):
        chunk_scaled_dot_kkt_fwd(
            torch.zeros(1, 8, 1, 4),
            torch.zeros(1, 8, 4),
            cu_seqlens=torch.tensor([0, 8]),
            chunk_indices=torch.tensor([[2, 0]]),
        )


# ======================================================================================= wu: chain
def _chain(k, v, beta, g, cu_seqlens, chunk_size=BT):
    """kkt -> solve_tril(bf16) -> recompute_w_u, i.e. the production slice of chunk.py:41-60."""
    a_kkt = chunk_scaled_dot_kkt_fwd(
        k, beta, g, cu_seqlens=cu_seqlens, chunk_size=chunk_size
    )
    a_inv = solve_tril(a_kkt, cu_seqlens=cu_seqlens, output_dtype=k.dtype)
    w, u = recompute_w_u_fwd(k, v, beta, g, a_inv, cu_seqlens)
    return a_kkt, a_inv, w, u


def test_wu_chain_matches_oracle_single_chunk_bf16():
    total = BT
    cu = torch.tensor([0, total], dtype=torch.int32)
    k, v, beta = _make_kv(total, seed=40)
    g = _make_g(total, 4, None, seed=41)
    a_kkt, a_inv, w, u = _chain(k, v, beta, g, cu)
    assert a_inv.dtype is torch.bfloat16  # the caller's output_dtype=k.dtype, chunk.py:50
    assert w.dtype is torch.bfloat16 and u.dtype is torch.bfloat16
    assert w.shape == (1, total, 4, 8)  # value-head count, key width
    assert u.shape == (1, total, 4, 8)
    w_ref, u_ref, _ = _wu_oracle(k, v, beta, g, cu, a_kkt)
    _assert_close(w, w_ref, WU_TOL)
    _assert_close(u, u_ref, WU_TOL)


def test_wu_chain_varlen_two_sequences():
    total = BT
    cu = torch.tensor([0, 37, 64], dtype=torch.int32)
    k, v, beta = _make_kv(total, seed=42)
    g = _make_g(total, 4, cu, seed=43)
    a_kkt, _, w, u = _chain(k, v, beta, g, cu)
    w_ref, u_ref, _ = _wu_oracle(k, v, beta, g, cu, a_kkt)
    _assert_close(w, w_ref, WU_TOL)
    _assert_close(u, u_ref, WU_TOL)
    # rows beyond the last sequence end (none here) and per-sequence independence
    assert torch.equal(
        recompute_w_u_fwd(
            k[:, 37:].contiguous(),
            v[:, 37:].contiguous(),
            beta[:, 37:].contiguous(),
            g[:, 37:].contiguous(),
            solve_tril(
                chunk_scaled_dot_kkt_fwd(
                    k[:, 37:].contiguous(),
                    beta[:, 37:].contiguous(),
                    g[:, 37:].contiguous(),
                    cu_seqlens=torch.tensor([0, 27]),
                ),
                cu_seqlens=torch.tensor([0, 27]),
                output_dtype=k.dtype,
            ),
            torch.tensor([0, 27]),
        )[0],
        w[:, 37:],
    )


def test_wu_chain_multi_chunk_batch_two():
    total, batch = 3 * BT + 7, 2
    cu = torch.tensor([0, total], dtype=torch.int32)
    k, v, beta = _make_kv(total, batch=batch, seed=44)
    g = _make_g(total, 4, None, batch=batch, seed=45)
    a_kkt, _, w, u = _chain(k, v, beta, g, cu)
    w_ref, u_ref, _ = _wu_oracle(k, v, beta, g, cu, a_kkt)
    _assert_close(w, w_ref, WU_TOL)
    _assert_close(u, u_ref, WU_TOL)
    assert w.shape == (batch, total, 4, 8) and u.shape == (batch, total, 4, 8)


def test_wu_exact_math_fp32():
    """With A_inv supplied and fp32 operands, w/u must match the oracle to fp32 precision."""
    total = BT + 9
    cu = torch.tensor([0, total], dtype=torch.int32)
    k, v, beta = _make_kv(total, seed=46, dtype=torch.float32)
    g = _make_g(total, 4, None, seed=47)
    a_kkt = chunk_scaled_dot_kkt_fwd(k, beta, g, cu_seqlens=cu)
    # independent fp64 inverse of the same A, rounded to fp32: isolates this operator's math
    _, _, a_inv = _wu_oracle(k, v, beta, g, cu, a_kkt)
    w, u = recompute_w_u_fwd(k, v, beta, g, a_inv, cu)
    w_ref, u_ref, _ = _wu_oracle(k, v, beta, g, cu, a_kkt)
    _assert_close(w, w_ref, 1e-5)
    _assert_close(u, u_ref, 1e-5)


def test_wu_bt_is_read_from_A_not_hardcoded():
    """``BT = A.shape[-1]`` [sglang wy_fast.py:122]: a 32-wide A must chunk at 32, not at 64."""
    total, bt = 80, 32
    cu = torch.tensor([0, total], dtype=torch.int32)
    k, v, beta = _make_kv(total, seed=64)
    g = _make_g(total, 4, None, chunk_size=bt, seed=65)
    a_kkt = chunk_scaled_dot_kkt_fwd(k, beta, g, cu_seqlens=cu, chunk_size=bt)
    a_inv = solve_tril(a_kkt, cu_seqlens=cu, output_dtype=k.dtype)
    assert a_inv.shape[-1] == bt
    w, u = recompute_w_u_fwd(k, v, beta, g, a_inv, cu)
    w_ref, u_ref, _ = _wu_oracle(k, v, beta, g, cu, a_kkt, chunk_size=bt)
    _assert_close(w, w_ref, WU_TOL)
    _assert_close(u, u_ref, WU_TOL)


def test_wu_fp64_is_exact():
    total = 2 * BT + 5
    cu = torch.tensor([0, total], dtype=torch.int32)
    k, v, beta = _make_kv(total, seed=48, dtype=torch.float64)
    g = _make_g(total, 4, None, seed=49, dtype=torch.float64)
    a_kkt = chunk_scaled_dot_kkt_fwd(k, beta, g, cu_seqlens=cu, output_dtype=torch.float64)
    a_inv = solve_tril(a_kkt, cu_seqlens=cu, output_dtype=torch.float64)
    w, u = recompute_w_u_fwd(k, v, beta, g, a_inv, cu)
    w_ref, u_ref, _ = _wu_oracle(k, v, beta, g, cu, a_kkt)
    _assert_close(w, w_ref, 1e-12)
    _assert_close(u, u_ref, 1e-12)


# ======================================================================================== wu: math
def test_wu_identity_inv_closed_form_bf16():
    """A_inv = I: w = k*beta*exp(g), u = v*beta, with the GQA map and the pre-dot downcast."""
    total, hk, hv, d, v_dim = 2 * BT + 5, 2, 4, 8, 6
    k, v, beta = _make_kv(total, hk=hk, hv=hv, d=d, v_dim=v_dim, seed=50)
    g = _make_g(total, hv, None, seed=51)
    cu = torch.tensor([0, total], dtype=torch.int32)
    a_inv = _identity_inv(1, total, hv)
    w, u = recompute_w_u_fwd(k, v, beta, g, a_inv, cu)
    assert w.shape == (1, total, hv, d) and u.shape == (1, total, hv, v_dim)
    assert w.dtype is torch.bfloat16 and u.dtype is torch.bfloat16

    kb = k[:, :, torch.arange(hv) // (hv // hk)]  # [B, T, H, K]
    w_exp = (kb.float() * beta.float().unsqueeze(-1) * torch.exp(g).unsqueeze(-1)).to(k.dtype)
    u_exp = (v.float() * beta.float().unsqueeze(-1)).to(v.dtype)
    torch.testing.assert_close(w, w_exp, rtol=0, atol=0)
    torch.testing.assert_close(u, u_exp, rtol=0, atol=0)


def test_wu_rounds_operands_before_the_dots():
    """Contract: ``(v*beta)`` and ``(k*beta*exp(g))`` are cast to bf16 *before* each dot
    [sglang wy_fast.py:84,106].

    This is a dtype-contract probe, not a correctness oracle: ``v = k = 1``, ``g = 0``, ``beta``
    values sit between bf16 grid points, and ``A_inv = [[1, 0], [-1, 1]]`` turns the two roundings
    into a cancellation (0.0078125 vs 0.001) instead of hiding them under the bf16 output quantum.
    """
    total = 2
    k = torch.ones(1, total, 1, 1, dtype=torch.bfloat16)
    v = torch.ones(1, total, 1, 1, dtype=torch.bfloat16)
    beta = torch.tensor([[[1.3], [1.301]]], dtype=torch.float32)
    g = torch.zeros(1, total, 1, dtype=torch.float32)
    a_inv = torch.zeros(1, total, 1, BT, dtype=torch.float32)
    a_inv[0, 0, 0, 0] = 1.0
    a_inv[0, 1, 0, 0] = -1.0
    a_inv[0, 1, 0, 1] = 1.0
    w, u = recompute_w_u_fwd(k, v, beta, g, a_inv, torch.tensor([0, total], dtype=torch.int32))

    a = a_inv[0, :, 0, :total].double()  # [T, T]
    rounded = beta[0, :, 0].to(torch.bfloat16).double()  # v == k == 1, exp(g) == 1
    plain = beta[0, :, 0].double()
    expected_rounded = (a @ rounded).to(torch.bfloat16)
    expected_plain = (a @ plain).to(torch.bfloat16)
    assert not torch.equal(expected_rounded, expected_plain), "probe does not separate the cases"
    torch.testing.assert_close(u[0, :, 0, 0], expected_rounded, rtol=0, atol=0)
    torch.testing.assert_close(w[0, :, 0, 0], expected_rounded, rtol=0, atol=0)


def test_wu_mixed_input_dtypes_flow_independently():
    """wy_fast takes w's dtype from k and u's from v; chunk.py:212's equal-dtype assert is upstream."""
    total, d, v_dim = BT, 8, 6
    k, v, beta = _make_kv(total, d=d, v_dim=v_dim, seed=52)
    v = v.float()  # k/beta bf16, v fp32
    g = _make_g(total, 4, None, seed=53)
    cu = torch.tensor([0, total], dtype=torch.int32)
    w, u = recompute_w_u_fwd(k, v, beta, g, _identity_inv(1, total, 4), cu)
    assert w.dtype is torch.bfloat16
    assert u.dtype is torch.float32
    kb = k[:, :, torch.arange(4) // 2]
    w_exp = (kb.float() * beta.float().unsqueeze(-1) * torch.exp(g).unsqueeze(-1)).to(k.dtype)
    u_exp = v.float() * beta.float().unsqueeze(-1)
    torch.testing.assert_close(w, w_exp, rtol=0, atol=0)
    torch.testing.assert_close(u, u_exp, rtol=0, atol=0)


def test_wu_u_is_not_gated_and_w_is():
    """u must not depend on g at all; w must depend on it strongly.  A swapped gate fails this."""
    total = BT
    cu = torch.tensor([0, total], dtype=torch.int32)
    k, v, beta = _make_kv(total, seed=54)
    g = _make_g(total, 4, None, seed=55)
    a_inv = solve_tril(
        chunk_scaled_dot_kkt_fwd(k, beta, g, cu_seqlens=cu), cu_seqlens=cu, output_dtype=k.dtype
    )
    w, u = recompute_w_u_fwd(k, v, beta, g, a_inv, cu)
    w0, u0 = recompute_w_u_fwd(k, v, beta, torch.zeros_like(g), a_inv, cu)
    assert torch.equal(u, u0), "u picked up the gate; wy_fast must not gate u"
    assert _rel_err(w, w0) > 1e-2, "w ignored exp(g)"


def test_wu_partial_tail_first_row_closed_form():
    """A chunk's first row has A_inv row = e_0, so T = BT + 1's last row is closed-form."""
    total = BT + 1
    cu = torch.tensor([0, total], dtype=torch.int32)
    k, v, beta = _make_kv(total, seed=56)
    g = _make_g(total, 4, None, seed=57)
    _, _, w, u = _chain(k, v, beta, g, cu)
    kb = k[:, :, torch.arange(4) // 2]
    w_exp = (kb.float() * beta.float().unsqueeze(-1) * torch.exp(g).unsqueeze(-1)).to(k.dtype)
    u_exp = (v.float() * beta.float().unsqueeze(-1)).to(v.dtype)
    torch.testing.assert_close(w[0, BT], w_exp[0, BT], rtol=0, atol=0)
    torch.testing.assert_close(u[0, BT], u_exp[0, BT], rtol=0, atol=0)


def test_wu_uncovered_trailing_rows_stay_zero():
    total, cu = 16, torch.tensor([0, 5], dtype=torch.int32)
    k, v, beta = _make_kv(total, seed=58)
    g = _make_g(total, 4, cu, seed=59)
    _, _, w, u = _chain(k, v, beta, g, cu)
    assert torch.count_nonzero(w[0, 5:]) == 0
    assert torch.count_nonzero(u[0, 5:]) == 0


def test_wu_non_contiguous_inputs():
    total = BT + 11
    cu = torch.tensor([0, total], dtype=torch.int32)
    k, v, beta = _make_kv(total, seed=60)
    g = _make_g(total, 4, None, seed=61)
    a_inv = _identity_inv(1, total, 4)
    expected_w, expected_u = recompute_w_u_fwd(k, v, beta, g, a_inv, cu)

    vw = torch.zeros(1, total, 4, 12, dtype=v.dtype)
    vw[..., :8] = v
    vs = vw[..., :8]
    aw = torch.zeros(1, total, 4, 2 * BT, dtype=a_inv.dtype)
    aw[..., :BT] = a_inv
    as_ = aw[..., :BT]
    gw = torch.zeros(1, total, 8, dtype=g.dtype)
    gw[..., :4] = g
    gs = gw[..., :4]
    bt_w = torch.zeros(1, total, 8, dtype=beta.dtype)
    bt_w[..., :4] = beta
    bs = bt_w[..., :4]
    assert not vs.is_contiguous() and not as_.is_contiguous() and not gs.is_contiguous()
    w, u = recompute_w_u_fwd(k, vs, bs, gs, as_, cu)
    assert torch.equal(w, expected_w) and torch.equal(u, expected_u)


# ================================================================================ wu: plumbing
def test_wu_chunk_indices_derived_matches_explicit_and_ignored_when_dense():
    total = 70
    cu = torch.tensor([0, 37, 70], dtype=torch.int32)
    k, v, beta = _make_kv(total, seed=62)
    g = _make_g(total, 4, cu, seed=63)
    a_inv = _identity_inv(1, total, 4)
    auto = recompute_w_u_fwd(k, v, beta, g, a_inv, cu)
    ci = prepare_chunk_indices(cu, BT)
    explicit = recompute_w_u_fwd(k, v, beta, g, a_inv, cu, chunk_indices=ci)
    flipped = recompute_w_u_fwd(k, v, beta, g, a_inv, cu, chunk_indices=ci.flip(0))
    assert torch.equal(auto[0], explicit[0]) and torch.equal(auto[1], explicit[1])
    assert torch.equal(auto[0], flipped[0]) and torch.equal(auto[1], flipped[1])

    dense = recompute_w_u_fwd(k, v, beta, g, a_inv, None)
    bogus = torch.zeros((3, 2), dtype=torch.int64)
    stray = recompute_w_u_fwd(k, v, beta, g, a_inv, None, chunk_indices=bogus)
    assert torch.equal(dense[0], stray[0]) and torch.equal(dense[1], stray[1])


def test_wu_rejects_malformed_input():
    k = torch.zeros(1, 8, 1, 4)
    v = torch.zeros(1, 8, 4, 4)
    beta = torch.zeros(1, 8, 4)
    g = torch.zeros(1, 8, 4)
    a = torch.zeros(1, 8, 4, 16)
    with pytest.raises(ValueError, match=r"v must be \[B, T, H, V\]"):
        recompute_w_u_fwd(k, v[0], beta, g, a, None)
    with pytest.raises(ValueError, match="beta must be"):
        recompute_w_u_fwd(k, v, beta[0], g, a, None)
    with pytest.raises(ValueError, match="A must be"):
        recompute_w_u_fwd(k, v, beta, g, a[0], None)
    with pytest.raises(ValueError, match="same \[B, T\]"):
        recompute_w_u_fwd(k, v[:, :7], beta[:, :7], g[:, :7], a[:, :7], None)
    with pytest.raises(ValueError, match="positive multiple"):
        recompute_w_u_fwd(torch.zeros(1, 8, 3, 4), v, beta, g, a, None)
