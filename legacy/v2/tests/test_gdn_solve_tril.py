"""CPU oracle tests for the torch port of FLA's ``solve_tril``
(``tileinfer/vllm/ops/fla/solve_tril.py``).

The oracle is built independently of the implementation: it materialises each ``BT x BT`` chunk
matrix explicitly as ``I + tril(A_block, -1)`` in float64 and calls ``torch.linalg.inv``.  The
implementation instead uses ``torch.linalg.solve_triangular`` on a unit-lower system, so the two
only agree if the *semantics* agree - not the code.  ``torch.linalg.solve`` is cross-checked against
``torch.linalg.inv`` in ``test_solve_and_inv_oracles_agree``.

Semantics under test (see the implementation module docstring for the quoted kernel lines):
out = (I + A)^-1 inside every chunk, input diagonal/upper triangle masked away, rows a program does
not cover keep the zeros of the ``zeros_like`` buffer, varlen chunks restart at every
``cu_seqlens`` boundary.
"""

from __future__ import annotations

import pytest
import torch

from tileinfer.vllm.ops.fla.solve_tril import prepare_chunk_indices, solve_tril

BT_SIZES = (16, 32, 64)
SCALE = 0.25  # keeps the lower-triangular systems well conditioned so fp32-vs-fp64 is meaningful


# --------------------------------------------------------------------------------------- builders
def _strict_lower_mask(bt: int) -> torch.Tensor:
    return torch.tril(torch.ones(bt, bt, dtype=torch.bool), diagonal=-1)


def _dense_chunks(total: int, bt: int) -> list[tuple[int, int]]:
    """``(t0, length)`` of every chunk of a run of ``total`` tokens, straight from the host NT."""
    return [(t0, min(bt, total - t0)) for t0 in range(0, total, bt)]


def _varlen_chunks(cu_seqlens, bt: int) -> list[tuple[int, int]]:
    """Same, but chunk boundaries restart at every sequence start (the IS_VARLEN branch)."""
    bounds = [int(v) for v in cu_seqlens]
    chunks = []
    for bos, eos in zip(bounds[:-1], bounds[1:]):
        t0 = bos
        while t0 < eos:
            chunks.append((t0, min(bt, eos - t0)))
            t0 += bt
    return chunks


def _make_A(batch_chunks, total, heads, bt, seed=0, dtype=torch.float32):
    """Strictly-lower ``[B, T, H, BT]`` filled with the given ``(t0, length)`` chunks per batch."""
    generator = torch.Generator().manual_seed(seed)
    a = torch.zeros(len(batch_chunks), total, heads, bt, dtype=dtype)
    mask = _strict_lower_mask(bt)
    for i_b, chunks in enumerate(batch_chunks):
        for t0, length in chunks:
            values = torch.randn(length, heads, bt, generator=generator) * SCALE
            a[i_b, t0 : t0 + length] = values * mask[:length].unsqueeze(1)
    return a


def _dense_blocks(batch, total, bt):
    return [(i_b, t0, length) for i_b in range(batch) for t0, length in _dense_chunks(total, bt)]


def _varlen_blocks(batch, cu_seqlens, bt):
    chunks = _varlen_chunks(cu_seqlens, bt)
    return [(i_b, t0, length) for i_b in range(batch) for t0, length in chunks]


# ----------------------------------------------------------------------------------------- oracle
def _oracle(A, blocks, sign=+1.0):
    """Float64 ``(I + sign*A)^{-1}`` per ``(i_b, t0, length)`` block, zeros elsewhere."""
    source = A.to(torch.float64)
    out = torch.zeros_like(source)
    for i_b, t0, length in blocks:
        block = torch.tril(source[i_b, t0 : t0 + length, :, :length].permute(1, 0, 2), diagonal=-1)
        eye = torch.eye(length, dtype=torch.float64)
        inverse = torch.linalg.inv(eye + sign * block)  # [H, l, l]
        out[i_b, t0 : t0 + length, :, :length] = inverse.permute(1, 0, 2)
    return out


def _blockdiag16_oracle(A, blocks):
    """The plausible-but-wrong answer: invert only the 16x16 diagonal blocks, skip the merge."""
    source = A.to(torch.float64)
    out = torch.zeros_like(source)
    for i_b, t0, length in blocks:
        for start in range(t0, t0 + length, 16):
            n = min(16, t0 + length - start)
            block = torch.tril(source[i_b, start : start + n, :, start : start + n].permute(1, 0, 2), -1)
            eye = torch.eye(n, dtype=torch.float64)
            out[i_b, start : start + n, :, start : start + n] = torch.linalg.inv(
                eye + block
            ).permute(1, 0, 2)
    return out


def _rel_error(actual, expected):
    """Max abs deviation normalised by the oracle's largest magnitude (scale-free tolerance)."""
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    diff = (actual.to(torch.float64) - expected).abs().max().item()
    return diff / max(expected.abs().max().item(), 1.0)


def _assert_matches(actual, expected, tol):
    error = _rel_error(actual, expected)
    assert error <= tol, f"relative error {error:.3e} exceeds {tol:.3e}"


# ------------------------------------------------------------------------------------------ tests
@pytest.mark.parametrize("bt", [16, 32])
def test_single_chunk_shorter_than_bt(bt):
    """One partial chunk: the leading ``length x length`` block is inverted, the rest stays zero."""
    total, heads, length = 5, 2, 5
    A = _make_A([_dense_chunks(total, bt)], total, heads, bt)
    got = solve_tril(A)
    assert got.dtype is torch.float32 and got.shape == A.shape
    _assert_matches(got, _oracle(A, _dense_blocks(1, total, bt)), 1e-5)
    block = got[0, :length, 0, :length]
    assert torch.allclose(torch.diagonal(block), torch.ones(length))
    assert torch.count_nonzero(torch.triu(block, diagonal=1)) == 0
    assert torch.count_nonzero(got[0, :length, 0, length:]) == 0


@pytest.mark.parametrize("bt", BT_SIZES)
def test_exact_multiple_of_bt(bt):
    """T == 2*BT with B=2, H=2: every chunk is full and every chunk is independent."""
    total, heads, batch = 2 * bt, 2, 2
    A = _make_A([_dense_chunks(total, bt)] * batch, total, heads, bt, seed=7)
    got = solve_tril(A)
    _assert_matches(got, _oracle(A, _dense_blocks(batch, total, bt)), 1e-5)


@pytest.mark.parametrize("length", [1, 4, 16, 17, 31, 33])
def test_partial_trailing_chunk_of_each_bt(length):
    """Ragged tail length, including lengths that straddle a 16-row sub-block (17, 31, 33)."""
    for bt in BT_SIZES:
        total = 2 * bt + length  # two full chunks plus a partial one
        A = _make_A([_dense_chunks(total, bt)], total, 2, bt, seed=length)
        got = solve_tril(A)
        _assert_matches(got, _oracle(A, _dense_blocks(1, total, bt)), 1e-5)
        # strictly-upper part of the partial block stays zero
        assert torch.count_nonzero(got[0, 2 * bt :, :, length:]) == 0


@pytest.mark.parametrize("bt", BT_SIZES)
def test_block_inverse_residual_is_identity(bt):
    """(I + A_block) @ Ai_block == I for every chunk: pins the block structure on its own."""
    total, heads = 3 * bt, 2
    A = _make_A([_dense_chunks(total, bt)], total, heads, bt, seed=11)
    got = solve_tril(A)
    for t0, length in _dense_chunks(total, bt):
        lhs = A[0, t0 : t0 + length, :, :length].to(torch.float64).permute(1, 0, 2)
        eye = torch.eye(length, dtype=torch.float64)
        product = (eye + torch.tril(lhs, -1)) @ got[0, t0 : t0 + length, :, :length].to(
            torch.float64
        ).permute(1, 0, 2)
        assert (product - eye).abs().max().item() < 1e-5


def test_bt64_merges_off_diagonal_blocks():
    """A 64x64 chunk is one inverse, not four 16x16 ones: the gap is O(1), not noise."""
    bt, total, heads = 64, 64, 2
    A = _make_A([_dense_chunks(total, bt)], total, heads, bt, seed=3)
    got = solve_tril(A)
    blocks = _dense_blocks(1, total, bt)
    expected = _oracle(A, blocks)
    wrong_diag = _blockdiag16_oracle(A, blocks)
    _assert_matches(got, expected, 1e-5)
    assert _rel_error(got, wrong_diag) > 1e-2, "output looks like a 16x16-only block inverse"
    # the merged (off-diagonal) part is exactly where the two answers diverge
    assert _rel_error(got[0, 16:32, :, 0:16], wrong_diag[0, 16:32, :, 0:16]) > 1e-2


def test_ragged_two_sequences():
    """cu = [0, 5, 17] at BT=16: chunk boundaries restart per sequence, no cross-talk."""
    cu = torch.tensor([0, 5, 17], dtype=torch.int32)
    bt, heads, total = 16, 2, 17
    blocks = _varlen_blocks(1, cu, bt)
    A = _make_A(
        [[(t0, length) for _, t0, length in blocks]], total, heads, bt, seed=5
    )
    got = solve_tril(A, cu_seqlens=cu)
    _assert_matches(got, _oracle(A, blocks), 1e-5)
    # sequence 2 is 12 tokens long -> columns 12..15 of its rows are strictly upper -> zeros
    assert torch.count_nonzero(got[0, 5:17, :, 12:]) == 0
    # each sequence must be solved as if it were alone: dense calls on the two slices agree
    assert torch.equal(solve_tril(A[:, :5].contiguous()), got[:, :5])
    assert torch.equal(solve_tril(A[:, 5:17].contiguous()), got[:, 5:17])


def test_ragged_multiple_chunks_and_zero_length_sequence():
    """BT=32, cu = [0, 40, 40, 71]: 40 -> 2 chunks, an empty sequence, then 31 -> 1 partial chunk."""
    cu = torch.tensor([0, 40, 40, 71], dtype=torch.int64)
    bt, heads, total = 32, 2, 71
    blocks = _varlen_blocks(1, cu, bt)
    A = _make_A([[(t0, length) for _, t0, length in blocks]], total, heads, bt, seed=9)
    got = solve_tril(A, cu_seqlens=cu)
    _assert_matches(got, _oracle(A, blocks), 1e-5)
    # the third sequence is 31 of 32 tokens: its last column is above the diagonal -> zero
    assert torch.count_nonzero(got[0, 40:71, :, 31:]) == 0


def test_ragged_batch_of_two_shares_the_row_layout():
    """Varlen with B=2: both batch rows use the same cu_seqlens, each with its own values."""
    cu = torch.tensor([0, 3, 20], dtype=torch.int32)
    bt, heads, total = 16, 3, 20
    blocks = _varlen_blocks(2, cu, bt)
    per_batch = [(t0, length) for _, t0, length in _varlen_blocks(1, cu, bt)]
    A = _make_A([per_batch] * 2, total, heads, bt, seed=4)
    got = solve_tril(A, cu_seqlens=cu)
    assert got.shape == A.shape
    _assert_matches(got, _oracle(A, blocks), 1e-5)


def test_uncovered_trailing_rows_stay_zero_not_identity():
    """cu = [0, 5] with T = 16: rows 5..15 belong to no sequence, so no program writes them.

    The host allocates ``Ai = torch.zeros_like(A)`` and the stores use ``boundary_check``, so the
    result is *zero* there - an implementation that emits the identity would be wrong.
    """
    cu = torch.tensor([0, 5], dtype=torch.int32)
    bt, heads, total = 16, 2, 16
    A = _make_A([_dense_chunks(5, bt)], total, heads, bt, seed=31)
    got = solve_tril(A, cu_seqlens=cu)
    assert torch.count_nonzero(got[0, 5:]) == 0
    _assert_matches(got, _oracle(A, _varlen_blocks(1, cu, bt)), 1e-5)


def test_explicit_chunk_indices_and_their_order_irrelevance():
    """Passing chunk_indices must reproduce the derived path, and their order cannot matter."""
    cu = torch.tensor([0, 5, 37], dtype=torch.int64)
    bt, heads, total = 16, 2, 37
    derived = prepare_chunk_indices(cu, bt)
    assert derived.tolist() == [[0, 0], [1, 0], [1, 1]]
    blocks = _varlen_blocks(1, cu, bt)
    A = _make_A([[(t0, length) for _, t0, length in blocks]], total, heads, bt, seed=6)
    auto = solve_tril(A, cu_seqlens=cu)
    assert torch.equal(auto, solve_tril(A, cu_seqlens=cu, chunk_indices=derived))
    assert torch.equal(auto, solve_tril(A, cu_seqlens=cu, chunk_indices=derived.flip(0)))
    _assert_matches(auto, _oracle(A, blocks), 1e-5)


def test_chunk_indices_are_ignored_without_cu_seqlens():
    """Upstream keys IS_VARLEN off cu_seqlens alone, so a stray chunk_indices must be inert."""
    bt, total = 16, 32
    A = _make_A([_dense_chunks(total, bt)], total, 2, bt, seed=2)
    bogus = torch.zeros((3, 2), dtype=torch.int64)
    assert torch.equal(solve_tril(A), solve_tril(A, chunk_indices=bogus))


def test_non_contiguous_input_matches_its_contiguous_copy():
    """The Triton kernel addresses memory by constants; this port indexes logically."""
    bt, total, heads, batch = 16, 40, 2, 2
    base = _make_A([_dense_chunks(total, bt)] * batch, total, heads, bt, seed=13)
    expected = solve_tril(base)
    _assert_matches(expected, _oracle(base, _dense_blocks(batch, total, bt)), 1e-5)

    # last-dim slice of a wider buffer: stride(BT axis) is 2*BT, not BT
    wide = torch.zeros(batch, total, heads, 2 * bt)
    wide[..., :bt] = base
    sliced = wide[..., :bt]
    assert not sliced.is_contiguous() and sliced.shape == base.shape
    assert torch.equal(solve_tril(sliced), expected)

    # token-axis slice of a longer buffer: stride(T axis) spans the padded rows
    long = torch.zeros(batch, 2 * total, heads, bt)
    long[:, :total] = base
    strided_t = long[:, :total]
    assert not strided_t.is_contiguous()
    assert torch.equal(solve_tril(strided_t), expected)

    # batch slice of a larger batch
    pair = torch.zeros(2 * batch, total, heads, bt)
    pair[:batch] = base
    assert torch.equal(solve_tril(pair[:batch]), expected)


@pytest.mark.parametrize("bt", [16, 64])
def test_diagonal_and_upper_triangle_are_masked(bt):
    """Any value on or above the diagonal of the input must not change the result.

    For well-formed input (``A.triu() == 0``) this is a no-op, i.e. it agrees with the kernels.
    It is deliberately *stricter* than the Triton substitution loop, which reads full 16-wide rows
    (``tl.load(A + ... + o_i + offset)``) and would leak such entries; upstream therefore documents
    ``A.triu() == 0`` as a requirement, while this port enforces it by masking.
    """
    total, heads = 2 * bt, 2
    A = _make_A([_dense_chunks(total, bt)], total, heads, bt, seed=17)
    polluted = A.clone()
    for t0 in range(0, total, bt):
        length = min(bt, total - t0)
        polluted[0, t0 : t0 + length, :, :] += (
            1e3 * torch.triu(torch.ones(length, length), diagonal=0).unsqueeze(1)
        )
    assert torch.count_nonzero(torch.triu(polluted[0, :, 0], diagonal=0)) > 0
    assert torch.equal(solve_tril(polluted), solve_tril(A))


def test_wrong_sign_convention_is_rejected():
    """The most plausible misreading: inverse of (I - A). It differs by O(1), not by noise."""
    bt, total, heads = 16, 3 * 16, 2
    A = _make_A([_dense_chunks(total, bt)], total, heads, bt, seed=19)
    got = solve_tril(A)
    blocks = _dense_blocks(1, total, bt)
    _assert_matches(got, _oracle(A, blocks, sign=+1.0), 1e-5)
    assert _rel_error(got, _oracle(A, blocks, sign=-1.0)) > 1e-2
    # second, independent statement of the same fact: (I + A) @ out == I but (I - A) @ out != I
    for t0, length in _dense_chunks(total, bt):
        lhs = A[0, t0 : t0 + length, :, :length].to(torch.float64).permute(1, 0, 2)
        block = got[0, t0 : t0 + length, :, :length].to(torch.float64).permute(1, 0, 2)
        eye = torch.eye(length, dtype=torch.float64)
        plus = ((eye + torch.tril(lhs, -1)) @ block - eye).abs().max().item()
        minus = ((eye - torch.tril(lhs, -1)) @ block - eye).abs().max().item()
        assert plus < 1e-5 < minus


def test_dtype_preservation():
    """Default output is float32 (upstream default), None means "same as input", fp64 stays exact."""
    bt, total, heads = 16, 32, 2
    blocks = _dense_blocks(1, total, bt)
    A32 = _make_A([_dense_chunks(total, bt)], total, heads, bt, seed=23)
    assert solve_tril(A32).dtype is torch.float32
    assert solve_tril(A32, output_dtype=None).dtype is torch.float32

    A64 = A32.to(torch.float64)
    exact = solve_tril(A64, output_dtype=None)
    assert exact.dtype is torch.float64
    _assert_matches(exact, _oracle(A32, blocks), 1e-12)  # fp64 in -> fp64 oracle, no slack
    assert solve_tril(A64).dtype is torch.float32  # the upstream default wins over the input dtype

    out_bf = solve_tril(A32.to(torch.bfloat16), output_dtype=None)
    assert out_bf.dtype is torch.bfloat16
    _assert_matches(out_bf, _oracle(A32, blocks), 5e-2)  # bf16 keeps ~8 mantissa bits
    assert solve_tril(A32, output_dtype=torch.bfloat16).dtype is torch.bfloat16


def test_solve_and_inv_oracles_agree():
    """Cross-check the oracle itself: inv(I + A) == solve(I + A, I) in float64."""
    bt, total, heads = 32, 64, 3
    A = _make_A([_dense_chunks(total, bt)], total, heads, bt, seed=29)
    for t0, length in _dense_chunks(total, bt):
        block = A[0, t0 : t0 + length, :, :length].to(torch.float64).permute(1, 0, 2)
        eye = torch.eye(length, dtype=torch.float64)
        system = eye + torch.tril(block, -1)
        assert (torch.linalg.inv(system) - torch.linalg.solve(system, eye)).abs().max() < 1e-12


def test_rejects_malformed_input():
    with pytest.raises(AssertionError, match="BT must be"):
        solve_tril(torch.zeros(1, 8, 2, 48))
    with pytest.raises(ValueError, match=r"\[B, T, H, BT\]"):
        solve_tril(torch.zeros(1, 8, 2))
    A = _make_A([_dense_chunks(16, 16)], 16, 1, 16)
    with pytest.raises(ValueError, match="cu_seqlens must start at 0"):
        solve_tril(A, cu_seqlens=torch.tensor([1, 16], dtype=torch.int32))
    with pytest.raises(ValueError, match="non-decreasing"):
        solve_tril(A, cu_seqlens=torch.tensor([0, 9, 4, 16], dtype=torch.int32))
    with pytest.raises(ValueError, match="span <="):
        solve_tril(A, cu_seqlens=torch.tensor([0, 32], dtype=torch.int32))
    with pytest.raises(ValueError, match="references sequence"):
        solve_tril(
            A,
            cu_seqlens=torch.tensor([0, 16], dtype=torch.int32),
            chunk_indices=torch.tensor([[3, 0]], dtype=torch.int64),
        )


def test_degenerate_shapes_return_zeros():
    assert solve_tril(torch.zeros(0, 8, 2, 16)).shape == (0, 8, 2, 16)
    assert torch.count_nonzero(solve_tril(torch.zeros(1, 0, 2, 16))) == 0
    out = solve_tril(torch.zeros(1, 8, 0, 16))
    assert out.shape == (1, 8, 0, 16) and torch.count_nonzero(out) == 0


def test_prepare_chunk_indices_layout():
    cu = torch.tensor([0, 5, 37], dtype=torch.int32)
    assert prepare_chunk_indices(cu, 16).tolist() == [[0, 0], [1, 0], [1, 1]]
    assert prepare_chunk_indices(torch.tensor([0, 0, 16], dtype=torch.int32), 16).tolist() == [[1, 0]]
    assert prepare_chunk_indices(None, 16) is None
    with pytest.raises(ValueError, match="chunk_size"):
        prepare_chunk_indices(cu, 0)
