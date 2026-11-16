"""CPU oracle tests for the torch FLA ops (l2norm / chunk-local cumsum)."""

from __future__ import annotations

import pytest
import torch

from tileinfer.vllm.ops.gdn import chunk_local_cumsum, l2norm_fwd


def test_l2norm_matches_the_closed_form_with_eps_inside_the_sqrt():
    x = torch.randn(4, 8, dtype=torch.float32)
    out = l2norm_fwd(x, eps=1e-6)
    expected = x / torch.sqrt((x * x).sum(-1, keepdim=True) + 1e-6)
    assert torch.allclose(out, expected, atol=1e-6)


def test_l2norm_eps_convention_differs_from_norm_clamping_on_small_rows():
    # The trap: 1/sqrt(sum + eps) is NOT x/max(norm, eps). A near-zero row exposes the difference.
    x = torch.full((1, 4), 1e-3, dtype=torch.float32)
    eps = 1e-6
    ours = l2norm_fwd(x, eps=eps)
    clamp_form = x / torch.clamp(x.norm(dim=-1, keepdim=True), min=eps)
    assert not torch.allclose(ours, clamp_form, atol=1e-4)
    # With eps inside the sqrt the result is *not* exactly unit norm - it is sum/(sum+eps):
    ratio = (x * x).sum(-1) / ((x * x).sum(-1) + eps)
    assert torch.allclose(ours.norm(dim=-1) ** 2, ratio, atol=1e-6)


def test_l2norm_keeps_and_casts_dtype():
    x = torch.randn(3, 16)
    assert l2norm_fwd(x).dtype is torch.float32
    bf = x.to(torch.bfloat16)
    assert l2norm_fwd(bf).dtype is torch.bfloat16
    assert l2norm_fwd(x, output_dtype=torch.bfloat16).dtype is torch.bfloat16


def test_l2norm_rejects_bad_input():
    with pytest.raises(ValueError):
        l2norm_fwd(torch.tensor(1.0))


@pytest.mark.parametrize("chunk,length", [(4, 8), (4, 10), (1, 5), (8, 8), (8, 3)])
def test_chunk_local_cumsum_resets_each_chunk(chunk, length):
    g = torch.arange(length, dtype=torch.float32).reshape(length, 1) + 1.0
    out = chunk_local_cumsum(g, chunk_size=chunk)
    expected = []
    for start in range(0, length, chunk):
        block = g[start : start + chunk, 0]
        expected.extend(torch.cumsum(block, 0).tolist())
    assert out[:, 0].tolist() == pytest.approx(expected)


def test_chunk_local_cumsum_reverse_is_the_suffix_sum():
    g = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
    rev = chunk_local_cumsum(g, chunk_size=3, reverse=True)
    # chunk 0 = [1,2,3] -> suffix [6,5,3]; chunk 1 = [4,5] -> suffix [9,5]
    assert rev[:, 0].tolist() == [6.0, 5.0, 3.0, 9.0, 5.0]


def test_chunk_local_cumsum_ragged_resets_at_sequence_boundaries():
    g = torch.ones(6, 1)
    cu = torch.tensor([0, 2, 6], dtype=torch.int32)
    out = chunk_local_cumsum(g, chunk_size=4, cu_seqlens=cu)
    # first sequence is 2 tokens (one chunk) -> [1,2]; second is 4 tokens -> [1,2,3,4]
    assert out[:, 0].tolist() == [1.0, 2.0, 1.0, 2.0, 3.0, 4.0]


def test_chunk_local_cumsum_rejects_bad_varlen_and_chunk():
    with pytest.raises(ValueError, match="chunk_size"):
        chunk_local_cumsum(torch.ones(4, 1), chunk_size=0)
    with pytest.raises(ValueError, match="cu_seqlens must span"):
        chunk_local_cumsum(torch.ones(4, 1), chunk_size=2, cu_seqlens=torch.tensor([0, 3]))
