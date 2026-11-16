"""Device-free contract tests for the native prefill glue (§17 ②③).

The kernel itself is covered by ``test_tilelang_ascend950_prefill.py`` (needs an NPU).  What can be
checked without a device — and what has actually broken elsewhere — is the *host* half:

* the compile-variant ladder, because a wrong rung means either a recompile per prompt length or
  dead tiles the kernel is not allowed to see;
* the padded (dead) tiles the schedule hands the kernel: they must carry ``q_len = 0`` and
  ``split_counts = 1`` (a 0 there would look like a KV split and demand a merge kernel);
* the ragged -> rectangular query reshape, since the kernel indexes a rectangle per request;
* the per-request query lengths, which are the one place a bad ``query_start_loc`` could silently
  shift every row.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tileinfer.vllm.attn.prefill import (
    BLOCK_Q,
    PrefillPath,
    PrefillUnsupported,
    _query_lengths,
    prefill_tile_bucket,
)


def _path(num_heads=16, num_kv_heads=2, head_size=256) -> PrefillPath:
    return PrefillPath(num_heads, num_kv_heads, head_size, scale=head_size**-0.5)


# --------------------------------------------------------------------------------------------
# compile-variant ladder
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "live, expected",
    [
        (0, 0),
        (1, 8),
        (7, 8),
        (8, 8),
        (9, 16),
        (32, 32),
        (33, 64),
        (128, 128),
        (129, 256),
        (2048, 2048),
        # past the ladder: a multiple of the last rung, so the variant count stays bounded
        (2049, 4096),
        (5000, 6144),
    ],
)
def test_tile_bucket_rounds_up(live, expected):
    assert prefill_tile_bucket(live) == expected


def test_tile_bucket_at_512_tokens_is_32():
    """The shape the M1 exit criterion is measured at: 512 tokens, group 8, block_q 128."""
    rows = 512 * 8
    assert rows // BLOCK_Q == 32
    assert prefill_tile_bucket(32) == 32


# --------------------------------------------------------------------------------------------
# schedule padding
# --------------------------------------------------------------------------------------------


def test_exact_tiles_are_not_padded():
    path = _path()
    qo = [512]
    sched = path._schedule(qo, 1, torch.device("cpu"))
    assert sched.num_tiles == 32
    assert sched.q_lens.tolist() == [BLOCK_Q] * 32
    assert sched.seq_ids.tolist() == [0] * 32
    assert sched.needs_merge is False


def test_short_prefill_is_padded_with_dead_tiles():
    path = _path()
    # 100 tokens x group 8 = 800 flattened rows -> 7 live tiles, ladder rung 8
    sched = path._schedule([100], 1, torch.device("cpu"))
    assert sched.num_tiles == 8
    live = 100 * 8
    expected_lens = [BLOCK_Q] * (live // BLOCK_Q) + [live % BLOCK_Q]
    assert sched.q_lens.tolist() == expected_lens + [0]
    # a dead tile must not look like a KV split (needs_merge is (split_counts > 1).any())
    assert sched.split_counts.tolist()[-1] == 1
    assert sched.needs_merge is False
    assert sched.seq_ids.tolist() == [0] * 8


def test_ragged_requests_get_their_own_tiles():
    path = _path()
    sched = path._schedule([2, 1], 2, torch.device("cpu"))
    # request 0: 16 rows, request 1: 8 rows -> one tile each, padded to the rung
    assert sched.num_tiles == 8
    assert sched.seq_ids.tolist()[:2] == [0, 1]
    assert sched.q_lens.tolist()[:2] == [16, 8]
    assert sched.q_offsets.tolist()[:2] == [0, 0]
    assert sched.q_lens.tolist()[2:] == [0] * 6


# --------------------------------------------------------------------------------------------
# query reshape
# --------------------------------------------------------------------------------------------


def test_rectify_single_request_transposes_the_head_axis():
    path = _path(head_size=256)
    q3 = torch.randn(5, 16, 256)
    rect = path._rectify(q3, [5], 1, 5, 5)
    assert rect.shape == (1, 16, 5, 256)
    # (token, head, dim) -> (head, token, dim): even a batch of one cannot be a view
    assert not rect.is_contiguous() or rect.data_ptr() != q3.data_ptr()
    assert rect[0, :, 3].equal(q3[3])
    assert rect[0, 7, :, :].equal(q3[:, 7, :])


def test_rectify_ragged_places_rows_in_order():
    path = _path(head_size=256)
    q3 = torch.arange(3 * 16 * 256, dtype=torch.float32).reshape(3, 16, 256)
    rect = path._rectify(q3, [2, 1], 2, 2, 3)
    assert rect.shape == (2, 16, 2, 256)
    assert rect[0, :, 0].equal(q3[0])
    assert rect[0, :, 1].equal(q3[1])
    assert rect[1, :, 0].equal(q3[2])
    # rows past a request's live length are never addressed by a tile, but must stay finite
    assert torch.equal(rect[1, :, 1], torch.zeros(16, 256))


# --------------------------------------------------------------------------------------------
# per-request query lengths
# --------------------------------------------------------------------------------------------


def _meta(starts):
    return SimpleNamespace(
        query_start_loc_cpu=torch.tensor(starts, dtype=torch.int32),
        query_start_loc=torch.tensor(starts, dtype=torch.int32),
    )


def test_query_lengths_from_host_copy():
    assert _query_lengths(_meta([0, 2, 3]), 2, 3) == [2, 1]
    assert _query_lengths(_meta([0, 512]), 1, 512) == [512]


def test_query_lengths_reject_inconsistent_metadata():
    with pytest.raises(PrefillUnsupported, match="sum to"):
        _query_lengths(_meta([0, 2, 3]), 2, 4)
    with pytest.raises(PrefillUnsupported, match="non-positive"):
        _query_lengths(_meta([0, 2, 2]), 2, 2)
    with pytest.raises(PrefillUnsupported, match="too short|entries"):
        _query_lengths(_meta([0, 2]), 2, 2)


# --------------------------------------------------------------------------------------------
# capability gate
# --------------------------------------------------------------------------------------------


def test_store_unpacks_without_swapping_head_and_token():
    """The kernel returns head-major ``[batch, heads, qo, dim]``; a ``reshape`` would swap the axes."""
    path = _path(head_size=256)
    heads, dim, tokens = 16, 256, 4
    # make every (head, token) value unique so a swap cannot go unnoticed
    result = torch.arange(heads * tokens * dim, dtype=torch.float32).reshape(
        1, heads, tokens, dim
    )
    out = torch.zeros(tokens, heads * dim)
    path._store(result, [tokens], 1, out, tokens)
    for t in range(tokens):
        for h in range(heads):
            assert torch.equal(out[t, h * dim : (h + 1) * dim], result[0, h, t])


def test_store_ragged_matches_per_request_rows():
    path = _path(head_size=256)
    heads, dim = 16, 256
    qo = [3, 2]
    result = torch.zeros(2, heads, 3, dim)
    for b, length in enumerate(qo):
        for t in range(length):
            for h in range(heads):
                result[b, h, t] = b * 1000 + t * 10 + h
    out = torch.zeros(5, heads * dim)
    path._store(result, qo, 2, out, 5)
    assert float(out[0, 0]) == 0.0 and float(out[0, dim]) == 1.0
    assert float(out[3, 0]) == 1000.0 and float(out[4, dim]) == 1011.0


def test_head_dim_gate_matches_the_kernel():
    assert _path(head_size=256).supported is True
    assert _path(head_size=128).supported is True
    assert _path(head_size=192).supported is False
