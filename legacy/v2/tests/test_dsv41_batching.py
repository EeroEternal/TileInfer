"""`dsv41_batching.request_slices` 的离线单测（P3 第 1 步）。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tileinfer.vllm.models.dsv41_batching import is_single, request_slices, run_batched


def meta(**kw):
    return SimpleNamespace(**kw)


def test_none_metadata_falls_back_to_single_sequence():
    assert request_slices(None, 5) is None
    assert is_single(None, 5) is True


def test_query_start_loc_two_requests():
    # 两个请求：长度 3 和 2，共 5 个 token
    sl = request_slices(meta(query_start_loc=[0, 3, 5]), 5)
    assert sl == [(0, 3), (3, 5)]
    assert is_single(sl, 5) is False


def test_query_start_loc_single_request_is_equivalent_to_single():
    sl = request_slices(meta(query_start_loc=[0, 5]), 5)
    assert sl == [(0, 5)]
    assert is_single(sl, 5) is True


def test_query_start_loc_four_requests_decode_one_token_each():
    sl = request_slices(meta(query_start_loc=[0, 1, 2, 3, 4]), 4)
    assert sl == [(0, 1), (1, 2), (2, 3), (3, 4)]


def test_nested_metadata_is_supported():
    inner = meta(query_start_loc=[0, 2, 5])
    sl = request_slices(meta(attn_metadata=inner), 5)
    assert sl == [(0, 2), (2, 5)]


def test_seq_lens_path_when_start_loc_missing():
    sl = request_slices(meta(seq_lens=[3, 2]), 5)
    assert sl == [(0, 3), (3, 5)]


def test_seq_lens_mismatch_is_not_guessed():
    # decode 阶段 seq_lens 是含历史的总长，与本批 token 数不一致时不能拿来切分
    assert request_slices(meta(seq_lens=[7, 9]), 2) is None


def test_empty_and_degenerate_inputs():
    assert request_slices(meta(query_start_loc=[0]), 5) is None
    assert request_slices(meta(query_start_loc=[0, 0]), 5) is None   # 全空片段
    assert request_slices(meta(query_start_loc=[0, 3, 2]), 5) is None  # 非单调
    assert request_slices(meta(query_start_loc=[0, 3, 5]), 0) is None


def test_slices_are_clamped_to_num_tokens():
    sl = request_slices(meta(query_start_loc=[0, 4, 9]), 5)
    assert sl == [(0, 4), (4, 5)]


def test_run_batched_uses_each_request_own_positions():
    """并发时各请求的位置互不相同；run_batched 必须传各自的 pos，而不是 positions[0]。"""
    ids = ["a0", "a1", "b0", "b1", "b2"]
    positions = [10, 11, 0, 1, 2]
    seen = []

    def forward_one(state, ids_slice, pos_slice):
        seen.append((state, list(ids_slice), list(pos_slice)))
        return [f"{state}:{i}" for i in ids_slice]

    per_req, last = run_batched([(0, 2), (2, 5)], ids, positions, lambda a, b: f"req{a}", forward_one)
    assert seen == [("req0", ["a0", "a1"], [10, 11]), ("req2", ["b0", "b1", "b2"], [0, 1, 2])]
    assert last == ["req0:a1", "req2:b2"]


def test_run_batched_raises_when_slots_exhausted():
    with pytest.raises(RuntimeError):
        run_batched([(0, 1), (1, 2)], ["x", "y"], [0, 0],
                    lambda a, b: None, lambda s, i, p: list(i))


def test_run_batched_single_slice_matches_whole_batch():
    per_req, last = run_batched([(0, 3)], ["x", "y", "z"], [0, 1, 2],
                                lambda a, b: "s", lambda s, i, p: list(i))
    assert per_req == [["x", "y", "z"]]
    assert last == ["z"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
