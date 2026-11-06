"""批处理窗口槽位 `window_topk_idxs_batched` 的单测（§24 的 B 步，不需要 NPU）。

核心性质：批处理版本必须与"逐请求单独调用"**逐位一致**——这是后续 A（共享 batched KV）
与 C（batched sparse attention）能安全改写的前提。
"""

from __future__ import annotations

import torch

from tileinfer.models.dsv41.layer import window_slots_for, window_topk_idxs_batched


def test_matches_per_request_calls():
    win = 8
    positions = [0, 1, 7, 8, 23, 100]
    got = window_topk_idxs_batched(win, positions)
    assert got.shape == (len(positions), win)
    for i, p in enumerate(positions):
        assert got[i].tolist() == window_slots_for(p, win).tolist(), p


def test_positions_beyond_window_wrap_correctly():
    win = 4
    got = window_topk_idxs_batched(win, [5, 6, 7])
    # 每个位置都是满环（无 -1），且相邻行滚动一位
    for row in got.tolist():
        assert -1 not in row
    assert got[0].tolist() == [2, 3, 0, 1]
    assert got[1].tolist() == [3, 0, 1, 2]
    assert got[2].tolist() == [0, 1, 2, 3]


def test_early_positions_are_masked_per_row():
    win = 6
    got = window_topk_idxs_batched(win, [0, 2])
    assert got[0].tolist() == [-1, -1, -1, -1, -1, 0]
    assert got[1].tolist() == [-1, -1, -1, 0, 1, 2]


def test_dtype_and_empty():
    got = window_topk_idxs_batched(4, [1, 2])
    assert got.dtype == torch.int32
    empty = window_topk_idxs_batched(4, [])
    assert empty.shape == (0, 4)


def test_single_position_equals_existing_helper():
    win = 8
    one = window_topk_idxs_batched(win, [11])
    assert one[0].tolist() == window_slots_for(11, win).tolist()


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
