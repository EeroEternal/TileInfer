"""`BatchedKV` 的离线单测（§24 的 A 步容器，不需要 NPU）。"""

from __future__ import annotations

import pytest
import torch

from tileinfer.models.dsv41.batched_kv import BatchedKV


def make(max_slots=4, window=6, dim=3):
    return BatchedKV(max_slots, window, dim, dtype=torch.float32)


def test_write_then_view_matches_per_request_state():
    kv = make()
    slots, positions = [2, 0], [5, 11]
    vals = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    kv.write_window(slots, positions, vals)

    view = kv.window_view(slots)
    assert view.shape == (2, 6, 3)
    assert view[0, 5 % 6].tolist() == [1.0, 2.0, 3.0]
    assert view[1, 11 % 6].tolist() == [4.0, 5.0, 6.0]


def test_views_are_copies_not_aliases():
    kv = make()
    kv.write_window([0], [0], torch.ones(1, 3))
    view = kv.window_view([0])
    view[0, 0] = torch.tensor([9.0, 9.0, 9.0])
    # 写副本不应影响容器（提醒调用方：写必须走 write_window）
    assert kv.window_view([0])[0, 0].tolist() == [1.0, 1.0, 1.0]


def test_slots_independent():
    kv = make()
    kv.write_window([1], [3], torch.full((1, 3), 7.0))
    # 其他槽位仍为零
    assert kv.window_view([0]).abs().sum().item() == 0.0
    assert kv.window_view([1])[0, 3].tolist() == [7.0, 7.0, 7.0]


def test_reset_clears():
    kv = make()
    kv.write_window([0, 1], [0, 1], torch.ones(2, 3))
    kv.reset()
    assert kv.window_kv.abs().sum().item() == 0.0


def test_validation_errors():
    kv = make()
    with pytest.raises(ValueError):
        kv.write_window([0, 1], [0], torch.ones(1, 3))          # 长度不一致
    with pytest.raises(ValueError):
        kv.write_window([0], [0], torch.ones(2, 3))             # kv 行数不一致
    with pytest.raises(ValueError):
        BatchedKV(0, 4, 2)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
