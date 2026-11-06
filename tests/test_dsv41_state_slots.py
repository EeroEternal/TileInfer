"""`StateSlots` 的离线单测（P3 第 2 步）。"""

from __future__ import annotations

import pytest

from tileinfer.vllm.models.dsv41_state_slots import StateSlots


class FakeState:
    def __init__(self, req_id: str) -> None:
        self.req_id = req_id


def make(max_slots=2, calls=None):
    calls = calls if calls is not None else []

    def factory(req_id):
        calls.append(req_id)
        return FakeState(req_id)

    return StateSlots(factory, max_slots=max_slots), calls


def test_acquire_creates_once_and_reuses():
    slots, calls = make()
    a = slots.acquire("r1")
    b = slots.acquire("r1")
    assert a is b
    assert calls == ["r1"]
    assert slots.created == 1
    assert slots.reused == 1


def test_acquire_returns_none_when_full():
    slots, calls = make(max_slots=2)
    assert slots.acquire("r1") is not None
    assert slots.acquire("r2") is not None
    assert slots.acquire("r3") is None      # 达到上限，拒绝
    assert calls == ["r1", "r2"]
    # 已存在的请求仍可取
    assert slots.acquire("r1") is not None


def test_release_frees_capacity():
    slots, _ = make(max_slots=1)
    assert slots.acquire("r1") is not None
    assert slots.acquire("r2") is None
    assert slots.release("r1") is True
    assert slots.release("r1") is False     # 重复释放
    assert slots.acquire("r2") is not None


def test_prune_drops_stale_requests():
    slots, _ = make(max_slots=4)
    for rid in ("a", "b", "c"):
        slots.acquire(rid)
    dead = slots.prune(["b"])
    assert sorted(dead) == ["a", "c"]
    assert slots.ids() == ["b"]
    assert len(slots) == 1


def test_get_does_not_create():
    slots, calls = make()
    assert slots.get("nope") is None
    assert calls == []
    assert "nope" not in slots


def test_clear_and_max_slots_validation():
    slots, _ = make()
    slots.acquire("x")
    slots.clear()
    assert len(slots) == 0
    with pytest.raises(ValueError):
        StateSlots(lambda r: None, max_slots=0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
