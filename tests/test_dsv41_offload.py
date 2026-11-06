"""Host-offload slot cache (S4): the invariants the paging mechanism depends on.

``SlotCache`` only does bookkeeping (pre-allocated slots, LRU, in-flight protection, watermark), so
it is fully testable on a CPU box -- which is the point: the device run then only has to prove the
*copies* are right, not the cache policy.
"""

from __future__ import annotations

import pytest

from tileinfer.models.dsv41.offload import GB, CacheStats, SlotCache

SLOT = 18_800_640  # one routed expert, bytes


def make(n_slots: int = 4) -> SlotCache:
    return SlotCache(slot_bytes=SLOT, n_slots=n_slots, watermark_bytes=SLOT * n_slots)


def test_miss_then_hit_reuses_the_slot():
    cache = make()
    slot, hit = cache.acquire(("L0", 7))
    assert hit is False and cache.stats.bytes_copied == SLOT
    cache.release(("L0", 7))
    again, hit2 = cache.acquire(("L0", 7))
    assert hit2 is True and again == slot
    assert cache.stats.bytes_copied == SLOT, "a hit must not copy"
    assert cache.stats.hit_rate == 0.5


def test_full_cache_evicts_the_lru_and_never_exceeds_capacity():
    cache = make(3)
    for e in range(3):
        cache.acquire(("L0", e))
        cache.release(("L0", e))
    cache.acquire(("L0", 9))
    assert not cache.contains(("L0", 0)), "the oldest key must be the one evicted"
    assert cache.stats.evictions == 1
    assert cache.resident_bytes == SLOT * 3
    cache.release(("L0", 9))


def test_inflight_blocks_are_not_evicted():
    cache = make(2)
    cache.acquire(("L0", 1))  # in flight (no release)
    cache.acquire(("L0", 2))
    cache.release(("L0", 2))
    cache.acquire(("L0", 3))  # must evict 2, not the in-flight 1
    assert cache.contains(("L0", 1))
    assert not cache.contains(("L0", 2))


def test_all_in_flight_raises_instead_of_corrupting():
    cache = make(2)
    cache.acquire(("L0", 1))
    cache.acquire(("L0", 2))
    with pytest.raises(MemoryError):
        cache.acquire(("L0", 3))
    assert cache.stats.failed_acquires == 1


def test_abort_returns_the_slot():
    cache = make(2)
    free_before = cache.free_slots
    cache.acquire(("L0", 1))
    cache.abort(("L0", 1))
    assert cache.free_slots == free_before
    assert not cache.contains(("L0", 1))


def test_evict_until_and_snapshot():
    cache = make(4)
    for e in range(4):
        cache.acquire(("L0", e))
        cache.release(("L0", e))
    assert cache.evict_until(SLOT) >= 1
    assert cache.resident_bytes <= SLOT
    snap = cache.snapshot()
    assert snap["resident_gib"] <= snap["watermark_gib"]
    assert snap["slots"] == 4 and snap["evictions"] >= 1


def test_over_watermark_configuration_is_rejected():
    with pytest.raises(ValueError):
        SlotCache(slot_bytes=SLOT, n_slots=8, watermark_bytes=SLOT * 4)


def test_stats_are_plain_data():
    stats = CacheStats(hits=3, misses=1, bytes_copied=4 * SLOT)
    assert stats.hit_rate == 0.75
    payload = stats.as_dict()
    assert payload["hits"] == 3 and payload["hit_rate"] == 0.75
    assert GB == 1024**3
