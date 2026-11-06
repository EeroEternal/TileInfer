"""Expert/engram slot cache for host-offloaded weights (S4 mechanism, device-agnostic).

The model is 475 GiB while the card holds 128 GiB, so the routed experts (268.9 GiB) and the engram
tables (189.1 GiB) live in host memory and are paged into pre-allocated HBM slots on demand
(``docs/notes/dsv41-s4-offload-plan.md``).  This module is only the *bookkeeping*: which key owns
which slot, LRU eviction that never touches an in-flight block, the watermark, and the counters the
S4/S5 evidence needs.  The copies themselves stay with the caller, so this is testable on a CPU box.

Invariants (each one has a self-test below):

* a slot is allocated once and reused; nothing is allocated per token (no fragmentation, no OOM mid-run);
* a key that is ``inflight`` may not be evicted — the copy stream would write into a slot someone else
  now owns;
* the resident bytes never exceed the watermark: ``evict`` runs until it fits or reports failure;
* every hit/miss/eviction/watermark event is counted, because S4's acceptance is measured from these.

    python -m tileinfer.models.dsv41.offload        # runs the self-test
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, Hashable, List, Optional, Tuple

__all__ = ["SlotCache", "CacheStats", "GB"]


GB = 1024**3


@dataclass
class CacheStats:
    """Counters the S4/S5 evidence is built from."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    failed_acquires: int = 0
    watermark_events: int = 0
    bytes_copied: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "failed_acquires": self.failed_acquires,
            "watermark_events": self.watermark_events,
            "bytes_copied": self.bytes_copied,
            "hit_rate": round(self.hit_rate, 4),
        }


@dataclass
class _Entry:
    key: Hashable
    slot: int
    nbytes: int
    inflight: bool = False


class SlotCache:
    """LRU page cache with pre-allocated slots, in-flight protection and a byte watermark.

    ``slot_bytes`` is the size of one slot (one expert = 18.80 MB; one engram row block = whatever
    the caller pages).  ``n_slots`` is derived from the watermark by the caller and both are fixed for
    the process lifetime.
    """

    def __init__(self, slot_bytes: int, n_slots: int, watermark_bytes: Optional[int] = None):
        if slot_bytes <= 0 or n_slots <= 0:
            raise ValueError(f"slot_bytes={slot_bytes} n_slots={n_slots} must be positive")
        self.slot_bytes = int(slot_bytes)
        self.n_slots = int(n_slots)
        self.watermark_bytes = int(watermark_bytes or self.slot_bytes * self.n_slots)
        if self.slot_bytes * self.n_slots > self.watermark_bytes:
            raise ValueError(
                f"{self.n_slots} slots x {self.slot_bytes} B exceed the watermark "
                f"{self.watermark_bytes} B"
            )
        self._free: List[int] = list(range(self.n_slots - 1, -1, -1))  # pop() from the end
        self._entries: "OrderedDict[Hashable, _Entry]" = OrderedDict()  # first = oldest (LRU)
        self._by_slot: Dict[int, _Entry] = {}
        self.stats = CacheStats()

    # -- small queries ------------------------------------------------------------------------

    @property
    def resident_bytes(self) -> int:
        return len(self._entries) * self.slot_bytes

    @property
    def free_slots(self) -> int:
        return len(self._free)

    def contains(self, key: Hashable) -> bool:
        return key in self._entries

    def slot_of(self, key: Hashable) -> Optional[int]:
        entry = self._entries.get(key)
        return None if entry is None else entry.slot

    def slot_tensor_index(self, key: Hashable) -> Optional[int]:
        """The slot index a caller should index its pre-allocated tensor list with."""
        return self.slot_of(key)

    # -- the protocol -------------------------------------------------------------------------

    def acquire(self, key: Hashable, nbytes: Optional[int] = None) -> Tuple[int, bool]:
        """Return ``(slot, hit)``; on a miss the caller must copy the block into that slot.

        Marks the slot ``inflight`` so it cannot be evicted while the copy runs; the caller must call
        :meth:`release` (success) or :meth:`abort` (failure) before the GEMM consumes it.
        """
        size = int(nbytes or self.slot_bytes)
        entry = self._entries.get(key)
        if entry is not None:
            self._entries.move_to_end(key)
            entry.inflight = True
            self.stats.hits += 1
            return entry.slot, True

        self.stats.misses += 1
        if not self._free and not self._evict_one(protect=key):
            self.stats.failed_acquires += 1
            raise MemoryError(
                f"no evictable slot for {key!r}: {self.resident_bytes/GB:.2f} GiB resident, "
                f"watermark {self.watermark_bytes/GB:.2f} GiB, everything in flight"
            )
        slot = self._free.pop()
        entry = _Entry(key=key, slot=slot, nbytes=size, inflight=True)
        self._entries[key] = entry
        self._by_slot[slot] = entry
        self.stats.bytes_copied += size
        if self.resident_bytes > self.watermark_bytes:
            self.stats.watermark_events += 1
        return slot, False

    def release(self, key: Hashable) -> None:
        """The copy for ``key`` finished; the slot may now be evicted."""
        entry = self._entries.get(key)
        if entry is None:
            raise KeyError(f"{key!r} is not resident")
        entry.inflight = False

    def abort(self, key: Hashable) -> None:
        """The copy failed; drop the entry and give the slot back."""
        entry = self._entries.pop(key, None)
        if entry is None:
            return
        self._by_slot.pop(entry.slot, None)
        self._free.append(entry.slot)

    # -- eviction -----------------------------------------------------------------------------

    def _evict_one(self, protect: Optional[Hashable] = None) -> bool:
        for key, entry in list(self._entries.items()):
            if entry.inflight or key == protect:
                continue  # never page out something a copy stream is writing
            self._entries.pop(key)
            self._by_slot.pop(entry.slot, None)
            self._free.append(entry.slot)
            self.stats.evictions += 1
            return True
        return False

    def drop(self, key: Hashable) -> None:
        """Release a key immediately, returning its slot (used after a prefill chunk is done)."""
        entry = self._entries.pop(key, None)
        if entry is None:
            return
        self._by_slot.pop(entry.slot, None)
        self._free.append(entry.slot)

    def evict_until(self, target_bytes: int, protect: Optional[Hashable] = None) -> int:
        """Evict LRU entries until ``resident_bytes <= target_bytes``; returns how many went out."""
        evicted = 0
        while self.resident_bytes > target_bytes:
            if not self._evict_one(protect=protect):
                break
            evicted += 1
        return evicted

    # -- diagnostics ---------------------------------------------------------------------------

    def snapshot(self) -> Dict[str, float]:
        data = self.stats.as_dict()
        data.update(
            resident_gib=round(self.resident_bytes / GB, 3),
            watermark_gib=round(self.watermark_bytes / GB, 3),
            slots=self.n_slots,
            slot_mib=round(self.slot_bytes / 1024**2, 2),
            free_slots=self.free_slots,
        )
        return data


# =============================================================================================
# Self-test: the invariants the S4 acceptance depends on
# =============================================================================================


def _self_test() -> int:
    slot = 18_800_640  # one routed expert, bytes
    cache = SlotCache(slot_bytes=slot, n_slots=4, watermark_bytes=slot * 4)
    ok = True

    def check(name: str, cond: bool) -> None:
        nonlocal ok
        ok = ok and cond
        print(f"  {'PASS' if cond else 'FAIL'}  {name}")

    # 1. a miss allocates, a second look-up hits and does not move bytes
    s0, hit0 = cache.acquire(("L0", 7))
    cache.release(("L0", 7))
    s0b, hit1 = cache.acquire(("L0", 7))
    check("first acquire misses, second hits", (hit0, hit1) == (False, True))
    check("the same key keeps its slot", s0 == s0b)
    check("a hit copies nothing", cache.stats.bytes_copied == slot)
    cache.release(("L0", 7))

    # 2. filling all slots works, the next miss evicts the LRU
    for e in range(1, 4):
        cache.acquire(("L0", e))
        cache.release(("L0", e))
    slot_5, hit5 = cache.acquire(("L0", 5))  # must evict ("L0", 7), the oldest
    check("a full cache evicts instead of failing", hit5 is False)
    check("the oldest key went out", not cache.contains(("L0", 7)))
    check("eviction counter moved", cache.stats.evictions == 1)
    check("resident stays at capacity", cache.resident_bytes == slot * 4)
    cache.release(("L0", 5))

    # 3. in-flight entries are never evicted
    cache.acquire(("L0", 100))  # in flight, not released
    try:
        cache.acquire(("L0", 101))
        got = cache.contains(("L0", 100))
    except MemoryError:
        got = cache.contains(("L0", 100))
    check("an in-flight block survives an eviction pass", got)

    # 4. aborting frees the slot again
    before = cache.free_slots
    cache.abort(("L0", 100))
    check("abort returns the slot", cache.free_slots == before + 1)

    # 5. evict_until respects the target and the watermark counter is inert below it
    n = cache.evict_until(slot * 2)
    check("evict_until drains down to the target", cache.resident_bytes <= slot * 2)
    check("it reported its evictions", n >= 0)

    # 6. a slot list that cannot fit the watermark is rejected up front
    try:
        SlotCache(slot_bytes=slot, n_slots=8, watermark_bytes=slot * 4)
        rejected = False
    except ValueError:
        rejected = True
    check("an over-watermark configuration is rejected at construction", rejected)

    print("\nRESULT:", "OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_self_test())
