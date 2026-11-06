"""Plan cache and pre-warm: one compiled plan per ``(shape, bucket, pool capacity)``.

What a "plan" is
----------------
A TileInfer plan owns the tiling schedule, the workspace and the JIT'd kernel handle for one static
batch shape.  Building it is the *slow* part of the whole library (a TileLang compile is ~70-110 s
cold, a cache hit when warm), and it must never happen on a serving step - least of all inside a
captured graph.  So the serving layer keeps a process-wide cache and drives it in two ways:

* **pre-warm**, before capture: the first non-captured step (the engine's warm-up run) builds every
  bucket a deployment can reach, synchronously, so the captured region only ever *looks up*;
* **background build**, as a safety net: a shape the pre-warm did not cover compiles on a worker
  thread and the step that needed it falls back for a few steps instead of stalling the engine.

Two properties matter more than they look:

* **single-flight.**  An engine builds one attention layer per model layer, and every layer asks for
  the same plan on the same step.  Without the ``_building`` set, a 40-layer model would start 40
  compiles of the same kernel and exhaust the process.
* **the key uses the *declared* pool capacity, not the exact pool.**  The engine derives its KV pool
  from a memory snapshot, so the page count drifts between runs (9425 in one, 9899 in the next); an
  exact-pool key silently invalidated an out-of-process pre-warm, which pushed the compile back
  inside the engine and stopped it (documented in ``docs/status.md`` §5).  ``declared_pool_cap``
  rounds the pool up to a power of two so drift lands in the same bucket.

Dependency rule: this module imports ``torch`` but no engine - it is unit-tested on a CPU
(``tests/test_plan_cache.py``), with a fake builder standing in for the kernel.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, Hashable, Iterable, List, Optional, Sequence, Set, Tuple

from .buckets import BATCH_BUCKETS, KERNEL_PAGE_SIZE, sub_pages
from .counters import Counters

__all__ = [
    "PlanKey",
    "plan_key",
    "plan_width",
    "PlanCache",
    "prewarm",
]

#: A plan key is opaque to the cache; the serving layer decides what goes in it.
PlanKey = Tuple[Any, ...]

PlanFactory = Callable[[], Any]


def plan_width(max_model_len: int, block_size: int) -> int:
    """Block-table width, in **kernel pages**, for a model served at ``max_model_len``.

    The engine gives a request ``ceil(max_model_len / block_size)`` page-table columns; each engine
    page expands to ``sub_pages(block_size)`` kernel pages.  Keeping the width a constant per run is
    what lets the plan own exactly-sized staging buffers.
    """
    if max_model_len <= 0:
        raise ValueError(f"max_model_len must be positive, got {max_model_len}")
    sub = sub_pages(int(block_size))
    engine_pages = -(-int(max_model_len) // int(block_size))
    return engine_pages * sub


def plan_key(
    config_key: Hashable,
    split_pages: int,
    bucket: int,
    width: int,
    block_size: int,
    pool_cap: int,
) -> PlanKey:
    """The cache key: everything that changes the compiled kernel or its staging buffers.

    ``split_pages`` (KV pages per split tile) is part of it because buckets with different split
    factors have different schedules and workspace sizes, and sharing a plan between them produces
    a silently wrong merge.
    """
    if bucket < 1:
        raise ValueError(f"bucket must be >= 1, got {bucket}")
    if width < 1:
        raise ValueError(f"width must be >= 1, got {width}")
    if pool_cap < 1:
        raise ValueError(f"pool_cap must be >= 1, got {pool_cap}")
    return (config_key, int(split_pages), int(bucket), int(width), int(block_size), int(pool_cap))


class PlanCache:
    """Thread-safe plan cache with single-flight background builds."""

    def __init__(self) -> None:
        self._plans: Dict[PlanKey, Any] = {}
        self._building: Set[PlanKey] = set()
        self._failed: Set[PlanKey] = set()
        self._lock = threading.Lock()
        self._threads: List[threading.Thread] = []
        self.counters = Counters(
            {
                "hits": 0,
                "misses": 0,
                "built": 0,
                "background_builds": 0,
                "build_failures": 0,
            }
        )

    # ------------------------------------------------------------------ reads

    def __len__(self) -> int:
        with self._lock:
            return len(self._plans)

    def __contains__(self, key: PlanKey) -> bool:
        with self._lock:
            return key in self._plans

    def get(self, key: PlanKey) -> Optional[Any]:
        """The plan for ``key`` if it is ready, else ``None`` (never builds)."""
        with self._lock:
            plan = self._plans.get(key)
        self.counters.bump("hits" if plan is not None else "misses")
        return plan

    def is_failed(self, key: PlanKey) -> bool:
        with self._lock:
            return key in self._failed

    def is_building(self, key: PlanKey) -> bool:
        with self._lock:
            return key in self._building

    def ready_keys(self) -> List[PlanKey]:
        with self._lock:
            return sorted(self._plans)

    # ------------------------------------------------------------------ writes

    def build(self, key: PlanKey, factory: PlanFactory) -> Any:
        """Build synchronously (pre-warm).  A failure is remembered and re-raised."""
        with self._lock:
            plan = self._plans.get(key)
            if plan is not None:
                return plan
            if key in self._failed:
                raise RuntimeError(f"plan {key!r} previously failed to compile")
            if key in self._building:
                raise RuntimeError(f"plan {key!r} is already being built")
            self._building.add(key)
        try:
            plan = factory()
        except BaseException:
            with self._lock:
                self._failed.add(key)
                self._building.discard(key)
            self.counters.bump("build_failures")
            raise
        with self._lock:
            self._plans[key] = plan
            self._building.discard(key)
        self.counters.bump("built")
        return plan

    def get_or_start(self, key: PlanKey, factory: PlanFactory) -> Optional[Any]:
        """Return the plan, or start one background build and return ``None``.

        Returning ``None`` is the contract the serving step relies on: it falls back for this step
        and tries again the next one.  It must never block, because it may be called from inside a
        captured graph.
        """
        with self._lock:
            plan = self._plans.get(key)
            if plan is not None or key in self._failed or key in self._building:
                return plan
            self._building.add(key)
        self.counters.bump("background_builds")
        thread = threading.Thread(
            target=self._build_in_background, args=(key, factory), daemon=True, name="tileinfer-plan"
        )
        with self._lock:
            self._threads.append(thread)
        thread.start()
        return None

    def _build_in_background(self, key: PlanKey, factory: PlanFactory) -> None:
        try:
            plan = factory()
        except BaseException:  # noqa: BLE001 - a background failure must not kill the engine
            with self._lock:
                self._failed.add(key)
                self._building.discard(key)
            self.counters.bump("build_failures")
            return
        with self._lock:
            self._plans[key] = plan
            self._building.discard(key)
        self.counters.bump("built")

    def wait(self, key: PlanKey, timeout: float = 30.0) -> Optional[Any]:
        """Bounded wait for a background build (tests / pre-warm verification only)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                plan = self._plans.get(key)
                if plan is not None or key not in self._building:
                    return plan
            time.sleep(0.005)
        return None

    def failure_count(self) -> int:
        with self._lock:
            return len(self._failed)

    def snapshot(self) -> Dict[str, int]:
        with self._lock:
            stats = {
                "ready_plans": len(self._plans),
                "building": len(self._building),
                "failed": len(self._failed),
            }
        stats.update(self.counters.snapshot())
        return stats

    def clear(self) -> None:
        with self._lock:
            self._plans.clear()
            self._building.clear()
            self._failed.clear()
            self._threads.clear()


def buckets_up_to(max_num_reqs: int, buckets: Sequence[int] = BATCH_BUCKETS) -> Tuple[int, ...]:
    """The buckets a deployment can reach (see :func:`tileinfer.serving.buckets.buckets_for`)."""
    limit = max(1, int(max_num_reqs))
    kept = tuple(bucket for bucket in buckets if bucket <= limit)
    return kept or (1,)


def prewarm(
    cache: PlanCache,
    *,
    config_key: Hashable,
    split_pages_for: Callable[[int], int],
    width: int,
    block_size: int,
    pool_cap: int,
    max_num_reqs: int,
    buckets: Sequence[int] = BATCH_BUCKETS,
    factory_for: Callable[[int], PlanFactory],
) -> int:
    """Build every reachable bucket synchronously; returns how many were newly built.

    One bucket failing does not abort the rest: a shape that cannot compile should cost that bucket
    its fallback, not the whole pre-warm.
    """
    built = 0
    for bucket in buckets_up_to(max_num_reqs, buckets):
        key = plan_key(config_key, split_pages_for(bucket), bucket, width, block_size, pool_cap)
        if cache.get(key) is not None or cache.is_failed(key):
            continue
        try:
            cache.build(key, factory_for(bucket))
        except Exception:  # noqa: BLE001 - one bad bucket must not kill the pre-warm
            continue
        built += 1
    return built
