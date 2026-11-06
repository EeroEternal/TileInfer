"""CPU tests for the plan cache: single-flight, failure memory, pre-warm, key stability."""

from __future__ import annotations

import threading

import pytest

from tileinfer.serving.planner import (
    PlanCache,
    buckets_up_to,
    plan_key,
    plan_width,
    prewarm,
)


def test_plan_width_is_constant_under_block_size():
    # 8192 tokens is 64 kernel pages whether the engine page is 128 or 2048 tokens - that is why
    # the staging buffers can be sized once per run.
    assert plan_width(8192, 128) == 64
    assert plan_width(8192, 2048) == 64
    assert plan_width(1, 2048) == 16  # one engine page still expands to 16 kernel pages
    with pytest.raises(ValueError):
        plan_width(0, 128)


def test_plan_key_parts_and_validation():
    key = plan_key("cfg", 1, 4, 64, 2048, 16384)
    assert key == ("cfg", 1, 4, 64, 2048, 16384)
    with pytest.raises(ValueError):
        plan_key("cfg", 0, 0, 64, 2048, 16384)
    with pytest.raises(ValueError):
        plan_key("cfg", 0, 1, 0, 2048, 16384)
    with pytest.raises(ValueError):
        plan_key("cfg", 0, 1, 64, 2048, 0)


def test_hit_and_miss_counters():
    cache = PlanCache()
    key = plan_key("cfg", 0, 1, 8, 128, 1024)
    assert cache.get(key) is None
    cache.build(key, lambda: "PLAN")
    assert cache.get(key) == "PLAN"
    assert cache.counters.get("misses") == 1
    assert cache.counters.get("hits") == 1
    assert cache.counters.get("built") == 1
    assert len(cache) == 1 and key in cache


def test_build_is_memoised():
    cache = PlanCache()
    calls = []
    key = plan_key("cfg", 0, 2, 8, 128, 1024)
    assert cache.build(key, lambda: calls.append(1) or "P1") == "P1"
    assert cache.build(key, lambda: calls.append(1) or "P2") == "P1"
    assert calls == [1]


def test_failure_is_remembered_and_reraised():
    cache = PlanCache()
    key = plan_key("cfg", 0, 1, 8, 128, 1024)

    def boom():
        raise RuntimeError("compile failed")

    with pytest.raises(RuntimeError, match="compile failed"):
        cache.build(key, boom)
    assert cache.is_failed(key)
    assert cache.counters.get("build_failures") == 1
    with pytest.raises(RuntimeError, match="previously failed"):
        cache.build(key, lambda: "never")
    # a failed key is never rebuilt implicitly
    assert cache.get_or_start(key, lambda: "never") is None


def test_concurrent_build_of_the_same_key_is_rejected():
    cache = PlanCache()
    key = plan_key("cfg", 0, 1, 8, 128, 1024)
    started, release = threading.Event(), threading.Event()

    def slow():
        started.set()
        release.wait(5)
        return "PLAN"

    worker = threading.Thread(target=lambda: cache.build(key, slow))
    worker.start()
    assert started.wait(5)
    with pytest.raises(RuntimeError, match="already being built"):
        cache.build(key, slow)
    release.set()
    worker.join(5)
    assert cache.get(key) == "PLAN"


def test_background_build_is_single_flight_and_never_blocks():
    cache = PlanCache()
    key = plan_key("cfg", 0, 4, 8, 128, 1024)
    calls = []
    release = threading.Event()

    def slow():
        calls.append(1)
        release.wait(5)
        return "PLAN"

    assert cache.get_or_start(key, slow) is None  # first call starts the build
    assert cache.get_or_start(key, slow) is None  # second call joins, does not start another
    assert cache.counters.get("background_builds") == 1
    assert cache.is_building(key)
    release.set()
    assert cache.wait(key, timeout=5) == "PLAN"
    assert calls == [1]


def test_background_failure_is_recorded_not_raised():
    cache = PlanCache()
    key = plan_key("cfg", 0, 1, 8, 128, 1024)
    cache.get_or_start(key, lambda: (_ for _ in ()).throw(RuntimeError("nope")))
    assert cache.wait(key, timeout=5) is None
    assert cache.is_failed(key)
    assert cache.counters.get("build_failures") == 1


def test_prewarm_builds_every_reachable_bucket_and_survives_one_failure():
    cache = PlanCache()
    keys = []

    def factory_for(bucket):
        def factory():
            keys.append(bucket)
            if bucket == 4:
                raise RuntimeError("bad bucket")
            return f"plan-{bucket}"

        return factory

    built = prewarm(
        cache,
        config_key="cfg",
        split_pages_for=lambda bucket: 0,
        width=64,
        block_size=2048,
        pool_cap=16384,
        max_num_reqs=5,
        factory_for=factory_for,
    )
    assert built == 2  # buckets 1 and 2 built; 4 failed; 8 is above max_num_reqs=5
    assert keys == [1, 2, 4]
    assert cache.get(plan_key("cfg", 0, 2, 64, 2048, 16384)) == "plan-2"
    assert cache.is_failed(plan_key("cfg", 0, 4, 64, 2048, 16384))
    # running it again is a no-op for the ready ones and does not retry the failed one
    assert prewarm(
        cache,
        config_key="cfg",
        split_pages_for=lambda bucket: 0,
        width=64,
        block_size=2048,
        pool_cap=16384,
        max_num_reqs=5,
        factory_for=factory_for,
    ) == 0


def test_buckets_up_to():
    assert buckets_up_to(5) == (1, 2, 4)
    assert buckets_up_to(1) == (1,)
    assert buckets_up_to(0) == (1,)
    assert buckets_up_to(300)[-1] == 256


def test_snapshot_reports_cache_state():
    cache = PlanCache()
    cache.build(plan_key("cfg", 0, 1, 8, 128, 1024), lambda: "P")
    snap = cache.snapshot()
    assert snap["ready_plans"] == 1 and snap["failed"] == 0 and snap["built"] == 1
