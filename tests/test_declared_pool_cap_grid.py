"""The declared-capacity grid: coverage, key stability, bounded extension.

The declared capacity quantizes a KV pool's page count onto an eighth-of-a-pow2 grid so that
(a) a pre-warm compiled for one pool size keeps matching a drifted pool, and (b) the gap the
capacity view must realise as a storage extension stays small - on torch_npu that extension is
a real allocation, which OOMed a hybrid two-group deployment when a plain pow2 rounding
demanded up to 2x the pool.
"""

from tileinfer.serving.buckets import declared_pool_cap, next_pow2


def test_aligned_pools_get_no_extension():
    for n in (1, 2, 4, 8, 1024, 8192, 16384):
        assert declared_pool_cap(n) == n, n


def test_small_pools_land_on_exact_integers():
    # base < 8 makes base*(8+j)//8 cover every integer up to the enclosing power of two
    assert declared_pool_cap(3) == 3
    assert declared_pool_cap(5) == 5
    assert declared_pool_cap(6) == 6
    assert declared_pool_cap(7) == 7
    assert declared_pool_cap(30) == 30
    assert declared_pool_cap(31) == 32
    assert declared_pool_cap(33) == 36


def test_nearby_snapshot_sizes_share_a_grid_point():
    # the two pool sizes actually observed on the reference box, 5% apart: same compiled key
    assert declared_pool_cap(9425) == declared_pool_cap(9899) == 10240


def test_coverage_and_bounded_extension():
    for pool in (1, 2, 3, 17, 100, 8193, 9425, 12000, 58000, 58064):
        cap = declared_pool_cap(pool)
        assert cap >= pool, pool
        assert cap <= next_pow2(pool), pool
        assert cap - pool <= max(1, pool // 8), (pool, cap, cap - pool)


def test_monotone_under_drift():
    prev = 0
    for pool in range(1, 400):
        cap = declared_pool_cap(pool)
        assert cap >= pool and cap >= prev, pool
        prev = max(prev, cap)


def test_explicit_pin_wins(monkeypatch):
    monkeypatch.setenv("TILEINFER_POOL_CAP", "64")
    assert declared_pool_cap(50) == 64
    import pytest

    with pytest.raises(ValueError, match="below the real pool"):
        declared_pool_cap(65)
