"""CPU tests for the serving arithmetic: buckets, sub-pages, ragged page counts.

These pin the three conversions that are silent when wrong: a batch padded to the wrong bucket, an
engine page expanded to the wrong sub-page order, and a last-page length that lets the kernel read
uninitialised cache.  No device and no engine required.
"""

from __future__ import annotations

import pytest
import torch

from tileinfer.serving import (
    BATCH_BUCKETS,
    KERNEL_PAGE_SIZE,
    Counters,
    bucket_for,
    buckets_for,
    capacity_view,
    expand_page_table,
    fused_halves,
    last_page_len,
    last_page_len_,
    next_pow2,
    pages_for_lens,
    pool_sub_view,
    sub_page_width,
    sub_pages,
)


# --------------------------------------------------------------------------- buckets


def test_kernel_page_size_is_the_kernel_n_tile():
    assert KERNEL_PAGE_SIZE == 128


@pytest.mark.parametrize(
    "batch,expected", [(1, 1), (2, 2), (3, 4), (5, 8), (16, 16), (17, 32), (256, 256)]
)
def test_bucket_for_rounds_up(batch, expected):
    assert bucket_for(batch) == expected


def test_bucket_for_rejects_batches_above_the_largest_bucket():
    with pytest.raises(ValueError, match="exceeds the largest bucket"):
        bucket_for(BATCH_BUCKETS[-1] + 1)


def test_bucket_for_rejects_non_positive_batches():
    with pytest.raises(ValueError, match="num_reqs must be positive"):
        bucket_for(0)


def test_bucket_for_rejects_unsorted_buckets():
    with pytest.raises(ValueError, match="ascending"):
        bucket_for(3, buckets=(1, 8, 4))


def test_buckets_for_truncates_at_max_num_reqs():
    assert buckets_for(5) == (1, 2, 4)
    assert buckets_for(1) == (1,)
    assert buckets_for(0) == (1,)
    assert buckets_for(1000)[-1] == BATCH_BUCKETS[-1]


@pytest.mark.parametrize("value,expected", [(1, 1), (2, 2), (3, 4), (17, 32), (9425, 16384)])
def test_next_pow2(value, expected):
    assert next_pow2(value) == expected


# --------------------------------------------------------------------------- sub pages


@pytest.mark.parametrize(
    "block_size,expected", [(128, 1), (256, 2), (2048, 16), (4096, 32)]
)
def test_sub_pages(block_size, expected):
    assert sub_pages(block_size) == expected


@pytest.mark.parametrize("block_size", [64, 96, 192, 200])
def test_sub_pages_rejects_non_multiples(block_size):
    with pytest.raises(ValueError, match="not a positive multiple"):
        sub_pages(block_size)


def test_sub_page_width():
    assert sub_page_width(4, 2048) == 64
    assert sub_page_width(4, 128) == 4
    with pytest.raises(ValueError):
        sub_page_width(0, 128)


def test_expand_page_table_is_identity_for_sub_one():
    table = torch.arange(6, dtype=torch.int32).reshape(2, 3)
    out = expand_page_table(table, 1)
    assert out is table  # zero-copy: no repack at the backend boundary


def test_expand_page_table_maps_engine_page_to_consecutive_sub_pages():
    table = torch.tensor([[0, 1], [3, 2]], dtype=torch.int32)
    out = expand_page_table(table, 4)
    assert out.shape == (2, 8)
    # page p -> sub-page ids p*4 + [0,1,2,3], in that order
    assert out.tolist() == [[0, 1, 2, 3, 4, 5, 6, 7], [12, 13, 14, 15, 8, 9, 10, 11]]


def test_pool_sub_view_is_a_view_of_the_same_bytes():
    blocks, block_size, heads, dim = 2, 512, 2, 4  # 4 kernel pages of 128 tokens per engine page
    sub = 4
    pool = torch.arange(blocks * block_size * heads * dim, dtype=torch.float32).reshape(
        blocks, block_size, heads, dim
    )
    view = pool_sub_view(pool, sub)
    assert view.shape == (blocks * sub, 128, heads, dim)
    assert view.data_ptr() == pool.data_ptr()
    # sub-page (p, t) is engine page p//sub, token offset (p%sub)*128 + t
    for p in (0, 3, 4, 7):
        for t in (0, 3, 127):
            engine_page, offset = divmod(p, sub)
            assert torch.equal(view[p, t], pool[engine_page, offset * 128 + t])


def test_pool_sub_view_identity_and_misalignment():
    pool = torch.zeros(2, 128, 2, 4)
    assert pool_sub_view(pool, 1) is pool
    with pytest.raises(ValueError, match="misaligned|pass sub_pages"):
        pool_sub_view(torch.zeros(2, 256, 2, 4), 1)
    with pytest.raises(ValueError, match="contiguous"):
        pool_sub_view(torch.zeros(2, 256, 2, 4)[:, ::2], 1)


def test_pool_sub_view_of_a_block_major_half_is_the_same_page_view():
    """The hybrid contract: block-first logical shape, K/V-outermost physical order.

    ``get_kv_cache_shape`` + ``get_kv_cache_stride_order`` make ``kv_cache[:, 0]`` a contiguous
    ``[blocks, page, heads, dim]`` tensor, so the kernel's 128-token view is the same zero-copy
    reshape it always was - the fused layout does not reach the kernel.
    """
    blocks, block_size, heads, dim, sub = 3, 512, 2, 4, 4
    physical = torch.arange(
        2 * blocks * block_size * heads * dim, dtype=torch.float32
    ).reshape(2, blocks, block_size, heads, dim)
    pool = physical.permute(1, 0, 2, 3, 4)
    assert not pool.is_contiguous()
    k_half, v_half = fused_halves(pool)
    assert k_half.is_contiguous() and v_half.is_contiguous()
    page = pool_sub_view(k_half, sub)
    assert page.data_ptr() == k_half.data_ptr()
    for p in (0, 1, sub - 1, sub, blocks * sub - 1):
        engine_page, offset = divmod(p, sub)
        assert torch.equal(page[p, 0], k_half[engine_page, offset * 128])


def test_capacity_view_keeps_every_half_at_its_own_offset():
    """The ABI capacity view must not move the V half onto the K half.

    ``pool_view`` re-binds the pool's storage at the declared capacity with ``set_``.  Re-binding
    at offset 0 (the original form) made the V half alias K - the kernel would attend K twice with
    no error.  The capacity view must keep the half's storage offset and only inflate the extent.
    """
    blocks, block_size, heads, dim = 3, 512, 2, 4
    for layout in ("kv_major", "block_major"):
        physical = torch.zeros(2, blocks, block_size, heads, dim)
        pool = physical if layout == "kv_major" else physical.permute(1, 0, 2, 3, 4)
        k_half, v_half = fused_halves(pool, block_dim=1 if layout == "kv_major" else 0)
        cap = next_pow2(blocks * (block_size // KERNEL_PAGE_SIZE))
        assert cap > blocks * (block_size // KERNEL_PAGE_SIZE)  # the interesting (non-pow2) case
        k_view = capacity_view(k_half, cap)
        v_view = capacity_view(v_half, cap)
        assert k_view.data_ptr() == k_half.data_ptr()
        assert v_view.data_ptr() == v_half.data_ptr(), (
            f"{layout}: the V capacity view lost its storage offset and now reads K"
        )
        assert v_view.data_ptr() - k_view.data_ptr() == blocks * block_size * heads * dim * 4
        assert tuple(v_view.shape) == (cap, block_size, heads, dim)
        assert v_view.stride() == v_half.stride()


def test_capacity_view_is_zero_copy_and_covers_the_real_pages():
    blocks, block_size, heads, dim = 3, 128, 2, 4
    pool = torch.zeros(blocks, block_size, heads, dim)
    pool[2, 7] = 3.5
    cap = next_pow2(blocks + 5)
    view = capacity_view(pool, cap)
    assert view.data_ptr() == pool.data_ptr()
    assert tuple(view.shape) == (cap, block_size, heads, dim)
    assert torch.equal(view[2, 7], pool[2, 7])
    assert view.storage_offset() == pool.storage_offset()


def test_capacity_view_refuses_to_hide_pages_and_bad_shapes():
    pool = torch.zeros(4, 128, 2, 4)
    assert capacity_view(pool, 4) is pool  # already at the declared capacity: no view needed
    with pytest.raises(ValueError, match="compiled capacity"):
        capacity_view(pool, 3)
    with pytest.raises(ValueError, match="4-D|\\[pages"):
        capacity_view(torch.zeros(4, 128, 8), 4)


def test_pool_sub_view_refuses_the_interleaved_hybrid_layout_with_the_reason():
    """vLLM's hybrid rewrite lays K and V out per *block*; that has no flat page array.

    A page would be ``(block, kv, sub)``, so a single ``[pages, 128, heads, dim]`` view cannot be
    formed without copying the pool.  The refusal must say so (and point at the block-major form),
    because "KV pool must be contiguous" alone sent the last debugging session down the wrong path.
    """
    blocks, block_size, heads, dim = 3, 512, 2, 4
    pool = torch.zeros(2, blocks, block_size, heads, dim)
    hidden = block_size * heads * dim
    pool.as_strided_(
        size=pool.shape, stride=(hidden, 2 * hidden, heads * dim, dim, 1)
    )
    k_half, _ = fused_halves(pool)
    assert not k_half.is_contiguous()
    with pytest.raises(ValueError, match="interleaved per block"):
        pool_sub_view(k_half, block_size // KERNEL_PAGE_SIZE)


# --------------------------------------------------------------------------- lengths


@pytest.mark.parametrize(
    "lens,expected",
    [([1, 128, 129, 2048, 2049], [1, 128, 1, 128, 1]), ([2, 127, 130], [2, 127, 2])],
)
def test_last_page_len(lens, expected):
    out = last_page_len(torch.tensor(lens, dtype=torch.int32))
    assert out.tolist() == expected


def test_last_page_len_in_place_matches_functional():
    lens = torch.tensor([1, 7, 128, 129, 2048], dtype=torch.int32)
    out = torch.empty_like(lens)
    last_page_len_(lens, out)
    assert torch.equal(out, last_page_len(lens))
    with pytest.raises(ValueError, match="must match lens shape"):
        last_page_len_(lens, torch.empty(2, dtype=torch.int32))


def test_pages_for_lens():
    assert pages_for_lens(torch.tensor([1, 128, 129, 2048], dtype=torch.int32)).tolist() == [
        1,
        1,
        2,
        16,
    ]


# --------------------------------------------------------------------------- counters


def test_counters_snapshot_is_a_copy_and_bump_is_monotonic():
    counters = Counters({"capture_calls": 0})
    assert counters.bump("capture_calls") == 1
    assert counters.bump("capture_calls", 4) == 5
    assert counters.get("missing") == 0
    snap = counters.snapshot()
    snap["capture_calls"] = 999
    assert counters.get("capture_calls") == 5
    assert "capture_calls" in counters and len(counters) == 1
