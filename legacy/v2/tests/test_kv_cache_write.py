"""CPU tests for the KV-cache write path (no device needed).

The write is the one piece of the backend that fails *silently* when wrong: attention keeps
producing plausible output from a pool that holds stale tokens.  So every case here compares the
written pool against a naive reference, element for element, including the sub-page layout that
the kernel actually reads.

Three layouts, one address space
--------------------------------
vLLM keeps K and V in one tensor and the *physical* layout of that tensor depends on the backend's
declarations (``get_kv_cache_shape`` + ``get_kv_cache_stride_order``) and, for a hybrid run, on
whether vLLM rewrites the strides (``_update_hybrid_attention_mamba_layout``).  The write must put
a token at ``(kv, block, offset)`` in **all** of them, and this file pins that against a naive
per-element oracle rather than against another ``index_put_``:

* ``kv_major``   - ``[2, blocks, page, heads, dim]`` contiguous (the vLLM-native form),
* ``block_major`` - ``[blocks, 2, ...]`` logical, physically ``[2][blocks][...]`` (what this
  backend declares: the halves are contiguous and the kernel can read them),
* ``block_interleaved`` - the legacy shape after vLLM's hybrid rewrite: same *shape*, strides
  physically ``[blocks][2][page][heads][dim]`` (the write still works; the kernel cannot read it,
  which is exactly why the backend does not declare this layout).
"""

from __future__ import annotations

import pytest
import torch

from tileinfer.serving.buckets import KERNEL_PAGE_SIZE, fused_halves, pool_sub_view
from tileinfer.vllm.attn.kv_cache import (
    check_slot_mapping,
    flat_pool,
    write_kv_cache,
    write_kv_cache_fused,
)


def _pool(blocks: int, block_size: int, heads: int, dim: int, dtype=torch.float32):
    # A deterministic, non-zero pool so a stale read cannot look like a correct write.
    return torch.arange(
        blocks * block_size * heads * dim, dtype=torch.float32
    ).reshape(blocks, block_size, heads, dim).to(dtype)


def _reference(pool: torch.Tensor, rows: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    """Naive per-element oracle: single-element indexing, never another index_put_."""
    out = pool.clone()
    for index, slot in enumerate(slots.tolist()):
        out[slot // out.shape[1], slot % out.shape[1]] = rows[index]
    return out


def _fused(layout: str, blocks: int, block_size: int, heads: int, dim: int) -> torch.Tensor:
    """One logical fused pool in the requested physical layout."""
    numel = 2 * blocks * block_size * heads * dim
    base = torch.arange(numel, dtype=torch.float32).reshape(
        2, blocks, block_size, heads, dim
    )
    if layout == "kv_major":
        return base
    if layout == "block_major":
        # `_reshape_kv_cache_tensors` with stride order (1, 0, 2, 3, 4), then permute back.
        return base.permute(1, 0, 2, 3, 4)
    if layout == "block_interleaved":
        # `_update_hybrid_attention_mamba_layout`: same shape, strides physically [blocks][2][...].
        hidden = block_size * heads * dim
        base.as_strided_(
            size=base.shape, stride=(hidden, 2 * hidden, heads * dim, dim, 1)
        )
        return base
    raise AssertionError(layout)


LAYOUTS = ("kv_major", "block_major", "block_interleaved")


def _half(fused: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The K/V halves with the layout the test built (explicit, so the test never guesses)."""
    block_dim = 1 if tuple(fused.shape)[0] == 2 else 0
    return fused_halves(fused, block_dim)


# --------------------------------------------------------------------------- flat view


def test_flat_pool_is_a_view():
    pool = _pool(2, 4, 2, 8)
    flat = flat_pool(pool)
    assert flat.shape == (8, 2, 8)
    assert flat.data_ptr() == pool.data_ptr()


# --------------------------------------------------------------------------- plain writes


def test_decode_writes_one_slot_per_request():
    pool = _pool(2, 4, 2, 8)
    slots = torch.tensor([0, 5, 7], dtype=torch.int64)
    key = torch.randn(3, 2, 8)
    value = torch.randn(3, 2, 8)
    k_cache, v_cache = pool.clone(), pool.clone()
    write_kv_cache(key, value, k_cache, v_cache, slots)
    assert torch.equal(k_cache, _reference(pool, key, slots))
    assert torch.equal(v_cache, _reference(pool, value, slots))


def test_chunked_prefill_writes_many_tokens_per_request():
    pool = _pool(2, 4, 2, 8)
    # request 0 owns slots 0..2, request 1 owns 4..5 (the engine decides; we just obey)
    slots = torch.tensor([0, 1, 2, 4, 5], dtype=torch.int64)
    key = torch.randn(5, 2, 8)
    value = torch.randn(5, 2, 8)
    k_cache, v_cache = pool.clone(), pool.clone()
    write_kv_cache(key, value, k_cache, v_cache, slots)
    assert torch.equal(k_cache, _reference(pool, key, slots))
    assert torch.equal(v_cache, _reference(pool, value, slots))


def test_write_honours_the_sub_page_layout_the_kernel_reads():
    # engine page = 4 kernel sub-pages of 8 tokens; slots are flat token indices, so writing at a
    # sub-page boundary must land exactly where the 128-token view would read it.
    blocks, block_size, heads, dim = 2, 32, 2, 4
    pool = _pool(blocks, block_size, heads, dim)
    slots = torch.tensor([0, 31, 32, 63], dtype=torch.int64)
    key = torch.randn(4, heads, dim)
    value = torch.randn(4, heads, dim)
    k_cache, v_cache = pool.clone(), pool.clone()
    write_kv_cache(key, value, k_cache, v_cache, slots)
    sub = block_size // 8
    view = pool_sub_view_for_test(k_cache, sub, 8)
    # slot 32 is the first token of the second engine page == sub-page 4 of 8-token pages
    assert torch.equal(view[4, 0], key[2])


def pool_sub_view_for_test(pool: torch.Tensor, sub: int, page: int) -> torch.Tensor:
    blocks, block_size, heads, dim = pool.shape
    assert block_size == sub * page
    return pool.view(blocks * sub, page, heads, dim)


def test_fused_pool_writes_both_halves():
    k_pool = _pool(1, 4, 2, 8)
    v_pool = _pool(1, 4, 2, 8) + 100.0
    fused = torch.stack([k_pool, v_pool], dim=0)
    slots = torch.tensor([1, 2], dtype=torch.int64)
    key = torch.randn(2, 2, 8)
    value = torch.randn(2, 2, 8)
    write_kv_cache_fused(fused, key, value, slots)
    assert torch.equal(fused[0], _reference(k_pool, key, slots))
    assert torch.equal(fused[1], _reference(v_pool, value, slots))


def test_bf16_pool_and_rows_are_written_without_a_cast():
    pool = _pool(1, 4, 2, 8, dtype=torch.bfloat16)
    key = torch.randn(2, 2, 8).to(torch.bfloat16)
    value = torch.randn(2, 2, 8).to(torch.bfloat16)
    slots = torch.tensor([0, 3], dtype=torch.int64)
    k_cache, v_cache = pool.clone(), pool.clone()
    write_kv_cache(key, value, k_cache, v_cache, slots)
    assert k_cache.dtype is torch.bfloat16
    assert torch.equal(k_cache, _reference(pool, key, slots))


def test_empty_step_is_a_no_op():
    pool = _pool(1, 4, 2, 8)
    k_cache = pool.clone()
    write_kv_cache(
        torch.empty(0, 2, 8),
        torch.empty(0, 2, 8),
        k_cache,
        pool.clone(),
        torch.empty(0, dtype=torch.int64),
    )
    assert torch.equal(k_cache, pool)


# --------------------------------------------------------------------------- layouts


@pytest.mark.parametrize("layout", LAYOUTS)
def test_the_same_slots_land_on_the_same_logical_positions_in_every_layout(layout):
    """The core discrimination: layout changes the bytes, not the addressable positions."""
    blocks, block_size, heads, dim = 3, 4, 2, 3
    slots = torch.tensor([0, 5, 6, 11], dtype=torch.int64)
    key = torch.randn(len(slots), heads, dim)
    value = torch.randn(len(slots), heads, dim)

    fused = _fused(layout, blocks, block_size, heads, dim)
    # Snapshot the halves *before* the write: they are views, so a clone is the oracle's input.
    pristine_k, pristine_v = (half.clone() for half in _half(fused))
    write_kv_cache_fused(fused, key, value, slots)
    k_half, v_half = _half(fused)

    assert torch.equal(k_half, _reference(pristine_k, key, slots))
    assert torch.equal(v_half, _reference(pristine_v, value, slots))
    # ...and therefore every written row is at (block, offset) of the *logical* page array.
    for index, slot in enumerate(slots.tolist()):
        block, offset = divmod(slot, block_size)
        assert torch.equal(k_half[block, offset], key[index])
        assert torch.equal(v_half[block, offset], value[index])


def test_a_block_offset_swap_is_detected():
    """``block = slot // block_size`` and ``offset = slot % block_size`` are not interchangeable.

    Slot 6 of a 4-token page is block 1, offset 2 - and (block 2, offset 1) is a *different, valid*
    position, so a swapped implementation writes successfully and silently.
    """
    blocks, block_size, heads, dim = 3, 4, 2, 3
    slot = 6
    key = torch.randn(1, heads, dim)
    value = torch.randn(1, heads, dim)
    for layout in LAYOUTS:
        fused = _fused(layout, blocks, block_size, heads, dim)
        k_half, _ = _half(fused)
        before = k_half.clone()
        write_kv_cache_fused(fused, key, value, torch.tensor([slot], dtype=torch.int64))
        assert torch.equal(k_half[slot // block_size, slot % block_size], key[0]), layout
        swapped = (slot % block_size, slot // block_size)
        assert torch.equal(
            k_half[swapped], before[swapped]
        ), f"{layout}: block and offset were swapped"


def test_dropping_the_kv_axis_from_the_index_is_detected():
    """K and V must go to *different* halves; a write that ignores the K/V axis fails here."""
    blocks, block_size, heads, dim = 3, 4, 2, 3
    slot = 5
    key = torch.randn(1, heads, dim)
    value = torch.randn(1, heads, dim) + 1000.0  # never equal to `key`
    for layout in LAYOUTS:
        fused = _fused(layout, blocks, block_size, heads, dim)
        pristine_k, pristine_v = (half.clone() for half in _half(fused))
        write_kv_cache_fused(fused, key, value, torch.tensor([slot], dtype=torch.int64))
        k_half, v_half = _half(fused)
        block, offset = divmod(slot, block_size)
        # K half: K, not V (and not "whichever half came first").
        assert torch.equal(k_half[block, offset], key[0]), f"{layout}: K not written to K"
        # V half: V, not K.
        assert torch.equal(v_half[block, offset], value[0]), f"{layout}: V not written to V"
        # The rest of each half is untouched, which a both-halves write would break.
        one = torch.tensor([slot], dtype=torch.int64)
        assert torch.equal(k_half, _reference(pristine_k, key, one)), layout
        assert torch.equal(v_half, _reference(pristine_v, value, one)), layout


@pytest.mark.parametrize("layout", ("kv_major", "block_major"))
def test_the_kernel_page_view_sees_exactly_the_written_tokens(layout):
    """Write -> 128-token kernel view: page ``slot // 128``, token ``slot % 128``.

    This is the read side of the contract, so it also proves the halves are a *flat* page array
    (which is what lets the kernel keep its page-id arithmetic unchanged).  The interleaved hybrid
    layout is excluded on purpose: it has no flat page view at all - see
    :func:`test_the_interleaved_hybrid_layout_has_no_flat_page_view`.
    """
    blocks, block_size, heads, dim = 3, 512, 2, 4  # 4 kernel pages per engine page
    slots = torch.tensor([0, 127, 128, 511], dtype=torch.int64)
    key = torch.randn(len(slots), heads, dim)
    value = torch.randn(len(slots), heads, dim)

    fused = _fused(layout, blocks, block_size, heads, dim)
    write_kv_cache_fused(fused, key, value, slots)
    k_half, v_half = _half(fused)

    sub = block_size // KERNEL_PAGE_SIZE
    k_page = pool_sub_view(k_half, sub)
    v_page = pool_sub_view(v_half, sub)
    assert k_page.data_ptr() == k_half.data_ptr() and v_page.data_ptr() == v_half.data_ptr()
    for index, slot in enumerate(slots.tolist()):
        page, token = divmod(slot, KERNEL_PAGE_SIZE)
        assert torch.equal(k_page[page, token], key[index])
        assert torch.equal(v_page[page, token], value[index])


def test_the_interleaved_hybrid_layout_has_no_flat_page_view():
    """Why the backend does not declare the layout vLLM's hybrid rewrite produces.

    The *write* handles it (the three-layout tests above), but no ``[pages, 128, heads, dim]`` view
    of a half exists: a page is ``(block, kv, sub)``, so a flat page id cannot be formed without a
    copy.  The kernel takes such a view, so this layout is rejected - loudly, not repacked.
    """
    fused = _fused("block_interleaved", 2, 512, 2, 4)
    k_half, _ = _half(fused)
    assert not k_half.is_contiguous()
    with pytest.raises(ValueError, match="contiguous"):
        pool_sub_view(k_half, 512 // KERNEL_PAGE_SIZE)



def test_flat_pool_refuses_a_view_it_cannot_take_and_the_write_still_works():
    """The new contract: a strided pool is *writable*, it is only not *viewable*.

    ``flat_pool`` returns a view and therefore keeps refusing; ``write_kv_cache`` no longer needs
    that view, so the layout that used to raise "KV pool must be contiguous" is now handled.
    """
    page, block_size, heads, dim = 3, 4, 2, 3
    base = _pool(page, 2 * block_size, heads, dim)
    pool = base[:, ::2]  # [3, 4, 2, 3] with stride (48, 12, 3, 1): a valid, non-contiguous NHD pool
    assert pool.shape[1] == block_size and not pool.is_contiguous()

    with pytest.raises(ValueError, match="contiguous"):
        flat_pool(pool)

    slots = torch.tensor([6, 5], dtype=torch.int64)
    key = torch.randn(2, heads, dim)
    value = torch.randn(2, heads, dim)
    k_cache, v_cache = pool.clone(), pool.clone()
    write_kv_cache(key, value, k_cache, v_cache, slots)
    assert torch.equal(k_cache, _reference(pool, key, slots))
    assert torch.equal(v_cache, _reference(pool, value, slots))


# --------------------------------------------------------------------------- fused halves


def test_fused_halves_are_the_same_storage_one_kv_factor_apart():
    blocks, block_size, heads, dim = 7, 4, 2, 3
    fused = _fused("block_major", blocks, block_size, heads, dim)
    k_half, v_half = fused_halves(fused)  # inferred: dim 1 is the 2
    assert k_half.is_contiguous() and v_half.is_contiguous()
    assert k_half.data_ptr() == fused.data_ptr()
    assert v_half.storage_offset() - k_half.storage_offset() == blocks * block_size * heads * dim

    legacy = _fused("kv_major", blocks, block_size, heads, dim)
    k_legacy, v_legacy = fused_halves(legacy)
    assert k_legacy.is_contiguous() and v_legacy.is_contiguous()
    assert k_legacy.data_ptr() == legacy.data_ptr()


def test_fused_halves_refuses_to_guess_when_both_dims_are_two():
    """Two blocks and two K/V halves are metadata-identical; guessing would swap K and V."""
    ambiguous = torch.zeros(2, 2, 4, 2, 3)
    with pytest.raises(ValueError, match="block_dim"):
        fused_halves(ambiguous)
    # ...but an explicit answer is accepted, and it is the caller's (documented) responsibility.
    k_half, v_half = fused_halves(ambiguous, block_dim=0)
    assert k_half.data_ptr() == ambiguous.data_ptr()
    assert v_half.storage_offset() - k_half.storage_offset() == 4 * 2 * 3
    k_legacy, v_legacy = fused_halves(ambiguous, block_dim=1)
    assert k_legacy.data_ptr() == ambiguous.data_ptr()
    assert v_legacy.storage_offset() - k_legacy.storage_offset() == 2 * 4 * 2 * 3


@pytest.mark.parametrize(
    "shape,block_dim,match",
    [
        ((2, 3, 4, 2, 3), 0, "dim 1 is 3"),
        ((3, 2, 4, 2, 3), 1, "dim 0 is 3"),
        ((3, 2, 4, 2, 3), 2, "must be 0 or 1"),
        ((3, 2, 4, 2), None, "must be 5-D"),
    ],
)
def test_fused_halves_rejects_inconsistent_layouts(shape, block_dim, match):
    with pytest.raises(ValueError, match=match):
        fused_halves(torch.zeros(shape), block_dim)


# --------------------------------------------------------------------------- validation


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda k, v, s: (k, v, s.to(torch.int32)), "must be int64"),
        (lambda k, v, s: (k.double(), v.double(), s), "must match the pool dtype"),
        (lambda k, v, s: (k.unsqueeze(0), v, s), "key must be"),
        (lambda k, v, s: (k, v[:, :1], s), "must match key shape"),
        (lambda k, v, s: (k, v, s[:-1]), "slot_mapping has"),
        (lambda k, v, s: (k[:1], v[:1], s), "key has 1 tokens"),
    ],
)
def test_shape_and_dtype_mistakes_are_rejected(mutate, match):
    pool = _pool(1, 4, 2, 8)
    key = torch.randn(2, 2, 8)
    value = torch.randn(2, 2, 8)
    slots = torch.tensor([0, 1], dtype=torch.int64)
    key, value, slots = mutate(key, value, slots)
    with pytest.raises(ValueError, match=match):
        write_kv_cache(key, value, pool.clone(), pool.clone(), slots)


def test_a_pool_that_is_not_one_half_of_a_fused_pair_is_rejected():
    pool = _pool(1, 4, 2, 8)
    slots = torch.tensor([0], dtype=torch.int64)
    with pytest.raises(ValueError, match="one half of the fused pool"):
        write_kv_cache(torch.randn(1, 2, 8), torch.randn(1, 2, 8), pool, pool[0], slots)


def test_check_slot_mapping_catches_out_of_range_and_negative():
    pool = _pool(1, 4, 2, 8)
    flat = flat_pool(pool)
    check_slot_mapping(torch.tensor([0, 3], dtype=torch.int64), flat)
    with pytest.raises(ValueError, match="addresses slots"):
        check_slot_mapping(torch.tensor([0, 4], dtype=torch.int64), flat)
    with pytest.raises(ValueError, match="addresses slots"):
        check_slot_mapping(torch.tensor([-1], dtype=torch.int64), flat)
