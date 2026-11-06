"""CPU tests for the per-step ragged rebuild (dense page table -> indptr/indices/last_len).

This is the conversion that used to produce "plausible garbage": a dense indptr made every request
attend over every table column, including stale ids.  The oracle here is the obvious host-side
implementation (``expected_ragged``); the device path must agree with it exactly, including the
trash-slot parking that keeps stale ids unreachable.
"""

from __future__ import annotations

import pytest
import torch

from tileinfer.serving.step import (
    allocate_step_buffers,
    expected_ragged,
    refresh_step,
)


def _make(bucket=4, engine_width=8, block_size=2048, capacity=4096):
    sub = block_size // 128
    width = engine_width * sub
    buffers = allocate_step_buffers(bucket, width, capacity, block_size, torch.device("cpu"))
    kv_indices = torch.full((capacity,), -7, dtype=torch.int32)
    return buffers, kv_indices


def _check(buffers, kv_indices, table, lens, block_size):
    indptr, indices, last = expected_ragged(table, lens, block_size)
    n = int(lens.numel())
    used = int(indptr[-1])
    assert torch.equal(buffers.kv_indptr_buf[: n + 1], indptr), "indptr prefix mismatch"
    assert torch.equal(kv_indices[:used], indices), "compacted page ids mismatch"
    assert torch.equal(buffers.last_page_len_buf[:n], last), "last page length mismatch"
    # Padding rows keep a valid, minimal footprint: length 1 -> one page, table column 0 -> page 0.
    assert buffers.lens_buf[n:].tolist() == [1] * (buffers.bucket - n)
    assert torch.equal(buffers.pages_buf[n:], torch.ones(buffers.bucket - n, dtype=torch.int32))
    # The trash slot is past every valid destination, so it can never be read.
    assert int(buffers.kv_indptr_buf[-1]) <= buffers.trash_slot
    # Padding rows write zero page ids between the live span and the end.
    tail = kv_indices[used : int(buffers.kv_indptr_buf[-1])]
    assert torch.all(tail == 0)


def test_sub_page_expansion_matches_the_reference():
    buffers, kv_indices = _make(block_size=2048)  # 16 sub-pages per engine page
    table = torch.tensor(
        [[3, 5, 7, 9, 11, 13, 15, 17], [1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.int32
    )
    lens = torch.tensor([300, 2049], dtype=torch.int32)
    refresh_step(buffers, table, lens, 2048, kv_indices)
    _check(buffers, kv_indices, table, lens, 2048)
    # request 0 has 300 tokens -> 3 kernel pages -> the first 3 sub-pages of engine page 3
    assert kv_indices[:3].tolist() == [48, 49, 50]
    # request 1 has 2049 tokens -> 17 kernel pages, starting at engine page 1 (sub-pages 16..32)
    assert kv_indices[3:20].tolist() == list(range(16, 33))


def test_single_page_engines_need_no_expansion():
    buffers, kv_indices = _make(block_size=128, engine_width=8)
    table = torch.tensor(
        [[2, 4, 6, 8, 10, 12, 14, 16], [1, 3, 5, 7, 9, 11, 13, 15]], dtype=torch.int32
    )
    lens = torch.tensor([130, 1], dtype=torch.int32)
    refresh_step(buffers, table, lens, 128, kv_indices)
    _check(buffers, kv_indices, table, lens, 128)
    assert kv_indices[:2].tolist() == [2, 4]


def test_decode_progression_updates_in_place():
    buffers, kv_indices = _make(block_size=2048)
    table = torch.tensor([[3, 4, 5, 6, 7, 8, 9, 10]] * 4, dtype=torch.int32)
    for step, length in enumerate([1, 128, 129, 2048, 2049, 3000]):
        lens = torch.tensor([length], dtype=torch.int32)
        refresh_step(buffers, table, lens, 2048, kv_indices)
        _check(buffers, kv_indices, table, lens, 2048)
        assert int(buffers.pages_buf[0]) == -(-length // 128), f"step {step}"


def test_shrinking_batch_resets_padding_rows():
    buffers, kv_indices = _make(block_size=2048)
    table = torch.tensor(
        [[3, 4, 5, 6, 7, 8, 9, 10]] * 4, dtype=torch.int32
    )
    lens4 = torch.tensor([300, 400, 500, 600], dtype=torch.int32)
    refresh_step(buffers, table, lens4, 2048, kv_indices)
    assert int(buffers.pages_buf[2]) == -(-500 // 128)
    # next step: only one request is alive
    refresh_step(buffers, table, lens4[:1], 2048, kv_indices)
    _check(buffers, kv_indices, table, lens4[:1], 2048)


def test_logical_page_order_comes_from_the_table_not_arithmetic():
    buffers, kv_indices = _make(block_size=128)  # one kernel page per engine page
    table = torch.tensor(
        [[0, 0, 0, 0, 0, 0, 0, 0], [2, 1, 0, 0, 0, 0, 0, 0]], dtype=torch.int32
    )
    lens = torch.tensor([128, 256], dtype=torch.int32)
    refresh_step(buffers, table, lens, 128, kv_indices)
    _check(buffers, kv_indices, table, lens, 128)
    # request 1's pages are (2, 1) in logical order - a prefix-caching engine hands them out in
    # arbitrary order, so the kernel must read kv_indices, never request*max_pages + j.
    assert kv_indices[:3].tolist() == [0, 2, 1]


def test_pool_too_small_for_the_trash_slot_is_rejected():
    with pytest.raises(ValueError, match="nowhere to park"):
        allocate_step_buffers(4, 128, 4 * 128, 2048, torch.device("cpu"))


def test_refresh_rejects_bad_arguments():
    buffers, kv_indices = _make(block_size=2048)
    table = torch.zeros((2, 8), dtype=torch.int32)
    lens = torch.tensor([1, 1], dtype=torch.int32)
    with pytest.raises(ValueError, match="outside"):
        refresh_step(buffers, table, lens, 2048, kv_indices, num_reqs=0)
    with pytest.raises(ValueError, match="outside"):
        refresh_step(buffers, table, lens, 2048, kv_indices, num_reqs=99)
    with pytest.raises(ValueError, match="must be int32"):
        refresh_step(buffers, table.to(torch.int64), lens, 2048, kv_indices)
    with pytest.raises(ValueError, match="does not cover"):
        refresh_step(buffers, table[:, :1], lens, 2048, kv_indices)
    with pytest.raises(ValueError, match="kv_indices has"):
        refresh_step(buffers, table, lens, 2048, kv_indices[:128])
