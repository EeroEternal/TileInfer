"""Metadata is the contract with every engine — it gets its own tests."""

from __future__ import annotations

import pytest
import torch

from tileinfer.metadata import (
    AttentionMode,
    PageTable,
    RaggedMetadata,
    detect_mode,
    kv_lens_from_indptr,
    pad_to_multiple,
)


def test_kv_lens_from_indptr_token_granularity():
    indptr = torch.tensor([0, 3, 3, 7], dtype=torch.int32)
    lens = kv_lens_from_indptr(indptr, page_size=1)
    assert lens.tolist() == [3, 0, 4]


def test_kv_lens_from_indptr_paged_uses_last_page_len():
    indptr = torch.tensor([0, 2, 5], dtype=torch.int32)
    last = torch.tensor([7, 128], dtype=torch.int32)
    # request 1 spans 3 pages with a full last page: 2 * 128 + 128 = 384
    assert kv_lens_from_indptr(indptr, last, page_size=128).tolist() == [135, 384]


def test_page_table_round_trip_drops_padding():
    seq_lens = torch.tensor([130, 256, 1], dtype=torch.int32)
    table = PageTable.from_seq_lens(seq_lens, page_size=128)
    assert table.table.shape == (3, 2)
    meta = table.to_ragged()
    assert meta.batch_size == 3
    # 3 requests: 2 + 2 + 1 pages
    assert meta.num_pages == 5
    assert meta.kv_lens.tolist() == [130, 256, 1]
    back = meta.to_page_table()
    assert back.table.tolist() == table.table.tolist()
    assert back.seq_lens.tolist() == seq_lens.tolist()


def test_ragged_validate_rejects_inconsistent_indptr():
    bad = RaggedMetadata(
        kv_indptr=torch.tensor([0, 5, 2], dtype=torch.int32),
        kv_indices=torch.arange(5, dtype=torch.int32),
        kv_last_page_len=torch.ones(2, dtype=torch.int32),
        page_size=128,
    )
    with pytest.raises(ValueError, match="non-decreasing"):
        bad.validate()

    over = RaggedMetadata(
        kv_indptr=torch.tensor([0, 9], dtype=torch.int32),
        kv_indices=torch.arange(3, dtype=torch.int32),
        kv_last_page_len=torch.ones(1, dtype=torch.int32),
        page_size=128,
    )
    with pytest.raises(ValueError, match="exceeds the number of"):
        over.validate()


def test_ragged_validate_requires_last_page_len_when_paged():
    meta = RaggedMetadata(
        kv_indptr=torch.tensor([0, 1], dtype=torch.int32),
        kv_indices=torch.tensor([0], dtype=torch.int32),
        page_size=128,
    )
    with pytest.raises(ValueError, match="kv_last_page_len is required"):
        meta.validate()


def test_contiguous_builder_matches_expected_layout():
    meta = RaggedMetadata.contiguous([128, 129, 1], page_size=128, qo_lens=[1, 1, 1])
    assert meta.kv_indptr.tolist() == [0, 1, 3, 4]
    assert meta.kv_indices.tolist() == [0, 1, 2, 3]
    assert meta.kv_last_page_len.tolist() == [128, 1, 1]
    assert meta.qo_lens.tolist() == [1, 1, 1]
    assert meta.max_kv_len == 129


@pytest.mark.parametrize(
    "qo_lens, kv_lens, expected",
    [
        ([1, 1, 1], [10, 20, 30], AttentionMode.DECODE),
        ([10, 20], [10, 20], AttentionMode.PREFILL),
        ([4, 4], [10, 20], AttentionMode.APPEND),
    ],
)
def test_detect_mode(qo_lens, kv_lens, expected):
    qo = torch.tensor(qo_lens, dtype=torch.int32)
    kv = torch.tensor(kv_lens, dtype=torch.int32)
    assert detect_mode(kv, qo) is expected


def test_detect_mode_without_query_metadata_is_decode():
    assert detect_mode(torch.tensor([5, 5], dtype=torch.int32), None) is AttentionMode.DECODE


def test_signature_is_value_independent_but_shape_sensitive():
    a = RaggedMetadata.contiguous([128, 64], page_size=128)
    b = RaggedMetadata.contiguous([256, 64], page_size=128)
    assert a.signature() == a.signature()
    assert a.signature() != b.signature()


def test_pad_to_multiple():
    assert pad_to_multiple(1, 128) == 128
    assert pad_to_multiple(128, 128) == 128
    assert pad_to_multiple(129, 128) == 256
    with pytest.raises(ValueError):
        pad_to_multiple(4, 0)
