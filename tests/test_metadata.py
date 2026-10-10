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


def test_ragged_from_page_table_device_matches_the_host_conversion():
    """The device-side conversion must agree with the (slow) host one, padding and all."""
    from tileinfer.metadata import ragged_from_page_table_device

    seq_lens = torch.tensor([130, 256, 1, 128], dtype=torch.int32)
    table = PageTable.from_seq_lens(seq_lens, page_size=128).table

    host = PageTable(table=table, seq_lens=seq_lens, page_size=128).to_ragged()
    device = ragged_from_page_table_device(table, seq_lens, page_size=128)

    assert device.kv_indptr.tolist() == host.kv_indptr.tolist()
    assert device.kv_indices.tolist() == host.kv_indices.tolist()
    assert device.kv_last_page_len.tolist() == host.kv_last_page_len.tolist()
    device.validate()


def test_ragged_from_page_table_device_with_a_perturbed_table():
    """A table whose contents moved (the serving case) converts to the moved contents."""
    from tileinfer.metadata import ragged_from_page_table_device

    seq_lens = torch.tensor([300, 100], dtype=torch.int32)
    table = PageTable.from_seq_lens(seq_lens, page_size=128).table
    table = table[:, :3].clone()
    table[0, 0], table[1, 0] = 7, 9  # values change between steps

    meta = ragged_from_page_table_device(table, seq_lens, page_size=128)
    assert meta.kv_indptr.tolist() == [0, 3, 4]
    assert meta.kv_indices.tolist() == [7, 1, 2, 9]
    assert meta.kv_last_page_len.tolist() == [44, 100]


def test_pad_batch_keeps_the_live_requests_and_makes_padding_valid():
    from tileinfer.metadata import pad_batch

    meta = RaggedMetadata.contiguous([128, 200], page_size=128, qo_lens=[1, 1])
    padded = pad_batch(meta, bucket=4, page_size=128)

    assert padded.batch_size == 4
    # the live part is untouched
    assert padded.kv_lens.tolist()[:2] == [128, 200]
    assert padded.kv_indptr.tolist() == [0, 1, 3, 4, 5]
    # padded requests are *valid*: one page, one token (otherwise the softmax divides by zero)
    assert padded.kv_last_page_len.tolist()[2:] == [1, 1]
    assert padded.kv_lens.tolist()[2:] == [1, 1]
    assert padded.qo_lens.tolist() == [1, 1, 1, 1]
    padded.validate()


def test_vllm_integration_imports_without_vllm_and_buckets():
    """The integration module must import in a plain environment (vLLM is optional) and its bucket
    lookup must never silently overflow."""
    import tileinfer.integrations.vllm_ascend as ti

    assert ti._bucket_for(1) == 1
    assert ti._bucket_for(3) == 4
    assert ti._bucket_for(256) == 256
    with pytest.raises(ValueError, match="exceeds the largest TileInfer bucket"):
        ti._bucket_for(257)
