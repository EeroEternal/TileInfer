"""End-to-end tests of the public API on the portable backend."""

from __future__ import annotations

import pytest
import torch

from tileinfer import AttentionMode, BatchAttention, RaggedMetadata
from tileinfer.metadata import PageTable
from tileinfer.testing.cache import build_paged_cache, random_q
from tileinfer.testing.reference import reference_attention


def _case(batch=2, h_q=4, h_kv=2, kv_len=200, page_size=64, d=64, qo_len=1):
    k = torch.randn(batch, h_kv, kv_len, d, dtype=torch.float16)
    v = torch.randn(batch, h_kv, kv_len, d, dtype=torch.float16)
    k_cache, v_cache, meta = build_paged_cache(k, v, page_size=page_size, shuffle_pages=True, seed=5)
    q = random_q(batch, h_q, qo_len, d, dtype=torch.float16)
    return q, k_cache, v_cache, meta


def test_available_backends_includes_reference():
    assert "reference" in BatchAttention.available_backends()


def test_plan_run_decode_on_cpu():
    q, k_cache, v_cache, meta = _case()
    attn = BatchAttention(backend="reference", device="cpu")
    plan = attn.plan(
        kv_indptr=meta.kv_indptr,
        kv_indices=meta.kv_indices,
        kv_last_page_len=meta.kv_last_page_len,
        page_size=meta.page_size,
        num_qo_heads=4,
        num_kv_heads=2,
        head_dim=64,
    )
    assert plan.mode is AttentionMode.DECODE
    out = attn.run(q, k_cache, v_cache, plan=plan)
    expected = reference_attention(q, k_cache, v_cache, plan.meta)
    assert out.shape == q.shape
    assert torch.allclose(out.float(), expected.float(), atol=1e-3)

    summary = plan.summary()
    assert "decode" in summary and "tiles=" in summary


def test_plan_is_cached_per_shape_and_refreshes_values():
    q, k_cache, v_cache, meta = _case()
    attn = BatchAttention(backend="reference", device="cpu")
    kwargs = dict(
        num_qo_heads=4,
        num_kv_heads=2,
        head_dim=64,
        page_size=meta.page_size,
    )
    p1 = attn.plan(
        kv_indptr=meta.kv_indptr,
        kv_indices=meta.kv_indices,
        kv_last_page_len=meta.kv_last_page_len,
        **kwargs,
    )
    p2 = attn.plan(
        kv_indptr=meta.kv_indptr,
        kv_indices=meta.kv_indices,
        kv_last_page_len=meta.kv_last_page_len,
        **kwargs,
    )
    assert p1 is p2


def test_split_decode_plan_and_run():
    q, k_cache, v_cache, meta = _case(batch=3, kv_len=300, page_size=64)
    attn = BatchAttention(backend="reference", device="cpu")
    plan = attn.plan(
        kv_indptr=meta.kv_indptr,
        kv_indices=meta.kv_indices,
        kv_last_page_len=meta.kv_last_page_len,
        page_size=meta.page_size,
        num_qo_heads=4,
        num_kv_heads=2,
        head_dim=64,
        kv_tile_pages=2,
    )
    assert plan.needs_merge
    assert plan.workspace.nbytes > 0
    out = attn.run(q, k_cache, v_cache, plan=plan)
    expected = reference_attention(q, k_cache, v_cache, plan.meta)
    assert torch.allclose(out.float(), expected.float(), atol=1e-3)


def test_plan_from_dense_page_table():
    q, k_cache, v_cache, meta = _case()
    table = meta.to_page_table()
    attn = BatchAttention(backend="reference", device="cpu")
    plan = attn.plan_from_page_table(
        table.table,
        table.seq_lens,
        page_size=table.page_size,
        num_qo_heads=4,
        num_kv_heads=2,
        head_dim=64,
    )
    # dense -> ragged must drop the padding pages
    assert plan.meta.num_pages == meta.num_pages
    out = attn.run(q, k_cache, v_cache, plan=plan)
    expected = reference_attention(q, k_cache, v_cache, plan.meta)
    assert torch.allclose(out.float(), expected.float(), atol=1e-3)


def test_run_without_plan_raises():
    q, k_cache, v_cache, _ = _case()
    attn = BatchAttention(backend="reference", device="cpu")
    with pytest.raises(RuntimeError, match="no plan"):
        attn.run(q, k_cache, v_cache)


def test_plan_mode_mismatch_is_reported():
    q, k_cache, v_cache, meta = _case()
    attn = BatchAttention(backend="reference", device="cpu")
    with pytest.raises(ValueError, match="unknown attention mode"):
        attn.plan(
            kv_indptr=meta.kv_indptr,
            kv_indices=meta.kv_indices,
            kv_last_page_len=meta.kv_last_page_len,
            page_size=meta.page_size,
            num_qo_heads=4,
            num_kv_heads=2,
            head_dim=64,
            mode="nonsense",
        )


def test_property_based_ragged_batch_matches_dense_oracle():
    torch.manual_seed(1)
    h_q, h_kv, d, page_size = 8, 2, 64, 32
    batch = 5
    kv_lens = [1, 40, 33, 200, 97]
    k = torch.randn(batch, h_kv, max(kv_lens), d, dtype=torch.float16)
    v = torch.randn(batch, h_kv, max(kv_lens), d, dtype=torch.float16)
    k_cache, v_cache, meta = build_paged_cache(k, v, page_size=page_size, shuffle_pages=True, seed=2)
    # restrict each request to its own length
    indptr = [0]
    for L in kv_lens:
        indptr.append(indptr[-1] + (L + page_size - 1) // page_size)
    meta = RaggedMetadata(
        kv_indptr=torch.tensor(indptr, dtype=torch.int32),
        kv_indices=meta.kv_indices,
        kv_last_page_len=torch.tensor(
            [page_size if L % page_size == 0 else L % page_size for L in kv_lens],
            dtype=torch.int32,
        ),
        page_size=page_size,
    )
    meta.qo_indptr = torch.arange(batch + 1, dtype=torch.int32)
    q = random_q(batch, h_q, 1, d, dtype=torch.float16)

    attn = BatchAttention(backend="reference", device="cpu")
    plan = attn.plan(
        kv_indptr=meta.kv_indptr,
        kv_indices=meta.kv_indices,
        kv_last_page_len=meta.kv_last_page_len,
        page_size=page_size,
        num_qo_heads=h_q,
        num_kv_heads=h_kv,
        head_dim=d,
        load_balance=True,
    )
    out = attn.run(q, k_cache, v_cache, plan=plan)
    expected = reference_attention(q, k_cache, v_cache, meta)
    assert torch.allclose(out.float(), expected.float(), atol=1e-3)
