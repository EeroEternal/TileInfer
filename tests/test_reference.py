"""The tiled reference must agree with the straight-line oracle.

If these fail, the *plan* is wrong and there is no point looking at kernels.
"""

from __future__ import annotations

import pytest
import torch

from tileinfer.metadata import RaggedMetadata
from tileinfer.plan import plan_decode_tiles, plan_query_tiles
from tileinfer.testing.cache import build_paged_cache, random_q
from tileinfer.testing.reference import reference_attention, reference_attention_tiled

# fp16 outputs carry ~1e-3 relative error, so the oracle comparisons below use fp16-appropriate
# tolerances.  They are still tight enough to catch masking/offset bugs, which produce O(1) errors.
RTOL, ATOL = 1e-2, 2e-3


def assert_close(a, b, msg=""):
    a = a.float()
    b = b.float()
    diff = (a - b).abs().max().item()
    assert torch.allclose(a, b, rtol=RTOL, atol=ATOL), f"{msg} max abs diff {diff:.3e}"



def _paged_case(batch, h_q, h_kv, qo_len, kv_len, page_size, shuffle=False, dtype=torch.float16):
    k = torch.randn(batch, h_kv, kv_len, 64, dtype=dtype)
    v = torch.randn(batch, h_kv, kv_len, 64, dtype=dtype)
    k_cache, v_cache, meta = build_paged_cache(
        k, v, page_size=page_size, shuffle_pages=shuffle, seed=7
    )
    q = random_q(batch, h_q, qo_len, 64, dtype=dtype)
    return q, k_cache, v_cache, meta


def test_prefill_matches_dense_oracle():
    q, k_cache, v_cache, meta = _paged_case(2, 4, 2, 16, 128, 32)
    qo_indptr = torch.tensor([0, 16, 32], dtype=torch.int32)
    meta.qo_indptr = qo_indptr

    dense = reference_attention(q, k_cache, v_cache, meta, causal=True)
    sched = plan_query_tiles(
        qo_indptr[1:] - qo_indptr[:-1],
        meta.kv_indptr[1:] - meta.kv_indptr[:-1],
        block_q=8,
    )
    tiled, _, _ = reference_attention_tiled(q, k_cache, v_cache, meta, sched, causal=True)
    assert_close(dense, tiled, "tiled prefill vs dense oracle")


def test_tail_page_is_masked_not_read_as_garbage():
    """A non-multiple KV length must not leak padding into the softmax."""
    q, k_cache, v_cache, meta = _paged_case(1, 2, 1, 1, 100, 128)
    assert meta.kv_lens.tolist() == [100]
    qo_indptr = torch.tensor([0, 1], dtype=torch.int32)
    meta.qo_indptr = qo_indptr

    # poison the padding region of the last page: correct code never reads it
    k_cache[0, 100:, 0, :] = 1e4
    v_cache[0, 100:, 0, :] = 1e4

    dense = reference_attention(q, k_cache, v_cache, meta, causal=True)
    assert torch.isfinite(dense).all()
    assert dense.abs().max() < 100


def test_shuffled_pages_give_identical_result():
    q = random_q(1, 2, 1, 64)
    k = torch.randn(1, 1, 256, 64, dtype=torch.float16)
    v = torch.randn(1, 1, 256, 64, dtype=torch.float16)
    k_a, v_a, meta_a = build_paged_cache(k, v, page_size=64, shuffle_pages=False)
    k_b, v_b, meta_b = build_paged_cache(k, v, page_size=64, shuffle_pages=True, seed=3)
    meta_a.qo_indptr = meta_b.qo_indptr = torch.tensor([0, 1], dtype=torch.int32)

    out_a = reference_attention(q, k_a, v_a, meta_a)
    out_b = reference_attention(q, k_b, v_b, meta_b)
    assert_close(out_a, out_b, "shuffled pages vs contiguous pages")


def test_split_kv_decode_matches_single_pass():
    batch, h_q, h_kv, kv_len, page_size = 3, 4, 2, 320, 64
    k = torch.randn(batch, h_kv, kv_len, 64, dtype=torch.float16)
    v = torch.randn(batch, h_kv, kv_len, 64, dtype=torch.float16)
    k_cache, v_cache, meta = build_paged_cache(k, v, page_size=page_size, shuffle_pages=True, seed=11)
    q = random_q(batch, h_q, 1, 64)
    meta.qo_indptr = torch.arange(batch + 1, dtype=torch.int32)

    oracle = reference_attention(q, k_cache, v_cache, meta)

    pages = meta.kv_indptr[1:] - meta.kv_indptr[:-1]
    sched = plan_decode_tiles(pages, kv_tile_pages=2)  # 2 pages per tile -> 3 splits
    assert sched.needs_merge
    split, partial_out, partial_lse = reference_attention_tiled(
        q, k_cache, v_cache, meta, sched, return_partials=True
    )
    assert partial_out is not None and partial_lse is not None
    assert_close(oracle, split.reshape_as(oracle), "split-KV merge vs single pass")


def test_gqa_grouping_matches_head_repeat():
    """GQA must be exactly 'repeat KV heads', not an approximation."""
    batch, h_kv, kv_len, d = 1, 2, 64, 32
    group = 3
    k = torch.randn(batch, h_kv, kv_len, d, dtype=torch.float16)
    v = torch.randn(batch, h_kv, kv_len, d, dtype=torch.float16)
    k_cache, v_cache, meta = build_paged_cache(k, v, page_size=32)
    q = random_q(batch, h_kv * group, 1, d, dtype=torch.float16)
    meta.qo_indptr = torch.tensor([0, 1], dtype=torch.int32)

    ours = reference_attention(q, k_cache, v_cache, meta).float()

    # brute force: materialise per-head attention with expanded KV heads.
    # KV as [tokens, head, dim] so that head h of the query pairs with head h of the KV
    kk = k[0].permute(1, 0, 2).repeat_interleave(group, dim=1).float()
    vv = v[0].permute(1, 0, 2).repeat_interleave(group, dim=1).float()
    qq = q[0].float().permute(1, 0, 2)  # [qo, H, D]
    scores = torch.einsum("qhd,khd->hqk", qq, kk) / d**0.5
    probs = torch.softmax(scores, dim=-1)
    expected = torch.einsum("hqk,khd->qhd", probs, vv)
    assert_close(ours[0, :, 0, :], expected, "GQA grouping vs head repeat")


def test_append_uses_correct_causal_offset():
    """Chunked prefill: the chunk's row 0 sees the cached prefix, not position 0."""
    batch, h_q, h_kv, qo_len, kv_len, page_size, d = 1, 4, 2, 4, 12, 8, 32
    k = torch.randn(batch, h_kv, kv_len, d, dtype=torch.float16)
    v = torch.randn(batch, h_kv, kv_len, d, dtype=torch.float16)
    k_cache, v_cache, meta = build_paged_cache(k, v, page_size=page_size)
    meta.qo_indptr = torch.tensor([0, qo_len], dtype=torch.int32)
    q = random_q(batch, h_q, qo_len, d, dtype=torch.float16)

    got = reference_attention(q, k_cache, v_cache, meta, causal=True).float()

    # reference by hand, with positions offset by kv_len - qo_len
    kk = k[0].permute(1, 0, 2).repeat_interleave(h_q // h_kv, dim=1).float()
    vv = v[0].permute(1, 0, 2).repeat_interleave(h_q // h_kv, dim=1).float()
    qq = q[0].float().permute(1, 0, 2)
    scores = torch.einsum("qhd,khd->hqk", qq, kk) / d**0.5
    pos = torch.arange(kv_len - qo_len, kv_len)
    mask = torch.arange(kv_len)[None, :] > pos[:, None]
    scores = scores.masked_fill(mask, float("-inf"))
    expected = torch.einsum("hqk,khd->qhd", torch.softmax(scores, -1), vv).permute(1, 0, 2)
    assert_close(got[0], expected, "append causal offset")
