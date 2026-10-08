"""Decode kernel tests — these are the ones that must run on the Ascend machine.

They are skipped (not failed) when no NPU/TileLang is present, so that the same suite is usable
on a laptop and on the 950.
"""

from __future__ import annotations

import pytest
import torch

from tileinfer.utils import have_tilelang, is_npu_available

pytestmark = [
    pytest.mark.npu,
    pytest.mark.skipif(not is_npu_available(), reason="no Ascend NPU available"),
    pytest.mark.skipif(not have_tilelang(), reason="no TileLang Ascend toolchain"),
]


@pytest.fixture
def npu():
    return torch.device("npu", 0)


def _make_case(batch, h_q, h_kv, kv_len, page_size, d, npu, seed=0, shuffle=True):
    from tileinfer.testing.cache import build_paged_cache, random_q

    torch.manual_seed(seed)
    k = torch.randn(batch, h_kv, kv_len, d, dtype=torch.float16, device=npu)
    v = torch.randn(batch, h_kv, kv_len, d, dtype=torch.float16, device=npu)
    k_cache, v_cache, meta = build_paged_cache(
        k, v, page_size=page_size, shuffle_pages=shuffle, seed=seed, device=npu
    )
    q = random_q(batch, h_q, 1, d, dtype=torch.float16, device=npu)
    return q, k_cache, v_cache, meta


@pytest.mark.parametrize(
    "batch,h_q,h_kv,kv_len,page_size,d",
    [
        (1, 4, 2, 128, 128, 128),
        (4, 32, 8, 512, 128, 128),
        (2, 8, 2, 200, 64, 128),  # partial tail page
        (1, 16, 4, 1024, 256, 64),
    ],
)
def test_decode_matches_reference(batch, h_q, h_kv, kv_len, page_size, d, npu):
    from tileinfer import BatchAttention
    from tileinfer.testing.reference import reference_attention

    q, k_cache, v_cache, meta = _make_case(batch, h_q, h_kv, kv_len, page_size, d, npu)
    meta.qo_indptr = torch.arange(batch + 1, dtype=torch.int32, device=npu)

    attn = BatchAttention(backend="tilelang", device=npu, dtype=torch.float16)
    plan = attn.plan(
        kv_indptr=meta.kv_indptr,
        kv_indices=meta.kv_indices,
        kv_last_page_len=meta.kv_last_page_len,
        page_size=page_size,
        num_qo_heads=h_q,
        num_kv_heads=h_kv,
        head_dim=d,
        k_cache=k_cache,
        v_cache=v_cache,
    )
    got = attn.run(q, k_cache, v_cache, plan=plan)

    expected = reference_attention(q, k_cache, v_cache, meta, causal=False)
    torch.npu.synchronize()
    diff = (got.float() - expected.float()).abs().max().item()
    assert diff < 5e-2, f"max abs diff {diff} too large"


def test_plan_reuse_avoids_recompilation(npu):
    from tileinfer import BatchAttention

    q, k_cache, v_cache, meta = _make_case(2, 8, 2, 256, 128, 128, npu)
    meta.qo_indptr = torch.arange(3, dtype=torch.int32, device=npu)
    attn = BatchAttention(backend="tilelang", device=npu)
    kwargs = dict(
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim=128,
        page_size=128,
        k_cache=k_cache,
        v_cache=v_cache,
    )
    plan = attn.plan(
        kv_indptr=meta.kv_indptr,
        kv_indices=meta.kv_indices,
        kv_last_page_len=meta.kv_last_page_len,
        **kwargs,
    )
    spec_key = plan.backend_state["spec_key"]
    attn.run(q, k_cache, v_cache, plan=plan)
    plan2 = attn.plan(
        kv_indptr=meta.kv_indptr,
        kv_indices=meta.kv_indices,
        kv_last_page_len=meta.kv_last_page_len,
        **kwargs,
    )
    assert plan2.backend_state["spec_key"] == spec_key
