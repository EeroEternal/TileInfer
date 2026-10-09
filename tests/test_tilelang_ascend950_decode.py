"""Device tests for the Ascend 950 paged-decode backend (official TileLang wheel).

These are the tests that were used to land the port; they run on the reference machine as

    source <CANN 9.3.x>/set_env.sh          # e.g. a user-local side-by-side install
    <venv with tilelang>=0.1.15>/bin/python -m pytest tests/test_tilelang_ascend950_decode.py

and they skip (never fail) anywhere the backend is not importable, so the rest of the suite keeps
running on a laptop.

The cases are chosen to be the ones that actually broke during development:

* a **partial tail page** — the classic silent paged-attention bug (padding leaking into the
  softmax denominator);
* a **single-token** request — the heaviest possible masking;
* **several ragged requests** in one batch — exercises the grid and the per-request page lists;
* **group == M-tile** (32) — exercises the path with no padding waste.
"""

from __future__ import annotations

import pytest
import torch

from tileinfer.testing.cache import build_paged_cache
from tileinfer.testing.reference import reference_attention
from tileinfer.utils import is_npu_available

try:  # pragma: no cover
    import tilelang.ascend  # noqa: F401

    _HAVE_ASCEND950 = True
except Exception:  # pragma: no cover
    _HAVE_ASCEND950 = False

pytestmark = [
    pytest.mark.npu,
    pytest.mark.skipif(not is_npu_available(), reason="no Ascend NPU available"),
    pytest.mark.skipif(not _HAVE_ASCEND950, reason="tilelang.ascend (>=0.1.15) not importable"),
]

DTYPE = torch.bfloat16
TOL = 3e-2  # bf16 vs the fp32 torch reference


def _case(batch, kv_heads, group, page_size, dim, kv_lens, seed=0):
    """Build a ragged paged cache plus its metadata, then run TileInfer's public plan/run path.

    The cache blocks are filled directly and each request's page list is shuffled *within the
    request*: the kernel and the reference both address memory through ``kv_indices``, so anything
    the metadata says is what both must agree on — which is exactly what we want to pin down
    (indirection, tail page, per-request lengths).
    """
    from tileinfer import BatchAttention
    from tileinfer.metadata import RaggedMetadata

    device = torch.device("npu", 0)
    torch.manual_seed(seed)

    indptr = [0]
    last_page_len = []
    for length in kv_lens:
        indptr.append(indptr[-1] + -(-length // page_size))
        last_page_len.append(page_size if length % page_size == 0 else length % page_size)
    total_pages = indptr[-1]

    k_cache = torch.randn(total_pages, page_size, kv_heads, dim, dtype=DTYPE, device=device)
    v_cache = torch.randn(total_pages, page_size, kv_heads, dim, dtype=DTYPE, device=device)

    # logical -> physical: identity, then permute inside each request (always a valid page table)
    indices = torch.arange(total_pages, dtype=torch.int32, device=device)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    for b in range(batch):
        start, end = indptr[b], indptr[b + 1]
        perm = torch.randperm(end - start, generator=gen) + start
        indices[start:end] = indices[start:end][perm.to(device)]

    meta = RaggedMetadata(
        kv_indptr=torch.tensor(indptr, dtype=torch.int32, device=device),
        kv_indices=indices,
        kv_last_page_len=torch.tensor(last_page_len, dtype=torch.int32, device=device),
        page_size=page_size,
    )
    q = torch.randn(batch, kv_heads * group, 1, dim, dtype=DTYPE, device=device) * 0.2

    attn = BatchAttention(backend="tilelang-ascend950", device=device, dtype=DTYPE)
    plan = attn.plan(
        kv_indptr=meta.kv_indptr,
        kv_indices=meta.kv_indices,
        kv_last_page_len=meta.kv_last_page_len,
        page_size=page_size,
        num_qo_heads=kv_heads * group,
        num_kv_heads=kv_heads,
        head_dim=dim,
        causal=False,
        k_cache=k_cache,
        v_cache=v_cache,
    )
    got = attn.run(q, k_cache, v_cache, plan=plan)
    torch.npu.synchronize()

    host_meta = RaggedMetadata(
        kv_indptr=meta.kv_indptr.cpu(),
        kv_indices=meta.kv_indices.cpu(),
        kv_last_page_len=meta.kv_last_page_len.cpu(),
        page_size=page_size,
    )
    host_meta.qo_indptr = torch.arange(batch + 1, dtype=torch.int32)
    expected = reference_attention(
        q.cpu().float(), k_cache.cpu().float(), v_cache.cpu().float(), host_meta
    ).float()
    return got.cpu().float(), expected


@pytest.mark.parametrize(
    "batch,kv_heads,group,kv_lens,label",
    [
        (1, 2, 8, [128], "one full page"),
        (1, 2, 8, [200], "partial tail page"),
        (1, 2, 8, [1], "single token"),
        (2, 2, 8, [128, 256], "ragged batch"),
        (1, 1, 32, [256], "group == M tile"),
    ],
)
def test_paged_decode_matches_reference(batch, kv_heads, group, kv_lens, label):
    got, expected = _case(batch, kv_heads, group, 128, 128, kv_lens)
    diff = (got - expected).abs().max().item()
    assert diff < TOL, f"{label}: max abs diff {diff:.4f} exceeds {TOL}"
