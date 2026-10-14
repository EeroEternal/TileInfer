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

import os

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


def _case(batch, kv_heads, group, page_size, dim, kv_lens, seed=0, kv_tile_pages=0):
    """Run one case through the public API.  Split-KV cases opt in explicitly (see KI-1)."""
    if kv_tile_pages:
        os.environ["TILEINFER_ALLOW_SPLIT_KV"] = "1"
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

    # logical -> physical: identity, then permute inside each request (always a valid page table).
    # NOTE: the permutation must be 0-based for the slice it indexes -- adding `start` here sent the
    # gather out of range, and Ascend does not bounds-check gathers: it returned garbage page ids
    # that looked exactly like a kernel scribbling over memory (see docs/known-issues.md).
    indices = torch.arange(total_pages, dtype=torch.int32, device=device)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    for b in range(batch):
        start, end = indptr[b], indptr[b + 1]
        perm = torch.randperm(end - start, generator=gen)
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
        kv_tile_pages=kv_tile_pages,
        k_cache=k_cache,
        v_cache=v_cache,
    )
    got = attn.run(q, k_cache, v_cache, plan=plan)
    torch.npu.synchronize()

    # diagnostic: the reference below reads kv_indices on the host, so any device-side overwrite
    # of that tensor would surface as a garbage page id rather than as a wrong number
    idx_after = meta.kv_indices.cpu()
    assert bool((idx_after >= 0).all()) and int(idx_after.max()) < total_pages, (
        f"kv_indices corrupted after run: {idx_after.tolist()}"
    )

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
    "batch,kv_heads,group,kv_lens,label,kv_tile_pages,dim",
    [
        (1, 2, 8, [128], "one full page", 0, 128),
        (1, 2, 8, [200], "partial tail page", 0, 128),
        (1, 2, 8, [1], "single token", 0, 128),
        (2, 2, 8, [128, 256], "ragged batch", 0, 128),
        (1, 1, 32, [256], "group == M tile", 0, 128),
        # split-KV: 32 pages -> 4 tiles, the tail lands in the last split
        (1, 2, 8, [4000], "split-KV, 4 tiles", 8, 128),
        # split-KV across several ragged requests (different split counts per request)
        (2, 2, 8, [4096, 1024], "split-KV, ragged", 8, 128),
        # split-KV with the *serving* GQA config (32 qo / 8 kv heads -> group 4): this is the
        # KI-1 regression case - the merge kernel's GM staging used to fault with a vector-core
        # exception for group 4 (sub-32-byte PartLse row) while group 8 passed
        (1, 8, 4, [4096], "split-KV, serving group 4", 16, 128),
        (2, 8, 4, [4096, 512], "split-KV, group 4 ragged", 16, 128),
        # head_dim 256 (Qwen3.5/3.6 full-attention layers): non-power-of-two pool on purpose,
        # it is the regression case for the kernel's quantized page-capacity binding
        (1, 2, 8, [128], "d256 one full page", 0, 256),
        (1, 2, 8, [200], "d256 partial tail page", 0, 256),
        (2, 2, 8, [128, 256], "d256 ragged batch", 0, 256),
        (1, 1, 32, [256], "d256 group == M tile", 0, 256),
        # d256 + non-power-of-two pool (3 pages, kernel compiled for 4) - ABI binding regression
        (2, 2, 8, [130, 300], "d256 ragged, odd pool", 0, 256),
    ],
)
def test_paged_decode_matches_reference(batch, kv_heads, group, kv_lens, label, kv_tile_pages, dim):
    got, expected = _case(batch, kv_heads, group, 128, dim, kv_lens, kv_tile_pages=kv_tile_pages)
    diff = (got - expected).abs().max().item()
    assert diff < TOL, f"{label}: max abs diff {diff:.4f} exceeds {TOL}"


# -------------------------------------------------------------------------------------------
# L2: pool-agnostic kernel ABI — the declared capacity is quantized, so kernel/plan/pre-warm
# keys never contain vLLM's snapshot-derived pool size, and a kernel compiled for a declared
# cap larger than the real pool stays numerically exact (pages beyond the pool are never
# dereferenced; the pool is presented through the zero-copy ``pool_view``).
# -------------------------------------------------------------------------------------------


def test_declared_pool_cap_semantics(monkeypatch):
    from tileinfer.kernels.attention.paged_decode_ascend950 import declared_pool_cap

    assert declared_pool_cap(1) == 1
    assert declared_pool_cap(2) == 2
    assert declared_pool_cap(3) == 4
    assert declared_pool_cap(30) == 32
    # the two snapshot-derived pool sizes actually observed on the reference box: same bucket
    assert declared_pool_cap(9425) == declared_pool_cap(9899) == 16384
    for pool in (1, 2, 3, 17, 100, 8192, 8193, 9425, 9899):
        cap = declared_pool_cap(pool)
        assert cap >= pool and cap & (cap - 1) == 0, pool
    # an explicit pin wins; going below the real pool is a hard error
    monkeypatch.setenv("TILEINFER_POOL_CAP", "64")
    assert declared_pool_cap(50) == 64
    with pytest.raises(ValueError):
        declared_pool_cap(65)


def test_pool_view_capacity_view():
    from tileinfer.kernels.attention.paged_decode_ascend950 import declared_pool_cap, pool_view

    pool = torch.randn(3, 128, 2, 8)
    cap = declared_pool_cap(3)
    view = pool_view(pool, cap)
    # capacity view: declared extent, same storage/strides, real pages untouched
    assert view.shape == (cap, 128, 2, 8)
    assert view.data_ptr() == pool.data_ptr()
    assert view.is_contiguous()
    assert torch.equal(view[:3], pool)
    assert pool_view(pool, 3).shape == (3, 128, 2, 8)
    with pytest.raises(ValueError):
        pool_view(pool, 2)  # declared capacity below the real pool is a caller bug


@pytest.mark.parametrize(
    "builder,extra",
    [
        ("build_decode_kernel", {}),
        ("build_split_kernel", {"num_tiles": 4, "max_splits": 2}),
        ("build_prefill_kernel", {"num_tiles": 4}),
    ],
)
def test_kernel_cache_keys_ignore_exact_pool_size(monkeypatch, builder, extra):
    """Two snapshot-drifted pool sizes must compile one artifact (same declared cap)."""
    import tileinfer.kernels.attention.paged_decode_ascend950 as K

    def _fake_compile(fn, **kwargs):
        return object()

    monkeypatch.setattr(K.tilelang, "compile", _fake_compile)
    common = dict(batch=7, kv_heads=2, group=4, dim=128, page_size=128, dtype="bfloat16", **extra)
    before = len(K._KERNEL_CACHE)
    getattr(K, builder)(**common, num_pages_cap=9425)
    getattr(K, builder)(**common, num_pages_cap=9899)
    # one entry only: the second size hit the first one's cache slot
    assert len(K._KERNEL_CACHE) == before + 1
    if builder == "build_decode_kernel":
        getattr(K, builder)(**common, num_pages_cap=9425, out_manual=True)
        assert len(K._KERNEL_CACHE) == before + 2  # manual-out is a distinct artifact


@pytest.mark.parametrize("kv_tile_pages", [0, 8])
def test_declared_cap_beyond_pool_is_exact(monkeypatch, kv_tile_pages):
    """Device binding regression: a 5-page pool served by kernels compiled for cap=64."""
    monkeypatch.setenv("TILEINFER_POOL_CAP", "64")
    got, expected = _case(2, 2, 8, 128, 128, [130, 300], kv_tile_pages=kv_tile_pages)
    diff = (got - expected).abs().max().item()
    assert diff < TOL, f"declared-cap binding (kv_tile_pages={kv_tile_pages}): {diff:.4f} > {TOL}"


# -------------------------------------------------------------------------------------------
# P8 (L3): one *full* split plan serves any page-count distribution.  Graph capture pins one
# plan per bucket while replay-time sequences grow a page per step, so the plan's schedule is
# built as if every request spanned the whole table width; the split kernel clamps each tile to
# the pages that actually exist (empty tiles write the merge's zero-weight filler) and the merge
# reads the same slots regardless.  This is the device-level contract behind the integration's
# `_build_plan` full-plan path and the TILEINFER_KV_TILE_PAGES opt-in.
# -------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "batch,kv_heads,group,dim,width_pages,kv_tile_pages,kv_lens,label",
    [
        (2, 8, 4, 128, 8, 2, [260, 130], "full plan, ragged lens, empty tail splits"),
        (2, 8, 4, 128, 8, 2, [1, 128], "full plan, near-empty batch"),
        (1, 8, 4, 128, 16, 4, [513], "full plan, partial last split"),
        (4, 8, 4, 128, 16, 4, [2050, 1024, 129, 1], "full plan, serving-config skew"),
    ],
)
def test_full_split_plan_clamps_to_real_pages(
    batch, kv_heads, group, dim, width_pages, kv_tile_pages, kv_lens, label
):
    os.environ["TILEINFER_ALLOW_SPLIT_KV"] = "1"
    from tileinfer import BatchAttention
    from tileinfer.metadata import RaggedMetadata

    device = torch.device("npu", 0)
    torch.manual_seed(7)
    page_size = 128

    def _meta(lens, pool_pages):
        indptr = [0]
        lpl = []
        for length in lens:
            indptr.append(indptr[-1] + -(-length // page_size))
            lpl.append(page_size if length % page_size == 0 else length % page_size)
        n = indptr[-1]
        ids = torch.arange(pool_pages, dtype=torch.int32, device=device)
        idx = ids.repeat(-(-n // pool_pages))[:n].contiguous()
        return RaggedMetadata(
            kv_indptr=torch.tensor(indptr, dtype=torch.int32, device=device),
            kv_indices=idx,
            kv_last_page_len=torch.tensor(lpl, dtype=torch.int32, device=device),
            page_size=page_size,
        )

    # the pool must hold at least batch*width pages (the full plan's kv_indices slot is
    # bucket*width long — in the engine bucket*width <= pool holds by the trash-slot check)
    pool_pages = batch * width_pages + 1
    k_cache = torch.randn(pool_pages, page_size, kv_heads, dim, dtype=DTYPE, device=device)
    v_cache = torch.randn_like(k_cache)
    real = _meta(kv_lens, pool_pages)
    # the plan-side lens describe the *full* table width (what `_build_plan` does); the values
    # are irrelevant beyond their shape — run() re-binds the real metadata every step
    full = _meta([width_pages * page_size] * batch, pool_pages)
    q = torch.randn(batch, kv_heads * group, 1, dim, dtype=DTYPE, device=device) * 0.2

    attn = BatchAttention(backend="tilelang-ascend950", device=device, dtype=DTYPE)
    plan = attn.plan(
        kv_indptr=full.kv_indptr,
        kv_indices=full.kv_indices,
        kv_last_page_len=full.kv_last_page_len,
        page_size=page_size,
        num_qo_heads=kv_heads * group,
        num_kv_heads=kv_heads,
        head_dim=dim,
        causal=False,
        kv_tile_pages=kv_tile_pages,
        k_cache=k_cache,
        v_cache=v_cache,
    )
    assert plan.schedule.needs_merge
    got = attn.run(
        q,
        k_cache,
        v_cache,
        plan=plan,
        kv_indptr=real.kv_indptr,
        kv_indices=real.kv_indices,
        kv_last_page_len=real.kv_last_page_len,
    )
    torch.npu.synchronize()

    host = RaggedMetadata(
        kv_indptr=real.kv_indptr.cpu(),
        kv_indices=real.kv_indices.cpu(),
        kv_last_page_len=real.kv_last_page_len.cpu(),
        page_size=page_size,
    )
    host.qo_indptr = torch.arange(batch + 1, dtype=torch.int32)
    expected = reference_attention(
        q.cpu().float(), k_cache.cpu().float(), v_cache.cpu().float(), host
    ).float()
    diff = (got.cpu().float() - expected).abs().max().item()
    assert diff < TOL, f"{label}: max abs diff {diff:.4f} exceeds {TOL}"
