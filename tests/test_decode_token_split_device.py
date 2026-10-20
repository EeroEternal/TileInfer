"""H67 token 级 decode split — 设备数值验证(npu 标记)。

CPU 契约(tests/test_decode_token_split.py)锁调度语义;本文件锁**设备数值**:

1. 单页短上下文(c1 的结构性场景):页级 split 恒为 1 tile(plain 路径),
   token 级(tps=32)切出 4 tile(merge 路径)—— 两者都必须逐位对上 oracle;
2. token 窗口在 **SIMD 与非 SIMD softmax 两个内核分支**都必须生效(E294:
   修复前非 SIMD 分支忽略窗口、静默算全量)。内核 cache key 现已含 SIMD 标志,
   同一进程内先后切换 env 是安全的(不同 key → 不同编译产物);
3. tail 页 / ragged batch 与 `TileSchedule.to()` 的 token 字段保全。
"""

from __future__ import annotations

import pytest
import torch

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
TOL = 3e-2


def _build(batch, kv_heads, group, page_size, dim, kv_lens, seed=0):
    """Ragged paged cache + metadata(与 test_tilelang_ascend950_decode 同构造)。"""
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
    return q, k_cache, v_cache, meta, indptr, last_page_len, total_pages


def _run_plan(q, k_cache, v_cache, meta, batch, kv_heads, group, dim, page_size, **plan_kw):
    from tileinfer import BatchAttention

    attn = BatchAttention(backend="tilelang-ascend950", device=torch.device("npu", 0), dtype=DTYPE)
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
        **plan_kw,
    )
    return attn.run(q, k_cache, v_cache, plan=plan), plan


def _oracle(q, k_cache, v_cache, meta, batch, indptr, last_page_len, page_size):
    from tileinfer.metadata import RaggedMetadata

    host_meta = RaggedMetadata(
        kv_indptr=torch.tensor(indptr, dtype=torch.int32),
        kv_indices=meta.kv_indices.cpu(),
        kv_last_page_len=torch.tensor(last_page_len, dtype=torch.int32),
        page_size=page_size,
    )
    host_meta.qo_indptr = torch.arange(batch + 1, dtype=torch.int32)
    return reference_attention(
        q.cpu().float(), k_cache.cpu().float(), v_cache.cpu().float(), host_meta
    ).float()


@pytest.mark.parametrize("simd", [True, False], ids=["simd", "nosimd"])
def test_h67_single_page_token_split_matches_oracle(monkeypatch, simd):
    """1 页 128 token:页级 split=1(plain),token 级 tps=32 切 4 tile(merge)——都对上 oracle。"""
    monkeypatch.setenv("TILEINFER_ALLOW_SPLIT_KV", "1")
    monkeypatch.setenv("TILEINFER_SIMD_SOFTMAX", "1" if simd else "0")

    batch, kv_heads, group, page, dim = 1, 2, 8, 128, 128
    q, k_cache, v_cache, meta, indptr, lpl, _ = _build(batch, kv_heads, group, page, dim, [page])

    expected = _oracle(q, k_cache, v_cache, meta, batch, indptr, lpl, page)

    got_plain, plan_plain = _run_plan(
        q, k_cache, v_cache, meta, batch, kv_heads, group, dim, page, kv_tile_pages=0
    )
    assert not plan_plain.schedule.needs_merge

    got_tok, plan_tok = _run_plan(
        q, k_cache, v_cache, meta, batch, kv_heads, group, dim, page, tokens_per_split=32
    )
    assert plan_tok.schedule.token_split and plan_tok.schedule.needs_merge
    assert plan_tok.schedule.num_tiles == 4, plan_tok.schedule.num_tiles

    for name, got in (("plain", got_plain), ("token_split", got_tok)):
        diff = (got.cpu().float() - expected).abs().max().item()
        assert diff < TOL, f"simd={simd} {name}: max abs diff {diff:.4f} exceeds {TOL}"


@pytest.mark.parametrize("simd", [True, False], ids=["simd", "nosimd"])
def test_h67_tail_and_ragged_token_split_matches_oracle(monkeypatch, simd):
    """tail 页 + ragged batch:token 窗口跨界页、各请求 split 数不同。"""
    monkeypatch.setenv("TILEINFER_ALLOW_SPLIT_KV", "1")
    monkeypatch.setenv("TILEINFER_SIMD_SOFTMAX", "1" if simd else "0")

    batch, kv_heads, group, page, dim = 2, 2, 8, 128, 128
    kv_lens = [200, 512]
    q, k_cache, v_cache, meta, indptr, lpl, _ = _build(
        batch, kv_heads, group, page, dim, kv_lens, seed=3
    )
    expected = _oracle(q, k_cache, v_cache, meta, batch, indptr, lpl, page)

    got, plan = _run_plan(
        q, k_cache, v_cache, meta, batch, kv_heads, group, dim, page, tokens_per_split=64
    )
    assert plan.schedule.token_split and plan.schedule.needs_merge
    # 200 -> ceil(200/64)=4 块;512 -> 8 块
    assert plan.schedule.num_tiles == 12, plan.schedule.num_tiles

    diff = (got.cpu().float() - expected).abs().max().item()
    assert diff < TOL, f"simd={simd}: max abs diff {diff:.4f} exceeds {TOL}"
