"""设备测试:B1 prefill 接线(集成层 ``prefill_step`` 的数值正确性)。

不经过 vLLM,直接以引擎形态喂参数(引擎页 2048-token 池、子页展开、query_start_loc、
全量 seq_lens),覆盖三种引擎状态共有的语义:

1. PrefillNoCache 型:kv_len == qo_len(纯 prefill,K/V 已由 reshape_and_cache 写入池);
2. ChunkedPrefill / PrefillCacheHit 型:append 到已缓存前缀(kv_len > qo_len);
3. 混合批:decode 行(qo=1)+ 两个 prefill 行同批。

池构造:每个请求的完整 KV 序列(缓存前缀 + 本步 chunk)按引擎页写入池;参考值用子页视图
gather 后按请求逐个过 ``reference_attention``(causal=True)。首个新形状走后台编译并返回
False(与引擎内"FIA 先顶、编译就绪后接管"的行为一致),就绪后重试并比对。

CPU 机器自动跳过;真实 head_dim=256 / group=8(Qwen3.6 full-attn 形状)。
"""

from __future__ import annotations

import time

import pytest
import torch

from tileinfer.integrations.vllm_ascend import (
    KERNEL_PAGE_SIZE,
    _expand_page_table,
    _pool_sub_view,
    prefill_step,
)
from tileinfer.metadata import RaggedMetadata
from tileinfer.testing.reference import reference_attention
from tileinfer.utils import is_npu_available

try:  # pragma: no cover
    from tileinfer.integrations import vllm_ascend as _va_mod

    _HAVE_VA = True
except Exception:  # pragma: no cover - CPU 机器无 vllm_ascend 依赖时跳过
    _HAVE_VA = False

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
ENGINE_BLOCK = 2048
SUB = ENGINE_BLOCK // KERNEL_PAGE_SIZE
KVH, GROUP, DIM = 2, 8, 256
HEADS = KVH * GROUP
SCALE = DIM**-0.5


def _run_case(kv_lens: list, qo_lens: list, seed: int):
    device = torch.device("npu", 0)
    gen = torch.Generator().manual_seed(seed)

    # 每请求:完整 KV 序列(前 kv-qo 个 token = 已缓存前缀,后 qo 个 = 本步 chunk),按引擎页入池
    pool: list[torch.Tensor] = []  # 引擎页池(按需增长)
    tables: list[list[int]] = []
    fulls: list[torch.Tensor] = []
    for kv in kv_lens:
        row = list(range(len(pool), len(pool) + -(-kv // ENGINE_BLOCK)))
        pool.extend(
            torch.zeros(ENGINE_BLOCK, KVH, DIM, dtype=DTYPE) for _ in range(len(row))
        )
        tables.append(row)
        full = torch.randn(kv, KVH, DIM, generator=gen).to(DTYPE) * 0.3
        fulls.append(full)
        page = pool[row[0]]
        flat = page.view(-1, KVH, DIM)
        flat[:kv] = full

    num_tokens = sum(qo_lens)
    table = torch.tensor(tables, dtype=torch.int32, device=device)
    seq_lens = torch.tensor(kv_lens, dtype=torch.int32, device=device)
    qsl = torch.tensor([0] + list(torch.cumsum(torch.tensor(qo_lens), 0)), dtype=torch.int32, device=device)
    q = (torch.randn(num_tokens, HEADS, DIM, generator=gen).to(DTYPE) * 0.2).to(device).contiguous()
    out = torch.empty_like(q)
    kc = torch.stack(pool).to(device)
    vc = torch.randn(kc.shape, generator=gen).to(DTYPE).to(device) * 0.3

    served = prefill_step(
        query=q,
        key_cache=kc,
        value_cache=vc,
        block_tables=table,
        seq_lens=seq_lens,
        query_start_loc=qsl,
        block_size=ENGINE_BLOCK,
        output=out,
        num_heads=HEADS,
        num_kv_heads=KVH,
        head_size=DIM,
        scale=SCALE,
    )
    if not served:
        return None  # 首次调用:该形状后台编译中(引擎内此时由 FIA 顶住)

    # 参考值:子页视图 gather 每请求的页 → 逐请求均匀-qo 参考
    table_exp = _expand_page_table(table, SUB).cpu()
    kc_sub = _pool_sub_view(kc, SUB).cpu().float()
    vc_sub = _pool_sub_view(vc, SUB).cpu().float()
    offset = 0
    for b, (kv, qo) in enumerate(zip(kv_lens, qo_lens)):
        pages = -(-kv // KERNEL_PAGE_SIZE)
        page_ids = table_exp[b, :pages].to(torch.long)
        kb = kc_sub[page_ids]  # [pages, 128, kvh, dim]
        vb = vc_sub[page_ids]
        sub_meta = RaggedMetadata(
            kv_indptr=torch.tensor([0, pages], dtype=torch.int32),
            kv_indices=torch.arange(pages, dtype=torch.int32),
            # 尾页长度:整除 → 满页,否则为余数(注意条件方向,写反会把 44 变成 128)
            kv_last_page_len=torch.tensor(
                [kv % KERNEL_PAGE_SIZE or KERNEL_PAGE_SIZE], dtype=torch.int32
            ),
            page_size=KERNEL_PAGE_SIZE,
        )
        sub_meta.qo_indptr = torch.tensor([0, qo], dtype=torch.int32)
        qb = q[offset : offset + qo].cpu().float().permute(1, 0, 2).unsqueeze(0)
        exp = reference_attention(qb, kb, vb, sub_meta, causal=True)
        got = out[offset : offset + qo].cpu().float().permute(1, 0, 2).unsqueeze(0)
        diff = (got - exp).abs().amax(dim=(0, 1, 3))  # 每 qo 行的最大差(跨头/维)
        bad = (diff > TOL).nonzero().flatten().tolist()
        assert not bad, (
            f"request {b} (kv={kv}, qo={qo}) mismatch: max|diff|={diff.max().item()}; "
            f"bad rows ({len(bad)}/{qo}): {bad[:16]}{'...' if len(bad) > 16 else ''}"
        )
        offset += qo
    return True


@pytest.mark.parametrize(
    "kv_lens,qo_lens,label",
    [
        ([300], [300], "prefill-no-cache"),
        ([500, 129], [100, 129], "chunked-append+fresh"),
        ([200, 37, 77], [1, 37, 5], "mixed-decode-row"),
    ],
)
def test_prefill_step_numerics(kv_lens, qo_lens, label, monkeypatch):
    """首调触发后台编译(返回 False/FIA),该形状就绪后数值必须对齐参考。"""
    if _HAVE_VA:
        # H61 终审后 _PREFILL_MAX_CTX 默认 1(全量交还 FIA),默认档下 prefill_step 对
        # kv>128 恒 shunt → served 恒 None,接管等待永不满足。测试期关闭阈值
        # (显式 0 = 全 TI 接管),结束由 monkeypatch 自动还原。
        monkeypatch.setattr(_va_mod, "_PREFILL_MAX_CTX", 0)

    seed = abs(hash(label)) % 1000
    deadline = time.time() + 600  # 每个新形状一次后台编译(约 90-120 s)
    got = None
    while time.time() < deadline:
        got = _run_case(kv_lens, qo_lens, seed)
        if got is not None:
            break
        time.sleep(5)
    assert got is True, f"{label}: TileInfer did not serve the step after compilation"
