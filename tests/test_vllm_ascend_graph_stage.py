"""设备测试:图捕获改造后集成层 staging 的数值正确性(H4 修复回归,B5d)。

背景:稠密 indptr 布局(arange×width)让内核按全宽页数读页,陈旧页 id 被 attend,
输出垃圾/NaN。修复 = 每步设备侧重建真 ragged 三元组(cumsum indptr + scatter 压缩)。
本文件把修复钉死在三件事上:

1. eager 下集成层 ``run()`` 的数值正确性(多请求、不同页数、部分尾页);
2. 填充行不可读性:未用表行塞满越界页 id,输出仍正确——修复前的稠密布局会 attend 它们;
3. ``torch.npu.NPUGraph`` 捕获后更新页表/lens/query/池再 replay 的正确性,含 replay 期
   页数**变化**(数据依赖的 ragged 重建在图内成立)——这是引擎 replay 数据流的库级等价物。

CPU 机器自动跳过;运行方式同其他设备套件(约 2 个内核编译,batch 1/2 × d128)。
"""

from __future__ import annotations

import pytest
import torch

from tileinfer.integrations.vllm_ascend import (
    KERNEL_PAGE_SIZE,
    TileInferDecodeAttention,
    _expand_page_table,
    _pool_sub_view,
)
from tileinfer.metadata import RaggedMetadata
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

KVH, GROUP, DIM = 2, 4, 128
NUM_HEADS = KVH * GROUP
ENGINE_POOL_PAGES = 8  # 引擎页;池按引擎形态分配(与 vLLM 一致),内核用 8*16=128 个子页
KERNEL_POOL_PAGES = ENGINE_POOL_PAGES * (2048 // KERNEL_PAGE_SIZE)  # > bucket(2)x32,trash 槽在外
ENGINE_BLOCK = 2048  # 16 个 128-token 子页,Qwen3.6 混合模型的引擎页
TABLE_WIDTH = 2  # 引擎页表宽(引擎页);内核页宽 = 2*16 = 32
SUB = ENGINE_BLOCK // KERNEL_PAGE_SIZE
SCALE = DIM**-0.5


def _reference(table: torch.Tensor, lens: torch.Tensor, q: torch.Tensor, kc, vc):
    """集成层数据流(引擎页表 + token lens)→ fp32 参考输出 [num, heads, dim]。

    参考在 128-token 子页视图上取数:先展开页表(引擎页 id → 子页 id),再逐行取有效页。
    """
    table_exp = _expand_page_table(table, SUB).cpu()
    kc = _pool_sub_view(kc, SUB)
    vc = _pool_sub_view(vc, SUB)
    lens_host = lens.cpu().tolist()
    indptr = [0]
    lpl = []
    for n in lens_host:
        indptr.append(indptr[-1] + -(-n // KERNEL_PAGE_SIZE))
        lpl.append(KERNEL_PAGE_SIZE if n % KERNEL_PAGE_SIZE == 0 else n % KERNEL_PAGE_SIZE)
    meta = RaggedMetadata(
        kv_indptr=torch.tensor(indptr, dtype=torch.int32),
        kv_indices=torch.cat([table_exp[b, : indptr[b + 1] - indptr[b]] for b in range(len(lens_host))]),
        kv_last_page_len=torch.tensor(lpl, dtype=torch.int32),
        page_size=KERNEL_PAGE_SIZE,
    )
    meta.qo_indptr = torch.arange(len(lens_host) + 1, dtype=torch.int32)
    return reference_attention(
        q.cpu().float().unsqueeze(2), kc.cpu().float(), vc.cpu().float(), meta, causal=False
    ).squeeze(2)


def _make_impl(device) -> TileInferDecodeAttention:
    return TileInferDecodeAttention(
        num_heads=NUM_HEADS,
        num_kv_heads=KVH,
        head_size=DIM,
        scale=SCALE,
        device=device,
        dtype=DTYPE,
        kv_tile_pages=0,
        max_num_reqs=2,  # buckets 1,2 get prewarmed; every case below runs on bucket 2
    )


def _run_case(impl, device, table, lens, q, kc, vc):
    out = torch.empty(table.shape[0], NUM_HEADS, DIM, dtype=DTYPE, device=device)
    got = impl.run(
        query=q,
        key_cache=kc,
        value_cache=vc,
        block_tables=table,
        seq_lens=lens,
        block_size=ENGINE_BLOCK,
        output=out,
    )
    torch.npu.synchronize()
    assert got is not None, "run() fell back (plan missing) - prewarm did not cover this shape"
    return got


def test_eager_ragged_correctness():
    """多请求、页数不同、部分尾页:eager 数值 vs fp32 参考。"""
    device = torch.device("npu", 0)
    torch.manual_seed(7)
    impl = _make_impl(device)
    kc = torch.randn(ENGINE_POOL_PAGES, ENGINE_BLOCK, KVH, DIM, dtype=DTYPE, device=device)
    vc = torch.randn_like(kc)
    impl.prewarm(width=TABLE_WIDTH * SUB, block_size=ENGINE_BLOCK, key_cache=kc)

    lens = torch.tensor([300, 129], dtype=torch.int32, device=device)  # 3 页 + 2 页(尾页 1)
    table = torch.randint(0, ENGINE_POOL_PAGES, (2, TABLE_WIDTH), dtype=torch.int32, device=device)
    table[0, 3:] = 0  # 修复后不可达;塞 0 便于诊断
    q = torch.randn(2, NUM_HEADS, DIM, dtype=DTYPE, device=device) * 0.2
    got = _run_case(impl, device, table, lens, q, kc, vc)
    exp = _reference(table, lens, q, kc, vc)
    diff = (got.cpu().float() - exp).abs().max().item()
    assert diff < TOL, f"eager mismatch: max|diff|={diff}"


def test_padding_row_never_read():
    """num_reqs < bucket:未用表行塞越界 id,输出仍正确(H4 修复前的直接失败场景)。"""
    device = torch.device("npu", 0)
    torch.manual_seed(11)
    impl = _make_impl(device)
    kc = torch.randn(ENGINE_POOL_PAGES, ENGINE_BLOCK, KVH, DIM, dtype=DTYPE, device=device)
    vc = torch.randn_like(kc)
    impl.prewarm(width=TABLE_WIDTH * SUB, block_size=ENGINE_BLOCK, key_cache=kc)

    lens = torch.tensor([77, 1], dtype=torch.int32, device=device)  # 第二行是填充
    table = torch.randint(0, ENGINE_POOL_PAGES, (2, TABLE_WIDTH), dtype=torch.int32, device=device)
    table[1].fill_(ENGINE_POOL_PAGES + 999)  # 越界页 id:修复前会被 attend(垃圾/NaN 的来源)
    q = torch.randn(1, NUM_HEADS, DIM, dtype=DTYPE, device=device) * 0.2
    got = _run_case(impl, device, table[:1], lens[:1], q, kc, vc)
    exp = _reference(table[:1], lens[:1], q, kc, vc)
    diff = (got[:1].cpu().float() - exp).abs().max().item()
    assert diff < TOL, f"padding-row mismatch: max|diff|={diff}"
    assert bool(torch.isfinite(got.float()).all()), "non-finite output leaked from padding rows"


def test_graph_capture_replay_data_updates():
    """NPUGraph 捕获 → 原地更新页表/lens/query/池 → replay,含 replay 期页数变化。"""
    device = torch.device("npu", 0)
    torch.manual_seed(3)
    impl = _make_impl(device)
    kc = torch.randn(ENGINE_POOL_PAGES, ENGINE_BLOCK, KVH, DIM, dtype=DTYPE, device=device)
    vc = torch.randn_like(kc)
    impl.prewarm(width=TABLE_WIDTH * SUB, block_size=ENGINE_BLOCK, key_cache=kc)

    # 捕获形态:2 请求(= bucket 2,与引擎 capture 一致),lens 数据无关地进图
    lens_g = torch.tensor([128, 129], dtype=torch.int32, device=device)
    table_g = torch.randint(0, ENGINE_POOL_PAGES, (2, TABLE_WIDTH), dtype=torch.int32, device=device)
    q_g = torch.randn(2, NUM_HEADS, DIM, dtype=DTYPE, device=device) * 0.2
    out_g = torch.empty(2, NUM_HEADS, DIM, dtype=DTYPE, device=device)
    args = dict(
        key_cache=kc,
        value_cache=vc,
        block_tables=table_g,
        seq_lens=lens_g,
        block_size=ENGINE_BLOCK,
        output=out_g,
    )
    got0 = impl.run(query=q_g, **args)
    torch.npu.synchronize()
    exp0 = _reference(table_g, lens_g, q_g, kc, vc)
    assert (got0.cpu().float() - exp0).abs().max().item() < TOL

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        impl.run(query=q_g, **args)
    torch.npu.synchronize()

    # replay 1:页数变化(1 页 → 3 页)+ 全部数据更新
    kc.copy_(torch.randn_like(kc))
    vc.copy_(torch.randn_like(vc))
    table_g.copy_(torch.randint(0, ENGINE_POOL_PAGES, (2, TABLE_WIDTH), dtype=torch.int32, device=device))
    lens_g.copy_(torch.tensor([250, 300], dtype=torch.int32, device=device))  # 2 页 + 3 页
    q_g.copy_(torch.randn(2, NUM_HEADS, DIM, dtype=DTYPE, device=device) * 0.2)
    graph.replay()
    torch.npu.synchronize()
    exp1 = _reference(table_g, lens_g, q_g, kc, vc)
    diff1 = (out_g.cpu().float() - exp1).abs().max().item()
    assert diff1 < TOL, f"replay-1 mismatch: max|diff|={diff1}"

    # replay 2:页数再变(两请求都到 1 页,尾页长度不同)
    table_g.copy_(torch.randint(0, ENGINE_POOL_PAGES, (2, TABLE_WIDTH), dtype=torch.int32, device=device))
    lens_g.copy_(torch.tensor([1, 128], dtype=torch.int32, device=device))
    q_g.copy_(torch.randn(2, NUM_HEADS, DIM, dtype=DTYPE, device=device) * 0.2)
    graph.replay()
    torch.npu.synchronize()
    exp2 = _reference(table_g, lens_g, q_g, kc, vc)
    diff2 = (out_g.cpu().float() - exp2).abs().max().item()
    assert diff2 < TOL, f"replay-2 mismatch: max|diff|={diff2}"
