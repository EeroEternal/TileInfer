"""子页(KERNEL_PAGE_SIZE)变换的单元测试 —— 纯 CPU,不依赖 NPU。

混合模型(Qwen3.5/3.6 的 GDN 层)会把引擎的 attention block size 强制对齐到 mamba 页
(例如 2048 token),而 TileLang 内核按 128-token 页编译。集成层把 k×128 的引擎页零拷贝地
视为 k 个连续 128-token 子页:页池做 reshape 视图、页表做子页 id 展开。这里验证展开后的
ragged 元数据与"直接按引擎页大小计算"的语义完全等价。
"""

import pytest
import torch

from tileinfer.integrations.vllm_ascend import (
    KERNEL_PAGE_SIZE,
    _expand_page_table,
    _pool_sub_view,
    _sub_pages,
)
from tileinfer.metadata import pad_batch, ragged_from_page_table_device


def test_sub_pages():
    assert _sub_pages(128) == 1
    assert _sub_pages(2048) == 16
    assert _sub_pages(256) == 2
    with pytest.raises(ValueError):
        _sub_pages(100)


def test_expand_page_table_identity_for_sub1():
    table = torch.tensor([[5, 7], [2, 0]], dtype=torch.int32)
    assert torch.equal(_expand_page_table(table, 1), table)


def test_expand_page_table_scales_and_interleaves():
    table = torch.tensor([[5, 7], [2, 0]], dtype=torch.int32)
    got = _expand_page_table(table, 4)
    expected = torch.tensor(
        [[20, 21, 22, 23, 28, 29, 30, 31], [8, 9, 10, 11, 0, 1, 2, 3]], dtype=torch.int32
    )
    assert torch.equal(got, expected)


def test_pool_sub_view_is_zero_copy_and_correct():
    nb, block, heads, dim = 4, 256, 2, 128
    pool = torch.arange(nb * block * heads * dim, dtype=torch.float32).reshape(
        nb, block, heads, dim
    )
    view = _pool_sub_view(pool, 2)
    assert view.shape == (8, 128, heads, dim)
    # 零拷贝:同一 storage
    assert view.data_ptr() == pool.data_ptr()
    # 值映射:引擎页 p 的 token t == 子页 p*2 + t//128 的 token t%128
    p, t, h, d = 3, 200, 1, 55
    assert view[p * 2 + t // 128, t % 128, h, d] == pool[p, t, h, d]
    # sub=1 恒等
    assert _pool_sub_view(pool, 1) is pool


def test_expanded_metadata_matches_engine_page_semantics():
    """展开后的 (indptr, indices, last_page_len) + 子页池视图 ≡ 引擎页大小下的直接计算。"""
    nb, block, heads, dim = 6, 512, 2, 16
    sub = block // KERNEL_PAGE_SIZE
    assert sub == 4
    pool = torch.arange(nb * block * heads * dim, dtype=torch.float32).reshape(
        nb, block, heads, dim
    )
    view = _pool_sub_view(pool, sub)

    # 两个请求:600 token(占 2 个引擎页,尾页 88),130 token(1 个引擎页,尾页 130)
    block_tables = torch.tensor([[0, 3], [5, 0]], dtype=torch.int32)
    seq_lens = torch.tensor([600, 130], dtype=torch.int32)

    bt_sub = _expand_page_table(block_tables, sub)
    meta = ragged_from_page_table_device(bt_sub, seq_lens, KERNEL_PAGE_SIZE)

    # 逐请求回放子页序列,收集 token 流(取 h=0,d=0 一个通道的值即可验证布局)
    for b in range(2):
        n = int(seq_lens[b])
        start, end = int(meta.kv_indptr[b]), int(meta.kv_indptr[b + 1])
        sub_ids = meta.kv_indices[start:end].tolist()
        tokens = []
        for sid in sub_ids:
            tokens.extend(view[sid, :, 0, 0].tolist())
        live = int(meta.kv_last_page_len[b])
        # 有效 token 流 = 前 seq_len 个(页从头部填充;尾页前 live 个有效)
        assert live == n - KERNEL_PAGE_SIZE * (len(sub_ids) - 1)
        tokens = tokens[:n]

        # 直接按引擎页语义计算同一条 token 流
        engine_pages = block_tables[b].tolist()
        direct = []
        for page in engine_pages:
            direct.extend(pool[page, :, 0, 0].tolist())
        assert len(tokens) == n, f"request {b}: 子页 token 流长度 {len(tokens)} != seq_len {n}"
        assert tokens == direct[:n], f"request {b}: 子页回放与引擎页语义不一致"


def test_pad_batch_on_expanded_metadata():
    block_tables = torch.tensor([[2, 4]], dtype=torch.int32)
    seq_lens = torch.tensor([300], dtype=torch.int32)
    meta = ragged_from_page_table_device(
        _expand_page_table(block_tables, 2), seq_lens, KERNEL_PAGE_SIZE
    )
    padded = pad_batch(meta, bucket=4, page_size=KERNEL_PAGE_SIZE)
    assert padded.batch_size == 4
    assert padded.kv_indptr.numel() == 5
    # 填充请求必须合法:1 个子页、1 个有效 token(否则 softmax 分母为 0)
    assert int(padded.kv_indptr[4]) - int(padded.kv_indptr[3]) == 1
    assert int(padded.kv_last_page_len[3]) == 1
