"""L-B 选择质量 dump hook 契约(纯 CPU,不编译内核,H81/H85 判据输入)。

dump 是离线判据的唯一数据通道,锁死三件事:
- 采集正确性:从 paged cache 按 block_tables 收集的 K/V 与稠密参照逐位一致(含 sub>1 展开);
- 步边界累积:逐层累积 → 步切换 flush → 一个步一个文件,E364 格式;
- 门控安全:env 未启用时零行为;采集内部异常静默返回 None(dump 永不影响服务)。
"""

from __future__ import annotations

import os

import pytest
import torch

from tileinfer.integrations import vllm_ascend as integ

HQ, HKV, D, KERNEL_PAGE = 16, 2, 256, 128


def _make_cache(n_pages: int, block: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    total = n_pages * block
    k = torch.randn(1, total, HKV, D, generator=g, dtype=torch.float32).to(torch.bfloat16)
    v = torch.randn(1, total, HKV, D, generator=g, dtype=torch.float32).to(torch.bfloat16)
    return k.view(n_pages, block, HKV, D), v.view(n_pages, block, HKV, D)


def _collect(query, key_cache, value_cache, block_tables, seq_len, block_size):
    seq_lens = torch.tensor([seq_len])
    return integ._lb_dump_collect(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_tables=block_tables,
        seq_lens=seq_lens,
        block_size=block_size,
    )


def test_collect_matches_dense_reference_sub1():
    """sub=1(block=128):收集的 K/V = 稠密参照的前 S 行,逐位一致。"""
    n_pages, block, seq_len = 8, 128, 1000
    key_cache, value_cache = _make_cache(n_pages, block)
    block_tables = torch.arange(n_pages).view(1, n_pages)
    query = torch.randn(1, HQ, D, dtype=torch.bfloat16)

    payload = _collect(query, key_cache, value_cache, block_tables, seq_len, block)
    assert payload is not None
    assert payload["q"].shape == (HQ, D)
    assert payload["k"].shape == (HKV, seq_len, D)
    assert payload["v"].shape == (HKV, seq_len, D)
    # 稠密参照:pool 拍平成序列
    assert torch.equal(payload["k"], key_cache.reshape(-1, HKV, D)[:seq_len].permute(1, 0, 2))
    assert torch.equal(payload["v"], value_cache.reshape(-1, HKV, D)[:seq_len].permute(1, 0, 2))


def test_collect_matches_dense_reference_sub2_and_permutation():
    """sub=2(block=256):页表乱序放置,收集须按页表展开且子页连续。"""
    block, n_pages, seq_len = 256, 6, 500  # sub=2,需要 ceil(500/128)=4 个 128 子页=2 个引擎页
    key_cache, value_cache = _make_cache(n_pages, block, seed=3)
    # 页表:引擎页 3,1(乱序),覆盖子页 6,7 / 2,3
    block_tables = torch.tensor([[3, 1]])
    query = torch.randn(1, HQ, D, dtype=torch.bfloat16)

    payload = _collect(query, key_cache, value_cache, block_tables, seq_len, block)
    assert payload is not None
    sub_pages = [3 * 2 + 0, 3 * 2 + 1, 1 * 2 + 0, 1 * 2 + 1]
    flat = key_cache.reshape(-1, HKV, D)  # (n_pages*block, HKV, D) 逐 token
    ref_k = torch.cat([flat[p * 128 : (p + 1) * 128] for p in sub_pages])[:seq_len]
    assert torch.equal(payload["k"], ref_k.permute(1, 0, 2))


def test_collect_rejects_bad_shapes():
    key_cache, value_cache = _make_cache(2, 128)
    block_tables = torch.tensor([[0, 1]])
    query = torch.randn(1, HQ, D, dtype=torch.bfloat16)
    # 表宽不足
    assert _collect(query, key_cache, value_cache, block_tables, 300, 128) is None
    # 非法长度
    assert _collect(query, key_cache, value_cache, block_tables, 0, 128) is None
    # 3D cache(NZ 变体形态)不 dump
    assert _collect(query, key_cache.flatten(2, 3), value_cache.flatten(2, 3), block_tables, 100, 128) is None


def test_collect_swallows_exceptions(monkeypatch):
    """采集内部异常必须静默(dump 永不影响服务)。"""
    key_cache, value_cache = _make_cache(2, 128)
    block_tables = torch.tensor([[0, 1]])
    query = torch.randn(1, HQ, D, dtype=torch.bfloat16)
    monkeypatch.setattr(integ, "_pool_sub_view", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    assert _collect(query, key_cache, value_cache, block_tables, 100, 128) is None


def test_record_flush_step_boundary(tmp_path, monkeypatch):
    """逐层累积 → 步切换 flush:一个步一个文件,格式 {"layers": [...]}。"""
    out_dir = str(tmp_path)
    monkeypatch.setenv("TILEINFER_LB_DUMP", out_dir)
    monkeypatch.setenv("TILEINFER_LB_DUMP_STEPS", "2")
    integ._LB_DUMP_STATE.update({"acc_key": None, "acc": None, "saved": 0})

    def layer_payload(seed):
        g = torch.Generator().manual_seed(seed)
        return {
            "q": torch.randn(HQ, D, generator=g).bfloat16(),
            "k": torch.randn(HKV, 128, D, generator=g).bfloat16(),
            "v": torch.randn(HKV, 128, D, generator=g).bfloat16(),
        }

    step_a, step_b = object(), object()
    for li in range(3):  # 步 A:3 层
        integ._lb_dump_record(step_a, layer_payload(li))
    for li in range(2):  # 步 B 首层触发步 A 落盘
        integ._lb_dump_record(step_b, layer_payload(10 + li))
    integ._lb_dump_flush()  # 兜底 flush 步 B

    files = sorted(os.listdir(out_dir))
    assert files == ["lb_dump_ctx128_step0.pt", "lb_dump_ctx128_step1.pt"]
    d0 = torch.load(os.path.join(out_dir, files[0]), weights_only=True)
    assert set(d0["layers"][0].keys()) == {"q", "k", "v"}
    assert len(d0["layers"]) == 3
    d1 = torch.load(os.path.join(out_dir, files[1]), weights_only=True)
    assert len(d1["layers"]) == 2


def test_record_respects_max_steps(tmp_path, monkeypatch):
    out_dir = str(tmp_path)
    monkeypatch.setenv("TILEINFER_LB_DUMP", out_dir)
    monkeypatch.setenv("TILEINFER_LB_DUMP_STEPS", "1")
    integ._LB_DUMP_STATE.update({"acc_key": None, "acc": None, "saved": 0})

    p = {"q": torch.zeros(HQ, D).bfloat16(), "k": torch.zeros(HKV, 128, D).bfloat16(), "v": torch.zeros(HKV, 128, D).bfloat16()}
    integ._lb_dump_record(object(), p)
    integ._lb_dump_record(object(), p)  # 触发第 1 步落盘;第 2 步达上限不再累积
    integ._lb_dump_record(object(), p)
    integ._lb_dump_flush()

    files = os.listdir(out_dir)
    assert files == ["lb_dump_ctx128_step0.pt"]


def test_dump_disabled_by_default(monkeypatch):
    monkeypatch.delenv("TILEINFER_LB_DUMP", raising=False)
    assert integ._lb_dump_enabled() is False
    monkeypatch.setenv("TILEINFER_LB_DUMP", "/tmp/x")
    assert integ._lb_dump_enabled() is True
