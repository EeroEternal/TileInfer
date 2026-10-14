"""CPU 单测:H18 prefill 步守卫(共享元数据跨层去重)+ plan_query_tiles_from_host。

prefill_step 的共享面构建(qo 长度、ragged 页表三元组、tile schedule、形状判定)只依赖
attn_metadata 与池形状,与层无关;守卫按 id(attn_metadata) 命中时零同步复用(E45)。
本文件把 forward_prefill(TileLang,需 NPU)与 PrefillShapeManager.usable(会触发后台
编译)打桩,在 CPU 上验证守卫语义与共享对象身份;内核数值正确性由
test_vllm_ascend_prefill_wiring.py(设备)覆盖。
"""

from __future__ import annotations

import pytest
import torch

import tileinfer.integrations.vllm_ascend as va
from tileinfer.plan import plan_query_tiles, plan_query_tiles_from_host


@pytest.fixture(autouse=True)
def _reset_step_guard():
    va._PREFILL_STEP_GUARD = 0
    va._PREFILL_STEP_SHARED.clear()
    va._PREFILL_STEP_OBJ = None
    va.TileInferDecodeAttention._stats.pop("prefill_refreshes", None)
    va.TileInferDecodeAttention._stats.pop("prefill_calls", None)
    yield
    va._PREFILL_STEP_GUARD = 0
    va._PREFILL_STEP_SHARED.clear()
    va._PREFILL_STEP_OBJ = None


class _FakeLaunch:
    """记录每次 forward_prefill 调用的实参;返回可解包的全零 out4。"""

    def __init__(self, heads: int, dim: int):
        self.calls: list = []
        self.heads, self.dim = heads, dim

    def __call__(self, q_pad, key_cache, value_cache, meta, schedule, qo_dev, **kw):
        self.calls.append(
            {
                "q_pad_shape": tuple(q_pad.shape),
                "meta": meta,
                "schedule": schedule,
                "qo_dev": qo_dev,
                "kernel_batch": kw.get("kernel_batch"),
                "group": kw.get("group"),
            }
        )
        b, h, q, d = q_pad.shape
        return torch.zeros(b, h, q, d, dtype=q_pad.dtype, device=q_pad.device)


@pytest.fixture()
def env(monkeypatch):
    monkeypatch.setattr(va.PrefillShapeManager, "usable", staticmethod(lambda spec: True))
    fake = _FakeLaunch(heads=8, dim=32)
    import tileinfer.kernels.attention.paged_decode_ascend950 as paged

    monkeypatch.setattr(paged, "forward_prefill", fake)
    return fake


def _make_step(kv_lens, qo_lens, device="cpu"):
    num_tokens = sum(qo_lens)
    heads, dim = 8, 32
    pool = torch.zeros(8, 128, heads // 4, dim, dtype=torch.bfloat16)
    table = torch.arange(len(kv_lens), dtype=torch.int32, device=device).unsqueeze(1)
    seq_lens = torch.tensor(kv_lens, dtype=torch.int32, device=device)
    qsl = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(qo_lens, dtype=torch.int64), 0)),
        dtype=torch.int32,
        device=device,
    )
    q = torch.randn(num_tokens, heads, dim, dtype=torch.bfloat16)
    out = torch.zeros_like(q)
    return dict(
        query=q,
        key_cache=pool,
        value_cache=pool.clone(),
        block_tables=table,
        seq_lens=seq_lens,
        query_start_loc=qsl,
        block_size=128,
        output=out,
        num_heads=heads,
        num_kv_heads=heads // 4,
        head_size=dim,
        scale=dim**-0.5,
        batch_cap=4,
    )


def test_plan_from_host_matches_tensor_entry():
    qo = [300, 0, 129, 1]
    pages = [3, 5, 2, 1]
    dev = torch.device("cpu")
    for lb in (False, True):
        a = plan_query_tiles(
            torch.tensor(qo, dtype=torch.int64), torch.tensor(pages, dtype=torch.int64),
            block_q=128, load_balance=lb,
        )
        b = plan_query_tiles_from_host(qo, pages, device=dev, block_q=128, load_balance=lb)
        for f in ("seq_ids", "q_offsets", "q_lens", "kv_page_starts", "kv_page_lens",
                  "split_ids", "split_counts"):
            assert torch.equal(getattr(a, f), getattr(b, f)), (lb, f)
        assert a.num_tiles == b.num_tiles


def test_step_guard_shares_payload_across_layers(env):
    step = object()
    kw = _make_step([200, 129, 37], [100, 129, 5])
    assert va.prefill_step(**kw, step_obj=step) is True
    assert va.prefill_step(**kw, step_obj=step) is True
    assert len(env.calls) == 2
    # 第二层零重建:meta/schedule/qo_dev 是同一批对象;形状判定只发生一次
    assert env.calls[0]["meta"] is env.calls[1]["meta"]
    assert env.calls[0]["schedule"] is env.calls[1]["schedule"]
    assert env.calls[0]["qo_dev"] is env.calls[1]["qo_dev"]
    assert va.TileInferDecodeAttention._stats["prefill_refreshes"] == 1
    assert va.TileInferDecodeAttention._stats["prefill_calls"] == 2
    assert env.calls[0]["kernel_batch"] == 5  # bucket(4)+1 trash 行


def test_step_guard_rebuilds_for_new_metadata(env):
    kw = _make_step([200, 129], [100, 129])
    s1, s2 = object(), object()
    assert va.prefill_step(**kw, step_obj=s1) is True
    assert va.prefill_step(**kw, step_obj=s2) is True
    assert va.TileInferDecodeAttention._stats["prefill_refreshes"] == 2
    assert env.calls[0]["meta"] is not env.calls[1]["meta"]


def test_step_guard_none_keeps_per_call_build(env):
    kw = _make_step([200], [200])
    assert va.prefill_step(**kw) is True
    assert va.prefill_step(**kw) is True
    # 无步身份:保持旧行为(每调用重建,不计入守卫计数)
    assert "prefill_refreshes" not in va.TileInferDecodeAttention._stats
    assert len(env.calls) == 2
    assert env.calls[0]["meta"] is not env.calls[1]["meta"]


def test_step_guard_belt_and_braces_on_content_change(env):
    step = object()
    kw = _make_step([200], [200])
    assert va.prefill_step(**kw, step_obj=step) is True
    # 同一 id 但内容不符(id 复用防护):必须重建,不得复用陈旧共享面
    kw2 = _make_step([200, 129], [200, 100])
    assert va.prefill_step(**kw2, step_obj=step) is True
    assert va.TileInferDecodeAttention._stats["prefill_refreshes"] == 2


def test_unusable_shape_never_stores_guard(monkeypatch, env):
    step = object()
    kw = _make_step([200], [200])
    monkeypatch.setattr(va.PrefillShapeManager, "usable", staticmethod(lambda spec: False))
    assert va.prefill_step(**kw, step_obj=step) is False
    assert va._PREFILL_STEP_GUARD == 0  # 拒绝路径不得落守卫,后续步可重试
    monkeypatch.setattr(va.PrefillShapeManager, "usable", staticmethod(lambda spec: True))
    assert va.prefill_step(**kw, step_obj=step) is True
    assert va.TileInferDecodeAttention._stats["prefill_refreshes"] == 1
