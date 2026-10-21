"""D57-①(W1 收尾):长上下文 prefill 步分流回退 FIA(CPU 契约,零设备依赖)。

背景(E41/E46/E47/E63/D57):TI prefill 内核单位功效率随 KV 上下文线性劣化
(aiv_vec_ratio=0.96、同功线性扩展),C8 饱和下 8K/16K 场景吞吐仅 0.213x/0.121x,
split 路线(ktp=64)亦无增益(0.226x/0.130x)。D57-① 把 KV 上下文超阈的 prefill 步
交还原生 FIA(decode 接管不受影响),env ``TILEINFER_PREFILL_MAX_CTX``(token;
H59 起默认启用(4096);H61 终审起**默认 1**(全量交还 FIA),显式 0 关闭)。

锁定契约:
  1. 决策纯函数 ``_prefill_ctx_shunt``:阈值关闭(<=0)恒 False;空页表恒 False;
     严格大于才分流(等于阈值不算超阈);页数按 KERNEL_PAGE_SIZE=128 换算;
  2. 模块状态:``_PREFILL_STATS`` 计数器在位(在役探针 prefill_long_skips 视图);
     ``_PREFILL_SKIP_GUARD`` 初始空(分流步其余层的短路键);
  3. env 旋钮默认 4096(H59);显式 0 = 关闭,恢复旧行为;
  4. metrics 合并语义:dump 视图含 prefill_long_skips 键(空进程静默守卫不受影响,
     该键不参与静默判定);
  5. H86:全分流极端口径(``0 < MAX_CTX < 128``)下分流判决在 D2H 之前短路——
     零 ``.cpu()`` 同步、计数器/守卫与 "ctx shunt (early)" 分支等价;H59 档(4096)
     不短路,判据仍读设备侧 seq_lens。
"""

from __future__ import annotations

import pytest
import torch

try:
    from tileinfer.integrations import vllm_ascend as mod

    _HAVE_MOD = True
except Exception:  # pragma: no cover - CPU 机器无 vllm_ascend 依赖时跳过
    _HAVE_MOD = False


pytestmark = pytest.mark.skipif(not _HAVE_MOD, reason="vllm_ascend integration unavailable")


# ---------------------------------------------------------------- 1. 决策纯函数


def test_shunt_disabled_by_default_and_zero_threshold():
    # H59:默认值由 0 改为 4096(E238-2);H61 终审:再由 4096 改为 1(E244,
    # 同会话三档 A/B 后按 H61.3 预登记判据取 1)。阈值 <=0 仍表示"关闭",
    # 显式 0 可恢复旧行为(全走 TI 内核)。
    assert isinstance(mod._PREFILL_MAX_CTX, int)
    assert mod._PREFILL_MAX_CTX == 1, "H61:默认应全量交还 FIA(阈值 1)"
    assert mod._prefill_ctx_shunt([999999], 0) is False
    assert mod._prefill_ctx_shunt([999999], -1) is False


def test_h59_default_shunts_long_context_only():
    """H61 默认契约:默认阈值 1 下,>1 页(>128 token)的 prefill 交还 FIA。

    依据 E244:三档 A/B 中 c1 档吞吐全场景名义最高、TPOT 恒等、短场景 TTFT
    −16~−21%。1 页(≤128 token)的 prefill 仍由 TI 内核接管
    (预展开上界 ceil(seq/128) ≤ 1 不超阈,与在役实测 prefill_long_skips 一致)。
    """
    th = mod._PREFILL_MAX_CTX
    assert th == 1
    # 任意真实页(128 token 粒度)都超阈:全量交还 FIA(128 > 1)
    assert mod._prefill_ctx_shunt([1], th) is True     # 128 token > 1
    assert mod._prefill_ctx_shunt([2], th) is True     # 256 token
    assert mod._prefill_ctx_shunt([32], th) is True    # 4096
    assert mod._prefill_ctx_shunt([33], th) is True    # 4224
    assert mod._prefill_ctx_shunt([64], th) is True    # 8192
    assert mod._prefill_ctx_shunt([128], th) is True   # 16384


def test_shunt_empty_pages_never_shunts():
    assert mod._prefill_ctx_shunt([], 4096) is False


def test_shunt_page_to_token_conversion():
    # 1 页 = KERNEL_PAGE_SIZE token;换算后才比较(阈值本身是 token 数)
    ps = mod.KERNEL_PAGE_SIZE
    assert ps == 128
    assert mod._prefill_ctx_shunt([1], ps) is False        # 1 页=128 == 阈值:不分流
    assert mod._prefill_ctx_shunt([2], ps) is True         # 2 页=256 > 128
    assert mod._prefill_ctx_shunt([31], 4096) is False     # 31*128=3968 <= 4096
    assert mod._prefill_ctx_shunt([32], 4096) is False     # 32*128=4096 == 阈值
    assert mod._prefill_ctx_shunt([33], 4096) is True      # 33*128=4224 > 4096


def test_shunt_max_over_requests():
    # 分流看的是批内最大上下文:一个长请求即整步分流
    assert mod._prefill_ctx_shunt([1, 1, 31], 4096) is False
    assert mod._prefill_ctx_shunt([1, 1, 33], 4096) is True


# ---------------------------------------------------------------- 2. 模块状态与探针视图


def test_probe_stats_and_guard_in_place():
    # 在役探针依赖的计数键与分流守卫初始态
    assert "prefill_long_skips" in mod._PREFILL_STATS
    assert mod._PREFILL_SKIP_GUARD == ()
    # 计数器可累加(dumper 每秒读取,不做清零语义——进程生命周期累计值)
    before = mod._PREFILL_STATS["prefill_long_skips"]
    mod._PREFILL_STATS["prefill_long_skips"] = before + 1
    assert mod._PREFILL_STATS["prefill_long_skips"] == before + 1
    mod._PREFILL_STATS["prefill_long_skips"] = before


def test_metrics_dump_merges_prefill_stats():
    # dumper 的合并语义:stats.update(_PREFILL_STATS) 后视图含该键;
    # (_stats 是类初始计数器集合;kernel_markers 由 dumper 从 cls._marker 另行读出)
    stats = dict(getattr(mod.TileInferDecodeAttention, "_stats", {}))
    stats.update(mod._PREFILL_STATS)
    assert "prefill_long_skips" in stats
    assert "fallbacks" in stats


# ---------------------------------------------------------------- 3. env 旋钮契约


def test_env_knob_name_documented():
    # 旋钮名与 docs/性能口径一致(防手滑改名破坏部署脚本)
    import inspect

    src = inspect.getsource(mod)
    assert "TILEINFER_PREFILL_MAX_CTX" in src
    assert "TILEINFER_PREFILL_KV_TILE_PAGES" in src  # 既有旋钮不被误改


# ------------------------------------------------- 5. H86:全分流口径的 D2H 前短路


def _make_step(kv_lens, qo_lens):
    """最小 CPU 步参数(与 test_vllm_ascend_prefill_step_guard 的同款骨架)。"""
    num_tokens = sum(qo_lens)
    heads, dim = 8, 32
    pool = torch.zeros(8, 128, heads // 4, dim, dtype=torch.bfloat16)
    table = torch.arange(len(kv_lens), dtype=torch.int32).unsqueeze(1)
    seq_lens = torch.tensor(kv_lens, dtype=torch.int32)
    qsl = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(qo_lens, dtype=torch.int64), 0)),
        dtype=torch.int32,
    )
    q = torch.randn(num_tokens, heads, dim, dtype=torch.bfloat16)
    return dict(
        query=q,
        key_cache=pool,
        value_cache=pool.clone(),
        block_tables=table,
        seq_lens=seq_lens,
        query_start_loc=qsl,
        block_size=128,
        output=torch.zeros_like(q),
        num_heads=heads,
        num_kv_heads=heads // 4,
        head_size=dim,
        scale=dim**-0.5,
        batch_cap=4,
    )


@pytest.fixture()
def _reset_shunt_state():
    before_stats = dict(mod._PREFILL_STATS)
    before_guard = mod._PREFILL_SKIP_GUARD
    yield
    mod._PREFILL_STATS.clear()
    mod._PREFILL_STATS.update(before_stats)
    mod._PREFILL_SKIP_GUARD = before_guard


def test_h86_all_shunt_refuses_without_any_d2h(_reset_shunt_state, monkeypatch):
    """H86 主契约:MAX_CTX=1 下判决短路——``.cpu()`` 一次都不许发生。"""
    monkeypatch.setattr(mod, "_PREFILL_MAX_CTX", 1)

    def _no_cpu(self, *a, **k):
        raise AssertionError("H86: 全分流口径下 prefill 判决不得触发 D2H(.cpu())")

    monkeypatch.setattr(torch.Tensor, "cpu", _no_cpu)
    before = mod._PREFILL_STATS["prefill_long_skips"]
    step = object()
    kw = _make_step([200], [200])
    assert mod.prefill_step(**kw, step_obj=step) is False
    assert mod._PREFILL_STATS["prefill_long_skips"] == before + 1
    # 与 "ctx shunt (early)" 分支同款守卫键:同一步其余层零成本短路
    assert mod._PREFILL_SKIP_GUARD == (id(step), 1, kw["query"].shape[0])


def test_h86_second_layer_hits_skip_guard_still_zero_d2h(_reset_shunt_state, monkeypatch):
    """同一步第二层:守卫短路,计数器不重复累加,依旧零 D2H。"""
    monkeypatch.setattr(mod, "_PREFILL_MAX_CTX", 1)

    def _no_cpu(self, *a, **k):
        raise AssertionError("H86: 分流步其余层走守卫短路,同样不得触发 D2H")

    monkeypatch.setattr(torch.Tensor, "cpu", _no_cpu)
    step = object()
    kw = _make_step([200], [200])
    assert mod.prefill_step(**kw, step_obj=step) is False
    first = mod._PREFILL_STATS["prefill_long_skips"]
    assert mod.prefill_step(**kw, step_obj=step) is False
    assert mod._PREFILL_STATS["prefill_long_skips"] == first  # 第二层不计数


def test_h86_short_circuit_not_armed_at_h59_threshold(_reset_shunt_state, monkeypatch):
    """H59 档(4096 ≥ 1 页)不得短路:判据仍需读设备侧 seq_lens。"""
    monkeypatch.setattr(mod, "_PREFILL_MAX_CTX", 4096)
    seen: list = []
    orig_cpu = torch.Tensor.cpu

    def _rec(self, *a, **k):
        seen.append(1)
        return orig_cpu(self, *a, **k)

    monkeypatch.setattr(torch.Tensor, "cpu", _rec)
    monkeypatch.setattr(mod.PrefillShapeManager, "usable", staticmethod(lambda spec: False))
    kw = _make_step([200], [200])  # 2 页 = 256 token ≤ 4096:本不该分流
    assert mod.prefill_step(**kw, step_obj=object()) is False  # usable=False 拒绝
    assert seen, "H59 档不允许短路:分流判决需要真实的 seq_lens"


def test_h86_short_circuit_not_armed_when_disabled(_reset_shunt_state, monkeypatch):
    """MAX_CTX=0(全 TI 接管,shunt 关闭)更不允许短路。"""
    monkeypatch.setattr(mod, "_PREFILL_MAX_CTX", 0)
    seen: list = []
    orig_cpu = torch.Tensor.cpu

    def _rec(self, *a, **k):
        seen.append(1)
        return orig_cpu(self, *a, **k)

    monkeypatch.setattr(torch.Tensor, "cpu", _rec)
    monkeypatch.setattr(mod.PrefillShapeManager, "usable", staticmethod(lambda spec: False))
    kw = _make_step([200], [200])
    assert mod.prefill_step(**kw, step_obj=object()) is False
    assert seen


# ------------------------------------------------- 6. H60:解包尾部搬运的缓冲契约


def test_h60_unpack_tail_contract():
    """H60 契约:``forward_prefill`` 的解包尾部在"有 out / 无 out"两条路径上
    搬运次数不同,但**结果必须逐元素相同**,且无 out 时返回的是连续张量。

    这里只做静态检查(CPU 无 tilelang 编译):确认有 ``out`` 时不再先
    ``contiguous()`` 再 ``copy_``(那会付两次全量搬运)。
    """
    import inspect

    from tileinfer.kernels.attention import paged_decode_ascend950 as attn

    src = inspect.getsource(attn.forward_prefill)
    # 有 out 的快速路径:copy_ 直接吃 strided view 的 reshape,不先物化
    assert "out.copy_(" in src
    # 尾部必须仍有 contiguous 兜底(无 out 的返回路径),否则调用方拿到非连续张量
    assert ".contiguous()" in src
    # 关键:不能出现 "先 contiguous 到临时变量、再 copy_ 到 out" 的旧形态
    tail = src.split("out is not None")[-1]
    assert "result.contiguous()" not in tail.split("return out")[0], (
        "H60: 有 out 时不应先物化一份临时张量再拷贝"
    )
