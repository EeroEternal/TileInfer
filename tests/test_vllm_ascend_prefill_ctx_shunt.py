"""D57-①(W1 收尾):长上下文 prefill 步分流回退 FIA(CPU 契约,零设备依赖)。

背景(E41/E46/E47/E63/D57):TI prefill 内核单位功效率随 KV 上下文线性劣化
(aiv_vec_ratio=0.96、同功线性扩展),C8 饱和下 8K/16K 场景吞吐仅 0.213x/0.121x,
split 路线(ktp=64)亦无增益(0.226x/0.130x)。D57-① 把 KV 上下文超阈的 prefill 步
交还原生 FIA(decode 接管不受影响),env ``TILEINFER_PREFILL_MAX_CTX``(token;
H59 起**默认 4096** 即启用,显式 0 关闭)。

锁定契约:
  1. 决策纯函数 ``_prefill_ctx_shunt``:阈值关闭(<=0)恒 False;空页表恒 False;
     严格大于才分流(等于阈值不算超阈);页数按 KERNEL_PAGE_SIZE=128 换算;
  2. 模块状态:``_PREFILL_STATS`` 计数器在位(在役探针 prefill_long_skips 视图);
     ``_PREFILL_SKIP_GUARD`` 初始空(分流步其余层的短路键);
  3. env 旋钮默认 4096(H59);显式 0 = 关闭,恢复旧行为;
  4. metrics 合并语义:dump 视图含 prefill_long_skips 键(空进程静默守卫不受影响,
     该键不参与静默判定)。
"""

from __future__ import annotations

import pytest

try:
    from tileinfer.integrations import vllm_ascend as mod

    _HAVE_MOD = True
except Exception:  # pragma: no cover - CPU 机器无 vllm_ascend 依赖时跳过
    _HAVE_MOD = False


pytestmark = pytest.mark.skipif(not _HAVE_MOD, reason="vllm_ascend integration unavailable")


# ---------------------------------------------------------------- 1. 决策纯函数


def test_shunt_disabled_by_default_and_zero_threshold():
    # H59:默认值由 0 改为 4096(见 vllm_ascend.py 的 H59 注释与 E238-2 证据)。
    # 阈值 <=0 仍表示"关闭",显式 0 可恢复旧行为。
    assert isinstance(mod._PREFILL_MAX_CTX, int)
    assert mod._PREFILL_MAX_CTX == 4096, "H59:默认应启用 shunt(4096 token)"
    assert mod._prefill_ctx_shunt([999999], 0) is False
    assert mod._prefill_ctx_shunt([999999], -1) is False


def test_h59_default_shunts_long_context_only():
    """H59 契约:默认阈值下,≤4K 走 TI 内核、>4K 交还 FIA。

    依据 E238-2:未开 shunt 时 8K/16K TTFT 慢 1.53x/1.71x,而 ITL 已领先
    9.2%/13.6% —— decode 接管是正收益,唯一负项是长 prefill。
    """
    th = mod._PREFILL_MAX_CTX
    assert th == 4096
    # 短/中上下文:仍由 TI 内核接管(不 shunt)
    assert mod._prefill_ctx_shunt([1], th) is False
    assert mod._prefill_ctx_shunt([32], th) is False    # 32*128 == 4096
    # 长上下文:交还 FIA
    assert mod._prefill_ctx_shunt([33], th) is True     # 33*128 == 4224
    assert mod._prefill_ctx_shunt([64], th) is True     # 8192
    assert mod._prefill_ctx_shunt([128], th) is True    # 16384


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


# ------------------------------------------------- 4. H60:解包尾部搬运的缓冲契约


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
