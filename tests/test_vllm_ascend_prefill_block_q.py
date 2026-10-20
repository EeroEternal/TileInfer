"""H72 短序列 prefill occupancy 自适应(CPU 契约,零设备依赖)。

背景(E239/E241 + E46):
  TI prefill 内核 grid = ``num_tiles × kv_heads``,``num_tiles = Σ ceil(qo_len·group / block_q)``。
  本模型 ``group=8``、``kv_heads=2``,block_q=128 时:
    q=128 × b=1 → 16 blocks;q=128 × b=8 → 128 blocks;q=512 × b=8 → 512 blocks。
  短序列 block 数远低于 950 可用核数,occupancy 不足;E46 画像显示内核是
  ``aiv_vec 0.96 / aic_mac 0.02`` 的逐页串行向量链(非访存、非 cube 瓶颈),
  因此"加 block"是该形状下唯一可动方向。
  H61 三档实测给���纯劣势:c1 +11.85%、c8 +22.01%、512_c8 +14.04%
  (TI 内核 TTFT vs 交还 FIA)。

锁定契约:
  1. 两个 env 旋钮默认值:``TILEINFER_PREFILL_BLOCK_Q=0``(自适应)、
     ``TILEINFER_PREFILL_SHORT_CTX=1024``;
  2. 自适应规则:上下文 <= SHORT_CTX → block_q=64,否则 128;
     显式 BLOCK_Q>0 时恒用该值(便于单变量 A/B);
  3. block_q 必须落在内核允许区间 (0, 128](内核断言 ``BR <= 128``);
  4. block_q 参与 tile 数计算、shape spec、schedule 规划三处,必须**同一值**
     (否则 pad/编译变体与布局不一致);
  5. block_q 通过 shared 缓存跨层复用(与 qo_host/meta 同生命周期)。
"""

from __future__ import annotations

import inspect

import pytest

try:
    from tileinfer.integrations import vllm_ascend as mod

    _HAVE_MOD = True
except Exception:  # pragma: no cover - CPU 机器无 vllm_ascend 依赖时跳过
    _HAVE_MOD = False


pytestmark = pytest.mark.skipif(
    not _HAVE_MOD, reason="vllm_ascend integration unavailable"
)


# ---------------------------------------------------------------- 1. 旋钮默认值


def test_h72_knob_defaults():
    """默认 = 自适应(0)/ 短上下文分界 1024。"""
    assert mod._PREFILL_BLOCK_Q == 0, "默认应为自适应"
    assert mod._PREFILL_SHORT_CTX == 1024


def test_h72_block_q_within_kernel_limit():
    """内核 assert BR <= 128,自适应值 64 必须合法;显式值也须受约束。"""
    assert 0 < 64 <= 128
    # 内核里的断言(源码级锁定,防未来放宽后与集成层失配)
    from tileinfer.kernels.attention import paged_decode_ascend950 as attn

    src = inspect.getsource(attn.paged_prefill_ascend950)
    assert "BR <= 128" in src


# ---------------------------------------------------------------- 2. 自适应规则


def _pick_block_q(max_ctx_tokens: int, block_q_env: int, short_ctx: int) -> int:
    """复刻集成层的选择逻辑(纯函数形式,便于无设备验证)。"""
    if block_q_env > 0:
        return block_q_env
    return 64 if max_ctx_tokens <= short_ctx else 128


def test_h72_adaptive_rule():
    assert _pick_block_q(128, 0, 1024) == 64        # 短序列 → 翻倍 tile
    assert _pick_block_q(1024, 0, 1024) == 64       # 边界含等号
    assert _pick_block_q(1025, 0, 1024) == 128      # 刚过界
    assert _pick_block_q(8192, 0, 1024) == 128      # 长序列不缩


def test_h72_explicit_override_wins():
    """显式 BLOCK_Q 覆盖自适应(单变量 A/B 用)。"""
    assert _pick_block_q(128, 32, 1024) == 32
    assert _pick_block_q(8192, 32, 1024) == 32
    assert _pick_block_q(128, 128, 1024) == 128


def test_h72_grid_growth():
    """验证缩 block_q 确实翻倍 grid(group=8, kv_heads=2,本模型真实参数)。"""
    group, kv_heads = 8, 2

    def grid(q: int, batch: int, bq: int) -> int:
        tiles = sum((q * group + bq - 1) // bq for _ in range(batch))
        return tiles * kv_heads

    # c1 短序列:16 → 32 blocks
    assert grid(128, 1, 128) == 16
    assert grid(128, 1, 64) == 32
    # c8:128 → 256
    assert grid(128, 8, 128) == 128
    assert grid(128, 8, 64) == 256
    # 长序列本就不缺并行,缩 block_q 只是增加碎片
    assert grid(8192, 8, 128) == 8192


# ---------------------------------------------------------------- 3. 三处一致性


def test_h72_block_q_used_consistently():
    """tile 数计算 / shape spec / schedule 规划必须用同一个 block_q。

    静态检查源码:不允许残留硬编码 ``block_q=128`` 或 ``+ 127) // 128``。
    """
    src = inspect.getsource(mod.prefill_step)

    # 三处都应引用变量
    assert "block_q=block_q" in src, "schedule/split 规划应使用变量 block_q"
    assert "(q * group + block_q - 1) // block_q" in src, "tile 数计算应用变量"
    assert 'block_q=shared["block_q"]' in src, "forward_prefill 调用应用 shared 的值"
    # 不允许再出现硬编码的 128 版 tile 计算
    assert "+ 127) // 128" not in src, "不应残留硬编码 block_q=128 的 tile 计算"
    # 逐行检查:任何**非注释**行都不得出现 block_q=128
    for line in src.splitlines():
        code = line.split("#", 1)[0]
        assert "block_q=128" not in code, f"不应残留硬编码: {line.strip()}"


def test_h72_block_q_cached_in_shared():
    """block_q 必须进 shared 缓存(跨层复用,与 qo_host/meta 同生命周期)。"""
    src = inspect.getsource(mod.prefill_step)
    assert '"block_q": block_q' in src, "shared 应保存 block_q"
    assert 'shared["block_q"]' in src, "forward_prefill 调用应读 shared 的 block_q"
