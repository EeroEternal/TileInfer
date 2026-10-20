"""H74 契约:编译器调度页流水开关(`TILEINFER_SW_PIPELINE`)的静态与语义约束。

纯 CPU,不依赖 NPU(tilelang 编译在设备测试里做)。

背景:deepseek/DeepInfer 的 Ascend 算子(`tilelang_ops/asc/`,26 个文件)全部用
`T.Persistent`/`T.Pipelined(..., num_stages=N)` 让**编译器**编排流水;我们此前是
`for p in T.serial(num_pages)`。H74 把页循环换成 `T.Pipelined(num_stages=sw_pipeline)`,
`num_stages=0` 时 tilelang 不启用流水(等价串行)。

这里锁定四件事:
1. `sw_pipeline=0` 不进 cache key(默认路径不因多一个键而重复编译);
2. `sw_pipeline>0` **必须**进 cache key(否则与串行版共享同一份产物,静默错配);
3. 环境变量解析健壮(非法值退回 0,不抛);
4. 内核源码里页循环确实是 `T.Pipelined(..., num_stages=sw_pipeline)`——
   防止后续回改时把开关变成哑参数(本仓 H56 踩过"挂点从未生效")。
"""

from __future__ import annotations

import inspect
import os

import pytest

from tileinfer.kernels.attention import paged_decode_ascend950 as mod


def _base_spec():
    return dict(
        batch=1, kv_heads=2, group=8, dim=128, page_size=128,
        num_pages_cap=512, dtype="bfloat16", out_manual=True,
    )


# ------------------------------------------------------------------ 1. 环境变量


def test_env_default_is_zero(monkeypatch):
    monkeypatch.delenv("TILEINFER_SW_PIPELINE", raising=False)
    assert mod._sw_pipeline_stages() == 0


@pytest.mark.parametrize("raw,expected", [("0", 0), ("2", 2), ("4", 4), ("16", 16)])
def test_env_parses_positive(monkeypatch, raw, expected):
    monkeypatch.setenv("TILEINFER_SW_PIPELINE", raw)
    assert mod._sw_pipeline_stages() == expected


@pytest.mark.parametrize("raw", ["", "abc", "-3", "1.5"])
def test_env_invalid_falls_back_to_zero(monkeypatch, raw):
    """非法值必须静默退回 0,不能抛——否则 serve 启动即崩。"""
    monkeypatch.setenv("TILEINFER_SW_PIPELINE", raw)
    assert mod._sw_pipeline_stages() == 0


# ------------------------------------------------------------------ 2. cache key


def _key_via_build(spec):
    """走 `build_decode_kernel` 的规范化逻辑拿到 cache key(不实际编译)。

    `build_decode_kernel` 会先 pop/normalize 再算 key;直接调 `_cache_key` 会绕过这一步,
    测不到真实行为。这里用 monkeypatch 截获 `tilelang.compile`。
    """
    captured = {}

    class _FakeKernel:
        pass

    def fake_compile(*args, **kwargs):
        captured["called"] = True
        return _FakeKernel()

    import tilelang

    mod._KERNEL_CACHE.clear()
    orig = tilelang.compile
    tilelang.compile = fake_compile
    try:
        mod.build_decode_kernel(**spec)
        # 编译后缓存里只有一个条目,其 key 即规范化结果
        return next(iter(mod._KERNEL_CACHE))
    finally:
        tilelang.compile = orig
        mod._KERNEL_CACHE.clear()


def test_zero_stages_not_in_cache_key():
    """默认路径(0)不得改变 cache key,否则默认产物会多编译一份。"""
    k_none = _key_via_build(_base_spec())
    k_zero = _key_via_build(dict(_base_spec(), sw_pipeline=0))
    assert k_none == k_zero


@pytest.mark.parametrize("stages", [2, 3, 4])
def test_positive_stages_change_cache_key(stages):
    """非 0 必须改变 key:串行版与流水版是两个不同的编译产物。"""
    k0 = _key_via_build(_base_spec())
    k1 = _key_via_build(dict(_base_spec(), sw_pipeline=stages))
    assert k0 != k1


def test_different_depths_do_not_collide():
    k2 = _key_via_build(dict(_base_spec(), sw_pipeline=2))
    k3 = _key_via_build(dict(_base_spec(), sw_pipeline=3))
    assert k2 != k3


# ------------------------------------------------------------------ 3. 源码锁


def test_page_loop_uses_pipelined_with_switch():
    """页循环必须是 `T.Pipelined(num_pages, num_stages=sw_pipeline)`。

    这条锁防两件事:
    - 回改成 `T.serial` → 开关变哑参数(静默不生效);
    - 写成固定的 `num_stages=2` → 默认路径被强制开流水(E296 已证会回归)。
    """
    src = inspect.getsource(mod.paged_decode_ascend950)
    assert "T.Pipelined(num_pages, num_stages=sw_pipeline)" in src, (
        "页循环必须用 T.Pipelined 且阶段数来自 sw_pipeline 参数"
    )
    # 不得退化成固定常量
    assert "T.Pipelined(num_pages, num_stages=2)" not in src


def test_build_decode_kernel_forwards_switch():
    """`build_decode_kernel` 必须把 sw_pipeline 传下去(而不是丢弃)。"""
    src = inspect.getsource(mod.build_decode_kernel)
    assert 'spec.pop("sw_pipeline"' in src, "build_decode_kernel 必须消费 sw_pipeline"


def test_backend_passes_switch():
    """backend 必须从 env 取值并传给 build_decode_kernel。"""
    from tileinfer.attention.backends import tilelang_ascend950 as be

    src = inspect.getsource(be)
    assert "_sw_pipeline_stages()" in src, "backend 未接线 _sw_pipeline_stages"
    assert "sw_pipeline=_sw_pipeline_stages()" in src
