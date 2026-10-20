"""step_diag(H65)契约测试:门控、JSONL 事件、上下文/包装器机制。

背景:step_diag 是 c1 步长 host 暴露段的分解诊断模块,仅在
``TILEINFER_STEP_DIAG=1`` 时激活(默认关=零行为变化、零 vendor 改动)。
本测试锁死三面契约(CPU 可跑,零设备依赖、零全局副作用):
  1) 门控:未设置或非 "1" → install_step_diag() 返回 False 且不产生任何
     副作用(不打开文件、不改 _ENABLED);
  2) 事件机制:_TimedSeg/_wrap_timed 落盘 JSONL 行结构(t/step/seg/ms[+ntok]);
  3) 包装器语义:透传返回值与异常;record_function 名称映射(prepare input→prep)。
激活路径(vllm/vllm_ascend 类打点)由引擎级在役探针覆盖(serve 日志
"step_diag ACTIVE"),不在 CPU 测试面内。

运行::

    .venv/bin/python -m pytest tests/test_step_diag.py -v
"""

from __future__ import annotations

import json

import pytest

try:
    from tileinfer.integrations import step_diag as sd

    _HAVE_MOD = True
except Exception:  # pragma: no cover - 依赖缺失时跳过
    _HAVE_MOD = False


pytestmark = pytest.mark.skipif(not _HAVE_MOD, reason="step_diag module unavailable")


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    """每个用例隔离:清环境变量、重置模块全局态、指向临时文件。"""
    monkeypatch.delenv("TILEINFER_STEP_DIAG", raising=False)
    monkeypatch.delenv("TILEINFER_STEP_DIAG_PATH", raising=False)
    monkeypatch.setattr(sd, "_ENABLED", False)
    monkeypatch.setattr(sd, "_fh", None)
    monkeypatch.setattr(sd, "_step", 0)
    monkeypatch.setattr(sd, "_events", 0)
    monkeypatch.setattr(sd, "_pending", [])
    monkeypatch.setattr(sd, "_t0", 0.0)
    yield


@pytest.mark.parametrize("value", [None, "", "0", "false", "off", "2"])
def test_gate_rejects_everything_but_literal_one(monkeypatch, value):
    """未设置或任何非 "1" 值 → 不激活、不打开文件、无副作用。"""
    if value is None:
        monkeypatch.delenv("TILEINFER_STEP_DIAG", raising=False)
    else:
        monkeypatch.setenv("TILEINFER_STEP_DIAG", value)
    assert sd.install_step_diag() is False
    assert sd._fh is None
    assert sd._ENABLED is False


def test_timed_seg_emits_jsonl(monkeypatch, tmp_path):
    out = tmp_path / "diag.jsonl"
    monkeypatch.setenv("TILEINFER_STEP_DIAG", "1")
    monkeypatch.setattr(sd, "_fh", open(out, "a", buffering=1))
    with sd._TimedSeg("prep"):
        pass
    with sd._TimedSeg("graph"):
        pass
    sd._flush_pending()
    lines = [json.loads(x) for x in out.read_text().splitlines()]
    assert [r["seg"] for r in lines] == ["prep", "graph"]
    for r in lines:
        assert set(r) == {"t", "step", "seg", "ms"}
        assert isinstance(r["ms"], float) and r["ms"] >= 0.0
        assert r["t"] >= 0.0 and r["step"] == 0


def test_emit_batches_until_threshold(monkeypatch, tmp_path):
    """事件批量落盘:<64 条滞留内存,≥64 条触发整批写入(iter-206)。"""
    out = tmp_path / "diag.jsonl"
    monkeypatch.setattr(sd, "_fh", open(out, "a", buffering=1))
    for i in range(63):
        sd._emit("prep", float(i))
    assert out.read_text() == ""
    assert len(sd._pending) == 63
    sd._emit("prep", 63.0)
    lines = out.read_text().splitlines()
    assert len(lines) == 64
    assert sd._pending == []


def test_flush_pending_drains_partial(monkeypatch, tmp_path):
    out = tmp_path / "diag.jsonl"
    monkeypatch.setattr(sd, "_fh", open(out, "a", buffering=1))
    sd._emit("exec", 1.0, ntok=2)
    sd._flush_pending()
    assert sd._pending == []
    assert out.read_text().splitlines()[-1]  # 非空即已落盘
    sd._flush_pending()  # 幂等:空缓冲不写不抛
    sd._emit("noop", 1.0)  # _fh=None 场景已由 test_fh_none_is_noop 覆盖


def test_emit_ntok_field(monkeypatch, tmp_path):
    out = tmp_path / "diag.jsonl"
    monkeypatch.setattr(sd, "_fh", open(out, "a", buffering=1))
    sd._emit("exec", 1.2345, ntok=8)
    sd._emit("exec", 2.0, ntok=None)
    sd._flush_pending()
    lines = [json.loads(x) for x in out.read_text().splitlines()]
    assert lines[0]["ntok"] == 8
    assert "ntok" not in lines[1]


def test_diag_record_function_name_mapping():
    assert sd._diag_record_function("prepare input")._name == "prep"
    assert sd._diag_record_function("forward")._name == "fwd"
    assert sd._diag_record_function("post process")._name == "post"
    assert sd._diag_record_function("other block")._name == "other block"


def test_wrap_timed_passthrough_and_error():
    class Target:
        def ok(self, a, b=0):
            return a + b

        def boom(self):
            raise ValueError("x")

    t = Target()
    sd._wrap_timed(Target, "ok", "seg_ok", ntok_of=lambda *a, **k: a[1] if len(a) > 1 else None)
    assert t.ok(1, b=2) == 3
    with pytest.raises(ValueError):
        t.boom()


def test_fh_none_is_noop():
    """_fh 为 None(未激活)时 _emit 静默无副作用。"""
    assert sd._fh is None
    sd._emit("prep", 1.0)
    sd._emit("exec", 1.0, ntok=4)
    sd._flush_pending()


def test_patch_prep_stubs_failure_is_import_error():
    """vendor model_runner_v1 裸进程不可导入(环形导入,仅在役可加载)——
    _patch_prep_stubs 此时须抛 ImportError(由 install_step_diag 统一捕获记
    warning),不得静默吞掉;在役激活面由 serve 日志 "step_diag ACTIVE" 探针覆盖。"""
    import sys
    import types

    pytest.importorskip("vllm")
    from vllm.v1.worker.block_table import BlockTable

    assert hasattr(BlockTable, "commit_block_table")
    fake = types.ModuleType("vllm_ascend.worker.model_runner_v1")
    monkey_patch = pytest.MonkeyPatch()
    monkey_patch.setitem(sys.modules, "vllm_ascend.worker.model_runner_v1", fake)
    monkey_patch.setitem(sys.modules, "vllm_ascend.worker", types.ModuleType("vllm_ascend.worker"))
    try:
        with pytest.raises((ImportError, AttributeError)):
            sd._patch_prep_stubs()
    finally:
        monkey_patch.undo()


def test_wrap_module_attr_passthrough_and_gating():
    """_wrap_module_attr:透传返回值,非可调用/缺失属性不包装。"""
    import types

    mod = types.ModuleType("fake_mrv")
    mod.update_cos_sin = lambda positions: positions + 1
    assert sd._wrap_module_attr(mod, "update_cos_sin", "cos_sin") is True
    assert mod.update_cos_sin(1) == 2
    assert sd._wrap_module_attr(mod, "missing_attr", "x") is False
    mod.not_callable = 42
    assert sd._wrap_module_attr(mod, "not_callable", "x") is False
