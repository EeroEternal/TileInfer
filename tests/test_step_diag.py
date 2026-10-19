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
    sd._fh.flush()
    lines = [json.loads(x) for x in out.read_text().splitlines()]
    assert [r["seg"] for r in lines] == ["prep", "graph"]
    for r in lines:
        assert set(r) == {"t", "step", "seg", "ms"}
        assert isinstance(r["ms"], float) and r["ms"] >= 0.0
        assert r["t"] >= 0.0 and r["step"] == 0


def test_emit_ntok_field(monkeypatch, tmp_path):
    out = tmp_path / "diag.jsonl"
    monkeypatch.setattr(sd, "_fh", open(out, "a", buffering=1))
    sd._emit("exec", 1.2345, ntok=8)
    sd._emit("exec", 2.0, ntok=None)
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
