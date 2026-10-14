"""TILEINFER_VLLM opt-in 门回归测试(P0 off 侧语义)。

背景:install() 曾用真值判断 ``not os.environ.get("TILEINFER_VLLM")``,使
``TILEINFER_VLLM=0``(FIA 对照侧)也被视为启用——权威口径 P0 基线的 off 侧
serve 日志出现 TileInfer plan READY 即该缺陷的直接证据。修复后只有显式
``TILEINFER_VLLM=1`` 才安装插件;本测试锁死拒绝侧契约(CPU 可跑,零设备
依赖、零全局副作用)。接受侧("1" 路径)由引擎级在役探针覆盖(6 plan READY)。

运行::

    .venv/bin/python -m pytest tests/test_vllm_ascend_optin_gate.py -v
"""

from __future__ import annotations

import pytest

try:
    from tileinfer.integrations import vllm_ascend as mod

    _HAVE_MOD = True
except Exception:  # pragma: no cover - CPU 机器无 vllm_ascend 依赖时跳过
    _HAVE_MOD = False


pytestmark = pytest.mark.skipif(not _HAVE_MOD, reason="vllm_ascend integration unavailable")


@pytest.mark.parametrize("value", [None, "", "0", "false", "off", "2"])
def test_optin_gate_rejects_everything_but_literal_one(monkeypatch, value):
    """未设置或任何非 "1" 值 → install() 返回 False,不注册 backend。"""
    if value is None:
        monkeypatch.delenv("TILEINFER_VLLM", raising=False)
    else:
        monkeypatch.setenv("TILEINFER_VLLM", value)
    assert mod.install() is False
