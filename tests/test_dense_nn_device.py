"""H77 dense_nn 设备数值 oracle:三族真实形状 bf16,shim(NN matmul)vs 原 apply(NT)。

纯数值门槛(与 h77_nn_bench 的 allclose 同判据,引擎内集成面独立复核):
  任一形状 M∈{1,8} 下 |NN−NT|_atol ≤ 2e-2 且相对误差 ≤ 2e-2 → PASS。
运行(pytest -m npu 设备窗):
    .venv/bin/python -m pytest tests/test_dense_nn_device.py -q -m npu
"""

from __future__ import annotations

import pytest
import torch

try:
    from tileinfer.integrations import dense_nn as dn

    _HAVE_MOD = True
except Exception:  # pragma: no cover
    _HAVE_MOD = False

pytestmark = pytest.mark.skipif(not _HAVE_MOD, reason="dense_nn module unavailable")

SHAPES = [(12288, 2048, "qkvz"), (9216, 2048, "attnqkv"), (2048, 4096, "oproj")]


class _QM:
    def apply(self, layer, x, bias=None):
        return torch.nn.functional.linear(x, layer.weight.data, bias)


class _Layer:
    def __init__(self, n, k):
        self.prefix = "test"
        self.weight = torch.nn.Parameter(
            torch.randn(n, k, dtype=torch.bfloat16, device="npu") * 0.02)
        self.bias = None
        self.quant_method = _QM()


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    monkeypatch.setenv("TILEINFER_DENSE_NN", "1")
    monkeypatch.delenv("TILEINFER_DENSE_NN_MAXM", raising=False)
    monkeypatch.delenv("TILEINFER_DENSE_NN_DRYRUN", raising=False)
    monkeypatch.setattr(dn, "_INSTALLED", False)
    for k in dn._STATS:
        dn._STATS[k] = 0
    yield


@pytest.mark.parametrize("n,k,name", SHAPES)
@pytest.mark.parametrize("m", [1, 8])
def test_dense_nn_device_oracle(n, k, name, m):
    torch.manual_seed(0)
    layer = _Layer(n, k)
    assert dn._shim_layer(layer) is True, name
    x = torch.randn(m, k, dtype=torch.bfloat16, device="npu")
    ref = torch.nn.functional.linear(x, layer.weight.data).float()
    out = layer.quant_method.apply(layer, x).float()
    assert dn.stats()["dense_nn_forward"] >= 1
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)
    # 大 M 回落路径数值 = 原 apply(NT)本体
    x_big = torch.randn(64, k, dtype=torch.bfloat16, device="npu")
    out_big = layer.quant_method.apply(layer, x_big).float()
    ref_big = torch.nn.functional.linear(x_big, layer.weight.data).float()
    torch.testing.assert_close(out_big, ref_big, atol=0.0, rtol=0.0)
