"""H79 dense_nn 设备数值 oracle:三族真实形状 bf16,weight-swap 后引擎同构 apply
(F.linear on 换形权重)vs 原始 ND 权重 F.linear。

纯数值门槛(与 h79_stride_bench 的 allclose 同判据,引擎内集成面独立复核):
  任一形状 M∈{1,8} 下 |swap−orig|_atol ≤ 2e-2 且相对误差 ≤ 2e-2 → PASS;
  并断言物理形态:stride=(1,N)、.t() 连续(NN 存储)。
运行(pytest -m npu 设备窗):
    .venv/bin/python -m pytest tests/test_dense_nn_device.py -q -m npu
"""

from __future__ import annotations

import pytest
import torch

from tileinfer.utils import is_npu_available

try:
    from tileinfer.integrations import dense_nn as dn

    _HAVE_MOD = True
except Exception:  # pragma: no cover
    _HAVE_MOD = False

# E315(iter-231):补 npu 标记——文件内张量在 device="npu" 上创建,属设备测试;
# 缺标记导致 gate 链 `-m npu` 全量滤选(rc=5),且 CPU 套件会误收集并触碰设备。
pytestmark = [
    pytest.mark.npu,
    pytest.mark.skipif(not is_npu_available(), reason="no Ascend NPU available"),
    pytest.mark.skipif(not _HAVE_MOD, reason="dense_nn module unavailable"),
]

SHAPES = [(12288, 2048, "qkvz"), (9216, 2048, "attnqkv"), (2048, 4096, "oproj")]


class _QM:
    def apply(self, layer, x, bias=None):
        # 与引擎 AscendUnquantizedLinearMethod.apply→unquantized_gemm 同构
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
    monkeypatch.delenv("TILEINFER_DENSE_NN_DRYRUN", raising=False)
    monkeypatch.setattr(dn, "_INSTALLED", False)
    dn._REGISTERED.clear()
    for k in dn._STATS:
        dn._STATS[k] = 0
    yield


@pytest.mark.parametrize("n,k,name", SHAPES)
@pytest.mark.parametrize("m", [1, 8])
def test_dense_nn_device_oracle(n, k, name, m):
    torch.manual_seed(0)
    layer = _Layer(n, k)
    w_orig = layer.weight.data.clone()
    assert dn._swap_layer(layer) is True, name
    # 物理形态断言:逻辑形状不变、stride=(1,N)、B 视图 NN 连续
    assert tuple(layer.weight.data.shape) == (n, k), name
    assert tuple(layer.weight.data.stride()) == (1, n), name
    assert layer.weight.data.t().is_contiguous(), name
    assert dn.stats()["dense_nn_stride_ok"] == 1, name
    x = torch.randn(m, k, dtype=torch.bfloat16, device="npu")
    ref = torch.nn.functional.linear(x, w_orig).float()
    out = layer.quant_method.apply(layer, x).float()
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)
