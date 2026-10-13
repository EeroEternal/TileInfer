"""设备测试:GDN gating 接管(NPU)。

1. tileinfer 实现与 vllm-ascend A5 在役 triton patch 逐位对齐(decode/prefill 形状);
2. ``install_gdn_gating`` 的 monkey-patch 分派:装上后 DeviceOperator 路径走 TileInfer,
   TILEINFER_OPS 去掉 gdn 时回落原实现,计数器递增;
3. ACLGraph 捕获+replay:门控为固定形状 out-of-place 算子,捕获安全,数据更新后 replay 正确。

CPU 机器自动跳过;运行前 source .agent_loop/run/env.sh(D26)。
"""

from __future__ import annotations

import pytest
import torch

from tileinfer.kernels.gdn import fused_gdn_gating
from tileinfer.utils import is_npu_available

try:  # pragma: no cover
    import vllm_ascend.ops.gdn  # noqa: F401  先拉起完整链(device_op 反向依赖 ops)
    import vllm_ascend.device.device_op as _device_op

    from vllm_ascend.ops.triton.fused_gdn_gating import fused_gdn_gating_patch

    _HAVE_VLLM_ASCEND = True
except Exception:  # pragma: no cover
    _HAVE_VLLM_ASCEND = False

pytestmark = [
    pytest.mark.npu,
    pytest.mark.skipif(not is_npu_available(), reason="no Ascend NPU available"),
    pytest.mark.skipif(not _HAVE_VLLM_ASCEND, reason="vllm-ascend not importable"),
]

H = 16


@pytest.fixture(autouse=True, scope="session")
def _init_triton_props():
    """独立测试进程没有 vLLM worker 的初始化链,需手动初始化 triton 设备属性。"""
    from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

    torch.npu.set_device(0)
    init_device_properties_triton()


def _make(t, device):
    g = torch.Generator(device="cpu").manual_seed(t)
    a = (torch.randn(t, H, generator=g) * 2).to(device=device, dtype=torch.bfloat16)
    b = torch.rand(t, H, generator=g).to(device=device, dtype=torch.bfloat16)
    A_log = torch.randn(H, generator=g).to(device=device, dtype=torch.float32) * 0.3
    dt_bias = torch.randn(H, generator=g).to(device=device, dtype=torch.float32) * 0.1
    return A_log, a, b, dt_bias


@pytest.mark.parametrize("t", [1, 32, 128, 1024])
def test_parity_with_vllm_ascend_triton_patch(t):
    device = torch.device("npu", 0)
    A_log, a, b, dt_bias = _make(t, device)
    ref_g, ref_beta = fused_gdn_gating_patch(A_log, a, b, dt_bias)
    got_g, got_beta = fused_gdn_gating(A_log, a, b, dt_bias)
    torch.testing.assert_close(got_g, ref_g, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(got_beta.float(), ref_beta.float(), rtol=1e-6, atol=1e-7)
    assert got_g.dtype == torch.float32 and got_beta.dtype == torch.bfloat16


def test_install_dispatch_and_ops_switch():
    from tileinfer.integrations.vllm_ascend import (
        TileInferDecodeAttention,
        install_gdn_gating,
        ops_enabled,
    )

    assert install_gdn_gating()
    device_op = _device_op
    DeviceOperator = device_op.get_device_adaptor()
    assert DeviceOperator is _device_op.A5DeviceAdaptor
    assert getattr(_device_op.A5DeviceAdaptor, "_tileinfer_gdn_installed", False)

    device = torch.device("npu", 0)
    A_log, a, b, dt_bias = _make(8, device)
    before = TileInferDecodeAttention._stats.get("gdn_gating_calls", 0)
    g, beta = DeviceOperator.fused_gdn_gating(A_log, a, b, dt_bias)
    assert TileInferDecodeAttention._stats["gdn_gating_calls"] == before + 1

    ref_g, ref_beta = fused_gdn_gating_patch(A_log, a, b, dt_bias)
    torch.testing.assert_close(g, ref_g, rtol=1e-5, atol=1e-5)

    # TILEINFER_OPS 关掉 gdn → 走原 triton patch,计数不再增长(B6 秒级回退语义)
    if ops_enabled("gdn"):  # 环境缺省全开才做开关翻转
        import os

        old = os.environ.get("TILEINFER_OPS")
        os.environ["TILEINFER_OPS"] = "decode,prefill,moe"
        try:
            calls = TileInferDecodeAttention._stats["gdn_gating_calls"]
            g2, beta2 = DeviceOperator.fused_gdn_gating(A_log, a, b, dt_bias)
            assert TileInferDecodeAttention._stats["gdn_gating_calls"] == calls
            ref2_g, ref2_beta = fused_gdn_gating_patch(A_log, a, b, dt_bias)
            torch.testing.assert_close(g2, ref2_g, rtol=1e-5, atol=1e-5)
        finally:
            if old is None:
                os.environ.pop("TILEINFER_OPS", None)
            else:
                os.environ["TILEINFER_OPS"] = old


def test_graph_capture_and_replay():
    """torch.npu.NPUGraph 捕获门控算子链,更新数据后 replay 输出随数据变化。"""
    device = torch.device("npu", 0)
    t = 128
    A_log, a, b, dt_bias = _make(t, device)
    g = torch.empty(1, t, H, device=device, dtype=torch.float32)
    beta = torch.empty(1, t, H, device=device, dtype=torch.bfloat16)

    def _run():
        gg, bb = fused_gdn_gating(A_log, a, b, dt_bias)
        g.copy_(gg)
        beta.copy_(bb)

    _run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        _run()
    torch.npu.synchronize()

    a2 = (torch.randn(t, H) * 3).to(device=device, dtype=torch.bfloat16)
    # 就地改被捕获的输入缓冲再 replay(引擎 replay 的库级等价物)
    a.copy_(a2)
    g.fill_(float("nan"))
    graph.replay()
    torch.npu.synchronize()
    ref_g, ref_beta = fused_gdn_gating(A_log, a, b, dt_bias)
    torch.testing.assert_close(g, ref_g, rtol=1e-5, atol=1e-5)
    assert not torch.isnan(g).any()
