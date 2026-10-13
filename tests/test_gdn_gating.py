"""GDN gating 融合算子:数值正确性(CPU 可跑,fp64 oracle 对照)。

M-GDN(E20):替换 vllm-ascend A5 在役的 gating triton patch。本文件验证
``tileinfer.kernels.gdn.fused_gdn_gating`` 的语义:
1. 与 fp64 oracle(softplus/sigmoid 数学定义)对齐;
2. softplus 阈值分支(beta·x > threshold 线性化)两侧都覆盖;
3. dtype 契约:g fp32、beta 与 b 同 dtype、形状 (1, T, H);
4. 广播语义:A_log/dt_bias 为 (H,) 头共享参数。
"""

from __future__ import annotations

import pytest
import torch

from tileinfer.kernels.gdn import fused_gdn_gating, reference_gdn_gating


@pytest.mark.parametrize("t,h", [(1, 16), (4, 16), (128, 16), (333, 7)])
def test_gdn_gating_matches_fp64_oracle(t, h):
    g = torch.randn(t, h) * 0.5
    A_log = torch.randn(h, dtype=torch.float32) * 0.3
    dt_bias = torch.randn(h, dtype=torch.float32) * 0.1
    b = torch.randn(t, h)
    got_g, got_beta = fused_gdn_gating(A_log, g, b, dt_bias)
    ref_g, ref_beta = reference_gdn_gating(A_log, g.double(), b, dt_bias)
    torch.testing.assert_close(got_g, ref_g.to(torch.float32), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        got_beta.double(), ref_beta, rtol=1e-6, atol=1e-7
    )


def test_gdn_gating_softplus_threshold_branch():
    """大输入走线性分支(与 triton 的 where(beta*x<=threshold, log1p, x) 一致)。"""
    h = 4
    A_log = torch.zeros(h, dtype=torch.float32)
    dt_bias = torch.zeros(h, dtype=torch.float32)
    a = torch.tensor([[30.0, -30.0, 20.0, 21.0]], dtype=torch.bfloat16)
    b = torch.zeros(1, h, dtype=torch.bfloat16)
    got_g, got_beta = fused_gdn_gating(A_log, a, b, dt_bias)
    # beta=1, threshold=20: x>20 → softplus(x)=x;x<=20 → log1p(exp(x))
    expect = torch.tensor(
        [[30.0, torch.log1p(torch.exp(torch.tensor(-30.0))).item(), torch.log1p(torch.exp(torch.tensor(20.0))).item(), 21.0]],
    )
    torch.testing.assert_close(got_g[0], -expect, rtol=1e-5, atol=1e-5)
    assert torch.allclose(got_beta, torch.full((1, h), 0.5, dtype=torch.bfloat16))


def test_gdn_gating_dtype_and_shape_contract():
    t, h = 7, 16
    A_log = torch.randn(h, dtype=torch.float32)
    dt_bias = torch.randn(h, dtype=torch.float32)
    a = torch.randn(t, h, dtype=torch.bfloat16)
    b = torch.randn(t, h, dtype=torch.bfloat16)
    g, beta = fused_gdn_gating(A_log, a, b, dt_bias)
    assert g.dtype == torch.float32 and g.shape == (1, t, h)
    assert beta.dtype == torch.bfloat16 and beta.shape == (1, t, h)
    # fp32 输入同样成立(recurrent 路径可能传 fp32)
    g2, beta2 = fused_gdn_gating(A_log, a.float(), b.float(), dt_bias)
    assert g2.dtype == torch.float32 and beta2.dtype == torch.float32
    torch.testing.assert_close(g2, g, rtol=1e-6, atol=1e-6)
