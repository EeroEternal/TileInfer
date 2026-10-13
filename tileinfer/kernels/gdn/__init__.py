"""GDN(gated delta net)融合算子(Qwen3.5/3.6 线性注意力层)。

定位(BACKLOG M-GDN,E20 取证):替换 vllm-ascend A5 适配器在役的 gating triton 内核
(``fused_gdn_gating_patch``,E20 实测 prefill 形状逐调用 ~2.6ms 设备时间,占 prefill
设备时间 11%)。decode 递推核/conv1d 在 E20 实测每步设备时间合计 <0.2ms,不做替换。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["fused_gdn_gating", "reference_gdn_gating"]


def fused_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """gated delta net 的门控参数:g = -exp(A_log)·softplus(a + dt_bias),beta = sigmoid(b)。

    与 vllm-ascend ``fused_gdn_gating_patch``(triton)逐位同语义:
    softplus 在 beta·x > threshold 时线性化为 x(F.softplus 同一约定);
    beta 在 fp32 域做 sigmoid 后转回 ``b`` 的 dtype(triton 先 ``to(fp32)`` 再 cast)。
    返回 ``(g, beta_out)``:``g`` 为 fp32、形状 ``(1, T, H)``;``beta_out`` 与 ``b`` 同
    dtype、形状 ``(1, T, H)``。全部为 out-of-place 固定形状算子,ACLGraph 捕获安全。
    """
    g = -torch.exp(A_log.float()) * F.softplus(
        a.float() + dt_bias.float(), beta, threshold
    )
    beta_out = torch.sigmoid(b.float()).to(b.dtype)
    return g.unsqueeze(0), beta_out.unsqueeze(0)


def reference_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp64 oracle(测试参照):直接按数学定义计算,不经过 F.softplus 的快速路径。"""
    x = (a.double() + dt_bias.double()) * beta
    softplus = torch.where(x <= threshold, torch.log1p(torch.exp(x)) / beta, x / beta)
    g = -torch.exp(A_log.double()) * softplus
    beta_out = torch.sigmoid(b.double())
    return g.unsqueeze(0), beta_out.unsqueeze(0)
