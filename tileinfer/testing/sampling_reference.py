"""采样 top-k 阈值选择的参考/对照实现(BACKLOG 采样条目,仅测试用)。

对照目标:vllm-ascend A5 原生路径 ``_apply_top_k_top_p_pytorch`` 的 top-k-only
分支——对整张词表 ``probs.sort(descending=True)``,取第 k 大概率作阈值,
``probs < cutoff`` 丢弃。softmax 严格单调,故在 logits 上取第 k 大 logit 作阈值
与在 probs 上取第 k 大 prob 等价;两者都保留并列项(``< cutoff`` 丢弃,``==``保留)。

本文件不依赖 deep_select / vllm_ascend,供设备数值一致性测试对照
(``tileinfer.kernels.sampling.topk.apply_top_k_top_p``,以 bf16 ULP 量级 maxdiff
为通过口径,对齐 npu_flash 侧三层判据)。
"""

from __future__ import annotations

from typing import Optional

import torch

__all__ = ["apply_top_k_top_p_reference"]


def apply_top_k_top_p_reference(
    logits: torch.Tensor,
    p: Optional[torch.Tensor],
    k: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """原生语义对照:第 k 大 logit 作阈值,``logits < cutoff`` 原位置 -inf。

    ``p is not None``(top-p 混合)或 ``k`` 非标量/越界时返回 ``None``(与接管实现
    同一窗口判据,供门控对照),否则返回原位掩码后的 ``logits``。
    """
    if p is not None or k is None or not isinstance(k, torch.Tensor):
        return None
    if logits.dim() != 2 or logits.numel() == 0 or k.numel() == 0:
        return None
    k0 = int(k.to(torch.long).reshape(-1)[0].item())
    vocab = int(logits.shape[-1])
    if any(v != k0 for v in k.to(torch.long).reshape(-1).tolist()):
        return None  # 异构 k 交还原生(与接管实现一致)
    if not 0 < k0 < vocab:
        return None
    # 第 k 大 logit(1-based):sorted=True 后取第 k-1 列
    topk = torch.topk(logits.to(torch.float32), k0, dim=-1, sorted=True, largest=True).values
    cutoff = topk[:, -1:].to(logits.dtype)
    return logits.masked_fill_(logits < cutoff, float("-inf"))
