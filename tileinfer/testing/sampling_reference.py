"""采样 top-k 阈值选择的 oracle(测试参照)。

对照 ``tileinfer.kernels.sampling.topk.apply_top_k_top_p``(DeepSelect 路径)与原生
``vllm_ascend.sample.sampler.apply_top_k_top_p``(对整张词表 ``probs.sort``)。

等价性:``softmax`` 严格单调 ⇒ ``prob_i < 第k大 prob`` ⟺ ``logit_i < 第k大 logit``。
本 oracle 直接在 logits 上用 ``torch.topk`` 求第 k 大作阈值,与 DeepSelect 路径同语义
(两者都在 logits 域取值、都保留并列项),差异应仅在 bf16 舍入量级。
"""

from __future__ import annotations

import torch

__all__ = ["mask_top_k_by_logits", "mask_top_k_by_probs"]


def mask_top_k_by_logits(
    logits: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """logits 域 oracle:取每行第 k 大 logit 作阈值,``< 阈值`` 者置 ``-inf``(保留并列)。

    在 fp32 域计算阈值(避免 bf16 topk 的舍入影响),返回与 ``logits`` 同 dtype 的新张量
    (不原地修改入参,便于与 in-place 内核输出逐元素对比)。
    """
    cutoff = torch.topk(logits.float(), int(k), dim=-1).values.min(dim=-1, keepdim=True).values
    return logits.masked_fill(logits < cutoff.to(logits.dtype), float("-inf"))


def mask_top_k_by_probs(
    logits: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """probs 域 oracle(对齐原生 ``probs.sort`` 语义):softmax 后取第 k 大 prob 作阈值。

    与 :func:`mask_top_k_by_logits` 应一致(单调性);用于交叉核对两种取值域是否等价。
    """
    probs = torch.softmax(logits.float(), dim=-1)
    cutoff = torch.topk(probs, int(k), dim=-1).values.min(dim=-1, keepdim=True).values
    return logits.masked_fill(probs < cutoff, float("-inf"))
