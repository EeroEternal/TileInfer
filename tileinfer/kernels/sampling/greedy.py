"""Sampler greedy 快路径(H36,采样域;依据 E105/E106 #19033 d-1、E97)。

全批 greedy(temperature=0——权威口径基准即此场景)时,vLLM 基类
``Sampler.forward`` 仍执行 bf16→fp32 **全词表 cast**(E91/E97:lm_head 采样路径
3.74% 设备份额的主要物料化)外加一串对本场景可证为 no-op 的处理器链。本模块
生成 ``AscendSampler.forward`` 覆写(0.23.0 的 AscendSampler 未覆写 forward,
原实现即基类 Sampler.forward),守卫全满足时直接原 dtype ``argmax``:

* 数值契约:bf16→fp32 逐元素精确可表且严格单调 → 与「先 cast 再 argmax」
  逐位一致;并列极值场景两次 argmax 看到同一值集、同一 tie-break 规则 →
  同索引(测试含人工并列极值用例);
* 跳过分支的可证 no-op 论证(与基类 forward 源码逐条对应):
  - allowed_token_ids_mask 为 None → masked_fill_ 不执行;
  - bad_words_token_ids 为空 → apply_bad_words 不调用;
  - logitsprocs.non_argmax_invariant 为空(min_tokens/logit_bias 均属该桶,
    基类 docstring 步骤 5)→ 贪心敏感处理器不执行;
  - no_penalties → apply_all_penalties 原样返回;
  - thinking_budget holder 为 None 或无追踪 → update_state/apply_to_logits 不执行;
  - max_num_logprobs None 且 logprob_token_ids 空 → 无 logprobs 物料需求;
  - predict_bonus_token False → 无 spec token 合并语义;
  - all_greedy → sample() 直接 greedy_sample 返回,temperature/min_p/top-k
    分支不可达。

任何守卫不满足或异常均回退原 forward(参数原样透传,含
``logprobs_mode_override``)。``TILEINFER_OPS=greedy`` 每步读取翻转;计数器
``greedy_fast_calls``/``greedy_fast_fallbacks`` 供在役探针断言。
"""

from __future__ import annotations

from typing import Any, Callable

import torch

#: 探针计数器(安装时同步进后端类 _stats,metrics dumper 读取)
COUNTERS: dict[str, int] = {
    "greedy_fast_calls": 0,
    "greedy_fast_fallbacks": 0,
}

_stats_sink: Callable[[str], None] | None = None


def _inc(name: str) -> None:
    COUNTERS[name] = COUNTERS.get(name, 0) + 1
    if _stats_sink is not None:
        _stats_sink(name)


def _ops_enabled() -> bool:
    try:
        from ...integrations.vllm_ascend import ops_enabled

        return ops_enabled("greedy")
    except Exception:  # noqa: BLE001 - 开关读取失败按关闭处理
        return False


def greedy_fast_applicable(logits: Any, md: Any, predict_bonus_token: bool) -> bool:
    """严格守卫:仅 host 侧标量/集合检查(零 D2H),任一不满足即 False。"""
    if predict_bonus_token:
        return False
    if not getattr(md, "all_greedy", False):
        return False
    if getattr(md, "max_num_logprobs", 0) is not None:
        return False
    if getattr(md, "logprob_token_ids", None):
        return False
    if not getattr(md, "no_penalties", False):
        return False
    if getattr(md, "bad_words_token_ids", None):
        return False
    if getattr(md, "allowed_token_ids_mask", None) is not None:
        return False
    procs = getattr(md, "logitsprocs", None)
    if procs is None or len(getattr(procs, "non_argmax_invariant", [None])):
        # non_argmax_invariant 桶含 min_tokens/logit_bias(贪心敏感),必须为空
        return False
    holder = getattr(md, "thinking_budget_state_holder", None)
    if holder is not None and holder.has_tracked_requests():
        return False
    if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
        return False
    return True


def greedy_fast_forward(logits: torch.Tensor, md: Any, predict_bonus_token: bool):
    """守卫 + 原 dtype argmax 快路径;不适用返回 None(调用方回退原 forward)。"""
    if not greedy_fast_applicable(logits, md, predict_bonus_token):
        return None
    from vllm.v1.outputs import SamplerOutput

    sampled = logits.argmax(dim=-1).view(-1).to(torch.int32)
    return SamplerOutput(sampled_token_ids=sampled.unsqueeze(-1), logprobs_tensors=None)


def make_greedy_forward(orig_forward: Callable) -> Callable:
    """生成覆写 ``AscendSampler.forward`` 的包装(签名与基类 forward 一致)。"""

    def forward(
        self,
        logits: torch.Tensor,
        sampling_metadata: Any,
        predict_bonus_token: bool = False,
        logprobs_mode_override: Any = None,
    ):
        if _ops_enabled():
            try:
                out = greedy_fast_forward(logits, sampling_metadata, predict_bonus_token)
                if out is not None:
                    _inc("greedy_fast_calls")
                    return out
            except Exception:  # noqa: BLE001 - 接管失败必须静默回退,绝不破坏服务
                _inc("greedy_fast_fallbacks")
        return orig_forward(
            self, logits, sampling_metadata, predict_bonus_token, logprobs_mode_override
        )

    return forward


def reset_counters() -> None:
    """计数器清零(测试辅助)。"""
    for name in list(COUNTERS):
        COUNTERS[name] = 0


def reset_state() -> None:
    """恢复模块态(测试辅助;引擎内不使用——翻转 TILEINFER_OPS 即回退)。"""
    global _stats_sink
    reset_counters()
    _stats_sink = None
