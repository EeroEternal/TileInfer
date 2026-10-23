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
  - logitsprocs 贪心敏感面为空:0.23.0 持久实例语义下 MinTokens/LogitBias
    处理器常驻 non_argmax_invariant 桶(E409),空追踪状态(min_toks/biases
    为空 dict)时 apply() 短路返回原 logits=可证 no-op;有追踪或未知处理器
    即拦截;
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

import logging
from typing import Any, Callable

import torch

try:  # 0.23.0 内建处理器类型(E409 守卫语义依赖);缺失则退回保守拦截
    from vllm.v1.sample.logits_processor.builtin import (
        LogitBiasLogitsProcessor as _LBProc,
        MinTokensLogitsProcessor as _MTProc,
    )
except Exception:  # noqa: BLE001 - 非 vLLM 环境(纯 CPU 契约)退回保守拦截
    _LBProc = None
    _MTProc = None

#: 探针计数器(安装时同步进后端类 _stats,metrics dumper 读取)
COUNTERS: dict[str, int] = {
    "greedy_fast_calls": 0,
    "greedy_fast_fallbacks": 0,
    "greedy_fast_guard_skips": 0,
}

_stats_sink: Callable[[str], None] | None = None

_logger = logging.getLogger("tileinfer.sampling.greedy")


def _first_guard_fail(logits: Any, md: Any, predict_bonus_token: bool) -> str | None:
    """返回首个不满足的守卫名(诊断可观测面);全满足返回 None。

    与 ``greedy_fast_applicable`` 逐条同序同语义(单一事实源,后者委托本函数)。
    """
    if predict_bonus_token:
        return "predict_bonus_token"
    if not getattr(md, "all_greedy", False):
        return "not_all_greedy"
    if getattr(md, "max_num_logprobs", 0) is not None:
        return "logprobs_requested"
    if getattr(md, "logprob_token_ids", None):
        return "logprob_token_ids"
    if not getattr(md, "no_penalties", False):
        return "penalties_present"
    if getattr(md, "bad_words_token_ids", None):
        return "bad_words"
    if getattr(md, "allowed_token_ids_mask", None) is not None:
        return "allowed_token_ids_mask"
    procs = getattr(md, "logitsprocs", None)
    if procs is None:
        return "non_argmax_invariant_procs"
    for _proc in getattr(procs, "non_argmax_invariant", ()) or ():
        # 0.23.0 持久实例语义(E409):MinTokens/LogitBias 处理器无条件构建且
        # 常驻 non_argmax_invariant 桶——桶非空≠贪心敏感状态在轨。两者空状态
        # 时 apply() 逐字返回原 logits(builtin.py:apply 的 `if not self.min_toks`
        # /`if self.biases` 短路)=可证 no-op,放行;有追踪请求或未知处理器
        # (无法证明 argmax 不变性)一律保守拦截。
        if _MTProc is not None and isinstance(_proc, _MTProc):
            if getattr(_proc, "min_toks", None):
                return "min_tokens_tracked"
        elif _LBProc is not None and isinstance(_proc, _LBProc):
            if getattr(_proc, "biases", None):
                return "logit_bias_tracked"
        else:
            return "non_argmax_invariant_procs"
    holder = getattr(md, "thinking_budget_state_holder", None)
    if holder is not None and holder.has_tracked_requests():
        return "thinking_budget_tracked"
    if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
        return "logits_shape"
    return None


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
    return _first_guard_fail(logits, md, predict_bonus_token) is None


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
                # 守卫不满足:按因计数+一次性告警(诊断可观测面——引擎 A/B 探针
                # 需要区分「包装器从未被调用」与「守卫拦截」;此前该路径完全静默,
                # E402:h36 on 臂零计数无法归因即此盲区)。
                reason = _first_guard_fail(logits, sampling_metadata, predict_bonus_token)
                _inc("greedy_fast_guard_skips")
                if reason:
                    _logger.warning_once(
                        "greedy fast path guard skip: %s", reason
                    )
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
