"""Fused sampling 集成层:把 :mod:`tileinfer.kernels.sampling.fused_sample` 接到 vLLM。

## 与 H36(greedy 快路径)的关系

两者是**互补**的,不是替代:

| | H36(`greedy.py`) | 本模块(`fused_sample`) |
|---|---|---|
| 覆盖场景 | 全批 greedy(temperature=0) | 有 top-p/top-k 的真实采样 |
| 做法 | **跳过** fp32 cast,直接 argmax | 把 temp+softmax+阈值+采样**融成一个 kernel** |
| 收益来源 | 省一次全词表 cast 与处理器链 | 省掉全词表 `sort`(O(V log V)) + 多次往返 |

在 greedy 场景下 H36 更轻(它连 softmax 都不做),所以**本模块的守卫会显式跳过
全 greedy 批次**,把舞台留给 H36。只有当批次里存在 `top_p < 1` 或 `top_k < vocab`
的请求时才接管。

## 守卫(任何一条不满足即回退原生实现)

1. `ops_enabled("sampling")` —— `TILEINFER_OPS` 含 `sampling`;
2. logits 是 2D、float32、连续、last-dim stride 1;
3. `vocab_size` 是 64 的倍数且 ≥ `MIN_VOCAB_FOR_FUSED_SAMPLE`;
4. 批次里**至少有一个**请求启用 top-p/top-k(否则交给 H36/原生 argmax);
5. 无 logprobs 需求(`max_num_logprobs` 为 None 且 `logprob_token_ids` 空);
6. 无 penalties / bad words / allowed-token mask 等会改写 logits 的处理器
   —— 这些会改变概率分布,融合内核看不到它们;
7. 无 spec token(`predict_bonus_token=False`)。

## 回退语义

任何异常都静默回退原生 `forward`,并 bump 计数器。**绝不抛出**,避免采样失败
拖垮整个 serve。

## 开关

`TILEINFER_OPS` 含 `sampling` 才安装;`ops_enabled("sampling")` 每步读取,
翻转 env + 重启即回退。计数器:
- `fused_sample_calls` / `fused_sample_fallbacks` —— 内核接管/回退次数。
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import torch

from ..kernels.sampling.fused_sample import (
    MIN_VOCAB_FOR_FUSED_SAMPLE,
    fused_sample_plan,
    fused_sample_tl,
)

logger = logging.getLogger(__name__)

__all__ = [
    "COUNTERS",
    "install_fused_sample",
    "fused_sample_applicable",
    "reset_state",
]

COUNTERS: dict[str, int] = {
    "fused_sample_calls": 0,
    "fused_sample_fallbacks": 0,
    "fused_sample_skipped": 0,
}

_stats_sink: Callable[[str], None] | None = None
_INSTALLED = False
# (vocab, batch, do_sample) -> 已编译内核
_KERNEL_CACHE: dict[tuple, Any] = {}


def _inc(name: str) -> None:
    COUNTERS[name] = COUNTERS.get(name, 0) + 1
    if _stats_sink is not None:
        _stats_sink(name)


def _ops_enabled() -> bool:
    try:
        from .vllm_ascend import ops_enabled

        return ops_enabled("sampling")
    except Exception:  # noqa: BLE001 - 开关读取失败按关闭处理
        return False


def _has_real_sampling(md: Any) -> bool:
    """批次里是否真的存在 top-p/top-k 请求。

    vLLM 的采样元数据字段在不同版本略有差异,这里做宽松读取:
    拿不到就当作「有采样」,让后续守卫去判断(宁可不接管,也不误判)。
    """
    top_p = getattr(md, "top_p", None)
    top_k = getattr(md, "top_k", None)
    try:
        if top_p is not None and bool((top_p < 1.0).any().item()):
            return True
        if top_k is not None:
            # vLLM 语义:`top_k <= 0` 表示**不启用**,不是"取前 0 个"。
            # (写成 `>= 0` 会把 top_k=0 误判为启用,导致全 greedy 批次被本内核
            # 抢走,而那种场景 H36 更轻。)
            vocab = getattr(md, "vocab_size", 1 << 30)
            if bool(((top_k > 0) & (top_k < vocab)).any().item()):
                return True
        return False
    except Exception:  # noqa: BLE001
        return True


def fused_sample_applicable(logits: Any, md: Any, predict_bonus_token: bool) -> bool:
    """严格守卫:全部是 host 侧标量/形状检查(零 D2H)。"""
    if predict_bonus_token:
        return False
    if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
        return False
    if logits.dtype != torch.float32:
        return False
    if not logits.is_contiguous():
        return False
    vocab = int(logits.shape[-1])
    if vocab < MIN_VOCAB_FOR_FUSED_SAMPLE or vocab % 64 != 0:
        return False
    # logprobs 需求 → 内核不产出,必须回退
    if getattr(md, "max_num_logprobs", 0) is not None:
        return False
    if getattr(md, "logprob_token_ids", None):
        return False
    # 会改写 logits 的处理器 → 概率分布已变,内核看不到
    if not getattr(md, "no_penalties", False):
        return False
    if getattr(md, "bad_words_token_ids", None):
        return False
    if getattr(md, "allowed_token_ids_mask", None) is not None:
        return False
    procs = getattr(md, "logprobsprocs", None) or getattr(md, "logitsprocs", None)
    if procs is not None and len(getattr(procs, "non_argmax_invariant", [None])):
        return False
    # 全 greedy 批次交给 H36(它更轻:连 softmax 都不做)
    if not _has_real_sampling(md):
        return False
    return True


def _get_kernel(vocab: int, batch: int, do_sample: bool):
    key = (vocab, batch, do_sample)
    kern = _KERNEL_CACHE.get(key)
    if kern is None:
        import tilelang

        kern = tilelang.compile(fused_sample_tl(vocab, batch, do_sample), out_idx=None)
        _KERNEL_CACHE[key] = kern
    return kern


def _extract_params(md: Any, batch: int, device) -> tuple[torch.Tensor, ...]:
    """从采样元数据取出内核需要的四个张量(detach + float32/int32/int64)。"""
    def _col(name: str, default: float, dtype):
        v = getattr(md, name, None)
        if v is None:
            return torch.full((batch,), default, dtype=dtype, device=device)
        if not isinstance(v, torch.Tensor):
            return torch.full((batch,), default, dtype=dtype, device=device)
        v = v.to(device=device, dtype=dtype).view(-1)
        if v.numel() == 1 and batch > 1:
            v = v.expand(batch).contiguous()
        return v[:batch].contiguous()

    temps = _col("temperature", 1.0, torch.float32)
    # temperature=0 表示 greedy;内核里 1/temp 会爆,钳到 1e-6(等价于极锐 softmax)
    temps = torch.clamp(temps, min=1e-6)
    top_p = _col("top_p", 1.0, torch.float32)
    top_k = _col("top_k", 0, torch.int32)
    seeds = getattr(md, "generators", None)
    seed = torch.arange(batch, dtype=torch.int64, device=device) + 12345
    if isinstance(seeds, (list, tuple)) and seeds:
        try:
            vals = [int(getattr(g, "initial_seed", lambda: 12345)()) for g in seeds]
            seed = torch.tensor(vals[:batch], dtype=torch.int64, device=device)
        except Exception:  # noqa: BLE001
            pass
    return temps, top_p, top_k, seed


def make_fused_forward(orig_forward: Callable) -> Callable:
    """生成覆写 ``AscendSampler.forward`` 的包装(签名与基类一致)。"""

    def forward(
        self,
        logits: torch.Tensor,
        sampling_metadata: Any,
        predict_bonus_token: bool = False,
        logprobs_mode_override: Any = None,
    ):
        if _ops_enabled():
            try:
                if fused_sample_applicable(logits, sampling_metadata, predict_bonus_token):
                    batch = int(logits.shape[0])
                    vocab = int(logits.shape[-1])
                    do_sample = True
                    temps, top_p, top_k, seed = _extract_params(
                        sampling_metadata, batch, logits.device
                    )
                    out = torch.empty(batch, dtype=torch.int32, device=logits.device)
                    kern = _get_kernel(vocab, batch, do_sample)
                    kern(logits.contiguous(), temps, top_p, top_k, seed, out)
                    from vllm.v1.outputs import SamplerOutput

                    _inc("fused_sample_calls")
                    return SamplerOutput(
                        sampled_token_ids=out.unsqueeze(-1), logprobs_tensors=None
                    )
            except Exception:  # noqa: BLE001 - 接管失败必须静默回退,绝不破坏服务
                _inc("fused_sample_fallbacks")
        return orig_forward(
            self, logits, sampling_metadata, predict_bonus_token, logprobs_mode_override
        )

    return forward


def install_fused_sample() -> bool:
    """在采样器类上安装 fused-sample forward(须在采样器构造前调用)。"""
    global _INSTALLED
    if _INSTALLED:
        return True
    try:
        from vllm_ascend.sample.sampler import AscendSampler as _cls
    except Exception:
        logger.warning("fused_sample not installed: vllm_ascend sampler unavailable")
        return False

    _cls.forward = make_fused_forward(_cls.forward)
    _INSTALLED = True
    logger.info(
        "TileInfer fused_sample installed (governed by TILEINFER_OPS=sampling)"
    )
    return True


def reset_state() -> None:
    """测试辅助:清缓存与计数器(引擎内不使用)。"""
    global _INSTALLED, _stats_sink
    _KERNEL_CACHE.clear()
    for k in list(COUNTERS):
        COUNTERS[k] = 0
    _INSTALLED = False
    _stats_sink = None


def set_stats_sink(fn: Callable[[str], None] | None) -> None:
    """把计数器接到后端的 _stats(供 metrics dumper 读取)。"""
    global _stats_sink
    _stats_sink = fn
