"""采样 top-k 阈值选择:用 DeepSelect ``topk`` 替换原生对整张词表的 ``probs.sort``。

定位(A5 采样路径):``vllm_ascend.sample.sampler.AscendTopKTopPSampler.apply_top_k_top_p``
在 **top-k-only**(``p is None``)时仍对整张词表(V=151936)做 ``probs.sort``(O(V log V))
求第 k 大阈值。本模块在**采样门控包络**(词表 >= ``SAMPLING_MIN_VOCAB`` 且 batch >=
``SAMPLING_MIN_BATCH``)内,改用 DeepSelect ``topk``(``sorted=False``、int32)在 logits 上
直接求第 k 大(``values.min``)作阈值,语义等价且省去整词表排序。

等价性:``softmax`` 严格单调 ⇒ ``prob_i < 第k大 prob`` ⟺ ``logit_i < 第k大 logit``。
原生在 probs 上取第 k 大后 ``probs < cutoff`` 丢弃(保留并列项);本实现取第 k 大 logit
后 ``logits < cutoff``,同样保留并列项,语义一致(当 logits 为 bf16 可表示值、无 penalties/
temperature 引入的额外 fp32 尾数时完全一致)。

回退契约:``apply_top_k_top_p`` 在**包络外**或**硬约束不满足**时返回 ``None``,由集成层
(``integrations/vllm_ascend.py``)交还原生实现——保证小 batch / top-p 场景零额外开销。
"""

from __future__ import annotations

import os

import torch

__all__ = [
    "apply_top_k_top_p",
    "sampling_window_ok",
    "resolve_k_scalar_cached",
    "reset_k_scalar_cache",
    "SAMPLING_MIN_VOCAB",
    "SAMPLING_MIN_BATCH",
    "SAMPLING_INDICES_DTYPE",
    "SAMPLING_STRIDE_ALIGN_BYTES",
]


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


#: 词表门限:实测 vocab >= 16384 起 DeepSelect 任意 batch 均为正(V=16384 时 b=64/256/4096
#: → 0.39/0.25/0.23x);V<16384 即便放大 batch 也可能为负(V=2048 b=4096 仍 1.42x neg)。
#: 保守取 16384。可用 ``TILEINFER_SAMPLING_MIN_VOCAB`` 覆盖。
SAMPLING_MIN_VOCAB = _env_int("TILEINFER_SAMPLING_MIN_VOCAB", 16384)

#: batch 门限:采样窗口在每次调用上多一层 host 侧窗口判定与 ``masked_fill`` 固定开销
#: (~90~100us,随 batch 近乎恒定),b<8 无法摊薄(b=1 持平、b=2/4 略负 1.14x/1.09x);
#: b>=8 起稳定转正并随 batch 单调改善(0.54→0.11x)。可用 ``TILEINFER_SAMPLING_MIN_BATCH`` 覆盖。
SAMPLING_MIN_BATCH = _env_int("TILEINFER_SAMPLING_MIN_BATCH", 8)

#: DeepSelect Ascend 后端仅支持 int32 索引,且要求 stride(0) 为 32B 整数倍。
SAMPLING_INDICES_DTYPE = torch.int32
SAMPLING_STRIDE_ALIGN_BYTES = 32


# -- k 标量缓存(图模式关键:稳态零 D2H) --------------------------------------
# 键 = ``k`` 张量**对象身份**(强引用,防内存复用致 id 假命中)+ vocab。稳态解码下
# vLLM 复用同一 ``sampling_metadata.top_k`` 切片对象逐 step 传入,故稳态**零 D2H**;
# 批次组成变化时 ``_make_sampling_metadata`` 用切片重建**新对象**(值也仅在该重建时更新),
# 身份随之变化 → 自动失效重解析。正确性依据:**同一对象 ⟺ 值未变**。
_K_CACHE_K: object = None
_K_CACHE_VOCAB: int = -1
_K_CACHE_RESULT: tuple = (False, 0)


def reset_k_scalar_cache() -> None:
    """清空 k 标量缓存(挂/卸载接管时调用,释放强引用)。"""
    global _K_CACHE_K, _K_CACHE_VOCAB, _K_CACHE_RESULT
    _K_CACHE_K = None
    _K_CACHE_VOCAB = -1
    _K_CACHE_RESULT = (False, 0)


def resolve_k_scalar_cached(k: torch.Tensor, vocab: int) -> tuple:
    """解析标量 k(要求各行同值、且 ``0 < k0 < vocab``)并跨步缓存,返回 ``(ok, k0)``。

    仅在缓存未命中时执行一次 D2H(``k.to(long).reshape(-1).cpu().tolist()``,~15us);
    命中(``k`` 为同一对象且 vocab 一致)时**零 D2H**。这是图模式(NPU-bound)下消除
    「每步 D2H 打断异步流水」的关键。
    """
    global _K_CACHE_K, _K_CACHE_VOCAB, _K_CACHE_RESULT
    if _K_CACHE_K is k and _K_CACHE_VOCAB == vocab:
        return _K_CACHE_RESULT
    vals = k.to(torch.long).reshape(-1).cpu().tolist()
    k0 = vals[0]
    ok = 0 < k0 < vocab
    if ok:
        for v in vals:
            if v != k0:  # DeepSelect 仅支持标量 k;异构批次交还原生
                ok = False
                break
    _K_CACHE_K = k
    _K_CACHE_VOCAB = vocab
    _K_CACHE_RESULT = (ok, k0 if ok else 0)
    return _K_CACHE_RESULT


def _deepselect_topk_ok(input: torch.Tensor, k: int) -> bool:
    """是否满足 DeepSelect ``topk`` 的 Ascend 硬约束(对齐其 ``interface.py``)。"""
    if input.device.type != "npu":
        return False
    if input.dtype != torch.bfloat16:  # Ascend 仅支持 bf16
        return False
    if input.dim() != 2:
        return False
    if input.stride(-1) != 1:
        return False
    if (input.stride(0) * input.element_size()) % SAMPLING_STRIDE_ALIGN_BYTES != 0:
        return False
    if int(input.shape[-1]) < int(k):
        return False
    return True


def _deepselect_topk(input: torch.Tensor, k: int) -> tuple:
    import deep_select  # 懒 import:无 deep_select 的机器仍可 ``import tileinfer``

    return deep_select.topk(
        input,
        k,
        sorted=False,
        indices_type=SAMPLING_INDICES_DTYPE,
        return_value=True,
    )


def sampling_window_ok(
    logits: torch.Tensor,
    k: object,
    p: object,
) -> tuple:
    """判定是否落进「DeepSelect 可替代原生全词表 sort」的干净窗口,返回 ``(ok, k0)``。

    窗口条件:仅 top-k(``p is None``)、二维 NPU 浮点 logits、落进采样门控包络
    (``vocab >= SAMPLING_MIN_VOCAB`` **且** ``batch >= SAMPLING_MIN_BATCH``),且每行
    k 相同、``0 < k < V``。任一不满足即不做替换(交还原生)。

    **检查顺序很重要**:所有不需要 device→host 同步的条件(含 batch/词表门控)都在取 ``k``
    之前完成,确保**回退路径零额外开销**——否则单请求等回退场景会先付出一次 D2H 同步
    (本机实测 ~15~25us)再交还原生,反被拖慢。
    """
    if p is not None or k is None:
        return False, 0
    if logits.device.type != "npu" or logits.dim() != 2:
        return False, 0
    if not logits.is_floating_point() or logits.numel() == 0:
        return False, 0
    if not isinstance(k, torch.Tensor) or k.numel() == 0:
        return False, 0
    batch = int(logits.shape[0])
    vocab = int(logits.shape[-1])
    # 前置**无 D2H** 门控(可在此短路回退,避免为回退付出 device 同步开销):
    if batch < SAMPLING_MIN_BATCH or vocab < SAMPLING_MIN_VOCAB:
        return False, 0
    # 仅在通过门控后才取 k(跨步缓存,稳态零 D2H)。
    ok, k0 = resolve_k_scalar_cached(k, vocab)
    if not ok:
        return False, 0
    return True, k0


def apply_top_k_top_p(
    logits: torch.Tensor,
    k: object,
    p: object,
    top_k: object = None,
) -> "torch.Tensor | None":
    """纯内核路径:包络内用 DeepSelect 求第 k 大 *logit* 作阈值并 ``masked_fill_``。

    返回 ``None`` 表示「未接管」,由集成层交还原生 ``apply_top_k_top_p``。
    ``top_k`` 仅为对齐原生签名(不使用)。
    """
    ok, k0 = sampling_window_ok(logits, k, p)
    if not ok:
        return None
    x = logits.to(torch.bfloat16)
    if not _deepselect_topk_ok(x, k0):
        return None
    values, _ = _deepselect_topk(x, k0)
    cutoff = values.min(dim=-1, keepdim=True).values.to(logits.dtype)
    return logits.masked_fill_(logits < cutoff, float("-inf"))
