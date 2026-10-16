"""Sampling top-k threshold selection via DeepSelect ``topk`` (BACKLOG 采样条目).

vllm-ascend A5 的原生采样路径对**整张词表**（V≈152K）做 ``probs.sort``
（O(B·V log V)）来求第 k 大概率作 top-k 阈值。softmax 严格单调，故 ``prob_i <
第k大 prob`` ⟺ ``logit_i < 第k大 logit``——用 DeepSelect ``topk``（bf16、
``sorted=False``、int32 索引）在 logits 上求第 k 大 *logit* 作阈值即可，设备耗时
实测降至原生 0.12~0.17x（词表/batch 包络内，见 tileinfer_topk_poc/PERFORMANCE.md）。

来源与证据：``npu_flash`` 侧 ``ops/topk.py`` / ``integration/vllm_ascend.py``
（离线 L1 wall 0.87~0.88x、L2 设备 0.12~0.17x、L3 门控零越界；在线 TPOT +16~22%），
按 ``install_gdn_gating`` 的「替换原函数 + 逐调用门控 + 失败回退 + 在役探针」范式
映射进 TileInfer。

硬约束（对齐 DeepSelect ``interface.py``，不满足一律回退原生路径）：

- 2D ``(batch, vocab)`` **bf16**（Ascend 仅支持 bf16）；
- last-dim ``stride == 1``，``stride(0)`` 为 32B 整数倍；
- 仅支持 ``sorted=False``、``indices_type=torch.int32``。
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

__all__ = [
    "SAMPLING_MIN_VOCAB",
    "SAMPLING_MIN_BATCH",
    "TOPK_INDICES_DTYPE",
    "TOPK_STRIDE_ALIGN_BYTES",
    "sampler_in_envelope",
    "deepselect_topk_ok",
    "deepselect_topk",
    "apply_top_k_top_p",
]

# 门控包络（PERFORMANCE.md L2/L3 实测依据）：
# - 词表 >= 16384 起任意 batch 均正优化（V=8192 在 b=64 仍 1.17x neg）；
# - batch >= 8 起固定 host 开销被摊薄（b=2/4 为 1.14x/1.09x neg，b>=8 → 0.54~0.11x）。
SAMPLING_MIN_VOCAB = 16384
SAMPLING_MIN_BATCH = 8

# DeepSelect Ascend 后端仅支持 int32 索引；输入 stride(0) 须为 32B 整数倍
TOPK_INDICES_DTYPE = torch.int32
TOPK_STRIDE_ALIGN_BYTES = 32


def sampler_in_envelope(shape: Tuple[int, int]) -> bool:
    """采样窗口包络门控：词表 ≥ 16384 且 batch ≥ 8（2D ``(batch, vocab)``）。

    与 op 级门控（任意 batch 均正）不同，采样链每次调用多一层 host 侧窗口判定与
    ``masked_fill`` 固定开销，小 batch 无法摊薄（b=1 持平、b=2/4 为负），故 b<8
    一律交还原生。**纯宿主侧 shape 读取，零设备同步**——调用方（graph 路径）依赖
    这一点保证回退路径零 D2H。
    """
    if shape is None or len(shape) != 2:
        return False
    batch, vocab = int(shape[0]), int(shape[-1])
    return vocab >= SAMPLING_MIN_VOCAB and batch >= SAMPLING_MIN_BATCH


def deepselect_topk_ok(
    input: torch.Tensor,
    k: int,
    dim: int,
    sorted: bool,
    indices_type: torch.dtype,
) -> bool:
    """是否满足 DeepSelect ``topk`` 的 Ascend 硬约束（不满足即回退 torch.topk）。"""
    if input.device.type != "npu":
        return False
    if input.dtype != torch.bfloat16:  # Ascend 仅支持 bf16
        return False
    if input.dim() != 2:
        return False
    if dim not in (-1, 1):
        return False
    if sorted:  # Ascend 仅支持 sorted=False
        return False
    if indices_type != torch.int32:
        return False
    if input.stride(-1) != 1:
        return False
    if (input.stride(0) * input.element_size()) % TOPK_STRIDE_ALIGN_BYTES != 0:
        return False
    if int(input.shape[-1]) < int(k):
        return False
    return True


def deepselect_topk(
    input: torch.Tensor,
    k: int,
    dim: int = -1,
    *,
    sorted: bool = False,
    indices_type: torch.dtype = TOPK_INDICES_DTYPE,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """DeepSelect ``topk``（bf16 / 2D / ``sorted=False`` / int32），约束不满足回退 torch。"""
    if deepselect_topk_ok(input, k, dim, sorted, indices_type):
        import deep_select

        assert dim in (-1, input.dim() - 1), "DeepSelect 仅支持在最后一维做 topk"
        return deep_select.topk(input, k, sorted=sorted, indices_type=indices_type)
    values, indices = torch.topk(input, k, dim=dim, sorted=sorted, largest=True)
    if indices.dtype != indices_type:
        indices = indices.to(indices_type)
    return values, indices


# -- 采样 k 标量的跨步缓存 ------------------------------------------------------
# 键 = `k` 张量**对象身份**（强引用，防内存复用致 id 假命中）+ vocab。稳态解码下
# vLLM 复用同一 `sampling_metadata.top_k` 切片对象逐 step 传入，故稳态**零 D2H**；
# 批次组成变化时 `_make_sampling_metadata` 用切片 `self.top_k[:num_reqs]` 重建**新
# 对象**，身份随之变化 → 自动失效重解析。正确性依据：**同一对象 ⟺ 值未变**。
#
# 【与 TileInfer「plan/run 分离、run 零 host 同步」设计的关系】这是唯一必需例外：
# k 是采样元数据（逐步变化、无 plan 期可锚定的设备侧副本），plan 期无法解析；
# 缓存后稳态零 D2H，图模式（ACLGraph）下不引入 host 同步——若每步 `k.item()` 会
# 打断异步流水、抵消收益（npu_flash §5.5 回退根因）。
_K_CACHE_K: Optional[torch.Tensor] = None
_K_CACHE_VOCAB: int = -1
_K_CACHE_RESULT: tuple = (False, 0)


def reset_k_cache() -> None:
    """清空采样 `k` 标量缓存（挂/卸载 patch 时调用，释放强引用）。"""
    global _K_CACHE_K, _K_CACHE_VOCAB, _K_CACHE_RESULT
    _K_CACHE_K = None
    _K_CACHE_VOCAB = -1
    _K_CACHE_RESULT = (False, 0)


def _resolve_k_scalar_cached(k: torch.Tensor, vocab: int) -> tuple:
    """解析标量 k（要求各行同值、且 ``0 < k0 < vocab``）并跨步缓存，返回 ``(ok, k0)``。

    仅在缓存未命中时执行一次 D2H（``k.to(long).reshape(-1).cpu().tolist()``，~15us）；
    命中（``k`` 为同一对象且 vocab 一致）时**零 D2H**。
    """
    global _K_CACHE_K, _K_CACHE_VOCAB, _K_CACHE_RESULT
    if _K_CACHE_K is k and _K_CACHE_VOCAB == vocab:
        return _K_CACHE_RESULT
    vals = k.to(torch.long).reshape(-1).cpu().tolist()
    k0 = vals[0]
    ok = 0 < k0 < vocab
    if ok:
        for v in vals:
            if v != k0:  # DeepSelect 仅支持标量 k；异构批次交还原生
                ok = False
                break
    _K_CACHE_K = k
    _K_CACHE_VOCAB = vocab
    _K_CACHE_RESULT = (ok, k0 if ok else 0)
    return _K_CACHE_RESULT


def apply_top_k_top_p(logits: torch.Tensor, p: Optional[torch.Tensor], k: Optional[torch.Tensor]):
    """top-k 阈值选择（DeepSelect 替代原生全词表 ``probs.sort``），原生语义保持。

    等价性：softmax 严格单调，``prob_i < 第k大 prob`` ⟺ ``logit_i < 第k大 logit``；
    原生保留并列项（``probs < cutoff`` 丢弃），本实现在 logits 上取第 k 大 logit 作
    阈值后 ``logits < cutoff``，同样保留并列项，语义一致。

    调用契约（``tileinfer.integrations.vllm_ascend.install_sampler_topk`` 保证）：
    仅在 top-k-only 且落包络窗口时被调用；本函数仍自检窗口与硬约束，任一不满足
    返回 ``None``（调用方回退原实现）。

    **检查顺序即性能契约**：所有零 D2H 的门控（含包络）都在取 k 之前完成，确保
    回退路径零额外开销。
    """
    # 1) 零 D2H 门控：仅 top-k（p is None）、2D 浮点 NPU logits、包络内
    if p is not None or k is None:
        return None
    if logits.device.type != "npu" or logits.dim() != 2:
        return None
    if not logits.is_floating_point() or logits.numel() == 0:
        return None
    if not isinstance(k, torch.Tensor) or k.numel() == 0:
        return None
    batch, vocab = int(logits.shape[0]), int(logits.shape[-1])
    if not sampler_in_envelope((batch, vocab)):
        return None
    # 2) 仅在通过全部门控后才解析 k（缓存命中零 D2H；未命中付一次）
    ok, k0 = _resolve_k_scalar_cached(k, vocab)
    if not ok:
        return None
    # 3) DeepSelect 求第 k 大 logit（bf16 / sorted=False / int32），min 作阈值
    values, _ = deepselect_topk(
        logits.to(torch.bfloat16), k0, dim=-1, sorted=False, indices_type=torch.int32
    )
    cutoff = values.min(dim=-1, keepdim=True).values.to(logits.dtype)
    return logits.masked_fill_(logits < cutoff, float("-inf"))
