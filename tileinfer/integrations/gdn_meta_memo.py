"""H70-b GDN attention 元数据 builder 稳态 memo 化(E285;opt-in,默认关)。

针对 c1/c8 decode 稳态下 _build_attention_metadata 的最大单点
(E285:gdn_build 0.377-0.492ms/步,Qwen3.6 hybrid 3 个 GDN 组 × builder.build):

  Qwen3.6-35B-A3B decode 稳态(纯 decode、无 spec、mamba_cache_mode="none")
  下,GDNAttentionMetadataBuilder.build 的全部输出内容仅由
  「请求集合 → mamba 状态行(请求生命周期内固定)+ 均匀 qsl [0,1..] +
  purity」决定,逐步重算(treat 数据类拷贝/compute_num_computed_tokens/
  block_table 取表/pad fill+D2D×3/元数据构造/3 attach)为纯冗余 host+发射税。

  手法:build 包装层 memo——键全 CPU 侧可取(req_ids 元组、num_reqs、
  num_actual_tokens、attn_state、purity),命中即复用上一步返回的元数据
  对象(其字段引用持久设备缓冲,内容未变即语义等价);未命中走原实现并
  更新缓存。每 builder 实例独立缓存(id() 槽位)。

语义安全:
  - actual_seq_lengths = diff(padded qsl)(gdn_attn_builder.py:129),
    稳态常量;context_lens_tensor 仅 prefill 分支消费(miss 重算);
  - mamba_get_block_table_tensor 在 mode="none" 早退直返(E285 源码审查;
    非 "none" 直接 passthrough 不 memo);
  - spec decode(num_accepted_tokens/num_decode_draft_tokens 入参)passthrough;
  - 任一 purity/前置不满足(有 prefill、is_prefilling 非 CPU 张量、runner
    上下文缺失)passthrough;prefill/混合步恒 miss,捕获期 dummy 键与运行期
    真实键不相交。
  - 命中复用对象字段引用的持久缓冲由 reshape_and_cache/图 replay 读侧消费,
    跳过重写仅在「写入内容恒等」时发生。

门控:``TILEINFER_GDN_META_MEMO=1`` 显式开启(默认关=零行为变化);翻转
env + 重启即整体回退。安装点=tileinfer.integrations.vllm_ascend.install()。
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Optional

logger = logging.getLogger("tileinfer.gdn_meta_memo")

_INSTALLED = False

#: 引擎线程上的当前 runner(_build_attention_metadata 进入时置位,退出清除;
#: prep 块在引擎线程串行执行)。builder.build 无 runner 引用,经此取 req_ids。
_tls = threading.local()

_cache: dict[int, tuple[tuple[Any, ...], Any]] = {}
_hits = 0
_misses = 0
_pass = 0
_LOG_EVERY = 20000


def memo_enabled() -> bool:
    return (os.environ.get("TILEINFER_GDN_META_MEMO") or "").strip() == "1"


def stats() -> dict[str, int]:
    return {"hits": _hits, "misses": _misses, "passthrough": _pass}


def _maybe_log() -> None:
    global _hits, _misses, _pass
    total = _hits + _misses
    if total and total % _LOG_EVERY == 0:
        logger.warning(
            "gdn_meta_memo stats: hits=%d misses=%d passthrough=%d hit_rate=%.3f",
            _hits, _misses, _pass, _hits / max(1, total),
        )


def _memo_key(runner: Any, cm: Any) -> Optional[tuple[Any, ...]]:
    """构造全 CPU 侧 memo 键;任一前置不满足返回 None(passthrough)。"""
    pre = getattr(cm, "is_prefilling", None)
    if pre is None or not hasattr(pre, "device") or pre.device.type != "cpu":
        return None
    try:
        if bool(pre.any().item()):
            return None
    except Exception:
        return None
    req_ids = getattr(getattr(runner, "input_batch", None), "req_ids", None)
    if not isinstance(req_ids, (list, tuple)):
        return None
    nr = int(cm.num_reqs)
    if nr > len(req_ids):
        return None
    ids = tuple(req_ids[:nr])
    return (ids, nr, int(cm.num_actual_tokens), cm.attn_state)


def install_gdn_meta_memo() -> bool:
    """安装 memo 化;失败仅降级本模块,不影响在役路径。"""
    global _INSTALLED
    if _INSTALLED:
        return True
    if not memo_enabled():
        return False
    import vllm_ascend.worker.model_runner_v1 as mrv
    from vllm_ascend.ops.gdn_attn_builder import AscendGDNAttentionMetadataBuilder

    orig_bam = mrv.NPUModelRunner._build_attention_metadata

    def _build_attention_metadata(self, *args: Any, **kwargs: Any):
        _tls.runner = self
        try:
            return orig_bam(self, *args, **kwargs)
        finally:
            _tls.runner = None

    try:
        _build_attention_metadata.__wrapped__ = orig_bam  # type: ignore[attr-defined]
    except Exception:
        pass
    mrv.NPUModelRunner._build_attention_metadata = _build_attention_metadata  # type: ignore[method-assign]

    orig_build = AscendGDNAttentionMetadataBuilder.build

    def _build(
        bself: Any,
        common_prefix_len: int = 0,
        common_attn_metadata: Any = None,
        **extra: Any,
    ):
        global _hits, _misses, _pass
        slot = id(bself)
        if extra.get("num_accepted_tokens") is not None or extra.get("num_decode_draft_tokens_cpu") is not None:
            # spec 步会覆写 spec/持久缓冲:清槽防 stale 命中。
            _pass += 1
            _cache.pop(slot, None)
            return orig_build(bself, common_prefix_len, common_attn_metadata, **extra)
        if getattr(bself.vllm_config.cache_config, "mamba_cache_mode", "none") != "none":
            _pass += 1
            _cache.pop(slot, None)
            return orig_build(bself, common_prefix_len, common_attn_metadata, **extra)
        runner = getattr(_tls, "runner", None)
        slot = id(bself)
        if runner is None or common_attn_metadata is None:
            _pass += 1
            _cache.pop(slot, None)
            return orig_build(bself, common_prefix_len, common_attn_metadata, **extra)
        key = _memo_key(runner, common_attn_metadata)
        if key is None:
            # prefill/混合/捕获期步:重建会覆写持久缓冲,旧缓存对象一并失效。
            _misses += 1
            _cache.pop(slot, None)
            return orig_build(bself, common_prefix_len, common_attn_metadata, **extra)
        cached = _cache.get(slot)
        if cached is not None and cached[0] == key:
            _hits += 1
            _maybe_log()
            return cached[1]
        _misses += 1
        out = orig_build(bself, common_prefix_len, common_attn_metadata, **extra)
        _cache[slot] = (key, out)
        return out

    try:
        _build.__wrapped__ = orig_build  # type: ignore[attr-defined]
    except Exception:
        pass
    AscendGDNAttentionMetadataBuilder.build = _build  # type: ignore[method-assign]

    _INSTALLED = True
    logger.warning("TileInfer gdn meta memo ACTIVE (opt-in): memo key=(req_ids,nreqs,ntoks,attn_state,purity)")
    return True
