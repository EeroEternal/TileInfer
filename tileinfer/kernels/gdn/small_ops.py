"""GDN 小核链削减(H35,S-GDN 域;依据 E105/vllm-ascend #19033)。

decode 每步每层三个定距命中的小核开销(E105,decode 窗 4345.9ms 归因):
conv1d 权重每步转置 0.8% + rearrange cat 材料化 0.7% + q/k l2norm 双调用 0.8%。
本模块提供三个相互独立、均可按 ``TILEINFER_OPS=gdnsmall`` 每步读取翻转的包装:

1. rearrange cat-free:覆写 ``rearrange_mixed_qkv``,split 后直接 reshape(视图,
   c=1 时全连续零拷贝),消除每层每步 reshape(-1)×3 + ``torch.cat`` 的
   ConcatD 材料化拷贝(#19033 b 项;上游 vllm-ascend #18723 同构);
2. conv1d 权重转置缓存:包装 ``torch.ops._C_ascend.npu_causal_conv1d_custom``,
   第三位权重参数(转置视图)按 (storage_ptr, shape, stride, dtype) 键惰性物化
   连续版本并缓存(storage 弱引用保活校验,防指针复用陈旧),消除每层每步
   ``aclnnContiguous_Transpose``(#19033 a 项);
3. qk l2norm 配对融合:包装 ``vllm_ascend.ops.gdn.l2norm_fwd``,q/k 为同一
   连续缓冲相邻区域(seq_len==1)时一次内核计算两份归一化,返回各自视图。
   l2norm 沿末维逐行归一,行分组与结果无关 → 与两次调用数值等价(#19033 c 项)。

数值契约:1/3 与两次调用/原始路径 TOL 级等价(同一内核、同逐行归一语义);
2 为同一存储的预物化拷贝,逐位一致。任何守卫不满足或异常均回退原实现。
"""

from __future__ import annotations

import weakref
from typing import Any, Callable

import torch

#: 探针计数器(安装时同步进后端类 _stats,metrics dumper 读取)
COUNTERS: dict[str, int] = {
    "gdnsmall_rearrange_calls": 0,
    "gdnsmall_conv_hits": 0,
    "gdnsmall_l2norm_fused": 0,
    "gdnsmall_fallbacks": 0,
}

_stats_sink: Callable[[str], None] | None = None
_conv_ns: Any = None  # torch.ops._C_ascend 命名空间(安装时保存,供卸载/测试)
_orig_conv: Callable | None = None  # npu_causal_conv1d_custom 原始绑定
_orig_l2norm: Callable | None = None  # vllm_ascend.ops.gdn.l2norm_fwd 原始符号
_conv_cache: dict[tuple, tuple] = {}
_conv_cache_max = 64
_pair_stash: dict[str, Any] | None = None  # rearrange 产出的 q 融合候选
_pair_slot: dict[str, Any] | None = None  # q 融合完成后等待 k 的槽


def _inc(name: str) -> None:
    COUNTERS[name] = COUNTERS.get(name, 0) + 1
    if _stats_sink is not None:
        _stats_sink(name)


# ---------------------------------------------------------------- conv1d 缓存


def _conv_cached_contig(w: torch.Tensor) -> torch.Tensor | None:
    """对非连续二维权重视图返回缓存的连续版本;不可缓存返回 None。"""
    try:
        storage = w.untyped_storage()
    except Exception:  # noqa: BLE001
        return None
    key = (storage.data_ptr(), tuple(w.shape), tuple(w.stride()), w.dtype)
    ent = _conv_cache.get(key)
    if ent is not None:
        sref = ent[0]()
        if sref is not None and sref.data_ptr() == key[0]:
            return ent[1]
        del _conv_cache[key]
    contig = w.contiguous()
    if len(_conv_cache) >= _conv_cache_max:
        _conv_cache.pop(next(iter(_conv_cache)))
    _conv_cache[key] = (weakref.ref(storage), contig)
    return contig


def make_conv1d_wrapper(orig: Callable) -> Callable:
    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        if not _ops_enabled():
            return orig(*args, **kwargs)
        try:
            if len(args) >= 3:
                w = args[2]
                if (
                    torch.is_tensor(w)
                    and not w.is_contiguous()
                    and w.dim() == 2
                    and w.numel() <= (1 << 20)
                    and w.is_floating_point()
                ):
                    cached = _conv_cached_contig(w)
                    if cached is not None:
                        args = args[:2] + (cached,) + args[3:]
                        _inc("gdnsmall_conv_hits")
        except Exception:  # noqa: BLE001 - 任何异常回退原调用
            _inc("gdnsmall_fallbacks")
        return orig(*args, **kwargs)

    return _wrapped


# ------------------------------------------------------- rearrange cat-free


def cat_free_rearrange(self, mixed_qkv: torch.Tensor | None):
    """rearrange_mixed_qkv 替身:split + reshape 视图,无 cat;语义与上游一致。

    上游(vllm 0.23.0 qwen_gdn_linear_attn.py:808)对 None 返回 (None,)*3、
    按 [q_dim, k_dim, v_dim] 沿末维 split 后经 reshape(-1)+cat+view 物化;
    本替身保留 split 与 view 语义、跳过全部材料化(c=1 时 split 切片本身
    连续,reshape 返回零拷贝视图)。
    """
    global _pair_stash
    orig = getattr(self, "_tileinfer_rearrange_orig", None)
    enabled = _ops_enabled()
    if not enabled or mixed_qkv is None:
        return orig(self, mixed_qkv)
    try:
        seq_len = mixed_qkv.shape[0]
        q_dim = self.key_dim // self.tp_size
        k_dim = self.key_dim // self.tp_size
        v_dim = self.value_dim // self.tp_size
        query, key, value = torch.split(mixed_qkv, [q_dim, k_dim, v_dim], dim=-1)
        query = query.reshape(1, seq_len, -1, self.head_k_dim)
        key = key.reshape(1, seq_len, -1, self.head_k_dim)
        value = value.reshape(1, seq_len, -1, self.head_v_dim)
        _inc("gdnsmall_rearrange_calls")
        if (
            seq_len == 1
            and mixed_qkv.is_contiguous()
            and mixed_qkv.is_floating_point()
        ):
            # qk 相邻区融合候选:q 位于缓冲 0..q_numel,k 位于 q_numel..2*q_numel
            _pair_stash = {
                "tref": weakref.ref(mixed_qkv),
                "ptr": mixed_qkv.data_ptr(),
                "dtype": mixed_qkv.dtype,
                "shape": tuple(query.shape),
                "q_numel": query.numel(),
            }
        else:
            _pair_stash = None
        return query, key, value
    except Exception:  # noqa: BLE001
        _inc("gdnsmall_fallbacks")
        return orig(self, mixed_qkv)


# ------------------------------------------------------ l2norm 配对融合


def make_l2norm_wrapper(orig: Callable) -> Callable:
    def _wrapped(x: torch.Tensor, *a: Any, **k: Any) -> Any:
        global _pair_slot, _pair_stash
        if not _ops_enabled() or not torch.is_tensor(x):
            return orig(x, *a, **k)
        try:
            if _pair_slot is not None:
                slot, _pair_slot = _pair_slot, None
                base = slot["tref"]()
                if (
                    base is not None
                    and x.dtype == slot["dtype"]
                    and tuple(x.shape) == slot["shape"]
                    and x.data_ptr() == slot["ptr"] + slot["q_numel"] * x.element_size()
                ):
                    # x 是配对的 k:q 调用时已融合计算,直接返回预存的 k 视图
                    return slot["k_out"]
            stash, _pair_stash = _pair_stash, None
            if stash is not None:
                base = stash["tref"]()
                if (
                    base is not None
                    and x.dtype == stash["dtype"]
                    and tuple(x.shape) == stash["shape"]
                    and x.data_ptr() == stash["ptr"]
                ):
                    q_numel = stash["q_numel"]
                    flat = base.reshape(-1)
                    # x 形如 (1, 1, H, D):融合视图 (1, 2, H, D),flat 前 2*q_numel
                    # 恰为 [q 区 | k 区](seq_len==1,缓冲连续)
                    qk = (
                        flat[: 2 * q_numel].reshape(1, 2, *x.shape[2:])
                        if x.dim() == 4
                        else None
                    )
                    if qk is not None:
                        fused = orig(qk, *a, **k)
                        _pair_slot = {
                            "tref": stash["tref"],
                            "ptr": stash["ptr"],
                            "dtype": stash["dtype"],
                            "shape": stash["shape"],
                            "q_numel": q_numel,
                            "k_out": fused[:, 1:2],
                        }
                        _inc("gdnsmall_l2norm_fused")
                        return fused[:, 0:1]
        except Exception:  # noqa: BLE001
            _pair_slot = None
            _pair_stash = None
            _inc("gdnsmall_fallbacks")
        return orig(x, *a, **k)

    return _wrapped


# ------------------------------------------------------------ 安装/卸载


def _ops_enabled() -> bool:
    try:
        from ...integrations.vllm_ascend import ops_enabled

        return ops_enabled("gdnsmall")
    except Exception:  # noqa: BLE001
        return False


def reset_state() -> None:
    """清空缓存/配对状态(测试与卸载辅助)。"""
    global _pair_stash, _pair_slot
    _conv_cache.clear()
    _pair_stash = None
    _pair_slot = None


def reset_counters() -> None:
    """计数器清零(测试辅助)。"""
    for name in list(COUNTERS):
        COUNTERS[name] = 0


def uninstall() -> None:
    """还原三处补丁(测试辅助;引擎内不使用——翻转 TILEINFER_OPS 即回退)。"""
    global _stats_sink, _orig_conv, _orig_l2norm
    if _conv_ns is not None and _orig_conv is not None:
        try:
            setattr(_conv_ns, "npu_causal_conv1d_custom", _orig_conv)
        except Exception:  # noqa: BLE001
            pass
    if _orig_l2norm is not None:
        try:
            import vllm_ascend.ops.gdn as _gdn_mod

            _gdn_mod.l2norm_fwd = _orig_l2norm
        except Exception:  # noqa: BLE001
            pass
    try:
        from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
            QwenGatedDeltaNetAttention as _gdn_cls,
        )

        orig = getattr(_gdn_cls, "_tileinfer_rearrange_orig", None)
        if orig is not None:
            _gdn_cls.rearrange_mixed_qkv = orig
            del _gdn_cls._tileinfer_rearrange_orig
        _gdn_cls._tileinfer_gdnsmall_installed = False
    except Exception:  # noqa: BLE001
        pass
    _stats_sink = None
    _orig_conv = None
    _orig_l2norm = None
    reset_state()
