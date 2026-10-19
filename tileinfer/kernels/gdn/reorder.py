"""GDN fused(qkvz+ba)权重 load 期行重排(H34,系统级专项 S-GDN 投影族)。

机制(E102/E105):H32 融合后,decode 每步仍执行 ``fix_query_key_value_ordering``
( interleaved 布局的 view+split+reshape)→ 3×rearrange → cat,每次都有真实拷贝
(E105:Cat [32768+32768+65536] ×7650 定距命中)。本模块在 load 期把 fused 权重行序
重排为消费终序 ``[q|k|v|z|b|a]``(q/k 按 k-head 组主序扁平,v/z/b/a 按组主序),
使单 GEMM 输出列序即终态,forward 仅需零拷贝切片:

* ``mixed_qkv = fused[:, :2*dk+dv]`` 直接即上游 cat 结果;
* ``z`` 为末维纯拆分 view(E101 同 legality);``b``/``a`` 为行连续切片;
* 权重行重排=输出列纯置换,GEMM 每列同 K 序点积 → 逐位等价(CPU 契约验证;
  设备上不同 N 的 MatMulV3 tiling 差异由 TOL 断言覆盖,与 H32 同口径)。

本文件为纯新增路径,不触碰 H32 在役代码;接线(独立 opt-in token)留待 H32
权威口径 A/B 收口后进行(防在役链单变量污染)。
"""

from __future__ import annotations

import torch

from tileinfer.kernels.gdn.proj_fusion import build_fused_inproj_weight


def reorder_rows_plan(num_k_heads: int, num_v_heads: int, head_k_dim: int,
                      head_v_dim: int, tp_size: int = 1) -> torch.Tensor:
    """构建行重排置换 ``plan``:新权重第 i 行=旧 fused 权重第 ``plan[i]`` 行。

    旧 fused 行序(与上游 checkpoint 一致):qkvz 按 k-head 组交错
    ``[g0: q|k|v|z, g1: q|k|v|z, ...]``,ba 按组交错 ``[g0: b|a, g1: b|a, ...]``;
    新行序=消费终序 ``[q_all|k_all|v_all|z_all|b_all|a_all]``(组主序扁平)。
    """
    if num_k_heads <= 0 or num_v_heads <= 0 or tp_size <= 0:
        raise ValueError("invalid head config")
    if num_v_heads % num_k_heads != 0 or num_k_heads % tp_size != 0:
        raise ValueError("head counts not divisible (interleaved layout)")
    ng = num_k_heads // tp_size
    vpg = num_v_heads // num_k_heads
    dk, dv = head_k_dim, head_v_dim
    group = 2 * dk + 2 * vpg * dv
    n_qkvz = ng * group

    qkvz: list[int] = []
    for lo in range(0, 2 * dk, dk):  # q_all, k_all
        for g in range(ng):
            base = g * group + lo
            qkvz.extend(range(base, base + dk))
    for lo in (2 * dk, 2 * dk + vpg * dv):  # v_all, z_all
        for g in range(ng):
            base = g * group + lo
            qkvz.extend(range(base, base + vpg * dv))

    ba: list[int] = []
    for g in range(ng):  # b_all(组主序:g 外层、j 内层,与 reshape 语义一致)
        for j in range(vpg):
            ba.append(n_qkvz + g * 2 * vpg + j)
    for g in range(ng):  # a_all
        for j in range(vpg):
            ba.append(n_qkvz + g * 2 * vpg + vpg + j)
    return torch.tensor(qkvz + ba, dtype=torch.long)


def build_reordered_inproj_weight(layer: torch.nn.Module) -> torch.Tensor | None:
    """构建终序 fused 权重;守卫(继承 H32)或形状不符返回 None。

    仅支持 ``gqa_interleaved_layout=True``(Qwen3-Next):行重排 plan 是按
    "qkvz/ba 各自按 k-head 组交错"的 checkpoint 序推导的。Qwen3.5 布局
    (``False``,本模型 Qwen3.6-35B-A3B)的权重已是 ``[q|k|v|z]``/``[b|a]`` 段序,
    没有可消除的 ordering 拷贝 —— 此时返回 None,由 H32 路径接管即可。
    """
    if not getattr(layer, "gqa_interleaved_layout", False):
        return None
    base = build_fused_inproj_weight(layer)
    if base is None:
        return None
    try:
        plan = reorder_rows_plan(
            layer.num_k_heads, layer.num_v_heads,
            layer.head_k_dim, layer.head_v_dim, layer.tp_size,
        )
    except (AttributeError, ValueError, ZeroDivisionError):
        return None
    if plan.numel() != base.shape[0]:
        return None
    return base[plan].contiguous()


def split_reordered(fused: torch.Tensor, num_k_heads: int, num_v_heads: int,
                    head_k_dim: int, head_v_dim: int, tp_size: int = 1):
    """终序 GEMM 输出的零拷贝拆分:返回 ``(mixed_qkv, z, b, a)``。

    形状与上游 ordering+rearrange+cat 的消费侧逐位一致(见模块 docstring)。
    """
    ng = num_k_heads // tp_size
    dk_total = ng * head_k_dim
    nv_local = num_v_heads // tp_size
    dv_total = nv_local * head_v_dim
    n_qkv = 2 * dk_total + dv_total
    tokens = fused.size(0)
    mixed_qkv = fused[:, :n_qkv]
    z = fused[:, n_qkv:n_qkv + dv_total].view(tokens, nv_local, head_v_dim)
    b = fused[:, n_qkv + dv_total:n_qkv + dv_total + nv_local]
    a = fused[:, n_qkv + dv_total + nv_local:]
    return mixed_qkv, z, b, a


def fused_forward_reordered(layer: torch.nn.Module, hidden_states: torch.Tensor,
                            output: torch.Tensor) -> bool:
    """H34 终序路径的 ``forward_cuda`` 替身;返回是否接管。

    尾部(core op→output projection)与 H32 ``fused_forward_cuda`` 逐行同构,
    仅前置 ordering/rearrange/cat 被终序切片取代。懒构建失败缓存属性独立命名,
    与 H32 路径互不可见。
    """
    weight = getattr(layer, "_tileinfer_reordered_inproj", None)
    if weight is None:
        if getattr(layer, "_tileinfer_reordered_inproj_failed", False):
            return False
        weight = build_reordered_inproj_weight(layer)
        if weight is None:
            object.__setattr__(layer, "_tileinfer_reordered_inproj_failed", True)
            return False
        object.__setattr__(layer, "_tileinfer_reordered_inproj", weight)

    num_tokens = hidden_states.size(0)
    fused = torch.nn.functional.linear(hidden_states, weight)
    mixed_qkv, z, b, a = split_reordered(
        fused, layer.num_k_heads, layer.num_v_heads,
        layer.head_k_dim, layer.head_v_dim, layer.tp_size,
    )

    core_attn_out = torch.zeros(
        (num_tokens, layer.num_v_heads // layer.tp_size, layer.head_v_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        _encode_layer_name,
    )

    torch.ops.vllm.qwen_gdn_attention_core(
        mixed_qkv,
        b,
        a,
        core_attn_out,
        layer_name=_encode_layer_name(layer.prefix),
    )
    layer._output_projection(core_attn_out, z, output, num_tokens)
    return True
