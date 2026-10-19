"""GDN in_proj_ba → in_proj_qkvz 权重融合(H32,系统级专项 S-GDN 投影族)。

机制(E100/E99):decode 每步每个 GDN 层执行两次 skinny GEMM——in_proj_qkvz
(M=16,K=2048,N=12288)与 in_proj_ba(M=16,K=2048,N=64,0.26MB,纯内核启动延迟税,
in-engine 3.8µs/次 ≈ 0.67% TPOT,E97)。把 ba 权重 cat 到 qkvz 权重尾部成单 GEMM
(N=12352),输出用零拷贝 strided 切片喂原 ``fix_query_key_value_ordering``:

* ``fused[:, :n_qkvz]`` / ``fused[:, n_qkvz:]`` 的 strided 视图传给上游 ordering 函数,
  其内部 ``view`` 仅做末维拆分(torch 对非连续张量的纯维度拆分合法,E101);
* 下游 rearrange/cat 的拷贝与现状逐次一致(现状的 split 视图同样非连续);
* GEMM 按元素同 K 序累加 → 融合与分离逐位等价(E100:bafused_maxdiff=0.0;
  vLLM 上游同构先例 #41457)。

守卫:仅 TP=1、无量化、gqa_interleaved_layout、双投影为普通线性权重时接管;
任何守卫不满足或运行期异常均回退原 ``forward_cuda``(``TILEINFER_OPS`` 每步读取,
翻转即回退,重启免编译)。fused 权重在首次 forward(预热期、图捕获前)懒构建,
额外显存 = 12352×2048×2B × 30 层 ≈ 1.5GB。
"""

from __future__ import annotations

import torch

try:  # einops 是 vllm 的硬依赖,forward_cuda 原实现同样使用
    from einops import rearrange
except ImportError:  # pragma: no cover - vllm 缺席时本模块不会被引擎加载
    rearrange = None


def build_fused_inproj_weight(layer: torch.nn.Module) -> torch.Tensor | None:
    """为 GDN 层构建 ``[n_qkvz + n_ba, hidden]`` fused 权重;守卫不满足返回 None。

    返回 None 时调用方应缓存失败标记,避免每步重复守卫(静态条件,不会自愈)。

    两种 checkpoint 布局都支持(Qwen3.6-35B-A3B 是后者):
    * ``gqa_interleaved_layout=True``(Qwen3-Next):qkvz / ba 各自按 k-head 组交错;
    * ``gqa_interleaved_layout=False``(Qwen3.5):qkvz 为 ``[q|k|v|z]`` 四段拼接,
      ba 为 ``[b|a]`` 两段——权重 cat 后按段切回即可,无需重排。
    """
    if getattr(layer, "tp_size", 1) != 1:
        # TP>1 时 in_proj_ba 可能 disable_tp(复制)而 qkvz 分片,cat 语义不成立
        return None
    qkvz = getattr(layer, "in_proj_qkvz", None)
    ba = getattr(layer, "in_proj_ba", None)
    if qkvz is None or ba is None:
        return None
    wq = getattr(qkvz, "weight", None)
    wb = getattr(ba, "weight", None)
    if (
        wq is None
        or wb is None
        or wq.dim() != 2
        or wb.dim() != 2
        or wq.shape[1] != wb.shape[1]
        or not wq.is_floating_point()
        or not wb.is_floating_point()
    ):
        return None
    if getattr(layer, "quant_config", None) is not None:
        return None
    return torch.cat([wq.data, wb.data], dim=0).contiguous()


def fused_forward_cuda(layer: torch.nn.Module, hidden_states: torch.Tensor,
                       output: torch.Tensor) -> bool:
    """H32 融合路径的 forward 替身;返回是否接管。

    同时支持两种布局(vllm 0.23.0 qwen_gdn_linear_attn.py:908-967):
    * ``gqa_interleaved_layout=True``:ordering → 3×rearrange → cat;
    * ``gqa_interleaved_layout=False``(Qwen3.5,本模型):qkvz 按
      ``[qkv_size, z_size]`` 切分、ba 按 ``chunk(2)`` 切 b/a —— 与上游逐行同构。
    上游演进时本函数需同步复核。
    """
    weight = getattr(layer, "_tileinfer_fused_inproj", None)
    if weight is None:
        if getattr(layer, "_tileinfer_fused_inproj_failed", False):
            return False
        weight = build_fused_inproj_weight(layer)
        if weight is None:
            object.__setattr__(layer, "_tileinfer_fused_inproj_failed", True)
            return False
        object.__setattr__(layer, "_tileinfer_fused_inproj", weight)

    num_tokens = hidden_states.size(0)
    fused = torch.nn.functional.linear(hidden_states, weight)
    n_qkvz = layer.in_proj_qkvz.weight.shape[0]
    # 零拷贝 strided 切片:下游只做切分/纯维度 view(E101)
    mixed_qkvz = fused[:, :n_qkvz]
    ba = fused[:, n_qkvz:]

    if getattr(layer, "gqa_interleaved_layout", False):
        query, key, value, z, b, a = layer.fix_query_key_value_ordering(mixed_qkvz, ba)
        query, key, value = map(
            lambda x: rearrange(x, "l p d -> l (p d)"), (query, key, value)
        )
        mixed_qkv = torch.cat((query, key, value), dim=-1)
    else:
        # Qwen3.5:权重已是 [q, k, v, z] / [b, a] 顺序,与上游 forward_cuda 同构
        qkv_size = (layer.key_dim * 2 + layer.value_dim) // layer.tp_size
        z_size = layer.value_dim // layer.tp_size
        mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
        z = z.reshape(z.size(0), -1, layer.head_v_dim)
        b, a = layer.split_ba(ba)
        b = b.contiguous()
        a = a.contiguous()

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
