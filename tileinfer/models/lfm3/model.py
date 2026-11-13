"""vLLM model implementation for LFM3 (route O).

LFM3 has no upstream vLLM class and its HF remote code is not compatible with vLLM's
attention layout.  This module provides the model structure against vLLM's layer primitives:
gated short convolutions, sliding/full GQA attention with per-head output gates and two RoPE
parameter sets, sandwich norms and a SwiGLU MLP.  The convolution math lives in the
engine-free provider ``tileinfer.providers.state.short_conv``; this file only owns the state
cache and metadata plumbing.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from itertools import islice

import torch
import torch.nn as nn

from vllm.config import (
    CacheConfig,
    ModelConfig,
    VllmConfig,
    get_current_vllm_config,
)
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    is_conv_state_dim_first,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsHybrid,
    SupportsLoRA,
    SupportsPP,
    SupportsQuant,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

from ...providers.state.short_conv import ShortConvState


def _padded_state_len(
    dim: int,
    kernel: int,
    head_dim: int,
    num_kv_heads: int,
    dtype_size: int,
) -> int:
    """Pad the tiny convolution state to vllm-ascend's attention page size.

    The Ascend mamba patch asserts that the per-block SSM page equals
    ``128 * head_dim * num_kv_heads * dtype_size`` bytes.  A short convolution state is
    ``dim * (kernel - 1) * dtype_size`` bytes, far smaller, so the state allocation is padded;
    the math still uses only the first ``kernel - 1`` slots.
    """
    state_len = kernel - 1
    target = 128 * head_dim * num_kv_heads * dtype_size
    base = dim * state_len * dtype_size
    blocks = max(1, math.ceil(base / target))
    padded = (blocks * target) // (dim * dtype_size)
    return max(state_len, padded)


class Lfm3MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.w13 = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.w13",
        )
        self.w2 = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.w2",
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.w13(x)
        x = self.act_fn(gate_up)
        x, _ = self.w2(x)
        return x


class Lfm3RMSNorm(nn.Module):
    """HF-equivalent RMSNorm (float32 variance), used to keep the reference comparison exact."""

    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(dtype)


class Lfm3RotaryEmbedding(nn.Module):
    """LFM3's zero-tail partial RoPE.

    HF keeps the full-head frequency progression and zeroes the tail of ``inv_freq`` instead
    of shrinking the rotated dimension.  The full 128-dim ``rotate_half`` then rotates only
    the first ``partial * head_dim`` dimensions (pairs 0..31 with 64..95 for a 0.5 factor).
    vLLM's ``RotaryEmbedding`` applies partial rotary to the first 64 dims with NeoX pairing,
    which is a different transform, so LFM3 carries its own small implementation.
    """

    def __init__(self, head_dim: int, base: float, partial: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (
            base
            ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / float(head_dim))
        )
        angles = int(partial * head_dim) // 2
        inv_freq = torch.cat(
            (inv_freq[:angles], inv_freq.new_zeros(head_dim // 2 - angles))
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = positions.float()[:, None] * self.inv_freq.to(positions.device)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(query.dtype)[:, None, :]
        sin = emb.sin().to(query.dtype)[:, None, :]
        return (
            query * cos + self._rotate_half(query) * sin,
            key * cos + self._rotate_half(key) * sin,
        )


class Lfm3Attention(nn.Module):
    def __init__(
        self,
        config,
        layer_idx: int,
        cache_config: CacheConfig | None = None,
        quant_config=None,
        sliding_window: int | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_heads = self.total_num_heads
        self.num_kv_heads = self.total_num_kv_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.sliding_window = sliding_window

        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.gate_proj = ColumnParallelLinear(
            input_size=self.hidden_size,
            output_size=self.total_num_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_proj",
        )
        self.q_norm = Lfm3RMSNorm(self.head_dim, eps=config.norm_eps)
        self.k_norm = Lfm3RMSNorm(self.head_dim, eps=config.norm_eps)

        rope_params = dict(config.rope_parameters[self.layer_type])
        self.rotary_emb = Lfm3RotaryEmbedding(
            self.head_dim,
            base=float(rope_params["rope_theta"]),
            partial=float(rope_params.get("partial_rotary_factor", 1.0)),
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=sliding_window,
            prefix=f"{prefix}.attn",
        )

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        n_tokens, _ = hidden_states.shape
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(n_tokens, self.num_heads, self.head_dim)
        k = k.view(n_tokens, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        q = q.reshape(n_tokens, -1).contiguous()
        k = k.reshape(n_tokens, -1).contiguous()
        v = v.contiguous()
        attn_output = self.attn(q, k, v)
        gate, _ = self.gate_proj(hidden_states)
        gate = torch.sigmoid(gate).unsqueeze(-1)
        attn_output = attn_output.view(n_tokens, self.num_heads, self.head_dim)
        attn_output = (attn_output * gate).reshape(n_tokens, -1)
        output, _ = self.o_proj(attn_output)
        return output


class Lfm3ShortConv(MambaBase, nn.Module):
    """Gated depthwise short convolution with vLLM's short-conv state metadata."""

    def __init__(
        self,
        config,
        dim: int,
        layer_idx: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        if tp_size != 1:
            raise NotImplementedError("LFM3 short convolution is single-card only")
        self.config = config
        self.layer_idx = layer_idx
        self.conv_dim = dim
        self.kernel = config.conv_L_cache
        self.bias = config.conv_bias
        self.model_config = model_config
        self.cache_config = cache_config
        self.prefix = prefix

        self.in_proj = ColumnParallelLinear(
            input_size=dim,
            output_size=3 * dim,
            bias=self.bias,
            quant_config=quant_config,
            prefix=f"{prefix}.in_proj",
        )
        self.out_proj = RowParallelLinear(
            input_size=dim,
            output_size=dim,
            bias=self.bias,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )
        self.conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=self.kernel,
            groups=dim,
            bias=self.bias,
            padding=self.kernel - 1,
        )
        self.state_len = self.kernel - 1
        dtype = model_config.dtype if model_config is not None else torch.bfloat16
        self.state_len_padded = _padded_state_len(
            dim,
            self.kernel,
            config.head_dim,
            config.num_key_value_heads,
            torch.tensor([], dtype=dtype).element_size(),
        )
        self.kv_cache = (torch.tensor([]),)

        compilation_config = get_current_vllm_config().compilation_config
        compilation_config.static_forward_context[prefix] = self

    def forward(self, hidden_states: torch.Tensor, output: torch.Tensor) -> None:
        forward_context = get_forward_context()
        metadata = forward_context.attn_metadata
        BCx, _ = self.in_proj(hidden_states)
        if not isinstance(metadata, dict) or self.prefix not in metadata:
            B, C, x = BCx.chunk(3, dim=-1)
            y = C * (B * x)
            out, _ = self.out_proj(y)
            output[: y.shape[0]] = out
            return

        md = metadata[self.prefix]
        num_decode = int(md.num_decode_tokens)
        num_prefill = int(md.num_prefill_tokens)
        num_actual = num_decode + num_prefill
        B, C, x = BCx[:num_actual].chunk(3, dim=-1)
        Bx = B * x
        raw_state = self.kv_cache[0]
        dim_first = is_conv_state_dim_first()
        weight = self.conv.weight
        bias = self.conv.bias
        pieces: list[torch.Tensor] = []

        def gather(indices: torch.Tensor) -> torch.Tensor:
            selected = raw_state.index_select(0, indices)
            if dim_first:
                return selected[..., : self.state_len].contiguous()
            return selected[:, : self.state_len, :].transpose(1, 2).contiguous()

        def scatter(indices: torch.Tensor, values: torch.Tensor) -> None:
            # index_copy_ needs a full source slice, so pad back to the allocated state width.
            if dim_first:
                padded = values.new_zeros(
                    (values.shape[0], values.shape[1], raw_state.shape[-1])
                )
                padded[..., : self.state_len] = values
            else:
                padded = values.new_zeros(
                    (values.shape[0], raw_state.shape[1], values.shape[1])
                )
                padded[:, : self.state_len, :] = values.transpose(1, 2)
            raw_state.index_copy_(0, indices, padded)

        if num_decode:
            indices = md.state_indices_tensor_d.reshape(-1)
            decoded, new_states = ShortConvState.step_batch(
                Bx[:num_decode], weight, gather(indices), bias
            )
            scatter(indices, new_states)
            pieces.append(decoded * C[:num_decode])

        if num_prefill:
            indices = md.state_indices_tensor_p.reshape(-1)
            states = gather(indices)
            has_initial = md.has_initial_states_p
            if has_initial is not None:
                states = states * has_initial.to(states.dtype).view(-1, 1, 1)
            starts = md.query_start_loc_p[: int(md.num_prefills)].to(Bx.device)
            ends = md.query_start_loc_p[1 : int(md.num_prefills) + 1].to(Bx.device)
            prefilled, new_states = ShortConvState.prefill_batch(
                Bx[num_decode:], weight, states, starts, ends, bias
            )
            scatter(indices, new_states)
            pieces.append(prefilled * C[num_decode:])

        y = torch.cat(pieces, dim=0)
        out, _ = self.out_proj(y)
        output[:num_actual] = out

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.SHORT_CONV

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        return MambaStateDtypeCalculator.short_conv_state_dtype(
            self.model_config.dtype,
            self.cache_config.mamba_cache_dtype,
        )

    def get_state_shape(self) -> tuple[tuple[int, ...]]:
        if is_conv_state_dim_first():
            return ((self.conv_dim, self.state_len_padded),)
        return ((self.state_len_padded, self.conv_dim),)


class Lfm3AttentionDecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        layer_idx: int,
        cache_config: CacheConfig | None = None,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        sliding = config.sliding_window if self.layer_type == "sliding_attention" else None
        self.self_attn = Lfm3Attention(
            config,
            layer_idx,
            cache_config=cache_config,
            quant_config=quant_config,
            sliding_window=sliding,
            prefix=f"{prefix}.self_attn",
        )
        self._build_tail(config, quant_config, prefix)

    def _build_tail(self, config, quant_config, prefix: str) -> None:
        self.feed_forward = Lfm3MLP(
            config.hidden_size,
            config.intermediate_size,
            quant_config=quant_config,
            prefix=f"{prefix}.feed_forward",
        )
        self.operator_norm = Lfm3RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ffn_norm = Lfm3RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.post_operator_norm = (
            Lfm3RMSNorm(config.hidden_size, eps=config.norm_eps)
            if config.use_sandwich_norm
            else None
        )
        self.post_ffn_norm = (
            Lfm3RMSNorm(config.hidden_size, eps=config.norm_eps)
            if config.use_sandwich_norm
            else None
        )

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.operator_norm(hidden_states)
        hidden_states = self.self_attn(positions, hidden_states)
        if self.post_operator_norm is not None:
            hidden_states = self.post_operator_norm(hidden_states)
        hidden_states = residual + hidden_states
        ffn = self.feed_forward(self.ffn_norm(hidden_states))
        if self.post_ffn_norm is not None:
            ffn = self.post_ffn_norm(ffn)
        return hidden_states + ffn


class Lfm3ConvDecoderLayer(Lfm3AttentionDecoderLayer):
    def __init__(
        self,
        config,
        layer_idx: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.layer_type = config.layer_types[layer_idx]
        self.short_conv = Lfm3ShortConv(
            config,
            config.hidden_size,
            layer_idx,
            model_config=model_config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.short_conv",
        )
        self._build_tail(config, quant_config, prefix)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.operator_norm(hidden_states)
        output = torch.empty_like(hidden_states)
        self.short_conv(hidden_states, output)
        hidden_states = output
        if self.post_operator_norm is not None:
            hidden_states = self.post_operator_norm(hidden_states)
        hidden_states = residual + hidden_states
        ffn = self.feed_forward(self.ffn_norm(hidden_states))
        if self.post_ffn_norm is not None:
            ffn = self.post_ffn_norm(ffn)
        return hidden_states + ffn


class Lfm3Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size, org_num_embeddings=config.vocab_size
        )

        def get_layer(prefix: str):
            layer_idx = extract_layer_index(prefix)
            if config.layer_types[layer_idx] == "conv":
                return Lfm3ConvDecoderLayer(
                    config,
                    layer_idx,
                    model_config=model_config,
                    cache_config=cache_config,
                    quant_config=quant_config,
                    prefix=prefix,
                )
            return Lfm3AttentionDecoderLayer(
                config,
                layer_idx,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], config.hidden_size
        )
        if get_pp_group().is_last_rank:
            self.embedding_norm = Lfm3RMSNorm(config.hidden_size, eps=config.norm_eps)
        else:
            self.embedding_norm = PPMissingLayer()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if get_pp_group().is_first_rank:
            hidden_states = (
                inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
            )
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states = layer(positions, hidden_states)
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})
        return self.embedding_norm(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".w13", ".w1", 0),
            (".w13", ".w3", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if ".conv." in name:
                name = name.replace(".conv.", ".short_conv.", 1)
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name + "." not in name:
                    continue
                name = name.replace(weight_name + ".", param_name + ".")
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                param.weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class Lfm3ForCausalLM(
    nn.Module, HasInnerState, SupportsLoRA, SupportsPP, IsHybrid, SupportsQuant
):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "w13": ["w1", "w3"],
    }
    hf_to_vllm_mapper = WeightsMapper(orig_to_new_substr={".conv.": ".short_conv."})
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig):
        return MambaStateDtypeCalculator.short_conv_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig):
        hf_config = vllm_config.model_config.hf_config
        dtype_size = torch.tensor(
            [], dtype=vllm_config.model_config.dtype
        ).element_size()
        padded = _padded_state_len(
            hf_config.hidden_size,
            hf_config.conv_L_cache,
            hf_config.head_dim,
            hf_config.num_key_value_heads,
            dtype_size,
        )
        if is_conv_state_dim_first():
            return ((hf_config.hidden_size, padded),)
        return ((padded, hf_config.hidden_size),)

    @classmethod
    def get_mamba_state_copy_func(cls):
        return MambaStateCopyFuncCalculator.short_conv_state_copy_func()

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        if cache_config.mamba_cache_mode == "all":
            raise NotImplementedError(
                "LFM3 does not support 'all' prefix caching; use '--mamba-cache-mode=align'"
            )
        self.config = config
        self.model = Lfm3Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)


__all__ = ["Lfm3ForCausalLM"]
