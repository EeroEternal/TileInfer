"""``tileinfer.vllm.models.muse_glimmer`` — MuseGlimmer text decoder for vLLM 0.23.

Backported from upstream vLLM main (``vllm/model_executor/models/muse_glimmer.py`` at
``ef5f913``, 2026-11-09) because the frozen stack carries **vLLM 0.23.0**, whose registry has
no MuseGlimmer architecture, and its bundled transformers (5.5.4) does not know
``model_type: muse_glimmer``.  This module supplies both halves of the missing glue:

* ``MuseGlimmerHFConfig`` + :func:`register_config` — a ``PretrainedConfig`` shim registered
  under ``muse_glimmer`` so ``AutoConfig.from_pretrained`` can parse the checkpoint at all
  (same trick as ``tileinfer.vllm.config.Dsv41HFConfig``).
* ``MuseGlimmerForConditionalGeneration`` + :func:`register_models` — the architecture,
  registered into ``ModelRegistry`` by class path so the parameter tree is imported only when
  such a checkpoint is actually resolved.

Phase-1 scope: **text-only**.  The upstream file pairs the text decoder with a ViT-G/14 vision
tower, a two-layer adapter and a full multimodal processor (which needs
``transformers.MuseGlimmerProcessor``, absent from transformers 5.5.4).  None of that is
backported here: the class deliberately does **not** inherit ``SupportsMultiModal``, so vLLM
treats the architecture as a text model, never requests a processor, and refuses image inputs.
Phase 2 (vision + DFlash) restores the multimodal path.

Model math (all deltas vs Gemma2 handled explicitly, mirroring upstream):

* SiLU-gated MLP (``hidden_activation="silu"``).
* Scaleless RMSNorm on token embeddings (no ``sqrt(hidden)`` multiplier).
* Per-layer sandwich RMSNorms with baked ``+1`` offset (Gemma style) and distinct pre/post eps.
* QK-norm: weightless, fp32, per head over ``head_dim``, applied *before* RoPE; the query is
  additionally pre-scaled by ``qk_scale_factor``.
* iRoPE: ``layer_rope_theta`` of 0 marks NoPE + full attention; non-zero marks RoPE +
  sliding-window attention (``per_layer_sliding_window``).
* Logits pre-scaled by ``output_multiplier`` then tanh soft-capped at
  ``final_logit_softcapping``.
* Untied lm_head.

0.23 API deltas applied during the port (checked against
``.venv_native213`` vLLM 0.23.0, not assumed): everything the text path imports exists at the
same name/signature upstream — including ``per_layer_sliding_window`` /
``logits_soft_cap`` on ``Attention``, ``has_weight``/``dtype`` on ``RMSNorm``,
``rope_parameters`` on ``get_rope``, ``EagleModelMixin._maybe_add_hidden_state`` and
``AutoWeightsLoader.load_weights(..., mapper=...)``.  The only dropped import is
``vllm.config.multimodal.MultiModalDummyOptions`` (multimodal-only; 0.23 spells the type
``MMDummyOptions``), which this module does not need.
"""

from itertools import islice
from typing import Any, ClassVar, Dict, Literal

import torch
from torch import nn
from transformers import PretrainedConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (
    divide,
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces import (
    EagleModelMixin,
    SupportsLoRA,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    WeightsMapper,
    default_weight_loader,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors

logger = init_logger(__name__)

MODEL_TYPE = "muse_glimmer"
MODEL_ARCH = "MuseGlimmerForConditionalGeneration"
__all__ = ["MuseGlimmerHFConfig", "MODEL_ARCH", "MODEL_TYPE", "register_models"]


def _text_config(config):
    """MuseGlimmer checkpoints may nest the text config under ``text_config``
    (multimodal ``MuseGlimmerConfig``) or expose it directly
    (``MuseGlimmerTextConfig``)."""
    return getattr(config, "text_config", config)


# =============================================================================================
# Config shim: teach transformers.AutoConfig the checkpoint's model_type
# =============================================================================================


class MuseGlimmerHFConfig(PretrainedConfig):
    """Carries the checkpoint's keys verbatim, turning the nested blocks into configs.

    ``config.json`` declares ``model_type: muse_glimmer`` with ``text_config`` and
    ``vision_config`` blocks plus ``image_token_id`` / ``video_token_id``.  ``PretrainedConfig``
    would store the blocks as plain dicts and vLLM's ``get_text_config()`` then fails with
    "does not have `num_attention_heads`" — so the sub-blocks are promoted to config objects,
    which is what transformers does for real checkpoints.
    """

    model_type = MODEL_TYPE

    def __init__(self, **kwargs: Any) -> None:
        # Phase-1 scope: only ``text_config`` is promoted.  The vision block is left as a
        # raw dict because transformers 5.5.4's strict dataclass validation rejects its
        # layer_types (``window_attention``, a Muse-only value) while building a plain
        # ``PretrainedConfig``; the text-only model never reads ``vision_config``.
        # Phase 2 (vision) must provide a proper vision config class instead.
        sub = kwargs.get("text_config")
        if isinstance(sub, dict):
            kwargs["text_config"] = PretrainedConfig(**sub)
        kwargs.setdefault("architectures", [MODEL_ARCH])
        super().__init__(**kwargs)


def register_config() -> Dict[str, str]:
    """Idempotently teach ``AutoConfig`` the ``muse_glimmer`` name."""
    from transformers import AutoConfig

    existing = getattr(AutoConfig, "_mapping", {}).get(MODEL_TYPE)
    if existing is MuseGlimmerHFConfig:
        return {"config": "already registered", "model_type": MODEL_TYPE}
    AutoConfig.register(MODEL_TYPE, MuseGlimmerHFConfig, exist_ok=True)
    return {"config": "registered", "model_type": MODEL_TYPE}


# =============================================================================================
# Text decoder
# =============================================================================================


class MuseGlimmerMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_activation: str,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_activation != "silu":
            raise ValueError(
                f"MuseGlimmer uses `silu` as the hidden activation; "
                f"got `{hidden_activation}`."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class MuseGlimmerAttention(nn.Module):
    def __init__(
        self,
        config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = extract_layer_index(prefix)

        tp_size = get_tensor_model_parallel_world_size()
        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        # MuseGlimmer overrides Gemma2's query_pre_attn_scalar scaling with the
        # standard 1/sqrt(head_dim). The query is *additionally* pre-scaled by
        # scale_query_by after QK-norm (see forward).
        self.scaling = self.head_dim**-0.5

        # iRoPE: layers with a zero RoPE theta are NoPE
        self.use_rope = bool(config.layer_rope_theta[self.layer_idx])

        # Weightless, computed in fp32, applied per head over head_dim.
        self.qk_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, has_weight=False)
        self.scale_query_by = config.qk_scale_factor

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.output_gate_proj = ColumnParallelLinear(
            self.hidden_size,
            self.total_num_heads * self.head_dim,
            bias=False,
            gather_output=False,
            quant_config=quant_config,
            prefix=f"{prefix}.output_gate_proj",
        )

        self.rotary_emb = (
            get_rope(
                self.head_dim,
                max_position=config.max_position_embeddings,
                rope_parameters=config.rope_parameters,
                # HF converter permutes q/k to NEOX (half-split) layout via
                # _permute_for_rope
                is_neox_style=True,
            )
            if self.use_rope
            else None
        )

        is_sliding = config.layer_types[self.layer_idx] == "sliding_attention"
        sliding_window = config.sliding_window if is_sliding else None
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            logits_soft_cap=None,  # MuseGlimmer sets attn_logit_softcapping = None
            per_layer_sliding_window=sliding_window,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # QK-norm over head_dim, fp32, applied BEFORE RoPE; then pre-scale q.
        q = q.reshape(-1, self.head_dim)
        q = self.qk_norm(q).reshape(-1, self.q_size) * self.scale_query_by
        k = k.reshape(-1, self.head_dim)
        k = self.qk_norm(k).reshape(-1, self.kv_size)
        q = q.to(v.dtype)
        k = k.to(v.dtype)

        if self.rotary_emb is not None:
            q, k = self.rotary_emb(positions, q, k)

        # NOTE (port workaround, not upstream): the fused-qkv ``split`` above leaves ``v`` a
        # strided view for multi-token prefill (single-token decode is contiguous).  vllm-ascend
        # 0.23.0's sliding-window branch calls CANN's FIA without making the inputs contiguous,
        # and the operator's non-PA checker rejects a strided ``value`` (ACLNN tiling error
        # 561002, ``paged_attention_checker.cpp: "In non-PA scenarios, value tensors must be
        # contiguous"``).  Upstream fixed the vendor side in cfb01b1c8 (#17925, first in
        # v0.30.0); this copy is the model-side equivalent for pinned stacks and is harmless on
        # the v2 native backend.
        v = v.contiguous()

        attn_output = self.attn(q, k, v)

        # Gate reads the layer input hidden states (not the attn output).
        gate, _ = self.output_gate_proj(hidden_states)
        attn_output = torch.sigmoid(gate) * attn_output

        output, _ = self.o_proj(attn_output)
        return output


class MuseGlimmerDecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.self_attn = MuseGlimmerAttention(
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = MuseGlimmerMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_activation=config.hidden_activation,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        # Sandwich norms with baked +1 offset. Pre-norms use rms_norm_eps; the
        # post-norms use the (typically smaller) post_norm_eps.
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.post_norm_eps
        )
        self.pre_feedforward_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.post_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Gemma2-style sandwich, replicated explicitly (matches HF MuseGlimmer).
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, residual


@support_torch_compile
class MuseGlimmerModel(nn.Module, EagleModelMixin):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = _text_config(vllm_config.model_config.hf_config)
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        # MuseGlimmer normalizes token embeddings with a weightless RMSNorm
        # instead of Gemma's sqrt(hidden_size) multiplier.
        self.embed_norm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, has_weight=False
        )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: MuseGlimmerDecoderLayer(
                config, cache_config, quant_config, prefix=prefix
            ),
            prefix=f"{prefix}.layers",
        )
        # Final norm: weight-as-scale, no offset.
        self.norm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dtype=torch.float32
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_norm(self.embed_tokens(input_ids))

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        aux_hidden_states = self._maybe_add_hidden_state(
            [], self.start_layer, hidden_states, None
        )
        for layer_idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            hidden_states, residual = layer(positions, hidden_states, residual)
            self._maybe_add_hidden_state(
                aux_hidden_states, layer_idx + 1, hidden_states, None
            )
        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states = self.norm(hidden_states)
        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states


# =============================================================================================
# The conditional-generation wrapper
# =============================================================================================


class MuseGlimmerForConditionalGeneration(nn.Module, SupportsLoRA, SupportsPP):
    # vLLM 0.23 classifies architectures through ``VllmModelForTextGeneration``
    # (interfaces_base.py): the ``is_text_generation_model`` flag plus ``embed_input_ids`` /
    # ``forward`` / ``compute_logits`` decide whether ``--runner generate`` is accepted.
    is_text_generation_model: ClassVar[Literal[True]] = True

    # Weight-name normalization. Two checkpoint conventions are supported:
    #
    #   * HF MuseGlimmer export (``convert_muse_glimmer_weights_to_hf.py``): the
    #     multimodal ``MuseGlimmerConfig`` prefixes the language model with
    #     ``model.language_model.``; the per-layer sandwich norms are already
    #     named ``input_layernorm`` / ``post_attention_layernorm`` /
    #     ``pre_feedforward_layernorm`` / ``post_feedforward_layernorm``.
    #
    #   * Legacy HF export (an earlier checkpoint convention): uses ``model.``
    #     and a different sandwich-norm naming where ``post_attn_norm`` is the
    #     true post-attention norm and ``post_attention_layernorm`` is actually
    #     the pre-feedforward norm. We remap those to MuseGlimmer's names.
    #
    # The unambiguous discriminator is the PREFIX: legacy keys start with
    # ``model.layers.`` while canonical keys start with
    # ``model.language_model.layers.``. ``orig_to_new_regex`` runs BEFORE the
    # prefix strip (see WeightsMapper._map_name_with_shard), so we anchor the
    # legacy renames on ``^model\.layers\.`` — they fire ONLY on legacy keys and
    # leave canonical/partner checkpoints untouched. Rule order within the regex
    # dict matters: the ``post_attention_layernorm`` -> ``pre_feedforward_...``
    # rule must precede the ``post_attn_norm`` -> ``post_attention_layernorm``
    # rule so the latter's output is not re-captured by the former (regex rules
    # apply as a single forward pass).
    #
    # The vision-tower rules are inert for the text-only phase (no such keys hit
    # the model tree) and are kept verbatim so phase 2 does not re-derive them.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={
            "model.vision_tower.patch_embedder.position_embedding_table.weight": (
                "model.vision_tower.positional_embedding_vlm"
            ),
            "model.vision_tower.layers.": "model.vision_tower.transformer.",
            ".norm1.": ".ln_1.",
            ".norm2.": ".ln_2.",
            ".attn.proj.": ".attn.o_proj.",
            ".mlp.fc1.": ".mlp.c_fc.",
            ".mlp.fc2.": ".mlp.c_proj.",
            ".self_attn.gate_proj": ".self_attn.output_gate_proj",
        },
        orig_to_new_prefix={
            "model.rotary_emb.": None,
            "model.language_model.": "model.",
            "language_model.": "model.",
            "model.vision_tower.patch_embedder.patch_embedding.": (
                "model.vision_tower.conv1_linear."
            ),
            "model.vision_tower.": "vision_encoder.",
            "vision_tower.": "vision_encoder.",
            "model.vision_encoder.": "vision_encoder.",
            "model.vision_adapter.fc1.": "model.vision_adapter.c_fc.",
            "model.vision_adapter.fc2.": "model.vision_adapter.c_proj.",
            "model.vision_adapter.": "vision_adapter.",
            "model.vision_projection.": "vision_projection.",
            "model.perception_emb_norm.": "perception_emb_norm.",
        },
    )
    # 0.23 has no ``WeightsMapper.orig_to_new_stacked`` (a main-only addition), so the
    # q/k/v and gate/up shard folding happens here, classic-loader style.
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    # Phase-1 (text-only): checkpoint keys under these markers belong to the vision tower /
    # adapter / projection, which this build does not instantiate.  Skipped instead of
    # raising; phase 2 restores the tower and drops this list.
    _VISION_KEY_MARKERS = (
        "vision_tower",
        "vision_encoder",
        "vision_adapter",
        "vision_projection",
        "perception_emb_norm",
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        text_config = _text_config(config)
        quant_config = vllm_config.quant_config
        self.config = config
        self.text_config = text_config
        self.quant_config = quant_config

        # Phase-1 note: upstream wraps this in ``self._mark_language_model(...)`` from
        # ``SupportsMultiModal``; that interface is deliberately not inherited here, so the
        # marking (an mm-encoder-only bookkeeping concern) is simply absent.
        self.model = MuseGlimmerModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )

        self.lm_head = ParallelLMHead(
            text_config.vocab_size,
            text_config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.output_multiplier = text_config.output_multiplier
        self.final_logit_softcapping = text_config.final_logit_softcapping
        self.logits_processor = LogitsProcessor(text_config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        if logits is None:
            return None
        logits = logits * self.output_multiplier
        if self.final_logit_softcapping is not None:
            cap = self.final_logit_softcapping
            logits = cap * torch.tanh(logits / cap)
        return logits

    def load_weights(self, weights) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        vision_seen = False
        for name, loaded_weight in self.hf_to_vllm_mapper.apply(weights):
            if any(marker in name for marker in self._VISION_KEY_MARKERS):
                if not vision_seen:
                    logger.info(
                        "MuseGlimmer text-only build: skipping vision weights "
                        "(first key %r; further ones not logged)",
                        name,
                    )
                    vision_seen = True
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)
        return loaded_params


# =============================================================================================
# Registration
# =============================================================================================


def register_models(registry=None) -> Dict[str, object]:
    """Teach transformers the ``muse_glimmer`` config, then register the architecture.

    The config half must run first: vLLM reads ``config.json`` through
    ``transformers.AutoConfig`` before it looks at architectures, and the checkpoint's
    ``model_type: muse_glimmer`` is unknown to released transformers.  Registration is by
    class path string, so importing this module's parameter tree waits until a checkpoint
    with that architecture is actually resolved.

    Failure-tolerant like every other plugin step: a missing vLLM registry or a duplicate
    registration is a logged detail, never a boot failure.
    """
    info: Dict[str, object] = {
        "arch": MODEL_ARCH,
        "class_path": "tileinfer.vllm.models.muse_glimmer:MuseGlimmerForConditionalGeneration",
        "registered": False,
        "detail": "",
    }
    details = []
    try:
        details.append(str(register_config()))
    except Exception as exc:  # noqa: BLE001 - a plugin step must never abort the boot
        details.append(f"config registration raised {exc!r}")
    try:
        if registry is None:
            from vllm.model_executor.models.registry import ModelRegistry

            registry = ModelRegistry
        registry.register_model(
            MODEL_ARCH,
            "tileinfer.vllm.models.muse_glimmer:MuseGlimmerForConditionalGeneration",
        )
        info["registered"] = True
        details.append(f"registered {MODEL_ARCH} -> tileinfer.vllm.models.muse_glimmer")
    except Exception as exc:  # noqa: BLE001 - a plugin step must never abort the boot
        details.append(f"registration failed: {exc!r}")
    info["detail"] = "; ".join(details)
    return info
