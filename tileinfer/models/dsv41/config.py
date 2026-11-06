"""``tileinfer.models.dsv41.config`` — every shape decision of DeepSeek-V4.1-Flash in one record.

Two spellings of the same config exist on disk and both are accepted:

* ``/home/models/DeepSeek-V4.1-Flash/config.json`` — HF style, model facts under ``text_config``
  (``hidden_size``, ``qk_rope_head_dim``, ``kv_source_layer_ids``, ...), plus ``vision_config`` and
  ``quantization_config``.
* ``.../inference/config.json`` — the reference runtime's flat ``ModelArgs`` spelling
  (``dim``, ``rope_head_dim``, ``kv_source_layers``, ...).

:func:`Dsv41Config.from_dict` normalises both to one set of attribute names; nothing downstream
reads the raw dicts.  All facts and their provenance: ``docs/notes/dsv41-facts.md`` §0–§14.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = ["Dsv41Config", "FP8_BLOCK", "FP4_BLOCK"]


#: One fp8 scale per 32x32 weight block / 32 activations (weight_block_size [32, 32]).
FP8_BLOCK = 32
#: One fp4 scale per 32 elements along K.
FP4_BLOCK = 32


def _pick(src: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    """First present, non-``None`` value among ``names`` (HF and flat spellings interleaved)."""
    for name in names:
        if name in src and src[name] is not None:
            return src[name]
    return default


@dataclass(frozen=True)
class Dsv41Config:
    """DeepSeek-V4.1-Flash text + vision shape facts.

    Everything is derived from the checkpoint's own ``config.json``; the defaults are the released
    values so a test can build a *small* model by overriding the few fields it needs
    (see ``tests/test_dsv41_tree.py``).
    """

    # --- text backbone -----------------------------------------------------------------------
    vocab_size: int = 129280
    hidden_size: int = 5120
    moe_intermediate_size: int = 2304
    num_hidden_layers: int = 40
    num_attention_heads: int = 64
    head_dim: int = 512
    qk_rope_head_dim: int = 64
    q_lora_rank: int = 1280
    o_lora_rank: int = 1024
    o_groups: int = 8
    rms_norm_eps: float = 1e-20
    sliding_window: int = 128
    # --- MoE ---------------------------------------------------------------------------------
    n_routed_experts: int = 384
    n_shared_experts: int = 1
    num_experts_per_tok: int = 6
    scoring_func: str = "sqrtsoftplus"
    routed_scaling_factor: float = 1.5
    swiglu_limit: float = 10.0
    # --- sparse compressed attention ---------------------------------------------------------
    compress_ratios: Tuple[int, ...] = ()
    kv_source_layers: Tuple[int, ...] = (2, 8, 14, 20)
    index_source_layers: Tuple[int, ...] = (2, 8, 14, 20, 24, 28, 32, 36)
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 512
    candidate_source_layer: int = 20
    candidate_topk_blocks: int = 2048
    candidate_block_size: int = 8
    compress_rope_theta: float = 160000.0
    # --- rope ---------------------------------------------------------------------------------
    rope_theta: float = 10000.0
    rope_factor: float = 16.0
    beta_fast: int = 32
    beta_slow: int = 1
    original_seq_len: int = 65536
    max_position_embeddings: int = 1048576
    # --- Hyper-Connections -------------------------------------------------------------------
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    # --- Engram ------------------------------------------------------------------------------
    engram_layer_ids: Tuple[int, ...] = (1, 14)
    engram_num_embeddings: Tuple[int, ...] = (384006168, 384016682)
    engram_max_ngram_size: int = 4
    engram_vocab_size: int = 16000000
    engram_n_heads: int = 8
    engram_head_dim: int = 256
    engram_pad_token_id: int = 2
    engram_compressed_vocab_size: int = 99092
    # --- DSpark draft head -------------------------------------------------------------------
    num_nextn_predict_layers: int = 3
    dspark_block_size: int = 5
    dspark_noise_token_id: int = 128799
    dspark_target_layer_ids: Tuple[int, ...] = (37, 38, 39)
    dspark_markov_rank: int = 256
    dspark_n_routed_experts: int = 128
    dspark_n_activated_experts: int = 3
    # --- vision ------------------------------------------------------------------------------
    vision_num_hidden_layers: int = 32
    vision_hidden_size: int = 1024
    vision_num_attention_heads: int = 16
    vision_intermediate_size: int = 2816
    vision_patch_size: int = 14
    vision_downsample_ratio: int = 3
    vision_rope_theta: float = 10000.0
    image_token_id: int = 129264
    # --- quantization ------------------------------------------------------------------------
    expert_dtype: str = "fp4"
    weight_dtype: str = "fp8"
    # --- runtime (not checkpoint shape) ------------------------------------------------------
    max_seq_len: int = 4096
    max_batch_size: int = 1

    # =========================================================================================
    # Derived facts
    # =========================================================================================

    @property
    def n_mtp_layers(self) -> int:
        return int(self.num_nextn_predict_layers)

    @property
    def rope_head_dim(self) -> int:
        return int(self.qk_rope_head_dim)

    @property
    def nope_head_dim(self) -> int:
        return int(self.head_dim) - int(self.qk_rope_head_dim)

    @property
    def mix_hc(self) -> int:
        """Rows of one ``hc_*_fn`` matrix: ``(2 + hc_mult) * hc_mult``."""
        return (2 + int(self.hc_mult)) * int(self.hc_mult)

    @property
    def hc_dim(self) -> int:
        """Columns of one ``hc_*_fn`` matrix: ``hc_mult * hidden_size``."""
        return int(self.hc_mult) * int(self.hidden_size)

    @property
    def n_hash_cols(self) -> int:
        """Engram rows fetched per token per layer: ``(max_ngram_size - 1) * n_heads``."""
        return (int(self.engram_max_ngram_size) - 1) * int(self.engram_n_heads)

    @property
    def engram_wkv_in(self) -> int:
        return self.n_hash_cols * int(self.engram_head_dim)

    @property
    def engram_wkv_out(self) -> int:
        return int(self.hidden_size) * (int(self.hc_mult) + 1)

    def compress_ratio(self, layer_id: int) -> int:
        """``0`` = sliding window only, ``1`` = 1:1 compressed stream, ``r`` = pool r tokens."""
        ratios = tuple(int(r) for r in self.compress_ratios)
        if not ratios:
            return 0
        if layer_id >= len(ratios):
            raise IndexError(f"layer_id {layer_id} has no compress ratio (len={len(ratios)})")
        return ratios[layer_id]

    @property
    def n_layers(self) -> int:
        """Backbone layers (DSpark stages are ``n_layers .. n_layers+n_mtp_layers``)."""
        return int(self.num_hidden_layers)

    def is_kv_source(self, layer_id: int) -> bool:
        return layer_id < self.n_layers and layer_id in tuple(int(x) for x in self.kv_source_layers)

    def is_index_source(self, layer_id: int) -> bool:
        return layer_id < self.n_layers and layer_id in tuple(int(x) for x in self.index_source_layers)

    def owns_index_k(self, layer_id: int) -> bool:
        """Only a layer that compresses its own KV can publish index keys (facts §7.1)."""
        return self.is_kv_source(layer_id)

    def is_candidate_source(self, layer_id: int) -> bool:
        return layer_id == int(self.candidate_source_layer)

    def uses_candidates(self, layer_id: int) -> bool:
        return 0 <= int(self.candidate_source_layer) < layer_id

    def has_engram(self, layer_id: int) -> bool:
        return layer_id in tuple(int(x) for x in self.engram_layer_ids)

    def engram_rows(self, layer_id: int) -> int:
        """Checkpoint rows of this layer's engram table (sum of its 24 prime buckets)."""
        ids = tuple(int(x) for x in self.engram_layer_ids)
        return int(self.engram_num_embeddings[ids.index(layer_id)])

    def moe_config(self, layer_id: int) -> Tuple[int, int]:
        """``(n_routed_experts, n_activated_experts)``; DSpark stages use their own counts."""
        if layer_id < self.n_layers:
            return int(self.n_routed_experts), int(self.num_experts_per_tok)
        return (
            int(self.dspark_n_routed_experts) or int(self.n_routed_experts),
            int(self.dspark_n_activated_experts) or int(self.num_experts_per_tok),
        )

    @property
    def full_compress_ratios(self) -> Tuple[int, ...]:
        """One ratio per layer including DSpark stages (they are pure sliding window)."""
        ratios = tuple(int(r) for r in self.compress_ratios)
        if len(ratios) >= self.n_layers + self.n_mtp_layers:
            return ratios
        return ratios + (0,) * (self.n_layers + self.n_mtp_layers - len(ratios))

    # =========================================================================================
    # Construction
    # =========================================================================================

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Dsv41Config":
        """Normalise a HF-style *or* flat ``config.json`` dict into this record."""
        raw = dict(raw or {})
        text: Mapping[str, Any] = raw
        nested = _pick(raw, "text_config")
        if isinstance(nested, Mapping):
            text = nested
            raw_vision = _pick(raw, "vision_config", default={}) or {}
            quant = _pick(raw, "quantization_config", default={}) or {}
        else:
            raw_vision = raw
            quant = raw

        def t(*names: str, default: Any = None) -> Any:
            return _pick(text, *names, default=default)

        def v(*names: str, default: Any = None) -> Any:
            return _pick(raw_vision, *names, default=default)

        rope_scaling = t("rope_scaling", default={}) or {}
        if not isinstance(rope_scaling, Mapping):
            rope_scaling = {}

        ratios = tuple(int(x) for x in (t("compress_ratios", default=()) or ()))
        beta_fast = t("beta_fast", default=rope_scaling.get("beta_fast", 32))
        beta_slow = t("beta_slow", default=rope_scaling.get("beta_slow", 1))
        orig = t(
            "original_seq_len",
            "original_max_position_embeddings",
            default=rope_scaling.get("original_max_position_embeddings", 65536),
        )
        factor = t("rope_factor", default=rope_scaling.get("factor", 16.0))

        expert_dtype = _pick(quant, "expert_dtype", default=quant.get("expert_dtype", "fp4"))
        weight_dtype = t("dtype", default=_pick(raw, "dtype", default="fp8"))
        if str(weight_dtype) not in ("fp8", "bf16"):
            weight_dtype = "fp8"

        return cls(
            vocab_size=int(t("vocab_size", default=129280)),
            hidden_size=int(t("hidden_size", "dim", default=5120)),
            moe_intermediate_size=int(t("moe_intermediate_size", "moe_inter_dim", default=2304)),
            num_hidden_layers=int(t("num_hidden_layers", "n_layers", default=40)),
            num_attention_heads=int(t("num_attention_heads", "n_heads", default=64)),
            head_dim=int(t("head_dim", default=512)),
            qk_rope_head_dim=int(t("qk_rope_head_dim", "rope_head_dim", default=64)),
            q_lora_rank=int(t("q_lora_rank", default=1280)),
            o_lora_rank=int(t("o_lora_rank", default=1024)),
            o_groups=int(t("o_groups", default=8)),
            rms_norm_eps=float(t("rms_norm_eps", "norm_eps", default=1e-20)),
            sliding_window=int(t("sliding_window", "window_size", default=128)),
            n_routed_experts=int(t("n_routed_experts", default=384)),
            n_shared_experts=int(t("n_shared_experts", default=1)),
            num_experts_per_tok=int(t("num_experts_per_tok", "n_activated_experts", default=6)),
            scoring_func=str(t("scoring_func", "score_func", default="sqrtsoftplus")),
            routed_scaling_factor=float(t("routed_scaling_factor", "route_scale", default=1.5)),
            swiglu_limit=float(t("swiglu_limit", default=10.0)),
            compress_ratios=ratios,
            kv_source_layers=tuple(
                int(x) for x in t("kv_source_layer_ids", "kv_source_layers", default=(2, 8, 14, 20))
            ),
            index_source_layers=tuple(
                int(x)
                for x in t(
                    "index_source_layer_ids",
                    "index_source_layers",
                    default=(2, 8, 14, 20, 24, 28, 32, 36),
                )
            ),
            index_n_heads=int(t("index_n_heads", default=32)),
            index_head_dim=int(t("index_head_dim", default=128)),
            index_topk=int(t("index_topk", default=512)),
            candidate_source_layer=int(
                t("candidate_source_layer_id", "candidate_source_layer", default=20)
            ),
            candidate_topk_blocks=int(t("candidate_topk_blocks", default=2048)),
            candidate_block_size=int(t("candidate_block_size", default=8)),
            compress_rope_theta=float(t("compress_rope_theta", default=160000.0)),
            rope_theta=float(t("rope_theta", default=10000.0)),
            rope_factor=float(factor),
            beta_fast=int(beta_fast),
            beta_slow=int(beta_slow),
            original_seq_len=int(orig),
            max_position_embeddings=int(t("max_position_embeddings", default=1048576)),
            hc_mult=int(t("hc_mult", default=4)),
            hc_sinkhorn_iters=int(t("hc_sinkhorn_iters", default=20)),
            hc_eps=float(t("hc_eps", default=1e-6)),
            engram_layer_ids=tuple(int(x) for x in t("engram_layer_ids", default=(1, 14))),
            engram_num_embeddings=tuple(int(x) for x in t("engram_num_embeddings", default=())),
            engram_max_ngram_size=int(t("engram_max_ngram_size", default=4)),
            engram_vocab_size=int(t("engram_vocab_size", default=16000000)),
            engram_n_heads=int(t("engram_n_heads", default=8)),
            engram_head_dim=int(t("engram_head_dim", default=256)),
            engram_pad_token_id=int(t("engram_pad_token_id", "engram_pad_id", default=2)),
            engram_compressed_vocab_size=int(t("engram_compressed_vocab_size", default=99092)),
            num_nextn_predict_layers=int(t("num_nextn_predict_layers", "n_mtp_layers", default=3)),
            dspark_block_size=int(t("dspark_block_size", default=5)),
            dspark_noise_token_id=int(t("dspark_noise_token_id", default=128799)),
            dspark_target_layer_ids=tuple(
                int(x) for x in t("dspark_target_layer_ids", default=(37, 38, 39))
            ),
            dspark_markov_rank=int(t("dspark_markov_rank", default=256)),
            dspark_n_routed_experts=int(t("dspark_n_routed_experts", default=128)),
            dspark_n_activated_experts=int(t("dspark_n_activated_experts", default=3)),
            vision_num_hidden_layers=int(v("num_hidden_layers", "vision_n_layers", default=32)),
            vision_hidden_size=int(v("hidden_size", "vision_dim", default=1024)),
            vision_num_attention_heads=int(v("num_attention_heads", "vision_n_heads", default=16)),
            vision_intermediate_size=int(v("intermediate_size", "vision_inter_dim", default=2816)),
            vision_patch_size=int(v("patch_size", "vision_patch_size", default=14)),
            vision_downsample_ratio=int(v("downsample_ratio", "vision_downsample_ratio", default=3)),
            vision_rope_theta=float(v("rope_theta", "vision_rope_theta", default=10000.0)),
            image_token_id=int(_pick(raw, "image_token_id", default=129264)),
            expert_dtype=str(expert_dtype),
            weight_dtype=str(weight_dtype),
        )

    @classmethod
    def from_json(cls, path: str) -> "Dsv41Config":
        import json

        with open(path) as handle:
            return cls.from_dict(json.load(handle))

    @classmethod
    def from_hf(cls, hf_config: Any) -> "Dsv41Config":
        """Build from a ``transformers`` config object *or* a plain dict.

        vLLM hands us ``model_config.hf_config``; transformers has no ``deepseek_v41`` class, so
        the object is a generic ``PretrainedConfig`` whose nested ``text_config`` may be a dict.
        """
        if isinstance(hf_config, Mapping):
            return cls.from_dict(hf_config)
        text = getattr(hf_config, "text_config", None)
        raw: Dict[str, Any] = {}
        if isinstance(text, Mapping):
            raw["text_config"] = dict(text)
        elif text is not None and hasattr(text, "to_dict"):
            raw["text_config"] = text.to_dict()
        for key in ("image_token_id", "quantization_config", "dtype"):
            value = getattr(hf_config, key, None)
            if value is not None:
                raw[key] = value
        vision = getattr(hf_config, "vision_config", None)
        if isinstance(vision, Mapping):
            raw["vision_config"] = dict(vision)
        elif vision is not None and hasattr(vision, "to_dict"):
            raw["vision_config"] = vision.to_dict()
        cfg = cls.from_dict(raw)
        if not cfg.compress_ratios:
            # a generic config object that kept everything flat: fall back to the attributes
            ratios = getattr(hf_config, "compress_ratios", None)
            if ratios:
                cfg = cls.from_dict({**raw, "compress_ratios": list(ratios)})
        return cfg

    # =========================================================================================
    # Sanity
    # =========================================================================================

    def validate(self) -> List[str]:
        """Structural checks that must hold for the released checkpoint; returns warnings.

        The compressed stream is shared by depth: a layer with ``ratio > 0`` that is not itself a kv
        source reads the stream published by the *last* source at or before it, and that source must
        carry the same ratio (facts §6.3).  That is the invariant checked here — "a layer with a
        ratio must be a source" would be wrong, 35 layers legitimately are not.
        """
        problems: List[str] = []
        total_layers = self.n_layers + self.n_mtp_layers
        ratios = self.full_compress_ratios
        if len(ratios) < total_layers:
            problems.append("compress_ratios is shorter than n_layers + n_mtp_layers")
            return problems
        sources = tuple(int(x) for x in self.kv_source_layers)
        for lid in range(self.n_layers):
            ratio = int(ratios[lid])
            if ratio == 0:
                continue
            preceding = [s for s in sources if s <= lid]
            if not preceding:
                problems.append(f"layer {lid} has ratio {ratio} but no kv source precedes it")
                continue
            owner = max(preceding)
            owner_ratio = int(ratios[owner])
            if owner_ratio != ratio:
                problems.append(
                    f"layer {lid} (ratio {ratio}) would read layer {owner}'s stream "
                    f"(ratio {owner_ratio}); the shared stream must keep one ratio per segment"
                )
        for lid in range(self.n_layers, total_layers):
            if int(ratios[lid]) != 0:
                problems.append(f"DSpark stage {lid} must not compress (ratio != 0)")
        if len(self.engram_layer_ids) != len(self.engram_num_embeddings):
            problems.append("engram_layer_ids and engram_num_embeddings disagree in length")
        if self.hidden_size % self.o_groups:
            problems.append("hidden_size is not a multiple of o_groups")
        if (self.num_attention_heads * self.head_dim) % self.o_groups:
            problems.append("n_heads*head_dim is not a multiple of o_groups")
        if self.engram_max_ngram_size < 2:
            problems.append("engram_max_ngram_size must be >= 2 (1-gram lookup has no hash)")
        if self.index_topk <= 0:
            problems.append("index_topk must be positive")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for name in self.__dataclass_fields__:  # type: ignore[attr-defined]
            out[name] = getattr(self, name)
        return out
