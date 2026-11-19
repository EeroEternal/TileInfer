"""Nemotron 3.5 Lightning 30B-A3B — declaration from the real checkpoint config.

The architecture fields below come from the checkpoint's own ``config.json`` (snapshot:
``docs/notes/m5-nemotron-config.json``), not from vLLM's ``NemotronHConfig`` defaults.  The
checkpoint's config spells the layer plan as ``layers_block_type``; this declaration keeps
it as layer types so the decision layer can match Mamba2 / MoE / attention without
depending on the vendor's config class.

**DRAFT, NOT REGISTERED.**  The C2 benchmark top-5 is still empty: it needs the HF
reference run on the real weights.  Do not add this module to ``MANIFESTS`` or use it as a
deployable declaration until ``BENCHMARK_PENDING`` is false.
"""

from __future__ import annotations

from ..contracts.capabilities import Capability
from ..contracts.manifest import AttentionSpec, MambaSpec, ModelManifest, MoESpec

BENCHMARK_PENDING = True

LAYER_TYPES = (
    "mamba", "moe", "mamba", "moe",
    "mamba", "full_attention", "moe", "mamba",
    "moe", "mamba", "moe", "mamba",
    "full_attention", "moe", "mamba", "moe",
    "mamba", "moe", "mamba", "full_attention",
    "moe", "mamba", "moe", "mamba",
    "moe", "mamba", "full_attention", "moe",
    "mamba", "moe", "mamba", "moe",
    "mamba", "full_attention", "moe", "mamba",
    "moe", "mamba", "moe", "mamba",
    "moe", "mamba", "full_attention", "moe",
    "mamba", "moe", "mamba", "moe",
    "mamba", "moe", "mamba", "moe",
)

NEMOTRON_3_5_DRAFT = ModelManifest(
    name="Nemotron-3.5-Lightning-30B-A3B",
    manifest_id="nemotron_3_5_lightning",
    arch="NemotronHForCausalLM",
    model_type="nemotron_h",
    num_layers=52,
    layer_types=LAYER_TYPES,
    attention=AttentionSpec(
        kind="gqa",
        num_heads=32,
        num_kv_heads=2,
        head_dim=128,
        rope="rope",
        partial_rotary_factor=1.0,
    ),
    mamba=MambaSpec(
        kind="mamba2",
        num_heads=64,
        head_dim=64,
        state_size=128,
        n_groups=8,
        conv_kernel=4,
        expand=2,
        state_dtype="float32",
    ),
    moe=MoESpec(
        num_experts=128,
        top_k=6,
        intermediate_size=1856,
        shared_experts=1,
        shared_intermediate_size=3712,
    ),
    quantization="none",
    kv={
        "hybrid": True,
        "full_attention_layers": 6,
        "mamba_layers": 23,
        "moe_layers": 23,
        "max_position_embeddings": 262144,
        "mamba_state_dtype": "float32",
        "num_nextn_predict_layers": 1,
        "mtp_weights": "found: 270 mtp.* keys in model.safetensors.index.json; mtp.layers.0/.1 present",
        "config_snapshot": "docs/notes/m5-nemotron-config.json",
        "note": "config uses layers_block_type; vLLM 0.23 needs a config shim; MTP weights present but Ascend spec path unverified",
    },
    required_capabilities=tuple(
        cap.value
        for cap in (
            Capability.ATTENTION_FULL,
            Capability.MOE_ROUTED,
            Capability.NORM_RMS,
            Capability.ROTARY,
            Capability.SAMPLING_TOPK,
            Capability.STATE_LINEAR,
            Capability.QUANT_VERIFY,
        )
    ),
    benchmark={
        "prompt": "The capital of France is",
        "max_tokens": 16,
        "temperature": 0.0,
        "top_logprobs": 5,
        "first_token": "",
        "top5": [],
        "pending": "HF/transformers reference run on the real weights",
    },
    support={
        "text": "partial",
        "vision": "not_applicable",
        "mtp": "failed",
        "quant": ["bf16"],
        # Vendor-path TP=2/4/8 bring-up only: our Mamba2/MoE candidates declare
        # parallel=("single",) and the decision refuses them under sharding until they are
        # validated.  Do not read this as provider TP support.
        "tp": ["1"],
        "tp_note": "vendor path only; candidate providers are single-card and fail closed",
        "max_context_verified": 131072,
    },
    config_impl="tileinfer.bridge.nemotron_config:register_nemotron_h_alias",
    source="docs/notes/m5-nemotron-config.json; docs/notes/m5-nemotron-declaration.md",
)
