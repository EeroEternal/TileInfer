"""Qwen3.6-35B-A3B - the first in-service declaration.

Numbers below are *model facts*, copied from the checkpoint's ``config.json`` (40 layers =
30 GDN linear-attention + 10 full-attention, 16 Q / 2 KV heads at head_dim 256, 256 experts
top-8, GDN 16 K / 32 V heads at 128/128, conv width 4, bf16, no quantization).  They are
code, not a data copy: the decision layer must be able to answer "what does this model
need?" before the 67 GB of weights exist.

The benchmark sample was frozen against the vllm-ascend 0.23 baseline on this box
(2026-11-12, ``temperature=0``, ``top_logprobs=5``).  It is the model-local half of C2:
a run that does not reproduce these five numbers has failed, regardless of how readable
its text looks.
"""

from __future__ import annotations

from ..contracts.capabilities import Capability
from ..contracts.manifest import AttentionSpec, LinearSpec, ModelManifest, MoESpec

LAYER_TYPES = tuple(
    "full_attention" if index % 4 == 3 else "linear_attention" for index in range(40)
)

QWEN3_6 = ModelManifest(
    name="Qwen3.6-35B-A3B",
    manifest_id="qwen3_6",
    arch="Qwen3_5MoeForConditionalGeneration",
    model_type="qwen3_5_moe",
    num_layers=40,
    layer_types=LAYER_TYPES,
    attention=AttentionSpec(
        kind="full",
        num_heads=16,
        num_kv_heads=2,
        head_dim=256,
        output_gate=True,
        rope="mrope_interleaved",
        partial_rotary_factor=0.25,
    ),
    linear=LinearSpec(
        kind="gdn",
        num_k_heads=16,
        num_v_heads=32,
        head_k_dim=128,
        head_v_dim=128,
        conv_kernel_dim=4,
        state_dtype="float32",
    ),
    moe=MoESpec(num_experts=256, top_k=8, intermediate_size=512, shared_experts=0),
    quantization="none",
    kv={
        "hybrid": True,
        "full_attention_layers": 10,
        "linear_attention_layers": 30,
        "engine_block_size": 2048,
        "note": "vllm-ascend owns the pool; the page size is a vendor decision, recorded here "
        "so a mismatch is visible before it becomes a slow path",
    },
    required_capabilities=tuple(
        cap.value
        for cap in (
            Capability.ATTENTION_FULL,
            Capability.ATTENTION_LINEAR,
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
        # First generated token and its top-5, frozen from the vllm-ascend baseline.
        "first_token": "Here",
        "top5": [
            ["Here", -0.030266],
            ["Thinking", -3.530266],
            ["The", -8.655266],
            ["Okay", -9.030266],
            ["用户", -10.030266],
        ],
        "recorded_at": "2026-11-12T14:47:00+08:00",
        "recorded_on": "Ascend950PR, CANN 9.3.0, vllm-ascend 0.23.0",
    },
    source="docs/notes/op-coverage-qwen36.md; /home/models/Qwen3.6-35B-A3B/config.json",
)

QWEN3_6.validate()

#: Registry used by ``resolve.decide``; adding a model is one entry here.
MANIFESTS = {QWEN3_6.manifest_id: QWEN3_6, QWEN3_6.arch: QWEN3_6, QWEN3_6.name: QWEN3_6}

__all__ = ["QWEN3_6", "MANIFESTS", "LAYER_TYPES"]
