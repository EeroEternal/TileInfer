"""LFM3-1.7B: a hybrid convolution / sliding-attention / full-attention model.

Facts are copied from the checkpoint's ``config.json`` (24 layers: 13 gated short
convolutions, 5 sliding-attention layers and 6 full-attention layers; 16 Q / 4 KV heads at
head_dim 128; short-conv window 7 and sliding window 256; two RoPE parameter sets, full
attention theta 1e6 with partial rotary 0.5 and sliding attention theta 1e4).  The model has
no usable upstream vLLM implementation, so the declaration names the route-O model class.
"""

from __future__ import annotations

from ..contracts.capabilities import Capability
from ..contracts.manifest import AttentionSpec, ModelManifest, StateSpec

LAYER_TYPES = tuple(
    "full_attention"
    if index % 4 == 2
    else "sliding_attention"
    if index % 4 == 0 and index > 0
    else "conv"
    for index in range(24)
)

LFM3_1_7 = ModelManifest(
    name="LFM3-1.7B",
    manifest_id="lfm3_1_7",
    arch="Lfm3ForCausalLM",
    model_type="lfm3",
    num_layers=24,
    layer_types=LAYER_TYPES,
    attention=AttentionSpec(
        kind="full",
        num_heads=16,
        num_kv_heads=4,
        head_dim=128,
        output_gate=True,
        rope="per_layer",
        partial_rotary_factor=0.5,
        sliding_window=256,
        rope_by_layer={
            "full_attention": {
                "rope_type": "default",
                "rope_theta": 1000000.0,
                "partial_rotary_factor": 0.5,
            },
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": 10000.0,
            },
        },
    ),
    state=StateSpec(kind="short_conv", dim=2048, kernel=7, bias=False),
    quantization="none",
    kv={
        "hybrid": True,
        "full_attention_layers": 6,
        "sliding_attention_layers": 5,
        "conv_layers": 13,
        "sliding_window": 256,
        "conv_kernel": 7,
        "engine_block_size": 128,
        "max_position_embeddings": 262144,
    },
    required_capabilities=(
        Capability.ATTENTION_FULL.value,
        Capability.ATTENTION_SLIDING.value,
        Capability.STATE_LINEAR.value,
        Capability.NORM_RMS.value,
        Capability.ROTARY.value,
        Capability.SAMPLING_TOPK.value,
        Capability.QUANT_VERIFY.value,
    ),
    benchmark={
        "prompt": "The capital of France is",
        "max_tokens": 16,
        "temperature": 0.0,
        "top5": [
            [" Paris", -0.1842],
            [" the", -2.6842],
            [":\n\n", -4.3717],
            ["Paris", -4.4967],
            [":", -5.4342],
        ],
        "first_token": " Paris",
        "reference": "hf-transformers-5.17-npu",
        "logprob_abs_tol": 0.1,
        "note": (
            "No vendor baseline: vLLM rejects Lfm3ForCausalLM. Reference is HF transformers "
            "on the same NPU; the top-5 set matches, ranks 3/4 may swap inside the tolerance "
            "because Ascend FIA and HF SDPA accumulate in a different order."
        ),
        "perf": {
            "reference": "hf-transformers-5.17-npu (conc=1)",
            "ttft_p50_ms": {"1024": 195.8, "4096": 73.7, "7680": 138.7},
            "tpot_ms": {"1024": 6.29, "4096": 5.27, "7680": 5.30},
            "throughput_tok_s": {
                "1024_conc16": 1293.9,
                "4096_conc16": 610.7,
                "7680_conc16": 364.3,
            },
            "hf_ttft_p50_ms": {"1024": 349.7, "4096": 78.9, "7680": 148.9},
            "hf_tpot_ms": {"1024": 13.77, "4096": 11.69, "7680": 11.73},
            "gate": "conc=1: TTFT <= 1.10x HF, TPOT <= 1.10x HF; conc>1: absolute only",
            "status": "pass_conc1_reference",
        },
    },
    support={
        "text": "verified",
        "vision": "not_applicable",
        "mtp": "not_applicable",
        "quant": ["bf16"],
    },
    hooks=("lfm3_rope_global_buffer",),
    source="docs/notes/lfm3-adaptation.md",
    model_impl="tileinfer.bridge.models.lfm3",
    # Validated single-card only.  LFM3's short convolution is depthwise over the full
    # channel count, so TP would need a per-rank channel shard and a verified state layout;
    # until that exists the decision refuses TP instead of feeding it sharded tensors.
    parallel=("single",),
)
