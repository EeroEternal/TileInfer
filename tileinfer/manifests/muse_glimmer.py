"""MuseGlimmer-30B — the first *vendor-absent* declaration.

The checkpoint is a 52-layer dense decoder with iRoPE attention: layers 3, 7, 11, ... 51
(every fourth) attend to the whole history and carry no rotary embedding; the other 39 layers
slide over a 2048-token window and use RoPE.  Layer 52 has an output gate on the attention
result, QK-norm over head_dim, and an untied lm_head.

Why this manifest exists: vLLM 0.23 has no architecture for the model and vllm-ascend has no
adaptation, so the vendor cannot serve it.  The two things the vendor lacks are declared here
as *our* obligation (``model_impl``) plus the parser for the checkpoint's private ATEM channel
format (``reasoning_parser_impl``); everything else — attention (including the window), KV
management, worker, graphs — stays with the vendor.

Fact source: ``/home/models/Muse-Glimmer-30B/config.json`` (text_config) and the weight index;
verified on the device (see ``docs/notes/muse-glimmer-static.md``).  The C2 sample is not
captured yet, so this declaration drives deployment but not numeric acceptance.
"""

from __future__ import annotations

from ..contracts.capabilities import Capability
from ..contracts.manifest import AttentionSpec, ModelManifest

#: Layers 3, 7, 11, ... 51 are full attention (NoPE); everything else slides.
FULL_ATTENTION_LAYERS = tuple(range(3, 52, 4))
SLIDING_LAYERS = tuple(i for i in range(52) if i not in FULL_ATTENTION_LAYERS)
LAYER_TYPES = tuple(
    "full_attention" if i in FULL_ATTENTION_LAYERS else "sliding_attention" for i in range(52)
)

MUSE_GLIMMER = ModelManifest(
    name="MuseGlimmer-30B",
    manifest_id="muse_glimmer",
    arch="MuseGlimmerForConditionalGeneration",
    model_type="muse_glimmer",
    num_layers=52,
    layer_types=LAYER_TYPES,
    attention=AttentionSpec(
        kind="irope",
        num_heads=32,
        num_kv_heads=2,
        head_dim=128,
        output_gate=True,
        rope="irope",
        sliding_window=2048,
        rope_disabled_layers=FULL_ATTENTION_LAYERS,
    ),
    linear=None,
    moe=None,
    quantization="none",
    kv={
        "layout": "block_first_fused_nhd",
        "stride_order": [1, 0, 2, 3, 4],
        "block_dim": 0,
        "mixed_specs": True,
    },
    required_capabilities=(
        Capability.ATTENTION_FULL.value,
        Capability.ATTENTION_SLIDING.value,
        Capability.NORM_RMS.value,
        Capability.ROTARY.value,
        Capability.SAMPLING_TOPK.value,
    ),
    benchmark={},  # C2 sample pending; deployment does not depend on it
    model_impl="tileinfer.bridge.models.muse_glimmer",
    reasoning_parser="atem",
    reasoning_parser_impl="tileinfer.bridge.models.atem:AtemReasoningParser",
    source="docs/notes/muse-glimmer-static.md",
)
