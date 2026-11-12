"""The vendor provider catalogue (data, not code).

vllm-ascend is the baseline on Ascend.  Its providers are recorded here with the same five
fields ours must carry, so the decision layer can compare them uniformly and so a gap report
can name the exact vendor implementation that is missing.

``measured_ratio`` is 1.0 by definition for the baseline: the A/B denominator is the vendor
implementation itself.  That is an identity, not a claim, and it is why a vendor provider is
allowed to be the default without a separate benchmark.
"""

from __future__ import annotations

from typing import Dict

from ..contracts.capabilities import Capability, ProviderSpec

VENDOR_PROVIDERS: Dict[Capability, ProviderSpec] = {
    Capability.ATTENTION_FULL: ProviderSpec(
        capability=Capability.ATTENTION_FULL,
        name="vllm_ascend.attention",
        impl="vllm_ascend.attention.attention_v1.AscendAttentionBackend",
        source="vendor",
        graphable=True,
        measured_ratio=1.0,
        evidence="baseline",
        delete_when="never - this is the baseline",
    ),
    Capability.ATTENTION_LINEAR: ProviderSpec(
        capability=Capability.ATTENTION_LINEAR,
        name="vllm_ascend.gdn",
        impl="vllm_ascend.ops.gdn_attn_builder.AscendGDNAttentionBackend",
        source="vendor",
        graphable=True,
        measured_ratio=1.0,
        evidence="baseline",
        delete_when="never - this is the baseline",
    ),
    Capability.MOE_ROUTED: ProviderSpec(
        capability=Capability.MOE_ROUTED,
        name="vllm_ascend.fused_moe",
        impl="vllm_ascend.ops.fused_moe",
        source="vendor",
        graphable=True,
        measured_ratio=1.0,
        evidence="baseline",
        delete_when="never - this is the baseline",
    ),
    Capability.NORM_RMS: ProviderSpec(
        capability=Capability.NORM_RMS,
        name="vllm_ascend.layernorm",
        impl="vllm_ascend.ops.layernorm",
        source="vendor",
        graphable=True,
        measured_ratio=1.0,
        evidence="baseline",
        delete_when="never - this is the baseline",
    ),
    Capability.ROTARY: ProviderSpec(
        capability=Capability.ROTARY,
        name="vllm_ascend.rotary",
        impl="vllm_ascend.ops.rotary_embedding",
        source="vendor",
        graphable=True,
        measured_ratio=1.0,
        evidence="baseline",
        delete_when="never - this is the baseline",
    ),
    Capability.SAMPLING_TOPK: ProviderSpec(
        capability=Capability.SAMPLING_TOPK,
        name="vllm_ascend.sample",
        impl="vllm_ascend.sample",
        source="vendor",
        graphable=True,
        measured_ratio=1.0,
        evidence="baseline",
        delete_when="never - this is the baseline",
    ),
    Capability.STATE_LINEAR: ProviderSpec(
        capability=Capability.STATE_LINEAR,
        name="vllm_ascend.gdn_state",
        impl="vllm_ascend.ops.gdn",
        source="vendor",
        graphable=True,
        measured_ratio=1.0,
        evidence="baseline",
        delete_when="never - this is the baseline",
    ),
    Capability.QUANT_VERIFY: ProviderSpec(
        capability=Capability.QUANT_VERIFY,
        name="vllm_ascend.quantization",
        impl="vllm_ascend.quantization",
        source="vendor",
        graphable=True,
        measured_ratio=1.0,
        evidence="baseline",
        delete_when="never - this is the baseline",
    ),
}

__all__ = ["VENDOR_PROVIDERS"]
