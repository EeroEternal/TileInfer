"""Nemotron v3 providers: implementations of the vendor gaps this model needs.

The entries here are **candidates**, not defaults.  They exist because vllm-ascend has
no Mamba2 SSD/state implementation and its non-gated relu2 expert MLP is rejected at the
target shape.  A candidate may be selected by a plan (for a bring-up run) but is not
registered in ``resolve.decide.TILEINFER_PROVIDERS`` until real-weight C1/C2 evidence exists.

The implementations are pure torch and import no engine symbol.  Installation is done by
``tileinfer.bridge.impl_table`` (L5), never by this package.
"""

from __future__ import annotations

from ...contracts.capabilities import Capability, ProviderSpec

MAMBA2_STATE = ProviderSpec(
    capability=Capability.STATE_LINEAR,
    name="tileinfer.mamba2_state",
    impl="tileinfer.providers.nemotron.mamba2",
    source="tileinfer",
    graphable=True,
    covers={"mixer": "mamba2"},
    measured_ratio=None,
    evidence="m5-nemotron-mamba2-ops",
    delete_when="vllm-ascend ships Mamba2 SSD/state at the target shapes",
    notes=(
        "candidate, not a default: target-geometry CPU reference and dummy launch "
        "mechanics pass; real-weight C1/C2 and C3 graph capture are still pending"
    ),
    status="candidate",
    scope="global",
    install="impl_table",
    graph_safe=True,
    evidence_level="device-c3",
)

NEMOTRON_MOE = ProviderSpec(
    capability=Capability.MOE_ROUTED,
    name="tileinfer.nemotron_moe",
    impl="tileinfer.providers.moe",
    source="tileinfer",
    graphable=True,
    covers={"model_type": "nemotron_h", "activation": "relu2_no_mul", "is_act_and_mul": False},
    measured_ratio=None,
    evidence="m5-nemotron-mamba2-ops",
    delete_when=(
        "vllm-ascend supports a non-gated relu2 expert MLP at the target shape"
    ),
    notes=(
        "candidate, not a default: the torch fused_experts path and the Ascend non-gated "
        "hook are implemented and CPU-tested; target real-weight C2 is pending"
    ),
    status="candidate",
    scope="global",
    install="impl_table",
    graph_safe=True,
    evidence_level="device-c3",
)

CANDIDATES = (MAMBA2_STATE, NEMOTRON_MOE)
PROVIDERS_BY_NAME = {provider.name: provider for provider in CANDIDATES}

__all__ = ["MAMBA2_STATE", "NEMOTRON_MOE", "CANDIDATES", "PROVIDERS_BY_NAME"]
