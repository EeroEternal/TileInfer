"""Route-O provider catalogue: declarations only, no engine imports.

``resolve`` reads this table only for a declaration that ships its own model
implementation: the provider is consumed *inside* that model, not installed through the
vLLM operator table.  Keeping it out of ``TILEINFER_PROVIDERS`` is deliberate - that table
is the global default set, and a model-scoped implementation must not become a default for
an unrelated model.
"""

from __future__ import annotations

from typing import Dict

from ..contracts.capabilities import Capability, ProviderSpec

ROUTE_O_PROVIDERS: Dict[Capability, ProviderSpec] = {
    Capability.STATE_LINEAR: ProviderSpec(
        capability=Capability.STATE_LINEAR,
        name="tileinfer.short_conv",
        impl="tileinfer.providers.state.short_conv:ShortConvState",
        source="tileinfer",
        graphable=True,
        covers={
            "mixer": "short_conv",
            "model_type": "lfm3",
            "kernel": [7],
            "dtype": ["bf16"],
            "layout": "depthwise_causal",
        },
        parallel=("single",),
        measured_ratio=None,
        evidence="docs/notes/lfm3-adaptation.md",
        delete_when="vllm-ascend gains a short-conv state layer usable by LFM3",
        notes="torch causal conv1d; consumed inside the LFM3 route-O model",
    ),
}

__all__ = ["ROUTE_O_PROVIDERS"]
