"""Our provider catalogue: declarations only, no engine imports.

``resolve`` reads this table; the implementation modules are imported by the model/bridge at
run time, never by the decision layer.  A provider exists here only for a capability the
vendor cannot serve at the shape a declaration requires.
"""

from __future__ import annotations

from typing import Dict

from ..contracts.capabilities import Capability, ProviderSpec

TILEINFER_PROVIDERS: Dict[Capability, ProviderSpec] = {
    Capability.STATE_LINEAR: ProviderSpec(
        capability=Capability.STATE_LINEAR,
        name="tileinfer.short_conv",
        impl="tileinfer.providers.state.short_conv:ShortConvState",
        source="tileinfer",
        graphable=True,
        covers={"kernel": [7], "dtype": ["bf16"], "layout": "depthwise_causal"},
        parallel=("single",),
        measured_ratio=None,
        evidence="docs/notes/lfm3-adaptation.md",
        delete_when="vllm-ascend gains a short-conv state layer usable by LFM3",
        notes="torch causal conv1d; the vendor state provider is GDN-only",
    ),
}

__all__ = ["TILEINFER_PROVIDERS"]
