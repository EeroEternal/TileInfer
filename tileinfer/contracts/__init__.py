"""L3 - the capability contracts.

This package is the *interface* between the decision layer (L2) and the implementation
providers (L4).  It is deliberately free of vLLM, torch_npu and any concrete kernel: it can
be imported and unit-tested on a machine with neither an NPU nor an engine installed.

The eight contracts live in :mod:`tileinfer.contracts.capabilities`; the data products that
cross layer boundaries live next to them (:mod:`~tileinfer.contracts.plan`,
:mod:`~tileinfer.contracts.manifest`, :mod:`~tileinfer.contracts.gaps`).

Nothing in ``tileinfer.bridge`` may be imported from here: the dependency direction is
``bridge -> resolve -> contracts``, never the other way around.
"""

from __future__ import annotations

from .capabilities import (
    CAPABILITIES,
    AttentionContract,
    Capability,
    LinearAttentionContract,
    MoEContract,
    NormContract,
    ProviderSpec,
    QuantContract,
    RotaryContract,
    SamplingContract,
    StateContract,
)
from .gaps import Gap, GapReport
from .manifest import AttentionSpec, LinearSpec, ModelManifest, MoESpec
from .plan import ModelRef, Plan, Selection, VersionTriangle

__all__ = [
    "CAPABILITIES",
    "AttentionContract",
    "Capability",
    "LinearAttentionContract",
    "MoEContract",
    "NormContract",
    "ProviderSpec",
    "QuantContract",
    "RotaryContract",
    "SamplingContract",
    "StateContract",
    "Gap",
    "GapReport",
    "AttentionSpec",
    "LinearSpec",
    "ModelManifest",
    "MoESpec",
    "ModelRef",
    "Plan",
    "Selection",
    "VersionTriangle",
]
