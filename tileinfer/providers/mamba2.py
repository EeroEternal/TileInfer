"""Candidate provider for the Mamba2 half of ``state.linear``.

This is deliberately **not** entered in ``resolve.decide.TILEINFER_PROVIDERS``.  The
capability contract is filled by a real torch implementation
(``tileinfer.vllm.ops.mamba2``); the target dummy service and the target-geometry CPU
reference tests exercise it, but the target's real-weight C1/C2 has not run because nine
checkpoint shards are absent.  Keeping the declaration here records the provider shape and
its evidence without turning it into a default or a support claim.

Delete condition: upstream vllm-ascend ships a Mamba2 SSD/state path at these shapes and the
replacement is no longer needed.
"""

from __future__ import annotations

from ..contracts.capabilities import Capability, ProviderSpec

MAMBA2_STATE = ProviderSpec(
    capability=Capability.STATE_LINEAR,
    name="tileinfer.mamba2_state",
    impl="tileinfer.vllm.ops.mamba2",
    source="tileinfer",
    graphable=True,
    covers={"mixer": "mamba2"},
    measured_ratio=None,
    evidence="m5-nemotron-mamba2-ops",
    delete_when="vllm-ascend ships Mamba2 SSD/state at the target shapes",
    notes=(
        "candidate, not a default: target-geometry CPU reference and target-dummy graph "
        "capture pass; real-weight C1/C2 is still pending because 9/14 shards are absent"
    ),
)

__all__ = ["MAMBA2_STATE"]
