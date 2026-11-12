"""The decision: a pure function from (declaration, environment, policy) to artefacts.

It has no side effects, no clock, no device access and no imports of implementations.  Two
runs with the same inputs produce byte-identical plans - which is what makes a deployment
reproducible and a regression bisectable.

Policy
------
``vendor_first`` (default): if vllm-ascend is installed and advertises a capability, use it.
The vendor is the baseline, and a transparent layer that does not regress is worth more than
a faster layer that cannot be compared.  A capability the vendor cannot supply is looked up
in our own provider catalogue; if neither has it, it becomes a gap.

``tileinfer_first`` exists for A/B and for machines without a vendor platform; it is not the
default because on Ascend it would trade a measured baseline for an unmeasured one.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from ..contracts.capabilities import Capability
from ..contracts.gaps import Gap, GapReport
from ..contracts.manifest import ModelManifest
from ..contracts.plan import (
    ACCESS_POINTS,
    ModelRef,
    Plan,
    Selection,
    VersionTriangle,
)
from .probe import Environment
from .vendor import VENDOR_PROVIDERS

#: Capabilities our own providers can serve.  Empty today, and empty is correct: this table
#: is for gaps only - a capability the vendor cannot run, or cannot run correctly at the shape
#: a model needs.  Qwen3.6 requires nothing the vendor lacks, so claiming our v2 kernels here
#: would invent work and make the gap report lie.  Entries appear when a model needs a
#: capability the vendor rejects, not when we believe we could beat the vendor on speed.
TILEINFER_PROVIDERS: dict = {}

POLICIES = ("vendor_first", "tileinfer_first")


def _select(
    manifest: ModelManifest,
    env: Environment,
    policy: str,
) -> Tuple[List[Selection], List[Gap]]:
    selections: List[Selection] = []
    gaps: List[Gap] = []
    for cap_name in manifest.required_capabilities:
        capability = Capability(cap_name)
        vendor = VENDOR_PROVIDERS.get(capability)
        ours = TILEINFER_PROVIDERS.get(capability)
        order = ("vendor", "tileinfer") if policy == "vendor_first" else ("tileinfer", "vendor")

        chosen: Optional[Selection] = None
        for source in order:
            if source == "vendor" and env.has_vendor and vendor is not None:
                chosen = Selection(
                    capability=capability.value,
                    provider=vendor.name,
                    impl=vendor.impl,
                    reason=f"{policy}: vendor advertises the capability",
                    source="vendor",
                    graphable=vendor.graphable,
                )
                break
            if source == "tileinfer" and ours is not None:
                chosen = Selection(
                    capability=capability.value,
                    provider=ours.name,
                    impl=ours.impl,
                    reason=f"{policy}: selected our provider",
                    source="tileinfer",
                    graphable=ours.graphable,
                )
                break

        if chosen is not None:
            selections.append(chosen)
            continue
        reason = (
            "vllm-ascend is not installed, so no vendor provider exists"
            if not env.has_vendor
            else "vllm-ascend is installed but does not advertise this capability"
        )
        gaps.append(
            Gap(
                capability=capability.value,
                required=True,
                reason=reason,
                candidates=[],
            )
        )
    return selections, gaps


def decide(
    manifest: ModelManifest,
    env: Environment,
    policy: str = "vendor_first",
) -> Tuple[Plan, GapReport]:
    """Compute the plan and the gap report from one declaration."""
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}; known: {POLICIES}")
    selections, gaps = _select(manifest, env, policy)
    servable = not any(g.required for g in gaps)

    # The access points that are actually installed.  ``impl_table`` and ``model_table`` are
    # only listed when the decision selects something that needs them; an unused mechanism
    # in the budget would be noise, not capability.
    access = ["plugin_entry", "attn_select", "method_wrap", "graph_hook"]
    for name in access:
        assert name in ACCESS_POINTS

    plan = Plan(
        model=ModelRef(
            name=manifest.name,
            manifest=manifest.manifest_id,
            arch=manifest.arch,
            fingerprint=manifest.fingerprint,
        ),
        versions=VersionTriangle(
            vllm=env.vllm or "missing",
            vllm_ascend=env.vllm_ascend or "missing",
            python=env.python,
            torch=env.torch or "",
        ),
        selections=selections,
        access_points=access,
        instrumentation={
            "graph_counters": True,
            "metrics_path": "/tmp/tileinfer-v3-metrics.json",
            "dump_every_decode_steps": 500,
        },
        fallback_policy="forbid",
    )
    plan.validate()

    report = GapReport(
        model=manifest.name,
        manifest_id=manifest.manifest_id,
        servable=servable,
        gaps=gaps,
        notes=[
            f"policy={policy}",
            f"vendor={env.vllm_ascend or 'missing'}",
            f"device={env.device_type or 'unknown'}",
            "the layer selects implementations; it does not own platform/worker/KV/graph",
        ],
    )
    return plan, report
