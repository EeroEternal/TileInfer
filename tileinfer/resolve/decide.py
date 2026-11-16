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


def _candidates(capability: Capability, manifest: ModelManifest) -> List:
    """Declared candidates that cover this manifest but are not allowed to be defaults.

    A candidate has implementation evidence but not target real-weight C1/C2.  It must appear
    in the gap report (so the gap is visibly implementable, not a blocker) without entering
    ``TILEINFER_PROVIDERS`` (which would make the plan claim support).
    """
    try:
        from ..providers import CANDIDATE_PROVIDERS
    except Exception:  # pragma: no cover - providers are optional during early bring-up
        return []
    return [
        provider
        for provider in CANDIDATE_PROVIDERS
        if provider.capability is capability and _covers(provider, manifest)
    ]


POLICIES = ("vendor_first", "tileinfer_first")


def _covers(provider, manifest: ModelManifest) -> bool:
    """Honour the provider's symbolic `covers` constraints.

    `state.linear` is shared by GDN and Mamba2.  Without this check a vendor GDN
    provider is selected for a Mamba2 declaration just because both spell the same
    capability.  Unknown cover keys are ignored here; they are evidence metadata, not
    routing keys.
    """
    if provider is None:
        return True
    wanted_mixer = provider.covers.get("mixer")
    if wanted_mixer is not None and wanted_mixer not in manifest.mixer_kinds:
        return False
    wanted_model = provider.covers.get("model_type")
    if wanted_model is not None and wanted_model != manifest.model_type:
        return False
    return True


def _select(
    manifest: ModelManifest,
    env: Environment,
    policy: str,
    allow_candidates: bool = False,
) -> Tuple[List[Selection], List[Gap]]:
    selections: List[Selection] = []
    gaps: List[Gap] = []
    for cap_name in manifest.required_capabilities:
        capability = Capability(cap_name)
        vendor = VENDOR_PROVIDERS.get(capability)
        ours = TILEINFER_PROVIDERS.get(capability)
        order = ("vendor", "tileinfer") if policy == "vendor_first" else ("tileinfer", "vendor")

        if allow_candidates:
            # A candidate with a model-type cover replaces the vendor even when the vendor
            # advertises the capability: the vendor's generic MoE does not cover Nemotron's
            # non-gated relu2 expert MLP.  This is bring-up only and still writes a required gap.
            overrides = [
                provider
                for provider in _candidates(capability, manifest)
                if provider.covers.get("model_type") == manifest.model_type
            ]
            if overrides:
                candidate = overrides[0]
                selections.append(
                    Selection(
                        capability=capability.value,
                        provider=candidate.name,
                        impl=candidate.impl,
                        reason=(
                            f"explicit candidate bring-up; {candidate.name} covers "
                            f"model_type={manifest.model_type}, the vendor does not"
                        ),
                        source="tileinfer",
                        graphable=candidate.graphable,
                    )
                )
                gaps.append(
                    Gap(
                        capability=capability.value,
                        required=True,
                        reason=(
                            "candidate overrides the vendor for this model type; "
                            f"target real-weight C1/C2 pending for {candidate.name}"
                        ),
                        candidates=[provider.name for provider in overrides],
                    )
                )
                continue

        chosen: Optional[Selection] = None
        for source in order:
            if source == "vendor" and env.has_vendor and vendor is not None and _covers(vendor, manifest):
                chosen = Selection(
                    capability=capability.value,
                    provider=vendor.name,
                    impl=vendor.impl,
                    reason=f"{policy}: vendor advertises the capability",
                    source="vendor",
                    graphable=vendor.graphable,
                )
                break
            if source == "tileinfer" and ours is not None and _covers(ours, manifest):
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
        if allow_candidates:
            candidates = _candidates(capability, manifest)
            if candidates:
                candidate = candidates[0]
                selections.append(
                    Selection(
                        capability=capability.value,
                        provider=candidate.name,
                        impl=candidate.impl,
                        reason=(
                            "explicit candidate bring-up; the plan is still servable=false "
                            "until real-weight C1/C2"
                        ),
                        source="tileinfer",
                        graphable=candidate.graphable,
                    )
                )
                gaps.append(
                    Gap(
                        capability=capability.value,
                        required=True,
                        reason=(
                            "candidate selected only for an explicit bring-up run; "
                            f"target real-weight C1/C2 pending for {candidate.name}"
                        ),
                        candidates=[provider.name for provider in candidates],
                    )
                )
                continue
        reason = (
            "vllm-ascend is not installed, so no vendor provider exists"
            if not env.has_vendor
            else "vllm-ascend is installed but does not advertise this capability"
        )
        candidates = _candidates(capability, manifest)
        if candidates:
            names = ", ".join(provider.name for provider in candidates)
            reason += f"; TileInfer candidate(s) implemented but not default until target C1/C2: {names}"
        gaps.append(
            Gap(
                capability=capability.value,
                required=True,
                reason=reason,
                candidates=[provider.name for provider in candidates],
            )
        )
    return selections, gaps


def decide(
    manifest: ModelManifest,
    env: Environment,
    policy: str = "vendor_first",
    allow_candidates: bool = False,
) -> Tuple[Plan, GapReport]:
    """Compute the plan and the gap report from one declaration.

    ``allow_candidates`` is for an explicit bring-up run.  It selects declared candidates
    into the plan (so the v3 ``impl_table`` can install them) but keeps the gap required and
    ``Plan.servable`` false; the bridge still refuses unless its own bring-up override is set.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}; known: {POLICIES}")
    selections, gaps = _select(manifest, env, policy, allow_candidates=allow_candidates)
    servable = not any(g.required for g in gaps)

    # The access points that are actually installed.  ``impl_table`` and ``model_table`` are
    # only listed when the decision selects something that needs them; an unused mechanism
    # in the budget would be noise, not capability.
    access = ["plugin_entry", "attn_select", "method_wrap", "graph_hook"]
    if manifest.model_impl:
        # The vendor has no architecture for this model: the layer must register it, which is
        # the model_table access point.  Parser registration rides along with it.
        access.append("model_table")
    if any(selection.source != "vendor" for selection in selections):
        access.append("impl_table")
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
        model_impl=manifest.model_impl,
        reasoning_parser=manifest.reasoning_parser,
        reasoning_parser_impl=manifest.reasoning_parser_impl,
        servable=servable,
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
            f"allow_candidates={bool(allow_candidates)}",
        ],
    )
    return plan, report
