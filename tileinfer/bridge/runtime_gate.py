"""The single gate that answers "is this plan still valid for the live run?".

The plan records versions, architecture and parallel axes.  Doctor used to re-implement
these checks, while ``install()`` checked only versions; the two could disagree.  This module
is the shared implementation: install, doctor and future deployment tools call it.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, Optional

from ..contracts.capabilities import parallel_supports


def _python_key(version: str) -> str:
    return ".".join(str(version).split(".")[:2])


def check_plan(
    plan: Any,
    live: Dict[str, str],
    *,
    parallel_axes: Optional[tuple[str, ...]] = None,
    check_versions: bool = True,
) -> Optional[str]:
    """Return a skip reason, or ``None`` when the plan may be injected."""
    if plan is None:
        return "no plan"
    planned = plan.versions
    if check_versions:
        for key, planned_value, live_value in (
            ("vllm", planned.vllm, live["vllm"]),
            ("vllm_ascend", planned.vllm_ascend, live["vllm_ascend"]),
        ):
            if planned_value != live_value:
                return f"version gate: plan {key}={planned_value}, live={live_value}"
        if _python_key(planned.python) != _python_key(live["python"]):
            return f"version gate: plan python={planned.python}, live={live['python']}"

    if parallel_axes is not None:
        plan_axes = tuple(getattr(plan, "parallel", ("single",)))
        live_axes = tuple(parallel_axes)
        if set(plan_axes) != set(live_axes):
            return (
                f"parallel gate: plan was built for {list(plan_axes)} but this run is "
                f"{list(live_axes)}; rebuild the plan with --tp/--ep/--dp/--pp"
            )
        try:
            from ..providers import find_provider
        except Exception:  # pragma: no cover - providers optional in early bring-up
            find_provider = lambda name: None  # noqa: E731
        bad = []
        for selection in getattr(plan, "selections", ()):
            if getattr(selection, "source", "vendor") == "vendor":
                continue
            provider = find_provider(selection.provider)
            if provider is not None and not parallel_supports(provider.parallel, live_axes):
                bad.append(f"{selection.provider} supports {list(provider.parallel)}")
        if bad:
            return "parallel gate: provider(s) do not cover this run: " + ", ".join(bad)
    return None


__all__ = ["check_plan"]
