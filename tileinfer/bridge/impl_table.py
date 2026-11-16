"""Plan-driven operator table for v3 providers.

The plan is the only authority here.  A provider is installed **only** when the plan has a
non-vendor selection for it **and** the plan lists the ``impl_table`` access point.  There is
no global operator installer in this module: the v2 ``install_ops`` table is frozen and not
imported.

Mamba2 and the non-gated MoE adapter are the first two entries.  Their implementations live in
``tileinfer.providers`` (L4); this module is the L5 bridge that knows the vLLM names and how to
write them.
"""

from __future__ import annotations

import importlib
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Sequence

logger = logging.getLogger("tileinfer.bridge.impl_table")

__all__ = ["install", "PROVIDER_BINDERS", "ImplReport"]


@dataclass
class ImplReport:
    ok: bool = True
    installed: List[str] = field(default_factory=list)
    failed: Dict[str, str] = field(default_factory=dict)
    skipped: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "installed": list(self.installed),
            "failed": dict(self.failed),
            "skipped": list(self.skipped),
        }

    def summary(self) -> str:
        return (
            f"impl_table: {len(self.installed)} installed, {len(self.failed)} failed, "
            f"{len(self.skipped)} skipped"
        )


# The vLLM 0.23 names.  The aliases are needed because the mixers import the names by value.
_MAMBA_PREFIX = "vllm.model_executor.layers.mamba"
_MAMBA2_TARGETS: Dict[str, Sequence[str]] = {
    "causal_conv1d_fn": (
        f"{_MAMBA_PREFIX}.ops.causal_conv1d.causal_conv1d_fn",
        f"{_MAMBA_PREFIX}.mamba_mixer2.causal_conv1d_fn",
    ),
    "causal_conv1d_update": (
        f"{_MAMBA_PREFIX}.ops.causal_conv1d.causal_conv1d_update",
        f"{_MAMBA_PREFIX}.mamba_mixer2.causal_conv1d_update",
    ),
    "mamba_chunk_scan_combined_varlen": (
        f"{_MAMBA_PREFIX}.ops.ssd_combined.mamba_chunk_scan_combined_varlen",
        f"{_MAMBA_PREFIX}.mamba_mixer2.mamba_chunk_scan_combined_varlen",
    ),
    "selective_state_update": (
        f"{_MAMBA_PREFIX}.ops.ssu_dispatch.selective_state_update",
        f"{_MAMBA_PREFIX}.mamba_mixer2.selective_state_update",
    ),
}


def _resolve(path: str) -> tuple[Any, str]:
    module_path, _, name = path.rpartition(".")
    if not module_path:
        raise ImportError(f"{path!r} is not a module attribute path")
    module = importlib.import_module(module_path)
    return module, name


def _bind_attribute(path: str, impl: Callable[..., Any]) -> str:
    """Set one module attribute; idempotent when it is already ours."""
    owner, name = _resolve(path)
    current = getattr(owner, name, None)
    if current is impl:
        return "already"
    if current is None:
        raise AttributeError(f"{path} does not exist")
    setattr(owner, name, impl)
    return "bound"


def _install_mamba2(report: ImplReport) -> None:
    from tileinfer.providers.nemotron import mamba2

    for op, paths in _MAMBA2_TARGETS.items():
        impl = getattr(mamba2, op)
        bound = 0
        for path in paths:
            try:
                status = _bind_attribute(path, impl)
            except Exception as exc:  # noqa: BLE001 - a missing vLLM name is data, not a crash
                report.failed[f"{op}:{path}"] = f"{type(exc).__name__}: {exc}"
                continue
            bound += 1
            if status == "bound":
                report.installed.append(path)
        if bound == 0:
            report.failed[op] = "no vLLM target resolves (version/module drift)"


def _install_nemotron_moe(report: ImplReport) -> None:
    """Install the vLLM/vllm-ascend non-gated MoE adapters (bridge code, not a kernel)."""
    from tileinfer.bridge import nemotron_moe

    for name, fn in (
        ("non_gated_guard", nemotron_moe.install_non_gated_guard),
        ("ascend_non_gated_experts", nemotron_moe.install_ascend_non_gated_experts),
        ("nemotron_shared_alias", nemotron_moe.install_nemotron_shared_alias),
    ):
        try:
            ok = bool(fn())
        except Exception as exc:  # noqa: BLE001 - report, never take the process down
            report.failed[name] = f"{type(exc).__name__}: {exc}"
            continue
        if ok:
            report.installed.append(f"moe:{name}")
        else:
            report.failed[f"moe:{name}"] = "adapter returned False (target unavailable)"


PROVIDER_BINDERS: Mapping[str, Callable[[ImplReport], None]] = {
    "tileinfer.mamba2_state": _install_mamba2,
    "tileinfer.nemotron_moe": _install_nemotron_moe,
}


def install(plan: Any) -> ImplReport:
    """Install the plan's non-vendor providers through the v3 ``impl_table`` seam.

    Rules, all fail-closed:
    * no non-vendor selection -> no-op (vendor table is kept);
    * a non-vendor selection without ``impl_table`` in ``plan.access_points`` -> failure;
    * an unknown provider -> failure, never guessed;
    * one provider failure does not hide the others, but the report is ``ok=False``.
    """
    report = ImplReport()
    non_vendor = [s for s in plan.selections if getattr(s, "source", "vendor") != "vendor"]
    if not non_vendor:
        report.skipped.append("no non-vendor selections (vendor table kept)")
        return report
    if "impl_table" not in list(plan.access_points):
        report.ok = False
        report.failed["plan"] = (
            "non-vendor selections require the impl_table access point; "
            f"plan has {list(plan.access_points)!r}"
        )
        return report
    if not bool(getattr(plan, "servable", True)):
        # The decision already called this model unservable.  An explicit bring-up run may
        # override it, but the default must fail closed rather than install a candidate.
        if os.environ.get("TILEINFER_ALLOW_UNSERVABLE", "").strip().lower() not in {
            "1",
            "true",
            "yes",
            "on",
        }:
            report.ok = False
            report.failed["plan"] = (
                "plan.servable=false; refusing non-vendor provider install "
                "(set TILEINFER_ALLOW_UNSERVABLE=1 only for a labelled bring-up run)"
            )
            return report
        logger.warning(
            "TileInfer impl_table: installing a non-vendor provider for an unservable plan "
            "because TILEINFER_ALLOW_UNSERVABLE is set"
        )

    seen: set[str] = set()
    for selection in non_vendor:
        name = str(selection.provider)
        if name in seen:
            continue
        seen.add(name)
        binder = PROVIDER_BINDERS.get(name)
        if binder is None:
            report.ok = False
            report.failed[name] = "no v3 provider binder (refusing to guess)"
            continue
        before = len(report.failed)
        binder(report)
        if len(report.failed) > before:
            report.ok = False
        else:
            report.installed.append(f"provider:{name}")
    return report
