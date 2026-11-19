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
from typing import Any, Callable, Dict, List, Sequence

from .results import InstallResult

logger = logging.getLogger("tileinfer.bridge.impl_table")

__all__ = ["install", "ImplReport"]


class ImplReport(InstallResult):
    """Provider-binder result; the common install-result type with a fixed name."""

    def __init__(self) -> None:
        super().__init__(name="impl_table")


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

    _install_mamba2_ascend_spec_metadata(report)
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


def _install_mamba2_ascend_spec_metadata(report: ImplReport) -> None:
    """Let vllm-ascend's V1 runner hand speculative metadata to the Mamba2 builder.

    The Ascend runner forwards ``num_accepted_tokens`` / ``num_decode_draft_tokens_cpu`` only
    when the builder is an instance of ``GDNAttentionMetadataBuilder``.  Mamba2 needs the same
    arguments: without them a 1+draft target verification batch is classified as a prefill and
    the SSM/conv state advances by both draft tokens instead of the accepted prefix.  Widening
    the runner's builder tuple is a bridge change; no vLLM or vllm-ascend source is edited.
    """
    try:
        from vllm.v1.attention.backends.mamba2_attn import (
            Mamba2AttentionMetadataBuilder as _mamba2_builder,
        )
    except Exception:
        return
    try:
        import vllm_ascend.worker.model_runner_v1 as _runner
    except Exception:
        return
    try:
        current = _runner.GDNAttentionMetadataBuilder
        builders = current if isinstance(current, tuple) else (current,)
        if _mamba2_builder not in builders:
            _runner.GDNAttentionMetadataBuilder = (*builders, _mamba2_builder)
        report.installed.append("mamba2:ascend_spec_metadata")
    except Exception as exc:  # noqa: BLE001 - report the bridge failure, never crash import
        report.failed["mamba2:ascend_spec_metadata"] = f"{type(exc).__name__}: {exc}"


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
    from tileinfer.providers import find_provider

    for selection in non_vendor:
        name = str(selection.provider)
        if name in seen:
            continue
        seen.add(name)
        spec = find_provider(name)
        if spec is None:
            report.ok = False
            report.failed[name] = "no v3 provider binder for unknown provider (refusing to guess)"
            continue

        actions = tuple(getattr(spec, "install_actions", ()) or ())
        if actions:
            before = len(report.failed)
            for action in actions:
                if action.kind == "model_call":
                    report.skipped.append(f"{name}: consumed inside model_impl")
                    continue
                if action.kind == "vendor_patch":
                    from .compat_hooks import install_compat_hook

                    result = install_compat_hook(action.hook)
                    if result.ok and not result.get("disabled"):
                        report.installed.append(f"{name}:{action.hook}")
                    else:
                        report.failed[f"{name}:{action.hook}"] = result.detail or "hook failed"
                    continue
                if action.kind == "impl_bind":
                    try:
                        module_name, _, actor = action.actor.partition(":")
                        binder = getattr(importlib.import_module(module_name), actor)
                        binder(report)
                    except Exception as exc:  # noqa: BLE001
                        report.failed[f"{name}:{action.actor}"] = (
                            f"{type(exc).__name__}: {exc}"
                        )
                    continue
                report.failed[f"{name}:{action.kind}"] = "unknown install action"
            if len(report.failed) > before:
                report.ok = False
            else:
                report.installed.append(f"provider:{name}")
            continue

        # Legacy declaration path (install_actor / install=model_impl).  New providers
        # should use install_actions so this branch eventually disappears.
        if spec.install == "model_impl":
            report.skipped.append(f"{name}: consumed inside model_impl")
            continue
        binder = None
        if spec.install_actor:
            try:
                module_name, _, actor = spec.install_actor.partition(":")
                binder = getattr(importlib.import_module(module_name), actor)
            except Exception as exc:  # noqa: BLE001
                report.ok = False
                report.failed[name] = (
                    f"cannot resolve install_actor {spec.install_actor!r}: "
                    f"{type(exc).__name__}: {exc}"
                )
                continue
        if binder is None:
            report.ok = False
            report.failed[name] = "no v3 provider binder (no install_actions declared)"
            continue
        before = len(report.failed)
        binder(report)
        if len(report.failed) > before:
            report.ok = False
        else:
            report.installed.append(f"provider:{name}")
    return report
