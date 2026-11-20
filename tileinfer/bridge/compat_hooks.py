"""Registry for temporary vendor compatibility hooks.

Hooks are not providers: they change an upstream code path that the vendor already owns
(platform guards, monkeypatches, global RoPE-buffer patches) or record a model-side numeric
fix that cannot be expressed as a provider.  They are the most likely place for a bring-up
patch to become permanent debt, so each one is declared with an owner, a disable switch and
a delete condition.

Convention:
    ``package.module:actor`` -> ``actor()``
The actor returns a bool or a mapping; ``False`` is a failed hook.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

from .results import InstallResult

_TRUE = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class CompatHook:
    name: str
    kind: str
    target: str
    installer: str
    disable_env: str
    delete_when: str
    evidence: str
    idempotent: bool = True
    notes: str = ""


HOOKS: Dict[str, CompatHook] = {
    "lfm3_rope_global_buffer": CompatHook(
        name="lfm3_rope_global_buffer",
        kind="rope_patch",
        target="vllm_ascend.ops.rotary_embedding.update_cos_sin",
        installer="tileinfer.bridge.models.lfm3:_patch_ascend_rope",
        disable_env="TILEINFER_DISABLE_LFM3_ROPE_PATCH",
        delete_when="vllm-ascend supports per-layer rotary dimensions",
        evidence="docs/notes/lfm3-adaptation.md",
        notes="LFM3 has 64/128 rotary dims; the vendor keeps one global cos/sin buffer",
    ),
    "nemotron_non_gated_guard": CompatHook(
        name="nemotron_non_gated_guard",
        kind="platform_guard",
        target="vllm.model_executor.layers.fused_moe.layer.FusedMoE.__init__",
        installer="tileinfer.bridge.nemotron_moe:install_non_gated_guard",
        disable_env="TILEINFER_DISABLE_NEMOTRON_MOE_GUARD",
        delete_when="vLLM/Ascend accepts is_act_and_mul=False on the NPU platform",
        evidence="docs/notes/m5-nemotron-mamba2-ops.md",
    ),
    "nemotron_ascend_non_gated_experts": CompatHook(
        name="nemotron_ascend_non_gated_experts",
        kind="monkeypatch",
        target="vllm_ascend.ops.fused_moe.AscendUnquantizedFusedMoEMethod.apply",
        installer="tileinfer.bridge.nemotron_moe:install_ascend_non_gated_experts",
        disable_env="TILEINFER_DISABLE_NEMOTRON_ASCEND_MOE",
        delete_when="vllm-ascend ships the non-gated relu2 expert MLP",
        evidence="docs/notes/m5-nemotron-mamba2-ops.md",
    ),
    "nemotron_shared_alias": CompatHook(
        name="nemotron_shared_alias",
        kind="monkeypatch",
        target="vllm.model_executor.models.nemotron_h.NemotronHMLP.gate_up_proj",
        installer="tileinfer.bridge.nemotron_moe:install_nemotron_shared_alias",
        disable_env="TILEINFER_DISABLE_NEMOTRON_SHARED_ALIAS",
        delete_when="Ascend shared-expert path accepts up_proj/down_proj without gate_up_proj",
        evidence="docs/notes/m5-nemotron-mamba2-ops.md",
    ),
    "muse_v_contiguous": CompatHook(
        name="muse_v_contiguous",
        kind="model_side_numeric_fix",
        target="tileinfer.bridge.models.muse_glimmer:...value.contiguous()",
        installer="",
        disable_env="",
        delete_when="vllm-ascend sliding-window prefill accepts a strided value",
        evidence="docs/notes/muse-glimmer-static.md",
        notes="Model-side fix, not a runtime hook; recorded so the workaround is not forgotten.",
    ),
}

_INSTALLED: Dict[str, InstallResult] = {}


def list_compat_hooks() -> List[Dict[str, Any]]:
    return [
        {
            "name": hook.name,
            "kind": hook.kind,
            "target": hook.target,
            "installer": hook.installer,
            "disable_env": hook.disable_env,
            "delete_when": hook.delete_when,
            "evidence": hook.evidence,
            "idempotent": hook.idempotent,
            "notes": hook.notes,
        }
        for hook in sorted(HOOKS.values(), key=lambda item: item.name)
    ]


def install_compat_hook(name: str) -> InstallResult:
    """Install one declared hook and return the common install result."""
    hook = HOOKS.get(name)
    if hook is None:
        return InstallResult(name=name, ok=False, detail=f"unknown compat hook {name!r}")
    if hook.disable_env and os.environ.get(hook.disable_env, "").strip().lower() in _TRUE:
        return InstallResult(
            name=name,
            ok=True,
            detail=f"disabled by {hook.disable_env}",
            extra={"disabled": True},
        )
    if not hook.installer:
        return InstallResult(
            name=name,
            ok=True,
            detail="metadata-only hook (model-side numeric fix)",
            extra={"disabled": False},
        )
    if hook.idempotent and name in _INSTALLED:
        cached = _INSTALLED[name]
        return InstallResult(
            name=name,
            ok=cached.ok,
            installed=cached.installed,
            skipped=cached.skipped,
            failed=cached.failed,
            detail="already installed",
            extra=dict(cached.extra),
        )

    module_name, _, actor = hook.installer.partition(":")
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001
        return InstallResult(
            name=name,
            ok=False,
            detail=f"cannot import {module_name!r}: {type(exc).__name__}: {exc}",
        )
    fn = getattr(module, actor, None)
    if not callable(fn):
        return InstallResult(
            name=name,
            ok=False,
            detail=f"{module_name!r} has no callable {actor!r}",
        )
    try:
        value = fn()
    except Exception as exc:  # noqa: BLE001
        return InstallResult(
            name=name,
            ok=False,
            detail=f"{hook.installer} raised {type(exc).__name__}: {exc}",
        )
    if value is False:
        return InstallResult(
            name=name,
            ok=False,
            detail=f"{hook.installer} returned False (target unavailable)",
        )
    result = InstallResult(
        name=name,
        ok=True,
        installed=[hook.installer],
        detail=f"installed {hook.installer}",
        extra={"disabled": False, "result": value if isinstance(value, Mapping) else None},
    )
    if hook.idempotent:
        _INSTALLED[name] = result
    return result


def reset_compat_hooks() -> None:
    """Test helper: forget which hooks ran in this process."""
    _INSTALLED.clear()


__all__ = [
    "CompatHook",
    "HOOKS",
    "install_compat_hook",
    "list_compat_hooks",
    "reset_compat_hooks",
]
