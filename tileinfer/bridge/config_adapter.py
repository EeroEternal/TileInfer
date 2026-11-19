"""Config adapters: the model/config gap that must be closed before ``ModelConfig``.

A checkpoint can declare a model vLLM already ships and still fail before any weight is
read: an unknown ``model_type``, a nested text/vision config the strict validator rejects,
or a layer plan spelled as a list where vLLM expects a pattern string.  The manifest's
``config_impl`` names the actor that fixes this; this module resolves it and returns the
common install result, never raising from the plugin entry point.

Convention (the same shape as ``model_impl``):
    ``package.module``              -> calls ``register()``
    ``package.module:actor``        -> calls ``actor()``

The actor must be idempotent; it runs once per process and may also be registered as a
``model_impl.register_config`` actor during a transition.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict

from .results import InstallResult

_INSTALLED: Dict[str, InstallResult] = {}


def install_config_adapter(spec: str) -> InstallResult:
    """Run the manifest's config adapter and return a common install result."""
    spec = str(spec or "").strip()
    if not spec:
        return InstallResult(name="config_adapter", ok=True, detail="no config adapter requested")
    cached = _INSTALLED.get(spec)
    if cached is not None:
        return InstallResult(
            name="config_adapter",
            ok=cached.ok,
            detail="already installed",
            extra={"result": cached.extra.get("result"), "actor": cached.extra.get("actor")},
        )

    module_name, _, actor = spec.partition(":")
    actor = actor or "register"
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 - report, never raise from the entry point
        return InstallResult(
            name="config_adapter",
            ok=False,
            detail=f"cannot import {module_name!r}: {type(exc).__name__}: {exc}",
            extra={"actor": actor},
        )
    fn = getattr(module, actor, None)
    if not callable(fn):
        return InstallResult(
            name="config_adapter",
            ok=False,
            detail=f"{module_name!r} has no callable {actor!r} actor",
            extra={"actor": actor},
        )
    try:
        value = fn()
    except Exception as exc:  # noqa: BLE001
        return InstallResult(
            name="config_adapter",
            ok=False,
            detail=f"{spec} raised {type(exc).__name__}: {exc}",
            extra={"actor": actor},
        )
    result = InstallResult(
        name="config_adapter",
        ok=True,
        detail=f"{spec} applied",
        extra={"result": value, "actor": actor},
    )
    _INSTALLED[spec] = result
    return result


def reset_config_adapters() -> None:
    """Test/doctor helper: forget which specs ran in this process."""
    _INSTALLED.clear()


__all__ = ["install_config_adapter", "reset_config_adapters"]
