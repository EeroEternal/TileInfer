"""Config adapters: the model/config gap that must be closed before ``ModelConfig``.

A checkpoint can declare a model vLLM already ships and still fail before any weight is
read: an unknown ``model_type``, a nested text/vision config the strict validator rejects,
or a layer plan spelled as a list where vLLM expects a pattern string.  The manifest's
``config_impl`` names the actor that fixes this; this module resolves it and reports the
result, never raising from the plugin entry point.

Convention (the same shape as ``model_impl``):
    ``package.module``              -> calls ``register()``
    ``package.module:actor``        -> calls ``actor()``

The actor must be idempotent; it runs once per process and may also be registered as a
``model_impl.register_config`` actor during a transition.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict

_INSTALLED: Dict[str, Dict[str, Any]] = {}


def install_config_adapter(spec: str) -> Dict[str, Any]:
    """Run the manifest's config adapter and return a machine-readable result."""
    spec = str(spec or "").strip()
    info: Dict[str, Any] = {
        "spec": spec,
        "actor": "",
        "ok": False,
        "result": None,
        "detail": "",
    }
    if not spec:
        info.update(ok=True, detail="no config adapter requested")
        return info
    cached = _INSTALLED.get(spec)
    if cached is not None:
        return dict(cached, detail="already installed")

    module_name, _, actor = spec.partition(":")
    actor = actor or "register"
    info["actor"] = actor
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 - report, never raise from the entry point
        info["detail"] = f"cannot import {module_name!r}: {type(exc).__name__}: {exc}"
        return info
    fn = getattr(module, actor, None)
    if not callable(fn):
        info["detail"] = f"{module_name!r} has no callable {actor!r} actor"
        return info
    try:
        result = fn()
    except Exception as exc:  # noqa: BLE001
        info["detail"] = f"{spec} raised {type(exc).__name__}: {exc}"
        return info
    info.update(ok=True, result=result, detail=f"{spec} applied")
    _INSTALLED[spec] = dict(info)
    return info


def reset_config_adapters() -> None:
    """Test/doctor helper: forget which specs ran in this process."""
    _INSTALLED.clear()


__all__ = ["install_config_adapter", "reset_config_adapters"]
