"""Resolve operator implementations from an external library instead of the local copies.

Why this file exists
--------------------
``AscendAttention`` is the engine-agnostic home of these operators: same math, no vLLM concepts, a
declarative spec and a measured ledger.  The direction of the dependency is supposed to be
**one-way** (this repo -> that library), so this module is the *only* place allowed to know the other
package exists.  It is tiny, it is inert by default, and it falls back on any error: a missing library,
a missing symbol or a version mismatch must degrade to the local implementation, never to a crash.

The switch
----------
``TILEINFER_OPS_LIB=ascend_attention`` turns it on.  Unset/empty means *off*: the shipping behaviour
is this repo's own operators, byte for byte.  Note the timing difference from ``TILEINFER_OPS``: the
family switch is consulted per call inside a wrapper, while this binding replaces module attributes at
import, so flipping it needs a **process restart** (same as this repo's other import-time decisions).

What may be bound
-----------------
Only names whose semantics are identical in both repos, listed in :data:`BINDABLE`.  Anything that
depends on a vLLM ABI shape (``fused_experts``, ``topk_softmax``, the takeovers in ``install.py``)
stays local: borrowing an implementation that does not answer the same question would be a rewrite
disguised as a refactor.

Verification
------------
``tests/test_ascend_attention_bridge.py`` (in the AscendAttention repo) compares both implementations
bitwise for the bound names, and ``tests/test_tileinfer_equivalence.py`` there does the same from the
other side.  This module's own ``report()`` is what the test prints when it fails.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

__all__ = ["BINDABLE", "LIB_ENV", "apply", "enabled", "library", "report", "resolve"]

logger = logging.getLogger(__name__)

#: ``TILEINFER_OPS_LIB=ascend_attention`` selects the external library.
LIB_ENV = "TILEINFER_OPS_LIB"
LIB_NAME = "ascend_attention"

#: Local name -> (module inside the external library, attribute).
BINDABLE: Dict[str, Tuple[str, str]] = {
    "chunk_local_cumsum": ("ascend_attention.ops.gdn", "chunk_local_cumsum"),
    "l2norm_fwd": ("ascend_attention.ops.gdn", "l2norm_fwd"),
}

_TRUTHY = {"1", "true", "yes", "on", "ascend_attention"}
_LOGGED = False


def enabled() -> bool:
    """Is the external library selected?  Per-call read; unset means off."""
    raw = os.environ.get(LIB_ENV)
    if raw is None:
        return False
    return raw.strip().lower() in _TRUTHY


def _ensure_importable() -> None:
    """Let a sibling checkout be used without installing it (development only, and logged once)."""
    global _LOGGED
    try:
        importlib.import_module(LIB_NAME)
        return
    except Exception:  # noqa: BLE001 - fall through to the sibling path
        pass
    # extern.py lives at <repo>/tileinfer/vllm/ops/, so walk up instead of hard-coding a depth that a
    # file move would silently break.
    candidate = None
    for parent in Path(__file__).resolve().parents:
        trial = parent / "AscendAttention" / "src"
        if trial.is_dir():
            candidate = trial
            break
    if candidate is None:
        return
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
    if not _LOGGED:
        logger.warning("%s: using the sibling checkout at %s", LIB_ENV, candidate)
        _LOGGED = True


def resolve(name: str) -> Optional[Any]:
    """The external implementation of ``name``, or ``None`` when it cannot be used here."""
    if not enabled() or name not in BINDABLE:
        return None
    module_name, attribute = BINDABLE[name]
    try:
        _ensure_importable()
        return getattr(importlib.import_module(module_name), attribute)
    except Exception as exc:  # noqa: BLE001 - the local copy is always a valid answer
        logger.warning("%s: %r unavailable (%s: %s); keeping the local implementation",
                       LIB_ENV, name, type(exc).__name__, exc)
        return None


def apply(namespace: Mapping[str, Any], names: Tuple[str, ...]) -> Dict[str, str]:
    """Rebind ``names`` inside ``namespace`` (a module ``globals()``) when the switch is on.

    Returns the mapping of ``name -> "external" | "local"`` actually applied, which is what
    :func:`report` prints and what a test compares.
    """
    applied: Dict[str, str] = {}
    for name in names:
        replacement = resolve(name)
        if replacement is None:
            applied[name] = "local"
            continue
        # ``namespace`` is a live module globals dict; the assignment is the whole patch.
        namespace[name] = replacement  # type: ignore[index]
        applied[name] = "external"
    return applied


def report() -> Dict[str, Any]:
    """What the bridge would do right now, for logs and for the equivalence test."""
    return {
        "env": LIB_ENV,
        "raw": os.environ.get(LIB_ENV),
        "enabled": enabled(),
        "bindable": {name: value[1] for name, value in BINDABLE.items()},
    }
