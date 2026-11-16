"""Operator-family registry: one module per model family's vLLM takeover table.

``install.py`` owns the *mechanism* (planning, the per-call switch, the ``Patch`` writes).  A family
module owns the *data*: ``required_names()`` plus ``bind()``, as in :mod:`.gdn_moe`.  A
:class:`~tileinfer.vllm.profiles.ModelProfile` names its family through ``op_family`` (a
dash-separated name); ``install.py`` resolves that to a module here and iterates its bindings.

Adding a family is a new file in this package plus a ``register_family`` call (or the module
self-registering at its bottom, which works because the lazy import below runs after this package
is fully initialised).  No edit to ``install.py`` is needed.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any, Dict, Tuple

__all__ = [
    "FAMILIES",
    "register_family",
    "unregister_family",
    "family_for",
    "family_names",
]

#: ``name -> family module`` (module objects expose ``required_names()`` and ``bind()``).
FAMILIES: Dict[str, Any] = {}


def register_family(name: str, module: Any) -> Any:
    """Register ``module`` under ``name``; returns it (so a module can self-register)."""
    if not name:
        raise ValueError("an op family needs a non-empty name")
    for attribute in ("required_names", "bind"):
        if not callable(getattr(module, attribute, None)):
            raise TypeError(
                f"op family {name!r} must expose callable {attribute}(); got {module!r}"
            )
    FAMILIES[str(name)] = module
    return module


def unregister_family(name: str) -> Any:
    """Remove a family (tests / a plugin reload); returns the removed module."""
    try:
        return FAMILIES.pop(str(name))
    except KeyError:
        raise KeyError(f"no op family named {name!r}; known families: {family_names()}") from None


def family_names() -> Tuple[str, ...]:
    return tuple(FAMILIES)


def _module_name(name: str) -> str:
    return str(name).strip().replace("-", "_")


def family_for(name: str) -> Any:
    """Resolve ``name`` to a family module, importing ``tileinfer.vllm.ops.families.<name>`` if needed.

    The lazy import is what makes "a new model family is a new file" true: a profile whose
    ``op_family`` is ``foo-bar`` loads ``tileinfer.vllm.ops.families.foo_bar`` with no edit here.
    A module that wants to register itself calls ``register_family`` at its bottom; a module already
    registered explicitly (``gdn_moe``) is used directly.
    """
    key = str(name)
    module = FAMILIES.get(key)
    if module is not None:
        return module
    module_name = _module_name(key)
    try:
        imported = importlib.import_module(f"{__name__}.{module_name}")
    except ImportError as exc:
        raise KeyError(
            f"no op family named {key!r} and {__name__}.{module_name} is not importable "
            f"({exc!r}); known families: {family_names()}"
        ) from exc
    module = FAMILIES.get(key)
    if module is None:
        # A new family file that defines ``required_names`` + ``bind`` is enough: register it under
        # the profile's spelling (its ``NAME`` when it has one) so adding a family stays "a new file".
        candidate = sys.modules.get(f"{__name__}.{module_name}", imported)
        if callable(getattr(candidate, "required_names", None)) and callable(
            getattr(candidate, "bind", None)
        ):
            registered_name = getattr(candidate, "NAME", key)
            module = register_family(registered_name, candidate)
            if key != registered_name:
                FAMILIES[key] = module
        else:
            raise KeyError(
                f"op family module {__name__}.{module_name} exposes no required_names()/bind(); "
                f"known families: {family_names()}"
            )
    return module


# The one family shipped today.  Imported (not lazily discovered) because its name contains a dash
# and because an explicit registry line makes the shipped surface visible in this file.
from . import gdn_moe  # noqa: E402  (the import above registers nothing on its own)

register_family(gdn_moe.NAME, gdn_moe)
