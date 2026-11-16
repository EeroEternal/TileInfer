"""Deprecated v2 location; the implementation moved to :mod:`tileinfer.providers.moe`.

Kept only as a compatibility shim for old v2 call sites/tests.  New code must import the
provider module directly; v2 is frozen and must not gain new behavior.
"""

from __future__ import annotations

import importlib as _importlib

_impl = _importlib.import_module("tileinfer.providers.moe")
globals().update({k: v for k, v in vars(_impl).items() if not k.startswith("__")})
try:
    __all__ = list(_impl.__all__)
except AttributeError:
    pass
