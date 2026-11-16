"""Deprecated v2 location; the bridge adapter moved to :mod:`tileinfer.bridge.nemotron_moe`.

Kept only for old v2 imports/tests.  v2 is frozen; new code uses the bridge module.
"""

from __future__ import annotations

import importlib as _importlib

_impl = _importlib.import_module("tileinfer.bridge.nemotron_moe")
globals().update({k: v for k, v in vars(_impl).items() if not k.startswith("__")})
try:
    __all__ = list(_impl.__all__)
except AttributeError:
    pass
