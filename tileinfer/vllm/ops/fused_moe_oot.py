"""Deprecated v2 location; moved to :mod:`tileinfer.providers.fused_moe_oot`."""

from __future__ import annotations

import importlib as _importlib

_impl = _importlib.import_module("tileinfer.providers.fused_moe_oot")
globals().update({k: v for k, v in vars(_impl).items() if not k.startswith("__")})
try:
    __all__ = list(_impl.__all__)
except AttributeError:
    pass
