"""Deprecated v2 location; moved to :mod:`tileinfer.bridge.nemotron_config`.

The bridge owns checkpoint-format compatibility.  This shim exists only for old imports;
new code must use the bridge module; v2 is frozen.
"""

from __future__ import annotations

import importlib as _importlib

_impl = _importlib.import_module("tileinfer.bridge.nemotron_config")
globals().update({k: v for k, v in vars(_impl).items() if not k.startswith("__")})
try:
    __all__ = list(_impl.__all__)
except AttributeError:
    pass
