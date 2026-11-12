"""L1 - one declaration per model.

Adding a model is adding a module here and one registry line; it is never a change to the
decision layer.  The declaration says what the model needs; it never says who provides it.
"""

from __future__ import annotations

from .qwen3_6 import LAYER_TYPES, QWEN3_6
from .registry import MANIFESTS, find_manifest

__all__ = ["MANIFESTS", "QWEN3_6", "LAYER_TYPES", "find_manifest"]
