"""Manifest registry: the only place a model id becomes an object.

Lookup accepts the arch name, the manifest id, or the human model name, because those are
the three strings a deployment script or a checkpoint naturally has.  An unknown string is
an explicit :class:`KeyError`, not a silent fallback to the first manifest - silently
serving the wrong declaration is how a KV-shape mismatch reaches the device.
"""

from __future__ import annotations

from typing import Dict

from ..contracts.manifest import ModelManifest
from .qwen3_6 import QWEN3_6

MANIFESTS: Dict[str, ModelManifest] = {
    QWEN3_6.manifest_id: QWEN3_6,
    QWEN3_6.arch: QWEN3_6,
    QWEN3_6.name: QWEN3_6,
}


def find_manifest(key: str) -> ModelManifest:
    """Resolve a manifest by id/arch/name, or raise with the known keys."""
    if key in MANIFESTS:
        return MANIFESTS[key]
    raise KeyError(f"no manifest for {key!r}; known: {sorted(MANIFESTS)}")
