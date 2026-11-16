"""Config-class aliases and raw-checkpoint normalisation that vLLM 0.23 needs (v3 bridge).

This module is imported by the general-plugin path before ``ModelConfig`` is built.  It only
registers a class that vLLM itself ships; it does not implement a model, and it never raises.

Two separate upstream gaps are handled here:

* vLLM 0.23 ships ``NemotronHConfig`` / ``NemotronHForCausalLM`` but does not put
  ``nemotron_h`` in its AutoConfig map, so the first ``ModelConfig`` fails before any model code.
* Nemotron checkpoints export the layer plan as ``layers_block_type`` (a list such as
  ``["mamba", "moe", ...]``) and ``mtp_layers_block_type``.  vLLM's class exposes
  ``layers_block_type`` as a read-only property derived from ``hybrid_override_pattern``; passing
  the raw key to ``from_dict`` therefore raises ``AttributeError: property ... has no setter``.
  The subclass below converts the list into the pattern the class expects and drops the raw keys.
  This is a checkpoint-format adapter, not an upstream edit.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_BLOCK_CHARS = {"mamba": "M", "attention": "*", "mlp": "-", "moe": "E"}


def _pattern_from_block_types(block_types: Any) -> str:
    """Convert a checkpoint layer-plan list to vLLM's ``[M*-E]`` pattern string."""
    if not isinstance(block_types, (list, tuple)):
        raise ValueError(f"layer plan must be a list, got {type(block_types).__name__}")
    chars: List[str] = []
    for item in block_types:
        key = str(item).lower()
        if key not in _BLOCK_CHARS:
            raise ValueError(f"unknown layer block type {item!r}; known: {sorted(_BLOCK_CHARS)}")
        chars.append(_BLOCK_CHARS[key])
    return "".join(chars)


def normalize_nemotron_config(data: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of a raw Nemotron config with list layer plans converted to patterns."""
    out = dict(data)
    if "layers_block_type" in out:
        value = out.pop("layers_block_type")
        pattern = _pattern_from_block_types(value)
        existing = out.get("hybrid_override_pattern")
        if existing and existing != pattern:
            raise ValueError(
                "checkpoint has both layers_block_type and a different hybrid_override_pattern"
            )
        out["hybrid_override_pattern"] = pattern
    if "mtp_layers_block_type" in out:
        value = out.pop("mtp_layers_block_type")
        pattern = _pattern_from_block_types(value)
        existing = out.get("mtp_hybrid_override_pattern")
        if existing and existing != pattern:
            raise ValueError(
                "checkpoint has both mtp_layers_block_type and a different "
                "mtp_hybrid_override_pattern"
            )
        out["mtp_hybrid_override_pattern"] = pattern
    return out


def register_nemotron_h_alias() -> Dict[str, Any]:
    """Make ``model_type="nemotron_h"`` resolvable, with raw layer-plan normalisation."""
    info: Dict[str, Any] = {
        "name": "nemotron_h",
        "registered": False,
        "normalized": False,
        "detail": "",
    }
    try:
        from transformers import AutoConfig
        from vllm.transformers_utils.configs.nemotron_h import (
            NemotronHConfig as VendorNemotronHConfig,
        )
    except Exception as exc:  # noqa: BLE001 - CPU/non-Nemotron environments are fine
        info["detail"] = f"not registered: imports unavailable ({exc!r})"
        return info

    class NemotronHConfig(VendorNemotronHConfig):
        """Vendor class plus checkpoint-format normalisation; deliberately same class name."""

        @classmethod
        def from_dict(cls, config_dict: Dict[str, Any], **kwargs: Any):
            return super().from_dict(normalize_nemotron_config(config_dict), **kwargs)

        def __init__(self, **kwargs: Any):
            super().__init__(**normalize_nemotron_config(kwargs))

    NemotronHConfig.__name__ = "NemotronHConfig"
    NemotronHConfig.__qualname__ = "NemotronHConfig"

    try:
        NemotronHConfig.model_type = "nemotron_h"
        AutoConfig.register("nemotron_h", NemotronHConfig, exist_ok=True)
        info["registered"] = True
        info["normalized"] = True
        info["detail"] = (
            "AutoConfig['nemotron_h'] -> vLLM NemotronHConfig; "
            "layers_block_type/mtp_layers_block_type normalised to patterns"
        )
        logger.info("TileInfer: registered AutoConfig alias nemotron_h -> NemotronHConfig")
    except Exception as exc:  # noqa: BLE001 - registration must not abort boot
        info["detail"] = f"not registered: {exc!r}"
        logger.warning("TileInfer: could not register nemotron_h config alias: %r", exc)
    return info
