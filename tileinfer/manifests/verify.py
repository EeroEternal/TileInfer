"""Compare a declaration with the checkpoint config it claims to describe.

The manifest is the decision layer's only input; a hand transcription error would otherwise
survive CPU tests and only show up as wrong layer geometry at load time.  This verifier is
model-agnostic and returns a list of concrete mismatches; model-specific audio/vision details
remain in model-specific tests.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence


def _text(raw: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = raw.get("text_config")
    return nested if isinstance(nested, Mapping) else raw


def _normalise_layer_types(raw: Mapping[str, Any]) -> Optional[Sequence[str]]:
    types = raw.get("layer_types")
    if isinstance(types, list):
        return tuple(types)
    blocks = raw.get("layers_block_type")
    if not isinstance(blocks, list):
        return None
    mapping = {"attention": "full_attention", "mamba": "mamba", "moe": "moe", "mlp": "mlp"}
    return tuple(mapping.get(str(item), str(item)) for item in blocks)


def verify_manifest(manifest: Any, raw_config: Mapping[str, Any]) -> List[str]:
    """Return every mismatch between ``manifest`` and one checkpoint config."""
    errors: List[str] = []
    text = _text(raw_config)

    archs = tuple(raw_config.get("architectures") or ())
    if archs and manifest.arch not in archs:
        errors.append(f"arch: manifest={manifest.arch!r} checkpoint={list(archs)!r}")
    model_type = raw_config.get("model_type") or text.get("model_type")
    if model_type and manifest.model_type != model_type:
        errors.append(f"model_type: manifest={manifest.model_type!r} checkpoint={model_type!r}")

    expected_layers = text.get("num_hidden_layers", raw_config.get("num_hidden_layers"))
    if expected_layers is not None and manifest.num_layers != int(expected_layers):
        errors.append(f"num_layers: manifest={manifest.num_layers} checkpoint={expected_layers}")

    layer_types = _normalise_layer_types(text)
    if layer_types is not None and tuple(manifest.layer_types) != tuple(layer_types):
        errors.append("layer_types: manifest and checkpoint differ")

    attn = getattr(manifest, "attention", None)
    for key in ("num_attention_heads", "num_key_value_heads", "head_dim"):
        value = text.get(key)
        field = {"num_attention_heads": "num_heads", "num_key_value_heads": "num_kv_heads"}.get(key, key)
        if value is not None and attn is not None and getattr(attn, field, value) != int(value):
            errors.append(f"{key}: manifest={getattr(attn, field)} checkpoint={value}")

    window = text.get("sliding_window")
    if window is not None and attn is not None and attn.sliding_window != int(window):
        errors.append(f"sliding_window: manifest={attn.sliding_window} checkpoint={window}")

    state = getattr(manifest, "state", None)
    if state is not None:
        kernel = text.get("conv_L_cache", text.get("conv_kernel"))
        if kernel is not None and state.kernel != int(kernel):
            errors.append(f"state.kernel: manifest={state.kernel} checkpoint={kernel}")

    return errors


__all__ = ["verify_manifest"]
