"""Transformers compatibility shims for the LFM3 checkpoint.

The LFM3 remote config declares ``conv`` layers and per-layer ``rope_parameters``.  Newer
transformers releases remap ``conv`` to ``linear_attention`` and add scalar entries to the
per-layer rope dict; both break a model whose own code keys off ``conv``.  These shims are
installed by the bridge before vLLM reads the model config and are idempotent.
"""

from __future__ import annotations


def apply() -> None:
    import transformers.configuration_utils as configuration_utils
    import transformers.modeling_rope_utils as rope_utils

    if getattr(configuration_utils, "_tileinfer_lfm3_compat", False):
        return

    if hasattr(configuration_utils, "remap_legacy_layer_types"):
        configuration_utils.remap_legacy_layer_types = lambda layers: [
            "linear_attention" if layer == "mamba" else "full_attention" if layer == "attention" else layer
            for layer in layers
        ]

    original_convert = rope_utils.RotaryEmbeddingConfigMixin.convert_rope_params_to_dict

    def convert(self, **kwargs):
        rope_params = getattr(self, "rope_parameters", None)
        layer_types = getattr(self, "layer_types", None)
        if isinstance(rope_params, dict) and layer_types and "conv" in layer_types:
            if "conv" not in rope_params:
                for key in ("sliding_attention", "full_attention"):
                    if isinstance(rope_params.get(key), dict):
                        rope_params["conv"] = dict(rope_params[key])
                        break
        result = original_convert(self, **kwargs)
        rope_params = getattr(self, "rope_parameters", None)
        if isinstance(rope_params, dict):
            for key in [k for k, v in rope_params.items() if v is not None and not isinstance(v, dict)]:
                rope_params.pop(key, None)
        return result

    rope_utils.RotaryEmbeddingConfigMixin.convert_rope_params_to_dict = convert

    original_validate = rope_utils.RotaryEmbeddingConfigMixin.validate_rope

    def validate(self):
        rope_params = getattr(self, "rope_parameters", None)
        if isinstance(rope_params, dict):
            for key in [k for k, v in rope_params.items() if v is not None and not isinstance(v, dict)]:
                rope_params.pop(key, None)
        return original_validate(self)

    rope_utils.RotaryEmbeddingConfigMixin.validate_rope = validate
    configuration_utils._tileinfer_lfm3_compat = True


__all__ = ["apply"]
