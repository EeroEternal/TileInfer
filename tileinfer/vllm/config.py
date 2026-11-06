"""Minimal ``PretrainedConfig`` shim so vLLM can parse this checkpoint.

vLLM reads ``config.json`` through ``transformers.AutoConfig`` before any of our code runs, and the
checkpoint declares ``model_type: deepseek_v41``, which no released transformers knows.  Registering a
trivial subclass under that name (``AutoConfig.register``) is enough for the loader: every model key
lands on the config object untouched, and vLLM then dispatches on the *architecture* -- which the plugin
already registers as ``DeepseekV41ForCausalLM``.
"""

from __future__ import annotations

from typing import Any, Dict

from transformers import PretrainedConfig

__all__ = ["Dsv41HFConfig", "register_config"]

MODEL_TYPE = "deepseek_v41"


class Dsv41HFConfig(PretrainedConfig):
    """Carries the checkpoint's keys verbatim, turning the nested blocks into configs.

    The checkpoint keeps the architecture inside ``text_config`` (59 keys, already in HF spelling:
    ``num_attention_heads``/``num_hidden_layers``/``hidden_size``/…), plus ``vision_config`` and
    ``quantization_config``.  ``PretrainedConfig`` would store those as plain dicts, and vLLM's
    ``get_text_config()`` then fails with "does not have `num_attention_heads`" -- so the sub-blocks are
    promoted to config objects here, which is what transformers does for real checkpoints.
    """

    model_type = MODEL_TYPE

    def __init__(self, **kwargs: Any) -> None:
        for key in ("text_config", "vision_config"):
            sub = kwargs.get(key)
            if isinstance(sub, dict):
                kwargs[key] = PretrainedConfig(**sub)
        # config.json carries no `architectures` field, and vLLM needs it to decide that this is a
        # generative model: without it, ModelConfig rejects `--runner generate`.
        kwargs.setdefault("architectures", ["DeepseekV41ForCausalLM"])
        super().__init__(**kwargs)


def register_config() -> Dict[str, str]:
    """Idempotently teach ``AutoConfig`` the ``deepseek_v41`` name."""
    from transformers import AutoConfig

    existing = getattr(AutoConfig, "_mapping", {}).get(MODEL_TYPE)
    if existing is Dsv41HFConfig:
        return {"config": "already registered", "model_type": MODEL_TYPE}
    AutoConfig.register(MODEL_TYPE, Dsv41HFConfig, exist_ok=True)
    return {"config": "registered", "model_type": MODEL_TYPE}
