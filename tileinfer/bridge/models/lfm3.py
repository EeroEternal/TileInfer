"""``tileinfer.bridge.models.lfm3`` — route-O registration for LFM3.

LFM3 has no upstream vLLM architecture, so the declaration names this module as its
``model_impl``.  The bridge calls two actors before vLLM reads the checkpoint config:

* :func:`register_config` — install the transformers compatibility shims and disable the
  vllm-ascend global RoPE buffer, which cannot represent LFM3's two rotary dimensions.
* :func:`register_models` — register ``Lfm3ForCausalLM`` in vLLM's ``ModelRegistry`` by
  class path, so the parameter tree is imported only when such a checkpoint is resolved.

The model implementation itself stays in ``tileinfer.models.lfm3``; that package is
engine-free by rule, so it must not be imported at module import time here.
"""

from __future__ import annotations

from typing import Any, Dict

MODEL_ARCH = "Lfm3ForCausalLM"
MODEL_IMPL = "tileinfer.models.lfm3.model:Lfm3ForCausalLM"


def _patch_ascend_rope() -> str:
    """Disable vllm-ascend's single global cos/sin buffer for LFM3's two rotary dims.

    The Ascend GQA rope path keeps one process-wide buffer sized from a model-wide
    ``partial_rotary_factor``.  LFM3 has 64 rotary dims on full-attention layers and 128 on
    sliding layers, so the global update cannot be correct; the model uses a native
    per-layer implementation instead.  A missing vllm-ascend is not an error here: this
    actor also runs in CPU-only plan/doctor tests.
    """
    try:
        import vllm_ascend.ops.rotary_embedding as ascend_rope

        ascend_rope.update_cos_sin = lambda positions: None
        ascend_rope.set_cos_and_sin = lambda *args, **kwargs: None
    except Exception:
        return "unavailable"
    try:
        import vllm_ascend.worker.model_runner_v1 as model_runner

        if hasattr(model_runner, "update_cos_sin"):
            model_runner.update_cos_sin = lambda positions: None
    except Exception:
        pass
    return "applied"


def register_config() -> Dict[str, str]:
    """Install the LFM3 transformers/rope compatibility shims (idempotent)."""
    from ...models.lfm3.compat import apply as apply_lfm3_compat

    apply_lfm3_compat()
    return {
        "model_type": "lfm3",
        "transformers_compat": "applied",
        "ascend_rope_patch": _patch_ascend_rope(),
    }


def register_models(registry: Any = None) -> Dict[str, object]:
    """Register ``Lfm3ForCausalLM`` by class path; never abort a vendor-served boot."""
    info: Dict[str, object] = {
        "arch": MODEL_ARCH,
        "class_path": MODEL_IMPL,
        "registered": False,
        "detail": "",
    }
    details = []
    try:
        details.append(str(register_config()))
    except Exception as exc:  # noqa: BLE001 - optional compatibility aid
        details.append(f"config registration raised {exc!r}")
    try:
        if registry is None:
            from vllm.model_executor.models.registry import ModelRegistry

            registry = ModelRegistry
        registry.register_model(MODEL_ARCH, MODEL_IMPL)
        info["registered"] = True
        details.append(f"registered {MODEL_ARCH} -> {MODEL_IMPL}")
    except Exception as exc:  # noqa: BLE001 - a plugin step must never abort the boot
        details.append(f"registration failed: {exc!r}")
    info["detail"] = "; ".join(details)
    return info


__all__ = ["MODEL_ARCH", "MODEL_IMPL", "register_config", "register_models"]
