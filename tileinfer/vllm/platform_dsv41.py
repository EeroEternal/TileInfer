"""Platform for DeepSeek-V4.1-Flash: the TileInfer Ascend platform with fp8/fp4 admitted.

``TileInferPlatform`` (the Qwen3.6 arm) deliberately rejects every quantisation method -- v0 of that
backend is bf16/fp16 only, and ``platform.py`` documents why.  DeepSeek-V4.1-Flash is different by
construction: the released checkpoint is fp8-block-32 for everything dense and **fp4 (e4m3 scales) for
the routed experts**, and the S1-S5 stack reads it exactly that way (``docs/notes/dsv41-facts.md`` §3).

Rather than weaken the shared gate -- which would change the Qwen3.6 arm's contract, C2 -- this module
subclasses it and overrides only :meth:`verify_quantization`.  The plugin picks this class when
``VLLM_DSV41=1`` is set (``tileinfer.vllm.plugin.register`` stays the same entry point, so no packaging
change is needed), and keeps returning the original platform otherwise.

Nothing else is overridden: device handling, attention backend selection and memory accounting are
inherited unchanged.
"""

from __future__ import annotations

from typing import Any

from tileinfer.vllm.platform import NO_QUANT_SPELLINGS, TileInferPlatform

__all__ = ["Dsv41Platform", "SUPPORTED_QUANTISATION"]

#: What this checkpoint actually contains (dense fp8-block-32, routed experts fp4, e8m0/ue8m0 scales).
SUPPORTED_QUANTISATION = (
    "fp8",
    "fp8_e4m3",
    "fp8_w8a8",
    "fp4",
    "mxfp4",
    "deepseek_v41",
    "compressed-tensors",
)


class Dsv41Platform(TileInferPlatform):
    """``TileInferPlatform`` with the quantisation gate opened for this checkpoint's formats."""

    @classmethod
    def support_hybrid_kv_cache(cls) -> bool:
        """False: this model has no hybrid (GDN + attention) layer mix.

        The Qwen3.6 arm does have one, and its platform answers True.  vLLM then picks the hybrid KV
        coordinator, which asserts ``at least two attention groups`` -- our spec is empty (no
        ``AttentionLayerBase`` layers at all), so the plain coordinator is the right one.
        """
        return False

    @classmethod
    def verify_quantization(cls, quant: Any) -> None:
        """Accept the formats the checkpoint is stored in; keep rejecting anything else.

        The base class rejects everything except the "no quantisation" spellings, which is the right
        contract for the bf16-only Qwen3.6 path.  Here the *weights* are fp8/fp4 by design, so the
        accepted set is exactly :data:`SUPPORTED_QUANTISATION` plus those spellings.
        """
        name = "" if quant is None else str(quant).strip().lower()
        if name in NO_QUANT_SPELLINGS:
            return
        if name in SUPPORTED_QUANTISATION:
            return
        raise ValueError(
            f"quantisation {quant!r} is not supported for DeepSeek-V4.1-Flash; the checkpoint is "
            f"fp8-block-32 (dense) with fp4 routed experts, so the accepted spellings are "
            f"{SUPPORTED_QUANTISATION} (docs/notes/dsv41-facts.md §3)."
        )
