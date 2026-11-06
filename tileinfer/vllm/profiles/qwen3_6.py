"""The Qwen3.6 model profile - the exact behaviour the port had before the profile layer existed.

This is the **only** profile for now, and it is transcribed value-by-value from the code it
replaced, so the refactor is a pure move:

=========================================  ==================================================
fact                                       value (and where it used to live)
=========================================  ==================================================
architectures                              ``Qwen3_5*`` (the registry names)
head sizes                                 128 / 256 (``attn/backend.py``)
GQA group floor                            2 (``attn/backend.py:check_head_layout``)
kernel page                                128 (``serving.buckets.KERNEL_PAGE_SIZE``)
hybrid page                                2048 (``platform.HYBRID_BLOCK_SIZE``)
fused KV shape                             ``(blocks, 2, page, heads, dim)`` block-first
stride order                               ``(1, 0, 2, 3, 4)`` (K/V outermost)
hybrid kind                                GDN; mamba types ``GDN_ATTN``; SSU ``MAMBA1/2``
state update                               ``fused_recurrent_gated_delta_rule_packed_decode``
op family                                  ``gdn-moe``
=========================================  ==================================================

The reject list keeps both wordings that existed: the platform's selector reasons
(``platform.SELECTOR_REJECT_REASONS``) and the attention backend's own sentences
(``TileInferAttentionBackend.validate_configuration``).  They were never identical, the tests pin
both, and the profile is the one place they now live.

The four config-level exclusions (speculative decoding, a non-``auto`` cache dtype, sleep mode,
MLA) are part of this profile too: they describe the feature set this model family is served with,
not a property of the TileInfer hardware, so a future profile can decide them differently.
"""

from __future__ import annotations

from typing import Tuple

from tileinfer.serving.buckets import KERNEL_PAGE_SIZE
from tileinfer.vllm.profiles.base import (
    BlockSizePolicy,
    HybridKind,
    KvShape,
    ModelProfile,
    UnsupportedFeature,
    register_profile,
)

__all__ = ["NAME", "QWEN3_6", "mamba_page_size_padded", "fused_kv_shape"]


#: Registry name; ``scripts/serve_qwen36.py`` pins this string on the generic launcher.
NAME = "qwen3.6"

#: vLLM's registered architecture spellings for the Qwen3.5/3.6 family.  The explicit names (rather
#: than one ``Qwen3_5*`` pattern) are what vLLM's ``_VLLM_MODELS`` table contains.
ARCHS: Tuple[str, ...] = (
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForConditionalGeneration",
    "Qwen3_5ForCausalLM",
    "Qwen3_5MoeForCausalLM",
)

#: The hybrid state page: one attention page per token is
#: ``2 (K/V) * num_kv_heads * head_dim * element_size`` bytes, times the engine page size.  This is
#: the formula ``platform._align_mamba_state_pages`` applied inline before the profile existed.
def mamba_page_size_padded(
    block_size: int, num_kv_heads: int, head_size: int, element_size: int
) -> int:
    return int(block_size) * 2 * int(num_kv_heads) * int(head_size) * int(element_size)


def fused_kv_shape(
    num_blocks: int, block_size: int, num_kv_heads: int, head_size: int
) -> Tuple[int, ...]:
    """Block-first fused NHD: ``(num_blocks, 2, block_size, num_kv_heads, head_size)``."""
    return (int(num_blocks), 2, int(block_size), int(num_kv_heads), int(head_size))


QWEN3_6 = register_profile(
    ModelProfile(
        name=NAME,
        archs=ARCHS,
        head_size=(128, 256),
        group_min=2,
        block_size_policy=BlockSizePolicy(
            kernel_page_size=int(KERNEL_PAGE_SIZE),
            hybrid_align=2048,
        ),
        kv_shape=KvShape(
            shape_fn=fused_kv_shape,
            stride_order=(1, 0, 2, 3, 4),
            block_dim=0,
        ),
        hybrid=HybridKind(
            name="gdn",
            mamba_types=("GDN_ATTN",),
            ssu_mamba_types=("MAMBA1", "MAMBA2"),
            page_size_padded=mamba_page_size_padded,
            state_update_op="fused_recurrent_gated_delta_rule_packed_decode",
            notes=(
                "vLLM's Mamba SSU backend is a no-op for mamba_type=GDN_ATTN "
                "(ssu_dispatch.py:202-208); the decode state update is the FLA packed-decode op "
                "bound by tileinfer.vllm.ops.install."
            ),
        ),
        op_family="gdn-moe",
        unsupported_features=(
            # ---- attention-selector booleans (platform tuple + backend sentence) --------------
            UnsupportedFeature(
                field="use_mla",
                reason=(
                    "MLA is out of v0 scope (plan §0); the paged kernels are GQA/MHA with an "
                    "explicit head dim and the backend declares is_mla() == False"
                ),
                backend_reason="MLA is not supported by the TileInfer attention backend",
            ),
            UnsupportedFeature(
                field="use_sparse",
                reason=(
                    "sparse attention is out of v0 scope; the backend declares is_sparse() == "
                    "False"
                ),
                backend_reason=(
                    "sparse attention is not supported by the TileInfer attention backend"
                ),
            ),
            UnsupportedFeature(
                field="has_sink",
                reason="attention sinks are not implemented (supports_sink() == False)",
                backend_reason=(
                    "attention sinks are not supported by the TileInfer attention backend"
                ),
            ),
            UnsupportedFeature(
                field="use_mm_prefix",
                reason=(
                    "partial-multimodal-token full attention is not implemented "
                    "(supports_mm_prefix() == False)"
                ),
                backend_reason="partial-multimodal-token full attention is not supported",
            ),
            UnsupportedFeature(
                field="use_per_head_quant_scales",
                reason=(
                    "quantised attention is out of v0 scope "
                    "(supports_per_head_quant_scales() == False)"
                ),
                backend_reason="per-head quant scales are not supported",
            ),
            UnsupportedFeature(
                field="use_non_causal",
                reason="the TileInfer page kernel is causal only (supports_non_causal() == False)",
                backend_reason="non-causal (bidirectional) attention is not supported",
            ),
            UnsupportedFeature(
                field="use_batch_invariant",
                kind="plain_selector",
                reason=(
                    "batch-invariant attention is not implemented "
                    "(supports_batch_invariance() == False)"
                ),
                backend_reason=(
                    "batch-invariant attention is not supported by the paged kernel"
                ),
            ),
            UnsupportedFeature(
                field="use_kv_connector",
                kind="plain_selector",
                reason=(
                    "a KV connector rewrites cache blocks behind the layer's back; v0 writes one "
                    "fused pool per process (supports_kv_connector() == False)"
                ),
                backend_reason=(
                    "KV connectors are not supported (the TileInfer impl writes the fused pool "
                    "itself)"
                ),
            ),
            # ---- the selector's non-boolean attn_type field -----------------------------------
            UnsupportedFeature(
                field="attn_type",
                kind="attn_type",
                reason=(
                    "only decoder self-attention is served (AttentionBackend.supports_attn_type "
                    "defaults to AttentionType.DECODER, v1/attention/backend.py:248-254); an "
                    "encoder / encoder-decoder layer needs a different mask contract"
                ),
            ),
            # ---- config-level exclusions (check_and_update_config) -----------------------------
            UnsupportedFeature(
                field="speculative_config",
                kind="config",
                reason="speculative decoding is out of v0 scope",
                path=("speculative_config",),
                owners=("VllmConfig",),
                predicate="is_not_none",
                why=(
                    "speculative decoding must be rejected, not ignored "
                    "(vllm/model_executor/models/qwen3_5.py:696-697, "
                    "vllm/v1/attention/backend.py:568-576)"
                ),
                message=(
                    "speculative decoding is not supported on the TileInfer Ascend platform "
                    "(speculative_config={value!r}). v0 has no draft model / MTP path "
                    "(docs/plan-vllm-native-backend.md §0: 投机解码 MTP/DSpark 砍掉). "
                    "Remove --speculative-config."
                ),
            ),
            UnsupportedFeature(
                field="cache_dtype",
                kind="config",
                reason="only an unquantised kv cache (the model dtype) is served",
                path=("cache_config", "cache_dtype"),
                owners=("VllmConfig", "CacheConfig"),
                predicate="not_in_cache_dtypes",
                why=(
                    "the kv-cache dtype must be 'auto' (interface.py:557-560 compares cache_config."
                    "cache_dtype against 'auto')"
                ),
                message=(
                    "kv-cache dtype {value!r} is not supported on the TileInfer Ascend platform "
                    "(only {supported!r}: unquantised, model dtype). The field is "
                    "CacheConfig.cache_dtype (interface.py:557-560), and v0 validates exactly one "
                    "kv-cache path - the TileInfer kernels write the fused NHD pool in the model "
                    "dtype."
                ),
            ),
            UnsupportedFeature(
                field="enable_sleep_mode",
                kind="config",
                reason="there is no NPU device memory-pool allocator for sleep mode",
                path=("model_config", "enable_sleep_mode"),
                owners=("VllmConfig", "ModelConfig"),
                predicate="is_truthy",
                why=(
                    "sleep mode must be rejected (gpu_worker.py:204-213 is the only local reader; "
                    "there is no NPU memory-pool allocator in v0)"
                ),
                message=(
                    "sleep mode (--enable-sleep-mode) is not supported on the TileInfer Ascend "
                    "platform: it needs a device memory-pool allocator (vllm.device_allocator) "
                    "that does not exist for NPU in v0, and is_sleep_mode_available() is False "
                    "for out-of-tree platforms (interface.py:195-200)."
                ),
            ),
            UnsupportedFeature(
                field="use_mla",
                kind="config",
                reason="the paged kernels implement GQA/MHA with an explicit head dimension",
                path=("model_config", "use_mla"),
                owners=("VllmConfig", "ModelConfig"),
                predicate="is_truthy",
                why="MLA must be rejected; interface.py:565 is the local reader of use_mla",
                message=(
                    "MLA (multi-head latent attention) is not supported on the TileInfer Ascend "
                    "platform: the paged kernels implement GQA/MHA with an explicit head "
                    "dimension (docs/plan-vllm-native-backend.md §0, 模型清单/MLA 砍掉). The "
                    "attention backend declares is_mla() == False "
                    "(tileinfer/vllm/attn/backend.py:125-127)."
                ),
            ),
        ),
        notes=(
            "Qwen3.6 is a 30 GDN (linear-attention) + 10 full-attention hybrid with 16 query / 2 "
            "KV heads at head_dim 256 and 128 for the GDN state. The 10/30 split comes from vLLM's "
            "own KV-cache specs, never from this profile; the profile only supplies the page "
            "geometry that makes one mamba state page equal one attention page."
        ),
    )
)
