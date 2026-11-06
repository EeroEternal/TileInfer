"""Model profile for DeepSeek-V4.1-Flash (``DeepseekV41ForCausalLM``).

The framework asks for a profile by architecture (``register_profile`` in
``tileinfer/vllm/profiles/base.py``); without one, ``VllmConfig`` refuses to start with
"no TileInfer model profile matches ['DeepseekV41ForCausalLM']".

This model does **not** use vLLM's paged KV cache: ``DeepseekV41ForCausalLM.forward`` ignores
``kv_caches``/``attn_metadata`` and drives the standalone stack, whose window / compressed-KV / index
buffers live in :class:`tileinfer.models.dsv41.block.ModelState` (1M context = 13.27 GiB measured,
``docs/notes/dsv41-s4-offload-plan.md`` §5).  So the KV geometry below only has to be *valid*, not
sizeable: the model exposes no ``AttentionLayerBase`` layers, ``get_kv_cache_spec()`` is therefore
empty, and vLLM allocates nothing (``vllm/v1/worker/gpu_model_runner.py:7440`` documents that layers
without KV cache are simply absent).

Numbers come from the checkpoint (``docs/notes/dsv41-facts.md`` §2): 64 heads x head_dim 512, MLA with
``num_key_value_heads = 1``, 40 layers.
"""

from __future__ import annotations

from typing import Any, Tuple

from .base import BlockSizePolicy, KvShape, ModelProfile, register_profile

NAME = "dsv41"
ARCHS: Tuple[str, ...] = ("DeepseekV41ForCausalLM",)

#: 64 heads x 512 per head (facts §2); the indexer's 128 is a separate, non-attention path.
HEAD_SIZE: Tuple[int, ...] = (512,)

#: MLA: every query head shares the single KV head, so the GQA floor is 1.
GROUP_MIN = 1


def _kv_shape(blocks: int, page: int, heads: int, dim: int) -> Tuple[int, ...]:
    """Identity geometry -- the buffers are allocated by us, never by vLLM's cache allocator.

    ``KvShape`` requires a 5-D block declaration (``stride_order`` must permute 0..4), so the extra
    axis is a size-1 placeholder: nothing reads it because no paged cache is ever allocated.
    """
    return (int(blocks), int(page), int(heads), int(dim), 1)


DSV41 = register_profile(
    ModelProfile(
        name=NAME,
        archs=ARCHS,
        head_size=HEAD_SIZE,
        group_min=GROUP_MIN,
        block_size_policy=BlockSizePolicy(kernel_page_size=16, hybrid_align=16),
        kv_shape=KvShape(shape_fn=_kv_shape, stride_order=(0, 1, 2, 3, 4), block_dim=0),
        hybrid=None,
        op_family="gdn-moe",
        unsupported_features=(),
        notes=(
            "fp8-block-32 dense weights + fp4 routed experts; KV state is held by the model itself, "
            "so no paged cache is allocated and the geometry above is inert."
        ),
    )
)
