"""Model profile for MuseGlimmer (``MuseGlimmerForConditionalGeneration``, text-only build).

The framework asks for a profile by architecture (``register_profile`` in
``tileinfer/vllm/profiles/base.py``); without one, ``VllmConfig`` refuses to start with
"no TileInfer model profile matches ['MuseGlimmerForConditionalGeneration']".

Checkpoint facts (``/home/models/Muse-Glimmer-30B/config.json`` ``text_config``, and the
backport docstring in ``tileinfer/vllm/models/muse_glimmer.py``): 52 dense layers, hidden 6656,
32 query / 2 KV heads (GQA group 16) at head_dim 128, SiLU-gated MLP, untied lm_head.

**Sliding windows (iRoPE)**: the checkpoint mixes 39 sliding-window and 13 full-attention
layers, so ``mixed_attention_specs=True`` puts the engine in the per-group mode: vLLM builds a
``FullAttentionSpec`` group (13 layers) and a ``SlidingWindowSpec`` group (39 layers), each with
its own pool, block table and slot mapping, and the sliding-window allocator keeps only the
in-window blocks real (out-of-window table entries point at the reserved NULL block, id 0).
The window itself lives in the **compute**: those 39 layers run on the attention backend's
reference path (``window_left`` masking over absolute positions), while the full-attention
layers use the paged kernels.  Consequences: windowed decode is eager-only (``--enforce-eager``;
the builder downgrades graph capture for a windowed spec), and the reference gather walks the
full-length table (null pages included, masked out) — a bandwidth cost, not a correctness one.
"""

from __future__ import annotations

from typing import Tuple

from tileinfer.serving.buckets import KERNEL_PAGE_SIZE
from tileinfer.vllm.profiles.base import (
    BlockSizePolicy,
    KvShape,
    ModelProfile,
    register_profile,
)

__all__ = ["NAME", "MUSE_GLIMMER", "fused_kv_shape"]

NAME = "muse_glimmer"

ARCHS: Tuple[str, ...] = ("MuseGlimmerForConditionalGeneration",)

#: 32 query heads x head_dim 128; the ViT tower's heads are a separate, phase-2 path.
HEAD_SIZE: Tuple[int, ...] = (128,)

#: 32 query heads share 2 KV heads -> every head reads a 16-way GQA group.
GROUP_MIN = 16


def fused_kv_shape(
    num_blocks: int, block_size: int, num_kv_heads: int, head_size: int
) -> Tuple[int, ...]:
    """Block-first fused NHD: ``(num_blocks, 2, block_size, num_kv_heads, head_size)``."""
    return (int(num_blocks), 2, int(block_size), int(num_kv_heads), int(head_size))


MUSE_GLIMMER = register_profile(
    ModelProfile(
        name=NAME,
        archs=ARCHS,
        head_size=HEAD_SIZE,
        group_min=GROUP_MIN,
        block_size_policy=BlockSizePolicy(
            kernel_page_size=int(KERNEL_PAGE_SIZE),
            hybrid_align=None,
        ),
        kv_shape=KvShape(
            shape_fn=fused_kv_shape,
            stride_order=(1, 0, 2, 3, 4),
            block_dim=0,
        ),
        hybrid=None,
        op_family="dense",
        unsupported_features=(),
        notes=(
            "Text-only backport of vLLM main's muse_glimmer (transformers 5.5.4 lacks the "
            "model_type; vLLM 0.23.0 lacks the architecture). Sliding windows compute on the "
            "backend's reference path (eager only); mixed_attention_specs puts the engine in "
            "the per-group mode so the sliding-window allocator drops out-of-window blocks."
        ),
        mixed_attention_specs=True,
    )
)
