"""KV-cache writes for the NHD paged pool.

Why this module exists
----------------------
A vLLM attention backend owns the cache write: ``AttentionImpl.forward`` receives this step's
``key``/``value`` plus the layer's ``kv_cache``, and it must place them at ``slot_mapping``
*before* attention reads the pool.  In the previous integration that step was inherited from
vLLM-Ascend's implementation (``torch_npu._npu_reshape_and_cache``); with vLLM-Ascend gone it has
to exist here.  It is the single most "invisible" missing piece of the port: without it attention
reads whatever the pool happened to contain, which is not a crash - it is quietly wrong output.

Why it is small
---------------
* **NHD is already flat.**  ``[blocks, block_size, kv_heads, dim]`` flattens to
  ``[blocks * block_size, kv_heads, dim]`` with no stride games, and the 128-token sub-page view
  the kernel reads is *the same memory* (a page of ``k * 128`` tokens is ``k`` consecutive
  sub-pages).  Decode, chunked prefill and the paged kernel therefore share one address space.
* **``slot_mapping`` is per token**, not per page or per request: one slot per request in decode,
  many slots per request in chunked prefill, identical code path.

Two pool layouts, one write
---------------------------
vLLM hands the layer a *fused* pool with the K/V factor in one dimension, and a hybrid run
(attention + mamba layers) rewrites its physical strides (``gpu_model_runner.py:7167-7199``).
The write therefore never assumes that a pool is flattenable:

* ``[blocks, block_size, heads, dim]`` contiguous -> the flat view (one ``index_put_`` per half,
  the steady-state path);
* the same shape with any other strides (block-major / K-V interleaved) -> a two-index scatter at
  ``(block, offset) = (slot // block_size, slot % block_size)``.  A *slot* is still a flat token
  index in engine-page space: that arithmetic is the contract, not the strides.

:func:`tileinfer.serving.buckets.fused_halves` owns "which axis of the fused tensor is the 2";
this module only consumes the halves.

TIming and capture safety
-------------------------
The write happens on the layer's stream, inside the same ``forward`` that launches the attention
kernel, which is the ordering the engine already relied on.  It performs no dtype conversion and no
host synchronisation, so it is legal inside an ACLGraph capture and on replay.  The contiguous
path allocates nothing at all; the strided path reuses two int64 scratch vectors per
``(device, num_tokens)`` (allocated on first sight, which is an eager warm-up step, never a new
size inside a captured graph).
:func:`check_slot_mapping` is the only function that reads values back and is meant for builder
time (and tests), never for the captured step.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from tileinfer.serving.buckets import fused_halves

__all__ = [
    "flat_pool",
    "write_kv_cache",
    "write_kv_cache_fused",
    "check_slot_mapping",
]

#: Scratch for the block/offset split of the strided write, keyed by ``(device, num_tokens)``.
#: See :func:`_slot_split`: the alternative (fresh ``div``/``remainder`` results every step) would
#: allocate inside the captured region.
_SLOT_SPLIT: Dict[Tuple[str, int], Tuple[torch.Tensor, torch.Tensor]] = {}


def flat_pool(pool: torch.Tensor) -> torch.Tensor:
    """``[blocks, block_size, kv_heads, dim]`` -> ``[blocks * block_size, kv_heads, dim]``.

    A view, not a copy: raising instead of falling back to ``reshape`` keeps the "the kernel and
    the engine see the same bytes" invariant, which the zero-copy sub-page view depends on.  A
    pool that is not contiguous is *not* rejected by the write path - :func:`write_kv_cache`
    scatters at ``(block, offset)`` instead - it is only rejected here, where a view is all that
    can be returned.
    """
    if pool.dim() != 4:
        raise ValueError(f"KV pool must be 4-D NHD, got shape {tuple(pool.shape)}")
    if not pool.is_contiguous():
        raise ValueError(
            "KV pool must be contiguous to take the flat view, got strides "
            f"{pool.stride()}; a strided pool (block-major / K-V interleaved) is written through "
            "write_kv_cache's (block, offset) scatter instead"
        )
    return pool.view(-1, pool.shape[2], pool.shape[3])


def _check_step_tensors(
    key: torch.Tensor,
    value: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
) -> None:
    """Shape/dtype/device checks only - never a value read (those would sync)."""
    if key.dim() != 3:
        raise ValueError(
            f"key must be [num_tokens, kv_heads, head_dim], got shape {tuple(key.shape)}"
        )
    if value.shape != key.shape:
        raise ValueError(
            f"value shape {tuple(value.shape)} must match key shape {tuple(key.shape)}"
        )
    if slot_mapping.dim() != 1:
        raise ValueError(
            f"slot_mapping must be 1-D, got shape {tuple(slot_mapping.shape)}"
        )
    if slot_mapping.dtype != torch.int64:
        # Deliberately not casting: an int32 -> int64 cast allocates, and this call is inside the
        # graph.  vLLM's slot mapping is int64; a change there must be handled in the builder.
        raise ValueError(
            f"slot_mapping must be int64 (got {slot_mapping.dtype}); converting here would "
            "allocate inside the captured step"
        )
    if key.shape[0] != slot_mapping.numel():
        raise ValueError(
            f"key has {key.shape[0]} tokens but slot_mapping has {slot_mapping.numel()} slots"
        )
    for name, pool in (("k_cache", k_cache), ("v_cache", v_cache)):
        if pool.dim() != 4:
            raise ValueError(
                f"{name} must be one half of the fused pool, [blocks, block_size, kv_heads, "
                f"head_dim], got shape {tuple(pool.shape)}"
            )
        if key.shape[1:] != pool.shape[2:]:
            raise ValueError(
                f"key trailing shape {tuple(key.shape[1:])} does not match the pool's "
                f"{tuple(pool.shape[2:])} (kv_heads, head_dim)"
            )
        if pool.shape[1] < 1:
            raise ValueError(f"{name} has a zero-length page")
    if key.dtype != k_cache.dtype:
        raise ValueError(
            f"key dtype {key.dtype} must match the pool dtype {k_cache.dtype}; casting would "
            "allocate and change rounding"
        )
    if value.dtype != v_cache.dtype:
        raise ValueError(
            f"value dtype {value.dtype} must match the pool dtype {v_cache.dtype}; casting would "
            "allocate and change rounding"
        )
    if key.device != k_cache.device:
        raise ValueError(f"key on {key.device} but pool on {k_cache.device}")
    if value.device != v_cache.device:
        raise ValueError(f"value on {value.device} but pool on {v_cache.device}")
    if value.device != key.device:
        raise ValueError(f"value on {value.device} but key on {key.device}")


def _slot_split(
    slot_mapping: torch.Tensor, block_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``slot -> (block, offset)`` into two reusable int64 buffers.

    ``block = slot // block_size``, ``offset = slot % block_size`` - the only arithmetic a slot
    needs, and the reason the strided path exists at all: in a block-major pool the byte offset of
    a token is ``block * stride(block) + offset * stride(token)``, which is *not* affine in the
    slot, so a single flat index cannot express it.

    Both results are written into cached scratch (``out=``) instead of being allocated: a fresh
    ``div``/``remainder`` per step would allocate two int64 vectors inside the captured region.
    The cache key includes ``num_tokens`` because ``out=`` requires the exact shape; a captured
    graph always replays the same size, so the allocation happens during eager warm-up only.
    The scratch is process-wide, like the plan cache: the step path is single-threaded per worker,
    which is the assumption that makes reuse safe.
    """
    if block_size < 1:
        raise ValueError(f"block_size must be positive, got {block_size}")
    key = (str(slot_mapping.device), int(slot_mapping.numel()))
    pair = _SLOT_SPLIT.get(key)
    if pair is None:
        pair = (torch.empty_like(slot_mapping), torch.empty_like(slot_mapping))
        _SLOT_SPLIT[key] = pair
    blocks, offsets = pair
    torch.div(slot_mapping, block_size, rounding_mode="floor", out=blocks)
    torch.remainder(slot_mapping, block_size, out=offsets)
    return blocks, offsets


def _write_strided(
    pool: torch.Tensor, rows: torch.Tensor, slot_mapping: torch.Tensor
) -> None:
    """Scatter ``rows`` into a ``[blocks, page, heads, dim]`` pool at ``(block, offset)``.

    Works for *any* strides: torch resolves the two leading indices through the tensor's own
    layout, so the same statement is correct for a contiguous pool, for a block-major half and for
    the K-V-interleaved pool a ``block_dim == 1`` backend gets from vLLM's hybrid rewrite.  It only
    assumes what the fused NHD shape guarantees: block is dim 0, token is dim 1, and the trailing
    ``(heads, dim)`` run is what ``key`` carries (checked before we get here).
    """
    blocks, offsets = _slot_split(slot_mapping, int(pool.shape[1]))
    pool[blocks, offsets] = rows


def write_kv_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Scatter this step's K/V into the paged pool, in place.

    ``key``/``value`` are ``[num_tokens, kv_heads, head_dim]``; ``k_cache``/``v_cache`` are the
    ``[blocks, block_size, kv_heads, head_dim]`` halves of the fused pool (see
    :func:`tileinfer.serving.buckets.fused_halves`); ``slot_mapping`` holds one flat token slot
    per row.  Slots are unique within a step (the engine assigns each new token its own slot), so
    the non-deterministic order of ``index_put_`` on duplicate indices never matters here.
    """
    if slot_mapping.numel() == 0:
        return
    _check_step_tensors(key, value, slot_mapping, k_cache, v_cache)
    if k_cache.is_contiguous() and v_cache.is_contiguous():
        # The steady-state path (and byte-for-byte what this module did before the hybrid layout
        # existed): one index_put_ per half, no allocation.
        flat_pool(k_cache)[slot_mapping] = key
        flat_pool(v_cache)[slot_mapping] = value
    else:
        _write_strided(k_cache, key, slot_mapping)
        _write_strided(v_cache, value, slot_mapping)


def write_kv_cache_fused(
    kv_cache: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_dim: Optional[int] = None,
) -> None:
    """Same as :func:`write_kv_cache` for vLLM's fused 5-D pool.

    ``block_dim`` says which dimension of ``kv_cache`` is the block index, exactly as vLLM's own
    ``AttentionBackend.get_kv_cache_block_dim`` defines it (``v1/attention/backend.py:98-116``):

    * ``0`` -> ``[blocks, 2, block_size, heads, dim]`` (what ``get_kv_cache_shape`` returns);
    * ``1`` -> ``[2, blocks, block_size, heads, dim]`` (the vLLM-native form, and the one
      ``_update_hybrid_attention_mamba_layout`` leaves behind as a *strided* tensor when it
      rewrites the physical layout to ``[blocks][2][...]``).

    ``None`` infers it from the shape (``fused_halves``) and refuses when both dims are 2, because
    two blocks cannot serve a real step and guessing would swap K/V silently.
    """
    if kv_cache.dim() != 5:
        raise ValueError(
            "fused KV cache must be 5-D - [blocks, 2, block_size, kv_heads, head_dim] with the "
            "backend's shape, or [2, blocks, block_size, kv_heads, head_dim] - got "
            f"{tuple(kv_cache.shape)}"
        )
    k_cache, v_cache = fused_halves(kv_cache, block_dim)
    write_kv_cache(key, value, k_cache, v_cache, slot_mapping)


def check_slot_mapping(
    slot_mapping: torch.Tensor,
    pool_flat: torch.Tensor,
    name: str = "slot_mapping",
) -> None:
    """Validate that every slot addresses the pool (builder time / tests only).

    This reads values back to the host and therefore **synchronises**: calling it inside the
    captured step would stall the engine and break the replay contract.

    ``pool_flat`` is one half in flat form (``flat_pool``), so its ``shape[0]`` is the number of
    token slots *per half* - the same capacity a slot may address, because K and V have 2x the
    page count each only when fused and are indexed identically.
    """
    if slot_mapping.numel() == 0:
        return
    capacity = int(pool_flat.shape[0])
    lowest = int(slot_mapping.min().item())
    highest = int(slot_mapping.max().item())
    # `min` only matters because it also catches a negative id, which would otherwise wrap around
    # to the end of the pool in Python's negative indexing - silent corruption, not an error.
    if lowest < 0 or highest >= capacity:
        raise ValueError(
            f"{name} addresses slots [{lowest}, {highest}] but the pool has {capacity} token "
            "slots; check the block table / page-size conversion"
        )
