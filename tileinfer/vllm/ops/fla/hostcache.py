"""Cached host-side reads of the *batch* meta tensors the GDN chunk chain keeps re-converting.

Why this exists
---------------
Every module in the chunked GDN prefill chain (``chunk``, ``wu``, ``kkt``, ``delta_h``,
``solve_tril``, and ``ops/gdn`` itself) reads ``cu_seqlens`` back to the host to bound its chunk loop,
and several of them do the same for ``chunk_indices``.  Those tensors describe the **batch**, not the
layer, so on Qwen3.6 (30 GDN layers, ~7 conversions per layer) a prefill step performs **~210
device->host synchronisations** whose answers are identical (§37.2).

Measured context for why that matters: §32 priced one such sync at ~0.5 ms in this stack, and the
fix there (removing a per-call ``bool(valid.any())``) was worth 14.6% end to end.

Safety
------
* The cache holds a **strong reference** to the source tensor, so an entry cannot outlive it and the
  key can never be recycled into a different tensor's address (the same argument as
  ``recurrent._decode_constants``).
* The key includes the shape and dtype, so a reshaped or re-dtyped tensor misses.
* ``TILEINFER_GDN_HOST_CACHE=0`` (or the ``gdn_host_cache`` file flag) turns it off, in which case
  every call reconverts -- i.e. exactly the old behaviour.
* Nothing here changes arithmetic: the values returned are the same Python ints ``tolist()`` gives.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

__all__ = ["bounds_int_list", "int_list", "pairs_int_list", "reset_cache"]

#: ``(id(tensor), data_ptr, shape, dtype)`` -> ``(host_values, source_tensor)``
_CACHE: Dict[Tuple[Any, ...], Tuple[Any, Any]] = {}
_CACHE_MAX = 256


def _enabled() -> bool:
    """**Default off, because it was measured**: 0.999 (§42).

    The reasoning behind the change was sound — the chunk chain converts batch-meta tensors ~7 times
    per layer and they are layer-invariant — but the paired prefill measurement came out at **0.999
    (4 pairs, sign mixed, spread 13.6%)**: those conversions are cheap enough that removing them is
    invisible next to the rest of a prefill step.  So it is kept as an opt-in
    (``TILEINFER_GDN_HOST_CACHE=1``) rather than paying cache bookkeeping for nothing.
    """
    from tileinfer.serving import flags  # noqa: PLC0415

    return flags.flag("gdn_host_cache", "TILEINFER_GDN_HOST_CACHE", False)


def _key(tensor: Any) -> Tuple[Any, ...]:
    return (
        id(tensor),
        tensor.data_ptr(),
        tuple(tensor.shape),
        str(tensor.dtype),
        str(tensor.device),
    )


def _cached(tensor: Any, build):
    if not _enabled():
        return build(tensor)
    key = _key(tensor)
    hit = _CACHE.get(key)
    if hit is not None:
        return hit[0]
    value = build(tensor)
    if len(_CACHE) >= _CACHE_MAX:
        _CACHE.clear()
    # The strong reference is the safety argument, not an implementation detail: keep it.
    _CACHE[key] = (value, tensor)
    return value


def int_list(tensor: Any) -> List[int]:
    """``[int(v) for v in tensor.tolist()]``, cached per tensor (cu_seqlens, chunk_offsets, ...)."""
    return _cached(tensor, lambda t: [int(v) for v in t.tolist()])


#: Alias kept for readability at the ``cu_seqlens`` call sites.
bounds_int_list = int_list


def pairs_int_list(chunk_indices: Any) -> List[Tuple[int, int]]:
    """``chunk_indices.tolist()`` as a list of ``(sequence, chunk)`` int pairs, cached per tensor."""
    return _cached(chunk_indices, lambda t: [(int(a), int(b)) for a, b in t.tolist()])


def reset_cache() -> None:
    """Drop every entry (tests, engine teardown)."""
    _CACHE.clear()
