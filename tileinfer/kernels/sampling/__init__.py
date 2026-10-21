"""Sampling kernels (top-k threshold selection over the vocabulary).

Imported lazily: ``import tileinfer`` must work on a machine with no DeepSelect / NPU.
"""

from .topk import (
    SAMPLING_MIN_BATCH,
    SAMPLING_MIN_VOCAB,
    apply_top_k_top_p,
    reset_k_scalar_cache,
    resolve_k_scalar_cached,
    sampling_window_ok,
)

__all__ = [
    "apply_top_k_top_p",
    "sampling_window_ok",
    "resolve_k_scalar_cached",
    "reset_k_scalar_cache",
    "SAMPLING_MIN_VOCAB",
    "SAMPLING_MIN_BATCH",
]
