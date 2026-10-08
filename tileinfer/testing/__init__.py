"""Test utilities: reference implementations and KV-cache builders."""

from .cache import build_paged_cache, random_q
from .reference import (
    expand_kv_heads,
    gather_kv,
    merge_partials,
    reference_attention,
    reference_attention_tiled,
)

__all__ = [
    "build_paged_cache",
    "random_q",
    "expand_kv_heads",
    "gather_kv",
    "merge_partials",
    "reference_attention",
    "reference_attention_tiled",
]
