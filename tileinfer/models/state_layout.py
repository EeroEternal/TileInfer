"""Engine-free state-cache layout arithmetic.

A recurrent mixer's state is not a KV page: it has a fixed logical shape and the vendor
allocator requires the per-block page to be sized by its own attention geometry.  LFM3's
short convolution is the first case (a 6-slot window padded to 32 columns); Mamba2 and a
future linear mixer need the same arithmetic.  Keep it here, away from vLLM imports, so it
can be unit-tested on CPU and reused by any model payload or provider.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

#: vllm-ascend's SSM page geometry: 128 tokens x head_dim x kv_heads x dtype bytes.
ASCEND_SSM_PAGE_TOKENS = 128


@dataclass(frozen=True)
class StateLayout:
    """A state shape before/after vendor page padding."""

    dim: int
    state_len: int
    padded_len: int
    dtype_size: int
    target_page_bytes: int
    blocks: int

    @property
    def logical_bytes(self) -> int:
        return self.dim * self.state_len * self.dtype_size

    @property
    def padded_bytes(self) -> int:
        return self.dim * self.padded_len * self.dtype_size

    def shape(self, dim_first: bool = True) -> Tuple[int, int]:
        """Return the 2-D shape the engine should allocate."""
        if dim_first:
            return (self.dim, self.padded_len)
        return (self.padded_len, self.dim)

    def as_dict(self) -> dict:
        return {
            "dim": self.dim,
            "state_len": self.state_len,
            "padded_len": self.padded_len,
            "dtype_size": self.dtype_size,
            "target_page_bytes": self.target_page_bytes,
            "blocks": self.blocks,
            "logical_bytes": self.logical_bytes,
            "padded_bytes": self.padded_bytes,
        }


def ascend_page_bytes(head_dim: int, num_kv_heads: int, dtype_size: int) -> int:
    """Bytes in one vllm-ascend SSM/attention page."""
    if head_dim <= 0 or num_kv_heads <= 0 or dtype_size <= 0:
        raise ValueError("head_dim, num_kv_heads and dtype_size must be positive")
    return ASCEND_SSM_PAGE_TOKENS * head_dim * num_kv_heads * dtype_size


def short_conv_state_layout(
    dim: int,
    kernel: int,
    head_dim: int,
    num_kv_heads: int,
    dtype_size: int,
) -> StateLayout:
    """LFM3-style short-convolution state padded to the Ascend SSM page size.

    The math only reads ``kernel - 1`` history slots; the remaining columns are allocator
    padding and must stay zero/unused.
    """
    if dim <= 0 or kernel <= 1:
        raise ValueError("dim must be positive and kernel must be > 1")
    state_len = kernel - 1
    target = ascend_page_bytes(head_dim, num_kv_heads, dtype_size)
    logical = dim * state_len * dtype_size
    blocks = max(1, math.ceil(logical / target))
    padded = (blocks * target) // (dim * dtype_size)
    padded = max(state_len, padded)
    return StateLayout(
        dim=dim,
        state_len=state_len,
        padded_len=padded,
        dtype_size=dtype_size,
        target_page_bytes=target,
        blocks=blocks,
    )


def state_layout_fits_page(layout: StateLayout) -> bool:
    """True when the padded state is an exact whole number of vendor pages."""
    return layout.padded_bytes % layout.target_page_bytes == 0


__all__ = [
    "ASCEND_SSM_PAGE_TOKENS",
    "StateLayout",
    "ascend_page_bytes",
    "short_conv_state_layout",
    "state_layout_fits_page",
]
