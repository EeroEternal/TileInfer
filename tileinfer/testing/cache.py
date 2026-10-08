"""KV-cache builders used by tests and benchmarks.

The point of these helpers is to make the *indirection* explicit: a request's pages are not
necessarily contiguous, and a page is not necessarily full.  A kernel that ignores ``kv_indices``
or ``kv_last_page_len`` will pass a naive test and fail on a real serving batch, so the builders
here deliberately offer page shuffling.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from ..metadata import RaggedMetadata

__all__ = ["build_paged_cache", "random_q"]


def build_paged_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    page_size: int = 128,
    layout: str = "NHD",
    shuffle_pages: bool = False,
    seed: int = 0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, RaggedMetadata]:
    """Build paged K and V caches plus the metadata addressing them.

    ``k`` / ``v`` are ``[batch, num_kv_heads, kv_len, head_dim]`` (uniform lengths).  Both caches
    share one :class:`~tileinfer.metadata.RaggedMetadata` because they are addressed identically —
    which is the whole point of a page table.
    """
    layout = layout.upper()
    if layout not in ("NHD", "HND"):
        raise ValueError(f"unknown kv layout {layout!r} (expected 'NHD' or 'HND')")
    batch, h_kv, kv_len, d = k.shape
    device = device or k.device
    pages_per_seq = (kv_len + page_size - 1) // page_size
    num_blocks = batch * pages_per_seq

    shape = (
        (num_blocks, page_size, h_kv, d)
        if layout == "NHD"
        else (num_blocks, h_kv, page_size, d)
    )
    k_cache = torch.zeros(shape, dtype=k.dtype, device=device)
    v_cache = torch.zeros(shape, dtype=v.dtype, device=device)

    for b in range(batch):
        for p in range(pages_per_seq):
            start = p * page_size
            end = min(start + page_size, kv_len)
            blk = b * pages_per_seq + p
            if layout == "NHD":
                k_cache[blk, : end - start] = k[b, :, start:end, :].permute(1, 0, 2)
                v_cache[blk, : end - start] = v[b, :, start:end, :].permute(1, 0, 2)
            else:
                k_cache[blk, :, : end - start] = k[b, :, start:end, :]
                v_cache[blk, :, : end - start] = v[b, :, start:end, :]

    indices = torch.arange(num_blocks, dtype=torch.int32, device=device)
    if shuffle_pages:
        gen = torch.Generator(device="cpu").manual_seed(seed)
        perm = torch.randperm(num_blocks, generator=gen)
        indices = indices[perm.cpu().to(device)].contiguous()

    indptr = torch.arange(
        0, (batch + 1) * pages_per_seq, pages_per_seq, dtype=torch.int32, device=device
    )
    last_page_len = torch.full(
        (batch,),
        page_size if kv_len % page_size == 0 else kv_len % page_size,
        dtype=torch.int32,
        device=device,
    )
    meta = RaggedMetadata(
        kv_indptr=indptr,
        kv_indices=indices,
        kv_last_page_len=last_page_len,
        page_size=page_size,
    )
    return k_cache, v_cache, meta


def random_q(
    batch: int,
    num_qo_heads: int,
    qo_len: int,
    head_dim: int,
    dtype: torch.dtype = torch.float16,
    device: Optional[torch.device] = None,
    scale: float = 1.0,
) -> torch.Tensor:
    """Random query tensor ``[batch, num_qo_heads, qo_len, head_dim]``."""
    return scale * torch.randn(
        (batch, num_qo_heads, qo_len, head_dim), dtype=dtype, device=device
    )
