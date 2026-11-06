"""Torch implementations of the FLA operators the GDN layers call.

Why these exist
---------------
vLLM 0.23 implements GDN through `vllm/model_executor/layers/fla/ops/`, whose every code path is a
Triton kernel (`chunk_delta_h`, `chunk_o`, `chunk_scaled_dot_kkt`, `cumsum`, `l2norm`, `solve_tril`,
`wy_fast`).  There is no torch branch on the NPU path, so a backend that does not install
vllm-ascend has to provide these operators itself.  This module starts with the two that carry no
cross-token state, because getting them exactly right is a precondition for the others: their output
feeds every chunk of the recurrence, so an `eps` convention off by one term shifts the whole model's
numerics without ever raising.

Where the semantics come from (not guessed)
-------------------------------------------
`runtime/vllm-api/fla_l2norm.py`, `l2norm_fwd_kernel1`:

```
b_var  = tl.sum(b_x * b_x, axis=0)
b_rstd = 1 / tl.sqrt(b_var + eps)     # eps is added to the SQUARED SUM, not clamped onto the norm
b_y    = b_x * b_rstd
```

so ``l2norm(x) = x / sqrt(sum(x^2, -1) + eps)``.  The common "``x / max(||x||, eps)``" formulation is
a *different* function; it agrees for large norms and diverges exactly on the near-zero rows that
the gate produces early in training/serving.

`chunk_local_cumsum` resets the running sum at every ``chunk_size`` boundary and, for a ragged batch,
also at every sequence boundary (that is what ``cu_seqlens`` is for); ``reverse=True`` makes it a
suffix sum within the chunk.  The kernels confirm the reset semantics; the exact ``head_first``
layout handling is validated by the tests here, not assumed.
"""

from __future__ import annotations

from typing import Optional

import torch

__all__ = ["l2norm_fwd", "chunk_local_cumsum"]


def l2norm_fwd(
    x: torch.Tensor, eps: float = 1e-6, output_dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """Row-wise L2 normalisation: ``x / sqrt(sum(x^2) + eps)``.

    The reduction runs in fp32 (the kernel loads into fp32 before the sum), then the result is cast
    to ``output_dtype`` - or back to ``x.dtype`` when not given, which matches the kernel storing
    into a buffer of the input dtype.
    """
    if x.dim() < 1:
        raise ValueError("l2norm_fwd expects at least a 1-D tensor")
    work = x.to(torch.float32)
    squared = (work * work).sum(dim=-1, keepdim=True)
    normalized = work * torch.rsqrt(squared + float(eps))
    return normalized.to(output_dtype or x.dtype)


def chunk_local_cumsum(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Cumulative sum inside ``chunk_size``-token chunks, along dimension 0.

    ``g`` is ``[T, ...]`` (tokens first).  The sum restarts at every chunk boundary and, when
    ``cu_seqlens`` is given, at every sequence start too - the ragged form a serving batch always
    uses.  ``reverse=True`` produces the suffix sum within each chunk.

    Implemented as a reshape + cumsum for the common case (no varlen) and a per-sequence loop for the
    ragged case; the loop is over sequences, never over tokens, so it stays cheap at batch sizes a
    serving step actually sees.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    if g.dim() < 1:
        raise ValueError("chunk_local_cumsum expects at least a 1-D tensor")
    if reverse:
        return _forward(g, chunk_size, cu_seqlens, reverse=True)
    return _forward(g, chunk_size, cu_seqlens)


def _chunk_forward(block: torch.Tensor, chunk_size: int, reverse: bool = False) -> torch.Tensor:
    """Prefix (or suffix) sum inside each chunk of a contiguous token block."""
    length = block.shape[0]
    if length <= chunk_size:
        return torch.cumsum(block.flip(0), 0).flip(0) if reverse else torch.cumsum(block, 0)
    padded = -(-length // chunk_size) * chunk_size
    tail = padded - length
    flat = block.reshape(length, -1)
    if tail:
        flat = torch.cat([flat, flat.new_zeros((tail, flat.shape[1]))], dim=0)
    chunks = flat.reshape(-1, chunk_size, flat.shape[1])
    if reverse:
        # flip inside the chunk axis only: the chunk boundaries stay anchored to the original order
        chunks = chunks.flip(1)
    out = torch.cumsum(chunks, dim=1)
    if reverse:
        out = out.flip(1)
    return out.reshape(padded, flat.shape[1])[:length].reshape(block.shape)


def _forward(
    g: torch.Tensor, chunk_size: int, cu_seqlens: Optional[torch.Tensor], reverse: bool = False
) -> torch.Tensor:
    if cu_seqlens is None:
        return _chunk_forward(g, chunk_size, reverse)
    from .fla.hostcache import int_list  # noqa: PLC0415 - cached batch-meta reads

    bounds = int_list(cu_seqlens)
    if len(bounds) < 2 or bounds[0] != 0 or bounds[-1] != g.shape[0]:
        raise ValueError(
            f"cu_seqlens must span [0, {g.shape[0]}], got {bounds[:4]}...{bounds[-1:]}"
        )
    out = torch.empty_like(g)
    for start, end in zip(bounds[:-1], bounds[1:]):
        if end > start:
            out[start:end] = _chunk_forward(g[start:end], chunk_size, reverse)
    return out
