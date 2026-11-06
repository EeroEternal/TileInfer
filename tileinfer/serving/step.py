"""The per-step metadata rebuild: dense page table -> ragged triple, entirely on the device.

What it is for
--------------
The engine hands a decode step a **dense** page table ``[num_reqs, max_pages]`` plus ``seq_lens``.
The kernel wants the **ragged** form: ``kv_indptr`` (cumulative page counts), ``kv_indices`` (the
physical page ids, in logical order, compacted) and ``kv_last_page_len`` (valid tokens in the last
page).  Compacting on the host would be a per-step allocation and, worse, a host-device
synchronisation - both illegal inside a captured ACLGraph - so the whole conversion runs on the
device with fixed shapes and in-place writes.

The invariant that makes it correct
-----------------------------------
``kv_indices`` has a **trash slot**: its last element.  Every table column that a request does not
own (``j >= pages_b``) is parked there, so the kernel - which only ever visits
``kv_indices[indptr[b] : indptr[b] + pages_b]`` - can never reach a stale page id.  An earlier
revision used a dense indptr (``arange(bucket + 1) * width``), which made every row attend over
``width`` pages including stale ids, and produced garbage that looked plausible.  The two
conditions that keep the trash slot meaningful are asserted here:

* ``bucket * width < kv_indices.numel()`` (there is always a slot past every valid destination);
* the trash slot is never a *valid* destination (its index is past all compacted spans).

Capture contract
----------------
Every operation below is fixed-shape and in place: no allocation, no compilation, no host read.
Two scratch buffers (``mul_buf``, and the constant ``repeat_index`` / ``offset_row`` vectors) are
allocated once at plan time, because the obvious formulation - ``repeat_interleave`` - allocates a
fresh ``[bucket, width]`` tensor on every step, which is exactly what the capture rule forbids.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .buckets import KERNEL_PAGE_SIZE, expand_page_table, last_page_len_

__all__ = ["StepBuffers", "allocate_step_buffers", "refresh_step"]


@dataclass
class StepBuffers:
    """Plan-owned buffers the step writes into (stable addresses, allocated once)."""

    bucket: int
    width: int
    engine_width: int
    trash_slot: int
    table_buf: torch.Tensor       # [bucket, width]        int32  expanded sub-page ids
    lens_buf: torch.Tensor        # [bucket]               int32  padded kv lengths
    last_page_len_buf: torch.Tensor  # [bucket]            int32
    pages_buf: torch.Tensor       # [bucket]               int32  ceil(len / 128)
    kv_indptr_buf: torch.Tensor   # [bucket + 1]           int32  cumulative pages
    valid_buf: torch.Tensor       # [bucket, width]        bool   j <  pages_b
    invalid_buf: torch.Tensor     # [bucket, width]        bool   j >= pages_b
    dst32_buf: torch.Tensor       # [bucket, width]        int32  scatter destination (compact)
    dst64_buf: torch.Tensor       # [bucket, width]        int64  same, for scatter_
    mul_buf: torch.Tensor         # [bucket, engine_width] int32  engine_page * sub scratch
    j_buf: torch.Tensor           # [width]                int32  column index (constant)
    repeat_index: torch.Tensor    # [width]                int64  constant, for index_select
    offset_row: torch.Tensor      # [width]                int32  constant, sub-page offset

    @property
    def device(self) -> torch.device:
        return self.table_buf.device


def allocate_step_buffers(
    bucket: int,
    width: int,
    kv_indices_capacity: int,
    block_size: int,
    device: torch.device,
) -> StepBuffers:
    """Allocate the step buffers for one ``(bucket, table width)`` shape.

    ``width`` is the table width in **kernel pages**; ``block_size`` is the engine page size, so
    ``engine_width = width // sub_pages(block_size)``.
    """
    from .buckets import sub_pages  # local import keeps the module's public surface small

    bucket = int(bucket)
    width = int(width)
    if bucket < 1 or width < 1:
        raise ValueError(f"bucket and width must be positive, got bucket={bucket}, width={width}")
    sub = sub_pages(int(block_size))
    if width % sub:
        raise ValueError(f"kernel width {width} is not a multiple of sub_pages={sub}")
    engine_width = width // sub
    capacity = int(kv_indices_capacity)
    trash_slot = capacity - 1
    if bucket * width >= trash_slot:
        raise ValueError(
            f"KV pool ({capacity} token slots) is smaller than the staging table "
            f"({bucket}x{width}); the ragged compaction has nowhere to park invalid columns"
        )

    i32 = torch.int32
    buffers = StepBuffers(
        bucket=bucket,
        width=width,
        engine_width=engine_width,
        trash_slot=trash_slot,
        # Padding rows keep table=0 / len=1: a *valid* page and a *valid* length, so the kernel's
        # softmax cannot divide by zero and the padded outputs are never read.
        table_buf=torch.zeros((bucket, width), dtype=i32, device=device),
        lens_buf=torch.ones(bucket, dtype=i32, device=device),
        last_page_len_buf=torch.ones(bucket, dtype=i32, device=device),
        pages_buf=torch.empty(bucket, dtype=i32, device=device),
        kv_indptr_buf=torch.zeros(bucket + 1, dtype=i32, device=device),
        valid_buf=torch.empty((bucket, width), dtype=torch.bool, device=device),
        invalid_buf=torch.empty((bucket, width), dtype=torch.bool, device=device),
        dst32_buf=torch.empty((bucket, width), dtype=i32, device=device),
        dst64_buf=torch.empty((bucket, width), dtype=torch.int64, device=device),
        mul_buf=torch.empty((bucket, engine_width), dtype=i32, device=device),
        j_buf=torch.arange(width, dtype=i32, device=device),
        repeat_index=torch.arange(engine_width, dtype=torch.int64, device=device).repeat_interleave(
            sub
        ),
        offset_row=torch.arange(sub, dtype=i32, device=device).repeat(engine_width),
    )
    return buffers


def _expand_into(
    buffers: StepBuffers, block_tables: torch.Tensor, num_reqs: int
) -> None:
    """``table_buf[:n] = expand_page_table(block_tables[:n], sub)`` without allocating.

    ``expand_page_table`` is the reference form (and the one the plan/tests use); it allocates a
    fresh tensor, so the step path re-implements the same mapping with three in-place kernels:
    scale engine pages by ``sub``, gather each column ``sub`` times (constant index), then add the
    per-position offset ``0..sub-1``.
    """
    if buffers.engine_width == buffers.width:  # sub == 1: the table is already in kernel pages
        buffers.table_buf[:num_reqs].copy_(block_tables[:num_reqs])
        return
    torch.mul(block_tables[:num_reqs], buffers.width // buffers.engine_width, out=buffers.mul_buf[:num_reqs])
    torch.index_select(
        buffers.mul_buf[:num_reqs], 1, buffers.repeat_index, out=buffers.table_buf[:num_reqs]
    )
    buffers.table_buf[:num_reqs].add_(buffers.offset_row)


def refresh_step(
    buffers: StepBuffers,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    kv_indices: torch.Tensor,
    num_reqs: Optional[int] = None,
) -> None:
    """Rebuild ``kv_indptr`` / ``kv_indices`` / ``kv_last_page_len`` for this step, in place.

    Parameters
    ----------
    block_tables:
        The engine's dense table, ``[num_reqs, engine_width]`` (its width may exceed the plan's -
        only the first ``buffers.engine_width`` columns are read).
    seq_lens:
        Full KV length in tokens per request (the cached prefix *plus* this step's tokens), on the
        device.  Its dtype is whatever the engine uses; ``copy_`` converts, which is in-place.
    block_size:
        The engine's page size in tokens (a multiple of :data:`KERNEL_PAGE_SIZE`).
    kv_indices:
        The plan-owned ``[capacity]`` int32 buffer the kernel reads; its last element is the
        trash slot.
    num_reqs:
        Defaults to ``seq_lens.numel()``; pass it explicitly when ``seq_lens`` is bucket-sized.

    Deviation from the previous adapter, deliberately: padding rows' lengths are reset to ``1``
    every step.  They used to keep whatever a *previous, larger* batch left there, which is still
    correct (their table rows are zeroed, so they only ever read page 0) but makes the padding
    rows walk up to ``old_len`` pages of page 0 for nothing.
    """
    from .buckets import sub_pages

    if num_reqs is None:
        num_reqs = int(seq_lens.numel())
    num_reqs = int(num_reqs)
    if num_reqs < 1 or num_reqs > buffers.bucket:
        raise ValueError(f"num_reqs={num_reqs} outside (0, bucket={buffers.bucket}]")
    if block_tables.dim() != 2 or block_tables.shape[0] < num_reqs:
        raise ValueError(
            f"block_tables must be [num_reqs >= {num_reqs}, max_pages], got "
            f"{tuple(block_tables.shape)}"
        )
    if block_tables.dtype != torch.int32:
        # The expansion runs in int32 (the kernel's page-id dtype); casting a fresh table each
        # step would allocate inside the captured region, so the engine must hand us int32.
        raise ValueError(
            f"block_tables must be int32 (got {block_tables.dtype}); the step path cannot cast "
            "without allocating"
        )
    sub = sub_pages(int(block_size))
    if block_tables.shape[1] * sub < buffers.width:
        raise ValueError(
            f"engine table width {block_tables.shape[1]} x sub_pages {sub} does not cover the "
            f"plan's kernel width {buffers.width}"
        )
    if kv_indices.numel() <= buffers.bucket * buffers.width:
        raise ValueError(
            f"kv_indices has {kv_indices.numel()} slots, needs more than "
            f"bucket*width={buffers.bucket * buffers.width} (trash slot)"
        )

    _expand_into(buffers, block_tables, num_reqs)
    # Padding rows: table 0 so they can only reach the (valid) page 0.
    buffers.table_buf[num_reqs:].zero_()

    buffers.lens_buf[:num_reqs].copy_(seq_lens[:num_reqs])
    buffers.lens_buf[num_reqs:].fill_(1)
    last_page_len_(buffers.lens_buf, buffers.last_page_len_buf, KERNEL_PAGE_SIZE)

    # pages_b = ceil(lens_b / 128); kv_indptr = exclusive cumsum (indptr[0] stays 0).
    torch.add(buffers.lens_buf, KERNEL_PAGE_SIZE - 1, out=buffers.pages_buf)
    buffers.pages_buf.floor_divide_(KERNEL_PAGE_SIZE)
    torch.cumsum(buffers.pages_buf, dim=0, out=buffers.kv_indptr_buf[1:])

    # Compaction: destination = indptr[b] + j for the columns a request owns, trash otherwise.
    j = buffers.j_buf
    pages = buffers.pages_buf
    torch.lt(j.unsqueeze(0), pages.unsqueeze(1), out=buffers.valid_buf)
    torch.ge(j.unsqueeze(0), pages.unsqueeze(1), out=buffers.invalid_buf)
    torch.add(buffers.kv_indptr_buf[:-1].unsqueeze(1), j.unsqueeze(0), out=buffers.dst32_buf)
    buffers.dst64_buf.copy_(buffers.dst32_buf)
    buffers.dst64_buf.masked_fill_(buffers.invalid_buf, buffers.trash_slot)
    kv_indices.scatter_(0, buffers.dst64_buf.view(-1), buffers.table_buf.view(-1))


def expected_ragged(
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    page_size: int = KERNEL_PAGE_SIZE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference implementation of the same conversion (host-side, allocating).

    Used by the tests as the oracle, and by callers that want the ragged form without a plan.
    """
    from .buckets import pages_for_lens, sub_pages

    n = int(seq_lens.numel())
    sub = sub_pages(int(block_size))
    table = expand_page_table(block_tables[:n], sub)
    lens = seq_lens[:n].to(torch.int64)
    pages = pages_for_lens(lens, page_size)
    indptr = torch.zeros(n + 1, dtype=torch.int64, device=lens.device)
    indptr[1:] = torch.cumsum(pages, dim=0)
    indices = torch.cat(
        [table[row, : int(pages[row])] for row in range(n)]
    ) if n else torch.empty(0, dtype=torch.int32, device=lens.device)
    last = (lens - 1).remainder(page_size) + 1
    return indptr.to(torch.int32), indices.to(torch.int32), last.to(torch.int32)
