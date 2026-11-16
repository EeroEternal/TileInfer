"""Exact sliding-window decode on the full-causal paged kernel, by scratch staging.

The paged decode kernel computes causal attention over ``[0, seq_len)`` of the pages a
request's block table lists — it cannot express "only the last ``w`` tokens".  This module
bridges the gap *without touching the kernel ABI*: for a windowed layer's decode step, the
last ``window`` tokens of every request are gathered from the real pool into a compact
per-request run of scratch pages, and the kernel machinery runs on the scratch pool with a
synthetic ``arange`` block table and clamped ``seq_lens``.  Causal attention over exactly the
gathered range *is* the window — no mask, no kernel change, and every fallback path
(``plan_not_ready`` / ``plan_failed`` -> the reference backend) stays correct for free,
because the scratch representation is semantically the window, not an approximation.

Shapes are fixed per bucket (``[bucket, window_tokens]``) and every index tensor is computed
on the device, so the steady-state step performs no host sync and no shape-dependent
allocation.  Out-of-range positions (bucket padding rows, the ``window_tokens - window`` tail
when the window is not a multiple of the kernel page) are *gathered* with clamped indices —
their values land in scratch rows the kernel may compute but nobody reads — which keeps the
index shapes data-independent.  Scratch pools are allocated per bucket on first sight (the
same eager-step boundary as ``_BucketState``); each gets two extra trash pages because the
staging-table compaction reserves the last ``kv_indices`` slot.

Sharing
-------
All windowed layers of one model share the same window and the same (per-group) pool, so the
gather each would perform is identical work.  :func:`shared_stager` hands out ONE stager per
geometry, and :meth:`WindowedDecodeStager.stage` skips the gather when the scratch already
holds this step's window: the engine hands every layer of a group the same metadata *object*,
so object identity is the step key.  The kernel only reads the scratch, so the skip is exact;
a new step brings a new metadata instance and re-gathers.  Each layer keeps its own
``_BucketState`` (query/output buffers are per-layer), but those state dicts validate against
the shared scratch by data pointer, which matches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch

from tileinfer.serving.buckets import KERNEL_PAGE_SIZE

__all__ = ["WindowedDecodeStager", "shared_stager", "reset_shared_stagers"]


@dataclass
class _ScratchState:
    """Per-bucket scratch: the pool the kernel reads, plus its synthetic geometry."""

    bucket: int
    #: ``[bucket, window_pages]`` synthetic block table (row ``r`` -> pages ``r*W..(r+1)*W-1``).
    table: torch.Tensor
    #: ``[bucket * window_tokens]`` flat destination indices for the gather (fixed).
    dst: torch.Tensor
    #: ``[bucket]`` scratch for the padded context lengths (filled from ``meta.seq_lens``).
    lens: torch.Tensor
    #: ``[bucket, table_width]`` scratch for the padded copy of the engine's block table.
    table_pad: torch.Tensor = field(repr=False)
    scratch_k: torch.Tensor = field(repr=False)
    scratch_v: torch.Tensor = field(repr=False)


class WindowedDecodeStager:
    """Owns the scratch pools and the per-step gather, shared across same-geometry layers."""

    def __init__(self, num_kv_heads: int, head_size: int, window: int):
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        self.num_kv_heads = int(num_kv_heads)
        self.head_size = int(head_size)
        self.window = int(window)
        # Whole 128-token kernel pages; the tail beyond ``window`` (when 128 does not divide it)
        # is gathered with clamped indices and never read by the causal kernel.
        self.window_pages = max(1, -(-self.window // int(KERNEL_PAGE_SIZE)))
        self.window_tokens = self.window_pages * int(KERNEL_PAGE_SIZE)
        self._states: Dict[int, _ScratchState] = {}
        # The metadata object the current scratch contents were staged for (identity key - the
        # engine hands every layer of a group the same per-step metadata instance).
        self._staged_for: object = None
        self._staged_result: Optional[Tuple[torch.Tensor, torch.Tensor, object]] = None

    def stage(
        self,
        k_half: torch.Tensor,
        v_half: torch.Tensor,
        block_size: int,
        meta,  # TileInferMetadata
    ) -> Tuple[torch.Tensor, torch.Tensor, object]:
        """Gather the window into the scratch pool; return ``(scratch_k, scratch_v, meta2)``.

        ``meta2`` is the incoming metadata with ``block_table_tensor`` / ``seq_lens`` replaced
        by the scratch geometry — every other field (the query view, ``num_reqs``, ``bucket``,
        ``slot_mapping``) describes the batch, not the pool, and is handed through unchanged.
        When the scratch already holds this step's window (another layer staged the same
        metadata instance), the gather is skipped and the staged result returned.
        """
        if self._staged_for is meta and self._staged_result is not None:
            return self._staged_result
        self._staged_for = meta
        self._staged_result = self._stage_now(k_half, v_half, block_size, meta)
        return self._staged_result

    def _stage_now(
        self,
        k_half: torch.Tensor,
        v_half: torch.Tensor,
        block_size: int,
        meta,  # TileInferMetadata
    ) -> Tuple[torch.Tensor, torch.Tensor, object]:
        import dataclasses

        bucket = int(meta.bucket)
        device = k_half.device
        state = self._state_for(bucket, k_half.dtype, device)

        table_width = int(meta.block_table_tensor.shape[1])
        if state.table_pad.shape != (bucket, table_width):
            state.table_pad = torch.zeros((bucket, table_width), dtype=torch.int32, device=device)
        rows = min(int(meta.block_table_tensor.shape[0]), bucket)
        state.table_pad[:rows].copy_(meta.block_table_tensor[:rows])

        # Padded context lengths: the engine's seq_lens (real rows) + ones (padding rows decay
        # to a one-token window, which the kernel may compute but nobody reads).
        lens_src = meta.seq_lens
        n = min(int(lens_src.shape[0]), bucket)
        state.lens[:n].copy_(lens_src[:n].to(torch.int32))
        if n < bucket:
            state.lens[n:].fill_(1)

        # pos[r, j] = start_r + j for j in [0, window_tokens): the absolute positions the
        # window covers, clamped into the table (padding rows and the over-window tail land on
        # duplicated valid tokens; their scratch rows are never attended with real weight).
        start = (state.lens - self.window).clamp(min=0)
        pos = start.unsqueeze(1) + torch.arange(
            self.window_tokens, dtype=torch.int32, device=device
        ).unsqueeze(0)
        pos_max = (state.lens - 1).clamp(min=0).unsqueeze(1)
        pos = torch.minimum(torch.maximum(pos, torch.zeros_like(pos)), pos_max)
        # The kernel-visible context length is the *clamped* one — the scratch holds exactly
        # the window, so causal attention over [0, eff) of the gathered range is the window.
        # (``lens`` is refilled from ``meta.seq_lens`` at the top of this method, so the
        # in-place clamp never leaks into the next step.)
        state.lens.clamp_(max=self.window)

        page_col = (pos // int(block_size)).clamp(max=table_width - 1)
        src_page = state.table_pad.gather(1, page_col.to(torch.int64))
        src_flat = (
            src_page * int(block_size) + (pos - (pos // int(block_size)) * int(block_size))
        ).to(torch.int64).reshape(-1)

        k_flat = k_half.reshape(-1, self.num_kv_heads, self.head_size)
        v_flat = v_half.reshape(-1, self.num_kv_heads, self.head_size)
        state.scratch_k.view(-1, self.num_kv_heads, self.head_size).index_copy_(
            0, state.dst, k_flat.index_select(0, src_flat)
        )
        state.scratch_v.view(-1, self.num_kv_heads, self.head_size).index_copy_(
            0, state.dst, v_flat.index_select(0, src_flat)
        )

        # The kernel-visible batch: the synthetic table and the clamped lengths.  ``max_seq_len``
        # is informational; the smallest legal value keeps any upper-bound reasoning sane.
        meta2 = dataclasses.replace(
            meta,
            block_table_tensor=state.table,
            seq_lens=state.lens,
            max_seq_len=min(int(meta.max_seq_len) or self.window, self.window_tokens),
        )
        return state.scratch_k, state.scratch_v, meta2

    # ------------------------------------------------------------------ per-bucket state

    def _state_for(self, bucket: int, dtype: torch.dtype, device: torch.device) -> _ScratchState:
        state = self._states.get(bucket)
        if state is not None:
            return state
        # +2 pages: the staging-table compaction reserves the last kv_indices slot, and the
        # impl's trash-slot check needs pool_cap - 1 > bucket * window_pages even when the
        # natural page count would round to a power of two.
        pages = bucket * self.window_pages + 2
        shape = (pages, int(KERNEL_PAGE_SIZE), self.num_kv_heads, self.head_size)
        state = _ScratchState(
            bucket=bucket,
            table=torch.arange(
                bucket * self.window_pages, dtype=torch.int32, device=device
            ).reshape(bucket, self.window_pages),
            dst=torch.arange(bucket * self.window_tokens, dtype=torch.int64, device=device),
            lens=torch.ones(bucket, dtype=torch.int32, device=device),
            # table_pad's width is filled on first use (the engine table's width is known then);
            # start with a 1-column placeholder and let stage() rebuild it if needed.
            table_pad=torch.zeros((bucket, 1), dtype=torch.int32, device=device),
            scratch_k=torch.zeros(shape, dtype=dtype, device=device),
            scratch_v=torch.zeros(shape, dtype=dtype, device=device),
        )
        self._states[bucket] = state
        return state


#: Shared stagers, keyed by geometry.  All windowed layers of one model share the window and
#: the per-group pool, so the key deliberately excludes the pool identity: the stager's scratch
#: is pool-agnostic (it is re-gathered from whatever halves the caller passes), and sharing it
#: across layers is what turns 39 identical gathers per step into one.
_SHARED_STAGERS: Dict[Tuple[int, int, int], WindowedDecodeStager] = {}


def shared_stager(num_kv_heads: int, head_size: int, window: int) -> WindowedDecodeStager:
    """The stager shared by every windowed layer of this geometry (see the module docstring)."""
    key = (int(num_kv_heads), int(head_size), int(window))
    stager = _SHARED_STAGERS.get(key)
    if stager is None:
        stager = WindowedDecodeStager(*key)
        _SHARED_STAGERS[key] = stager
    return stager


def reset_shared_stagers() -> None:
    """Drop the shared stagers (tests / engine teardown; the pools die with their owners)."""
    _SHARED_STAGERS.clear()
