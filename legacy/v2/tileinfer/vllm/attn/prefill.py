"""Kernel-backed **prefill / append** for the native vLLM attention path (§17 ②③).

Why this is not an ``AttentionPlan``
------------------------------------
The decode path compiles one kernel per ``(batch bucket, table width, pool cap)`` because a decode
step has a *shape-stable* meaning: ``batch`` rows, one query token each, one tile per request.  A
prefill step has no such stability — its tile count is ``sum_b ceil(qo_len_b * group / block_q)``,
which changes with every prompt length.  Baking that into a plan (which is compiled once and cached
forever) would either compile a variant per prompt length or freeze one step's schedule into every
later step.

So prefill is executed *eagerly* instead: the schedule is rebuilt per step on the host (the lengths
already sit on the host in ``query_start_loc_cpu``, so this costs **no** device sync), and the
kernel variant is quantized on two axes so the number of compiles stays bounded:

``num_tiles``
    rounded *up* to the next :data:`TILE_LADDER` rung.  Padding tiles carry ``q_len = 0`` and
    ``seq_ids = 0``; the kernel computes ``plen = 0`` for them and its page loop runs zero
    iterations, and :func:`~tileinfer.kernels.attention.paged_decode_ascend950.forward_prefill`
    masks their gathered query rows to zero (the P3 convention its docstring documents).

``batch``
    pinned to :data:`KERNEL_BATCH`.  ``batch`` only shapes ``kv_indptr`` / ``kv_last_page_len`` /
    ``causal_shift`` — the kernel never loops over it — so a single generous constant removes that
    axis from the compile key entirely.

Everything else (``kv_heads``/``group``/``dim``/``page_size``/``pool cap``/``dtype``) comes from the
layer and the pool and is constant per process.

Correctness contract
--------------------
This path is **not** a "best effort": it either returns the kernel's result or raises
:class:`PrefillUnsupported` before touching the device, and the caller then runs the reference
backend and counts the fallback.  Any *other* exception propagates — a kernel that runs and is
wrong is a bug, not a reason to silently degrade (``docs/notes/perf-comparison-rules.md`` C3).
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from tileinfer.plan import TileSchedule, plan_query_tiles_from_host
from tileinfer.serving.buckets import (
    KERNEL_PAGE_SIZE,
    declared_pool_cap,
    pool_sub_view,
    sub_pages,
)
from tileinfer.serving.step import StepBuffers, allocate_step_buffers, refresh_step

__all__ = [
    "BLOCK_Q",
    "KERNEL_BATCH",
    "TILE_LADDER",
    "PrefillPath",
    "PrefillUnsupported",
    "prefill_tile_bucket",
]

logger = logging.getLogger("tileinfer.vllm.attn.prefill")

#: The kernel's M tile: ``block_q`` query rows per block.  ``BR <= 128`` is a hard assert in
#: ``paged_prefill_ascend950`` (one L0C tile holds ``BR x BC`` floats at page_size 128).
BLOCK_Q = 128

#: Compiled-variant quantization on the tile axis (see the module docstring).  A compile is
#: tens of seconds, so the ladder trades a bounded number of variants for at most 2x dead tiles.
TILE_LADDER: Tuple[int, ...] = (8, 16, 32, 64, 128, 256, 512, 1024, 2048)

#: Compiled-variant quantization on the batch axis: one constant for every real batch below it.
KERNEL_BATCH = max(32, int(os.environ.get("TILEINFER_PREFILL_KERNEL_BATCH", "512") or 512))

#: Request-count ladder for the per-step ragged buffers.  ``refresh_step`` writes only
#: ``num_reqs`` rows, so the buffers only have to *cover* the batch; laddering them keeps the
#: number of ``kv_indices`` allocations (one per state) small.
REQ_LADDER: Tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)

_I32 = torch.int32


class PrefillUnsupported(RuntimeError):
    """A prefill step this path cannot serve; ``reason`` becomes a counter suffix."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"prefill kernel unsupported ({reason}): {detail}")
        self.reason = str(reason)
        self.detail = str(detail)


def _prefill_timing_enabled() -> bool:
    """``TILEINFER_PREFILL_TIMING=1`` prints launch vs device-inclusive time every 20 steps (§60)."""
    return os.environ.get("TILEINFER_PREFILL_TIMING", "").strip() not in ("", "0")


#: Aggregated across every ``PrefillPath`` instance (one per attention layer), so the 20-call window
#: is reachable: a single layer sees one call per request.
_TIMING_STATS: List[Tuple[float, float, int]] = []

#: ``(query lengths, num_reqs) -> (src_flat, valid_flat)``.  The pack indices depend only on the
#: schedule, and the schedule only on the query lengths, so every step of every layer with the same
#: shape reuses one entry.  Bounded in :meth:`PrefillPath.run`; holding device tensors, so it must
#: not grow without limit.
_PACK_CACHE: Dict[Tuple[Tuple[int, ...], int], Tuple[torch.Tensor, torch.Tensor]] = {}


def prefill_tile_bucket(live_tiles: int) -> int:
    """Round a live tile count up to the compile-variant ladder (see :data:`TILE_LADDER`)."""
    live = int(live_tiles)
    if live <= 0:
        return 0
    for rung in TILE_LADDER:
        if rung >= live:
            return rung
    step = TILE_LADDER[-1]
    return ((live + step - 1) // step) * step


def _req_bucket(num_reqs: int) -> int:
    for rung in REQ_LADDER:
        if rung >= num_reqs:
            return rung
    return int(num_reqs)


@dataclass
class _PrefillState:
    """Per-(batch bucket, table width) scratch: stable addresses, allocated once per shape."""

    req_bucket: int
    width: int
    block_size: int
    pool_cap: int
    dtype: torch.dtype
    device: torch.device
    step: StepBuffers
    kv_indices: torch.Tensor  # [pool_cap] int32, the page-id list the kernel reads


class PrefillPath:
    """Eager, kernel-backed prefill for one attention layer's geometry."""

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
        scale: float,
    ) -> None:
        self.num_heads = int(num_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.head_size = int(head_size)
        self.scale = float(scale)
        if self.num_kv_heads < 1 or self.num_heads % self.num_kv_heads:
            raise ValueError(
                f"num_heads={self.num_heads} is not a multiple of num_kv_heads={self.num_kv_heads}"
            )
        self.group = self.num_heads // self.num_kv_heads
        self.single_pass_pv = os.environ.get("TILEINFER_PREFILL_PV1PASS", "").strip() not in (
            "",
            "0",
        )
        self._states: Dict[Tuple[int, int, int, int, str], _PrefillState] = {}
        #: ``(launch_ms, total_ms, tokens)`` per step, only filled when ``TILEINFER_PREFILL_TIMING=1``.
        self._timing_stats: List[Tuple[float, float, int]] = []
        if _prefill_timing_enabled():
            # One line at construction time: §63.2 could not explain why the per-step timing never
            # printed although 100 steps ran, and the cheapest remaining hypothesis is that the
            # child process does not inherit the variable.  Printing the resolved value here settles
            # it in the first log line instead of after a measurement.
            print(
                "[tileinfer/prefill-timing] enabled "
                f"(TILEINFER_PREFILL_TIMING={os.environ.get('TILEINFER_PREFILL_TIMING', '<unset>')!r})",
                flush=True,
            )

    # ------------------------------------------------------------------ capability

    @property
    def supported(self) -> bool:
        """Whether the *kernel* can serve this layer's head geometry at all."""
        return self.head_size in (128, 256)

    # ------------------------------------------------------------------ execution

    def run(
        self,
        q3: torch.Tensor,
        k_half: torch.Tensor,
        v_half: torch.Tensor,
        block_size: int,
        meta,
        output: Optional[torch.Tensor],
        num_tokens: int,
    ) -> torch.Tensor:
        """Prefill/append this step on the kernel; raises :class:`PrefillUnsupported` if it cannot.

        ``q3``      ``[num_tokens, num_heads, head_size]`` — the step's rows, request-major
        ``k_half``  ``[blocks, block_size, kv_heads, head_size]`` — the K half, already written
        """
        if not self.supported:
            raise PrefillUnsupported(
                "head_dim", f"head_size {self.head_size} is not in (128, 256)"
            )
        num_reqs = int(meta.num_reqs)
        if num_reqs < 1:
            raise PrefillUnsupported("shape", "a batch with no requests cannot be attended")
        if num_reqs > KERNEL_BATCH:
            raise PrefillUnsupported(
                "batch", f"num_reqs {num_reqs} exceeds the pinned kernel batch {KERNEL_BATCH}"
            )
        sub = sub_pages(int(block_size))  # raises if the engine page is not k*128 tokens

        # Optional self-contained timing (§60.4): the phase framework's numbers disagreed with an
        # isolated benchmark by 10x, and the discrepancy has to be settled before anything here is
        # optimised.  Two numbers per call:
        #   launch  - wall time until every kernel for this step has been *issued* (host side only)
        #   total   - the same after a device synchronise, i.e. what the step really costs
        # Their difference is the device time; printed every 20 calls so a serving log shows whether
        # a 512-token prefill step is ~3 ms (the isolated measurement) or ~30 ms (the phase table).
        _timing = _prefill_timing_enabled()
        if _timing:
            _t0 = time.perf_counter()

        qo = _query_lengths(meta, num_reqs, num_tokens)  # host list, no device sync
        dev = q3.device
        k_pool = pool_sub_view(k_half, sub)
        v_pool = pool_sub_view(v_half, sub)

        state = self._state_for(
            num_reqs,
            int(meta.block_table_tensor.shape[1]),
            block_size,
            q3,
            int(k_pool.shape[0]),
        )

        # The ragged triple at *kernel* page granularity, rebuilt on the device exactly like a
        # decode step (`seq_lens` includes this step's tokens, so causal_shift is 0 for a fresh
        # prefill and > 0 for a chunked one — which is what the kernel's mask needs).
        refresh_step(
            state.step,
            meta.block_table_tensor,
            meta.seq_lens,
            block_size,
            state.kv_indices,
            num_reqs=num_reqs,
        )

        max_q = max(qo)
        q4 = self._rectify(q3, qo, num_reqs, max_q, num_tokens)
        schedule = self._schedule(qo, num_reqs, dev)
        # The pack indices are a pure function of the query lengths, so cache them per shape: they
        # were ~0.6 ms of the 1.17 ms host time measured per prefill step (§64.2).
        pack_key = (tuple(qo), num_reqs)
        pack = _PACK_CACHE.get(pack_key)
        if pack is None:
            from tileinfer.kernels.attention.paged_decode_ascend950 import _pack_indices

            pack = _pack_indices(
                schedule,
                self.num_kv_heads,
                max_q * self.group,
                BLOCK_Q,
                schedule.num_tiles,
                num_reqs * self.num_kv_heads * max_q * self.group,
                dev,
            )
            if len(_PACK_CACHE) >= 32:  # bounded: the key set is "shapes we actually serve"
                _PACK_CACHE.clear()
            _PACK_CACHE[pack_key] = pack
        _t_setup = time.perf_counter() if _timing else 0.0

        from tileinfer.kernels.attention.paged_decode_ascend950 import forward_prefill
        from tileinfer.metadata import RaggedMetadata

        ragged = RaggedMetadata(
            kv_indptr=state.step.kv_indptr_buf[: num_reqs + 1],
            kv_indices=state.kv_indices,
            kv_last_page_len=state.step.last_page_len_buf[:num_reqs],
            page_size=KERNEL_PAGE_SIZE,
            qo_indptr=meta.query_start_loc[: num_reqs + 1].to(_I32),
        )
        qo_dev = (meta.query_start_loc[1 : num_reqs + 1] - meta.query_start_loc[:num_reqs]).to(_I32)

        result = forward_prefill(
            q4,
            k_pool,
            v_pool,
            ragged,
            schedule,
            qo_dev,
            group=self.group,
            block_q=BLOCK_Q,
            dtype=_tilelang_dtype(q3.dtype),
            kernel_batch=KERNEL_BATCH,
            pv_single_pass=self.single_pass_pv,
            pack=_PACK_CACHE.get(pack_key),
        )
        out = self._store(result, qo, num_reqs, output, num_tokens)

        if _timing:
            _t_kernel = time.perf_counter()
            out = self._store(result, qo, num_reqs, output, num_tokens)
            _t_store = time.perf_counter()
            torch.npu.synchronize()
            _total = (time.perf_counter() - _t0) * 1000.0
            # Module-level, not per-instance: each ``PrefillPath`` belongs to one attention layer, so
            # an instance-local counter only ever reaches the number of *requests* and never prints
            # (§64.1 — the first implementation made exactly that mistake).
            _TIMING_STATS.append(
                (
                    _total,
                    (_t_setup - _t0) * 1000.0,     # shapes + step buffers + schedule
                    (_t_kernel - _t_setup) * 1000.0,  # kernel launch (tilelang marshalling)
                    (_t_store - _t_kernel) * 1000.0,  # store back into the caller's buffer
                    num_tokens,
                )
            )
            if len(_TIMING_STATS) % 20 == 0:
                def _med(index: int) -> float:
                    values = sorted(s[index] for s in _TIMING_STATS)
                    return values[len(values) // 2]

                toks = [s[4] for s in _TIMING_STATS]
                print(
                    f"[tileinfer/prefill-timing] calls={len(_TIMING_STATS)} "
                    f"tokens/call={sum(toks) / len(toks):.0f} total={_med(0):.2f}ms "
                    f"setup={_med(1):.2f} launch={_med(2):.2f} store={_med(3):.2f}",
                    flush=True,
                )
                _TIMING_STATS.clear()
            return out
        return self._store(result, qo, num_reqs, output, num_tokens)

    # ------------------------------------------------------------------ host shapes

    def _rectify(
        self, q3: torch.Tensor, qo: List[int], num_reqs: int, max_q: int, num_tokens: int
    ) -> torch.Tensor:
        """``[num_tokens, heads, dim]`` (request-major, ragged) -> ``[batch, heads, max_q, dim]``.

        The kernel indexes the query as a rectangle per request and takes the rows
        ``(query position, head)`` inside one KV head, while the layer hands us
        ``(token, head, dim)``.  A batch of one at exactly ``max_q`` rows is *not* a view: the two
        layouts put the head axis in a different major order, so even the single-request case is
        one transpose (a pass over Q, negligible next to the K/V a prefill reads).
        """
        heads, dim = self.num_heads, self.head_size
        if num_reqs == 1 and max_q == num_tokens:
            return q3.transpose(0, 1).unsqueeze(0).contiguous()
        rect = q3.new_zeros((num_reqs, heads, max_q, dim))
        off = 0
        for b, length in enumerate(qo):
            rect[b, :, :length].copy_(q3[off : off + length].transpose(0, 1))
            off += length
        return rect

    def _schedule(self, qo: List[int], num_reqs: int, dev: torch.device) -> TileSchedule:
        """Per-step tile schedule over the *flattened* ``(query position, head)`` rows.

        ``page_counts`` is passed as zeros on purpose: at ``kv_tile_pages = 0`` the partitioner
        only uses it to fill ``kv_page_starts``/``kv_page_lens``, which the plain prefill kernel
        does not read (it derives its own per-tile page window from the causal frontier inside
        :func:`forward_prefill`).  That keeps this call free of the ``seq_lens`` device read.
        """
        flat = [int(length) * self.group for length in qo]
        live = plan_query_tiles_from_host(
            flat,
            [0] * num_reqs,
            device=dev,
            block_q=BLOCK_Q,
            kv_tile_pages=0,
            load_balance=False,
        )
        bucket = prefill_tile_bucket(live.num_tiles)
        if bucket <= 0:
            raise PrefillUnsupported("shape", "the batch produced no query tiles")
        if bucket == live.num_tiles:
            return live

        def _pad(t: torch.Tensor, fill: int) -> torch.Tensor:
            out = torch.full((bucket,), fill, dtype=_I32, device=dev)
            out[: t.numel()].copy_(t)
            return out

        n = live.num_tiles
        return TileSchedule(
            seq_ids=_pad(live.seq_ids, 0),
            q_offsets=_pad(live.q_offsets, 0),
            q_lens=_pad(live.q_lens, 0),
            kv_page_starts=_pad(live.kv_page_starts, 0),
            kv_page_lens=_pad(live.kv_page_lens, 0),
            split_ids=_pad(live.split_ids, 0),
            # 1, not 0: ``TileSchedule.needs_merge`` is ``(split_counts > 1).any()``, and dead
            # tiles must not look like a KV split that needs a merge kernel.
            split_counts=_pad(live.split_counts, 1),
            order=torch.arange(bucket, dtype=_I32, device=dev),
            balanced=False,
            token_starts=None,
            token_lens=None,
            token_split=False,
        )

    def _store(
        self,
        result: torch.Tensor,
        qo: List[int],
        num_reqs: int,
        output: Optional[torch.Tensor],
        num_tokens: int,
    ) -> torch.Tensor:
        """``[batch, heads, max_q, dim]`` -> the step's ``[num_tokens, heads * dim]`` rows.

        The transpose matters even for a single request: the kernel's result is head-major
        ``[batch, heads, qo, dim]`` while the layer's rows are ``[token, head, dim]``, so
        ``reshape`` would swap the two axes (it reinterprets memory, it does not transpose).
        """
        heads, dim = self.num_heads, self.head_size
        if num_reqs == 1 and int(result.shape[2]) == num_tokens:
            flat = result[0].transpose(0, 1)  # [num_tokens, heads, dim]
            if output is None:
                return flat.reshape(num_tokens, heads * dim).clone()
            out3 = output.reshape(num_tokens, heads, dim)
            out3.copy_(flat)
            if out3.data_ptr() != output.data_ptr():
                output.copy_(out3.reshape_as(output))
            return output

        if output is None:
            parts = [
                result[b, :, :length].transpose(0, 1).reshape(length, heads * dim)
                for b, length in enumerate(qo)
            ]
            return parts[0].clone() if len(parts) == 1 else torch.cat(parts, dim=0)

        out3 = output.reshape(num_tokens, heads, dim)
        off = 0
        for b, length in enumerate(qo):
            out3[off : off + length].copy_(result[b, :, :length].transpose(0, 1))
            off += length
        if out3.data_ptr() != output.data_ptr():
            output.copy_(out3.reshape_as(output))
        return output

    # ------------------------------------------------------------------ buffers

    def _state_for(
        self,
        num_reqs: int,
        table_width: int,
        block_size: int,
        q3: torch.Tensor,
        pool_pages: int,
    ) -> _PrefillState:
        req_bucket = _req_bucket(num_reqs)
        sub = sub_pages(int(block_size))
        width = int(table_width) * sub
        if width < 1:
            raise PrefillUnsupported("shape", "block_table_tensor must have at least one column")
        device = q3.device
        pool_cap = declared_pool_cap(int(pool_pages))
        key = (req_bucket, width, block_size, pool_cap, str(q3.dtype))
        state = self._states.get(key)
        if state is not None:
            return state
        if req_bucket * width >= pool_cap - 1:
            raise PrefillUnsupported(
                "shape",
                f"the staging table ({req_bucket} x {width} kernel pages) does not fit the KV "
                f"capacity ({pool_cap} pages)",
            )
        step = allocate_step_buffers(
            bucket=req_bucket,
            width=width,
            kv_indices_capacity=pool_cap,
            block_size=block_size,
            device=device,
        )
        state = _PrefillState(
            req_bucket=req_bucket,
            width=width,
            block_size=block_size,
            pool_cap=pool_cap,
            dtype=q3.dtype,
            device=device,
            step=step,
            kv_indices=torch.zeros(pool_cap, dtype=_I32, device=device),
        )
        self._states[key] = state
        # WARNING, not INFO: vLLM's logging config filters this package's INFO records.
        logger.warning(
            "TileInfer prefill: prepared kernel path for batch<=%d, width=%d kernel pages, "
            "block_size=%d, pool_cap=%d, dtype=%s (the first step for a tile bucket compiles the "
            "kernel; later steps reuse it)",
            req_bucket,
            width,
            block_size,
            pool_cap,
            q3.dtype,
        )
        return state


def _query_lengths(meta, num_reqs: int, num_tokens: int) -> List[int]:
    """Per-request query lengths, straight from the host copy of ``query_start_loc``."""
    cpu = getattr(meta, "query_start_loc_cpu", None)
    if cpu is None:
        raise PrefillUnsupported("shape", "metadata has no query_start_loc_cpu")
    if int(cpu.numel()) < num_reqs + 1:
        raise PrefillUnsupported(
            "shape",
            f"query_start_loc_cpu has {int(cpu.numel())} entries but {num_reqs} requests need "
            f"{num_reqs + 1}",
        )
    starts = [int(v) for v in cpu[:num_reqs].tolist()]
    ends = [int(v) for v in cpu[1 : num_reqs + 1].tolist()]
    qo = [end - start for start, end in zip(starts, ends)]
    if min(qo) <= 0:
        raise PrefillUnsupported("shape", f"non-positive query length in {qo}")
    if sum(qo) != num_tokens:
        raise PrefillUnsupported(
            "shape",
            f"query lengths {qo} sum to {sum(qo)} but the step schedules {num_tokens} tokens",
        )
    return qo


def _tilelang_dtype(dtype: torch.dtype) -> str:
    table = {
        torch.bfloat16: "bfloat16",
        torch.float16: "float16",
        torch.float32: "float32",
    }
    try:
        return table[dtype]
    except KeyError:  # pragma: no cover - guarded by the backend's dtype validation
        raise PrefillUnsupported("dtype", f"unsupported compute dtype {dtype}") from None
