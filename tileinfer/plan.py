"""Plan stage: turn a batch description into a fixed-shape, load-balanced work schedule.

The plan/run split exists for three reasons, all of them serving-driven:

1. **Graph capture.**  Serving engines capture the decode step into an ACLGraph.  Anything that
   depends on the *values* of the batch (sequence lengths, page counts) must therefore happen
   before capture, in :func:`plan`.  The run stage only consumes tensors of static shape.
2. **Load balancing.**  On Ascend, kernel blocks (``cid``) are dispatched to AI cores in order.
   If block ``i`` is scheduled to core ``i % num_cores``, then a schedule whose work is unevenly
   distributed leaves cores idle.  Real serving batches are extremely skewed (one 100k-token
   request next to a 20-token one), so we re-partition the work at plan time: long requests are
   split along the KV axis, and the resulting units are ordered LPT-style (longest processing
   time first) so that per-core work is near-uniform.
3. **Kernel selection / JIT cache.**  The plan holds the concrete kernel variant (block sizes,
   dtypes, tile counts), so the run stage does a dict lookup instead of deciding.

The plan is deliberately a *host-side* object made of int32 tensors: it can be built with plain
Python control flow and, once built, is a static-shape description of one step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

from .metadata import AttentionMode, RaggedMetadata, as_int32, pad_to_multiple

__all__ = [
    "TileSchedule",
    "AttentionPlan",
    "Workspace",
    "WorkspaceManager",
    "plan_query_tiles",
    "plan_decode_tiles",
    "lpt_order",
]


# ---------------------------------------------------------------------------------------
# Work tiles
# ---------------------------------------------------------------------------------------


@dataclass
class TileSchedule:
    """A flat, device-friendly list of work tiles.

    A *tile* is the unit of parallel work handed to one kernel block: it reads
    ``q_lens[i]`` query rows starting at ``q_offsets[i]`` inside request ``seq_ids[i]`` and
    attends over ``kv_page_lens[i]`` KV pages starting at ``kv_page_starts[i]``.  When a
    request is split along the KV axis (``split_counts[i] > 1``) each tile only produces a
    *partial* output that a later merge step combines using ``partial_lse``.
    """

    seq_ids: torch.Tensor  # [num_tiles] int32
    q_offsets: torch.Tensor  # [num_tiles] int32, row offset inside the request
    q_lens: torch.Tensor  # [num_tiles] int32
    kv_page_starts: torch.Tensor  # [num_tiles] int32, page offset inside the request
    kv_page_lens: torch.Tensor  # [num_tiles] int32
    split_ids: torch.Tensor  # [num_tiles] int32
    split_counts: torch.Tensor  # [num_tiles] int32
    order: Optional[torch.Tensor] = None  # [num_tiles] int32, LPT dispatch order
    balanced: bool = False

    @property
    def num_tiles(self) -> int:
        return int(self.seq_ids.numel())

    @property
    def max_q_len(self) -> int:
        return int(self.q_lens.max().item()) if self.num_tiles else 0

    @property
    def max_kv_pages(self) -> int:
        return int(self.kv_page_lens.max().item()) if self.num_tiles else 0

    @property
    def needs_merge(self) -> bool:
        return bool((self.split_counts > 1).any()) if self.num_tiles else False

    @property
    def work(self) -> torch.Tensor:
        """Crude per-tile cost model: query rows x KV *tokens* (not pages)."""
        return (self.q_lens.to(torch.float32) * self.kv_page_lens.to(torch.float32))

    def to(self, device: torch.device, non_blocking: bool = False) -> "TileSchedule":
        """Move the whole schedule to ``device`` (plans are tiny; this is cheap)."""
        return TileSchedule(
            seq_ids=self.seq_ids.to(device, non_blocking=non_blocking),
            q_offsets=self.q_offsets.to(device, non_blocking=non_blocking),
            q_lens=self.q_lens.to(device, non_blocking=non_blocking),
            kv_page_starts=self.kv_page_starts.to(device, non_blocking=non_blocking),
            kv_page_lens=self.kv_page_lens.to(device, non_blocking=non_blocking),
            split_ids=self.split_ids.to(device, non_blocking=non_blocking),
            split_counts=self.split_counts.to(device, non_blocking=non_blocking),
            order=None if self.order is None else self.order.to(device, non_blocking=non_blocking),
            balanced=self.balanced,
        )

    def signature(self) -> Tuple[Any, ...]:
        counts = torch.bincount(self.q_lens.to(torch.int64)).tolist() if self.num_tiles else []
        return (
            self.num_tiles,
            self.max_q_len,
            self.max_kv_pages,
            int(self.split_counts.max().item()) if self.num_tiles else 1,
            tuple(counts[:16]),
            self.balanced,
        )


def lpt_order(work: torch.Tensor) -> torch.Tensor:
    """Longest-processing-time-first dispatch order.

    Sorting tiles by descending work (and, as a tie-break, descending KV pages) is what turns
    a skewed batch into a balanced one: the heaviest tiles are handed to the first cores, so
    the tail of the schedule is made of cheap tiles.  This is the same idea as FlashInfer's
    load-balanced decode scheduling, and it is free because it happens on the host.
    """
    if work.numel() == 0:
        return torch.empty(0, dtype=torch.int32, device=work.device)
    ordered = torch.argsort(work, descending=True, stable=True)
    return ordered.to(torch.int32)


def _empty_schedule(device: Optional[torch.device]) -> TileSchedule:
    z = torch.empty(0, dtype=torch.int32, device=device)
    return TileSchedule(
        seq_ids=z,
        q_offsets=z,
        q_lens=z,
        kv_page_starts=z,
        kv_page_lens=z,
        split_ids=z,
        split_counts=z,
        order=z,
    )


# ---------------------------------------------------------------------------------------
# Partitioners
# ---------------------------------------------------------------------------------------


def plan_query_tiles(
    qo_lens: torch.Tensor,
    page_counts: torch.Tensor,
    block_q: int = 128,
    kv_tile_pages: int = 0,
    load_balance: bool = True,
) -> TileSchedule:
    """Prefill / append partitioner: one tile per (request, query block).

    ``kv_tile_pages > 0`` additionally splits each tile along the KV axis (shared-prefix and
    very long context cases), which is what makes cascade/chunked prefill cheap.
    """
    return plan_query_tiles_from_host(
        [int(x) for x in qo_lens.to(torch.int64).cpu().tolist()],
        [int(x) for x in page_counts.to(torch.int64).cpu().tolist()],
        device=qo_lens.device,
        block_q=block_q,
        kv_tile_pages=kv_tile_pages,
        load_balance=load_balance,
    )


def frontier_page_splits(
    q_start_rows: int,
    qo_tokens: int,
    pages: int,
    lpl: int,
    *,
    group: int,
    page_size: int,
    kv_tile_pages: int,
) -> List[Tuple[int, int, bool]]:
    """H20 split-prefill page partitioning for ONE (request, query-block): the
    fully-visible head splits plus the frontier split.

    A KV page ``p`` is *fully visible* to the query block when its last token position
    ``(p + 1) * page_size - 1`` is ``<=`` the **lowest** row's causal threshold
    ``min_thr = q_start_rows // group + (kv_tokens - qo_tokens)`` — every row of the block
    then attends to the whole page, so the head kernel needs no per-row causal predicate
    (which is what sidesteps the tilelang SimtVF multi-block mask bug, E48): head splits tile
    ``[0, P_vis)`` where ``P_vis = (min_thr + 1) // page_size`` into ``kv_tile_pages``-page
    chunks; the single **frontier split** covers ``[P_vis, pages)`` (absent when
    ``P_vis >= pages``) and keeps the plain kernel's per-row ``RowThr`` form
    (multi-block-verified, 0a99edd).

    Returns ``[(kv_start, kv_len, is_frontier), ...]``; ``kv_len`` is the true page count (the
    frontier split is *not* frontier-truncated — the RowThr mask does the row-level cutting).
    """
    assert kv_tile_pages > 0
    if qo_tokens <= 0 or pages <= 0:
        return []
    kv_tokens = (pages - 1) * page_size + lpl
    min_thr = q_start_rows // group + (kv_tokens - qo_tokens)
    p_vis = (min_thr + 1) // page_size
    p_vis = max(0, min(p_vis, pages))
    splits: List[Tuple[int, int, bool]] = []
    for s in range(0, p_vis, kv_tile_pages):
        splits.append((s, min(kv_tile_pages, p_vis - s), False))
    if p_vis < pages:
        splits.append((p_vis, pages - p_vis, True))
    return splits


def plan_query_tiles_from_host(
    qo_lens: List[int],
    page_counts: List[int],
    device: Optional[torch.device] = None,
    block_q: int = 128,
    kv_tile_pages: int = 0,
    load_balance: bool = True,
) -> TileSchedule:
    """:func:`plan_query_tiles` on host lists: zero ``.item()`` syncs for callers that
    already hold the lengths on the host (the integration layer's per-step shared build).

    Semantics are identical to the tensor entry point (the original read the tensors row by
    row via ``.item()`` anyway); ``device`` is where the schedule tensors are allocated.
    """
    seq_ids: List[int] = []
    q_offsets: List[int] = []
    q_lens: List[int] = []
    kv_starts: List[int] = []
    kv_lens: List[int] = []
    split_ids: List[int] = []
    split_counts: List[int] = []

    for b, q_len in enumerate(qo_lens):
        pages = page_counts[b] if b < len(page_counts) else 0
        if q_len <= 0:
            continue
        if kv_tile_pages <= 0:
            splits = [(0, pages)]
        else:
            splits = [
                (start, min(kv_tile_pages, pages - start)) for start in range(0, pages, kv_tile_pages)
            ]
        for q_start in range(0, q_len, block_q):
            for split_id, (kv_start, kv_len) in enumerate(splits):
                seq_ids.append(b)
                q_offsets.append(q_start)
                q_lens.append(min(block_q, q_len - q_start))
                kv_starts.append(kv_start)
                kv_lens.append(kv_len)
                split_ids.append(split_id)
                split_counts.append(len(splits))

    if not seq_ids:
        return _empty_schedule(device)

    schedule = TileSchedule(
        seq_ids=torch.tensor(seq_ids, dtype=torch.int32, device=device),
        q_offsets=torch.tensor(q_offsets, dtype=torch.int32, device=device),
        q_lens=torch.tensor(q_lens, dtype=torch.int32, device=device),
        kv_page_starts=torch.tensor(kv_starts, dtype=torch.int32, device=device),
        kv_page_lens=torch.tensor(kv_lens, dtype=torch.int32, device=device),
        split_ids=torch.tensor(split_ids, dtype=torch.int32, device=device),
        split_counts=torch.tensor(split_counts, dtype=torch.int32, device=device),
        balanced=load_balance,
    )
    schedule.order = lpt_order(schedule.work) if load_balance else torch.arange(
        schedule.num_tiles, dtype=torch.int32, device=device
    )
    return schedule


def plan_decode_tiles(
    page_counts: torch.Tensor,
    block_q: int = 1,
    kv_tile_pages: int = 0,
    load_balance: bool = True,
) -> TileSchedule:
    """Decode partitioner: one query row per request, optionally split along the KV axis.

    ``kv_tile_pages`` is the *target* number of KV pages per tile.  Requests longer than that
    are split into ``ceil(pages / kv_tile_pages)`` tiles; short requests stay whole.  The
    resulting unit count is what gets balanced across cores, so this is the knob that pays off
    for long-context decode with a skewed batch.
    """
    page_counts = page_counts.to(torch.int64)
    seq_ids: List[int] = []
    splits: List[Tuple[int, int]] = []
    for b in range(page_counts.numel()):
        pages = int(page_counts[b].item())
        if pages <= 0:
            continue
        if kv_tile_pages <= 0 or pages <= kv_tile_pages:
            chunk = [(0, pages)]
        else:
            chunk = [
                (start, min(kv_tile_pages, pages - start))
                for start in range(0, pages, kv_tile_pages)
            ]
        for s in chunk:
            seq_ids.append(b)
            splits.append(s)

    if not seq_ids:
        return _empty_schedule(page_counts.device)

    n = len(seq_ids)
    kv_starts = [s[0] for s in splits]
    kv_lens = [s[1] for s in splits]
    split_counts_map: Dict[int, int] = {}
    for b in seq_ids:
        split_counts_map[b] = split_counts_map.get(b, 0) + 1
    running: Dict[int, int] = {}
    split_ids = []
    for b in seq_ids:
        running[b] = running.get(b, 0)
        split_ids.append(running[b])
        running[b] += 1

    schedule = TileSchedule(
        seq_ids=torch.tensor(seq_ids, dtype=torch.int32, device=page_counts.device),
        q_offsets=torch.zeros(n, dtype=torch.int32, device=page_counts.device),
        q_lens=torch.full((n,), max(1, block_q), dtype=torch.int32, device=page_counts.device),
        kv_page_starts=torch.tensor(kv_starts, dtype=torch.int32, device=page_counts.device),
        kv_page_lens=torch.tensor(kv_lens, dtype=torch.int32, device=page_counts.device),
        split_ids=torch.tensor(split_ids, dtype=torch.int32, device=page_counts.device),
        split_counts=torch.tensor(
            [split_counts_map[b] for b in seq_ids], dtype=torch.int32, device=page_counts.device
        ),
        balanced=load_balance,
    )
    schedule.order = (
        lpt_order(schedule.work)
        if load_balance
        else torch.arange(n, dtype=torch.int32, device=page_counts.device)
    )
    return schedule


# ---------------------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------------------


@dataclass
class Workspace:
    """Scratch memory owned by a plan.

    Kept as plain tensors so that an engine can hand them to an ACLGraph capture without any
    further allocation happening inside the captured region.  ``partial_out`` / ``partial_lse``
    are only materialised when the schedule actually splits a request, so the common decode
    path allocates nothing.
    """

    tensors: Dict[str, torch.Tensor] = field(default_factory=dict)

    def __getitem__(self, key: str) -> torch.Tensor:
        return self.tensors[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.tensors.get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self.tensors

    @property
    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.tensors.values())

    def clear(self) -> None:
        self.tensors.clear()


class WorkspaceManager:
    """Reusable, signature-keyed workspace pool.

    Serving steps have a small number of distinct *shapes* but run the same shape thousands of
    times.  Caching workspaces per signature keeps memory flat (no allocator churn, no growth
    during capture) and keeps the run stage allocation-free.
    """

    def __init__(self, max_entries: int = 64) -> None:
        self._pool: Dict[Tuple[Any, ...], Workspace] = {}
        self._max_entries = max_entries

    def get(
        self,
        signature: Tuple[Any, ...],
        spec: Dict[str, Tuple[Tuple[int, ...], torch.dtype, torch.device]],
    ) -> Workspace:
        key = (signature, tuple(sorted((k, v[0], v[1]) for k, v in spec.items())))
        workspace = self._pool.get(key)
        if workspace is None:
            if len(self._pool) >= self._max_entries:
                self._pool.pop(next(iter(self._pool)))
            workspace = Workspace(
                tensors={
                    name: torch.empty(shape, dtype=dtype, device=device)
                    for name, (shape, dtype, device) in spec.items()
                }
            )
            self._pool[key] = workspace
        return workspace

    def clear(self) -> None:
        self._pool.clear()

    def __len__(self) -> int:
        return len(self._pool)


# ---------------------------------------------------------------------------------------
# The plan object
# ---------------------------------------------------------------------------------------


@dataclass
class AttentionPlan:
    """Everything the run stage needs, with nothing left to decide.

    The plan is treated as immutable: rebuilding it is cheap (it is host-side tensor math) and
    happens whenever the batch *shape* changes, not every step.
    """

    mode: AttentionMode
    meta: RaggedMetadata
    schedule: TileSchedule
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    sm_scale: float
    page_size: int
    dtype: torch.dtype
    device: torch.device
    kv_layout: str = "NHD"
    causal: bool = True
    window_left: int = -1
    backend: str = "reference"
    block_q: int = 128
    workspace: Workspace = field(default_factory=Workspace)
    backend_state: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ derived quantities

    @property
    def batch_size(self) -> int:
        return self.meta.batch_size

    @property
    def max_q_len(self) -> int:
        """Static upper bound on query rows per tile — the shape the kernel compiles for."""
        return self.schedule.max_q_len

    @property
    def max_kv_pages(self) -> int:
        return self.schedule.max_kv_pages

    @property
    def needs_merge(self) -> bool:
        return self.schedule.needs_merge

    @property
    def num_splits(self) -> int:
        counts = self.schedule.split_counts
        return int(counts.max().item()) if counts.numel() else 1

    @property
    def gqa_group_size(self) -> int:
        if self.num_kv_heads == 0:
            return 1
        return max(1, self.num_qo_heads // self.num_kv_heads)

    def signature(self) -> Tuple[Any, ...]:
        return (
            self.mode.value,
            self.meta.signature(),
            self.schedule.signature(),
            self.num_qo_heads,
            self.num_kv_heads,
            self.head_dim,
            self.page_size,
            self.kv_layout,
            self.causal,
            self.window_left,
            self.backend,
            self.block_q,
        )

    def summary(self) -> str:
        m = self.meta
        qo = m.qo_lens
        return (
            f"<AttentionPlan {self.mode.value} backend={self.backend} "
            f"batch={self.batch_size} tiles={self.schedule.num_tiles} "
            f"kv_lens=[{int(m.kv_lens.min().item()) if self.batch_size else 0},"
            f"{m.max_kv_len}] "
            f"qo_lens=[{int(qo.min().item()) if qo is not None else 1},"
            f"{int(qo.max().item()) if qo is not None else 1}] "
            f"heads={self.num_qo_heads}/{self.num_kv_heads}x{self.head_dim} "
            f"splits={self.num_splits} balanced={self.schedule.balanced} "
            f"workspace={self.workspace.nbytes / 2**20:.1f}MiB>"
        )


def allocate_workspace(
    plan_like: AttentionPlan,
    manager: WorkspaceManager,
    extra_dtype: torch.dtype = torch.float32,
) -> Workspace:
    """Allocate (or reuse) the scratch buffers implied by ``plan_like``.

    Split schedules need one partial output and one log-sum-exp per tile; the merge step reads
    them in ``split_id`` order, so the layout is ``[num_tiles, ...]`` indexed by tile.
    """
    schedule = plan_like.schedule
    if not schedule.needs_merge:
        return Workspace()
    num_tiles = schedule.num_tiles
    h, d = plan_like.num_qo_heads, plan_like.head_dim
    device = plan_like.device
    spec = {
        "partial_out": ((num_tiles, h, d), extra_dtype, device),
        "partial_lse": ((num_tiles, h), extra_dtype, device),
    }
    return manager.get(plan_like.signature(), spec)
