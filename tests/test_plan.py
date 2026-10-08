"""The scheduler is where serving performance is decided — test it directly."""

from __future__ import annotations

import torch

from tileinfer.plan import (
    WorkspaceManager,
    lpt_order,
    plan_decode_tiles,
    plan_query_tiles,
)


def test_lpt_order_sorts_by_descending_work():
    work = torch.tensor([1.0, 9.0, 4.0, 4.0], dtype=torch.float32)
    order = lpt_order(work)
    assert work[order.long()].tolist() == [9.0, 4.0, 4.0, 1.0]


def test_lpt_balance_beats_natural_order_on_skewed_batch():
    """Under dynamic dispatch (each core picks up the next tile when it frees up), starting with
    the longest tiles is what keeps the makespan down — the classic LPT argument, and the reason
    the scheduler sorts by work instead of preserving arrival order."""
    # two long requests arriving *after* a handful of short ones
    pages = torch.tensor([1, 1, 1, 1, 100, 100], dtype=torch.int32)
    sched = plan_decode_tiles(pages, kv_tile_pages=0, load_balance=True)
    ordered = sched.work[sched.order.long()]
    natural = sched.work

    def makespan(values: torch.Tensor, cores: int = 4) -> float:
        loads = [0.0] * cores
        for v in values.tolist():
            loads[loads.index(min(loads))] += v
        return max(loads)

    assert makespan(ordered) < makespan(natural)


def test_load_balance_flag_controls_ordering():
    pages = torch.tensor([5, 1, 9, 3], dtype=torch.int32)
    balanced = plan_decode_tiles(pages, load_balance=True)
    fifo = plan_decode_tiles(pages, load_balance=False)
    assert balanced.order.tolist() == [2, 0, 3, 1]
    assert fifo.order.tolist() == [0, 1, 2, 3]


def test_decode_split_covers_every_page_exactly_once():
    pages = torch.tensor([10, 3, 25], dtype=torch.int32)
    sched = plan_decode_tiles(pages, kv_tile_pages=8)
    covered = {}
    for t in range(sched.num_tiles):
        b = int(sched.seq_ids[t].item())
        start = int(sched.kv_page_starts[t].item())
        count = int(sched.kv_page_lens[t].item())
        covered.setdefault(b, []).extend(range(start, start + count))
    assert covered[0] == list(range(10))
    assert covered[1] == list(range(3))
    assert covered[2] == list(range(25))
    # request 0 -> 2 tiles, request 1 -> 1 tile, request 2 -> 4 tiles
    assert sched.num_tiles == 7
    assert sched.needs_merge
    assert sched.split_counts.tolist() == [2, 2, 1, 4, 4, 4, 4]


def test_decode_without_split_needs_no_merge():
    pages = torch.tensor([10, 3, 25], dtype=torch.int32)
    sched = plan_decode_tiles(pages, kv_tile_pages=0)
    assert sched.num_tiles == 3
    assert not sched.needs_merge
    assert sched.q_lens.tolist() == [1, 1, 1]


def test_query_tiles_cover_every_row_once():
    qo_lens = torch.tensor([5, 200, 1], dtype=torch.int32)
    pages = (qo_lens + 127) // 128
    sched = plan_query_tiles(qo_lens, pages, block_q=128, load_balance=False)
    covered = {}
    for t in range(sched.num_tiles):
        b = int(sched.seq_ids[t].item())
        start = int(sched.q_offsets[t].item())
        count = int(sched.q_lens[t].item())
        covered.setdefault(b, []).extend(range(start, start + count))
    assert covered[0] == list(range(5))
    assert covered[1] == list(range(200))
    assert covered[2] == [0]


def test_workspace_manager_reuses_buffers_by_signature():
    manager = WorkspaceManager()
    spec = {"partial_out": ((4, 32, 128), torch.float32, torch.device("cpu"))}
    first = manager.get(("sig", 1), spec)
    second = manager.get(("sig", 1), spec)
    assert first is second
    third = manager.get(("sig", 2), spec)
    assert third is not first
    assert first["partial_out"].shape == (4, 32, 128)
