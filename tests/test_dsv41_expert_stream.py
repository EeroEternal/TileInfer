"""Flat expert arena (S4 step 3c groundwork): packing, slot reuse and dtype restoration.

Every case runs on a CPU box with small stand-ins for the real fp4/fp8 tensors; what is being pinned
down is the *layout* -- that a slot round-trips bit-exactly, that the byte offsets of the six tensors
are what the kernels will slice, and that the slot cache decides placement without ever handing out an
in-flight slot twice.
"""

from __future__ import annotations

import torch

from tileinfer.models.dsv41.expert_stream import ExpertArena, ExpertSlotStub, expert_specs


def make_expert(dim: int = 8, hidden: int = 4, seed: int = 0):
    """A stand-in expert with the same attribute shape as the real one (w1/w2/w3 + scales)."""
    gen = torch.Generator().manual_seed(seed)
    return ExpertSlotStub(
        w1_weight=torch.randn(dim, hidden, generator=gen),
        w1_scale=torch.rand(dim // 4, hidden // 4, generator=gen),
        w2_weight=torch.randn(hidden, dim, generator=gen),
        w2_scale=torch.rand(hidden // 4, dim // 4, generator=gen),
        w3_weight=torch.randn(dim, hidden, generator=gen),
        w3_scale=torch.rand(dim // 4, hidden // 4, generator=gen),
    )


def test_specs_tile_the_slot_without_gaps():
    expert = make_expert()
    specs = expert_specs(expert)
    expected = 0
    for spec in specs:
        assert spec.offset == expected, "the layout must be gap-free and ordered"
        expected = spec.end
    names = [s.name for s in specs]
    assert names == [
        "w1", "w1.scale", "w2", "w2.scale", "w3", "w3.scale",
    ], "weights and their scales must stay adjacent: the kernels slice them by offset"


def test_pack_round_trips_every_tensor():
    expert = make_expert(seed=7)
    arena = ExpertArena(specs=expert_specs(expert), n_slots=2, pin=False)
    moved = arena.pack(0, expert)
    assert moved == arena.slot_bytes
    views = arena.views(0)
    for name, want in (
        ("w1", expert.w1.weight),
        ("w2", expert.w2.weight),
        ("w3", expert.w3.weight),
        ("w1.scale", expert.w1.scale),
    ):
        got = views[name]
        assert got.shape == want.shape and got.dtype == want.dtype
        assert torch.equal(got, want), f"{name} did not survive the flat round-trip"


def test_two_slots_are_independent():
    a, b = make_expert(seed=1), make_expert(seed=2)
    arena = ExpertArena(specs=expert_specs(a), n_slots=2, pin=False)
    arena.pack(0, a)
    arena.pack(1, b)
    assert torch.equal(arena.views(0)["w1"], a.w1.weight)
    assert torch.equal(arena.views(1)["w1"], b.w1.weight)


def test_slot_cache_decides_placement_and_reuse():
    expert = make_expert()
    arena = ExpertArena(specs=expert_specs(expert), n_slots=2, pin=False)
    s0, hit0 = arena.bind(3, 11)
    arena.pack(s0, expert)
    arena.release(3, 11)
    assert hit0 is False

    s1, hit1 = arena.bind(3, 11)
    assert (s1, hit1) == (s0, True), "a resident expert must come back from the same slot"
    arena.release(3, 11)  # the GEMM consumer is done with it again

    s2, hit2 = arena.bind(5, 42)  # second slot, still free
    assert (s2, hit2) == (1, False) and s2 != s0
    arena.release(5, 42)

    s3, hit3 = arena.bind(9, 99)  # both slots busy -> evict the LRU (3, 11)
    assert hit3 is False and s3 in (0, 1)
    assert not arena.cache.contains((3, 11))
    assert arena.cache.stats.evictions == 1
    arena.release(9, 99)


def test_wrong_sized_tensor_is_rejected_not_silently_skewed():
    expert = make_expert()
    arena = ExpertArena(specs=expert_specs(expert), n_slots=1, pin=False)
    broken = make_expert(dim=16, hidden=4)
    try:
        arena.pack(0, broken)
        raised = False
    except ValueError:
        raised = True
    assert raised, "a shape mismatch must fail loudly -- a skewed slot would corrupt the GEMM"


def test_snapshot_reports_the_arena():
    arena = ExpertArena(specs=expert_specs(make_expert()), n_slots=3, pin=False)
    snap = arena.snapshot()
    assert snap["slots"] == 3
    assert snap["arena_bytes"] == arena.slot_bytes * 3
    assert snap["slot_bytes"] == arena.slot_bytes
