"""Contract tests for the torch MoE operators (``tileinfer/vllm/ops/moe.py``).

Oracle policy
-------------
Every expected value is produced **here**, by an explicit loop or by plain ``torch`` primitives,
never by the module under test.  Where the project already has an executable spec
(``tileinfer/testing/moe_reference.py``) it is used as an *additional* cross-check, not as the
primary oracle, so a mistake shared between the module and the reference would still show up
against the loops below.

What each op is pinned to
-------------------------
* ``topk_softmax``     softmax over all experts then top-k, ties to the **lower expert index**.
                       Random logits are checked against ``torch.softmax`` + ``torch.topk`` (the
                       task's stated oracle) on unambiguous rows, and every row's *value multiset*
                       is checked exactly, so the tie rule is the only degree of freedom.
* ``moe_align_block_size`` the padded slot layout: sentinel rule, per-expert contiguous runs that
                       are whole multiples of ``block_size``, zero slots for empty experts,
                       ``num_tokens_post_pad`` consistency, plus a brute-force expected layout and
                       the grouped-GEMM addressing contract (``experts_ids[tile]`` is the expert of
                       every real slot in that tile).
* ``fused_experts``    a naive per-token, per-slot loop (SwiGLU in fp32, weighted combine), the
                       production geometry (hidden 2048 / I 512 / 256 experts / top-8) and the
                       duplicate-expert accumulation rule.

Tolerances
----------
``fp32`` in/out: ``rtol=1e-5, atol=1e-5`` (the module also runs its projections in fp32; the only
difference is summation order).
``bf16`` in/out: ``rtol=2e-2, atol=2e-2`` - bf16 carries ~3 decimal digits (relative step 2^-8 =
3.9e-3), and the module casts the fp32 result to bf16 once at the end, so the error budget is the
output rounding plus the bf16 weight/activation quantisation that the oracle shares.
"""

from __future__ import annotations

import pytest
import torch

from tileinfer.vllm.ops import moe as moe_ops

FP32_TOL = dict(rtol=1e-5, atol=1e-5)
BF16_TOL = dict(rtol=2e-2, atol=2e-2)


# =============================================================================================
# Oracles (independent of the module)
# =============================================================================================


def _oracle_topk_softmax_topk(logits: torch.Tensor, k: int):
    """``torch.softmax`` over all experts then ``torch.topk`` - the task's stated oracle."""
    probs = torch.softmax(logits.to(torch.float32), dim=-1)
    values, ids = torch.topk(probs, k, dim=-1)
    return values, ids


def _oracle_topk_lower_index_wins(logits: torch.Tensor, k: int):
    """Explicit "largest probability, ties to the lowest expert index" selection loop."""
    probs = torch.softmax(logits.to(torch.float32), dim=-1)
    picks = []
    for t in range(probs.shape[0]):
        row = probs[t].tolist()
        chosen = []
        for _ in range(k):
            best = -1
            for e in range(len(row)):
                if e in chosen:
                    continue
                if best < 0 or row[e] > row[best]:
                    best = e
            chosen.append(best)
        picks.append(chosen)
    ids = torch.tensor(picks, dtype=torch.int64).reshape(probs.shape[0], k)
    return probs.gather(1, ids), ids


def _oracle_align(topk_ids: torch.Tensor, num_experts: int, block_size: int):
    """Brute-force expected slot layout: buckets per expert, pad each run, cumulative offsets."""
    flat = topk_ids.reshape(-1).tolist()
    num_expanded = len(flat)
    buckets = [[] for _ in range(num_experts)]
    for pair_idx, expert in enumerate(flat):
        buckets[expert].append(pair_idx)  # ascending flat pair index within an expert
    sorted_ids: list[int] = []
    experts_ids: list[int] = []
    for expert, run in enumerate(buckets):
        if not run:
            continue  # an empty expert contributes no slots at all
        tiles = (len(run) + block_size - 1) // block_size
        sorted_ids.extend(run)
        sorted_ids.extend([num_expanded] * (tiles * block_size - len(run)))
        experts_ids.extend([expert] * tiles)
    return sorted_ids, experts_ids, len(sorted_ids)


def _oracle_fused(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    weights: torch.Tensor,
    ids: torch.Tensor,
    expert_map: torch.Tensor | None = None,
    w1_shared: torch.Tensor | None = None,
    w2_shared: torch.Tensor | None = None,
    shared_gate: torch.Tensor | None = None,
    router_weight_on_input: bool = False,
) -> torch.Tensor:
    """Naive per-token, per-slot expert loop in fp32 (the reference the op must match)."""
    tokens, hidden = x.shape
    inter = w1.shape[1] // 2
    top_k = ids.shape[1]
    xf = x.to(torch.float32)
    out = torch.zeros((tokens, hidden), dtype=torch.float32)
    for t in range(tokens):
        for s in range(top_k):
            expert = int(ids[t, s])
            if expert_map is not None:
                expert = int(expert_map[expert])
            if expert < 0:
                continue
            h = xf[t] * float(weights[t, s]) if router_weight_on_input else xf[t]
            gate = h @ w1[expert, :inter, :].to(torch.float32).T
            up = h @ w1[expert, inter:, :].to(torch.float32).T
            act = torch.nn.functional.silu(gate) * up
            y = act @ w2[expert].to(torch.float32).T
            if not router_weight_on_input:
                y = y * float(weights[t, s])
            out[t] += y
    if w1_shared is not None:
        inter_s = w1_shared.shape[0] // 2
        gate = xf @ w1_shared[:inter_s].to(torch.float32).T
        up = xf @ w1_shared[inter_s:].to(torch.float32).T
        shared = (torch.nn.functional.silu(gate) * up) @ w2_shared.to(torch.float32).T
        if shared_gate is not None:
            scale = torch.sigmoid(xf @ shared_gate.reshape(-1).to(torch.float32))
            shared = shared * scale.unsqueeze(1)
        out = out + shared
    return out


def _fill_topk(logits: torch.Tensor, k: int):
    tokens, experts = logits.shape
    topk_weights = torch.empty((tokens, k), dtype=torch.float32)
    topk_ids = torch.empty((tokens, k), dtype=torch.int32)
    token_expert_indices = torch.empty((tokens, k), dtype=torch.int32)
    moe_ops.topk_softmax(topk_weights, topk_ids, token_expert_indices, logits)
    return topk_weights, topk_ids, token_expert_indices


def _ambiguous_rows(probs: torch.Tensor, k: int) -> torch.Tensor:
    """Rows where ``torch.topk``'s (unspecified) tie order could differ from ours."""
    sorted_probs = probs.sort(dim=-1, descending=True).values
    inside = (sorted_probs[:, :k].diff(dim=-1) == 0).any(dim=-1)
    boundary = sorted_probs[:, k - 1] == sorted_probs[:, k]
    return inside | boundary


# =============================================================================================
# topk_softmax
# =============================================================================================


@pytest.mark.parametrize("num_tokens,num_experts,k", [(64, 64, 8), (64, 64, 1), (16, 128, 8)])
def test_topk_softmax_matches_softmax_topk(num_tokens, num_experts, k):
    g = torch.Generator().manual_seed(0)
    logits = torch.randn((num_tokens, num_experts), generator=g)
    weights, ids, _ = _fill_topk(logits, k)
    oracle_w, oracle_ids = _oracle_topk_softmax_topk(logits, k)

    probs = torch.softmax(logits.to(torch.float32), dim=-1)
    unambiguous = ~_ambiguous_rows(probs, k)
    assert bool(unambiguous.any()), "test draws must contain unambiguous rows"

    # Values: exactly the softmax probabilities of the picked experts (same multiset every row,
    # so ordering ties cannot hide a wrong pick).
    assert torch.equal(weights.sort(dim=-1).values, oracle_w.sort(dim=-1).values)
    assert torch.equal(weights[unambiguous], oracle_w[unambiguous])
    assert torch.equal(ids[unambiguous], oracle_ids[unambiguous].to(torch.int32))
    assert ids.dtype == torch.int32 and weights.dtype == torch.float32


def test_topk_softmax_ties_break_to_lower_expert_index():
    # Deliberate ties: the two/three largest logits are identical, so only the tie rule decides.
    cases = [
        (torch.tensor([[2.0, 2.0, 2.0, 1.0, 0.5]]), 2, [0, 1]),
        (torch.tensor([[3.0, 2.0, 2.0, 2.0]]), 2, [0, 1]),          # tie at the k boundary
        (torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0]]), 3, [0, 1, 2]),  # uniform router
        (torch.tensor([[-1.0, -1.0, -2.0]]), 1, [0]),               # tie for the maximum
        (torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]), 2, [[0, 1], [0, 1]]),
    ]
    for logits, k, expected in cases:
        weights, ids, _ = _fill_topk(logits, k)
        expected_ids = torch.tensor(expected, dtype=torch.int32).reshape(ids.shape)
        assert torch.equal(ids, expected_ids), f"tie rule broke for {logits.tolist()}"
        oracle_w, oracle_ids = _oracle_topk_lower_index_wins(logits, k)
        assert torch.equal(ids, oracle_ids.to(torch.int32))
        assert torch.equal(weights, oracle_w)


def test_topk_softmax_k_equals_one_and_k_equals_num_experts():
    g = torch.Generator().manual_seed(1)
    logits = torch.randn((8, 8), generator=g)
    probs = torch.softmax(logits, -1)
    # k = 1: exactly the argmax
    weights, ids, _ = _fill_topk(logits, 1)
    assert torch.equal(ids.reshape(-1), probs.argmax(-1).to(torch.int32))
    torch.testing.assert_close(weights.reshape(-1), probs.max(-1).values, **FP32_TOL)
    # k = num_experts: the full sorted distribution, no expert lost
    weights_all, ids_all, _ = _fill_topk(logits, 8)
    torch.testing.assert_close(weights_all, probs.sort(dim=-1, descending=True).values, **FP32_TOL)
    torch.testing.assert_close(weights_all.sum(-1), torch.ones(8), **FP32_TOL)
    expected_ids = torch.arange(8, dtype=torch.int32).expand(8, 8)
    assert torch.equal(ids_all.sort(dim=-1).values, expected_ids)


def test_topk_softmax_large_256_expert():
    g = torch.Generator().manual_seed(2)
    logits = torch.randn((128, 256), generator=g) * 2.0
    weights, ids, _ = _fill_topk(logits, 8)
    oracle_w, oracle_ids = _oracle_topk_softmax_topk(logits, 8)
    probs = torch.softmax(logits, -1)
    unambiguous = ~_ambiguous_rows(probs, 8)
    assert torch.equal(weights.sort(-1).values, oracle_w.sort(-1).values)
    assert torch.equal(ids[unambiguous], oracle_ids[unambiguous].to(torch.int32))
    # Top-8 of 256 experts: ids must be in range and distinct within a row.
    assert int(ids.max()) < 256 and int(ids.min()) >= 0
    for row in ids:
        assert len(set(row.tolist())) == 8


def test_topk_softmax_bf16_logits_match_fp32_oracle():
    g = torch.Generator().manual_seed(3)
    logits = (torch.randn((32, 64), generator=g) * 2.0).to(torch.bfloat16)
    weights, ids, _ = _fill_topk(logits, 4)
    # bf16 has an 8-bit mantissa, so 64 experts out of 16-bit values tie *constantly*: the tie rule
    # is what makes this case well defined, hence the explicit lower-index oracle.
    oracle_w, oracle_ids = _oracle_topk_lower_index_wins(logits, 4)
    assert torch.equal(weights, oracle_w)
    assert torch.equal(ids, oracle_ids.to(torch.int32))

    # bf16 *output* buffer: documented tolerance (the op casts the fp32 probability down).
    topk_weights = torch.empty((32, 4), dtype=torch.bfloat16)
    topk_ids = torch.empty((32, 4), dtype=torch.int32)
    indices = torch.empty((32, 4), dtype=torch.int32)
    moe_ops.topk_softmax(topk_weights, topk_ids, indices, logits)
    assert topk_weights.dtype == torch.bfloat16
    torch.testing.assert_close(topk_weights.float(), oracle_w, **BF16_TOL)
    assert torch.equal(topk_ids, oracle_ids.to(torch.int32))


def test_topk_softmax_fills_token_expert_indices_both_modes():
    g = torch.Generator().manual_seed(4)
    logits = torch.randn((5, 6), generator=g)
    k = 3
    for mode in moe_ops.TOKEN_EXPERT_INDEX_MODES:
        topk_weights = torch.empty((5, k), dtype=torch.float32)
        topk_ids = torch.empty((5, k), dtype=torch.int32)
        indices = torch.empty((5, k), dtype=torch.int32)
        moe_ops.topk_softmax(
            topk_weights, topk_ids, indices, logits, token_expert_index_mode=mode
        )
        tokens = torch.arange(5, dtype=torch.int32).unsqueeze(1).expand(5, k)
        if mode == "token_expert":
            expected = tokens * 6 + topk_ids
        else:
            expected = tokens
        assert torch.equal(indices, expected), mode
        # The ids themselves do not depend on the mode.
        assert torch.equal(topk_ids, _oracle_topk_softmax_topk(logits, k)[1].to(torch.int32))


def test_topk_softmax_renormalize_sums_to_one():
    g = torch.Generator().manual_seed(5)
    logits = torch.randn((16, 32), generator=g)
    topk_weights = torch.empty((16, 4), dtype=torch.float32)
    topk_ids = torch.empty((16, 4), dtype=torch.int32)
    indices = torch.empty((16, 4), dtype=torch.int32)
    moe_ops.topk_softmax(topk_weights, topk_ids, indices, logits, renormalize=True)
    raw = torch.softmax(logits, -1).gather(1, topk_ids.to(torch.int64))
    torch.testing.assert_close(topk_weights.sum(-1), torch.ones(16), **FP32_TOL)
    torch.testing.assert_close(topk_weights, raw / raw.sum(-1, keepdim=True), **FP32_TOL)


def test_topk_softmax_validation():
    logits = torch.randn(4, 8)
    weights = torch.empty(4, 2)
    ids = torch.empty(4, 2, dtype=torch.int32)
    indices = torch.empty(4, 2, dtype=torch.int32)
    with pytest.raises(ValueError, match="2-D gating_output"):
        moe_ops.topk_softmax(weights, ids, indices, logits.unsqueeze(0))
    with pytest.raises(ValueError, match="top_k=9"):
        moe_ops.topk_softmax(torch.empty(4, 9), torch.empty(4, 9, dtype=torch.int32),
                            torch.empty(4, 9, dtype=torch.int32), logits)
    with pytest.raises(ValueError, match="same shape"):
        moe_ops.topk_softmax(torch.empty(4, 3), ids, indices, logits)
    with pytest.raises(ValueError, match="token_expert_index_mode"):
        moe_ops.topk_softmax(weights, ids, indices, logits, token_expert_index_mode="bogus")


# =============================================================================================
# moe_align_block_size
# =============================================================================================


def _run_align(topk_ids, num_experts, block_size, slack_tiles=2):
    """Call the op on a deliberately over-sized, garbage-filled buffer (proves full overwrite)."""
    counts = torch.bincount(topk_ids.reshape(-1).to(torch.int64), minlength=num_experts)
    tiles = int(((counts + block_size - 1) // block_size).sum())
    num_tiles = tiles + slack_tiles
    sorted_token_ids = torch.full((num_tiles * block_size,), -12345, dtype=torch.int32)
    experts_ids = torch.full((num_tiles,), -12345, dtype=torch.int32)
    num_tokens_post_pad = torch.full((1,), -1, dtype=torch.int32)
    moe_ops.moe_align_block_size(
        topk_ids, num_experts, block_size, sorted_token_ids, experts_ids, num_tokens_post_pad
    )
    return sorted_token_ids, experts_ids, num_tokens_post_pad


def _check_align_invariants(topk_ids, num_experts, block_size, sorted_token_ids, experts_ids,
                            num_tokens_post_pad):
    flat = topk_ids.reshape(-1).to(torch.int64)
    num_expanded = int(flat.numel())
    sentinel = num_expanded
    total = int(num_tokens_post_pad.reshape(-1)[0])
    counts = torch.bincount(flat, minlength=num_experts)
    tiles_per_expert = (counts + block_size - 1) // block_size

    # --- buffer / total bookkeeping ---------------------------------------------------------
    assert int(sorted_token_ids.min()) >= 0, "the whole buffer must be written (no stale slots)"
    assert int(experts_ids.min()) >= 0
    assert total == int(tiles_per_expert.sum()) * block_size
    assert total <= sorted_token_ids.numel()
    assert total // block_size <= experts_ids.numel()
    prefix = sorted_token_ids[:total]
    real = prefix[prefix != sentinel]
    assert int(real.min()) >= 0 and int(real.max()) < num_expanded, "sentinel must be >= R"
    assert int((prefix == sentinel).sum()) == total - num_expanded, "padding fills the run"
    assert torch.equal(real.sort().values, torch.arange(num_expanded, dtype=torch.int32))
    # A run of n tokens occupies exactly ceil(n/block)*block slots, i.e. padding is unavoidable
    # only at the tail of each run -> total is a whole number of tiles.
    assert total % block_size == 0

    # --- grouped-GEMM addressing contract: one expert per tile, no empty tile ---------------
    num_tiles = total // block_size
    grid = prefix.view(num_tiles, block_size)
    for tile in range(num_tiles):
        keep = grid[tile][grid[tile] != sentinel]
        assert keep.numel() > 0, "a padded tile inside the layout is never empty"
        experts = torch.unique(flat[keep.to(torch.int64)])
        assert experts.numel() == 1, "a tile must not span two experts"
        assert int(experts[0]) == int(experts_ids[tile])

    # --- per-expert runs: contiguous tiles, ceil(n/block) tiles, n occupied slots, nothing
    #     for experts with zero tokens -------------------------------------------------------
    slot_expert = torch.full((total,), -1, dtype=torch.int64)
    slot_expert[prefix != sentinel] = flat[real.to(torch.int64)]
    for expert in range(num_experts):
        n = int(counts[expert])
        expected_tiles = int(tiles_per_expert[expert])
        tile_idx = (experts_ids[:num_tiles] == expert).nonzero().reshape(-1)
        assert tile_idx.numel() == expected_tiles, f"expert {expert} tile count"
        if expected_tiles:
            # contiguous run of tiles => contiguous run of slots
            assert int(tile_idx[-1] - tile_idx[0]) == expected_tiles - 1
            slot_start = int(tile_idx[0]) * block_size
            span = slot_expert[slot_start: slot_start + expected_tiles * block_size]
            assert int((span == expert).sum()) == n, f"expert {expert} occupied slots"
            assert int((span != expert).sum()) + n == expected_tiles * block_size
        occupied = (slot_expert == expert).nonzero().reshape(-1)
        assert occupied.numel() == n
        if n:
            assert int(occupied[-1] - occupied[0]) == n - 1, "occupied slots are contiguous"
            # Documented ordering rule: within an expert the pairs keep their ascending flat
            # (token-major) index, so the layout is a pure permutation of the input order.
            pair_ids = prefix[occupied]
            assert torch.equal(pair_ids, pair_ids.sort().values), "run order is unstable"
        else:
            assert expert not in set(experts_ids[:num_tiles].tolist()), "empty expert"
    # experts_ids tail (beyond this step's tiles) is expert 0 / inactive
    assert torch.equal(experts_ids[num_tiles:], torch.zeros(experts_ids.numel() - num_tiles,
                                                            dtype=torch.int32))
    assert torch.equal(sorted_token_ids[total:],
                       torch.full((sorted_token_ids.numel() - total,), sentinel,
                                  dtype=torch.int32))


@pytest.mark.parametrize(
    "num_tokens,top_k,num_experts,block_size,seed",
    [
        (37, 8, 256, 32, 6),
        (16, 8, 64, 16, 7),
        (5, 4, 4, 4, 8),
        (1, 1, 8, 4, 9),
        (9, 2, 3, 8, 10),   # block_size > any run
        (11, 3, 5, 1, 11),  # no padding possible
    ],
)
def test_align_block_size_invariants(num_tokens, top_k, num_experts, block_size, seed):
    g = torch.Generator().manual_seed(seed)
    topk_ids = torch.randint(0, num_experts, (num_tokens, top_k), generator=g)
    buffers = _run_align(topk_ids, num_experts, block_size)
    _check_align_invariants(topk_ids, num_experts, block_size, *buffers)


def test_align_block_size_skewed_and_empty_experts():
    # Expert 0 takes 9 pairs (3 tiles at block=4), every other expert stays empty.
    topk_ids = torch.zeros((9,), dtype=torch.int64)
    buffers = _run_align(topk_ids, 64, 4)
    sorted_ids, experts_ids, npp = buffers
    _check_align_invariants(topk_ids, 64, 4, *buffers)
    assert int(npp[0]) == 12
    assert torch.equal(experts_ids[:3], torch.zeros(3, dtype=torch.int32))
    assert torch.equal(sorted_ids[:9], torch.arange(9, dtype=torch.int32))
    assert torch.equal(sorted_ids[9:12], torch.full((3,), 9, dtype=torch.int32))


def test_align_block_size_brute_force_small_case():
    g = torch.Generator().manual_seed(12)
    num_experts, block_size = 4, 4
    # Two shapes: a spread one (every run fits one tile) and a skewed one where expert 0 needs
    # 3 tiles, so the brute-force comparison also pins multi-tile padding.
    cases = [
        torch.randint(0, num_experts, (5, 2), generator=g),
        torch.tensor([[0, 0], [0, 0], [0, 0], [0, 1], [2, 3]], dtype=torch.int64),
    ]
    for topk_ids in cases:
        sorted_ids, experts_ids, npp = _run_align(topk_ids, num_experts, block_size)
        expected_sorted, expected_experts, expected_total = _oracle_align(
            topk_ids, num_experts, block_size
        )
        assert int(npp[0]) == expected_total
        assert sorted_ids[:expected_total].tolist() == expected_sorted
        assert experts_ids[: len(expected_experts)].tolist() == expected_experts
        # whole-buffer fill beyond the layout
        assert (sorted_ids[expected_total:] == topk_ids.numel()).all()
        assert (experts_ids[len(expected_experts):] == 0).all()


def test_align_block_size_prefill_scale_hot_experts():
    """Production geometry at prefill scale: R = 2048 pairs, block 32, routing concentrated on 8
    of 256 experts -> ~8 tiles per hot expert, 192 empty experts contributing zero slots."""
    g = torch.Generator().manual_seed(50)
    num_experts, block_size, hot = 256, 32, 8
    topk_ids = torch.randint(0, hot, (256, 8), generator=g)
    buffers = _run_align(topk_ids, num_experts, block_size)
    _check_align_invariants(topk_ids, num_experts, block_size, *buffers)
    sorted_ids, experts_ids, npp = buffers
    counts = torch.bincount(topk_ids.reshape(-1), minlength=num_experts)
    expected_total = int((((counts + block_size - 1) // block_size) * block_size).sum())
    assert int(npp[0]) == expected_total
    assert int(npp[0]) > topk_ids.numel(), "skewed routing must actually pad"
    assert int((experts_ids[: expected_total // block_size] >= hot).sum()) == 0
    # every hot expert spans several tiles
    for expert in range(hot):
        assert int((experts_ids == expert).sum()) >= 2


def test_align_block_size_matches_reference_build_slot_layout():
    from tileinfer.testing.moe_reference import build_slot_layout

    g = torch.Generator().manual_seed(13)
    num_experts, block_size = 16, 8
    topk_ids = torch.randint(0, num_experts, (23, 8), generator=g)
    sorted_ids, experts_ids, npp = _run_align(topk_ids, num_experts, block_size)
    ref = build_slot_layout(topk_ids, block_size, num_experts=num_experts)
    assert int(npp[0]) == ref["sorted_ids"].numel()
    assert torch.equal(sorted_ids[: int(npp[0])], ref["sorted_ids"])
    assert torch.equal(experts_ids[: ref["num_tiles"]], ref["expert_ids"])


def test_align_block_size_flat_and_2d_inputs_agree():
    g = torch.Generator().manual_seed(14)
    topk_ids = torch.randint(0, 6, (7, 3), generator=g)
    flat = topk_ids.reshape(-1)
    a = _run_align(topk_ids, 6, 4)
    b = _run_align(flat, 6, 4)
    for x, y in zip(a, b):
        assert torch.equal(x, y)


def test_align_block_size_block_first_alias_matches():
    g = torch.Generator().manual_seed(15)
    topk_ids = torch.randint(0, 5, (6, 2), generator=g)
    num_experts, block_size = 5, 4
    counts = torch.bincount(topk_ids.reshape(-1), minlength=num_experts)
    num_tiles = int(((counts + block_size - 1) // block_size).sum()) + 1
    a_sorted = torch.full((num_tiles * block_size,), -1, dtype=torch.int32)
    a_experts = torch.full((num_tiles,), -1, dtype=torch.int32)
    a_npp = torch.full((1,), -1, dtype=torch.int32)
    b_sorted = a_sorted.clone()
    b_experts = a_experts.clone()
    b_npp = a_npp.clone()
    moe_ops.moe_align_block_size(topk_ids, num_experts, block_size, a_sorted, a_experts, a_npp)
    moe_ops.moe_align_block_size_block_first(
        topk_ids, block_size, num_experts, b_sorted, b_experts, b_npp
    )
    assert torch.equal(a_sorted, b_sorted)
    assert torch.equal(a_experts, b_experts)
    assert torch.equal(a_npp, b_npp)


def test_align_block_size_validation():
    topk_ids = torch.tensor([[0, 1], [2, 9]], dtype=torch.int64)
    with pytest.raises(ValueError, match="num_experts=8"):
        moe_ops.moe_align_block_size(
            topk_ids, 8, 4,
            torch.empty(16, dtype=torch.int32), torch.empty(4, dtype=torch.int32),
            torch.empty(1, dtype=torch.int32),
        )
    with pytest.raises(ValueError, match="sorted_token_ids"):
        moe_ops.moe_align_block_size(
            torch.tensor([[0, 1], [2, 3]]), 8, 4,
            torch.empty(4, dtype=torch.int32), torch.empty(4, dtype=torch.int32),
            torch.empty(1, dtype=torch.int32),
        )
    with pytest.raises(ValueError, match="experts_ids"):
        moe_ops.moe_align_block_size(
            torch.tensor([[0, 1], [2, 3]]), 8, 4,
            torch.empty(16, dtype=torch.int32), torch.empty(1, dtype=torch.int32),
            torch.empty(1, dtype=torch.int32),
        )
    with pytest.raises(ValueError, match="block_size"):
        moe_ops.moe_align_block_size(
            torch.tensor([[0, 1]]), 8, 0,
            torch.empty(16, dtype=torch.int32), torch.empty(4, dtype=torch.int32),
            torch.empty(1, dtype=torch.int32),
        )
    # validate=False skips the host-side range check (documented escape hatch)
    sorted_ids = torch.empty(16, dtype=torch.int32)
    experts_ids = torch.empty(4, dtype=torch.int32)
    npp = torch.empty(1, dtype=torch.int32)
    moe_ops.moe_align_block_size(
        torch.tensor([[0, 1], [2, 3]]), 8, 4, sorted_ids, experts_ids, npp, validate=False
    )
    # 4 distinct experts x 1 token each -> 4 runs -> 4 tiles -> 16 slots
    assert int(npp[0]) == 16


# =============================================================================================
# fused_experts
# =============================================================================================


def _weights_and_ids(ids: torch.Tensor, seed: int) -> torch.Tensor:
    """Positive router weights that sum to one per row (topk_softmax's output shape/dtype)."""
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(ids.shape, generator=g)
    return torch.softmax(logits, dim=-1).float()


def _random_moe(seed, num_experts, hidden, inter, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    w1 = (torch.randn((num_experts, 2 * inter, hidden), generator=g) * 0.05).to(dtype)
    w2 = (torch.randn((num_experts, hidden, inter), generator=g) * 0.05).to(dtype)
    return w1, w2


def test_fused_experts_matches_per_token_loop():
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(20)
    hidden, inter, num_experts = 64, 32, 16
    w1, w2 = _random_moe(21, num_experts, hidden, inter)
    x = torch.randn((8, hidden), generator=g)
    ids = torch.randint(0, num_experts, (8, 2), generator=g)
    weights = _weights_and_ids(ids, 22)

    out = moe_ops.fused_experts(x, w1, w2, weights, ids)
    ref = _oracle_fused(x, w1, w2, weights, ids)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, ref, **FP32_TOL)


def test_fused_experts_256_experts_top8():
    g = torch.Generator().manual_seed(23)
    hidden, inter, num_experts, top_k = 16, 8, 256, 8
    w1, w2 = _random_moe(24, num_experts, hidden, inter)
    x = torch.randn((6, hidden), generator=g)
    ids = torch.stack([torch.randperm(num_experts, generator=g)[:top_k] for _ in range(6)])
    weights = _weights_and_ids(ids, 25)
    out = moe_ops.fused_experts(x, w1, w2, weights, ids)
    ref = _oracle_fused(x, w1, w2, weights, ids)
    torch.testing.assert_close(out, ref, **FP32_TOL)


def test_fused_experts_production_geometry_expert_parallel_shard():
    """hidden 2048 / I 512 / 256 experts / top-8, on a 4-expert local shard (EP).

    Materialising 256 real experts would need ~1.5 GiB of weights; the *routing* geometry is what
    this test pins down, so the shard aliases global ids into 4 local experts (``id % 4``) - the
    same code path an expert-parallel rank takes with a partial map.
    """
    g = torch.Generator().manual_seed(26)
    hidden, inter, global_experts, local_experts, top_k = 2048, 512, 256, 4, 8
    w1 = (torch.randn((local_experts, 2 * inter, hidden), generator=g) * 0.02).to(torch.bfloat16)
    w2 = (torch.randn((local_experts, hidden, inter), generator=g) * 0.02).to(torch.bfloat16)
    x = (torch.randn((3, hidden), generator=g) * 0.5).to(torch.bfloat16)
    ids = torch.stack([torch.randperm(global_experts, generator=g)[:top_k] for _ in range(3)])
    weights = _weights_and_ids(ids, 27)
    expert_map = torch.arange(global_experts) % local_experts

    out = moe_ops.fused_experts(
        x, w1, w2, weights, ids, expert_map=expert_map, global_num_experts=global_experts
    )
    ref = _oracle_fused(x, w1, w2, weights, ids, expert_map=expert_map)
    assert out.dtype == torch.bfloat16 and out.shape == (3, hidden)
    torch.testing.assert_close(out.float(), ref, **BF16_TOL)


def test_fused_experts_duplicate_expert_accumulates():
    g = torch.Generator().manual_seed(28)
    hidden, inter, num_experts = 32, 16, 8
    w1, w2 = _random_moe(29, num_experts, hidden, inter, dtype=torch.float32)
    x = torch.randn((4, hidden), generator=g)
    dup_ids = torch.tensor([[3, 3], [1, 1], [0, 2], [7, 7]], dtype=torch.int64)
    dup_w = torch.tensor([[0.5, 0.5], [0.3, 0.7], [0.9, 0.1], [1.0, 0.0]], dtype=torch.float32)

    out = moe_ops.fused_experts(x, w1, w2, dup_w, dup_ids)
    ref = _oracle_fused(x, w1, w2, dup_w, dup_ids)
    torch.testing.assert_close(out, ref, **FP32_TOL)

    # Same expert selected twice at 0.5/0.5 == once at 1.0 (accumulation, not overwrite).
    single_ids = torch.tensor([[3, 0], [1, 0], [0, 2], [7, 0]], dtype=torch.int64)
    single_w = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.9, 0.1], [1.0, 0.0]], dtype=torch.float32)
    out_single = moe_ops.fused_experts(x, w1, w2, single_w, single_ids)
    torch.testing.assert_close(out[:2], out_single[:2], **FP32_TOL)
    torch.testing.assert_close(out[3], out_single[3], **FP32_TOL)


def test_fused_experts_shared_expert_paths():
    g = torch.Generator().manual_seed(30)
    hidden, inter, num_experts, inter_s = 48, 24, 8, 12
    w1, w2 = _random_moe(31, num_experts, hidden, inter)
    x = torch.randn((5, hidden), generator=g)
    ids = torch.randint(0, num_experts, (5, 2), generator=g)
    weights = _weights_and_ids(ids, 32)
    w1_shared = torch.randn((2 * inter_s, hidden), generator=g) * 0.05
    w2_shared = torch.randn((hidden, inter_s), generator=g) * 0.05
    gate = torch.randn((1, hidden), generator=g) * 0.05

    routed = moe_ops.fused_experts(x, w1, w2, weights, ids)

    plain = moe_ops.fused_experts(x, w1, w2, weights, ids, w1_shared=w1_shared, w2_shared=w2_shared)
    torch.testing.assert_close(
        plain, _oracle_fused(x, w1, w2, weights, ids, w1_shared=w1_shared, w2_shared=w2_shared),
        **FP32_TOL,
    )

    gated = moe_ops.fused_experts(
        x, w1, w2, weights, ids,
        w1_shared=w1_shared, w2_shared=w2_shared, shared_gate_weight=gate,
    )
    torch.testing.assert_close(
        gated,
        _oracle_fused(x, w1, w2, weights, ids, w1_shared=w1_shared, w2_shared=w2_shared,
                      shared_gate=gate),
        **FP32_TOL,
    )
    # Gate is a per-token scalar: sigmoid(x @ gate) scales the whole shared row.
    shared_plain = plain - routed
    shared_gated = gated - routed
    scale = torch.sigmoid(x @ gate.reshape(-1))
    torch.testing.assert_close(shared_gated, shared_plain * scale.unsqueeze(1), **FP32_TOL)


def test_fused_experts_expert_map_drops_non_local_slots():
    g = torch.Generator().manual_seed(33)
    hidden, inter, num_experts = 32, 16, 8
    w1, w2 = _random_moe(34, num_experts, hidden, inter)
    x = torch.randn((4, hidden), generator=g)
    ids = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]], dtype=torch.int64)
    weights = _weights_and_ids(ids, 35)
    # expert 0 lives on another rank; 1..7 are local at rows 1..7; ids 4 and 6 are also remote.
    expert_map = torch.tensor([-1, 1, 2, 3, -1, 5, -1, 7], dtype=torch.int32)
    # Local weight rows are indexed by the map's *value*, so build a full-size weight table.
    out = moe_ops.fused_experts(x, w1, w2, weights, ids, expert_map=expert_map,
                                global_num_experts=num_experts)
    ref = _oracle_fused(x, w1, w2, weights, ids, expert_map=expert_map.long())
    torch.testing.assert_close(out, ref, **FP32_TOL)


def test_fused_experts_router_weight_on_input():
    g = torch.Generator().manual_seed(36)
    hidden, inter, num_experts = 32, 16, 8
    w1, w2 = _random_moe(37, num_experts, hidden, inter)
    x = torch.randn((6, hidden), generator=g)
    ids = torch.randint(0, num_experts, (6, 1), generator=g)
    weights = _weights_and_ids(ids, 38)

    on_input = moe_ops.fused_experts(x, w1, w2, weights, ids, apply_router_weight_on_input=True)
    torch.testing.assert_close(on_input, _oracle_fused(x, w1, w2, weights, ids,
                                                       router_weight_on_input=True), **FP32_TOL)
    # The two placements do NOT coincide in general: silu(w*g) * (w*u) != w * (silu(g) * u), so the
    # flag is a real semantic switch and not a no-op (the next test pins the difference).
    # The documented restriction: undefined for top_k > 1.
    ids2 = torch.randint(0, num_experts, (6, 2), generator=g)
    with pytest.raises(ValueError, match="top_k == 1"):
        moe_ops.fused_experts(x, w1, w2, _weights_and_ids(ids2, 39), ids2,
                              apply_router_weight_on_input=True)


def test_fused_experts_router_weight_on_input_matches_scaled_oracle():
    # With top_k == 1, scaling the input is *not* the same as scaling the output (silu is not
    # homogeneous), so pin the exact behaviour against the oracle that scales the input.
    g = torch.Generator().manual_seed(40)
    hidden, inter, num_experts = 16, 8, 4
    w1, w2 = _random_moe(41, num_experts, hidden, inter)
    x = torch.randn((3, hidden), generator=g)
    ids = torch.tensor([[0], [1], [3]], dtype=torch.int64)
    weights = torch.tensor([[0.4], [0.9], [0.1]], dtype=torch.float32)
    out = moe_ops.fused_experts(x, w1, w2, weights, ids, apply_router_weight_on_input=True)
    torch.testing.assert_close(out, _oracle_fused(x, w1, w2, weights, ids,
                                                  router_weight_on_input=True), **FP32_TOL)
    out_default = moe_ops.fused_experts(x, w1, w2, weights, ids)
    assert not torch.allclose(out, out_default, **FP32_TOL)


def test_fused_experts_bf16_in_out_tolerance():
    g = torch.Generator().manual_seed(42)
    hidden, inter, num_experts = 128, 64, 16
    w1 = (torch.randn((num_experts, 2 * inter, hidden), generator=g) * 0.05).to(torch.bfloat16)
    w2 = (torch.randn((num_experts, hidden, inter), generator=g) * 0.05).to(torch.bfloat16)
    x = (torch.randn((8, hidden), generator=g) * 0.5).to(torch.bfloat16)
    ids = torch.randint(0, num_experts, (8, 4), generator=g)
    weights = _weights_and_ids(ids, 43)

    out = moe_ops.fused_experts(x, w1, w2, weights, ids)
    assert out.dtype == torch.bfloat16
    ref = _oracle_fused(x, w1, w2, weights, ids)
    torch.testing.assert_close(out.float(), ref, **BF16_TOL)
    # And the documented tolerance is not hiding a systematically wrong result: the fp32 path is
    # much closer than the bf16 budget.
    out32 = moe_ops.fused_experts(x.float(), w1.float(), w2.float(), weights, ids)
    torch.testing.assert_close(out32, _oracle_fused(x.float(), w1.float(), w2.float(), weights, ids),
                               **FP32_TOL)


def test_fused_experts_matches_project_reference():
    from tileinfer.testing.moe_reference import reference_moe

    g = torch.Generator().manual_seed(44)
    hidden, inter, num_experts = 64, 32, 16
    w1, w2 = _random_moe(45, num_experts, hidden, inter, dtype=torch.float32)
    x = torch.randn((7, hidden), generator=g)
    ids = torch.randint(0, num_experts, (7, 3), generator=g)
    weights = _weights_and_ids(ids, 46)
    out = moe_ops.fused_experts(x, w1, w2, weights, ids)
    torch.testing.assert_close(out, reference_moe(x, w1, w2, ids, weights), **FP32_TOL)


def test_fused_experts_empty_batch():
    w1, w2 = _random_moe(47, 4, 8, 4)
    x = torch.zeros((0, 8))
    ids = torch.zeros((0, 2), dtype=torch.int64)
    weights = torch.zeros((0, 2))
    out = moe_ops.fused_experts(x, w1, w2, weights, ids)
    assert out.shape == (0, 8)


def test_fused_experts_validation():
    w1, w2 = _random_moe(48, 4, 8, 4)
    x = torch.randn(3, 8)
    ids = torch.zeros((3, 2), dtype=torch.int64)
    weights = torch.rand(3, 2)
    with pytest.raises(NotImplementedError, match="silu"):
        moe_ops.fused_experts(x, w1, w2, weights, ids, activation="gelu")
    with pytest.raises(ValueError, match="expert id"):
        moe_ops.fused_experts(x, w1, w2, weights, torch.full((3, 2), 9))
    with pytest.raises(ValueError, match="w2 must be"):
        moe_ops.fused_experts(x, w1, w2.transpose(1, 2).contiguous(), weights, ids)
    with pytest.raises(ValueError, match="w1_shared and w2_shared"):
        moe_ops.fused_experts(x, w1, w2, weights, ids, w1_shared=torch.randn(4, 8))
    with pytest.raises(ValueError, match="shared_gate_weight requires"):
        moe_ops.fused_experts(x, w1, w2, weights, ids, shared_gate_weight=torch.randn(8))
    with pytest.raises(ValueError, match="topk_weights"):
        moe_ops.fused_experts(x, w1, w2, weights[:, :1], ids)
    # expert_map values are local row indices into w1/w2, not global ids
    with pytest.raises(ValueError, match="local expert"):
        moe_ops.fused_experts(x, w1, w2, weights, torch.tensor([[0, 1], [1, 2], [2, 3]]),
                              expert_map=torch.tensor([0, 99, 1, 2], dtype=torch.int32),
                              global_num_experts=4)


def test_silu_and_mul_matches_torch_formula():
    g = torch.Generator().manual_seed(49)
    x = torch.randn((4, 16), generator=g)
    gate, up = x[:, :8], x[:, 8:]
    expected = torch.nn.functional.silu(gate) * up
    torch.testing.assert_close(moe_ops.silu_and_mul(x), expected, **FP32_TOL)
    xb = x.to(torch.bfloat16)
    out = moe_ops.silu_and_mul(xb)
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out.float(), torch.nn.functional.silu(xb[:, :8].float()) * xb[:, 8:].float(),
                               **BF16_TOL)
    with pytest.raises(ValueError, match="even last dimension"):
        moe_ops.silu_and_mul(torch.randn(2, 5))
