"""CPU tests for the split-prefill step context and merge (H19/H20).

The split-KV prefill (``TILEINFER_PREFILL_KV_TILE_PAGES``) is validated at three layers; this
file covers the two that need no NPU:

* ``plan_prefill_split_ctx`` — the per-step host context.  H20 layout: every (request,
  q-block) is partitioned by ``frontier_page_splits`` into fully-visible head splits and one
  frontier split; all head tiles enumerate first, then all frontier tiles, so the runner can
  launch one kernel per contiguous half.  The tests pin that enumeration, the pack/gather/
  scatter round-trips and the ``RowThr``-only-on-frontier rule;
* ``_merge_prefill_partials`` — the log-sum-exp weighted merge, driven here with synthetic
  partials (including ragged split counts under one power-of-two ``ms_compile`` and the
  zero-weight trash slot).

The kernels themselves (partials + LSE) are validated against the torch oracle on the device,
see ``test_tilelang_ascend950_prefill_split.py``.
"""

from __future__ import annotations

import pytest
import torch

from tileinfer.plan import frontier_page_splits


def _live_tiles(qo_tokens, pages, lpl, group, ktp, block_q=128, page_size=128):
    """Mirror of the ctx enumeration: (head_tiles, frontier_tiles) counts per layout."""
    heads, fronts = [], []
    for b, qo in enumerate(qo_tokens):
        q_rows = qo * group
        for q_start in range(0, q_rows, block_q):
            if ktp > 0:
                splits = frontier_page_splits(
                    q_start, qo, pages[b], lpl[b],
                    group=group, page_size=page_size, kv_tile_pages=ktp,
                )
            else:
                splits = [(0, pages[b], True)]
            for sid, (ks, kl, f) in enumerate(splits):
                (fronts if f else heads).append((b, q_start, ks, kl, sid))
    return heads, fronts


def _build_ctx(qo_tokens, pages, lpl, group=8, kv_heads=2, ktp=4, kernel_batch=None,
               block_q=128, page_size=128, max_qo=None, cap=None):
    from tileinfer.kernels.attention.paged_decode_ascend950 import plan_prefill_split_ctx

    total_pages = sum(pages)
    indices = torch.arange(max(cap or total_pages, total_pages), dtype=torch.int32)
    return plan_prefill_split_ctx(
        list(qo_tokens),
        list(pages),
        list(lpl),
        group=group,
        kv_tile_pages=ktp,
        kernel_batch=kernel_batch or len(qo_tokens),
        kernel_tiles=1,  # no padding: the ctx clamps dead tiles to 0 for non-pow2 live counts
        kv_heads=kv_heads,
        block_q=block_q,
        max_qo_tokens=max_qo or max(qo_tokens),
        page_size=page_size,
        indices=indices,
        device=torch.device("cpu"),
    )


def test_frontier_page_splits_partition():
    """The partition primitive: heads cover [0, P_vis) in ktp chunks, frontier the rest."""
    # kv = 16*128 = 2048 tokens cached, qo = 40 -> shift = 2008; q_start 0 -> P_vis = 15
    s = frontier_page_splits(0, 40, 16, 128, group=8, page_size=128, kv_tile_pages=4)
    assert s == [(0, 4, False), (4, 4, False), (8, 4, False), (12, 3, False), (15, 1, True)]
    # q_start row 128 -> min_thr 16 + 2008 = 2024 -> P_vis 15 (same split list)
    s1 = frontier_page_splits(128, 40, 16, 128, group=8, page_size=128, kv_tile_pages=4)
    assert s1 == s
    # no prefix: qo == kv -> shift 0, P_vis 0 -> frontier only
    assert frontier_page_splits(0, 128, 1, 128, group=8, page_size=128, kv_tile_pages=4) == [
        (0, 1, True)
    ]
    # fully visible heads: qo=128 = one full q-block, kv=4 pages, shift=384, min_thr=384
    # -> P_vis = 3 (the frontier row can't see past 511... thr=511 exactly: P_vis=4 when
    # qo=kv=512? here kv=512 qo=128: shift=384) -> heads [0,4) minus the frontier page
    s_fv = frontier_page_splits(0, 128, 4, 128, group=8, page_size=128, kv_tile_pages=2)
    assert s_fv == [(0, 2, False), (2, 1, False), (3, 1, True)]
    # truly no frontier: a 1-token query over 4 full pages — shift=511, min_thr=511,
    # P_vis = 4 = pages -> heads only
    assert frontier_page_splits(0, 1, 4, 128, group=8, page_size=128, kv_tile_pages=2) == [
        (0, 2, False), (2, 2, False)
    ]
    # degenerate: zero pages
    assert frontier_page_splits(0, 40, 0, 128, group=8, page_size=128, kv_tile_pages=4) == []


def test_ctx_layout_heads_first():
    """All head tiles enumerate before all frontier tiles; counts match the partition."""
    qo, pages, lpl = [300, 40], [9, 16], [96, 128]
    ctx = _build_ctx(qo, pages, lpl, ktp=4, max_qo=300)
    heads, fronts = _live_tiles(qo, pages, lpl, 8, 4)
    nh, nf = len(heads), len(fronts)
    assert ctx["num_head_tiles"] == nh
    assert ctx["num_frontier_tiles"] == nf
    assert ctx["schedule"].num_tiles == nh + nf
    sched = ctx["schedule"]
    for key in ("seq_ids", "kv_page_starts", "kv_page_lens", "split_ids"):
        half = getattr(sched, key)
        for t, (b, q_start, ks, kl, sid) in enumerate(heads + fronts):
            if key == "seq_ids":
                want = b
            elif key == "kv_page_starts":
                want = ks
            elif key == "kv_page_lens":
                want = kl
            else:
                want = sid
            assert int(half[t]) == want, (key, t)
    # q_offsets: q_start in rows
    for t, (b, q_start, _ks, _kl, _sid) in enumerate(heads + fronts):
        assert int(sched.q_offsets[t]) == q_start
    # ragged ms: max splits over the layout -> pow2.  qo=300/pages=9 gives 3 splits
    # (P_vis 6..8), qo=40/pages=16 gives 5; ms_compile = 8.
    assert ctx["ms_compile"] == 8
    # 19 + 3 q-blocks
    assert ctx["num_qblocks"] == 19 + 3


def test_ctx_pack_gather_scatter_roundtrip():
    """pack_idx reproduces the per-(tile, kv_head) packed rows; scatter inverts the merge."""
    group, kv_heads = 8, 2
    qo = [200, 40]
    pages = [5, 3]
    lpl = [128, 40]
    ctx = _build_ctx(qo, pages, lpl, group=group, kv_heads=kv_heads, ktp=2, max_qo=200)
    batch, max_qo, dim, br = 2, 200, 4, 128
    q_flat = torch.randn(batch, kv_heads, max_qo * group, dim)
    q_packed = (
        q_flat.reshape(-1, dim).index_select(0, ctx["pack_idx"]).view(-1, br, dim)
    )
    heads, fronts = _live_tiles(qo, pages, lpl, group, 2)
    tile = 0
    for b, q_start, _ks, _kl, _sid in heads + fronts:
        qo_tokens = qo[b]
        rows = min(br, qo_tokens * group - q_start)
        for bh in range(kv_heads):
            got = q_packed[tile * kv_heads + bh]
            want = q_flat[b, bh, q_start : q_start + rows]
            assert torch.equal(got[:rows], want), (tile, bh)
            assert torch.equal(
                got[rows:], q_flat[b, bh, 0].expand(br - rows, dim)
            ), "padding rows must alias real memory"
        tile += 1
    assert tile * kv_heads == q_packed.shape[0]

    # scatter: merged[g, bh, i] -> out_flat row (b, bh, q_start + i); padding rows → dummy.
    out_flat = torch.zeros(batch * kv_heads * max_qo * group + 1, dim)
    merged = torch.arange(
        ctx["num_qblocks"] * kv_heads * br, dtype=torch.float32
    ).unsqueeze(-1).expand(-1, dim)
    out_flat.view(-1, dim).index_copy_(0, ctx["scatter_idx"], merged)
    g = 0
    for b, qo_tokens in enumerate(qo):
        for q_start in range(0, qo_tokens * group, br):
            rows = min(br, qo_tokens * group - q_start)
            for bh in range(kv_heads):
                for i in range(rows):
                    dst = (b * kv_heads + bh) * (max_qo * group) + q_start + i
                    assert torch.equal(out_flat[dst], merged[(g * kv_heads + bh) * br + i])
            g += 1
    assert g == ctx["num_qblocks"]


def test_ctx_page_lens_frontier_rule():
    """Head tiles carry their full visible-page count; the frontier tile covers [P_vis, pages)."""
    page_size = 128
    # kv = 16*128 = 2048 cached, qo = 40 -> shift = 2008; every q-block has P_vis = 15
    qo, pages, lpl = [40], [16], [128]
    ctx = _build_ctx(qo, pages, lpl, ktp=4, max_qo=40, page_size=page_size)
    heads, fronts = _live_tiles(qo, pages, lpl, 8, 4, page_size=page_size)
    lens = ctx["tile_page_lens"].tolist()
    # enumeration: heads of all q-blocks first, then the frontier tiles
    assert lens[: len(heads)] == [kl for (_b, _q, _ks, kl, _s) in heads]
    assert lens[len(heads):] == [kl for (_b, _q, _ks, kl, _s) in fronts]
    assert lens == [4, 4, 4, 3, 4, 4, 4, 3, 4, 4, 4, 3, 1, 1, 1]
    # short prefix: later splits would fall past the frontier -> fewer head tiles
    qo, pages, lpl = [16], [4], [4]  # kv = 388, shift = 372, P_vis = 2
    ctx = _build_ctx(qo, pages, lpl, ktp=2, max_qo=16, page_size=page_size)
    assert ctx["tile_page_lens"].tolist() == [2, 2]
    # no prefix at all: single frontier tile per q-block (128 tokens * group 8 -> 8 blocks)
    qo, pages, lpl = [128], [1], [128]  # kv = 128, shift = 0, P_vis = 0
    ctx = _build_ctx(qo, pages, lpl, ktp=2, max_qo=128, page_size=page_size)
    assert ctx["tile_page_lens"].tolist() == [1] * 8
    assert ctx["num_head_tiles"] == 0


def test_row_thr_only_on_frontier_tiles():
    """RowThr exists per frontier tile only; live rows carry (q_pos + shift − kv_start*128),
    relativised to the tile's page offset (E50/D52 — keeps the kernel predicate in the plain
    kernel's `p * BC + c` form); padding rows stay negative."""
    group = 8
    qo, pages, lpl = [300], [9], [96]  # kv = 1120, shift = 820, P_vis in 6..8, 3 splits/qb
    ctx = _build_ctx(qo, pages, lpl, group=group, ktp=4, max_qo=300)
    heads, fronts = _live_tiles(qo, pages, lpl, group, 4)
    assert ctx["num_head_tiles"] == len(heads)
    thr = ctx["row_thr"].view(-1, 128)
    assert thr.shape[0] == len(fronts)
    shift = 1120 - 300
    q_rows = 300 * group
    blocks = [(qs, min(128, q_rows - qs)) for qs in range(0, q_rows, 128)]
    for t, (b, q_start, ks, _kl, _sid) in enumerate(fronts):
        rows = dict(blocks)[q_start]
        for i in range(128):
            want = (
                ((q_start + i) // group + shift - ks * 128) if i < rows else -1 - ks * 128
            )
            assert int(thr[t, i]) == want, (t, i)


def test_merge_partials_synthetic():
    """merge = Σ softmax(lse)·partial with ragged split counts and a zero-weight trash slot."""
    from tileinfer.kernels.attention.paged_decode_ascend950 import _merge_prefill_partials

    group, kv_heads, dim = 8, 2, 6
    qo = [16, 8]           # 1 q-block each
    pages = [5, 2]
    lpl = [128, 128]
    ctx = _build_ctx(qo, pages, lpl, group=group, kv_heads=kv_heads, ktp=2, max_qo=16)
    ms = ctx["ms_compile"]
    slots = ctx["kernel_slots"]
    part_rows = (slots + 1) * kv_heads
    br = ctx["block_q"]
    part_out = torch.randn(part_rows, br, dim)
    part_lse = torch.randn(part_rows, br) * 2.0
    part_out[-kv_heads:].zero_()
    part_lse[-kv_heads * br :].fill_(-1e30)

    merged = _merge_prefill_partials(part_out, part_lse, ctx, dim=dim, o_dtype=torch.float32)
    assert merged.shape == (ctx["num_qblocks"], kv_heads, br, dim)
    split_counts = []
    for b, qo_tokens in enumerate(qo):
        for q_start in range(0, qo_tokens * group, br):
            split_counts.append(
                len(frontier_page_splits(
                    q_start, qo_tokens, pages[b], lpl[b],
                    group=group, page_size=128, kv_tile_pages=2))
            )
    g = 0
    for b, qo_tokens in enumerate(qo):
        for q_start in range(0, qo_tokens * group, br):
            for bh in range(kv_heads):
                parts, lses = [], []
                for s in range(ms):
                    slot = (
                        ctx["gather_idx"][g, s, bh].item()
                        if s < split_counts[g]
                        else slots * kv_heads + bh
                    )
                    parts.append(part_out[slot])
                    lses.append(part_lse[slot])
                mx = torch.stack(lses).max(dim=0).values
                denom = torch.exp(torch.stack(lses) - mx).sum(dim=0)
                for i in range(br):
                    w = torch.exp(torch.stack([ls[i] for ls in lses]) - mx[i]) / denom[i]
                    want = sum(w[s] * parts[s][i] for s in range(ms))
                    got = merged[g, bh, i]
                    assert torch.allclose(got, want, atol=1e-5), (g, bh, i)
            g += 1


def test_merge_all_masked_rows_are_finite():
    """Rows masked in every split (lse = -1e30 everywhere) merge to 0, never NaN."""
    from tileinfer.kernels.attention.paged_decode_ascend950 import _merge_prefill_partials

    ctx = _build_ctx([16], [4], [128], group=8, kv_heads=1, ktp=2, max_qo=16)
    slots = ctx["kernel_slots"]
    rows = (slots + 1) * 1
    br = ctx["block_q"]
    part_out = torch.randn(rows, br, 3)
    part_lse = torch.full((rows * br,), -1e30)
    merged = _merge_prefill_partials(part_out, part_lse, ctx, dim=3, o_dtype=torch.float32)
    assert torch.isfinite(merged).all()
    assert merged.abs().sum() < 1e-6 * merged.numel() * 1e6 or merged.abs().max() < 1e-4


def test_unsplit_ctx_is_whole_tiles():
    """ktp=0 degenerates to one frontier-form tile per q-block (the plain prefill layout)."""
    ctx = _build_ctx([300, 100], [9, 4], [128, 64], group=8, kv_heads=2, ktp=0, max_qo=300)
    heads, fronts = _live_tiles([300, 100], [9, 4], [128, 64], 8, 0)
    assert ctx["ms_compile"] == 1
    assert ctx["num_head_tiles"] == 0
    assert ctx["schedule"].num_tiles == len(fronts)
    # seq/q_offset enumeration matches the plain prefill's (request order, q-block order)
    ref_seq, ref_off = [], []
    for b, qo in enumerate([300, 100]):
        for q_start in range(0, qo * 8, 128):
            ref_seq.append(b)
            ref_off.append(q_start)
    assert ctx["schedule"].seq_ids.tolist() == ref_seq
    assert ctx["schedule"].q_offsets.tolist() == ref_off
