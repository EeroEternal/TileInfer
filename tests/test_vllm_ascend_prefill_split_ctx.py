"""CPU tests for the split-prefill step context and merge (H19).

The split-KV prefill (``TILEINFER_PREFILL_KV_TILE_PAGES``) is validated at three layers; this
file covers the two that need no NPU:

* ``plan_prefill_split_ctx`` — the per-step host context.  Its tile enumeration must match
  ``plan_query_tiles_from_host`` exactly (the kernel consumes that schedule), its page-loop
  truncation must reproduce ``forward_prefill``'s causal-frontier rule, and its pack/gather/
  scatter indices must round-trip the packed layouts they address;
* ``_merge_prefill_partials`` — the log-sum-exp weighted merge, driven here with synthetic
  partials (including ragged split counts under one power-of-two ``ms_compile`` and the
  zero-weight trash slot).

The kernel itself (partials + LSE) is validated against the torch oracle on the device, see
``test_tilelang_ascend950_prefill_split.py``.
"""

from __future__ import annotations

import math

import pytest
import torch

from tileinfer.plan import plan_query_tiles_from_host


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
        kernel_tiles=1,  # no padding for these tests unless overridden
        kv_heads=kv_heads,
        block_q=block_q,
        max_qo_tokens=max_qo or max(qo_tokens),
        page_size=page_size,
        indices=indices,
        device=torch.device("cpu"),
    )


def test_ctx_schedule_matches_plan_and_ms():
    qo = [300, 700, 128]
    pages = [9, 16, 2]  # ragged split counts: ceil(9/4)=3, ceil(16/4)=4, ceil(2/4)=1
    ctx = _build_ctx(qo, pages, [128, 64, 2], ktp=4)
    ref = plan_query_tiles_from_host(
        [q * 8 for q in qo], pages, device=torch.device("cpu"), block_q=128, kv_tile_pages=4,
        load_balance=False,
    )
    assert ctx["ms_compile"] == 4
    assert ctx["num_qblocks"] == 19 + 44 + 8  # ceil(2400/128) + ceil(5600/128) + ceil(1024/128)
    for key in ("seq_ids", "q_offsets", "q_lens", "kv_page_starts", "kv_page_lens", "split_ids"):
        assert torch.equal(getattr(ctx["schedule"], key), getattr(ref, key)), key
    assert torch.equal(ctx["schedule"].split_counts, ref.split_counts)


def test_ctx_pack_gather_scatter_roundtrip():
    """pack_idx reproduces the per-(tile, kv_head) packed rows; scatter inverts the merge layout."""
    group, kv_heads = 8, 2
    qo = [200, 40]
    pages = [5, 3]
    ctx = _build_ctx(qo, pages, [128, 40], group=group, kv_heads=kv_heads, ktp=3, max_qo=200)
    batch, max_qo, dim, br = 2, 200, 4, 128
    q_flat = torch.randn(batch, kv_heads, max_qo * group, dim)
    q_packed = (
        q_flat.reshape(-1, dim).index_select(0, ctx["pack_idx"]).view(-1, br, dim)
    )
    # mirror of plan_query_tiles_from_host's enumeration
    tile = 0
    for b, qo_tokens in enumerate(qo):
        q_rows = qo_tokens * group
        kv_tokens = [0, 0]
        kv_tokens[b] = 1  # placeholder to keep the loop shape honest
        splits = [(s, min(3, pages[b] - s)) for s in range(0, pages[b], 3)]
        for q_start in range(0, q_rows, br):
            rows = min(br, q_rows - q_start)
            for kv_start, kv_len in splits:
                for bh in range(kv_heads):
                    got = q_packed[tile * kv_heads + bh]
                    want = q_flat[b, bh, q_start : q_start + rows]
                    assert torch.equal(got[:rows], want), (tile, bh)
                    assert torch.equal(
                        got[rows:], q_flat[b, bh, 0].expand(br - rows, dim)
                    ), "padding rows must alias real memory"
                tile += 1
    assert tile == ctx["schedule"].num_tiles

    # scatter: merged[g, bh, i] -> out_flat row (b, bh, q_start + i); padding rows resolve to
    # the dummy row.  Verify every live pair from the enumeration.
    out_flat = torch.zeros(batch * kv_heads * max_qo * group + 1, dim)
    merged = torch.arange(
        ctx["num_qblocks"] * kv_heads * br, dtype=torch.float32
    ).unsqueeze(-1).expand(-1, dim)
    out_flat.view(-1, dim).index_copy_(0, ctx["scatter_idx"], merged)
    # recompute the expected destination of merged row (g, bh, i) from the enumeration
    g = 0
    for b, qo_tokens in enumerate(qo):
        q_rows = qo_tokens * group
        for q_start in range(0, q_rows, br):
            rows = min(br, q_rows - q_start)
            for bh in range(kv_heads):
                for i in range(rows):
                    dst = (b * kv_heads + bh) * (max_qo * group) + q_start + i
                    assert torch.equal(out_flat[dst], merged[(g * kv_heads + bh) * br + i])
            g += 1
    assert g == ctx["num_qblocks"]


def test_ctx_page_lens_causal_frontier():
    """Page loop = clamp(needed - start, 0, kv_len): splits past the frontier run zero pages."""
    page_size = 128
    # one request: 40 q tokens over a 2000-token cached prefix (append) -> shift = 1960
    qo, pages, lpl = [40], [16, ], [16 * page_size - 2000 + 2000]  # kv=2000 exact pages? 2000/128=15.625
    pages, lpl = [16], [16 * page_size - 2000 + 2000]
    pages, lpl = [16], [16 * page_size - 16 * page_size + 2000 - 15 * page_size]
    # simpler: kv = 16*128 = 2048 tokens cached, qo=40 -> shift = 2048-40 = 2008
    qo, pages, lpl = [40], [16], [128]
    ctx = _build_ctx(qo, pages, lpl, ktp=4, max_qo=40, page_size=page_size)
    group = 8
    rows = 40 * group  # 320 rows -> 3 q-blocks (128,128,64)
    shift = 16 * 128 - 40
    exp_needed = []
    for q_start in range(0, rows, 128):
        r = min(128, rows - q_start)
        q_pos_last = (q_start + r - 1) // group
        exp_needed.append((q_pos_last + shift) // page_size + 1)
    lens = ctx["tile_page_lens"].tolist()
    # q-block 0: rows 0..127 -> q_pos 0..15, frontier 15+2008=2023 -> needed 16 pages: all 4 splits full
    assert lens[0:4] == [4, 4, 4, 4]
    # q-block 1: q_pos 16..31, frontier 2039 -> needed 16: all splits full
    assert lens[4:8] == [4, 4, 4, 4]
    # q-block 2: q_pos 32..39, frontier 2047 -> needed 16: all splits full (kv=16 pages exactly)
    assert lens[8:11] == [4, 4, 4]
    assert exp_needed == [16, 16, 16]
    # now a shorter prefix so later splits fall past the frontier
    qo, pages, lpl = [40], [4], [100]  # kv = 3*128+100 = 484, shift = 484-40 = 444
    ctx = _build_ctx(qo, pages, lpl, ktp=2, max_qo=40, page_size=page_size)
    lens = ctx["tile_page_lens"].tolist()
    shift = 484 - 40
    # q-block 0: frontier = 15+444 = 459 -> needed 4 pages: splits [(0,2),(2,2)] full
    assert lens[0:2] == [2, 2]
    # q-block 1: rows 128..255 -> q_pos 16..31, frontier 475 -> needed 4: full
    assert lens[2:4] == [2, 2]
    # q-block 2: rows 256..319 -> q_pos 32..39, frontier 483 -> needed 4: full
    assert lens[4:6] == [2, 2]
    # frontier at the very last token position: shrink kv so later splits empty out
    qo, pages, lpl = [16], [4], [4]  # kv = 3*128+4 = 388, shift = 372
    ctx = _build_ctx(qo, pages, lpl, ktp=2, max_qo=16, page_size=page_size)
    lens = ctx["tile_page_lens"].tolist()
    # single q-block, q_pos_last = 15, frontier = 387 -> needed = ceil? 387//128+1 = 4: full
    assert lens == [2, 2]
    qo, pages, lpl = [16], [4], [4]
    ctx = _build_ctx(qo, pages, lpl, ktp=1, max_qo=16, page_size=page_size)
    # ktp=1: 4 splits of 1 page; frontier 387 -> needed 4: all full
    assert ctx["tile_page_lens"].tolist() == [1, 1, 1, 1]
    # zero-page split: frontier page < split start
    qo, pages, lpl = [1], [4], [128]  # kv = 512, shift = 511
    ctx = _build_ctx(qo, pages, lpl, ktp=1, max_qo=1, page_size=page_size)
    # q_pos_last = 0, frontier = 511 -> needed = 4 (page 3): split 3 covers it, all non-empty
    assert ctx["tile_page_lens"].tolist() == [1, 1, 1, 1]


def test_row_thr_matches_forward_prefill_rule():
    """RowThr: global kv threshold q_pos + shift for live rows, -1 for padding rows."""
    group = 8
    qo, pages, lpl = [300], [9], [128 - 9 * 128 % 128 or 128]  # kv=1152? keep exact below
    qo, pages, lpl = [300], [9], [96]  # kv = 8*128+96 = 1120, shift = 1120-300 = 820
    ctx = _build_ctx(qo, pages, lpl, group=group, ktp=4, max_qo=300)
    thr = ctx["row_thr"].view(-1, 128)
    shift = 1120 - 300
    q_rows = 300 * group
    t = 0
    for q_start in range(0, q_rows, 128):
        rows = min(128, q_rows - q_start)
        for _split in range(3):
            for i in range(128):
                want = (q_start + i) // group + shift if i < rows else -1
                assert thr[t, i].item() == want, (t, i)
            t += 1
    assert t == ctx["schedule"].num_tiles


def test_merge_partials_synthetic():
    """merge = Σ softmax(lse)·partial with ragged split counts and a zero-weight trash slot."""
    from tileinfer.kernels.attention.paged_decode_ascend950 import _merge_prefill_partials

    group, kv_heads, dim = 8, 2, 6
    qo = [16, 8]           # 1 q-block each (16*8=128 rows -> 1 block; 8*8=64 -> 1 block)
    pages = [5, 2]         # ktp=2 -> splits 3 and 1 (ragged); ms_compile = 4
    ctx = _build_ctx(qo, pages, [128, 128], group=group, kv_heads=kv_heads, ktp=2, max_qo=16)
    ms = ctx["ms_compile"]
    assert ms == 4
    slots = ctx["kernel_slots"]
    part_rows = (slots + 1) * kv_heads
    br = ctx["block_q"]
    part_out = torch.randn(part_rows, br, dim)
    part_lse = torch.randn(part_rows, br) * 2.0
    # trash slot: (0, -1e30)
    part_out[-kv_heads:].zero_()
    part_lse[-kv_heads * br :].fill_(-1e30)

    merged = _merge_prefill_partials(part_out, part_lse, ctx, dim=dim, o_dtype=torch.float32)
    assert merged.shape == (ctx["num_qblocks"], kv_heads, br, dim)
    # brute-force per (g, bh, row)
    gmap = []  # (b, q_start, rows)
    for b, qo_tokens in enumerate(qo):
        for q_start in range(0, qo_tokens * group, br):
            gmap.append((b, q_start, min(br, qo_tokens * group - q_start)))
    # rebuild each q-block's split list to know which gathers are trash
    g = 0
    for b, qo_tokens in enumerate(qo):
        splits = [(s, min(2, pages[b] - s)) for s in range(0, pages[b], 2)]
        for _qs in range(0, qo_tokens * group, br):
            rows = min(br, qo_tokens * group - _qs)
            for bh in range(kv_heads):
                parts, lses = [], []
                for s in range(ms):
                    slot = (
                        ctx["gather_idx"][g, s, bh].item()
                        if s < len(splits)
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
    """ktp=0 degenerates to one split per q-block (the plain prefill layout)."""
    ctx = _build_ctx([300, 100], [9, 4], [128, 64], group=8, kv_heads=2, ktp=0, max_qo=300)
    ref = plan_query_tiles_from_host(
        [300 * 8, 100 * 8], [9, 4], device=torch.device("cpu"), block_q=128, kv_tile_pages=0,
        load_balance=False,
    )
    assert ctx["ms_compile"] == 1
    assert torch.equal(ctx["schedule"].split_counts, torch.ones(ref.num_tiles, dtype=torch.int32))
    assert torch.equal(ctx["schedule"].seq_ids, ref.seq_ids)
