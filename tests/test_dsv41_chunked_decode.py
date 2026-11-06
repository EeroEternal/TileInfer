"""投机解码前置件 `window_slots_for` 的离线单测（不需要 NPU）。

背景：我们的 `_window_kv` / `window_topk_idxs` 在 `start_pos > 0` 时只支持 `seqlen == 1`。
投机解码要一次验证 k 个 token（`seqlen = k > 1` 且 `start_pos > 0`），必须逐 query 计算环形槽位。
`window_slots_for(pos, win)` 就是那条逐 query 的基础件。
"""

from __future__ import annotations

import torch

from tileinfer.models.dsv41.layer import (
    compress_rope_positions,
    window_slots_for,
    window_topk_idxs,
    write_ring_chunk,
)


def test_first_positions_are_masked():
    # 位置 0：只有槽位 0 有效，其余 -1
    slots = window_slots_for(0, 4).tolist()
    assert slots == [-1, -1, -1, 0]


def test_ring_wraps_once_full():
    # window=4，pos=5：可读绝对位置 2..5 -> 槽位 2,3,0,1
    assert window_slots_for(5, 4).tolist() == [2, 3, 0, 1]


def test_each_position_matches_bruteforce():
    win = 8
    for pos in range(0, 20):
        lo = pos - win + 1
        expect = [(p % win) if p >= 0 else -1 for p in range(lo, pos + 1)]
        assert window_slots_for(pos, win).tolist() == expect, pos


def test_seqlen1_agrees_with_existing_decode_row():
    """seqlen==1 时，逐 query 结果应与现有 decode 分支给出的槽位集合一致（顺序可不同）。"""
    win, start = 8, 11
    per_query = window_slots_for(start, win)
    # window_topk_idxs 返回 [bsz, ..., win]，这里摊平成一行比较
    existing = window_topk_idxs(win, 1, 1, start).reshape(-1)
    assert sorted(per_query.tolist()) == sorted(existing.tolist())


def test_chunked_rows_are_distinct_and_consistent():
    """分块解码：连续 k 个位置各有一行，且第 i 行等于 window_slots_for(start+i)。

    注意：**环形缓冲填满后**（pos >= win-1）相邻两行的槽位集合本就相同（只是顺序滚动），
    所以"相邻差一个槽位"只在未填满时成立——这里取 start=2, win=8 来检验该性质。
    """
    win, start, k = 8, 2, 4
    rows = [window_slots_for(start + i, win) for i in range(k)]
    assert len(rows) == k
    for a, b in zip(rows, rows[1:]):
        # 未填满时：新版多进一个槽位，历史槽位不变
        assert len(set(b.tolist()) - set(a.tolist())) == 1
        assert len(set(a.tolist()) - set(b.tolist())) == 0
    assert torch.equal(torch.stack(rows)[-1], window_slots_for(start + k - 1, win))

    # 填满后：集合相同、顺序滚动（这正是环形缓冲应有的行为）
    full = [window_slots_for(win - 1 + i, win) for i in range(3)]
    assert all(sorted(r.tolist()) == sorted(full[0].tolist()) for r in full)
    assert [r[0].item() for r in full] == [0, 1, 2]


def test_window_topk_idxs_chunked_matches_per_position():
    """start_pos>0 且 seqlen>1 时，每个 query 行必须等于 window_slots_for(start+i)。"""
    win, start, k = 8, 5, 4
    got = window_topk_idxs(win, 2, k, start)          # [bsz=2, k, win]
    assert got.shape == (2, k, win)
    for i in range(k):
        want = window_slots_for(start + i, win).tolist()
        assert got[0, i].tolist() == want
        assert got[1, i].tolist() == want             # bsz 维是广播


def test_window_topk_idxs_seqlen1_unchanged():
    """seqlen==1 的既有行为必须逐位保持（回归门）。"""
    win, start = 8, 11
    got = window_topk_idxs(win, 3, 1, start)
    oldest = start % win + 1
    base = [s for s in list(range(oldest, win)) + list(range(oldest))]
    for b in range(3):
        assert got[b].reshape(-1).tolist() == base


def test_compress_rope_positions_prefill():
    # prefill：从 0 开始，每 ratio 一个压缩条目
    assert compress_rope_positions(0, 4, 3) == [0, 4, 8]
    assert compress_rope_positions(0, 4, 0) == []


def test_compress_rope_positions_single_token_matches_old_formula():
    """单 token 解码必须与老代码 `cos[start_pos + 1 - ratio]` 完全一致。"""
    ratio = 4
    for start in (3, 7, 11, 63):
        assert compress_rope_positions(start, ratio, 1) == [start + 1 - ratio]


def test_compress_rope_positions_chunked_advances_by_ratio():
    ratio = 4
    # 一次 8 个 token（跨两个压缩块）→ 两个条目，位置相差 ratio
    got = compress_rope_positions(3, ratio, 2)
    assert got == [0, 4]
    assert got[1] - got[0] == ratio
    # 三个条目
    assert compress_rope_positions(7, ratio, 3) == [4, 8, 12]


def test_write_ring_chunk_equals_token_by_token():
    """分块写入 k 个 token，必须与"逐个 token 写 k 次"完全一致（含跨环绕）。"""
    win, dim, k = 6, 3, 5
    for start in (0, 1, 4, 5, 11):
        chunk = torch.zeros(1, win, dim)
        kv = torch.arange(k * dim, dtype=torch.float32).reshape(1, k, dim)
        write_ring_chunk(chunk, start, kv, win)

        loop = torch.zeros(1, win, dim)
        for i in range(k):
            loop[0, (start + i) % win] = kv[0, i]
        assert torch.equal(chunk, loop), start


def test_write_ring_chunk_seqlen1_matches_old_slice_write():
    """seqlen==1 时等价于老代码 `ring[:bsz, start % win] = kv.squeeze(1)`。"""
    win, dim = 6, 3
    start = 4
    a = torch.zeros(1, win, dim)
    kv = torch.randn(1, 1, dim)
    write_ring_chunk(a, start, kv, win)

    b = torch.zeros(1, win, dim)
    b[:1, start % win] = kv.squeeze(1)
    assert torch.equal(a, b)


def test_write_ring_chunk_overwrites_stale_drafts():
    """被拒绝的草稿无需回滚：后续写入按新 start_pos 覆盖即可（这里是关键性质）。"""
    win, dim = 4, 2
    ring = torch.zeros(1, win, dim)
    stale = torch.full((1, 3, dim), -1.0)      # 3 个草稿 token（位置 0,1,2）
    write_ring_chunk(ring, 0, stale, win)
    good = torch.full((1, 1, dim), 7.0)        # 只接受 1 个，位置 0 重新写
    write_ring_chunk(ring, 0, good, win)
    assert ring[0, 0].tolist() == [7.0, 7.0]
    # 位置 1、2 仍是旧草稿，但它们已经超出 start_pos=1 的有效范围（由 idxs 掩码掉）
    assert ring[0, 1].tolist() == [-1.0, -1.0]
    assert window_slots_for(0, win).tolist() == [-1, -1, -1, 0]


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
