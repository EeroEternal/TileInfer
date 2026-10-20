"""H67 token 级 decode split(CPU 契约,零设备依赖)。

根因(E291,三链互证):
  TI decode attention 主核单次 0.024ms vs FIA 0.007ms = **3.4x**,与代码注释
  「split-KV measured 3x at batch 1」精确吻合。而 split 以 **KV 页**为单位,
  c1 的 128 token 只有 1 页 → splits=1 → grid = `1 tile × 2 kv_heads = 2 blocks`。
  ⇒ 缺口主体是**并行度**,不是内核算术效率。

本实现:
  ``plan_decode_tiles_token_split`` 按 token 切块(而非整页),
  内核侧用 ``token_starts/token_lens`` 做页内窗口裁剪,使同一页可被多个 tile 分摊。

锁定契约:
  1. tokens_per_split >= page_size 时,split 数/页范围与页级切分**一致**(等价性基线);
  2. tokens_per_split < page_size 时,同一页被多个 tile 覆盖,tile 数按 token 数线性增长;
  3. 各 tile 的 token 窗口**互不重叠、并集完整**(覆盖 [0, total),无空洞无重复);
  4. 每个 tile 的页范围必须**包含**其 token 窗口所需的所有页;
  5. 未启用 token_split 的旧 schedule 行为不变(token_split=False)。
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

try:
    from tileinfer.plan import (
        plan_decode_tiles_token_split,
        plan_query_tiles_from_host,
    )

    _HAVE = True
except Exception:  # pragma: no cover
    _HAVE = False


pytestmark = pytest.mark.skipif(not _HAVE, reason="tileinfer.plan unavailable")

DEV = torch.device("cpu")
PAGE = 128
GROUP = 8


# ---------------------------------------------------------------- 1. 等价性基线


def test_h67_token_split_equals_page_split_when_aligned():
    """tokens_per_split == page_size 时,split 数与页级切分一致(回归基线)。"""
    qo_lens = [1]
    seq_lens = [512]
    pages = [4]

    # 页级基线:显式 ktp=1(每页一个 split),这样 4 页 → 4 tiles
    page_sched = plan_query_tiles_from_host(
        [q * GROUP for q in qo_lens], pages, device=DEV, block_q=128,
        kv_tile_pages=1, load_balance=False,
    )
    tok_sched = plan_decode_tiles_token_split(
        qo_lens, seq_lens, pages, device=DEV, group=GROUP,
        page_size=PAGE, tokens_per_split=PAGE, load_balance=False,
    )

    assert tok_sched.num_tiles == page_sched.num_tiles, (
        f"对齐时 tile 数应相同: {tok_sched.num_tiles} vs {page_sched.num_tiles}"
    )
    assert tok_sched.token_split is True
    assert page_sched.token_split is False  # 旧路径不受影响


# ---------------------------------------------------------------- 2. 短上下文也能切


def test_h67_single_page_gets_multiple_tiles():
    """核心用例:1 页(128 token)c1 场景,页级切分只有 1 tile,token 切分能拿到 4 个。"""
    qo_lens = [1]
    seq_lens = [128]  # 恰好 1 页
    pages = [1]

    page_sched = plan_query_tiles_from_host(
        [q * GROUP for q in qo_lens], pages, device=DEV, block_q=128, load_balance=False
    )
    assert page_sched.num_tiles == 1, "页级切分在 1 页时只有 1 个 tile(这就是缺口)"

    tok_sched = plan_decode_tiles_token_split(
        qo_lens, seq_lens, pages, device=DEV, group=GROUP,
        page_size=PAGE, tokens_per_split=32, load_balance=False,
    )
    assert tok_sched.num_tiles == 4, f"token 切分应得到 4 个 tile,实际 {tok_sched.num_tiles}"

    # 每个 tile 都指向同一页(因为只有 1 页)
    starts = tok_sched.kv_page_starts.tolist()
    lens = tok_sched.kv_page_lens.tolist()
    assert starts == [0, 0, 0, 0], starts
    assert lens == [1, 1, 1, 1], lens


def test_h67_tile_count_scales_with_tokens():
    """tile 数 ≈ ceil(total_tokens / tokens_per_split)。"""
    for total, tps, expect in (
        (128, 32, 4), (128, 16, 8), (512, 128, 4), (512, 32, 16), (100, 32, 4),
    ):
        sched = plan_decode_tiles_token_split(
            [1], [total], [(total + PAGE - 1) // PAGE], device=DEV, group=GROUP,
            page_size=PAGE, tokens_per_split=tps, load_balance=False,
        )
        assert sched.num_tiles == expect, (total, tps, sched.num_tiles, expect)


# ---------------------------------------------------------------- 3. 窗口完整性


def test_h67_token_windows_are_disjoint_and_complete():
    """各 tile 的 token 窗口必须互不重叠且并集完整覆盖 [0, total)。"""
    total = 500
    tps = 32
    sched = plan_decode_tiles_token_split(
        [1], [total], [(total + PAGE - 1) // PAGE], device=DEV, group=GROUP,
        page_size=PAGE, tokens_per_split=tps, load_balance=False,
    )
    starts = sched.token_starts.tolist()
    lens = sched.token_lens.tolist()

    covered = []
    for s, ln in zip(starts, lens):
        assert ln > 0
        covered.extend(range(s, s + ln))

    assert covered == list(range(total)), "窗口必须无缝拼接且不重复"
    assert sched.token_split is True


def test_h67_token_window_covers_its_pages():
    """每个 tile 的页范围必须能覆盖它自己的 token 窗口(否则会漏读 KV)。"""
    total = 1000
    sched = plan_decode_tiles_token_split(
        [1], [total], [(total + PAGE - 1) // PAGE], device=DEV, group=GROUP,
        page_size=PAGE, tokens_per_split=64, load_balance=False,
    )
    for s, ln, p0, pl in zip(
        sched.token_starts.tolist(), sched.token_lens.tolist(),
        sched.kv_page_starts.tolist(), sched.kv_page_lens.tolist(),
    ):
        need_first = s // PAGE
        need_last = (s + ln - 1) // PAGE
        assert p0 <= need_first, (s, ln, p0, need_first)
        assert p0 + pl - 1 >= need_last, (s, ln, p0, pl, need_last)


# ---------------------------------------------------------------- 4. 多请求


def test_h67_multi_request_split_counts():
    """多请求时每个请求独立切分,split_counts 记录各自块数。"""
    qo_lens = [1, 1]
    seq_lens = [128, 256]
    pages = [1, 2]
    sched = plan_decode_tiles_token_split(
        qo_lens, seq_lens, pages, device=DEV, group=GROUP,
        page_size=PAGE, tokens_per_split=64, load_balance=False,
    )
    counts = sched.split_counts.tolist()
    seq_ids = sched.seq_ids.tolist()
    # 请求 0:128/64 = 2 块;请求 1:256/64 = 4 块
    assert counts[:2] == [2, 2], counts
    assert counts[2:] == [4, 4, 4, 4], counts
    assert seq_ids == [0, 0, 1, 1, 1, 1], seq_ids


# ---------------------------------------------------------------- 5. 边界与守卫


def test_h67_rejects_nonpositive_tokens_per_split():
    with pytest.raises(ValueError):
        plan_decode_tiles_token_split(
            [1], [128], [1], device=DEV, group=GROUP,
            page_size=PAGE, tokens_per_split=0,
        )


def test_h67_empty_input_returns_empty_schedule():
    sched = plan_decode_tiles_token_split(
        [], [], [], device=DEV, group=GROUP,
        page_size=PAGE, tokens_per_split=32,
    )
    assert sched.num_tiles == 0
