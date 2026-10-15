"""CPU 契约测试:设备侧 slot 布局重建(tileinfer.kernels.moe.slot_layout)。

内核消费契约的参照实现 = tileinfer.testing.moe_reference.build_slot_layout(探针链
背书)。本测试断言:由 compact+group_list 重建的 scatter 布局与参照 gather 布局在
「slot → 源行」「tile → expert」「X_slot 内容」三个层面逐元素等价;另覆盖上界公式、
极端分布(全同 expert/空 expert)与 zero-sync 纪律(不触发主机同步的算子集合)。
"""

from __future__ import annotations

import torch

from tileinfer.kernels.moe import em_upper_bound, slot_layout_from_group_list
from tileinfer.testing.moe_reference import build_slot_layout, gather_slots

BM = 32


def _case(tokens: int, topk: int, e: int, seed: int, skewed: bool = False):
    g = torch.Generator().manual_seed(seed)
    if skewed:
        flat = torch.zeros((tokens * topk,), dtype=torch.int64)
        flat[: tokens * topk // 2] = 0
        flat[tokens * topk // 2 :] = torch.randint(1, e, (tokens * topk // 2,), generator=g)
        topk_ids = flat.view(tokens, topk)
    else:
        topk_ids = torch.randint(0, e, (tokens, topk), generator=g)
    x = torch.randn((tokens, 64), generator=g).to(torch.bfloat16)
    return x, topk_ids


def _compact_from_topk(x: torch.Tensor, topk_ids: torch.Tensor, e: int):
    """引擎 dispatch 的等价物:compact 行序 = stable argsort(flat topk_ids),group_list = cumsum。"""
    topk = topk_ids.shape[1]
    expanded = x.unsqueeze(1).expand(-1, topk, -1).reshape(-1, x.shape[1])
    flat = topk_ids.reshape(-1).to(torch.int64)
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=e)
    gl = torch.cumsum(counts, 0).to(torch.int32)
    return expanded[order].contiguous(), gl, order


def _assert_equivalent(x, topk_ids, e, rows_static, em_static):
    num_rows = x.shape[0] * topk_ids.shape[1]
    compact, gl, order = _compact_from_topk(x, topk_ids, e)
    lay = slot_layout_from_group_list(gl, num_rows, rows_static, em_static, e, BM)
    ref = build_slot_layout(topk_ids, BM, num_experts=e)

    # 1) slot → 源行:参照 sorted_ids[s] = expanded 行索引(sentinel=num_expanded);
    #    scatter 布局 X_slot[slot_pos[i]] = compact[i] = expanded[order[i]]
    #    ⇔ ref_sorted[slot_pos[i]] == order[i]
    sp = lay["slot_pos"].to(torch.int64)
    ref_sorted = ref["sorted_ids"].to(torch.int64)
    assert torch.equal(ref_sorted[sp], order.to(torch.int64))
    # 2) tile → expert 一致(两布局的 expert_ids 前缀段)
    actual = int(lay["tiles_actual"][0])
    assert actual == ref["num_tiles"]
    assert torch.equal(lay["expert_ids"][:actual].to(torch.int64), ref["expert_ids"].to(torch.int64))
    # 3) 内核 active 判定数组:有效 slot = 自身位置,无效 = rows_static
    sids = lay["sorted_ids"]
    assert int(sids.numel()) == rows_static
    assert torch.equal(sids[sp], sp.to(torch.int32))
    pad_mask = torch.ones(rows_static, dtype=torch.bool)
    pad_mask[sp] = False
    assert torch.equal(sids[pad_mask], torch.full((int(pad_mask.sum()),), rows_static, dtype=torch.int32))
    # 4) X_slot 内容等价:scatter 后的有效行 == 参照 gather 的行
    x_slot = torch.empty((rows_static, x.shape[1]), dtype=torch.bfloat16)
    x_slot.index_copy_(0, sp, compact)
    ref_x_slots = gather_slots(x.repeat_interleave(topk_ids.shape[1], 0), ref["sorted_ids"], ref["num_expanded"])
    assert torch.equal(x_slot[sp], ref_x_slots[sp])
    # 5) 往返:输出 gather 恢复 compact 行序
    out = x_slot[sp]
    assert torch.equal(out, compact)
    return lay


def test_layout_matches_reference_uniform():
    x, topk_ids = _case(37, 8, 256, seed=1)
    e = 256
    em = em_upper_bound(37 * 8, e, BM)
    _assert_equivalent(x, topk_ids, e, em * BM, em)


def test_layout_matches_reference_skewed():
    x, topk_ids = _case(32, 8, 256, seed=2, skewed=True)
    e = 256
    em = em_upper_bound(32 * 8, e, BM)
    _assert_equivalent(x, topk_ids, e, em * BM, em)


def test_layout_all_rows_single_expert():
    # 极端:全部行落 expert 0(1 个 tile 段 + 255 个空 expert)
    x, _ = _case(8, 4, 256, seed=3)
    topk_ids = torch.zeros((8, 4), dtype=torch.int64)
    e = 256
    em = em_upper_bound(32, e, BM)
    lay = _assert_equivalent(x, topk_ids, e, em * BM, em)
    assert int(lay["tiles_actual"][0]) == 1


def test_layout_pad_tiles_inactive():
    # rows_static 预算放大 4 倍:额外 pad tile 必须 inactive(sentinel),expert clamp 安全
    x, topk_ids = _case(16, 8, 64, seed=4)
    e = 64
    em = em_upper_bound(16 * 8, e, BM)
    rows_static = em * BM * 4
    lay = _assert_equivalent(x, topk_ids, e, rows_static, em_upper_bound(rows_static, e, BM))
    actual = int(lay["tiles_actual"][0])
    sids = lay["sorted_ids"].view(-1, BM)
    assert torch.all(sids[actual:] == rows_static)  # pad tile 首行 sentinel → 内核跳过


def test_em_upper_bound_formula():
    assert em_upper_bound(256, 256, 32) == 8 + 256
    assert em_upper_bound(128, 8, 32) == 4 + 8
    rows = 100
    assert em_upper_bound(rows, 7, 32) == (rows + 31) // 32 + 7


def test_layout_no_host_sync_ops():
    # zero-sync 纪律:布局构建路径只允许 torch.arange/searchsorted/cumsum/index_copy_/
    # full/clamp 等异步安全算子;函数体内不得出现 .item()/.tolist()/int(tensor)
    import inspect

    from tileinfer.kernels.moe import slot_layout as mod

    src = inspect.getsource(mod.slot_layout_from_group_list)
    for forbidden in (".item()", ".tolist()", "int(tiles_actual", "synchronize"):
        assert forbidden not in src
