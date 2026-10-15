"""设备侧 slot 布局重建(D70 接入设计):compact(按 expert 排序)→ per-expert BM tile pad。

vLLM-Ascend 的 ``unified_apply_mlp`` 输入是 compact 布局(hidden_states 按 expert 排序、
``group_list`` cumsum);TileInfer MoE 内核吃 slot 布局(每 expert 的行段独立 pad 到
``block_m`` 边界,``sorted_ids``/``expert_ids`` 寻址,语义见
:mod:`tileinfer.testing.moe_reference`)。本模块用纯 torch 算子(searchsorted/cumsum/
scatter)在设备上零主机同步地完成重建,全部形状静态(``rows_static``/``em_static``
上界),图模式安全。

关键语义(与内核契约逐条对齐):
- ``slot_pos[i]``:compact 行 i 在 slot 布局中的目标行;``X_slot[slot_pos] = X`` 写入,
  pad 行不清零——GEMM 按行独立,pad 行只污染自身输出行,而输出只 gather ``slot_pos``
  有效行,残留垃圾不传播;
- ``sorted_ids[p]``:有效行 = 自身位置 p(< ``rows_static`` → 内核 active 判定通过),
  无效 slot = ``rows_static``(≥ 编译期 R → 整 tile 跳过);
- ``expert_ids[t]``:tile → expert;超出本步实际 tile 数的 pad tile clamp 到 E-1,
  其 ``sorted_ids`` 首行为 ``rows_static`` 必然 inactive,不会读权重。
"""

from __future__ import annotations

import torch

__all__ = ["em_upper_bound", "slot_layout_from_group_list"]


def em_upper_bound(rows_static: int, num_experts: int, block_m: int) -> int:
    """静态 tile 数上界:每 expert 至少 1 个不完整 tile ⇒ cdiv(R,BM) + E - 1。

    再 +1 保护(空 expert 平台/边界取整),开销可忽略(仅多一个 inactive tile)。
    """
    return (rows_static + block_m - 1) // block_m + num_experts


def slot_layout_from_group_list(
    group_list: torch.Tensor,
    num_rows: int,
    rows_static: int,
    em_static: int,
    num_experts: int,
    block_m: int,
) -> dict:
    """由 compact 布局的 ``group_list``(cumsum,type=0)重建 slot 布局索引。

    参数:
      group_list: [E] 单调非降 cumsum(设备张量,int32/int64),``group_list[-1] == num_rows``;
      num_rows: 本步实际 compact 行数 R(动态);
      rows_static: slot 缓冲编译期行数(≥ num_rows,静态上界);
      em_static: tile 数编译期上界(:func:`em_upper_bound`);
      num_experts: E;block_m: tile 行数(生产内核 BM=32)。

    返回 dict:
      slot_pos   [num_rows] int32 — compact 行 → slot 行
      sorted_ids [rows_static] int32 — 内核 active 判定数组(无效 slot = rows_static)
      expert_ids [em_static] int32 — tile → expert(pad tile clamp 到 E-1)
      tiles_actual [1] int64 — 本步实际 tile 数(设备张量,零主机同步;信息用)
    """
    assert group_list.ndim == 1 and group_list.shape[0] == num_experts
    assert rows_static >= num_rows >= 0
    dtype = group_list.dtype
    dev = group_list.device
    rows = torch.arange(num_rows, device=dev, dtype=torch.int64)
    # 每 compact 行的 expert:gl 单调,right=True 给出第一个 gl[e] > row 的 e
    e_row = torch.searchsorted(group_list, rows.to(dtype), right=True)
    prev = torch.cat((torch.zeros(1, dtype=dtype, device=dev), group_list[:-1]))
    seg_len = group_list - prev
    tiles_per_e = (seg_len + (block_m - 1)) // block_m
    tile_base = torch.cumsum(tiles_per_e, 0) - tiles_per_e  # exclusive cumsum
    slot_pos = (tile_base[e_row] * block_m + (rows - prev[e_row].to(torch.int64))).to(torch.int32)
    # active 判定数组:有效行 = 自身位置,无效 slot = rows_static(内核 <R 判定失败)
    sorted_ids = torch.full((rows_static,), rows_static, dtype=torch.int32, device=dev)
    sorted_ids.index_copy_(0, slot_pos.long(), slot_pos)
    tile_ends = (tile_base + tiles_per_e).to(torch.int64)
    tiles = torch.arange(em_static, device=dev, dtype=torch.int64)
    expert_ids = torch.searchsorted(tile_ends, tiles, right=True).clamp_(max=num_experts - 1)
    return {
        "slot_pos": slot_pos,
        "sorted_ids": sorted_ids,
        "expert_ids": expert_ids.to(torch.int32),
        "tiles_actual": tile_ends[-1:].clone(),
    }
