"""共享的 batched KV 容器（`perf-plan.md` §24 的 A 步）。

真批处理前向里，R 个请求不再各自持有一份 `window_kv [1, win, dim]`，而是共享一个
`[R_max, win, dim]` 缓冲，第 i 行属于槽位 i。本模块只做"按槽位读写"这一层，
不涉及模型计算，因此可以完全离线单测。

设计约束：

* **无回滚**：被拒绝的草稿 token 由下一次写入按新的位置覆盖（与 `write_ring_batched` 一致）；
* **视图是拷贝**：`window_view(slots)` 返回的是索引出来的新张量，调用方拿它做 attention
  读操作是安全的；写必须走 `write_window`（否则会写到副本上）。
"""

from __future__ import annotations

from typing import Sequence

import torch

from .layer import write_ring_batched


class BatchedKV:
    """`[R_max, window, dim]` 的共享窗口缓冲。

    ``layer_buffers`` 目前只放窗口 KV（压缩 KV 的结构按层不同，接入时再扩），
    所以构造参数保持最小：形状 + dtype + device。
    """

    def __init__(self, max_slots: int, window: int, dim: int, dtype=torch.bfloat16,
                 device: str = "cpu") -> None:
        if max_slots < 1 or window < 1 or dim < 1:
            raise ValueError("max_slots/window/dim 必须 >= 1")
        self.max_slots = int(max_slots)
        self.window = int(window)
        self.dim = int(dim)
        self.window_kv = torch.zeros(max_slots, window, dim, dtype=dtype, device=device)

    # ---- 写 ----
    def write_window(self, slots: Sequence[int], positions: Sequence[int],
                     kv: torch.Tensor) -> None:
        """把 R 个请求的当前 KV 散写进共享缓冲（``kv`` 形状 ``[R, dim]``）。"""
        if len(slots) != len(positions):
            raise ValueError("slots 与 positions 长度必须一致")
        if kv.shape[0] != len(slots):
            raise ValueError("kv 的第一维必须等于请求数")
        write_ring_batched(self.window_kv, slots, positions, kv, self.window)

    # ---- 读 ----
    def window_view(self, slots: Sequence[int]) -> torch.Tensor:
        """返回这些槽位对应的窗口视图 ``[R, window, dim]``（拷贝）。"""
        idx = torch.as_tensor(list(slots), dtype=torch.long, device=self.window_kv.device)
        return self.window_kv.index_select(0, idx)

    def reset(self) -> None:
        self.window_kv.zero_()
