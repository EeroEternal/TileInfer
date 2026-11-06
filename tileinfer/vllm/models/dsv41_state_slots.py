"""按请求管理 :class:`ModelState` 槽位（P3 并发改造第 2 步）。

设计见 `docs/notes/dsv41-perf-plan.md` §13：

* 每个请求一个槽位，首次出现时用 `factory()` 创建，请求结束后释放；
* **有上限**：`max_slots` 是硬闸门，超过就拒绝（返回 ``None``），避免"并发一高就把主机内存吃光"——
  我们的 `ModelState` 是 bsz=1 分配的一整套缓冲，4 份就是 4 倍；
* 槽位生命周期独立于 `forward` 里的 `pos == 0` 分支，这正是**图捕获**的前置条件（§12）。

本模块是纯 Python 逻辑，可离线单测；`forward` 暂不改用（等第 3 步一起接，届时用开关控制）。
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional


class StateSlots:
    """`req_id -> state` 的有界槽位表。

    ``factory`` 被调用时会收到请求 id，便于日志与按需分配 ``max_seq_len``。
    """

    def __init__(self, factory: Callable[[str], object], max_slots: int = 8) -> None:
        if max_slots < 1:
            raise ValueError("max_slots 必须 >= 1")
        self._factory = factory
        self._max = int(max_slots)
        self._slots: Dict[str, object] = {}
        self.created = 0          # 累计创建次数（测试与日志用）
        self.reused = 0           # 命中次数

    # ---- 查询 ----
    def __len__(self) -> int:
        return len(self._slots)

    def __contains__(self, req_id: str) -> bool:
        return req_id in self._slots

    @property
    def max_slots(self) -> int:
        return self._max

    def ids(self) -> List[str]:
        return list(self._slots)

    def get(self, req_id: str) -> Optional[object]:
        """取已有槽位；不存在返回 ``None``（不创建）。"""
        return self._slots.get(req_id)

    # ---- 生命周期 ----
    def acquire(self, req_id: str) -> Optional[object]:
        """取或创建。**达到上限且不是已有请求时返回 ``None``**，调用方须据此降级/排队。"""
        if req_id in self._slots:
            self.reused += 1
            return self._slots[req_id]
        if len(self._slots) >= self._max:
            return None
        state = self._factory(req_id)
        self._slots[req_id] = state
        self.created += 1
        return state

    def release(self, req_id: str) -> bool:
        """释放槽位；返回是否真的存在过。"""
        return self._slots.pop(req_id, None) is not None

    def prune(self, live: Iterable[str]) -> List[str]:
        """只保留 ``live`` 里的请求，返回被释放的 id 列表。"""
        alive = set(live)
        dead = [k for k in self._slots if k not in alive]
        for k in dead:
            self._slots.pop(k, None)
        return dead

    def clear(self) -> None:
        self._slots.clear()
