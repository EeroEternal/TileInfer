"""mmap 权重源：让 host-offload 的 457.78 GiB 变成**可回收页缓存**而不是 anon。

为什么需要（`perf-plan.md` §25）：现在 `load_weights` 把权重 `copy_` 进参数（anon 内存，
内核不可回收）。实测峰值 anon ≈490 GB，在 628 GB 机器上要求 available ≥505 GB，
一旦别的租户同时申请就被 OOM-kill（多次实测 `anon-rss 484–491 GB`）。

改为 mmap（`safetensors.safe_open` + 自持 handle）后，这部分内存是**文件页缓存**，
内核在压力下可以直接回收，而不是杀我们。理论 anon 峰值可降到几十 GB。

**关键陷阱（本会话踩过）**：vLLM 自带的 `safetensors_weights_iterator` 是
`with safe_open(...)` 用完即关，mmap 随之释放；若把参数指向它 yield 的张量，
之后访问就是悬垂指针 → 原生段错误（无 Python traceback，正是我们观测到的
"engine core exited unexpectedly"）。所以本模块**自己持有 handle 并保活**，直到显式 `close()`。
"""

from __future__ import annotations

import glob
import os
from typing import Dict, List, Optional

from safetensors import safe_open


class MmapWeightSource:
    """按名字取 mmap 视图；所有 shard 的 handle 在这里保活。"""

    def __init__(self, model_dir: str) -> None:
        paths = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
        if not paths:
            raise FileNotFoundError(f"no *.safetensors under {model_dir!r}")
        self._paths = paths
        # 每个 shard 一个打开句柄；**列表成员会一直存活**，保证 mmap 不被释放
        self._handles: List[object] = [safe_open(p, "pt", device="cpu") for p in paths]
        self._owner: Dict[str, int] = {}
        for i, h in enumerate(self._handles):
            for name in h.keys():
                self._owner[name] = i
        self._closed = False

    # ---- 查询 ----
    def __contains__(self, name: str) -> bool:
        return name in self._owner

    def names(self) -> List[str]:
        return list(self._owner)

    @property
    def num_shards(self) -> int:
        return len(self._handles)

    # ---- 取张量 ----
    def get(self, name: str):
        """返回该名字的**映射视图**（不复制；句柄由本对象保活）。"""
        if self._closed:
            raise RuntimeError("MmapWeightSource 已关闭，不能再取张量")
        idx = self._owner.get(name)
        if idx is None:
            raise KeyError(name)
        return self._handles[idx].get_tensor(name)

    def get_optional(self, name: str):
        return self.get(name) if name in self._owner else None

    # ---- 生命周期 ----
    def close(self) -> None:
        """显式释放所有 mmap。**只有在所有视图都不再被使用后**才能调用。"""
        for h in self._handles:
            try:
                h.__exit__(None, None, None)
            except Exception:  # noqa: BLE001 - 关闭失败不影响主流程
                pass
        self._handles = []
        self._owner = {}
        self._closed = True

    def __del__(self) -> None:  # pragma: no cover - GC 兜底
        try:
            if not self._closed:
                self.close()
        except Exception:  # noqa: BLE001
            pass
