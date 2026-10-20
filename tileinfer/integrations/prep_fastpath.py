"""H70 decode prep 域 host 固定税削减(E282/E281;opt-in,默认关)。

针对 c1/c8 decode 稳态下 prep 块的三处发射税(vllm-ascend PR#19033 同构手法,
本地证据 E282:bt_commit 0.087ms/步 + bt_slotmap 0.084ms/步 + snap_copy 0.049ms/步):

  1) bt_commit 内容不变跳过:MultiGroupBlockTable 4 个 KV 组每步无条件
     clone+pin+H2D;c1 稳态下块表内容每 ~128 步才变(page128)→ 与上次提交
     快照逐元素比较,相同则跳过全链。
  2) bt_slotmap CPU fast path:非 CP、非 MTP、duplicate_size==1 时,slot 映射
     退化为纯 numpy 公式(triton kernel 同构),n 个槽写 CpuGpuBuffer 后单次
     H2D;pad 区(kernel 契约:[n, max_num_batched_tokens) 填 PAD_ID)按 n 变化
     懒刷新,稳态零发射。消 triton python launch 税(~84μs/步)。
  3) snap_copy 持久 pinned 镜像:copy_snapshot_to_gpu 每次 clone+pin_memory()
     新分配 pinned 内存 → 每 buffer 复用一块持久 pinned 镜像 + event 门
     (覆盖前确认上一步 H2D 已完成)。

语义安全:
  - 全部手法为「同输入同输出」的发射数削减;块表/slot 消费方均为图内静态
    地址缓冲,跳过仅在「cpu 内容自上次提交后逐元素不变」时发生(gpu 必然
    已是同内容);clear() 同时清 cpu,比较天然失效触发重拷。
  - slot CPU 公式与 _compute_slot_mapping_kernel 在
    pcp==1/dcp==1/TOTAL_CP_WORLD_SIZE==1 分支逐项同构(E282/oracle 测试锁定)。
  - 镜像覆盖竞态由 npu event 门消除;fallback 全保留(任一前置不满足即回退
    原实现)。

门控:``TILEINFER_PREP_FASTPATH=1`` 显式开启(默认关=零行为变化);翻转
env + 重启即整体回退。安装点=tileinfer.integrations.vllm_ascend.install()。
"""

from __future__ import annotations

import logging
import os
import threading
import weakref
from typing import Any, Optional

import numpy as np
import torch

logger = logging.getLogger("tileinfer.prep_fastpath")

_INSTALLED = False
#: 引擎线程上的当前 runner(compute_slot_mapping 无 runner 引用,经此获取
#: num_computed_tokens_cpu/query_pos.np/num_scheduled_tokens.np;prep 在引擎
#: 线程串行执行,threadlocal 读写即进出 _prepare_inputs)。
_tls = threading.local()


_KNOWN = ("commit", "snap", "slotmap")


def fastpath_enabled() -> bool:
    return bool(_fastpath_subset())


def _fastpath_subset() -> set[str]:
    """机制子集门控:``1``=全开;逗号列表(commit,snap,slotmap)仅开所列;
    其余任何值(含 "0"/未设置/未知名)=关。"""
    spec = (os.environ.get("TILEINFER_PREP_FASTPATH") or "").strip()
    if spec == "1":
        return set(_KNOWN)
    parts = {s.strip().lower() for s in spec.split(",") if s.strip()}
    return parts if parts and parts <= set(_KNOWN) else set()


def _patch_bt_commit(BlockTable: type) -> str:
    """手法 1:commit_block_table 内容不变跳过(含 clone+pin+H2D 全链)。"""
    orig = BlockTable.commit_block_table
    snapshots: "weakref.WeakKeyDictionary[Any, tuple[int, np.ndarray]]" = (
        weakref.WeakKeyDictionary()
    )

    def commit_block_table(self, num_reqs: int) -> None:  # type: ignore[no-untyped-def]
        np_tab = self.block_table.np
        view = np_tab[:num_reqs]
        snap = snapshots.get(self)
        if (
            snap is not None
            and snap[0] == num_reqs
            and snap[1].shape == view.shape
            and np.array_equal(snap[1], view)
        ):
            return
        orig(self, num_reqs)
        snapshots[self] = (num_reqs, view.copy())

    BlockTable.commit_block_table = commit_block_table  # type: ignore[method-assign]
    return "bt_commit_skip"


def _patch_snap_copy(mrv: Any) -> str:
    """手法 3:copy_snapshot_to_gpu 持久 pinned 镜像 + event 门。"""
    orig = mrv.copy_snapshot_to_gpu
    mirrors: "weakref.WeakKeyDictionary[Any, tuple[Any, Any]]" = weakref.WeakKeyDictionary()

    def copy_snapshot_to_gpu(buffer: Any) -> Any:  # type: ignore[no-untyped-def]
        entry = mirrors.get(buffer)
        if entry is None or entry[0].shape != buffer.cpu.shape:
            mirror = torch.empty_like(buffer.cpu).pin_memory()
            event = torch.npu.Event()
            mirrors[buffer] = (mirror, event)
        else:
            mirror, event = entry
        if not event.query():
            event.synchronize()
        mirror.copy_(buffer.cpu)
        out = buffer.gpu.copy_(mirror, non_blocking=True)
        event.record()
        return out

    mrv.copy_snapshot_to_gpu = copy_snapshot_to_gpu
    return "snap_copy_mirror"


def _patch_bt_slotmap(BlockTable: type, mrv: Any) -> Optional[str]:
    """手法 2:slot_mapping CPU fast path(triton 同构公式 + pad 懒刷新)。

    线程契约:仅当 _prepare_inputs 包装器设置了当前 runner 时生效,否则回退。
    门控:非 CP(pcp/dcp==1)、块表无 MTP 行复制(shape 校验)、kernel_size 一致;
    公式与 _compute_slot_mapping_kernel 的 CP==1 分支逐项同构(块表内容在
    写入侧已逻辑映射,读侧两路径等价,见 E282/oracle 测试)。
    """
    orig = BlockTable.compute_slot_mapping

    def compute_slot_mapping(  # type: ignore[no-untyped-def]
        self,
        num_reqs: int,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
        *args: Any,
        positions_cpu: Any = None,
        query_start_loc_cpu: Any = None,
        **kwargs: Any,
    ):
        # ---- H49 优先路径:调用方已给出 CPU 侧数组,无需 threadlocal、无需重算 ----
        #
        # H49 的原补丁(#19033 b688ed1cb1 最小子集)在 model_runner 调用点直接传入
        # ``positions_np``(``_prepare_inputs`` 里第 913 行已算好)与
        # ``query_start_loc.np``。这条路比下面"从 runner 重算"少做三件事:
        #   ① 不依赖 _tls.runner(少一次 threadlocal 读写与包装层)
        #   ② 不重算 num_computed_tokens + query_pos(少一次 numpy 加法与索引)
        #   ③ 不需要 pad 懒刷新(调用方给的是精确切片)
        # 判据门控与慢路径相同。
        if (
            positions_cpu is not None
            and query_start_loc_cpu is not None
            and self.pcp_world_size == 1
            and self.dcp_world_size == 1
            and self.block_table.np.shape[0] == self.max_num_reqs
            and (getattr(self, "kernel_sizes", None) is None or self.kernel_sizes[0] == self.block_size)
        ):
            try:
                _slotmap_from_cpu_args(self, num_reqs, positions_cpu, query_start_loc_cpu)
                return None
            except Exception as exc:  # noqa: BLE001
                logger.warning("prep_fastpath slotmap direct-cpu fallback: %s", exc)
                return orig(self, num_reqs, query_start_loc, positions, *args, **kwargs)

        runner = getattr(_tls, "runner", None)
        # ---- H49 优先路径 B:复用 runner 在 _prepare_inputs 里**已经算好**的数组 ----
        #
        # model_runner_v1.py 第 913 行已经把 positions 写进 ``self._positions_np_buf``:
        #     positions_np = self._positions_np_buf[:total]
        #     np.add(num_computed_tokens_cpu[req_indices], query_pos.np[:n], out=positions_np)
        # 而 ``cu_num_tokens`` 是同一段的 query_start_loc 前缀和。
        # 因此这里**不需要任何重算**,直接读这两块 CPU 缓冲即可 —— 与 H49 原补丁
        # (#19033 b688ed1cb1 最小子集,调用方直传 positions_np/query_start_loc.np)
        # 在数据来源上等价,同时保持"不改 vendor 调用点"的纪律。
        if (
            runner is not None
            and self.pcp_world_size == 1
            and self.dcp_world_size == 1
            and self.block_table.np.shape[0] == self.max_num_reqs
            and (getattr(self, "kernel_sizes", None) is None or self.kernel_sizes[0] == self.block_size)
        ):
            pos_buf = getattr(runner, "_positions_np_buf", None)
            qsl_buf = getattr(getattr(runner, "query_start_loc", None), "np", None)
            if pos_buf is not None and qsl_buf is not None:
                try:
                    n = int(qsl_buf[num_reqs])
                    _slotmap_from_cpu_args(self, num_reqs, pos_buf[:n], qsl_buf[: num_reqs + 1])
                    return None
                except Exception as exc:  # noqa: BLE001
                    logger.warning("prep_fastpath slotmap runner-buffer fallback: %s", exc)
                    return orig(self, num_reqs, query_start_loc, positions, *args, **kwargs)

        if (
            runner is None
            or self.pcp_world_size != 1
            or self.dcp_world_size != 1
            or self.block_table.np.shape[0] != self.max_num_reqs
            or (getattr(self, "kernel_sizes", None) is not None and self.kernel_sizes[0] != self.block_size)
        ):
            return orig(self, num_reqs, query_start_loc, positions, *args, **kwargs)
        try:
            return _slotmap_cpu_fast(self, runner, num_reqs)
        except Exception as exc:  # noqa: BLE001 - fast path 失败必须回退原路径
            logger.warning("prep_fastpath slotmap fast path fallback: %s", exc)
            return orig(self, num_reqs, query_start_loc, positions, *args, **kwargs)

    def _slotmap_from_cpu_args(self: Any, num_reqs: int, positions_cpu: Any, qsl_cpu: Any) -> None:
        """H49 直接路径:用调用方给的 CPU 数组算 slot(与 triton 核同整数算术)。

        公式逐项同构 ``_compute_slot_mapping_kernel`` 的 CP==1 分支:
            slot = block_table[req, pos // bs] * bs + pos % bs
        ``pos`` 是全局 KV 位置(调用方的 positions 已含 computed prefix)。
        """
        qsl = np.asarray(qsl_cpu[: num_reqs + 1], dtype=np.int64)
        counts = qsl[1:] - qsl[:-1]
        n = int(counts.sum())
        if n == 0 or n > self.max_num_batched_tokens:
            raise ValueError(f"unsupported token count {n}")
        req_of_token = np.repeat(np.arange(num_reqs, dtype=np.int64), counts)
        pos = np.asarray(positions_cpu[:n], dtype=np.int64)
        block_idx = pos // self.block_size
        blocks = self.block_table.np[req_of_token, block_idx].astype(np.int64)
        np.add(
            blocks * self.block_size,
            pos - block_idx * self.block_size,
            out=self.slot_mapping.np[:n],
        )
        self.slot_mapping.copy_to_gpu(n)
        # 与 kernel 契约对齐:[n, max_num_batched_tokens) 填 PAD_ID(按 n 懒刷新)
        from vllm.v1.attention.backends.utils import PAD_SLOT_ID

        last = getattr(self, "_ti_pad_n_direct", -1)
        if last != n:
            self.slot_mapping.gpu[n : self.max_num_batched_tokens].fill_(PAD_SLOT_ID)
            self._ti_pad_n_direct = n

    def _slotmap_cpu_fast(self: Any, runner: Any, num_reqs: int) -> None:
        from vllm.v1.attention.backends.utils import PAD_SLOT_ID

        sched = runner.num_scheduled_tokens.np[:num_reqs].astype(np.int64, copy=False)
        n = int(sched.sum())
        if n == 0 or n > self.max_num_batched_tokens:
            raise ValueError(f"unsupported token count {n}")
        req_of_token = np.repeat(np.arange(num_reqs, dtype=np.int64), sched)
        pos = (
            runner.input_batch.num_computed_tokens_cpu[req_of_token].astype(np.int64)
            + runner.query_pos.np[:n].astype(np.int64)
        )
        block_idx = pos // self.block_size
        blocks = self.block_table.np[req_of_token, block_idx].astype(np.int64)
        np.add(
            blocks * self.block_size,
            pos - block_idx * self.block_size,
            out=self.slot_mapping.np[:n],
        )
        self.slot_mapping.copy_to_gpu(n)
        # kernel 契约:[n, max_num_batched_tokens) 填 PAD_ID;按 n 变化懒刷新
        last = getattr(self, "_ti_pad_n", -1)
        if last != n:
            self.slot_mapping.gpu[n : self.max_num_batched_tokens].fill_(PAD_SLOT_ID)
            self._ti_pad_n = n

    # _prepare_inputs 薄包装:进入/退出设置 threadlocal runner(供 slotmap 取数)
    orig_prep = mrv.NPUModelRunner._prepare_inputs

    def _prepare_inputs(self, *args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
        prev = getattr(_tls, "runner", None)
        _tls.runner = self
        try:
            return orig_prep(self, *args, **kwargs)
        finally:
            _tls.runner = prev

    mrv.NPUModelRunner._prepare_inputs = _prepare_inputs  # type: ignore[method-assign]
    BlockTable.compute_slot_mapping = compute_slot_mapping  # type: ignore[method-assign]
    return "bt_slotmap_cpu"


def install_prep_fastpath() -> bool:
    """安装三手法;任一失败仅降级该手法,不影响其余与在役路径。"""
    global _INSTALLED
    if _INSTALLED:
        return True
    if not fastpath_enabled():
        return False
    installed: list[str] = []
    subset = _fastpath_subset()
    import vllm_ascend.worker.model_runner_v1 as mrv
    from vllm_ascend.worker import block_table as asc_bt

    if "commit" in subset:
        try:
            installed.append(_patch_bt_commit(asc_bt.BlockTable))
        except Exception as exc:  # noqa: BLE001
            logger.warning("prep_fastpath bt_commit skip not installed: %s", exc)
    if "snap" in subset:
        try:
            installed.append(_patch_snap_copy(mrv))
        except Exception as exc:  # noqa: BLE001
            logger.warning("prep_fastpath snap mirror not installed: %s", exc)
    if "slotmap" in subset:
        try:
            seg = _patch_bt_slotmap(asc_bt.BlockTable, mrv)
            if seg:
                installed.append(seg)
        except Exception as exc:  # noqa: BLE001
            logger.warning("prep_fastpath slotmap fast path not installed: %s", exc)
    _INSTALLED = True
    logger.warning("TileInfer prep fastpath ACTIVE (opt-in): %s", installed)
    return True
