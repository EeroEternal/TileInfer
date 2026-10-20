"""Per-step host wall-clock breakdown for decode diagnosis (H65).

Diagnostic-only module: activated exclusively by ``TILEINFER_STEP_DIAG=1``
(default off, zero effect). Monkeypatches vLLM / vLLM-Ascend hot-path entry
points to record per-segment wall time as JSONL events; never alters
behaviour, never raises into serving paths.

Segments (all host wall time, serial on the engine-core thread for TP=1
UniProcExecutor):
    engine_step  EngineCoreProc._process_engine_step (step_fn + output
                 queue put + post_step)
    sched        Scheduler.schedule
    mask         Scheduler.get_grammar_bitmask
    exec         NPUModelRunner.execute_model (with ntok)
    prep         "prepare input" record_function block (inside exec)
    fwd          "forward" record_function block (inside exec)
    graph        ACLGraphWrapper.__call__ (inside fwd; stream sync included)
    upd          _update_full_graph_params_if_needed (inside fwd)
    post         "post process" record_function block (inside exec)
    sample_worker NPUModelRunner.sample_tokens
    upd_out      Scheduler.update_from_output
 Gaps between engine_step period and the segment sum are engine-loop residual
 (busy loop, zmq send, frontend backpressure).

 Sub-segment stubs (iter-206, H70 取证; all diagnostic wrappers):
    prep_upd     NPUModelRunner._update_states
    prep_inputs  NPUModelRunner._prepare_inputs
    prep_det     NPUModelRunner._determine_batch_execution_and_padding
    cos_sin      model_runner_v1.update_cos_sin
    snap_copy    model_runner_v1.copy_snapshot_to_gpu (clone+pin per call)
    prep_ids     NPUModelRunner._prepare_input_ids
    prep_prevpos NPUModelRunner._compute_prev_positions
    prep_attnstate NPUModelRunner._build_attn_state
    bt_commit    BlockTable.commit_block_table (H2D)
    bt_slotmap   BlockTable.compute_slot_mapping (triton launch)
 Events are batched (flush every 64 records) to keep diagnostic self-overhead
 low; _flush_pending() drains on atexit.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import threading
import time
from contextlib import nullcontext
from typing import Any, Callable, Optional

logger = logging.getLogger("tileinfer.step_diag")

_ENABLED = False
_fh = None
_lock = threading.Lock()
_step = 0
_t0 = 0.0
_events = 0
_pending: list[str] = []

_SEG_MAP = {"prepare input": "prep", "forward": "fwd", "post process": "post"}


def _flush_pending() -> None:
    """把积压事件一次性写入(批量化降低诊断自开销,iter-206)。"""
    if _fh is None or not _pending:
        return
    _fh.write("".join(_pending))
    _pending.clear()
    _fh.flush()


def _emit(seg: str, ms: float, ntok: Optional[int] = None) -> None:
    global _events
    if _fh is None:
        return
    rec: dict[str, Any] = {
        "t": round((time.perf_counter() - _t0) * 1000.0, 3),
        "step": _step,
        "seg": seg,
        "ms": round(ms, 4),
    }
    if ntok is not None:
        rec["ntok"] = ntok
    with _lock:
        _pending.append(json.dumps(rec) + "\n")
        _events += 1
        if len(_pending) >= 64:
            _flush_pending()


class _TimedSeg:
    __slots__ = ("_name", "_t0")

    def __init__(self, name: str) -> None:
        self._name = name
        self._t0 = 0.0

    def __enter__(self) -> "_TimedSeg":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        _emit(self._name, (time.perf_counter() - self._t0) * 1000.0)
        return False


def _diag_record_function(name: str, *args: Any, **kwargs: Any):
    return _TimedSeg(_SEG_MAP.get(name, name))


def _wrap_timed(obj: Any, attr: str, seg: str, ntok_of: Optional[Callable] = None) -> None:
    orig = getattr(obj, attr)

    def wrapper(*args: Any, **kwargs: Any):
        t0 = time.perf_counter()
        try:
            return orig(*args, **kwargs)
        finally:
            ntok = None
            if ntok_of is not None:
                try:
                    ntok = ntok_of(*args, **kwargs)
                except Exception:
                    ntok = None
            _emit(seg, (time.perf_counter() - t0) * 1000.0, ntok)

    try:
        wrapper.__name__ = getattr(orig, "__name__", attr)
    except Exception:
        pass
    setattr(obj, attr, wrapper)


def _ntok_from_scheduler_output(*args: Any, **kwargs: Any) -> Optional[int]:
    for a in args[1:]:
        v = getattr(a, "total_num_scheduled_tokens", None)
        if v is not None:
            return int(v)
    return None


def _patch_engine_core() -> list[str]:
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.engine.core import EngineCoreProc

    orig_pes = EngineCoreProc._process_engine_step

    def _process_engine_step(self):  # type: ignore[no-untyped-def]
        global _step
        t0 = time.perf_counter()
        _step += 1
        try:
            return orig_pes(self)
        finally:
            _emit("engine_step", (time.perf_counter() - t0) * 1000.0)

    EngineCoreProc._process_engine_step = _process_engine_step  # type: ignore[method-assign]
    _wrap_timed(Scheduler, "schedule", "sched")
    _wrap_timed(Scheduler, "update_from_output", "upd_out")
    _wrap_timed(Scheduler, "get_grammar_bitmask", "mask")
    return ["vllm.EngineCoreProc._process_engine_step", "Scheduler.schedule",
            "Scheduler.update_from_output", "Scheduler.get_grammar_bitmask"]


def _patch_model_runner() -> list[str]:
    import vllm_ascend.worker.model_runner_v1 as mrv

    _wrap_timed(
        mrv.NPUModelRunner,
        "execute_model",
        "exec",
        ntok_of=_ntok_from_scheduler_output,
    )
    _wrap_timed(mrv.NPUModelRunner, "sample_tokens", "sample_worker")
    _wrap_timed(mrv.NPUModelRunner, "_update_full_graph_params_if_needed", "upd")
    # H69-②(iter-205):prep 内部分解——E264 判 prep=1.126ms 为最大 host 暴露单段
    # (<1.5ms 门槛),但其内部构成未分解;"prepare input" 块两大子步独立记账,
    # 残差 = prep − prep_upd − prep_inputs。默认关,零在役影响。
    _wrap_timed(mrv.NPUModelRunner, "_update_states", "prep_upd")
    _wrap_timed(mrv.NPUModelRunner, "_prepare_inputs", "prep_inputs")
    mrv.record_function_or_nullcontext = _diag_record_function
    return ["NPUModelRunner.execute_model", "NPUModelRunner.sample_tokens",
            "NPUModelRunner._update_full_graph_params_if_needed", "record_function_or_nullcontext",
            "NPUModelRunner._update_states", "NPUModelRunner._prepare_inputs"]


def _patch_prep_stubs() -> list[str]:
    """H70(iter-206):prep 块内部子段 stub 二批——残差 0.709ms 与
    _prepare_inputs 0.512ms 的内部构成独立记账。全部诊断包装,透传语义,
    默认关(TILEINFER_STEP_DIAG!=1 时 install_step_diag 不调用本函数)。

    注意:运行时 BlockTable 是 vllm_ascend.worker.block_table 的自有实现
    (hybrid 模型 MultiGroupBlockTable 委派其内部 BlockTable;E280 复盘:
    包装 vllm 原版零事件),故以 vllm_ascend 版为准。"""
    patched: list[str] = []
    import vllm_ascend.worker.model_runner_v1 as mrv

    # 残差候选:_determine_batch_execution_and_padding(dispatch/pad)
    _wrap_timed(mrv.NPUModelRunner, "_determine_batch_execution_and_padding", "prep_det")
    patched.append("NPUModelRunner._determine_batch_execution_and_padding")
    # 残差候选:update_cos_sin(cos/sin 各一遍 index_select→view→repeat→chunk;
    # E280 实测本模型全局 cache 未注册早退 0.6μs,保留记账作证)
    if _wrap_module_attr(mrv, "update_cos_sin", "cos_sin"):
        patched.append("model_runner_v1.update_cos_sin")
    # 残差候选:mamba align 预处理与 attention 元数据构建(prep 块内 L2160/L2231)
    _wrap_timed(mrv.NPUModelRunner, "_build_attention_metadata", "build_attn_meta")
    patched.append("NPUModelRunner._build_attention_metadata")
    try:
        from vllm.v1.worker import mamba_utils

        if _wrap_module_attr(mamba_utils, "preprocess_mamba", "mamba_pre"):
            patched.append("mamba_utils.preprocess_mamba")
        if _wrap_module_attr(mamba_utils, "stage_postprocess_inputs_to_gpu", "mamba_stage"):
            patched.append("mamba_utils.stage_postprocess_inputs_to_gpu")
    except Exception:
        pass
    # _prepare_inputs 内部:cp 快照(clone+pin_memory 每次)/input_ids/prev_pos/attn_state
    if _wrap_module_attr(mrv, "copy_snapshot_to_gpu", "snap_copy"):
        patched.append("model_runner_v1.copy_snapshot_to_gpu")
    for attr, seg, name in (
        ("_prepare_input_ids", "prep_ids", "NPUModelRunner._prepare_input_ids"),
        ("_compute_prev_positions", "prep_prevpos", "NPUModelRunner._compute_prev_positions"),
        ("_build_attn_state", "prep_attnstate", "NPUModelRunner._build_attn_state"),
    ):
        if hasattr(mrv.NPUModelRunner, attr):
            _wrap_timed(mrv.NPUModelRunner, attr, seg)
            patched.append(name)
    # _prepare_inputs 内部:block_table 提交(H2D)与 triton slot_mapping 发射
    # (vllm_ascend 自有 BlockTable;vllm 原版仅在非 ascend 路径兜底)
    try:
        import vllm_ascend.worker.block_table as asc_bt

        for attr, seg, name in (
            ("commit_block_table", "bt_commit", "AscendBlockTable.commit_block_table"),
            ("compute_slot_mapping", "bt_slotmap", "AscendBlockTable.compute_slot_mapping"),
        ):
            if hasattr(asc_bt.BlockTable, attr):
                _wrap_timed(asc_bt.BlockTable, attr, seg)
                patched.append(name)
    except Exception:
        from vllm.v1.worker.block_table import BlockTable

        for attr, seg, name in (
            ("commit_block_table", "bt_commit", "BlockTable.commit_block_table"),
            ("compute_slot_mapping", "bt_slotmap", "BlockTable.compute_slot_mapping"),
        ):
            if hasattr(BlockTable, attr):
                _wrap_timed(BlockTable, attr, seg)
                patched.append(name)
    return patched


def _wrap_module_attr(module: Any, attr: str, seg: str) -> bool:
    fn = getattr(module, attr, None)
    if fn is None or not callable(fn):
        return False

    def wrapper(*args: Any, **kwargs: Any):
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            _emit(seg, (time.perf_counter() - t0) * 1000.0)

    try:
        wrapper.__name__ = getattr(fn, "__name__", attr)
    except Exception:
        pass
    setattr(module, attr, wrapper)
    return True


def _patch_acl_graph() -> list[str]:
    from vllm_ascend.compilation.acl_graph import ACLGraphWrapper

    _wrap_timed(ACLGraphWrapper, "__call__", "graph")
    return ["ACLGraphWrapper.__call__"]


def install_step_diag() -> bool:
    """Install per-step timers when TILEINFER_STEP_DIAG=1; no-op otherwise."""
    global _ENABLED, _fh, _t0
    if _ENABLED:
        return True
    if (os.environ.get("TILEINFER_STEP_DIAG") or "").strip() != "1":
        return False
    path = (os.environ.get("TILEINFER_STEP_DIAG_PATH")
            or "/tmp/opencode/step_diag.jsonl").strip()
    try:
        _fh = open(path, "a", buffering=1)
        _t0 = time.perf_counter()
        _emit("diag_start", 0.0)
        _flush_pending()
        atexit.register(_flush_pending)
    except Exception as exc:
        logger.warning("step_diag: cannot open %s: %s", path, exc)
        _fh = None
        return False

    ok: list[str] = []
    for patch_fn in (_patch_engine_core, _patch_model_runner, _patch_prep_stubs, _patch_acl_graph):
        try:
            ok.extend(patch_fn())
        except Exception as exc:
            logger.warning("step_diag: patch failed at %s: %s", patch_fn.__name__, exc)
    _ENABLED = True
    logger.warning("TileInfer step_diag ACTIVE (diagnostic): %s -> %s", ok, path)
    return True
