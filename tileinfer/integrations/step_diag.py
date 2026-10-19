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
"""

from __future__ import annotations

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

_SEG_MAP = {"prepare input": "prep", "forward": "fwd", "post process": "post"}


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
        _fh.write(json.dumps(rec) + "\n")
        _events += 1
        if _events % 64 == 0:
            _fh.flush()


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
    mrv.record_function_or_nullcontext = _diag_record_function
    return ["NPUModelRunner.execute_model", "NPUModelRunner.sample_tokens",
            "NPUModelRunner._update_full_graph_params_if_needed", "record_function_or_nullcontext"]


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
    except Exception as exc:
        logger.warning("step_diag: cannot open %s: %s", path, exc)
        _fh = None
        return False

    ok: list[str] = []
    for patch_fn in (_patch_engine_core, _patch_model_runner, _patch_acl_graph):
        try:
            ok.extend(patch_fn())
        except Exception as exc:
            logger.warning("step_diag: patch failed at %s: %s", patch_fn.__name__, exc)
    _ENABLED = True
    logger.warning("TileInfer step_diag ACTIVE (diagnostic): %s -> %s", ok, path)
    return True
