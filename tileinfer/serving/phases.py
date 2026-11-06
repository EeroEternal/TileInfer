"""Opt-in phase timing for the serving loop (M0/W0b of ``docs/plan-performance.md``).

Two clocks, because they answer different questions and on Ascend they are **not** the same:

* **host** (``time.perf_counter``): how long the call took to *enqueue*.  NPU kernels are
  asynchronous, so a 0.5 ms host time can be a 20 ms device kernel - the first probe table was
  host-only and could not tell which.
* **device** (``torch.npu.Event(enable_timing=True)``): how long the kernels actually ran, read with
  ``event.elapsed_time`` after one synchronise at report time.

Enable with ``TILEINFER_PHASE_TIMING``:

| value | host | device |
|---|---|---|
| unset / ``0`` | off | off |
| ``1`` / ``host`` | on | off |
| ``device`` / ``both`` | on | on |

Device events cost an allocation + two ``record`` calls per sample, so the sample list is capped
(``_DEVICE_MAX``): past the cap only the host clock keeps counting.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, List, Tuple

__all__ = [
    "enabled",
    "device_enabled",
    "phase",
    "timed",
    "record",
    "count_step",
    "report",
    "reset",
    "format_table",
]

_MODE = os.environ.get("TILEINFER_PHASE_TIMING", "").strip().lower()
_HOST_ON = _MODE in {"1", "true", "yes", "on", "host", "device", "both"}
_DEVICE_ON = _MODE in {"device", "both"}
_DEVICE_MAX = 40000

_LOCK = threading.Lock()
_TOTALS: Dict[str, float] = {}
_COUNTS: Dict[str, int] = {}
_DEVICE: List[Tuple[str, Any, Any]] = []
_DEVICE_DROPPED = 0
_STEPS: Dict[str, int] = {"decode": 0, "prefill": 0}

#: Warm-up window.  The cumulative table includes the first (cold) requests - kernel compilation,
#: plan building, allocator warm-up - which inflates every per-call mean.  Setting
#: ``TILEINFER_PHASE_RESET_AFTER_S=N`` makes the probe discard everything before ``N`` seconds after
#: the first sample, so the reported means describe steady state.  (Recording a cold average and
#: calling it a phase profile is exactly the mistake the plan's §11 corrects.)
_WARMUP_S = float(os.environ.get("TILEINFER_PHASE_RESET_AFTER_S", "0") or 0)
_FIRST_SAMPLE_AT: List[float] = []
_WINDOW_OPEN = [False]  # "the warm-up has been discarded already"


def _maybe_open_window() -> None:
    """Drop everything before the warm-up window closes (called from every record path)."""
    if _WARMUP_S <= 0 or _WINDOW_OPEN[0]:
        return
    now = time.monotonic()
    if not _FIRST_SAMPLE_AT:
        _FIRST_SAMPLE_AT.append(now)
        return
    if now - _FIRST_SAMPLE_AT[0] < _WARMUP_S:
        return
    _reset_locked()
    _WINDOW_OPEN[0] = True


def enabled() -> bool:
    """Whether the host clock is on."""
    return _HOST_ON


def device_enabled() -> bool:
    """Whether device events are on."""
    return _DEVICE_ON


def _new_event() -> Any:
    """An NPU timing event, or ``None`` when this build has no ``torch.npu`` (CPU-only box).

    A failure here also switches device mode off for the rest of the process, so a CPU run with
    ``TILEINFER_PHASE_TIMING=device`` degrades to host timing instead of raising.
    """
    global _DEVICE_ON
    try:
        import torch  # noqa: PLC0415 - torch is present whenever this runs

        return torch.npu.Event(enable_timing=True)
    except Exception:  # noqa: BLE001 - no torch.npu / no device
        _DEVICE_ON = False
        return None


def record(phase_name: str, seconds: float) -> None:
    """Add one *host* duration to ``phase_name``."""
    if not _HOST_ON:
        return
    _maybe_open_window()
    with _LOCK:
        _TOTALS[phase_name] = _TOTALS.get(phase_name, 0.0) + seconds
        _COUNTS[phase_name] = _COUNTS.get(phase_name, 0) + 1


def count_step(kind: str) -> None:
    """Count one scheduling step of ``kind`` ("decode" or "prefill")."""
    if not _HOST_ON:
        return
    _maybe_open_window()
    with _LOCK:
        _STEPS[kind] = _STEPS.get(kind, 0) + 1


def steps_counted(kind: str) -> int:
    """Constant-time read of a step counter.

    Never use :func:`report` for this: it synchronises the device and walks every queued event,
    so calling it per step turns the probe into an O(n^2) hot-path cost (observed: 64-token
    requests degrading from 5.8 s to 11.4 s across six runs).
    """
    with _LOCK:
        return _STEPS.get(kind, 0)


class phase:
    """Host clock only: ``with phase("moe"):`` (no-op when disabled)."""

    __slots__ = ("_name", "_start")

    def __init__(self, name: str) -> None:
        self._name = name
        self._start = 0.0

    def __enter__(self) -> "phase":
        if _HOST_ON:
            self._start = time.perf_counter()
        return self

    def __exit__(self, *exc_info: object) -> None:
        if _HOST_ON:
            record(self._name, time.perf_counter() - self._start)


class timed:
    """Host clock plus, when enabled, device events for the same block.

    The device pair is appended to a list and resolved once at :func:`report` time: calling
    ``elapsed_time`` inside the block would serialise every kernel and destroy the very pipelining
    the measurement is meant to observe.
    """

    __slots__ = ("_name", "_host_start", "_start_event", "_end_event")

    def __init__(self, name: str) -> None:
        self._name = name
        self._host_start = 0.0
        self._start_event = None
        self._end_event = None

    def __enter__(self) -> "timed":
        if _HOST_ON:
            _maybe_open_window()
            self._host_start = time.perf_counter()
        if _DEVICE_ON:
            self._start_event = _new_event()
            if self._start_event is not None:
                self._start_event.record()
        return self

    def __exit__(self, *exc_info: object) -> None:
        global _DEVICE_DROPPED
        if _HOST_ON:
            record(self._name, time.perf_counter() - self._host_start)
        if _DEVICE_ON and self._start_event is not None:
            self._end_event = _new_event()
            if self._end_event is None:
                return
            self._end_event.record()
            with _LOCK:
                if len(_DEVICE) < _DEVICE_MAX:
                    _DEVICE.append((self._name, self._start_event, self._end_event))
                else:
                    _DEVICE_DROPPED += 1


def reset() -> None:
    """Forget every sample (tests, and a fresh measurement window)."""
    _reset_locked()


def _reset_locked() -> None:
    global _DEVICE_DROPPED
    with _LOCK:
        _TOTALS.clear()
        _COUNTS.clear()
        _DEVICE.clear()
        _DEVICE_DROPPED = 0
        for key in _STEPS:
            _STEPS[key] = 0


def _device_totals() -> Tuple[Dict[str, float], Dict[str, int]]:
    """Resolve the queued event pairs into per-phase milliseconds (one synchronise)."""
    if not _DEVICE_ON:
        return {}, {}
    with _LOCK:
        samples = list(_DEVICE)
    if not samples:
        return {}, {}
    try:
        import torch  # noqa: PLC0415

        torch.npu.synchronize()
    except Exception:  # pragma: no cover - no device (CPU-only test box)
        return {}, {}
    totals: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for name, start, end in samples:
        try:
            totals[name] = totals.get(name, 0.0) + start.elapsed_time(end)
        except Exception:  # pragma: no cover - an event that never completed
            continue
        counts[name] = counts.get(name, 0) + 1
    return totals, counts


def report() -> Dict[str, object]:
    """The cumulative snapshot: host totals/means plus, when enabled, device totals/means."""
    with _LOCK:
        totals = dict(_TOTALS)
        counts = dict(_COUNTS)
        steps = dict(_STEPS)
        dropped = _DEVICE_DROPPED
    dev_totals, dev_counts = _device_totals()
    # A caller may record the whole step under "step"; without it, the sum of the phases is the
    # best available denominator (the shares are then "of the instrumented work", not of the step).
    step_total = totals.get("step", 0.0) or sum(v for k, v in totals.items() if k != "step")
    step_ms = (step_total / steps["decode"] * 1e3) if steps["decode"] else 0.0
    rows: List[Dict[str, object]] = []
    names = sorted(set(totals) | set(dev_totals), key=lambda n: dev_totals.get(n, totals.get(n, 0.0)),
                   reverse=True)
    for name in names:
        if name == "step":
            continue
        total = totals.get(name, 0.0)
        calls = counts.get(name, 0)
        dev_total_ms = dev_totals.get(name, 0.0)
        rows.append(
            {
                "phase": name,
                "calls": calls,
                "total_ms": round(total * 1e3, 2),
                "mean_ms": round((total / calls * 1e3) if calls else 0.0, 3),
                "per_step_ms": round((total / steps["decode"]) * 1e3, 3) if steps["decode"] else 0.0,
                "share_of_step": round(total / step_total, 4) if step_total else 0.0,
                "device_total_ms": round(dev_total_ms, 2),
                "device_mean_ms": round((dev_total_ms / dev_counts.get(name, 1)), 3)
                if dev_counts.get(name)
                else 0.0,
                "device_per_step_ms": round(dev_total_ms / steps["decode"], 3)
                if steps["decode"]
                else 0.0,
            }
        )
    return {
        "steps": steps,
        "step_mean_ms": round(step_ms, 2),
        "device_on": _DEVICE_ON,
        "device_samples": len(dev_counts) and sum(dev_counts.values()),
        "device_dropped": dropped,
        "phases": rows,
    }


def format_table(snapshot: Dict[str, object] | None = None) -> str:
    """A fixed-width table: host ms and, when enabled, device ms side by side.

    Read the columns with care — this table has been misread once already (§24.2):

    * ``calls`` — how many times the phase ran;
    * ``host/step`` — host milliseconds **per counted step**;
    * ``dev/call`` — device milliseconds per call of the phase (the only unambiguous device column);
    * ``dev/attncall`` — the phase's total device time divided by ``steps["decode"]``.  That
      denominator counts **attention-layer calls**, not engine steps (``count_step`` is bumped once
      per layer in ``impl._timed_forward``), so it is *not* "device time per generated token".
      A phase's per-token cost is ``dev/call × calls_per_token``; compute it from ``calls``.
    """
    data = report() if snapshot is None else snapshot
    steps = data["steps"]  # type: ignore[index]
    device_on = data["device_on"]  # type: ignore[index]
    header = f"{'phase':<22}{'calls':>9}{'host/step':>12}{'dev/attncall':>11}{'dev/call':>10}"
    lines = [
        "TileInfer phase timing: %d attn-layer calls (%s), %d prefill, host step=%.2f ms%s"
        % (
            steps["decode"],
            "decode",
            steps["prefill"],
            data["step_mean_ms"],  # type: ignore[index]
            "" if device_on else "  [device events off: TILEINFER_PHASE_TIMING=device]",
        ),
        header,
        "-" * len(header),
    ]
    for row in data["phases"]:  # type: ignore[index]
        lines.append(
            f"{row['phase']:<22}{row['calls']:>9}{row['per_step_ms']:>12.3f}"
            f"{row['device_per_step_ms']:>11.3f}{row['device_mean_ms']:>10.3f}"
        )
    if device_on and data["device_dropped"]:  # type: ignore[index]
        lines.append(f"(device samples dropped past the cap: {data['device_dropped']})")  # type: ignore[index]
    return "\n".join(lines)
