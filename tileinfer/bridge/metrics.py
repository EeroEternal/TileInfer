"""Counters for the one thing that must not be inferred: did the graph actually run?

The acceptance criterion is "captures > 0 and steady-state fallbacks == 0".  Neither number
is visible in stock vLLM logs, so the bridge counts them at the only two places that know:
the dispatcher that maps a batch to a graph, and the capture entry point.

Design constraints:

* the hot path only increments integers - no logging, no allocation, no device sync;
* the counters are dumped to JSON at most every ``dump_every`` decode steps (a file write
  every 500 steps is noise), on ``SIGUSR1``, and at process exit, so an A/B run can read
  them without instrumenting a second time;
* a fallback is only counted for a *uniform decode* step whose allowed modes included a
  graph mode - a prefill over 32 tokens is outside the captured range by design, not a
  regression.
"""

from __future__ import annotations

import atexit
import json
import os
import signal
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Counters:
    """Process-local graph counters.  Plain ints; safe to increment from one thread."""

    capture_calls: int = 0
    captured_graphs: int = 0
    capture_keys: Dict[str, int] = field(default_factory=dict)
    cudagraph_mode: str = ""
    capture_done: bool = False
    decode_calls: int = 0
    decode_graph_hits: int = 0
    decode_fallbacks: int = 0
    attention_backend: str = ""
    plan_id: str = ""
    started_at: float = field(default_factory=time.time)
    last_dump: float = 0.0
    errors: int = 0

    def snapshot(self) -> Dict[str, Any]:
        return {
            "pid": os.getpid(),
            "plan_id": self.plan_id,
            "capture_calls": self.capture_calls,
            "captured_graphs": self.captured_graphs,
            "capture_keys": dict(self.capture_keys),
            "cudagraph_mode": self.cudagraph_mode,
            "capture_done": self.capture_done,
            "decode_calls": self.decode_calls,
            "decode_graph_hits": self.decode_graph_hits,
            "decode_fallbacks": self.decode_fallbacks,
            "attention_backend": self.attention_backend,
            "errors": self.errors,
            "uptime_s": round(time.time() - self.started_at, 3),
        }


COUNTERS = Counters()
_LOCK = threading.Lock()
_PATH: Optional[str] = None
_DUMP_EVERY = 500
_HOOKED = False


def configure(path: Optional[str], dump_every: int = 500) -> None:
    global _PATH, _DUMP_EVERY
    _PATH = path
    _DUMP_EVERY = max(0, int(dump_every))
    if path:
        _install_hooks()


def path() -> Optional[str]:
    return _PATH


def dump(force: bool = False) -> Optional[str]:
    """Write a snapshot atomically.  Never raises: counters must not break serving.

    The API-server process installs the bridge too, but never captures a graph.  Without this
    guard its ``atexit`` dump overwrites the EngineCore's counters with zeros - which is
    exactly the "capture=0" false negative the acceptance report must not contain.  An
    explicit ``SIGUSR1`` (``force=True``) always writes, even from an idle process.
    """
    target = _PATH
    if not target:
        return None
    if not force and COUNTERS.capture_calls == 0 and COUNTERS.decode_calls == 0:
        return None
    try:
        snapshot = COUNTERS.snapshot()
        tmp = f"{target}.{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(snapshot, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        os.replace(tmp, target)
        COUNTERS.last_dump = time.time()
        return target
    except Exception:
        COUNTERS.errors += 1
        return None


def maybe_dump() -> None:
    if _DUMP_EVERY and COUNTERS.decode_calls and COUNTERS.decode_calls % _DUMP_EVERY == 0:
        dump()


def reset() -> None:
    """Test helper: clear counters without touching hooks."""
    fresh = Counters(plan_id=COUNTERS.plan_id)
    COUNTERS.__dict__.update(fresh.__dict__)


def _install_hooks() -> None:
    global _HOOKED
    if _HOOKED:
        return
    _HOOKED = True
    atexit.register(dump)

    def _on_usr1(_signum: int, _frame: Any) -> None:  # pragma: no cover - signal path
        dump(force=True)

    try:
        signal.signal(signal.SIGUSR1, _on_usr1)
    except Exception:
        pass


def record_attention_backend(backend: str) -> None:
    COUNTERS.attention_backend = backend


def record_capture(keys: Dict[str, int], mode: str) -> None:
    COUNTERS.capture_keys = dict(keys)
    COUNTERS.cudagraph_mode = mode


def record_capture_done(graphs: int) -> None:
    COUNTERS.capture_calls += 1
    if graphs > 0:
        COUNTERS.captured_graphs = graphs
    COUNTERS.capture_done = True
    dump()


def record_dispatch(uniform_decode: bool, mode: str, graph_allowed: bool) -> None:
    # Only steady state counts: before capture completes, `NONE` is expected and must not be
    # reported as a fallback.
    if not COUNTERS.capture_done or not uniform_decode or not graph_allowed:
        return
    COUNTERS.decode_calls += 1
    if mode == "NONE":
        COUNTERS.decode_fallbacks += 1
    else:
        COUNTERS.decode_graph_hits += 1
    maybe_dump()
