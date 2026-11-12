"""Which implementation did each engine op instance actually get?

vLLM picks an out-of-tree op by class name and, when the platform registered none, silently
calls ``forward_native`` - plain torch.  Nothing in the log says so, so "the model runs" and
"the model runs the vendor's kernels" are different statements and only the first one is
visible.  The graph counters cannot answer this either: they count dispatcher fallbacks (a
batch that could not use a captured graph), not ops that run as torch inside a captured graph.

The bridge fills this registry once per op instance, at construction time, and never on the
request path.  Per class it records the class, the module it came from, the forward that will
actually run, and how many instances the model built.  On a decode path where every op
instance runs once per token, the instance count *is* the per-step call count - which is the
difference between a cold op falling back (fine, do not touch it) and a hot one doing it
(a finding worth chasing).

The audit is off unless ``TILEINFER_AUDIT`` names an output path; it is a diagnostic, not a
seam the plan asks for.
"""

from __future__ import annotations

import atexit
import json
import os
import signal
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class OpRecord:
    """One op class as the model actually built it."""

    name: str
    base: str  # "custom_op" (per-op dispatch) | "pluggable" (whole-layer replacement)
    module: str
    forward: str
    forward_module: str
    count: int = 0
    calls: int = 0

    @property
    def impl(self) -> str:
        """Who executes it: the vendor, or a torch/vLLM default that never got replaced."""
        if self.forward_module.startswith("vllm_ascend"):
            return "vendor"
        return "torch" if self.base == "custom_op" else "vllm"

    @property
    def suspicious(self) -> bool:
        """A vendor class whose forward entry point is still vLLM's.

        A lead, not a verdict.  Two legitimate patterns produce this and both must be ruled out
        before calling it a fallback: (a) vLLM's ``forward_native`` is sometimes a thin shim
        that dispatches into a method the vendor *did* override - ``UnquantizedFusedMoEMethod``
        calls ``self.apply()``, which Ascend overrides; (b) the vendor may patch the method on
        the class rather than replace the class, which is exactly how it handles Qwen's GDN
        layer.  Check ``calls`` first: a flag on a class that never runs is noise.
        """
        return self.module.startswith("vllm_ascend") and self.impl != "vendor"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "base": self.base,
            "module": self.module,
            "forward": self.forward,
            "forward_module": self.forward_module,
            "impl": self.impl,
            "count": self.count,
            "calls": self.calls,
            "suspicious": self.suspicious,
        }


_RECORDS: Dict[str, OpRecord] = {}
_LOCK = threading.Lock()
_PATH: Optional[str] = None
_HOOKED = False
_STARTED = time.time()


def record(name: str, *, base: str, module: str, forward: str, forward_module: str) -> None:
    """Count one op instance.  Called at construction time only."""
    with _LOCK:
        existing = _RECORDS.get(name)
        if existing is None:
            _RECORDS[name] = OpRecord(name, base, module, forward, forward_module, 1)
            return
        existing.count += 1
        # Keep the first observation; if a later instance resolved differently, that is a
        # finding in itself - prefer the non-vendor one so it cannot hide.
        if existing.impl == "vendor" and existing.forward_module != forward_module:
            existing.forward = forward
            existing.forward_module = forward_module


def record_call(name: str) -> None:
    """Count one forward call.  Only installed while the audit is on, so it costs nothing by default."""
    rec = _RECORDS.get(name)
    if rec is not None:
        rec.calls += 1


def snapshot() -> Dict[str, Any]:
    """The audit as JSON-serialisable data, stable enough to diff between runs."""
    with _LOCK:
        ops: List[OpRecord] = list(_RECORDS.values())
    rows = sorted((op.as_dict() for op in ops), key=lambda r: (-r["count"], r["name"]))
    vendor = sum(r["count"] for r in rows if r["impl"] == "vendor")
    torch = sum(r["count"] for r in rows if r["impl"] == "torch")
    return {
        "pid": os.getpid(),
        "captured_at": time.time(),
        "uptime_s": round(time.time() - _STARTED, 3),
        "totals": {
            "classes": len(rows),
            "instances": sum(r["count"] for r in rows),
            "vendor": vendor,
            "torch": torch,
            "calls": sum(r["calls"] for r in rows),
            "suspicious": sum(1 for r in rows if r["suspicious"]),
        },
        "ops": rows,
    }


def dump(path: Optional[str] = None, *, force: bool = False) -> Optional[str]:
    """Write the audit.  A process that built no ops (the API server) writes nothing."""
    target = path or _PATH
    if not target:
        return None
    snap = snapshot()
    if not snap["ops"] and not force:
        return None
    tmp = f"{target}.{os.getpid()}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(snap, fh, ensure_ascii=False, indent=1)
            fh.write("\n")
        os.replace(tmp, target)
        return target
    except Exception:
        try:
            os.unlink(tmp)
        except Exception:
            pass
        return None


def reset() -> None:
    """Forget everything recorded so far.  Tests use it; the bridge never does."""
    with _LOCK:
        _RECORDS.clear()


def configure(path: str) -> None:
    """Enable the audit: remember the path, dump on SIGUSR1 and at exit.  Never raises."""
    global _PATH, _HOOKED
    _PATH = path
    if _HOOKED:
        return
    _HOOKED = True

    def _on_signal(signum, frame):  # noqa: ARG001 - signal handler signature
        dump()

    try:
        signal.signal(signal.SIGUSR1, _on_signal)
    except Exception:
        pass
    atexit.register(dump)
