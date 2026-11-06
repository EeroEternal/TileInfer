"""Serving counters: one thread-safe bag of integers per process.

The engines this backend serves are concurrent (an API-server thread, the EngineCore's step loop,
a background compiler), and the numbers are read by an out-of-process probe that decides whether a
strategy is actually in service.  Two rules follow, and both are cheap to keep here rather than at
every call site:

* every mutation goes through a lock (the probe reads a consistent snapshot, never a half-updated
  dict);
* ``snapshot`` returns a plain ``dict`` copy, so the probe can serialise it without holding the
  lock and without racing the step loop.
"""

from __future__ import annotations

import threading
from typing import Dict, Mapping, Optional

__all__ = ["Counters"]


def _is_tracing() -> bool:
    """Whether ``torch.compile`` is currently tracing (no lock, no allocations).

    Deliberately conservative: any failure to import/ask dynamo answers ``False``, so the counter
    keeps working in every ordinary run.
    """
    try:
        import torch._dynamo as _dynamo  # noqa: PLC0415

        return bool(_dynamo.is_compiling())
    except Exception:  # pragma: no cover - torch always ships dynamo
        return False


class Counters:
    """A small, thread-safe counter set with named keys."""

    __slots__ = ("_values", "_lock")

    def __init__(self, initial: Optional[Mapping[str, int]] = None) -> None:
        self._values: Dict[str, int] = {str(k): int(v) for k, v in (initial or {}).items()}
        self._lock = threading.Lock()

    def bump(self, name: str, n: int = 1) -> int:
        """Add ``n`` to ``name`` and return the new value.

        No-ops while ``torch.compile`` is tracing.  The lock is a ``with`` statement, which dynamo
        cannot trace (``Unsupported context manager``), and it is the *first* thing a traced
        attention ``forward`` would hit — which is why the impl used to be marked
        ``torch.compiler.disable`` and that in turn blocked vLLM's fullgraph graph-capture path
        (§66).  A counter is diagnostics, not semantics, so dropping it inside a trace costs nothing.
        """
        if _is_tracing():
            return 0
        with self._lock:
            value = self._values.get(name, 0) + int(n)
            self._values[name] = value
            return value

    def get(self, name: str, default: int = 0) -> int:
        with self._lock:
            return self._values.get(name, default)

    def set(self, name: str, value: int) -> None:
        with self._lock:
            self._values[name] = int(value)

    def update(self, mapping: Mapping[str, int]) -> None:
        with self._lock:
            for key, value in mapping.items():
                self._values[str(key)] = int(value)

    def snapshot(self) -> Dict[str, int]:
        """A consistent copy; safe to serialise outside the lock."""
        with self._lock:
            return dict(self._values)

    def reset(self) -> None:
        with self._lock:
            self._values.clear()

    def __contains__(self, name: object) -> bool:
        with self._lock:
            return name in self._values

    def __getitem__(self, name: str) -> int:
        return self.get(name)

    def __len__(self) -> int:
        with self._lock:
            return len(self._values)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Counters({self.snapshot()!r})"
