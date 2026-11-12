"""Observation: counters and timing that the acceptance report consumes.

The graph counters live in :mod:`tileinfer.bridge.metrics` because they are installed by the
seam that knows about vLLM's dispatcher.  This package is the engine-free home for future
observability that does not need the bridge (for example a CSV exporter).
"""

from __future__ import annotations

__all__: list = []
