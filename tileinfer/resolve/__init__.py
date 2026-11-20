"""L2 - capability detection and the pure decision (declaration -> plan + gap report).

Nothing here imports an implementation or the engine.  That property is what lets a new
model's cost be computed on a CPU laptop before anyone books a device window.

* :mod:`~tileinfer.resolve.probe` answers "what does this machine have?".
* :mod:`~tileinfer.resolve.decide` is the pure function "manifest x environment x policy".
* the vendor provider catalogue now lives in :mod:`tileinfer.providers.vendor`.
"""

from __future__ import annotations

from .decide import decide
from .probe import Environment, probe

__all__ = ["Environment", "probe", "decide"]
