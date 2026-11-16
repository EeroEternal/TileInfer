"""Upstream-facing model implementations (L5 payload of the ``model_table`` seam).

These modules import vLLM and live beside the bridge because the bridge is the only layer
allowed to depend on the engine.  A vendor-served model never reaches this package: its
architecture already exists upstream and the declaration leaves ``model_impl`` empty.

* :mod:`.muse_glimmer` — the text decoder vLLM 0.23 does not ship
  (``MuseGlimmerForConditionalGeneration``) plus the ``AutoConfig`` shim.
* :mod:`.atem` — the checkpoint's private channel reasoning parser
  (``--reasoning-parser atem``).

The model stays out of ``tileinfer/models/`` on purpose: that layer is engine-free by rule so
it can be constructed and tested on a CPU box without vLLM.  This package is the opposite
side of that boundary — it is imported only after a plan has asked for it.
"""

from __future__ import annotations

__all__: list = []
