"""Engine-free LFM3 helpers.

The vLLM-facing model lives in ``tileinfer.bridge.models.lfm3_model``.  This package keeps
only the transformers compatibility shim, which is imported by the bridge's
``register_config`` actor before vLLM reads the checkpoint config.
"""

from __future__ import annotations

from .compat import apply

__all__ = ["apply"]
