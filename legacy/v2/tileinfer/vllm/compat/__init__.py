"""Controlled compatibility layer for upstream vLLM.

``patches``
    The registry of every change we make to vLLM's own behaviour, with the metadata that keeps it
    auditable (why / since / remove-when).
``audit``
    Static scan of the vLLM tree for ``torch.ops._C.<op>`` call sites, so the operator gaps on the
    Qwen3.6 path are a list produced by a script instead of a list produced by memory.

Nothing here imports vLLM at module import time: the functions resolve their targets lazily, so
this package can be imported (and unit-tested) on a machine without vLLM installed.
"""

from __future__ import annotations

__all__: list[str] = []
