"""L4 - implementation providers.

Qwen3.6 needs no provider: every capability it declares is served by vllm-ascend.  LFM3 is
the first model with a real gap - its short convolution state is not GDN - so
``state/short_conv.py`` exists.  A provider is written only when the vendor cannot use a
capability at all - cannot run it, cannot run it correctly, or cannot run it at a shape we
need (``docs/extending.md`` §3.0).

A provider lands here only with the five declarations from ``docs/extending.md`` §3
(coverage, graphability, parallelism, evidence, delete condition).  ``measured_ratio`` is
optional - it exists only when the arm this provider replaces still runs - but ``evidence``
is not: without it the entry is a candidate, not a default.

The v2 operator families under ``tileinfer/vllm/ops`` are *reference material* for answering
"is the vendor really unable to do this on this shape?", not providers: they are written
against the v2 KV layout, and one becomes a provider only when a concrete model needs a shape
the vendor rejects.
"""

from __future__ import annotations

__all__: list = []
