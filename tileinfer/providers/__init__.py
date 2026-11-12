"""L4 - implementation providers.

Empty in M1: Qwen3.6 is served end-to-end by the vendor providers, and the decision layer
records that choice explicitly.  A provider lands here only with the five declarations from
``docs/extending.md`` §3 (coverage, graphability, parallelism, measured ratio + evidence,
delete condition); the v2 operator families under ``tileinfer/vllm/ops`` are candidates, not
providers, until they gain a vendor-compatible form.
"""

from __future__ import annotations

__all__: list = []
