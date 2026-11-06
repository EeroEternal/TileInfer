"""Engine-independent serving layer.

This package holds the part of a serving integration that is *not* about any particular engine:
batching into compile-time buckets, the 128-token kernel-page arithmetic, the plan cache and its
pre-warm, the per-step metadata rebuild, and the prefill/append step.  All of it was lifted out of
the former vLLM-Ascend adapter, where it had accumulated a year of hard-won fixes (see
``docs/known-issues.md``), and it keeps the same semantics.

Dependency rule: everything here may import ``torch`` and the TileInfer core, but **never**
``vllm``, the Ascend vendor package or ``torch_npu``.  That is what keeps the scheduling and
metadata logic testable on a CPU laptop, and it is enforced by ``tests/test_layer_deps.py``.
"""

from __future__ import annotations

from .buckets import (
    BATCH_BUCKETS,
    KERNEL_PAGE_SIZE,
    bucket_for,
    buckets_for,
    capacity_view,
    declared_pool_cap,
    expand_page_table,
    fused_halves,
    last_page_len,
    last_page_len_,
    next_pow2,
    padded_rows,
    pages_for_lens,
    pool_sub_view,
    sub_page_width,
    sub_pages,
)
from .counters import Counters
from .planner import PlanCache, buckets_up_to, plan_key, plan_width, prewarm
from .step import StepBuffers, allocate_step_buffers, expected_ragged, refresh_step

__all__ = [
    "BATCH_BUCKETS",
    "KERNEL_PAGE_SIZE",
    "Counters",
    "PlanCache",
    "StepBuffers",
    "allocate_step_buffers",
    "bucket_for",
    "buckets_for",
    "buckets_up_to",
    "capacity_view",
    "declared_pool_cap",
    "expand_page_table",
    "expected_ragged",
    "fused_halves",
    "last_page_len",
    "last_page_len_",
    "next_pow2",
    "padded_rows",
    "pages_for_lens",
    "plan_key",
    "plan_width",
    "pool_sub_view",
    "prewarm",
    "refresh_step",
    "sub_page_width",
    "sub_pages",
]
