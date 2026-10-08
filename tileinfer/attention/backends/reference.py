"""Portable torch backend — the oracle, and the fallback when no NPU is around.

It executes the *same plan* the TileLang backend would execute (per tile, honouring split-KV and
the merge step), so it is not only a correctness reference for the kernels: it also validates the
scheduler.  A mismatch between ``reference`` and ``tilelang`` is therefore a kernel bug, never a
planning discrepancy.

Written for clarity, not speed.  Do not use it in production.
"""

from __future__ import annotations

from typing import Optional

import torch

from ...plan import AttentionPlan
from ...testing.reference import reference_attention, reference_attention_tiled
from .base import AttentionBackend, register_backend

__all__ = ["ReferenceBackend"]


@register_backend
class ReferenceBackend(AttentionBackend):
    """Slow, obviously-correct torch implementation."""

    name = "reference"

    @classmethod
    def is_available(cls) -> bool:
        return True

    def run(
        self,
        plan: AttentionPlan,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        kv_indptr: Optional[torch.Tensor] = None,
        kv_indices: Optional[torch.Tensor] = None,
        kv_last_page_len: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        meta = plan.meta
        if kv_indptr is not None or kv_indices is not None or kv_last_page_len is not None:
            meta = RaggedMetadata(
                kv_indptr=kv_indptr if kv_indptr is not None else meta.kv_indptr,
                kv_indices=kv_indices if kv_indices is not None else meta.kv_indices,
                kv_last_page_len=(
                    kv_last_page_len if kv_last_page_len is not None else meta.kv_last_page_len
                ),
                page_size=meta.page_size,
                qo_indptr=meta.qo_indptr,
            )
        result, _, _ = reference_attention_tiled(
            q,
            k_cache,
            v_cache,
            meta,
            plan.schedule,
            layout=plan.kv_layout,
            causal=plan.causal,
            sm_scale=plan.sm_scale,
            window_left=plan.window_left,
            out_dtype=plan.dtype,
            return_partials=plan.needs_merge,
        )
        if plan.needs_merge:
            # the tiled path already merged; kernel-shaped output for consistency
            result = result.reshape(result.shape[0], plan.num_qo_heads, 1, plan.head_dim)
        if out is not None:
            out.copy_(result)
            return out
        return result

    def run_dense(
        self,
        plan: AttentionPlan,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Straight-line oracle that never goes through a schedule (used by the tests)."""
        result = reference_attention(
            q,
            k_cache,
            v_cache,
            plan.meta,
            layout=plan.kv_layout,
            causal=plan.causal,
            sm_scale=plan.sm_scale,
            window_left=plan.window_left,
            out_dtype=plan.dtype,
        )
        if out is not None:
            out.copy_(result)
            return out
        return result
