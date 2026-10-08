"""TileLang / Ascend attention backend.

Wiring only — the kernel itself lives in :mod:`tileinfer.kernels.attention`.

Two things are worth pointing out about how this file sits in the plan/run split:

* **Metadata is copied into plan-owned device buffers.**  The kernel is compiled for a fixed page
  *capacity* (the KV pool size), while a step only uses part of it.  Copying the step's
  ``kv_indptr`` / ``kv_indices`` / ``kv_last_page_len`` into buffers of static shape keeps the
  kernel's signature static — which is what makes the run stage capturable — and gives us a
  natural place to do the ``int32 -> float`` cast the vector unit needs for the tail mask.
* **Compilation happens once per shape, at finalisation time**, i.e. before ACLGraph capture.
  ``run`` never triggers a JIT: if the plan was not finalised (for instance a plan was built
  before the caches existed), the first ``run`` finalises it, and every later call is a lookup.
"""

from __future__ import annotations

from typing import Optional

import torch

from ...kernels.attention.paged_decode import DecodeKernelSpec, build_decode_kernel, is_available
from ...metadata import AttentionMode, RaggedMetadata
from ...plan import AttentionPlan
from ...utils import is_npu_available
from .base import AttentionBackend, register_backend

__all__ = ["TileLangAscendBackend"]

#: Page granularity below which the paged kernel has nothing to gain over the reference path.
MIN_PAGE_SIZE = 16


@register_backend
class TileLangAscendBackend(AttentionBackend):
    """Ascend decode attention written in TileLang."""

    name = "tilelang"

    # ------------------------------------------------------------------ availability

    @classmethod
    def is_available(cls) -> bool:
        return is_npu_available() and is_available()

    @classmethod
    def supports_mode(cls, mode: AttentionMode) -> bool:
        return mode is AttentionMode.DECODE

    # ------------------------------------------------------------------ planning

    def finalize_plan(
        self,
        plan: AttentionPlan,
        k_cache: Optional[torch.Tensor] = None,
        v_cache: Optional[torch.Tensor] = None,
    ) -> AttentionPlan:
        """Bind the plan to a concrete kernel variant.

        ``k_cache`` / ``v_cache`` are needed because the kernel is compiled for the pool size;
        only the *shape* is read, never the contents.
        """
        state = plan.backend_state
        if plan.mode is not AttentionMode.DECODE:
            raise NotImplementedError(
                f"the TileLang backend implements decode only so far (got mode={plan.mode.value}); "
                "use backend='reference' for prefill/append or wait for the prefill kernel"
            )
        if plan.num_splits > 1:
            raise NotImplementedError(
                "KV-split decode needs the merge kernel, which is not wired up yet; "
                "build the plan with kv_tile_pages=0"
            )
        group = plan.gqa_group_size
        if plan.gqa_group_size < 2:
            raise NotImplementedError(
                "the TileLang decode kernel requires GQA/MQA (num_qo_heads >= 2 * num_kv_heads); "
                "use backend='reference' for MHA"
            )
        if plan.page_size < MIN_PAGE_SIZE:
            raise NotImplementedError(
                f"page_size={plan.page_size} is too small for the paged kernel "
                f"(minimum {MIN_PAGE_SIZE}); repack the cache or use backend='reference'"
            )
        if k_cache is None or v_cache is None:
            raise ValueError("finalize_plan needs the K/V caches in order to size the kernel")
        if k_cache.shape != v_cache.shape:
            raise ValueError(
                f"K and V caches must have the same shape, got {tuple(k_cache.shape)} and "
                f"{tuple(v_cache.shape)}"
            )
        if k_cache.dim() != 4:
            raise ValueError(f"paged caches must be 4-D, got {tuple(k_cache.shape)}")
        if plan.kv_layout != "NHD":
            raise NotImplementedError("the TileLang kernel expects the NHD cache layout")

        num_pages_cap = int(k_cache.shape[0])
        spec = DecodeKernelSpec(
            batch=plan.batch_size,
            kv_heads=plan.num_kv_heads,
            group=group,
            dim=plan.head_dim,
            block_size=plan.page_size,
            num_pages_cap=num_pages_cap,
        )
        if state.get("spec_key") == spec.key():
            return plan

        kernel = build_decode_kernel(spec)
        device = plan.device
        state.update(
            spec_key=spec.key(),
            spec=spec,
            kernel=kernel,
            kv_indptr=torch.empty(plan.batch_size + 1, dtype=torch.int32, device=device),
            kv_indices=torch.empty(num_pages_cap, dtype=torch.int32, device=device),
            kv_last_page_len=torch.empty(plan.batch_size, dtype=torch.float32, device=device),
            tok_idx=torch.arange(plan.page_size, dtype=torch.float32, device=device),
        )
        state["kv_indptr"].copy_(plan.meta.kv_indptr.to(torch.int32))
        self._refresh_metadata(plan, plan.meta)
        return plan

    # ------------------------------------------------------------------ execution

    def _refresh_metadata(self, plan: AttentionPlan, meta: Optional[RaggedMetadata] = None) -> None:
        """Copy the (possibly updated) step metadata into the plan-owned buffers."""
        state = plan.backend_state
        meta = meta or plan.meta
        state["kv_indptr"].copy_(meta.kv_indptr.to(torch.int32))
        if meta.kv_last_page_len is None:
            raise ValueError("kv_last_page_len is required for a paged cache")
        state["kv_last_page_len"].copy_(meta.kv_last_page_len.to(torch.float32))
        indices = meta.kv_indices.to(torch.int32)
        cap = state["kv_indices"].numel()
        if indices.numel() > cap:
            raise ValueError(
                f"the step references {indices.numel()} pages but the kernel was compiled for a "
                f"pool of {cap} pages; rebuild the plan (cache growth requires a new plan)"
            )
        state["kv_indices"][: indices.numel()].copy_(indices)
        if indices.numel() < cap:
            # slots beyond the live pool are never dereferenced, but keep them deterministic so
            # ACLGraph replay compares bit-identical against an eager run
            state["kv_indices"][indices.numel() :].zero_()

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
        if "kernel" not in plan.backend_state:
            self.finalize_plan(plan, k_cache, v_cache)
        elif kv_indptr is not None or kv_indices is not None or kv_last_page_len is not None:
            plan.meta = RaggedMetadata(
                kv_indptr=kv_indptr if kv_indptr is not None else plan.meta.kv_indptr,
                kv_indices=kv_indices if kv_indices is not None else plan.meta.kv_indices,
                kv_last_page_len=(
                    kv_last_page_len
                    if kv_last_page_len is not None
                    else plan.meta.kv_last_page_len
                ),
                page_size=plan.meta.page_size,
                qo_indptr=plan.meta.qo_indptr,
            )
            self._refresh_metadata(plan, plan.meta)

        state = plan.backend_state
        group, kv_heads = plan.gqa_group_size, plan.num_kv_heads
        q_view = q.view(plan.batch_size, kv_heads, group, plan.head_dim)
        result = state["kernel"](
            q_view,
            k_cache,
            v_cache,
            state["kv_indptr"],
            state["kv_indices"],
            state["tok_idx"],
            state["kv_last_page_len"],
        )
        result = result.view(plan.batch_size, plan.num_qo_heads, 1, plan.head_dim)
        if out is not None:
            out.copy_(result)
            return out
        return result
