"""TileInfer backend for the **official** TileLang Ascend 950 support (``tilelang.ascend``).

This is the backend that actually runs on the reference machine, and it is deliberately thin: the
kernel lives in :mod:`tileinfer.kernels.attention.paged_decode_ascend950`, the planning (metadata,
schedule, workspace) is the shared layer in :mod:`tileinfer.plan`, and everything here is the glue
between them plus the padding convention the kernel requires.

Environment this needs (see ``docs/architecture.md``):

* ``tilelang>=0.1.15`` from PyPI — the wheel that ships the Ascend 950 backend,
* a CANN 9.3.x toolkit (a user-local, side-by-side install is fine; the driver is not touched),
* ``torch``/``torch_npu`` that agree with each other and can see the device.

Keep the imports lazy: none of this may break ``import tileinfer`` on a machine without any of it.
"""

from __future__ import annotations

import os
from typing import Optional

import torch

from ...metadata import AttentionMode
from ...plan import AttentionPlan
from .base import AttentionBackend, register_backend

__all__ = ["TileLangAscend950Backend"]


def _refresh_metadata(plan: AttentionPlan, state: dict, meta=None) -> None:
    """Copy the step's page table into the plan-owned buffers."""
    meta = meta or plan.meta
    state["kv_indptr_buf"].copy_(meta.kv_indptr.to(torch.int32))
    state["last_page_len_buf"].copy_(meta.kv_last_page_len.to(torch.int32))
    idx = meta.kv_indices.to(torch.int32)
    cap = state["kv_indices_buf"].numel()
    if idx.numel() > cap:
        raise ValueError(
            f"step references {idx.numel()} pages but the plan holds {cap}; rebuild the plan"
        )
    state["kv_indices_buf"][: idx.numel()].copy_(idx)
    if idx.numel() < cap:
        state["kv_indices_buf"][idx.numel() :].zero_()

_TILELANG_DTYPE = {
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
    torch.float32: "float32",
}

try:  # pragma: no cover - import guard, exercised by the CPU test suite
    import tilelang  # noqa: F401
    import tilelang.ascend  # noqa: F401

    _ASCEND950_IMPORT_ERROR: Optional[BaseException] = None
except Exception as _exc:  # pragma: no cover
    _ASCEND950_IMPORT_ERROR = _exc


@register_backend
class TileLangAscend950Backend(AttentionBackend):
    """Paged decode on Ascend 950 via the official TileLang wheel."""

    name = "tilelang-ascend950"

    # ------------------------------------------------------------------ availability

    @classmethod
    def is_available(cls) -> bool:
        from ...utils import is_npu_available

        return _ASCEND950_IMPORT_ERROR is None and is_npu_available()

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
        """Compile the kernel for this plan's shape.

        Compilation is slow (tens of seconds) and must therefore happen here, never in ``run``: the
        run stage of a serving engine is captured into an ACLGraph.
        """
        if plan.mode is not AttentionMode.DECODE:
            raise NotImplementedError(
                f"the Ascend 950 kernel implements decode only so far (got mode={plan.mode.value})"
            )
        if plan.kv_layout.upper() != "NHD":
            raise NotImplementedError("the Ascend 950 kernel expects the NHD cache layout")
        if k_cache is None or v_cache is None:
            raise ValueError("finalize_plan needs the K/V caches to size the kernel")
        if k_cache.shape != v_cache.shape:
            raise ValueError(
                f"K and V caches must have the same shape, got {tuple(k_cache.shape)} and "
                f"{tuple(v_cache.shape)}"
            )

        state = plan.backend_state
        spec_key = (
            plan.batch_size,
            plan.num_kv_heads,
            plan.gqa_group_size,
            plan.head_dim,
            plan.page_size,
            int(k_cache.shape[0]),
        )
        if state.get("spec_key") == spec_key:
            return plan

        from ...kernels.attention.paged_decode_ascend950 import (
            build_tile_slots,
            padded_rows,
        )

        br = padded_rows(plan.gqa_group_size)
        state["spec_key"] = spec_key
        state["br"] = br

        if plan.schedule.needs_merge and not os.environ.get("TILEINFER_ALLOW_SPLIT_KV"):
            # Split-KV is measured at 3x on the starved shapes and its tests pass, but it is not
            # enabled by default: inside a process that compiles and runs *several* shapes in
            # sequence it has twice wedged the device with a vector-core exception
            # (ACL_ERROR_RT_VECTOR_CORE_EXCEPTION), which is not reproducible for a single shape nor
            # across 20 consecutive launches of one plan.  Until that is understood, opting in is a
            # deliberate choice - see docs/known-issues.md.
            raise NotImplementedError(
                "split-KV is opt-in: set TILEINFER_ALLOW_SPLIT_KV=1 (measured 3x at batch 1, "
                "validated by tests) if you accept the multi-shape instability documented in "
                "docs/known-issues.md.  kv_tile_pages=0 uses the plain decode path."
            )

        if plan.schedule.needs_merge:
            # Split-KV: the schedule is fixed for this shape, so every buffer and lookup table the
            # two launches need is allocated here, once.  The run stage then only copies the step's
            # query in and launches - no allocation, no host work, capturable.
            max_splits = plan.num_splits
            num_slots = plan.batch_size * max_splits
            kv_heads = plan.num_kv_heads
            accum = torch.float32
            state["max_splits"] = max_splits
            state["tile_slots"] = build_tile_slots(plan.schedule, max_splits).to(plan.device)
            # Zeros, not empty: slots a request does not use are read by the merge kernel and
            # multiplied by a zero weight, so uninitialised memory (NaN/Inf) would poison the
            # result.  Real slots are overwritten by the split kernel every step.
            state["part_out"] = torch.zeros(
                (num_slots * kv_heads, br, plan.head_dim), dtype=plan.dtype, device=plan.device
            )
            state["part_lse"] = torch.full(
                (num_slots * kv_heads * br,),
                float("-inf"),
                dtype=accum,
                device=plan.device,
            )
            state["out_pad"] = torch.empty(
                (plan.batch_size, kv_heads * br, plan.head_dim),
                dtype=plan.dtype,
                device=plan.device,
            )
            state["q_pad"] = torch.zeros(
                (plan.batch_size, kv_heads * br, plan.head_dim),
                dtype=plan.dtype,
                device=plan.device,
            )
            # Plan-owned metadata buffers: the kernels read the engine's page table through their
            # own copies, so a kernel can never write into a caller's tensor, and the addresses
            # stay stable across steps (which is what graph capture wants).  This mirrors the
            # fork backend's `_refresh_metadata`.
            cap = int(k_cache.shape[0])
            state["page_cap"] = cap
            state["kv_indptr_buf"] = torch.empty(plan.batch_size + 1, dtype=torch.int32, device=plan.device)
            state["kv_indices_buf"] = torch.zeros(cap, dtype=torch.int32, device=plan.device)
            state["last_page_len_buf"] = torch.empty(plan.batch_size, dtype=torch.int32, device=plan.device)
            _refresh_metadata(plan, state, meta=None)
        else:
            from ...kernels.attention.paged_decode_ascend950 import build_decode_kernel

            state["kernel"] = build_decode_kernel(
                batch=plan.batch_size,
                kv_heads=plan.num_kv_heads,
                group=plan.gqa_group_size,
                dim=plan.head_dim,
                page_size=plan.page_size,
                num_pages_cap=int(k_cache.shape[0]),
                dtype=_TILELANG_DTYPE[plan.dtype],
            )
        return plan

    # ------------------------------------------------------------------ execution

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

        meta = plan.meta
        if kv_indptr is not None or kv_indices is not None or kv_last_page_len is not None:
            from ...metadata import RaggedMetadata

            meta = RaggedMetadata(
                kv_indptr=kv_indptr if kv_indptr is not None else meta.kv_indptr,
                kv_indices=kv_indices if kv_indices is not None else meta.kv_indices,
                kv_last_page_len=(
                    kv_last_page_len if kv_last_page_len is not None else meta.kv_last_page_len
                ),
                page_size=meta.page_size,
                qo_indptr=meta.qo_indptr,
            )

        state = plan.backend_state
        if "max_splits" in state:
            from ...kernels.attention.paged_decode_ascend950 import forward_split

            _refresh_metadata(plan, state, meta)
            return forward_split(
                q,
                k_cache,
                v_cache,
                meta,
                plan.schedule,
                group=plan.gqa_group_size,
                max_splits=state["max_splits"],
                tile_slots=state["tile_slots"],
                part_out=state["part_out"],
                part_lse=state["part_lse"],
                out_pad=state["out_pad"],
                q_pad=state["q_pad"],
                dtype=_TILELANG_DTYPE[plan.dtype],
                kv_indptr=state["kv_indptr_buf"],
                kv_indices=state["kv_indices_buf"],
                kv_last_page_len=state["last_page_len_buf"],
                out=out,
            )

        from ...kernels.attention.paged_decode_ascend950 import forward

        return forward(q, k_cache, v_cache, meta, group=plan.gqa_group_size, out=out)
