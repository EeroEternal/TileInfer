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

from typing import Optional

import torch

from ...metadata import AttentionMode
from ...plan import AttentionPlan
from .base import AttentionBackend, register_backend

__all__ = ["TileLangAscend950Backend"]

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
        if plan.num_splits > 1:
            raise NotImplementedError("KV-split decode is not implemented for this backend yet")
        if plan.kv_layout != "NHD":
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

        from ...kernels.attention.paged_decode_ascend950 import build_decode_kernel, padded_rows

        state["kernel"] = build_decode_kernel(
            batch=plan.batch_size,
            kv_heads=plan.num_kv_heads,
            group=plan.gqa_group_size,
            dim=plan.head_dim,
            page_size=plan.page_size,
            num_pages_cap=int(k_cache.shape[0]),
        )
        state["spec_key"] = spec_key
        state["br"] = padded_rows(plan.gqa_group_size)
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

        from ...kernels.attention.paged_decode_ascend950 import forward

        return forward(
            q,
            k_cache,
            v_cache,
            meta,
            group=plan.gqa_group_size,
            out=out,
        )
