"""``BatchAttention`` — the single entry point engines talk to.

The API is intentionally shaped like FlashInfer's, because that is the vocabulary the people
wiring up vLLM-Ascend / MindIE already have in their heads:

    attn = BatchAttention(backend="tilelang")
    plan = attn.plan(kv_indptr=..., kv_indices=..., kv_last_page_len=..., ...)
    out  = attn.run(q, k_cache, v_cache, plan=plan)

with the difference that the *plan* is an explicit, inspectable object: it carries the work
schedule, the workspace and the compiled kernel handle.  Engines that capture a graph call
``plan`` once per batch shape and then ``run`` forever; engines that prefer to pass metadata
every step can hand ``run`` fresh ``kv_indptr`` / ``kv_indices`` / ``kv_last_page_len``.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import torch

from ..metadata import AttentionMode, PageTable, RaggedMetadata, as_int32
from ..plan import AttentionPlan, WorkspaceManager
from ..utils import resolve_device
from .backends.base import AttentionBackend, build_plan, get_backend, list_backends

__all__ = ["BatchAttention"]


class BatchAttention:
    """Engine-agnostic batched attention with an explicit plan/run split.

    Parameters
    ----------
    backend:
        ``"auto"``, ``"tilelang"`` (Ascend, decode) or ``"reference"`` (portable oracle).
    device:
        ``"auto"``/``None`` picks the NPU when there is one, CPU otherwise.
    dtype:
        compute/output dtype; fp16 is what the Ascend kernels are built around.
    workspace_manager:
        shared across plans so that repeated shapes do not re-allocate scratch memory.
    """

    def __init__(
        self,
        backend: str = "auto",
        device: Union[str, torch.device, None] = None,
        dtype: Optional[torch.dtype] = None,
        workspace_manager: Optional[WorkspaceManager] = None,
        **backend_kwargs,
    ) -> None:
        self.device = resolve_device(device)
        self.dtype = dtype or torch.float16
        self.backend: AttentionBackend = get_backend(backend, **backend_kwargs)
        self.workspace_manager = workspace_manager or WorkspaceManager()
        self._last_plan: Optional[AttentionPlan] = None
        self._plan_cache: dict = {}

    # ------------------------------------------------------------------ introspection

    @property
    def backend_name(self) -> str:
        return self.backend.name

    @staticmethod
    def available_backends(available_only: bool = False) -> List[str]:
        return list_backends(available_only=available_only)

    def __repr__(self) -> str:
        return f"<BatchAttention backend={self.backend.name!r} device={self.device}>"

    # ------------------------------------------------------------------ plan

    def plan(
        self,
        *,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        kv_indptr: Optional[torch.Tensor] = None,
        kv_indices: Optional[torch.Tensor] = None,
        kv_last_page_len: Optional[torch.Tensor] = None,
        qo_indptr: Optional[torch.Tensor] = None,
        page_table: Optional[PageTable] = None,
        mode: Union[str, AttentionMode, None] = None,
        page_size: Optional[int] = None,
        sm_scale: Optional[float] = None,
        kv_layout: str = "NHD",
        causal: bool = True,
        window_left: int = -1,
        block_q: int = 128,
        kv_tile_pages: int = 0,
        load_balance: bool = True,
        k_cache: Optional[torch.Tensor] = None,
        v_cache: Optional[torch.Tensor] = None,
        finalize: bool = True,
        cache_key: Optional[tuple] = None,
    ) -> AttentionPlan:
        """Build a plan for one batch shape.

        Metadata sources, in order of preference: an engine's dense ``page_table`` (converted to
        ragged form, dropping padding), or the ragged triple
        ``kv_indptr`` / ``kv_indices`` / ``kv_last_page_len``.

        ``finalize=False`` skips kernel binding — useful when the caches do not exist yet; the
        first :meth:`run` will finalise instead.
        """
        meta = self._as_metadata(
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            kv_last_page_len=kv_last_page_len,
            qo_indptr=qo_indptr,
            page_table=page_table,
            page_size=page_size,
        )
        detected_mode = AttentionMode.from_str(mode) if mode is not None else None

        cache_key = cache_key or (
            meta.signature(),
            detected_mode,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            kv_layout,
            causal,
            window_left,
            block_q,
            kv_tile_pages,
            load_balance,
        )
        plan = self._plan_cache.get(cache_key)
        if plan is not None:
            # refresh the metadata *values* (engines update the same buffers in place)
            plan.meta = meta
            self._last_plan = plan
            return plan

        plan = build_plan(
            backend=self.backend.name,
            meta=meta,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=self.dtype,
            device=self.device,
            mode=detected_mode,
            page_size=page_size or meta.page_size,
            sm_scale=sm_scale,
            kv_layout=kv_layout,
            causal=causal,
            window_left=window_left,
            block_q=block_q,
            kv_tile_pages=kv_tile_pages,
            load_balance=load_balance,
            workspace_manager=self.workspace_manager,
        )
        if finalize:
            self.backend.finalize_plan(plan, k_cache, v_cache)
        self._plan_cache[cache_key] = plan
        self._last_plan = plan
        return plan

    def plan_from_page_table(
        self,
        block_table: torch.Tensor,
        seq_lens: Union[Sequence[int], torch.Tensor],
        *,
        qo_lens: Optional[Union[Sequence[int], torch.Tensor]] = None,
        page_size: int = 128,
        **kwargs,
    ) -> AttentionPlan:
        """Convenience wrapper for engines that keep a dense ``block_tables`` tensor.

        This is the one-liner a vLLM-Ascend metadata builder needs: the dense table is compacted
        here, once per batch shape, instead of being walked by every kernel block.
        """
        table = PageTable(
            table=block_table,
            seq_lens=as_int32(seq_lens, block_table.device),
            page_size=page_size,
        )
        meta = table.to_ragged()
        qo_indptr = None
        if qo_lens is not None:
            qo = as_int32(qo_lens, block_table.device)
            qo_indptr = torch.zeros(qo.numel() + 1, dtype=torch.int32, device=block_table.device)
            qo_indptr[1:] = torch.cumsum(qo, dim=0).to(torch.int32)
        return self.plan(
            kv_indptr=meta.kv_indptr,
            kv_indices=meta.kv_indices,
            kv_last_page_len=meta.kv_last_page_len,
            qo_indptr=qo_indptr,
            page_size=page_size,
            **kwargs,
        )

    # ------------------------------------------------------------------ run

    def run(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        plan: Optional[AttentionPlan] = None,
        *,
        kv_indptr: Optional[torch.Tensor] = None,
        kv_indices: Optional[torch.Tensor] = None,
        kv_last_page_len: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Execute a plan.  Pure compute: no allocation, no compilation, no host sync.

        Metadata may be refreshed per call (``kv_indptr`` / ``kv_indices`` /
        ``kv_last_page_len``) for engines that mutate their buffers in place; passing it is
        optional and identical in cost to mutating what the plan captured.
        """
        plan = plan or self._last_plan
        if plan is None:
            raise RuntimeError("no plan: call BatchAttention.plan(...) before run(...)")
        if plan.device != q.device:
            raise ValueError(f"query is on {q.device} but the plan targets {plan.device}")
        if out is not None and out.shape != q.shape:
            raise ValueError(
                f"out must have the query's shape {tuple(q.shape)}, got {tuple(out.shape)}"
            )
        return self.backend.run(
            plan,
            q,
            k_cache,
            v_cache,
            out,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            kv_last_page_len=kv_last_page_len,
        )

    def __call__(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return self.run(q, k_cache, v_cache, **kwargs)

    # ------------------------------------------------------------------ internals

    def _as_metadata(
        self,
        *,
        kv_indptr: Optional[torch.Tensor],
        kv_indices: Optional[torch.Tensor],
        kv_last_page_len: Optional[torch.Tensor],
        qo_indptr: Optional[torch.Tensor],
        page_table: Optional[PageTable],
        page_size: Optional[int],
    ) -> RaggedMetadata:
        if page_table is not None:
            if kv_indptr is not None or kv_indices is not None:
                raise ValueError("pass either page_table or ragged metadata, not both")
            meta = page_table.to_ragged()
            if qo_indptr is not None:
                meta.qo_indptr = as_int32(qo_indptr, self.device)
            return meta
        if kv_indptr is None or kv_indices is None:
            raise ValueError(
                "plan() needs either page_table=... or kv_indptr=... and kv_indices=..."
            )
        meta = RaggedMetadata(
            kv_indptr=as_int32(kv_indptr, self.device),
            kv_indices=as_int32(kv_indices, self.device),
            kv_last_page_len=(
                None if kv_last_page_len is None else as_int32(kv_last_page_len, self.device)
            ),
            page_size=int(page_size or 1),
            qo_indptr=None if qo_indptr is None else as_int32(qo_indptr, self.device),
        )
        if meta.page_size > 1 and meta.kv_last_page_len is None:
            raise ValueError("kv_last_page_len is required when page_size > 1")
        return meta

    def clear_plan_cache(self) -> None:
        """Drop cached plans (and with them, compiled kernel handles)."""
        self._plan_cache.clear()
        self._last_plan = None
        self.workspace_manager.clear()
