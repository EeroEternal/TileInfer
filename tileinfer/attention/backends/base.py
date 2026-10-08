"""Attention backend interface and registry.

A backend owns exactly two things: how a plan is *finalised* (kernel JIT handles, tile sizes)
and how a plan is *executed*.  Everything that is policy rather than mechanism — mode
detection, load-balanced partitioning, workspace sizing — is shared, so a new backend (say an
Ascend C / PTO one) only has to describe its kernels.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Type

import torch

from ...metadata import AttentionMode, RaggedMetadata, detect_mode
from ...plan import AttentionPlan, TileSchedule, WorkspaceManager, plan_decode_tiles, plan_query_tiles

__all__ = ["AttentionBackend", "register_backend", "get_backend", "list_backends", "build_plan"]


def build_plan(
    *,
    backend: str,
    meta: RaggedMetadata,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    mode: Optional[AttentionMode] = None,
    page_size: int = 1,
    sm_scale: Optional[float] = None,
    kv_layout: str = "NHD",
    causal: bool = True,
    window_left: int = -1,
    block_q: int = 128,
    kv_tile_pages: int = 0,
    load_balance: bool = True,
    workspace_manager: Optional[WorkspaceManager] = None,
) -> AttentionPlan:
    """Backend-independent part of planning.

    ``kv_tile_pages`` is the load-balancing knob: ``0`` keeps one tile per request (cheapest, no
    merge), a positive value splits requests into ~``kv_tile_pages``-page work units so that a
    skewed batch spreads over the cores.  Only the decode path can merge splits today, so
    ``kv_tile_pages`` together with ``block_q > 1`` raises rather than silently producing wrong
    output.
    """
    meta.validate()
    page_size = max(page_size, meta.page_size, 1)
    page_counts = (meta.kv_indptr[1:] - meta.kv_indptr[:-1]).to(torch.int64)

    if meta.qo_indptr is None:
        # no query-side metadata: every request contributes exactly one row (decode)
        qo_indptr = torch.arange(
            0, meta.batch_size + 1, dtype=torch.int32, device=meta.device
        )
        meta = RaggedMetadata(
            kv_indptr=meta.kv_indptr,
            kv_indices=meta.kv_indices,
            kv_last_page_len=meta.kv_last_page_len,
            page_size=meta.page_size,
            qo_indptr=qo_indptr,
        )
    qo_lens = meta.qo_lens
    assert qo_lens is not None

    detected = mode or detect_mode(meta.kv_lens, qo_lens)
    if mode is None and bool((qo_lens > 1).any()) and detected is AttentionMode.DECODE:
        detected = AttentionMode.PREFILL

    if detected is AttentionMode.DECODE:
        schedule = plan_decode_tiles(
            page_counts, block_q=1, kv_tile_pages=kv_tile_pages, load_balance=load_balance
        )
        block_q_eff = 1
    else:
        if kv_tile_pages > 0:
            raise NotImplementedError(
                "KV-split scheduling with the merge step is currently implemented for decode "
                "(block_q == 1) only; leave kv_tile_pages=0 for prefill/append"
            )
        schedule = plan_query_tiles(
            qo_lens, page_counts, block_q=block_q, kv_tile_pages=0, load_balance=load_balance
        )
        block_q_eff = block_q

    plan = AttentionPlan(
        mode=detected,
        meta=meta,
        schedule=schedule,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        sm_scale=sm_scale if sm_scale is not None else float(head_dim) ** -0.5,
        page_size=page_size,
        dtype=dtype,
        device=device,
        kv_layout=kv_layout.upper(),
        causal=causal,
        window_left=window_left,
        backend=backend,
        block_q=block_q_eff,
    )

    if schedule.needs_merge:
        from ...plan import allocate_workspace

        manager = workspace_manager or WorkspaceManager()
        plan.workspace = allocate_workspace(plan, manager)
    return plan


class AttentionBackend(ABC):
    """Base class for attention backends."""

    name: str = "base"

    # ------------------------------------------------------------------ capabilities

    @classmethod
    def is_available(cls) -> bool:
        """Whether this backend can actually run on the current machine."""
        return True

    @classmethod
    def supports_mode(cls, mode: AttentionMode) -> bool:
        return True

    # ------------------------------------------------------------------ lifecycle

    def finalize_plan(
        self,
        plan: AttentionPlan,
        k_cache: Optional[torch.Tensor] = None,
        v_cache: Optional[torch.Tensor] = None,
    ) -> AttentionPlan:
        """Attach kernel handles / compile-time decisions to a freshly built plan.

        Called once per *shape* change (not per step), which is where all JIT compilation is
        allowed to happen — the run stage must stay compilation-free for graph capture.  The
        caches are passed because a kernel is compiled for the KV *pool* size, which only the
        cache tensors know; backends that do not need them may ignore the arguments.
        """
        return plan

    @abstractmethod
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
        """Execute ``plan``; shapes are static, so this is capturable.

        The optional metadata arguments let an engine that mutates its buffers into new objects
        every step (rather than in place) still reuse the plan — for tile-based engines this is
        the common case.
        """

    # ------------------------------------------------------------------ convenience

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} name={self.name!r}>"


# ---------------------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------------------

_REGISTRY: Dict[str, Type[AttentionBackend]] = {}


def register_backend(cls: Type[AttentionBackend]) -> Type[AttentionBackend]:
    """Class decorator that adds a backend to the registry under ``cls.name``."""
    if not getattr(cls, "name", None):
        raise ValueError("a backend must define a non-empty `name`")
    _REGISTRY[cls.name] = cls
    return cls


def list_backends(available_only: bool = False) -> List[str]:
    _ensure_builtin_backends()
    names = sorted(_REGISTRY)
    if available_only:
        names = [n for n in names if _REGISTRY[n].is_available()]
    return names


def get_backend(name: str = "auto", **kwargs) -> AttentionBackend:
    """Instantiate a backend.

    ``"auto"`` picks the best *available* one, preferring specialised kernels (``tilelang``)
    over the portable reference path, so that the same user code runs on a laptop and on the NPU.
    """
    _ensure_builtin_backends()
    if name == "auto":
        for candidate in ("tilelang", "reference"):
            cls = _REGISTRY.get(candidate)
            if cls is not None and cls.is_available():
                return cls(**kwargs)
        raise RuntimeError(f"no usable attention backend among {list_backends()}")
    if name not in _REGISTRY:
        raise ValueError(f"unknown backend {name!r}; known: {list_backends()}")
    return _REGISTRY[name](**kwargs)


def _ensure_builtin_backends() -> None:
    """Import the bundled backends lazily so that heavy deps stay optional."""
    if "reference" in _REGISTRY:
        return
    from . import reference as _reference  # noqa: F401
    from . import tilelang_ascend as _tilelang  # noqa: F401
