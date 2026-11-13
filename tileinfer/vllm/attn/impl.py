"""``TileInferAttentionImpl`` — one attention layer's forward, wired to the serving layer.

The call graph of a decode step
-------------------------------
::

    k_half, v_half = fused_halves(kv_cache)                  # K / V [blocks, page, heads, dim]
        |
    write_kv_cache(key, value, k_half, v_half, slot_mapping) # (a) BEFORE attention reads the pool
        |
    k_pool, v_pool = pool_sub_view(k_half, sub)              # (b) zero-copy 128-token page views
        |                       pool_sub_view(v_half, sub)
        |
    state = self._state_for(meta, k_pool, v_pool)            # per-bucket, allocated once
    refresh_step(state.step, block_table, seq_lens, ...)     # device-side dense -> ragged
        |
    plan = PLAN_CACHE.get(key)  or  PLAN_CACHE.get_or_start(...)
        |
    BatchAttention.run(q_buf, k_pool, v_pool, out=out_buf, plan=plan, <ragged>)
        |
    output.view(-1, heads, dim).copy_(out_buf[:num_actual_tokens])

Why each step is where it is
----------------------------
* **(a) before attention.**  The engine hands the *new* tokens' K/V to this layer and expects
  attention to see them.  Writing after the kernel would read a pool that is missing this step's
  tokens — silently wrong output, no crash.
* **(b) zero copy.**  ``get_kv_cache_shape`` is block-first and ``get_kv_cache_stride_order`` puts
  the K/V factor outermost, so ``kv_cache[:, 0]``/``kv_cache[:, 1]`` are *contiguous* halves
  (``fused_halves``) and ``pool_sub_view`` is a ``view`` (it raises rather than falling back to a
  reshape).  The engine page (2048 tokens) is 16 consecutive kernel pages.  Page ids stay logical:
  a half is a plain ``[blocks * sub, 128, heads, dim]`` array, so the fused layout does not leak
  into the kernel's page-table arithmetic.
* **ragged rebuild on the device.**  ``refresh_step`` rewrites ``kv_indptr`` / ``kv_indices`` /
  ``kv_last_page_len`` in place with fixed shapes: no allocation, no host sync, legal inside a
  captured graph (``tileinfer/serving/step.py`` docstring).
* **bucket-padded staging.**  The kernel is compiled per batch bucket, so the query and output are
  staged into bucket-sized buffers owned by this layer.  Rows beyond ``num_actual_tokens`` stay
  zero (query) or are overwritten (output) and are never copied back.

Fallbacks are explicit and counted
----------------------------------
There is no ``except: pass`` and no "try the fast path, else return something" in this file.  The
degraded path is always the *reference* backend (``tileinfer.attention.backends.reference``: the
same plan, executed in plain torch — slow, obviously correct), always behind a counter:

=============================  =========================================================
counter                        when
=============================  =========================================================
``prefill_kernel``             ``max_query_len > 1`` and the paged-prefill kernel served it
                               (``.prefill.PrefillPath``)
``prefill_reference``          ``max_query_len > 1`` and the reference backend computed it:
                               either ``TILEINFER_PREFILL=0`` or the shape is unsupported
``prefill_unsupported``        the kernel path declined a prefill step; one
                               ``prefill_unsupported_<reason>`` counter records why
``plan_not_ready``             the plan's background build is in flight; this step degrades
                               and the next one hits (``serving/planner.py:165-184``)
``plan_failed``                the plan failed to compile; degrades for the rest of the
                               process
``kernel_backend_unavailable`` the kernel backend is not importable (tilelang/device
                               missing) — counted once, then every step is a
                               ``kernel_path_unavailable``
``kernel_path_unavailable``    per step, once the above has been detected
=============================  =========================================================

Hard errors (never degraded, because a degraded answer would be *wrong*): unsupported metadata
type, float output quantisation, an empty KV cache, missing K/V, a non-positive query length, and
any shape/dtype mismatch.  Each bumps a counter before raising, so a probe can tell "the engine
was misconfigured" from "the engine never got here".
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch

try:  # pragma: no cover - exercised only outside a vLLM env (the CPU suite stubs `vllm`)
    from vllm.v1.attention.backend import AttentionImpl, AttentionType
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "tileinfer.vllm.attn.impl requires vLLM 0.23's v1 attention API "
        "(vllm.v1.attention.backend.AttentionImpl / AttentionType); importing it failed with "
        f"{exc!r}. Install vllm>=0.23,<0.24 - this module is not importable without vLLM."
    ) from exc

from tileinfer.attention.api import BatchAttention
from tileinfer.serving.buckets import (
    KERNEL_PAGE_SIZE,
    declared_pool_cap,
    fused_halves,
    pool_sub_view,
    sub_pages,
)
from tileinfer.serving.counters import Counters
from tileinfer.serving.planner import PlanCache, plan_key
from tileinfer.serving import phases as _PHASES
from tileinfer.serving.step import StepBuffers, allocate_step_buffers, refresh_step

from .backend import SUPPORTED_KV_CACHE_DTYPES, TileInferAttentionBackend
from .kv_cache import write_kv_cache
from .metadata import TileInferMetadata
from .prefill import PrefillPath, PrefillUnsupported
from .window import reset_shared_stagers, shared_stager

__all__ = [
    "TileInferAttentionImpl",
    "IMPL_COUNTERS",
    "IMPL_COUNTER_INITIAL",
    "FALLBACK_REASONS",
    "PLAN_CACHE",
    "snapshot",
    "reset_counters",
    "reset_runtime_state",
]

logger = logging.getLogger("tileinfer.vllm.attn.impl")

#: Every degraded path this module can take.  Each name is also a counter key (pinned by the
#: contract test) — the port plan's §8 R5 requires "all fallbacks counted", and a set that a test
#: enforces is what keeps a new fallback from being added without one.
FALLBACK_REASONS: Tuple[str, ...] = (
    "prefill_reference",
    "plan_not_ready",
    "plan_failed",
    "kernel_path_unavailable",
)

IMPL_COUNTER_INITIAL = {
    # steps
    "forward_calls": 0,
    "decode_steps": 0,
    "prefill_steps": 0,
    "kv_writes": 0,
    # fallbacks (FALLBACK_REASONS, plus the one-time discovery below)
    "prefill_reference": 0,
    # M1: prefill served by the paged-prefill kernel (§17 ②③), and why it could not be
    "prefill_kernel": 0,
    "prefill_unsupported": 0,
    "plan_not_ready": 0,
    "plan_failed": 0,
    "kernel_path_unavailable": 0,
    "kernel_backend_unavailable": 0,
    # state / plan lifecycle (must be zero inside a captured region — W8 assertion)
    "state_builds": 0,
    "plan_builds": 0,
    "reference_plans": 0,
    # explicit non-fatal adaptations
    "padded_input_rows": 0,
    "input_reshape_copy": 0,
    "output_copyback": 0,
    "output_allocated": 0,
    # windowed layers (deliberate dispatch, not a fallback): the paged kernels are full-causal
    # only, so a sliding-window layer computes on the reference path (see the constructor note)
    "decode_window_staged": 0,
    "decode_window_reference": 0,
    "prefill_window_reference": 0,
    # per-step ragged triple sharing (one refresh per step instead of one per attention layer)
    "step_share_refresh": 0,
    "step_share_reuse": 0,
    # hard errors (counted before raising)
    "output_quant_unsupported": 0,
    "metadata_mismatch": 0,
    "empty_kv_cache": 0,
    "missing_kv": 0,
    "empty_step": 0,
    "bad_query_len": 0,
    "shape_mismatch": 0,
}

IMPL_COUNTERS = Counters(IMPL_COUNTER_INITIAL)

#: Process-wide plan cache: one compiled plan per (backend, dtype, device, bucket, width,
#: block_size, pool cap).  Shared on purpose — a 10-layer attention stack asking for the same
#: shape would otherwise start ten identical compiles (``serving/planner.py``'s single-flight is
#: per cache instance).
PLAN_CACHE = PlanCache()

_RUNTIME_LOCK = threading.Lock()
_DECODE_ATTN: Optional[BatchAttention] = None
_REFERENCE_ATTN: Optional[BatchAttention] = None
_REFERENCE_ERROR: Optional[BaseException] = None
#: Prefill fallback reasons already logged (one warning per reason, not per step).
_PREFILL_WARNED: set = set()
#: Sliding-window sizes already logged (one warning per window, not per layer/step).
_WINDOW_WARNED: set = set()


def snapshot() -> Dict[str, object]:
    """Counters + plan-cache state as plain data (probes, W8 assertions)."""
    data: Dict[str, object] = dict(IMPL_COUNTERS.snapshot())
    data["plans"] = PLAN_CACHE.snapshot()
    return data


def reset_counters() -> None:
    """Zero every counter *without* dropping the declared keys (a probe relies on the schema)."""
    IMPL_COUNTERS.reset()
    IMPL_COUNTERS.update(IMPL_COUNTER_INITIAL)


def reset_runtime_state() -> None:
    """Drop the process-wide plans and backend singletons (tests / engine teardown)."""
    global _DECODE_ATTN, _REFERENCE_ATTN, _REFERENCE_ERROR
    with _RUNTIME_LOCK:
        PLAN_CACHE.clear()
        _DECODE_ATTN = None
        _REFERENCE_ATTN = None
        _REFERENCE_ERROR = None
        _PREFILL_WARNED.clear()
        _WINDOW_WARNED.clear()
        _SHARED_STEPS.clear()
        reset_shared_stagers()


def _step_share_enabled() -> bool:
    """Whether the batch's ragged triple is computed once per step instead of once per layer.

    Default off until measured (C2).  ``TILEINFER_ATTN_STEP_SHARE=0`` forces the per-layer rebuild.
    """
    raw = os.environ.get("TILEINFER_ATTN_STEP_SHARE", "").strip().lower()
    return raw not in ("", "0", "false", "no")


@dataclass
class _SharedStep:
    """The per-batch ragged triple, shared by every attention layer of one step.

    ``refresh_step`` derives ``kv_indptr`` / ``kv_indices`` / ``kv_last_page_len`` from the batch's
    page table and ``seq_lens`` — that is a property of the *batch*, not of a layer.  Without
    sharing, Qwen3.6's ten full-attention layers redo the same ~15 device ops every step, and §22
    established that what costs at batch 1 is the dispatch (~25-45 us/op), not the arithmetic.

    ``meta_owner`` holds a **strong reference** to the metadata object the triple was built from, so
    "same object" really means "same step" (vLLM builds one metadata per step) and the address
    cannot be recycled into a different step's object.
    """

    step: StepBuffers
    kv_indices: torch.Tensor  # [pool_cap] int32
    meta_owner: Optional[object] = None


_SHARED_STEPS: Dict[Tuple[int, int, int, int, str], _SharedStep] = {}


def _shared_step_for(state: "_BucketState") -> _SharedStep:
    """The process-wide ragged triple for one bucket shape (allocated on first sight)."""
    key = (state.bucket, state.width, state.block_size, state.pool_cap, str(state.device))
    shared = _SHARED_STEPS.get(key)
    if shared is None:
        shared = _SharedStep(
            step=allocate_step_buffers(
                bucket=state.bucket,
                width=state.width,
                kv_indices_capacity=state.pool_cap,
                block_size=state.block_size,
                device=state.device,
            ),
            kv_indices=torch.zeros(state.pool_cap, dtype=torch.int32, device=state.device),
        )
        _SHARED_STEPS[key] = shared
        logger.warning(
            "TileInfer attention: shared step buffers for bucket=%d width=%d pool_cap=%d "
            "(the ragged triple is now rebuilt once per step, not once per layer)",
            state.bucket,
            state.width,
            state.pool_cap,
        )
    return shared


def _decode_backend_name() -> str:
    """Which TileInfer backend executes decode steps.

    ``tilelang-ascend950`` is the real kernel
    (``tileinfer/attention/backends/tilelang_ascend950.py``).  Overridable for A/B runs; **not**
    defaulted to ``"auto"``, because ``auto`` silently falls back to the reference backend on a
    machine without the toolchain, and a whole serving run on the oracle would look like a
    performance regression rather than a missing kernel.
    """
    raw = os.environ.get("TILEINFER_ATTN_BACKEND", "").strip()
    return raw or "tilelang-ascend950"


def _prefill_enabled() -> bool:
    """Whether prefill may use the paged-prefill kernel (``TILEINFER_PREFILL=0`` forces the
    reference backend, which is the same-process control for the C4 A/B)."""
    raw = os.environ.get("TILEINFER_PREFILL", "").strip().lower()
    return raw not in ("0", "off", "false", "no")


def _window_decode_enabled() -> bool:
    """Whether windowed decode stages the window for the kernel path (``TILEINFER_WINDOW_DECODE=0``
    forces the masked reference decode — the same-process A/B control, like ``TILEINFER_PREFILL``).
    On by default: the staging is exact, and the fallbacks inside the kernel path remain correct
    because the scratch *is* the window."""
    raw = os.environ.get("TILEINFER_WINDOW_DECODE", "").strip().lower()
    return raw not in ("0", "off", "false", "no")


def _dynamo_opaque(fn):
    """Make ``fn`` a graph break for ``torch.compile`` instead of something it tries to trace.

    vLLM 0.23 compiles the model body (``CompilationMode.VLLM_COMPILE``) and does **not** treat
    ``unified_attention_with_output`` as a graph boundary for every architecture, so this
    ``forward`` gets traced.  Tracing it fails on the first thing that is not tensor math — a
    counted step takes ``threading.Lock`` (``serving/counters.py:33``) and dynamo raises
    ``torch._dynamo.exc.Unsupported: Unsupported context manager`` (measured: engine refused to
    start once ``--enforce-eager`` was dropped, §18.8).

    **But this decorator is also what blocks vLLM's graph-capture path**: with ``fullgraph=True`` a
    ``torch.compiler.disable``'d callee is a hard error (§66.3), so capture can never engage while
    it is applied.  ``TILEINFER_DYNAMO_OPAQUE=0`` drops it, which is the switch to experiment with
    once the traced path is clean (`Counters.bump` now no-ops while tracing).
    """
    if os.environ.get("TILEINFER_DYNAMO_OPAQUE", "1").strip().lower() in ("0", "false", "no"):
        return fn
    try:
        from torch._dynamo import disable as _disable
    except Exception:  # pragma: no cover - torch always ships _dynamo
        return fn
    return _disable(fn)


def _decode_attn_or_none(dtype: Optional[torch.dtype] = None) -> Optional[BatchAttention]:
    """The decode backend, or ``None`` if it cannot be built (counted once, logged once).

    ``dtype`` must be the **pool/query** dtype: ``BatchAttention`` defaults to ``torch.float16``
    (``attention/api.py:54``), and the kernel variant is compiled from ``plan.dtype`` — so building
    it without the dtype compiled an fp16 kernel against a bf16 pool and the first step died with
    ``RuntimeError: kernel main input KCache dtype mismatch, expected float16`` (hardware run #1).
    """
    global _DECODE_ATTN
    if _DECODE_ATTN is not None:
        if dtype is None or getattr(_DECODE_ATTN, "dtype", dtype) == dtype:
            return _DECODE_ATTN
        # A backend built before the pool existed (pre-warm) may carry the fp16 default; the
        # kernel variant is compiled from that dtype, so rebuild it once the real dtype is known.
        with _RUNTIME_LOCK:
            _DECODE_ATTN = None
    with _RUNTIME_LOCK:
        if _DECODE_ATTN is None:
            name = _decode_backend_name()
            try:
                _DECODE_ATTN = BatchAttention(backend=name, dtype=dtype)
            except Exception as exc:  # noqa: BLE001 - missing toolchain / device
                IMPL_COUNTERS.bump("kernel_backend_unavailable")
                logger.warning(
                    "TileInfer attention: kernel backend %r is unavailable (%r); every decode "
                    "step will be computed by the reference backend",
                    name,
                    exc,
                )
                return None
    return _DECODE_ATTN


def _reference_attn() -> BatchAttention:
    """The degraded path's backend.  Raises (loudly) if even the oracle cannot be built."""
    global _REFERENCE_ATTN, _REFERENCE_ERROR
    if _REFERENCE_ATTN is not None:
        return _REFERENCE_ATTN
    with _RUNTIME_LOCK:
        if _REFERENCE_ATTN is None:
            if _REFERENCE_ERROR is not None:
                raise RuntimeError(
                    "the TileInfer reference attention backend failed to import earlier"
                ) from _REFERENCE_ERROR
            try:
                _REFERENCE_ATTN = BatchAttention(backend="reference")
            except Exception as exc:  # noqa: BLE001
                _REFERENCE_ERROR = exc
                raise RuntimeError(
                    "the TileInfer reference attention backend is unavailable; there is no "
                    "correctness-preserving fallback left"
                ) from exc
    return _REFERENCE_ATTN


@dataclass
class _BucketState:
    """Per-bucket scratch for one layer: buffers with stable addresses, allocated once."""

    bucket: int
    width: int
    block_size: int
    pool_cap: int
    dtype: torch.dtype
    device: torch.device
    k_pool: torch.Tensor  # [pool_pages, 128, kv_heads, head_size] view of the K half
    v_pool: torch.Tensor  # same, V half
    step: StepBuffers
    kv_indices: torch.Tensor  # [pool_cap] int32; the page-id list the kernel reads
    q_buf: torch.Tensor  # [bucket, num_heads, head_size]
    out_buf: torch.Tensor  # [bucket, num_heads, head_size]


class TileInferAttentionImpl(AttentionImpl):
    """Paged decode attention on the TileInfer kernels, one instance per attention layer."""

    #: The reference backend can return softmax lse for decode
    #: (``v1/attention/backend.py:705-707``); this impl never does.
    can_return_lse_for_decode = False
    supports_pcp = False
    supports_quant_query_input = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
    ) -> None:
        """The vLLM 0.23 constructor (``v1/attention/backend.py:780-794``), validated up front.

        No ``super().__init__()`` call: ``AttentionImpl.__init__`` is the abstract method being
        implemented (its body is ``raise NotImplementedError``, ``:794``), and
        ``AttentionImplBase`` has no ``__init__`` at all — its ``__new__`` (``:738-764``) has
        already run and set the DCP/PCP ranks.  Calling ``super().__init__()`` here would hit
        that ``raise``.

        Every unsupported knob raises *now* (model construction), not on the first step.  The
        ones ``AttentionBackend.validate_configuration`` cannot see — there are no head counts and
        no sliding window in its signature (``:275-291``) — are exactly the ones checked here.
        """
        num_kv_heads = num_heads if num_kv_heads is None else int(num_kv_heads)
        TileInferAttentionBackend.check_head_layout(num_heads, num_kv_heads, head_size)

        if alibi_slopes is not None:
            raise NotImplementedError(
                "alibi slopes are not supported by the TileInfer attention backend"
            )
        # Sliding-window (iRoPE) layers: the window is honoured on the reference path, whose
        # mask uses absolute positions (``window_left``: query at ``q_pos`` sees
        # ``[q_pos - window_left, q_pos]``; a window of ``w`` tokens is ``window_left = w - 1``).
        # The paged kernel paths are full-causal only, so a windowed layer never reaches them
        # (``forward`` branches before dispatch); the graph-capture support is downgraded to
        # match (``TileInferMetadataBuilder.get_cudagraph_support``).  What is NOT done here is
        # vLLM's hybrid-manager memory optimisation (dropping out-of-window blocks): the
        # unified-spec pool keeps every block, and the window exists only in the compute.
        window: Optional[int] = None
        if sliding_window is not None:
            window = int(sliding_window)
            if window < 1:
                raise ValueError(
                    f"sliding_window must be a positive number of tokens, got {sliding_window}"
                )
            if window not in _WINDOW_WARNED:
                _WINDOW_WARNED.add(window)
                logger.warning(
                    "TileInfer attention: sliding-window layers (window=%d) decode via exact "
                    "scratch staging onto the paged kernel path (prefill stays on the masked "
                    "reference path; capture is not verified for the staged path, so windowed "
                    "specs downgrade to --enforce-eager)",
                    window,
                )
        self.window_size: Optional[int] = window
        if logits_soft_cap is not None:
            raise NotImplementedError(
                "logits soft capping is not supported by the TileInfer attention backend"
            )
        if kv_sharing_target_layer_name is not None:
            raise NotImplementedError(
                "KV-sharing between layers is not supported by the TileInfer attention backend"
            )
        # Checked twice on purpose: a layer can be constructed before the backend's
        # `validate_configuration` runs, and this is the only place the impl can see the dtype.
        if kv_cache_dtype not in SUPPORTED_KV_CACHE_DTYPES:
            raise NotImplementedError(
                f"kv_cache_dtype {kv_cache_dtype!r} is not supported (unquantised only: "
                f"{list(SUPPORTED_KV_CACHE_DTYPES)})"
            )
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                f"attention type {attn_type!r} is not supported "
                f"(only {AttentionType.DECODER.value!r})"
            )

        self.num_heads = int(num_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.head_size = int(head_size)
        self.scale = float(scale)
        self.kv_cache_dtype = kv_cache_dtype
        self.attn_type = attn_type
        # Read by `vllm.v1.attention.backends.utils.get_per_layer_parameters` (:126-130); must be
        # `None` rather than an int, because that function indexes it as a tuple.  The window
        # itself lives in ``self.window_size`` (see the constructor's sliding-window note).
        self.sliding_window = None
        self.logits_soft_cap = None
        self.alibi_slopes = None
        self.kv_sharing_target_layer_name = None
        self.gqa_group = self.num_heads // self.num_kv_heads

        self._states: Dict[int, _BucketState] = {}
        self._no_kernel_backend = False
        self._backend_name = _decode_backend_name()
        self._split_kv_pages = _split_kv_pages()
        self._prefill_enabled = _prefill_enabled()
        # Built here, never lazily: this constructor runs outside any torch.compile/dynamo trace,
        # while a lazy build inside ``forward`` needs a ``threading.Lock`` — and dynamo refuses that
        # context manager outright ("Unsupported context manager: lock"), which is exactly how the
        # engine failed to start once ``--enforce-eager`` was dropped (§18.8).
        self._prefill: Optional[PrefillPath] = None
        if self._prefill_enabled:
            try:
                self._prefill = PrefillPath(
                    self.num_heads, self.num_kv_heads, self.head_size, self.scale
                )
            except Exception:  # noqa: BLE001 - geometry the kernel cannot serve
                logger.exception(
                    "TileInfer attention: the prefill kernel path is unavailable; every prefill "
                    "step will use the reference backend"
                )
                self._prefill_enabled = False
        # Windowed decode: the shared scratch stager + this layer's own per-bucket state dict,
        # so the scratch pools never share a bucket key (or a plan key) with the real-pool
        # states.  The stager is shared across same-geometry layers (they all stage the same
        # window from the same pool - one gather per step instead of one per layer); the states
        # stay per-layer because the query/output buffers are per-layer.  Built here (pure
        # geometry, no device), never lazily — same dynamo reasoning as ``_prefill``.
        self._window_states: Dict[int, _BucketState] = {}
        self._stager = None
        if self.window_size is not None:
            self._stager = shared_stager(self.num_kv_heads, self.head_size, self.window_size)

    # ------------------------------------------------------------------ vLLM contract

    @_dynamo_opaque
    def forward(
        self,
        layer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One layer's attention for one step.

        Signature copied verbatim from ``v1/attention/backend.py:796-809``; the parameter names are
        part of the contract test because ``Attention.forward`` passes them by name.

        ``layer`` is accepted and ignored: it only carries the quantisation scales
        (``AttentionLayer`` protocol, ``:673-689``), and this backend rejects every quantised
        configuration at construction time, so there is nothing to read from it.
        """
        IMPL_COUNTERS.bump("forward_calls")

        # Refusals.  Counted before raising; none of them degrades, because a degraded answer here
        # would be a wrong answer, not a slow one.
        if output_scale is not None or output_block_scale is not None:
            IMPL_COUNTERS.bump("output_quant_unsupported")
            raise NotImplementedError(
                "the TileInfer attention backend does not support fused output quantisation "
                "(output_scale / output_block_scale were provided); use an unquantised model"
            )
        meta = attn_metadata
        if meta is None:
            # vLLM's own contract: during a V1 *profile* run there is no attention metadata and
            # the layers return early (qwen_gdn_linear_attn.py:1186-1190 says so in as many words:
            # "_forward_core returns early because attn_metadata is None").  The profile run only
            # measures memory, so skipping the compute is the faithful behaviour - raising here
            # stopped the engine during load_model on the first hardware run.
            #
            # It is also the last moment *before* graph capture: build the decode backend here, or
            # the first construction (plan + tilelang compile, which synchronises) happens inside the
            # captured region and capture dies with ``rtStreamSynchronizeWithTimeout`` (§79).
            IMPL_COUNTERS.bump("profile_passthrough")
            if query is not None:
                _decode_attn_or_none(query.dtype)
            return output
        if not isinstance(meta, TileInferMetadata):
            IMPL_COUNTERS.bump("metadata_mismatch")
            raise TypeError(
                "TileInferAttentionImpl.forward expects TileInferMetadata (built by "
                f"TileInferMetadataBuilder); got {type(meta).__name__}. Metadata from another "
                "backend means the layer and the attention backend were mixed up."
            )
        if kv_cache is None or int(kv_cache.numel()) == 0:
            IMPL_COUNTERS.bump("empty_kv_cache")
            raise RuntimeError(
                "TileInferAttentionImpl.forward got an empty kv_cache; the layer has no KV pool "
                "(pipeline-parallel placeholder layer?) and cannot attend"
            )
        if key is None or value is None:
            IMPL_COUNTERS.bump("missing_kv")
            raise RuntimeError(
                "TileInferAttentionImpl.forward got key/value = None; this backend writes the KV "
                "cache itself (forward_includes_kv_cache_update = True) and cannot skip a step"
            )
        if kv_cache.dim() != 5:
            IMPL_COUNTERS.bump("shape_mismatch")
            raise ValueError(
                "kv_cache must be the fused NHD pool produced by get_kv_cache_shape "
                "([num_blocks, 2, block_size, kv_heads, head_size]) or the legacy "
                f"[2, num_blocks, ...] form; got {tuple(kv_cache.shape)}"
            )
        try:
            k_half, v_half = fused_halves(kv_cache)
        except ValueError as exc:
            # Two blocks would make the two orders indistinguishable by shape alone, and picking
            # the wrong one swaps K and V - refuse loudly instead (fused_halves explains why).
            IMPL_COUNTERS.bump("shape_mismatch")
            raise ValueError(f"kv_cache is not a usable fused NHD pool: {exc}") from exc
        # Both layouts keep the engine page size at dim 2; only the K/V axis moves.
        block_size = int(kv_cache.shape[2])

        num_tokens = int(meta.num_actual_tokens)
        if num_tokens == 0:
            # A legal empty step (the scheduler can hand out zero tokens).  Nothing to write,
            # nothing to attend; return the (zero-row) output unchanged.
            IMPL_COUNTERS.bump("empty_step")
            return self._empty_output(query, output)

        # (a) Stage this step's K/V as [num_tokens, heads, dim] and write the pool *first*.
        q3 = self._as_3d(query, num_tokens, self.num_heads, "query")
        k3 = self._as_3d(key, num_tokens, self.num_kv_heads, "key")
        v3 = self._as_3d(value, num_tokens, self.num_kv_heads, "value")
        write_kv_cache(k3, v3, k_half, v_half, meta.slot_mapping)
        IMPL_COUNTERS.bump("kv_writes")

        # Decode vs prefill, from max_query_len only (never from attn_state / is_prefilling).
        if meta.is_prefill:
            IMPL_COUNTERS.bump("prefill_steps")
            return self._prefill_forward(q3, k_half, v_half, block_size, meta, output, num_tokens)

        if self.window_size is not None:
            # A windowed layer on a decode step: the paged decode kernel is full-causal only,
            # so the window is presented to it exactly — the last ``window`` tokens are staged
            # into a per-layer scratch pool and the kernel runs on that (see
            # ``attn.window``).  TILEINFER_WINDOW_DECODE=0 switches back to the masked
            # reference decode (the C4 A/B control).
            if _window_decode_enabled():
                IMPL_COUNTERS.bump("decode_window_staged")
                return self._windowed_kernel_forward(
                    q3, k_half, v_half, block_size, meta, output, num_tokens
                )
            IMPL_COUNTERS.bump("decode_window_reference")
            return self._reference_forward(q3, k_half, v_half, block_size, meta, output, num_tokens)

        IMPL_COUNTERS.bump("decode_steps")
        return self._kernel_forward(q3, k_half, v_half, block_size, meta, output, num_tokens)

    # ------------------------------------------------------------------ prefill path

    def _prefill_forward(
        self,
        q3: torch.Tensor,
        k_half: torch.Tensor,
        v_half: torch.Tensor,
        block_size: int,
        meta: TileInferMetadata,
        output: Optional[torch.Tensor],
        num_tokens: int,
    ) -> torch.Tensor:
        """Prefill/append on the paged-prefill kernel, with the reference backend as a *counted*
        fallback (``docs/performance.md`` §17 ③, M1 exit needs the ratio).

        The kernel path is eager, not plan-backed: a prefill step's tile count changes with every
        prompt length, so the schedule is rebuilt per step (host-only, no device sync) and the
        kernel variant is quantized on the tile/batch axes — see ``prefill.py``.

        A ``PrefillUnsupported`` is a *declaration* that this shape cannot use the kernel and is
        decided before the device is touched.  Any other exception propagates: a kernel that runs
        and returns wrong numbers is a bug, and silently degrading would hide it (C3).
        """
        path = self._prefill_path()
        if path is not None and self.window_size is None:
            try:
                result = path.run(q3, k_half, v_half, block_size, meta, output, num_tokens)
            except PrefillUnsupported as exc:
                IMPL_COUNTERS.bump("prefill_unsupported")
                IMPL_COUNTERS.bump(f"prefill_unsupported_{exc.reason}")
                if exc.reason not in _PREFILL_WARNED:
                    _PREFILL_WARNED.add(exc.reason)
                    logger.warning(
                        "TileInfer attention: prefill kernel cannot serve this step (%s); the "
                        "reference backend computes it and the fallback is counted",
                        exc.detail,
                    )
            else:
                IMPL_COUNTERS.bump("prefill_kernel")
                self._log_prefill_share()
                return result
        if self.window_size is not None:
            # Windowed prefill cannot use the paged kernel at all; counting it separately keeps
            # the kernel-share log about kernel-eligible steps (M1's ratio).
            IMPL_COUNTERS.bump("prefill_window_reference")
        else:
            IMPL_COUNTERS.bump("prefill_reference")
        self._log_prefill_share()
        return self._reference_forward(q3, k_half, v_half, block_size, meta, output, num_tokens)

    def _log_prefill_share(self) -> None:
        """Report the kernel share of prefill steps (M1's exit criterion is >0.9).

        WARNING, not INFO: vLLM's logging config filters this package's INFO records.
        """
        steps = IMPL_COUNTERS.get("prefill_steps")
        if not steps or steps % 20:
            return
        kernel = IMPL_COUNTERS.get("prefill_kernel")
        reference = IMPL_COUNTERS.get("prefill_reference")
        total = kernel + reference
        logger.warning(
            "TileInfer prefill: %d steps, kernel=%d reference=%d unsupported=%d "
            "window_reference=%d (kernel share %.3f of kernel-eligible steps)",
            steps,
            kernel,
            reference,
            IMPL_COUNTERS.get("prefill_unsupported"),
            IMPL_COUNTERS.get("prefill_window_reference"),
            (kernel / total) if total else 0.0,
        )

    def _prefill_path(self) -> Optional[PrefillPath]:
        """The prefill kernel path for this layer, or ``None`` if it is switched off/unavailable.

        ``TILEINFER_PREFILL=0`` is the C4 same-process control: it forces every prefill through
        the reference backend without touching the rest of the stack, which is what the A/B
        needs to attribute a TTFT delta to this one change.

        Deliberately branch- and lock-free: this is called from inside the traced ``forward``.
        """
        return self._prefill if self._prefill_enabled else None

    # ------------------------------------------------------------------ decode path

    def _windowed_kernel_forward(
        self,
        q3: torch.Tensor,
        k_half: torch.Tensor,
        v_half: torch.Tensor,
        block_size: int,
        meta: TileInferMetadata,
        output: Optional[torch.Tensor],
        num_tokens: int,
    ) -> torch.Tensor:
        """Windowed decode on the kernel path, via the exact scratch staging.

        The stage replaces the pool and the batch geometry: the kernel (and every fallback it
        can take, including the reference backend) then sees a request whose whole context *is*
        the window, so causal attention over it is the windowed answer.  ``block_size`` is the
        scratch page (128) from here on, not the engine page.
        """
        scratch_k, scratch_v, meta2 = self._stager.stage(k_half, v_half, block_size, meta)
        return self._kernel_forward(
            q3,
            scratch_k,
            scratch_v,
            int(KERNEL_PAGE_SIZE),
            meta2,
            output,
            num_tokens,
            states=self._window_states,
            windowed=True,
        )

    def _kernel_forward(
        self,
        q3: torch.Tensor,
        k_half: torch.Tensor,
        v_half: torch.Tensor,
        block_size: int,
        meta: TileInferMetadata,
        output: Optional[torch.Tensor],
        num_tokens: int,
        *,
        states: Optional[Dict[int, _BucketState]] = None,
        windowed: bool = False,
    ) -> torch.Tensor:
        """The fast path: plan lookup -> refresh_step -> BatchAttention.run -> output.

        ``states``/``windowed`` serve the windowed scratch staging: a separate per-bucket dict
        (the scratch pools must not collide with the real pool's states) and a plan-key salt
        (a scratch pool could otherwise share a key with a real pool of the same geometry).
        """
        attn = self._decode_attn(q3.dtype)
        if attn is None:
            IMPL_COUNTERS.bump("kernel_path_unavailable")
            return self._reference_forward(
                q3, k_half, v_half, block_size, meta, output, num_tokens
            )

        sub = sub_pages(block_size)  # raises for a page that is not k*128 tokens
        # (b) Zero-copy kernel views of the two contiguous halves of the fused pool.
        k_pool = pool_sub_view(k_half, sub)
        v_pool = pool_sub_view(v_half, sub)

        state = self._state_for(meta, k_pool, v_pool, block_size, q3, states=states)

        # (c1) Plan lookup.  A miss starts a *background* build and degrades this step: blocking
        # here would stall the engine for a ~70 s compile, possibly inside a captured graph.
        key = self._plan_key(state, windowed=windowed)
        plan = PLAN_CACHE.get(key)
        if plan is None:
            if PLAN_CACHE.is_failed(key):
                IMPL_COUNTERS.bump("plan_failed")
                return self._reference_forward(
                    q3, k_half, v_half, block_size, meta, output, num_tokens
                )
            plan = PLAN_CACHE.get_or_start(
                key, lambda key=key, state=state: self._build_plan(key, state)
            )
            if plan is None:
                IMPL_COUNTERS.bump("plan_not_ready")
                return self._reference_forward(
                    q3, k_half, v_half, block_size, meta, output, num_tokens
                )

        # (c2) Ragged rebuild, entirely on the device (no allocation, no sync, capturable).  Done
        # *after* the plan lookup so a degraded step does not pay for device work it will not use.
        # With sharing on, the first attention layer of a step builds it and the rest reuse it: the
        # triple describes the batch, not the layer.
        if _step_share_enabled():
            shared = _shared_step_for(state)
            if shared.meta_owner is not meta:
                refresh_step(
                    shared.step,
                    meta.block_table_tensor,
                    meta.seq_lens,
                    block_size,
                    shared.kv_indices,
                    num_reqs=meta.num_reqs,
                )
                shared.meta_owner = meta
                IMPL_COUNTERS.bump("step_share_refresh")
            else:
                IMPL_COUNTERS.bump("step_share_reuse")
            step_buffers, kv_indices = shared.step, shared.kv_indices
        else:
            refresh_step(
                state.step,
                meta.block_table_tensor,
                meta.seq_lens,
                block_size,
                state.kv_indices,
                num_reqs=meta.num_reqs,
            )
            step_buffers, kv_indices = state.step, state.kv_indices

        # (c3) Stage the query into the bucket-sized buffer.  Rows [num_tokens, bucket) keep their
        # zero initialisation: they belong to padding requests, whose outputs are never read.
        state.q_buf[:num_tokens].copy_(q3)

        # (d) Run and write the output.  `out_buf` is bucket-sized and owned by this layer, so the
        # call allocates nothing.
        attn.run(
            state.q_buf,
            state.k_pool,
            state.v_pool,
            out=state.out_buf,
            plan=plan,
            kv_indptr=step_buffers.kv_indptr_buf,
            kv_indices=kv_indices,
            kv_last_page_len=step_buffers.last_page_len_buf,
        )
        return self._store_output(output, state.out_buf, num_tokens)

    def _build_plan(self, key: Tuple[object, ...], state: _BucketState) -> object:
        """Build one decode plan for a bucket (slow: the kernel compiles here).

        The metadata handed to the planner is **synthetic and schedule-shaping only**: every
        bucket row gets exactly one 128-token page (``kv_indptr = arange(bucket + 1)``).  The tile
        schedule then contains one tile per row with no KV split, which stays *correct* for every
        later step of that bucket — every row always has at least one page (``refresh_step`` gives
        padding rows length 1).  The live page ids, context lengths and last-page lengths are
        never taken from this object: ``BatchAttention.run`` receives the step's ragged triple
        explicitly and copies it into the plan's own buffers
        (``tilelang_ascend950.py:31-55``).  Building from live values instead would bake one
        step's page counts into the schedule.
        """
        attn = self._decode_attn(state.dtype)
        if attn is None:  # the caller checked, but keep the invariant local
            raise RuntimeError("decode backend is unavailable")
        bucket = int(state.bucket)
        indptr = torch.arange(bucket + 1, dtype=torch.int32, device=state.device)
        last_page_len = torch.ones(bucket, dtype=torch.int32, device=state.device)
        plan = attn.plan(
            kv_indptr=indptr,
            kv_indices=state.kv_indices,
            kv_last_page_len=last_page_len,
            num_qo_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_size,
            page_size=KERNEL_PAGE_SIZE,
            sm_scale=self.scale,
            kv_layout="NHD",
            causal=True,
            block_q=KERNEL_PAGE_SIZE,
            finalize=True,
            k_cache=state.k_pool,
            v_cache=state.v_pool,
            cache_key=key,
        )
        IMPL_COUNTERS.bump("plan_builds")
        logger.info(
            "TileInfer attention: built decode plan %r (bucket=%d, width=%d, pool_cap=%d)",
            key,
            bucket,
            state.width,
            state.pool_cap,
        )
        return plan

    # ------------------------------------------------------------------ degraded path

    def _reference_forward(
        self,
        q3: torch.Tensor,
        k_half: torch.Tensor,
        v_half: torch.Tensor,
        block_size: int,
        meta: TileInferMetadata,
        output: Optional[torch.Tensor],
        num_tokens: int,
    ) -> torch.Tensor:
        """Correct-but-slow path: the same plan semantics, executed in plain torch.

        Used when the step is switched off from the kernel path (``TILEINFER_PREFILL=0``), when a
        prefill shape the kernel cannot serve shows up, and whenever a decode plan is not ready.
        It runs through ``BatchAttention(backend="reference")`` so the batch is described by the
        same :class:`RaggedMetadata` vocabulary as the fast path — there is no second page-table
        implementation to keep correct.

        The reference backend indexes the pool by *engine page id* and never flattens it, so it is
        layout-agnostic and works on either half as handed in (contiguous or block-strided).

        This path *does* read the page table on the host (``PageTable.to_ragged`` walks requests)
        and builds a plan per new shape.  That is acceptable precisely because it is not captured
        and not the steady state; both costs are visible in the counters above.
        """
        ref = _reference_attn()
        num_reqs = int(meta.num_reqs)
        if num_reqs < 1:
            IMPL_COUNTERS.bump("bad_query_len")
            raise ValueError("a batch with no requests cannot be attended")
        # The window the reference mask applies, spelled in its vocabulary: a layer whose
        # window is ``w`` tokens (counting the query itself) masks
        # ``k_pos < q_pos - window_left`` with ``window_left = w - 1``.  ``-1`` disables
        # windowing; the mask's absolute positions make chunked-prefill extends correct.
        window_left = -1 if self.window_size is None else self.window_size - 1

        qo_lens = meta.query_start_loc_cpu[1 : num_reqs + 1] - meta.query_start_loc_cpu[:num_reqs]
        if int(qo_lens.numel()) != num_reqs or int(qo_lens.min().item()) <= 0:
            IMPL_COUNTERS.bump("bad_query_len")
            raise ValueError(
                "query_start_loc_cpu must describe one positive query length per request; got "
                f"{qo_lens.tolist()} for num_reqs={num_reqs}"
            )
        if int(qo_lens.sum().item()) != num_tokens:
            # The reference path indexes the query tensor by qo_indptr; a mismatch would read past
            # the staged rows.  Cheap host check (query_start_loc_cpu is already on the host).
            IMPL_COUNTERS.bump("bad_query_len")
            raise ValueError(
                f"query lengths sum to {int(qo_lens.sum().item())} but the step schedules "
                f"{num_tokens} tokens; the batch metadata is inconsistent"
            )

        out3 = None if output is None else self._as_output(output, num_tokens)
        qo_host = [int(v) for v in qo_lens.tolist()]

        if set(qo_host) == {1}:
            # Degraded **decode** (plan not ready / failed): one token per request, so the tiled
            # reference's ``[batch, h_q, qo_len, d]`` and our ``[tokens, h_q, d]`` are the same
            # memory and one batched call is both correct and the cheapest option.
            plan = ref.plan_from_page_table(
                meta.block_table_tensor[:num_reqs],
                meta.seq_lens[:num_reqs],
                qo_lens=qo_lens,
                page_size=block_size,
                num_qo_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_size,
                sm_scale=self.scale,
                kv_layout="NHD",
                causal=True,
                window_left=window_left,
                finalize=True,
                k_cache=k_half,
                v_cache=v_half,
            )
            IMPL_COUNTERS.bump("reference_plans")
            q4 = q3.reshape(num_reqs, self.num_heads, 1, self.head_size)
            result = ref.run(q4, k_half, v_half, plan=plan)  # [batch, h_q, 1, d]
            compact = result.reshape(num_tokens, self.num_heads, self.head_size)
            if output is None:
                IMPL_COUNTERS.bump("output_allocated")
                return compact.reshape(num_tokens, self.num_heads * self.head_size).clone()
            out3.copy_(compact)
            return self._finish_output(output, out3)

        # Every multi-token reference prefill runs **one request at a time, with an explicit
        # transpose in both directions**.  Two silent bugs live in the "reshape it, it is the same
        # memory" shortcut, and both were measured on Ascend950PR:
        #
        # * axes.  ``reference_attention_tiled`` takes and returns ``[batch, h_q, qo_len, d]``
        #   (``testing/reference.py:190,272``) while the layer hands us ``[tokens, h_q, d]``.
        #   Reshaping one into the other reinterprets memory, so head and token swap places: head 1
        #   of token 0 came back holding token 1's head 0 -- attention over a *future* token, i.e.
        #   silently wrong, and invisible at token 0 of a single-head model;
        # * batching.  The reference takes the query block length from ``q.shape`` but keys its mask
        #   off ``qo_indptr``, so a padded ragged batch produced the right first token and a wrong
        #   continuation (sequential "分析\n\n<think>" vs batched "分析!!!!!!!").
        #
        # The degraded path is host-bound anyway, so one plan + one call per request costs nothing
        # that matters, and leaving a *fast wrong* branch in the correctness oracle would poison
        # every later A/B (C3).
        IMPL_COUNTERS.bump("reference_ragged_split")
        compact_parts = []
        offset = 0
        for req, qo_len in enumerate(qo_host):
            sub_plan = ref.plan_from_page_table(
                meta.block_table_tensor[req : req + 1],
                meta.seq_lens[req : req + 1],
                qo_lens=torch.tensor(
                    [qo_len], dtype=torch.int32, device=meta.query_start_loc.device
                ),
                page_size=block_size,
                num_qo_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_size,
                sm_scale=self.scale,
                kv_layout="NHD",
                causal=True,
                window_left=window_left,
                finalize=True,
                k_cache=k_half,
                v_cache=v_half,
            )
            IMPL_COUNTERS.bump("reference_plans")
            sub_q = q3[offset : offset + qo_len]
            offset += qo_len
            q4 = sub_q.transpose(0, 1).unsqueeze(0).contiguous()  # [1, h_q, qo_len, d]
            rb = ref.run(q4, k_half, v_half, plan=sub_plan)  # [1, h_q, qo_len, d]
            compact_parts.append(rb[0].transpose(0, 1))  # [qo_len, h_q, d]
        compact = compact_parts[0] if len(compact_parts) == 1 else torch.cat(compact_parts, dim=0)
        if output is None:
            IMPL_COUNTERS.bump("output_allocated")
            return compact.reshape(num_tokens, self.num_heads * self.head_size)
        out3.copy_(compact.reshape(out3.shape))
        return self._finish_output(output, out3)

    # ------------------------------------------------------------------ state

    def _state_for(
        self,
        meta: TileInferMetadata,
        k_pool: torch.Tensor,
        v_pool: torch.Tensor,
        block_size: int,
        q3: torch.Tensor,
        states: Optional[Dict[int, _BucketState]] = None,
    ) -> _BucketState:
        """Per-bucket scratch, allocated on first sight of the bucket.

        Allocation is legal on the engine's warm-up step and illegal inside a captured graph, so
        this is exactly the boundary the W8 pre-warm must cover: one eager step per reachable
        bucket.  ``state_builds`` is the counter W8 asserts is zero during capture.

        ``states`` is the dict the state is cached in — the windowed scratch staging passes its
        own (see ``_kernel_forward``); ``None`` means the real pool's ``self._states``.
        """
        states = self._states if states is None else states
        bucket = int(meta.bucket)
        sub = sub_pages(block_size)
        table_width = int(meta.block_table_tensor.shape[1])
        if table_width < 1:
            IMPL_COUNTERS.bump("shape_mismatch")
            raise ValueError("block_table_tensor must have at least one page column")
        width = table_width * sub  # kernel pages covered by the engine table
        state = states.get(bucket)
        if state is not None:
            if (
                state.width != width
                or state.block_size != block_size
                or int(state.k_pool.shape[0]) != int(k_pool.shape[0])
                or state.k_pool.data_ptr() != k_pool.data_ptr()
            ):
                IMPL_COUNTERS.bump("shape_mismatch")
                raise ValueError(
                    f"bucket {bucket} was prepared for width={state.width}, block_size="
                    f"{state.block_size}, {int(state.k_pool.shape[0])} kernel pages; this step "
                    f"needs width={width}, block_size={block_size}, "
                    f"{int(k_pool.shape[0])} pages. The KV pool or max_model_len changed "
                    "mid-process."
                )
            if state.dtype != q3.dtype:
                IMPL_COUNTERS.bump("shape_mismatch")
                raise ValueError(
                    f"bucket {bucket} was prepared for {state.dtype} queries but got {q3.dtype}; "
                    "a layer's compute dtype cannot change between steps"
                )
            return state

        device = k_pool.device
        pool_cap = declared_pool_cap(int(k_pool.shape[0]))
        # `refresh_step` needs somewhere to park the columns a request does not own: the trash
        # slot is the last element of `kv_indices` (serving/step.py:14-21, :92-99), so the buffer
        # must cover every possible destination *plus* that slot.  This depends only on
        # max_model_len / max_num_seqs / pool size, so it is a start-up-shaped failure.
        if bucket * width >= pool_cap - 1:
            IMPL_COUNTERS.bump("shape_mismatch")
            raise ValueError(
                f"the staging table ({bucket} rows x {width} kernel pages) does not fit the "
                f"declared KV capacity ({pool_cap} pages): the ragged compaction needs a trash "
                "slot past every destination. Increase the KV pool or reduce max_model_len / "
                "max_num_seqs."
            )
        step = allocate_step_buffers(
            bucket=bucket,
            width=width,
            kv_indices_capacity=pool_cap,
            block_size=block_size,
            device=device,
        )
        state = _BucketState(
            bucket=bucket,
            width=width,
            block_size=block_size,
            pool_cap=pool_cap,
            dtype=q3.dtype,
            device=device,
            k_pool=k_pool,
            v_pool=v_pool,
            step=step,
            kv_indices=torch.zeros(pool_cap, dtype=torch.int32, device=device),
            # Zeros, not empty: padding rows must not hold NaN/Inf, and the kernel does read their
            # query rows (their output is discarded).
            q_buf=torch.zeros(
                (bucket, self.num_heads, self.head_size), dtype=q3.dtype, device=device
            ),
            out_buf=torch.empty(
                (bucket, self.num_heads, self.head_size), dtype=q3.dtype, device=device
            ),
        )
        states[bucket] = state
        IMPL_COUNTERS.bump("state_builds")
        logger.info(
            "TileInfer attention: prepared bucket %d (width=%d kernel pages, block_size=%d, "
            "pool_cap=%d, dtype=%s)",
            bucket,
            width,
            block_size,
            pool_cap,
            q3.dtype,
        )
        return state

    def prewarm(self, buckets: Optional[Sequence[int]] = None) -> int:
        """Build every plan a prepared bucket needs, synchronously; returns how many were built.

        Called from the engine's warm-up (W5/W8), *before* graph capture: the captured region may
        only look plans up.  A bucket that fails to compile is remembered by the plan cache and
        degrades to the reference backend for its lifetime (``plan_failed``); it does not abort
        the rest of the pre-warm.

        Only buckets that already have a :class:`_BucketState` can be pre-warmed — the plan needs
        the pool geometry, which is learned from the first eager forward of that bucket.  That is
        the same warm-up step the engine runs before capture; there is no way to pre-warm a pool
        that does not exist yet.  The windowed scratch states pre-warm the same way, off their
        own dict.
        """
        wanted = tuple(int(b) for b in (buckets if buckets is not None else sorted(self._states)))
        built = 0
        for bucket in wanted:
            for state, windowed in (
                (self._states.get(bucket), False),
                (self._window_states.get(bucket), True),
            ):
                if state is None:
                    if windowed:
                        continue  # windowed scratch is built lazily like the real one
                    logger.warning(
                        "TileInfer attention: prewarm asked for bucket %d before any step built "
                        "its buffers; run one eager step for that bucket first",
                        bucket,
                    )
                    continue
                key = self._plan_key(state, windowed=windowed)
                if PLAN_CACHE.get(key) is not None or PLAN_CACHE.is_failed(key):
                    continue
                try:
                    PLAN_CACHE.build(
                        key, lambda key=key, state=state: self._build_plan(key, state)
                    )
                except Exception:  # noqa: BLE001 - one bad bucket must not kill the pre-warm
                    logger.exception(
                        "TileInfer attention: pre-warm failed for bucket %d (%s)",
                        bucket,
                        "windowed" if windowed else "full",
                    )
                    continue
                built += 1
        return built

    # ------------------------------------------------------------------ helpers

    def prepare_windowed_buckets(
        self, buckets: Sequence[int], dtype: torch.dtype, device: torch.device
    ) -> int:
        """Pre-warm the windowed scratch states and their plans (the worker's pre-warm calls
        this for windowed layers instead of the real-pool path).

        The scratch pool, the synthetic table and the bucket state are built here — the same
        objects the first staged step would build lazily — so the first real windowed decode
        finds a ready plan instead of degrading for one step.  Returns how many plans were
        built; a bucket that fails to compile is remembered by the plan cache and degrades
        exactly like the real path's ``plan_failed``.
        """
        if self._stager is None:
            return 0
        from tileinfer.serving.buckets import pool_sub_view as _psv

        built = 0
        q3_shape = (0, self.num_heads, self.head_size)
        for bucket in buckets:
            bucket = int(bucket)
            scratch = self._stager._state_for(bucket, dtype, device)
            meta = TileInferMetadata(
                query_start_loc=torch.arange(bucket + 1, dtype=torch.int32, device=device),
                query_start_loc_cpu=torch.arange(bucket + 1, dtype=torch.int32),
                seq_lens=torch.ones(bucket, dtype=torch.int32, device=device),
                block_table_tensor=scratch.table,
                slot_mapping=torch.arange(bucket, dtype=torch.int64, device=device),
                num_reqs=bucket,
                num_actual_tokens=bucket,
                max_query_len=1,
                max_seq_len=1,
                bucket=bucket,
                block_size=int(KERNEL_PAGE_SIZE),
            )
            q3 = torch.empty(q3_shape, dtype=dtype, device=device)
            try:
                state = self._state_for(
                    meta,
                    _psv(scratch.scratch_k, 1),
                    _psv(scratch.scratch_v, 1),
                    int(KERNEL_PAGE_SIZE),
                    q3,
                    states=self._window_states,
                )
            except Exception:  # noqa: BLE001 - one bucket must not stop the others
                logger.exception(
                    "TileInfer attention: cannot prepare windowed bucket %d", bucket
                )
                continue
            key = self._plan_key(state, windowed=True)
            if PLAN_CACHE.get(key) is not None or PLAN_CACHE.is_failed(key):
                continue
            try:
                PLAN_CACHE.build(key, lambda key=key, state=state: self._build_plan(key, state))
            except Exception:  # noqa: BLE001 - one bad bucket must not kill the pre-warm
                logger.exception(
                    "TileInfer attention: windowed pre-warm failed for bucket %d", bucket
                )
                continue
            built += 1
        return built

    def _plan_key(self, state: _BucketState, *, windowed: bool = False) -> Tuple[object, ...]:
        """The plan-cache key for one bucket (see ``serving.planner.plan_key``).

        ``windowed`` salts the key: a scratch pool could otherwise share a key with a real pool
        of the same (bucket, width, block_size) geometry, and the two plans compile against
        different caches (``_build_plan`` finalizes on ``state.k_pool``).
        """
        base = (
            self._backend_name,
            "windowed-decode" if windowed else "decode",
            self.num_heads,
            self.num_kv_heads,
            self.head_size,
            str(state.dtype),
            str(state.device),
        )
        return plan_key(
            base,
            self._split_kv_pages,
            state.bucket,
            state.width,
            state.block_size,
            state.pool_cap,
        )

    def _decode_attn(self, dtype: Optional[torch.dtype] = None) -> Optional[BatchAttention]:
        if self._no_kernel_backend:
            return None
        # The pool dtype is the contract (see _decode_attn_or_none's docstring); a caller that
        # already has the step's tensors passes `state.dtype`, which is what the pre-warm needs
        # because `key_cache` is not bound yet at that point.
        wanted = dtype if dtype is not None else self._pool_dtype()
        attn = _decode_attn_or_none(wanted)
        if attn is None:
            self._no_kernel_backend = True
            return None
        return attn

    def _pool_dtype(self) -> Optional[torch.dtype]:
        """The KV pool's dtype, which is also the query dtype vLLM hands us."""
        cache = getattr(self, "key_cache", None)
        if cache is not None and hasattr(cache, "dtype"):
            return cache.dtype
        return getattr(self, "dtype", None)

    def _as_3d(self, tensor: torch.Tensor, num_tokens: int, heads: int, name: str) -> torch.Tensor:
        """``[tokens, heads, dim]`` or flattened ``[tokens, heads * dim]`` -> 3-D, copy-free when
        possible.

        vLLM 0.23's ``AttentionImpl.forward`` types these as bare ``torch.Tensor``
        (``v1/attention/backend.py:800-803``), so the layout is not part of the transcribed
        contract; both spellings are accepted rather than guessed at, and a reshape that had to
        copy is counted.
        """
        if tensor.dim() == 3:
            out = tensor
        elif tensor.dim() == 2:
            if int(tensor.shape[1]) != heads * self.head_size:
                IMPL_COUNTERS.bump("shape_mismatch")
                raise ValueError(
                    f"{name} has trailing dim {tensor.shape[1]} but expected "
                    f"{heads} x {self.head_size} = {heads * self.head_size}"
                )
            out = tensor.reshape(tensor.shape[0], heads, self.head_size)
            if out.data_ptr() != tensor.data_ptr():
                IMPL_COUNTERS.bump("input_reshape_copy")
        else:
            IMPL_COUNTERS.bump("shape_mismatch")
            raise ValueError(f"{name} must be 2-D or 3-D, got shape {tuple(tensor.shape)}")

        if out.shape[1] != heads or out.shape[2] != self.head_size:
            IMPL_COUNTERS.bump("shape_mismatch")
            raise ValueError(
                f"{name} must be [tokens, {heads}, {self.head_size}], got {tuple(out.shape)}"
            )
        if int(out.shape[0]) < num_tokens:
            IMPL_COUNTERS.bump("shape_mismatch")
            raise ValueError(
                f"{name} has {out.shape[0]} rows but the step schedules {num_tokens} tokens"
            )
        if int(out.shape[0]) > num_tokens:
            # A captured/padded input buffer: the extra rows have no slot mapping.  Explicit, and
            # counted, rather than silently attending over stale rows.
            IMPL_COUNTERS.bump("padded_input_rows")
            out = out[:num_tokens]
        return out

    def _as_output(self, output: torch.Tensor, num_tokens: int) -> torch.Tensor:
        """``[tokens, heads*dim]`` -> ``[tokens, heads, dim]``; a copy is counted, not hidden."""
        expected = num_tokens * self.num_heads * self.head_size
        if int(output.numel()) != expected:
            IMPL_COUNTERS.bump("shape_mismatch")
            raise ValueError(
                f"output has {output.numel()} elements but the step needs {expected} "
                f"({num_tokens} tokens x {self.num_heads} heads x {self.head_size})"
            )
        out3 = output.reshape(num_tokens, self.num_heads, self.head_size)
        if out3.data_ptr() != output.data_ptr():
            IMPL_COUNTERS.bump("output_copyback")
        return out3

    def _finish_output(self, output: torch.Tensor, out3: torch.Tensor) -> torch.Tensor:
        """Write back only if the 3-D view had to copy (non-contiguous output buffer)."""
        if out3.data_ptr() != output.data_ptr():
            output.copy_(out3.reshape_as(output))
        return output

    def _store_output(
        self, output: Optional[torch.Tensor], src3: torch.Tensor, num_tokens: int
    ) -> torch.Tensor:
        """Copy the first ``num_tokens`` rows of a bucket-sized result into vLLM's output."""
        if output is None:
            IMPL_COUNTERS.bump("output_allocated")
            return src3[:num_tokens].reshape(
                num_tokens, self.num_heads * self.head_size
            ).clone()
        out3 = self._as_output(output, num_tokens)
        out3.copy_(src3[:num_tokens])
        return self._finish_output(output, out3)

    def _empty_output(self, query: torch.Tensor, output: Optional[torch.Tensor]) -> torch.Tensor:
        if output is not None:
            return output
        return query.new_empty((0, self.num_heads * self.head_size))


def _split_kv_pages() -> int:
    """KV pages per split tile, or ``0`` for "one tile per request".

    Split-KV is opt-in upstream (``tilelang_ascend950.py:144-155`` raises unless
    ``TILEINFER_ALLOW_SPLIT_KV=1``), and it is part of the plan key because a split plan's tile
    count depends on the *page counts* of the batch it was built from — the one place where a
    per-bucket cached plan is not shape-stable.
    """
    raw = os.environ.get("TILEINFER_SPLIT_KV_PAGES", "").strip()
    if not raw:
        return 0
    value = int(raw)
    if value < 0:
        raise ValueError(f"TILEINFER_SPLIT_KV_PAGES must be >= 0, got {value}")
    return value


# --------------------------------------------------------------------------------------------
# Phase timing (docs/performance.md M0)
#
# Wrapped at module import rather than inline in ``forward``: the method has many early returns,
# and a try/finally wrapper keeps the measurement out of the logic and completely inert when
# TILEINFER_PHASE_TIMING is unset (the flag is read once, at import).
# --------------------------------------------------------------------------------------------
if _PHASES.enabled():  # pragma: no branch - resolved once at import
    _ORIGINAL_FORWARD = TileInferAttentionImpl.forward
    _TIMING_LOG_EVERY = 20

    def _timed_forward(
        self, layer, query, key, value, kv_cache, attn_metadata, output, *args, **kwargs
    ):
        try:
            with _PHASES.timed("attention"):
                return _ORIGINAL_FORWARD(
                    self, layer, query, key, value, kv_cache, attn_metadata, output, *args, **kwargs
                )
        finally:
            _max_q = getattr(attn_metadata, "max_query_len", None)
            if _max_q is not None:
                _kind = "decode" if int(_max_q) == 1 else "prefill"
                _PHASES.count_step(_kind)
                _seen = _PHASES.steps_counted(_kind)
                if _seen and _seen % _TIMING_LOG_EVERY == 0:
                    # WARNING, not INFO: vLLM's logging config filters our INFO records (observed
                    # on the device - the pre-warm's INFO messages never reached the log either).
                    # Both kinds log: the prefill breakdown is what says whether TTFT is attention
                    # at all, and a decode-only trigger left that question unanswered (M1, §18).
                    logger.warning("\n%s", _PHASES.format_table())

    TileInferAttentionImpl.forward = _timed_forward
