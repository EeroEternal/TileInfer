"""vLLM-Ascend integration: TileInfer as the decode attention backend.

Scope, deliberately narrow
--------------------------
vLLM-Ascend already has a working attention path (CANN's fused-infer operator, FIA).  TileInfer is
interesting where FIA is not the answer: **decode over a paged cache, dynamically batched**.  So this
integration overrides exactly one method — ``forward_paged_attention``, the decode path, whose parent
implementation calls ``torch_npu._npu_paged_attention`` — and lets everything else (prefill, chunked
prefill, speculative decoding, MLA/SFA models) fall through to vLLM-Ascend's own code.  A model
therefore keeps working even if TileInfer cannot handle a batch, and the comparison against FIA is
apples to apples because FIA is literally the parent call.

How it plugs in
---------------
vLLM 0.23 lets a third party take over a backend slot::

    from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
    register_backend(AttentionBackendEnum.CUSTOM, "my.module.MyBackend")

``CUSTOM`` is the same slot ``vllm_ascend/attention/fa3_v1.py`` uses (FA3 is only enabled in
batch-invariant mode), so registering last wins.  :func:`install` does that, and
``scripts/run_vllm_tileinfer.sh`` shows how to make vLLM load it before the backend is chosen.

Three serving realities this file has to deal with
--------------------------------------------------
1. **The batch size changes every step.**  A TileLang kernel is compiled per shape (~1-2 minutes), so
   compiling per step is impossible.  Requests are therefore padded up to a **bucket** and the kernel
   is compiled once per bucket; ``metadata.pad_batch`` adds padding requests that are *valid* (one
   page, one token) so the softmax cannot divide by zero, and their outputs are simply not read.
2. **The page table's contents move every step** (a new token lands in a new slot, a new page gets
   allocated).  The plan is therefore built once per (bucket, table shape) and the ragged triple is
   recomputed **on the device** each step by ``ragged_from_page_table_device`` and handed to
   ``run(kv_indptr=..., kv_indices=..., kv_last_page_len=...)``; the plan owns the buffers the kernel
   actually reads, so nothing is allocated inside the step.
3. **Prefix caching means a decode step is not always a decode step.**  ``attn_state`` tells us; the
   decode path is only taken for ``DecodeOnly``, which is exactly the case the kernel implements.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

logger = logging.getLogger("tileinfer.vllm_ascend")

#: Batch sizes the kernel is compiled for.  Serving batch sizes cluster well below these ceilings,
#: so the padding waste is small compared to recompiling.
BATCH_BUCKETS: Tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256)


def _bucket_for(num_reqs: int) -> int:
    for bucket in BATCH_BUCKETS:
        if bucket >= num_reqs:
            return bucket
    raise ValueError(
        f"batch of {num_reqs} exceeds the largest TileInfer bucket ({BATCH_BUCKETS[-1]}); "
        "add a larger bucket to BATCH_BUCKETS"
    )


class TileInferDecodeAttention:
    """Decode attention for one layer: bucketed plans plus the per-step metadata refresh."""

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
        scale: float,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        kv_tile_pages: int = 0,
    ) -> None:
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.scale = scale
        self.device = device
        self.dtype = dtype
        self.kv_tile_pages = kv_tile_pages
        self._attn = None
        self._plans: Dict[Tuple[int, int, int], Any] = {}
        self._q_pad: Dict[int, torch.Tensor] = {}
        self._warned = False

    # ------------------------------------------------------------------ availability

    @property
    def group_size(self) -> int:
        return max(1, self.num_heads // max(1, self.num_kv_heads))

    def supported(self, block_size: int) -> bool:
        """Whether the kernel can handle this configuration at all."""
        reasons = []
        if self.head_size != 128:
            reasons.append(f"head_size={self.head_size} (the Ascend 950 path is tuned for 128)")
        if block_size < 16:
            reasons.append(f"block_size={block_size} (the paged kernel needs >= 16)")
        if self.group_size < 2:
            reasons.append(f"group={self.group_size} (MHA needs the reference path)")
        if reasons:
            if not self._warned:
                logger.warning("TileInfer decode disabled: %s", "; ".join(reasons))
                self._warned = True
            return False
        return True

    # ------------------------------------------------------------------ plan cache

    def _plan(self, bucket: int, max_pages: int, block_size: int, key_cache: torch.Tensor):
        key = (bucket, max_pages, block_size)
        plan = self._plans.get(key)
        if plan is not None:
            return plan

        from tileinfer import BatchAttention
        from tileinfer.metadata import RaggedMetadata

        if self._attn is None:
            self._attn = BatchAttention(backend="tilelang-ascend950", device=self.device, dtype=self.dtype)

        # A *dummy* page table: only its shape matters.  Decode without KV splitting produces one
        # work tile per request, so the schedule does not depend on the lengths; the values are
        # refreshed every step by the caller.
        dummy_lens = torch.ones(bucket, dtype=torch.int32, device=self.device)
        dummy = RaggedMetadata(
            kv_indptr=torch.arange(0, bucket + 1, dtype=torch.int32, device=self.device),
            kv_indices=torch.zeros(bucket, dtype=torch.int32, device=self.device),
            kv_last_page_len=torch.ones(bucket, dtype=torch.int32, device=self.device),
            page_size=block_size,
        )
        plan = self._attn.plan_from_page_table(
            torch.zeros((bucket, max_pages), dtype=torch.int32, device=self.device),
            dummy_lens,
            qo_lens=torch.ones(bucket, dtype=torch.int32, device=self.device),
            page_size=block_size,
            num_qo_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_size,
            sm_scale=self.scale,
            causal=False,
            kv_tile_pages=self.kv_tile_pages,
            k_cache=key_cache,
            v_cache=torch.empty_like(key_cache),
        )
        self._plans[key] = plan
        self._q_pad[bucket] = torch.zeros(
            (bucket, self.num_heads, 1, self.head_size), dtype=self.dtype, device=self.device
        )
        logger.info(
            "TileInfer plan ready: bucket=%d max_pages=%d block_size=%d -> %s",
            bucket,
            max_pages,
            block_size,
            plan.summary(),
        )
        return plan

    # ------------------------------------------------------------------ the step

    def run(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        block_size: int,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """One decode step: ``query`` is ``[num_reqs, num_heads * head_size]``, writes into ``output``."""
        from tileinfer.metadata import pad_batch, ragged_from_page_table_device

        num_reqs = int(seq_lens.numel())
        bucket = _bucket_for(num_reqs)
        plan = self._plan(bucket, int(block_tables.shape[1]), block_size, key_cache)

        meta = ragged_from_page_table_device(
            block_tables[:num_reqs], seq_lens[:num_reqs], block_size, device=self.device
        )
        if bucket != num_reqs:
            meta = pad_batch(meta, bucket, block_size)

        q = self._q_pad[bucket]
        q[:num_reqs] = query[:num_reqs].view(num_reqs, self.num_heads, 1, self.head_size).to(self.dtype)

        out = self._attn.run(
            q,
            key_cache,
            value_cache,
            plan=plan,
            kv_indptr=meta.kv_indptr,
            kv_indices=meta.kv_indices,
            kv_last_page_len=meta.kv_last_page_len,
        )
        output[:num_reqs] = out[:num_reqs].reshape(num_reqs, self.num_heads * self.head_size).to(
            output.dtype
        )
        return output


# ---------------------------------------------------------------------------------------
# vLLM-Ascend glue.  Imported only inside a vLLM environment (see install()).
# ---------------------------------------------------------------------------------------


def _build_classes():
    """Create the backend/impl pair against the *installed* vLLM-Ascend.

    Called at import time (see the bottom of the file) because vLLM's registry stores backend
    *paths* - it imports the class by dotted name - so the classes have to be module globals.
    """
    from vllm.v1.attention.backend import AttentionBackend

    # Import-order workaround: `vllm_ascend.attention.attention_v1` pulls in `vllm_ascend.ops`, whose
    # `fused_moe` imports `vllm_ascend.device.device_op` *while that module is still initialising* -
    # a circular import that raises `cannot import name 'DeviceOperator'`.  Importing the ops package
    # first initialises it in the order vLLM-Ascend expects.  (Verified on vllm_ascend 0.23: only this
    # order works, `attention_v1` or `platform` first both fail.)
    import vllm_ascend.ops  # noqa: F401

    from vllm_ascend.attention.attention_v1 import (
        AscendAttentionBackendImpl,
        AscendAttentionMetadataBuilder,
        AscendAttentionState,
    )

    class TileInferBackend(AttentionBackend):
        """The ``CUSTOM`` slot, served by TileInfer for decode.

        It reuses vLLM-Ascend's metadata builder on purpose: ``block_tables`` / ``seq_lens`` /
        ``query_start_loc`` *is* the metadata TileInfer's planner consumes, so no conversion layer
        inside vLLM is needed.
        """

        @staticmethod
        def get_name() -> str:
            return "CUSTOM"

        @staticmethod
        def get_impl_cls() -> type:
            return TileInferImpl

        @staticmethod
        def get_builder_cls() -> type:
            return AscendAttentionMetadataBuilder

        @staticmethod
        def get_kv_cache_shape(
            num_blocks: int,
            block_size: int,
            num_kv_heads: int,
            head_size: int,
            cache_type: str = "",
        ) -> tuple:
            # NHD, identical to the FIA path so a model can switch backends without re-allocating
            return (2, num_blocks, block_size, num_kv_heads, head_size)

        @staticmethod
        def get_supported_kernel_block_sizes() -> list:
            return [128]

    class TileInferImpl(AscendAttentionBackendImpl):
        """Decode through TileInfer; every other path stays with vLLM-Ascend (FIA)."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._tileinfer: Optional[TileInferDecodeAttention] = None
            self._tileinfer_failed = False

        # -- helpers ---------------------------------------------------------------

        def _decode(self, block_size: int) -> Optional[TileInferDecodeAttention]:
            if self._tileinfer is None:
                self._tileinfer = TileInferDecodeAttention(
                    num_heads=self.num_heads,
                    num_kv_heads=self.num_kv_heads,
                    head_size=self.head_size,
                    scale=self.scale,
                    device=self.key_cache.device,
                )
            return self._tileinfer if self._tileinfer.supported(block_size) else None

        # -- the one overridden path ----------------------------------------------

        def forward_impl(self, query, key, value, kv_cache, attn_metadata, output):
            """The single dispatch point of vLLM-Ascend's attention.

            ``forward`` has already written the step's K/V into the cache by the time this runs, so
            for a decode-only batch TileInfer can read the same cache the FIA path would.

            Overriding ``forward_impl`` rather than ``forward_paged_attention`` matters: the parent
            only routes to that method when ``using_paged_attention(num_tokens, vllm_config,
            head_size)`` agrees, and when it does not the batch silently goes to FIA - which is what
            happened on the first attempt at this integration (a 0.47 s response with no TileInfer
            trace in the log).  Here the decision is explicit and, if the batch is one we cannot
            handle, the fallback is still the parent implementation.
            """
            if self._tileinfer_failed or self.key_cache is None:
                return super().forward_impl(query, key, value, kv_cache, attn_metadata, output)

            if attn_metadata.attn_state != AscendAttentionState.DecodeOnly:
                return super().forward_impl(query, key, value, kv_cache, attn_metadata, output)

            block_size = int(self.key_cache.shape[1])
            impl = self._decode(block_size)
            if impl is None:
                return super().forward_impl(query, key, value, kv_cache, attn_metadata, output)

            try:
                num_reqs = int(attn_metadata.seq_lens.numel())
                return impl.run(
                    query=query[:num_reqs],
                    key_cache=self.key_cache,
                    value_cache=self.value_cache,
                    block_tables=attn_metadata.block_tables[:num_reqs],
                    seq_lens=attn_metadata.seq_lens[:num_reqs],
                    block_size=block_size,
                    output=output,
                )
            except Exception:  # noqa: BLE001 - a serving process must survive
                self._tileinfer_failed = True
                logger.exception(
                    "TileInfer decode failed; falling back to the Ascend FIA path for this process"
                )
                return super().forward_impl(query, key, value, kv_cache, attn_metadata, output)

    return TileInferBackend, TileInferImpl


# Build at import time when vLLM is around; the module still imports cleanly without it (the CPU
# test suite relies on that).
TileInferBackend: Any = None
TileInferImpl: Any = None
try:  # pragma: no cover - needs vLLM + vllm_ascend
    if not __import__("os").environ.get("TILEINFER_SKIP_CLASS_BUILD"):
        TileInferBackend, TileInferImpl = _build_classes()
except Exception as _exc:  # pragma: no cover
    logger.debug("vLLM-Ascend classes not built here: %s", _exc)


def _install_backend_selection_shim() -> None:
    """Make ``vllm_ascend`` actually *use* the backend this plugin registers.

    Registering in vLLM's registry is necessary but not sufficient on this platform:
    ``vllm_ascend.platform.NPUPlatform.get_attn_backend_cls`` ignores the user's
    ``--attention-backend`` for everything except FLASH_ATTN (it logs "Ascend NPU will use its
    registered plugin backend instead. Resetting to None") and returns one of its own class paths
    from a fixed map.  So the selection is wrapped here: plain MHA/GQA goes to TileInfer, MLA /
    sparse / compressed keep the Ascend default.

    This is a shim over an upstream decision, not a capability of the plugin API, and it is isolated
    in this function so it can be deleted the day vLLM-Ascend exposes a hook.
    """
    from vllm_ascend.platform import NPUPlatform

    if getattr(NPUPlatform, "_tileinfer_shim", False):
        return
    original = NPUPlatform.get_attn_backend_cls.__func__
    path = f"{__name__}.TileInferBackend"

    @classmethod
    def get_attn_backend_cls(cls, selected_backend, attn_selector_config, num_heads=None):
        native = (
            getattr(attn_selector_config, "use_mla", False)
            or getattr(attn_selector_config, "use_sparse", False)
            or getattr(attn_selector_config, "use_compress", False)
        )
        if not native:
            logger.info("TileInfer: taking the attention backend slot for plain MHA/GQA")
            return path
        return original(cls, selected_backend, attn_selector_config, num_heads)

    NPUPlatform.get_attn_backend_cls = get_attn_backend_cls
    NPUPlatform._tileinfer_shim = True
    logger.info("TileInfer: vLLM-Ascend backend selection shim installed")


def _build_classes_into_module() -> None:
    """(Re)build the vLLM-Ascend classes into this module's globals."""
    global TileInferBackend, TileInferImpl
    TileInferBackend, TileInferImpl = _build_classes()


def install(raise_on_failure: bool = True) -> bool:
    """Register TileInfer in vLLM's ``CUSTOM`` attention slot.

    Call this *after* ``vllm_ascend`` has been imported and *before* vLLM selects a backend (the
    runner script does it before touching the CLI).  Returns ``True`` when the registration took.
    """
    if not __import__("os").environ.get("TILEINFER_VLLM"):
        logger.warning(
            "TileInfer's vLLM integration is opt-in: set TILEINFER_VLLM=1 to enable it.  Decode "
            "through the kernel currently faults the device inside an EngineCore (see "
            "docs/known-issues.md KI-1), so it stays off unless asked for explicitly."
        )
        return False

    try:
        from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

        if TileInferBackend is None:
            _build_classes_into_module()

        path = f"{__name__}.TileInferBackend"
        # vLLM's registry stores a *dotted path* and resolves it with importlib + rsplit; handing it
        # a class object registers nothing and vLLM then falls back to the default backend without
        # complaining.  Hence the path, and hence the check below.
        register_backend(AttentionBackendEnum.CUSTOM, path)

        _install_backend_selection_shim()
        resolved = AttentionBackendEnum.CUSTOM.get_class()
        if resolved is not TileInferBackend:
            raise RuntimeError(
                f"the CUSTOM slot resolves to {resolved!r}, not TileInferBackend - the registration "
                "did not take effect (another plugin registered after us?)"
            )
        logger.info(
            "TileInfer registered as the CUSTOM attention backend (decode accelerated, "
            "prefill falls back to vLLM-Ascend)"
        )
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not register the TileInfer attention backend: %s", exc)
        if raise_on_failure:
            raise
        return False
