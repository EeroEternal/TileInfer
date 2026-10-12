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
import threading
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

logger = logging.getLogger("tileinfer.vllm_ascend")

#: Batch sizes the kernel is compiled for.  Serving batch sizes cluster well below these ceilings,
#: so the padding waste is small compared to recompiling.
BATCH_BUCKETS: Tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256)

#: The TileLang kernels are compiled for 128-token pages: the page is the kernel's N tile (``BC``),
#: so a page of 2048 tokens cannot fit the 64 KB L0 buffers.  Hybrid models (Qwen3.5/3.6 with GDN
#: layers) force a much larger engine block size — vLLM-Ascend aligns the attention page with the
#: mamba state page (e.g. 2048 tokens).  A ``block_size = k * 128`` page is *k consecutive
#: 128-token sub-pages* in memory, so the pool is presented as a zero-copy view with 128-token
#: pages and the engine's page table is expanded to sub-page ids (``page * k + i``).  Kernel shapes
#: then stay within capacity and no KV copy happens; vLLM's block-size validation accepts this
#: because the framework block size only needs to be a *multiple* of the kernel's requirement.
KERNEL_PAGE_SIZE = 128


def _sub_pages(block_size: int) -> int:
    """How many 128-token sub-pages one engine page holds (>= 1; 1 = no transformation)."""
    if block_size % KERNEL_PAGE_SIZE != 0:
        raise ValueError(
            f"block_size={block_size} is not a multiple of {KERNEL_PAGE_SIZE}; "
            "the TileInfer kernel cannot tile it"
        )
    return max(1, block_size // KERNEL_PAGE_SIZE)


#: Operator-granularity switch (BACKLOG B6): ``TILEINFER_OPS=decode,prefill,gdn,moe`` enables
#: individual takeovers; unset or empty means *all on*.  One flip + server restart (a restart is
#: seconds, not the recompile a config change would be) takes a suspicious op out of the chain
#: without touching the others — the fallback for every disabled op is the stock FIA path.
_OPS_ALL = {"decode", "prefill", "gdn", "moe"}


def ops_enabled(op: str) -> bool:
    """Whether the given operator takeover is switched on (``TILEINFER_OPS``, default: all)."""
    import os

    spec = os.environ.get("TILEINFER_OPS")
    if spec is None or not spec.strip():
        return True
    unknown = {s.strip().lower() for s in spec.split(",") if s.strip()} - _OPS_ALL
    if unknown:
        raise ValueError(
            f"TILEINFER_OPS contains unknown ops {sorted(unknown)}; valid: {sorted(_OPS_ALL)}"
        )
    return op.lower() in {s.strip().lower() for s in spec.split(",") if s.strip()}


def _expand_page_table(block_tables: torch.Tensor, sub: int) -> torch.Tensor:
    """Expand a dense table of ``block_size``-token pages into 128-token sub-page ids.

    Column ``j * sub + i`` of the result holds ``table[j] * sub + i``: sub-page ``i`` of engine
    page ``table[j]``.  Unused columns expand to sub-pages of whatever stale id they hold — they
    are masked out by ``ragged_from_page_table_device`` (``cols < pages_per_seq``), so their
    contents never matter.
    """
    if sub == 1:
        return block_tables
    scaled = block_tables.to(torch.int32) * sub
    expanded = scaled.repeat_interleave(sub, dim=1)
    offsets = torch.arange(sub, dtype=torch.int32, device=block_tables.device)
    return expanded + offsets.repeat(scaled.shape[1])


def _pool_sub_view(pool: torch.Tensor, sub: int) -> torch.Tensor:
    """View a pool of ``block_size``-token NHD pages as ``sub``-times more 128-token pages."""
    if sub == 1:
        return pool
    nb, block, heads, dim = pool.shape
    return pool.reshape(nb * (block // KERNEL_PAGE_SIZE), KERNEL_PAGE_SIZE, heads, dim)


def _bucket_for(num_reqs: int) -> int:
    for bucket in BATCH_BUCKETS:
        if bucket >= num_reqs:
            return bucket
    raise ValueError(
        f"batch of {num_reqs} exceeds the largest TileInfer bucket ({BATCH_BUCKETS[-1]}); "
        "add a larger bucket to BATCH_BUCKETS"
    )


class TileInferDecodeAttention:
    """Decode attention for one layer: bucketed plans plus the per-step metadata refresh.

    Serving-safety contract (the reason this class looks the way it does):

    * Plans are **prebuilt before graph capture**: the first non-captured decode step (vLLM's
      warmup run) synchronously compiles every bucket up to ``max_num_reqs``, so by the time
      ACLGraph capture starts each ``run()`` finds its plan in the cache.  A plan built *during*
      capture would allocate/sync on the captured stream — ACL rejects that and the process used
      to silently fall back to FIA (BACKLOG B5).
    * ``run()`` never blocks and never allocates: the page table, the query and the output live in
      plan-owned buffers that are refreshed with pure device-side copies.
    * Under an active capture ``run()`` consults the plan cache only; a missing plan falls back to
      FIA by returning ``None`` — it must not raise and must not start a compile.
    """

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
        scale: float,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        kv_tile_pages: int = 0,
        max_num_reqs: int = 0,
    ) -> None:
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.scale = scale
        self.device = device
        self.dtype = dtype
        self.kv_tile_pages = kv_tile_pages
        self.max_num_reqs = max_num_reqs or 0
        self._warned = False

        # The plan cache, its in-flight/failed sets and the serving counters belong to the
        # *configuration*, not to the layer: vLLM instantiates one of these per layer (28 for this
        # model), and building a plan per layer multiplied a once-only cost by the layer count.
        cls = TileInferDecodeAttention
        if not hasattr(cls, "_plans"):
            cls._plans: Dict[Any, Any] = {}
            cls._building: set = set()
            cls._failed: set = set()
            cls._lock = threading.Lock()
            cls._attn = None
            cls._marker = None
            cls._stats = {
                "capture_calls": 0,
                "eager_calls": 0,
                "fallbacks": 0,
                "exceptions": 0,
            }
        self._config_key = (
            num_heads,
            num_kv_heads,
            head_size,
            round(float(scale), 6),
            str(device),
            str(dtype),
            int(kv_tile_pages),
        )

    # ------------------------------------------------------------------ availability

    @property
    def group_size(self) -> int:
        return max(1, self.num_heads // max(1, self.num_kv_heads))

    def supported(self, block_size: int) -> bool:
        """Whether the kernel can handle this configuration at all."""
        reasons = []
        if self.head_size not in (128, 256):
            reasons.append(
                f"head_size={self.head_size} (the Ascend 950 path is tuned for 128 and 256)"
            )
        if block_size < 16:
            reasons.append(f"block_size={block_size} (the paged kernel needs >= 16)")
        elif block_size % KERNEL_PAGE_SIZE != 0:
            reasons.append(
                f"block_size={block_size} (the kernel tiles 128-token pages; "
                "the engine page must be a multiple)"
            )
        if self.group_size < 2:
            reasons.append(f"group={self.group_size} (MHA needs the reference path)")
        if reasons:
            if not self._warned:
                logger.warning("TileInfer decode disabled: %s", "; ".join(reasons))
                self._warned = True
            return False
        return True

    # ------------------------------------------------------------------ plan cache

    def _plan_key(self, bucket: int, width: int, block_size: int, pool_pages: int) -> tuple:
        """Plan cache key.

        ``width`` is the request's *actual* block-table width in kernel pages (a constant buffer
        width in vLLM: ``ceil(max_model_len / engine_block)`` sub-pages), so one plan covers a
        whole run and the plan-owned staging buffers can be sized to it exactly.
        """
        return (self._config_key, bucket, width, block_size, pool_pages)

    def prewarm(self, width: int, block_size: int, key_cache: torch.Tensor) -> None:
        """Synchronously build a plan for *every* bucket up to ``max_num_reqs``.

        Called from the first non-captured decode step — vLLM's pre-capture warmup run — so the
        whole plan set exists before ACLGraph capture begins.  A compile takes ~100 s cold and
        hits the TileLang disk cache when warm, which is acceptable exactly once at startup.

        ``width`` counts kernel pages (engine table width × sub-pages).  Plans are built on
        kernel pages (see KERNEL_PAGE_SIZE) and the pool is presented as its 128-token view.
        """
        cls = TileInferDecodeAttention
        pool_eff = int(key_cache.shape[0])
        # ``max_num_reqs`` comes from the impl's vLLM config; when it is unavailable build every
        # bucket anyway (compiles hit the TileLang disk cache, so the cost is one-time).
        limit = int(self.max_num_reqs) if self.max_num_reqs else BATCH_BUCKETS[-1]
        built = 0
        for bucket in BATCH_BUCKETS:
            if bucket > limit:
                break
            key = self._plan_key(bucket, width, block_size, pool_eff)
            with cls._lock:
                if key in cls._plans or key in cls._failed or key in cls._building:
                    continue
                cls._building.add(key)
            try:
                self._build_plan(key, bucket, width, block_size, key_cache)
                built += 1
            except Exception:  # noqa: BLE001 - one bad bucket must not kill the engine
                logger.exception("TileInfer: prewarm compilation failed for bucket=%d", bucket)
                with cls._lock:
                    cls._failed.add(key)
            finally:
                with cls._lock:
                    cls._building.discard(key)
        if built:
            logger.info(
                "TileInfer: prewarmed %d plan(s) (buckets up to %d, table width %d pages) "
                "before graph capture",
                built,
                limit,
                width,
            )

    def plan_if_ready(self, bucket: int, max_pages: int, block_size: int, key_cache: torch.Tensor):
        """Return the plan for this shape, or ``None`` while it is being compiled.

        Kept as the *fallback* path for a step that needs a shape ``prewarm`` did not cover (for
        example a wider table after a config change): such a step compiles in the background and
        serves through FIA until the plan shows up.  The graph-capture path never comes here.
        """
        sub = _sub_pages(block_size)
        width = max_pages * sub
        pool_eff = int(key_cache.shape[0])
        key = self._plan_key(bucket, width, block_size, pool_eff)
        plan = self._plans.get(key)
        if plan is not None or key in self._failed:
            return plan

        with self._lock:
            if key in self._building:
                return None
            self._building.add(key)

        logger.info(
            "TileInfer: compiling the decode kernel for bucket=%d pages<=%d block_size=%d pool=%d "
            "pages in the background; serving through FIA until it is ready",
            bucket,
            width,
            block_size,
            pool_eff,
        )
        threading.Thread(
            target=self._compile_plan, args=(key, bucket, width, block_size, key_cache), daemon=True
        ).start()
        return None

    def _compile_plan(self, key, bucket, width, block_size, key_cache) -> None:
        try:
            self._build_plan(key, bucket, width, block_size, key_cache)
        except Exception:  # noqa: BLE001 - a background failure must not kill the engine
            logger.exception("TileInfer: background compilation failed; staying on the FIA path")
            with self._lock:
                self._failed.add(key)
        finally:
            with self._lock:
                self._building.discard(key)

    def _build_plan(self, key, bucket: int, width: int, block_size: int, key_cache: torch.Tensor):

        from tileinfer import BatchAttention
        from tileinfer.metadata import RaggedMetadata

        _diag("build_plan")
        cls = TileInferDecodeAttention
        if cls._attn is None:
            cls._attn = BatchAttention(backend="tilelang-ascend950", device=self.device, dtype=self.dtype)

        # A *dummy* page table: only its shape matters (it sizes the plan's workspace, so it is the
        # request's real table width, not the whole pool).  Decode without KV splitting produces
        # one work tile per request, so the schedule does not depend on the lengths; the values are
        # refreshed every step by the caller.  Plans are always built on kernel pages (128 tokens)
        # and the pool is presented as its 128-token sub-page view.
        sub = _sub_pages(block_size)
        pool = _pool_sub_view(key_cache, sub)
        dummy_lens = torch.ones(bucket, dtype=torch.int32, device=self.device)
        dummy = RaggedMetadata(
            kv_indptr=torch.arange(0, bucket + 1, dtype=torch.int32, device=self.device),
            kv_indices=torch.zeros(bucket, dtype=torch.int32, device=self.device),
            kv_last_page_len=torch.ones(bucket, dtype=torch.int32, device=self.device),
            page_size=KERNEL_PAGE_SIZE,
        )
        plan = cls._attn.plan_from_page_table(
            torch.zeros((bucket, width), dtype=torch.int32, device=self.device),
            dummy_lens,
            qo_lens=torch.ones(bucket, dtype=torch.int32, device=self.device),
            page_size=KERNEL_PAGE_SIZE,
            num_qo_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_size,
            sm_scale=self.scale,
            causal=False,
            kv_tile_pages=self.kv_tile_pages,
            k_cache=pool,
            # Only the *shape* is read (`finalize_plan` asserts K/V match); a full
            # `empty_like(key_cache)` duplicate of the V pool is ~0.8 GiB here and OOMs whenever
            # the KV cache already fills the budget.
            v_cache=pool,
        )
        state = plan.backend_state
        # Engine-side staging (plan-owned, refreshed every step with device-side copies only):
        # the expanded sub-page table and the padded lens vector.  Padding rows stay at
        # table=0 / len=1: a valid page id and a valid length, so the kernel's softmax cannot
        # divide by zero and the padded outputs are never read.
        state["table_buf"] = torch.zeros((bucket, width), dtype=torch.int32, device=self.device)
        state["lens_buf"] = torch.ones(bucket, dtype=torch.int32, device=self.device)
        state["table_width"] = width
        # The ragged triple is rebuilt on the device every step (H4): pages_b = ceil(lens_b/128),
        # kv_indptr = exclusive cumsum, and the row-major table columns are *compacted* into
        # kv_indices with a fixed-shape scatter.  The kernel derives pages_b from indptr diffs,
        # so an invalid (stale) table column must not be reachable through the indptr span — an
        # earlier revision used the dense indptr ``arange(bucket+1)*width`` instead, which made
        # every row attend over ``width`` pages including stale ids and produced garbage output.
        # All helpers below are fixed-shape plan-owned buffers: the per-step rebuild is pure
        # in-place device work (no allocation, no host sync), so it captures into ACLGraph.
        state["pages_buf"] = torch.empty(bucket, dtype=torch.int32, device=self.device)
        state["j_buf"] = torch.arange(width, dtype=torch.int32, device=self.device)
        state["valid_buf"] = torch.empty((bucket, width), dtype=torch.bool, device=self.device)
        state["invalid_buf"] = torch.empty((bucket, width), dtype=torch.bool, device=self.device)
        state["dst32_buf"] = torch.empty((bucket, width), dtype=torch.int32, device=self.device)
        state["dst64_buf"] = torch.empty((bucket, width), dtype=torch.int64, device=self.device)
        # kv_indices is pool-sized (the kernel's declared page cap); ``sum(pages_b)`` is at most
        # ``bucket*width``, far below the pool size, so the last slot is never covered by any
        # row's indptr span and serves as the scatter's trash destination for invalid columns.
        trash = int(state["kv_indices_buf"].numel()) - 1
        if bucket * width >= trash:
            raise ValueError(
                f"KV pool ({trash + 1} pages) smaller than the staging table "
                f"({bucket}x{width}); the ragged compaction has nowhere to park invalid columns"
            )
        state["trash_slot"] = trash
        state["kv_indptr_buf"].zero_()
        if cls._marker is None:
            # In-service probe: bumped once per kernel launch inside run().  Under ACLGraph the
            # increment is *captured into the graph*, so it also counts replays — the number grows
            # while the engine serves decode steps even though no Python runs (B5c).
            cls._marker = torch.zeros(1, dtype=torch.int64, device=self.device)
        self._plans[key] = plan
        logger.info(
            "TileInfer plan READY: bucket=%d table_width=%d block_size=%d -> %s",
            bucket,
            width,
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
        """One decode step: ``query`` is ``[num_reqs, num_heads, head_size]`` (the layout vLLM-Ascend
        passes here), writes into ``output`` with the same layout."""
        cls = TileInferDecodeAttention
        capturing = torch.npu.is_current_stream_capturing()
        num_reqs = int(seq_lens.numel())
        bucket = _bucket_for(num_reqs)
        sub = _sub_pages(block_size)
        width = int(block_tables.shape[1]) * sub
        pool_eff = int(key_cache.shape[0])
        key = self._plan_key(bucket, width, block_size, pool_eff)
        plan = cls._plans.get(key)
        if plan is None:
            if capturing:
                # Inside a capture we must not build, not allocate and not raise: let FIA serve
                # this step.  (prewarm() during the pre-capture warmup makes this path a no-op.)
                cls._stats["fallbacks"] += 1
                return None
            self.prewarm(width, block_size, key_cache)
            plan = cls._plans.get(key)
            if plan is None:
                cls._stats["fallbacks"] += 1
                return None

        state = plan.backend_state
        # Engine pages -> kernel pages: expand the page table to 128-token sub-page ids (zero-copy;
        # identity for block_size == 128) and stage everything into plan-owned buffers.  Every op
        # below is shape-static and device-side, so the whole step is capturable.
        table_exp = _expand_page_table(block_tables[:num_reqs], sub)
        table_buf: torch.Tensor = state["table_buf"]
        table_buf[:num_reqs].copy_(table_exp)
        # Zero the padding rows' columns so a padding row (pages=1 from its stale len=1) can only
        # ever reach page 0 — never a leftover id from some earlier longer batch.
        table_buf[num_reqs:].zero_()
        lens_buf: torch.Tensor = state["lens_buf"]
        lens_buf[:num_reqs].copy_(seq_lens[:num_reqs].to(torch.int32))
        # last-page-len from the padded lens, in place: ((lens - 1) % page) + 1
        lpl = state["last_page_len_buf"]
        torch.sub(lens_buf, 1, out=lpl)
        lpl.remainder_(KERNEL_PAGE_SIZE)
        lpl.add_(1)
        # Rebuild the ragged triple on the device (H4 fix): fixed shapes, in-place, capturable.
        # pages_b = ceil(lens_b / 128); kv_indptr = exclusive cumsum; request b's pages end up at
        # ``kv_indices[indptr[b] : indptr[b] + pages_b]`` — exactly the span the kernel derives
        # from indptr diffs.  Invalid columns are parked in the trash slot, outside every span.
        pages: torch.Tensor = state["pages_buf"]
        torch.add(lens_buf, KERNEL_PAGE_SIZE - 1, out=pages)
        pages.floor_divide_(KERNEL_PAGE_SIZE)
        indptr: torch.Tensor = state["kv_indptr_buf"]
        torch.cumsum(pages, dim=0, out=indptr[1:])
        j: torch.Tensor = state["j_buf"]
        torch.lt(j.unsqueeze(0), pages.unsqueeze(1), out=state["valid_buf"])
        torch.ge(j.unsqueeze(0), pages.unsqueeze(1), out=state["invalid_buf"])
        torch.add(indptr[:-1].unsqueeze(1), j.unsqueeze(0), out=state["dst32_buf"])
        dst64: torch.Tensor = state["dst64_buf"]
        dst64.copy_(state["dst32_buf"])
        dst64.masked_fill_(state["invalid_buf"], state["trash_slot"])
        idx_buf: torch.Tensor = state["kv_indices_buf"]
        idx_buf.scatter_(0, dst64.view(-1), table_buf.view(-1))

        # Stage the padded query.  vLLM hands us ``[num_tokens, num_heads, head_size]`` on this
        # stack (verified); the kernel wants ``[bucket, num_kv_heads * BR, head_size]`` with the
        # GQA group zero-padded to BR rows.  Head order is already ``kv_head * group + i``.
        from tileinfer.kernels.attention.paged_decode_ascend950 import padded_rows

        br = padded_rows(self.group_size)
        q_pad: torch.Tensor = state["q_pad"]
        q_pad.view(bucket, self.num_kv_heads, br, self.head_size)[:, :, : self.group_size].copy_(
            query[:num_reqs]
            .to(self.dtype)
            .reshape(num_reqs, self.num_kv_heads, self.group_size, self.head_size)
        )

        TileInferDecodeAttention._attn.run(
            q_pad,
            _pool_sub_view(key_cache, sub),
            _pool_sub_view(value_cache, sub),
            plan=plan,
            kv_indptr=state["kv_indptr_buf"],
            kv_indices=idx_buf,
            kv_last_page_len=lpl,
        )
        cls._stats["capture_calls" if capturing else "eager_calls"] += 1
        if cls._marker is not None:
            cls._marker += 1

        # Copy the live rows back with the engine's ``output`` layout
        # (``[num_tokens, num_heads, head_size]`` on this stack).
        result4 = state["out_pad"].view(bucket, self.num_kv_heads, br, self.head_size)
        output[:num_reqs].view(num_reqs, self.num_kv_heads, self.group_size, self.head_size).copy_(
            result4[:num_reqs, :, : self.group_size]
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
                import os

                try:
                    max_num_reqs = int(self.vllm_config.scheduler_config.max_num_seqs)
                except Exception:  # noqa: BLE001 - fall back to the largest bucket
                    max_num_reqs = 0
                self._tileinfer = TileInferDecodeAttention(
                    num_heads=self.num_heads,
                    num_kv_heads=self.num_kv_heads,
                    head_size=self.head_size,
                    scale=self.scale,
                    device=self.key_cache.device,
                    # Split-KV (`kv_tile_pages > 0`) was unreachable from the engine before: the ctor
                    # defaulted to 0 and nothing ever passed it.  It is still opt-in (the kernel also
                    # refuses split plans without TILEINFER_ALLOW_SPLIT_KV=1, see KI-1).
                    kv_tile_pages=int(os.environ.get("TILEINFER_KV_TILE_PAGES", 0)),
                    # Prewarm builds one plan per bucket up to this, before graph capture.
                    max_num_reqs=int(max_num_reqs or 0),
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

            if not ops_enabled("decode"):
                return super().forward_impl(query, key, value, kv_cache, attn_metadata, output)

            if attn_metadata.attn_state != AscendAttentionState.DecodeOnly:
                return super().forward_impl(query, key, value, kv_cache, attn_metadata, output)

            block_size = int(self.key_cache.shape[1])
            impl = self._decode(block_size)
            if impl is None:
                return super().forward_impl(query, key, value, kv_cache, attn_metadata, output)

            try:
                num_reqs = int(attn_metadata.seq_lens.numel())
                # Device-resident lengths only: the host-side `seq_lens` cannot be copied inside
                # an ACLGraph capture (aclrtMemcpy is rejected while the stream is capturing),
                # and replay would never refresh a plan-owned buffer from a host tensor.  The
                # common metadata's device seq_lens lives in vLLM's fixed-address graph input
                # buffer, so a D2D copy from it is capturable and replays correctly.  (The
                # `seq_lens_device` field is added to vllm-ascend's AscendMetadata by this
                # project - see .agent_loop/state/DECISIONS.md D17 for the exact patch.)
                device_lens = getattr(attn_metadata, "seq_lens_device", None)
                if device_lens is None:
                    device_lens = attn_metadata.seq_lens
                out = impl.run(
                    query=query[:num_reqs],
                    key_cache=self.key_cache,
                    value_cache=self.value_cache,
                    block_tables=attn_metadata.block_tables[:num_reqs],
                    seq_lens=device_lens[:num_reqs],
                    block_size=block_size,
                    output=output,
                )
                if out is None:  # kernel not compiled yet - keep serving through FIA
                    return super().forward_impl(query, key, value, kv_cache, attn_metadata, output)
                return out
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


def _diag(tag: str) -> None:
    """One-shot diagnostics for KI-2: what does the registry look like in *this* process?

    It names the module file so a duplicate-module-identity problem is visible at a glance.  Kept at
    debug: it is what identified the un-serialised backend import (see `_ensure_builtin_backends`),
    but it fires on every plan build and is noise in a normal run.
    """
    try:
        import tileinfer
        from tileinfer.attention.backends.base import _IMPORT_ERRORS, _REGISTRY, list_backends

        logger.debug(
            "TileInfer diag[%s]: module=%s registry=%s errors=%s",
            tag,
            getattr(tileinfer, "__file__", "?"),
            sorted(_REGISTRY),
            {k: f"{type(v).__name__}: {v}"[:90] for k, v in _IMPORT_ERRORS.items()},
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("TileInfer diag[%s] failed: %r", tag, exc)


def _build_classes_into_module() -> None:
    """(Re)build the vLLM-Ascend classes into this module's globals."""
    global TileInferBackend, TileInferImpl
    TileInferBackend, TileInferImpl = _build_classes()


_METRICS_INTERVAL_S = 1.0


def start_metrics_dumper() -> bool:
    """Serve the "is TileInfer actually serving" probe (BACKLOG B5c).

    When ``TILEINFER_METRICS_PATH`` is set, a daemon thread atomically writes one JSON line every
    second: plan READY count, kernel-call counters (``capture_calls`` inside a graph capture,
    ``eager_calls`` outside), FIA-fallback count, the ``kernel_markers`` device counter and the
    exception latch.  ``kernel_markers`` is bumped by a device-side add *inside* run() — under
    ACLGraph that increment is captured into the graph, so it counts replays too: it grows while
    the engine decodes even though no Python runs.  A benchmark asserts on this file *before*
    trusting any number: ready_plans >= 1, fallbacks == 0 and, in graph mode, capture_calls > 0.
    """
    import json
    import os
    import time

    path = os.environ.get("TILEINFER_METRICS_PATH")
    if not path:
        return False

    def _loop() -> None:
        while True:
            try:
                cls = TileInferDecodeAttention
                stats = dict(getattr(cls, "_stats", {}))
                stats["ready_plans"] = len(getattr(cls, "_plans", {}))
                marker = getattr(cls, "_marker", None)
                stats["kernel_markers"] = int(marker.item()) if marker is not None else -1
                stats["ts"] = time.time()
                # Only a process that actually hosts TileInfer state (the EngineCore) may write:
                # the API server imports the plugin too, and its empty view would keep overwriting
                # the real numbers.  Empty processes stay silent instead.
                if (
                    stats["ready_plans"] == 0
                    and stats["capture_calls"] == 0
                    and stats["eager_calls"] == 0
                    and stats["fallbacks"] == 0
                    and stats["kernel_markers"] < 0
                ):
                    time.sleep(_METRICS_INTERVAL_S)
                    continue
                tmp = f"{path}.tmp"
                with open(tmp, "w", encoding="utf-8") as handle:
                    json.dump(stats, handle)
                os.replace(tmp, path)
            except Exception:  # noqa: BLE001 - metrics must never break serving
                pass
            time.sleep(_METRICS_INTERVAL_S)

    threading.Thread(target=_loop, daemon=True, name="tileinfer-metrics").start()
    logger.info("TileInfer serving probe -> %s", path)
    return True


_INSTALLED = False


def install(raise_on_failure: bool = True) -> bool:
    """Register TileInfer in vLLM's ``CUSTOM`` attention slot.

    Call this *after* ``vllm_ascend`` has been imported and *before* vLLM selects a backend (the
    runner script does it before touching the CLI).  Returns ``True`` when the registration took.
    """
    global _INSTALLED
    if _INSTALLED:
        return True

    logging.getLogger("tileinfer").setLevel(logging.INFO)
    _diag("install")

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
        _INSTALLED = True
        start_metrics_dumper()
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
