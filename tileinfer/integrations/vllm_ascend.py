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
import os
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
#: ``gdnwu`` 是 chunk 链中段融合核(H10/阶段 3 P1)的独立开关,与 ``gdn`` 解耦做同侧 A/B 归因。
#: ``sampling`` 是采样 top-k 阈值替换(BACKLOG 采样条目澄清:非采样链整体融合,性质同
#: GDN gating 的 opt-in 小算子接管)的独立开关。
#: ``gdnproj`` 是 GDN in_proj_ba→in_proj_qkvz 权重融合(H32/S-GDN 投影族,E100:ba 为
#: 纯内核启动延迟税 ~0.67% TPOT,融合逐位等价)的独立开关。
#: ``moeprefetch`` 是 MoE 权重 prefetch 阈值对齐(H37/S-MoE 域,#19033 第 2 项移植:
#: 管道已在 vllm-ascend 0.23.0 在役路径上,E109 管道核查;阈值 96→1 + 惰性武装兜底)
#: 的独立开关。
#: ``gdnreorder`` 是 GDN fused(qkvz+ba)权重 load 期行重排(H34/S-GDN 投影族,
#: E102 vLLM #55260 同构:终序单 GEMM+零拷贝切片替代 forward 期 ordering/rearrange/cat)
#: 的独立开关;置于 gdnproj 之上(接管成功时 gdnproj 包装不触达)。
_OPS_ALL = {
    "decode",
    "prefill",
    "gdn",
    "gdnwu",
    "moe",
    "sampling",
    "gdnproj",
    "gdnreorder",
    "gdnsmall",
    "greedy",
    "moeprefetch",
}


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


def _parse_ktp_policy(raw: str) -> Dict[int, int]:
    """Parse ``TILEINFER_KV_TILE_PAGES_POLICY`` (H21): ``"1:16,2:32,8:128"`` → per-bucket ktp.

    The concurrency tax of decode split-KV is *wave-quantised* (E61): the schedule launches
    ``bucket × ceil(width / ktp)`` tiles regardless of the real context lengths, and a tile
    costs a full AIV wave slot even when its page loop is clamped to zero — b8 × ktp16 = 128
    tiles ≈ 4.6 waves vs plain's 1 wave (measured 4.54x), while the *same* split machinery at
    ≤28 total tiles is free (E61 matrix C: width-16 plan ≡ plain).  So the ktp that is right
    for batch 1 (max splits, starved device) is exactly wrong for batch 32 (device already
    full; unsplit).  The policy picks one ktp per bucket; buckets left out fall back to the
    global ``TILEINFER_KV_TILE_PAGES``.
    """
    policy: Dict[int, int] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        bucket, sep, ktp = part.partition(":")
        if not sep or not bucket.strip().isdigit() or not ktp.strip().isdigit():
            raise ValueError(
                f"bad TILEINFER_KV_TILE_PAGES_POLICY entry {part!r} "
                '(expected comma-separated "bucket:ktp" pairs, e.g. "1:16,8:128")'
            )
        policy[int(bucket)] = int(ktp)
    return policy


# ---------------------------------------------------------------------------------------
# B1: prefill / append through the TileLang prefill kernel.
#
# Unlike decode, a prefill step always runs eager (never inside an ACLGraph capture), so the
# host-side work here — per-request loops, ``.item()`` syncs, a ``masked_select`` page-table
# walk — is acceptable; the B3 profile decides later how much of it matters.  What *does*
# matter is kernel-variant control: the prefill kernel compiles per
# ``(batch, kv_heads, group, dim, page, pool_pages, num_tiles, block_q)``, and ``num_tiles``
# depends on the prompt/chunk sizes.  :class:`PrefillShapeManager` compiles every new shape in
# a background thread (FIA serves meanwhile) and caps the total number of variants so a weird
# workload can neither stall a step with a synchronous compile nor compile without bound.
# ---------------------------------------------------------------------------------------

#: Upper bound on tiles per prefill step; above this the shape stays on FIA for good (a tile is
#: 128 ``(q_pos, head)`` rows, so 4096 tiles ≈ a 64K-token chunk at group=8).
PREFILL_MAX_TILES = int(__import__("os").environ.get("TILEINFER_PREFILL_MAX_TILES", "4096"))

# H19: split-KV prefill knob (KV pages per split tile).  0 = off (the unsplit prefill kernel).
# Long-context serving is where the per-tile serial page walk hurts (W1-C8, E46/E47); short
# requests stay whole automatically (a request shorter than one split never splits).
_PREFILL_KV_TILE_PAGES = int(__import__("os").environ.get("TILEINFER_PREFILL_KV_TILE_PAGES", "0") or 0)

# D57-①(W1 收尾):长上下文 prefill 步分流回退 FIA。TI prefill 内核的单位功效率随 KV
# 上下文线性劣化(E46 aiv_vec_ratio=0.96/mac=0.02 + E47 同功线性扩展),C8 饱和下 8K/16K
# 场景整体吞吐仅 0.213x/0.121x(E41),split 路线亦无增益(E63);把 KV 上下文(按内核页
# 128 token 粒度上取整)超过该阈值的 prefill 步交还 vLLM 原生 FIA 路径——decode 接管
# (P8 已在役)不受影响。0 = 关闭(默认,行为零变化);决策零新增同步(读的 pages_host
# 是共享构建链本来就要算的量)。
_PREFILL_MAX_CTX = int(__import__("os").environ.get("TILEINFER_PREFILL_MAX_CTX", "0") or 0)

#: D57-① 在役探针计数(被分流回 FIA 的 prefill 步数;metrics dumper 合并输出)。
_PREFILL_STATS: Dict[str, int] = {"prefill_long_skips": 0}
#: 被分流步的守卫 (step_id, num_reqs, num_tokens):同一步的其余层快速短路,免重复构建
#: 共享面。键含形状,与 _PREFILL_STEP_GUARD 的 belt-and-braces 同款防 id 复用。
_PREFILL_SKIP_GUARD: tuple = ()


def _prefill_ctx_shunt(pages_host, threshold: int) -> bool:
    """D57-① 决策纯函数:该步 KV 上下文是否超阈(超阈 → 该步回退 FIA)。

    ``pages_host`` 是每请求的内核页数(128 token 粒度,上取整——与内核实际遍历量一致,
    故比较值天然保守:真上下文只会更小)。``threshold<=0`` 恒 False(开关关闭)。
    """
    if threshold <= 0 or not pages_host:
        return False
    return max(pages_host) * KERNEL_PAGE_SIZE > threshold

# Per-step metadata dedup for prefill (H18, the B3 step-guard extended to the prefill path):
# vLLM hands the *same* attention-metadata instance to every full-attention layer of a step and
# builds a fresh one per step, but everything ``prefill_step`` derives from it (qo lengths, the
# ragged page-table triple, the tile schedule, the compiled-shape decision) is layer-independent.
# E45 measured the redundancy at ~10x: per mixed step that was ~180 host-device syncs and a
# ~5.5 GB q_pad zero-fill just to re-derive identical values — the W1-C8 pathology's prime
# suspect.  Keyed on the metadata's identity exactly like the decode guard above: the strong
# reference keeps the guarded id alive, so no *other* live object can ever collide with it.
_PREFILL_STEP_GUARD: int = 0
_PREFILL_STEP_OBJ: Any = None  # strong ref: keeps the guarded id unique while it guards
_PREFILL_STEP_SHARED: dict = {}


def _next_pow2(n: int) -> int:
    """Smallest power of two ``>= n`` (P3 canonical tile quantization; 1 stays 1)."""
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def _pad_tile_schedule(schedule, num_tiles: int, trash_seq: int):
    """Pad a :class:`TileSchedule` to ``num_tiles`` entries with dead tiles (P3).

    A dead tile has ``q_len == 0`` (the pack/unpack loops skip it, every ``RowThr`` row is -1)
    and points at the *trash row* — a batch-padding request with zero KV pages, so the kernel's
    page loop runs zero iterations for it.  Pure tensor ops on the schedule's device.
    """
    t = schedule.num_tiles
    if num_tiles <= t:
        return schedule
    pad = num_tiles - t

    def _pad(tensor: torch.Tensor, fill: int) -> torch.Tensor:
        extra = torch.full((pad,), fill, dtype=torch.int32, device=tensor.device)
        return torch.cat([tensor.to(torch.int32), extra]).contiguous()

    from tileinfer.plan import TileSchedule

    return TileSchedule(
        seq_ids=_pad(schedule.seq_ids, trash_seq),
        q_offsets=_pad(schedule.q_offsets, 0),
        q_lens=_pad(schedule.q_lens, 0),
        kv_page_starts=_pad(schedule.kv_page_starts, 0),
        kv_page_lens=_pad(schedule.kv_page_lens, 0),
        split_ids=_pad(schedule.split_ids, 0),
        split_counts=_pad(schedule.split_counts, 1),
        balanced=schedule.balanced,
    )


class PrefillShapeManager:
    """Lazy background compiler for the prefill kernel's shape variants (BACKLOG B1)."""

    _ready: set = set()
    _building: set = set()
    _lock = threading.Lock()
    _cap = int(__import__("os").environ.get("TILEINFER_PREFILL_MAX_VARIANTS", "32"))

    @classmethod
    def usable(cls, spec: tuple) -> bool:
        """True when ``spec`` is compiled; otherwise kick off a background compile (once).

        The caller must fall back to FIA for this step when False — a compile takes ~90 s and
        must never run on the serving path.
        """
        if spec in cls._ready:
            return True
        with cls._lock:
            if spec in cls._building:
                return False
            if len(cls._ready) >= cls._cap:
                return False  # variant budget exhausted: this shape permanently stays on FIA
            cls._building.add(spec)
        logger.info("TileInfer: compiling prefill kernel for shape %s in the background", spec)
        threading.Thread(target=cls._compile, args=(spec,), daemon=True).start()
        return False

    @classmethod
    def prewarm_sync(cls, spec: tuple) -> bool:
        """Synchronously compile ``spec`` (P3 prewarm); True when it is (now) ready.

        Used from ``prewarm()`` — startup, off the serving path — to build the canonical variant
        ladder before any traffic.  Skips specs that are ready, already building (a lazy compile
        from a startup dummy step may be in flight) or beyond the variant cap.
        """
        if spec in cls._ready:
            return True
        with cls._lock:
            if spec in cls._building:
                return False
            if len(cls._ready) >= cls._cap:
                return False
            cls._building.add(spec)
        cls._compile(spec)
        return spec in cls._ready

    @classmethod
    def _compile(cls, spec: tuple) -> None:
        try:
            from tileinfer.kernels.attention.paged_decode_ascend950 import (
                build_prefill_kernel,
                build_prefill_split_head_kernel,
                build_prefill_split_frontier_kernel,
            )

            if len(spec) == 9:  # plain prefill variant
                (
                    batch,
                    kv_heads,
                    group,
                    dim,
                    page,
                    pool_pages,
                    num_tiles,
                    block_q,
                    dtype_name,
                ) = spec
                build_prefill_kernel(
                    batch=batch,
                    kv_heads=kv_heads,
                    group=group,
                    dim=dim,
                    page_size=page,
                    num_pages_cap=pool_pages,
                    num_tiles=num_tiles,
                    block_q=block_q,
                    dtype=dtype_name,
                )
            else:  # split-KV prefill variant: the 10th field is ``max_splits`` (H19/H20)
                (
                    batch,
                    kv_heads,
                    group,
                    dim,
                    page,
                    pool_pages,
                    num_tiles,
                    block_q,
                    dtype_name,
                    max_splits,
                ) = spec
                # H20: the split-prefill step runs two kernels (head = fully-visible chunks,
                # no per-row predicate; frontier = plain-form mask chunk).  Both share the
                # spec; the tile axes differ per step, so prewarm builds the padded-size pair
                # and lazy compiles fill in the exact sizes as traffic demands them.
                build_prefill_split_head_kernel(
                    batch=batch,
                    kv_heads=kv_heads,
                    group=group,
                    dim=dim,
                    page_size=page,
                    num_pages_cap=pool_pages,
                    num_tiles=num_tiles,
                    max_splits=max_splits,
                    num_slots=num_tiles + 1,
                    block_q=block_q,
                    dtype=dtype_name,
                )
                build_prefill_split_frontier_kernel(
                    batch=batch,
                    kv_heads=kv_heads,
                    group=group,
                    dim=dim,
                    page_size=page,
                    num_pages_cap=pool_pages,
                    num_tiles=num_tiles,
                    max_splits=max_splits,
                    num_slots=num_tiles + 1,
                    block_q=block_q,
                    dtype=dtype_name,
                )
            cls._ready.add(spec)
            logger.info("TileInfer prefill kernel READY: %s", spec)
        except Exception:  # noqa: BLE001 - a compile failure must not kill the engine
            logger.exception("TileInfer: prefill compile failed for %s; staying on FIA", spec)
        finally:
            with cls._lock:
                cls._building.discard(spec)


_REFUSE_LOGGED: set = set()


def _refuse(reason: str, got, expect) -> None:
    """Log a prefill refusal once per reason (the counters still record every step)."""
    if reason not in _REFUSE_LOGGED:
        _REFUSE_LOGGED.add(reason)
        logger.info("TileInfer prefill refused (%s): got=%s expect=%s", reason, got, expect)


def prefill_step(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    block_size: int,
    output: torch.Tensor,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    scale: float,
    batch_cap: int = 0,
    step_obj: Any = None,
) -> bool:
    """One prefill/append step through the TileLang prefill kernel (BACKLOG B1).

    ``query`` is ``[num_tokens, num_heads, head_size]`` with ``num_tokens == sum(qo_lens)``;
    ``seq_lens`` are the *full* KV lengths in tokens (the step's K/V has already been written to
    the pool by ``reshape_and_cache`` before this runs — verified against vllm-ascend's
    ``forward``); ``query_start_loc`` is the cumulative token offset per request.  Covers
    PrefillNoCache (``kv_len == qo_len``), PrefillCacheHit and ChunkedPrefill (append to a cached
    prefix) uniformly: the kernel's per-row causal threshold ``q_pos + (kv_len - qo_len)`` makes
    all three the same computation.

    ``step_obj`` is the step's attention-metadata object (vLLM passes the same instance to every
    layer of a step and builds a fresh one per step); when provided, the shared metadata-build
    chain (page-table walk, tile schedule, shape decision — everything layer-independent, H18/E45)
    runs only for the first layer of the step, mirroring the decode path's B3 step-guard.

    Returns True when TileInfer served the step; False means "use FIA" (shape not compiled yet
    or structurally unsupported).  Eager-only: must not be called under a graph capture.
    """
    global _PREFILL_STEP_GUARD, _PREFILL_STEP_OBJ, _PREFILL_SKIP_GUARD
    num_reqs = int(seq_lens.numel())
    if num_reqs == 0 or num_reqs != int(query_start_loc.numel()) - 1:
        _refuse("req/qsl mismatch", num_reqs, int(query_start_loc.numel()) - 1)
        return False

    step_id = id(step_obj) if step_obj is not None else None
    # D57-①:已被分流的步,同一步其余层直接短路(免重复的页表展开/共享面构建)。
    if (
        step_id is not None
        and _PREFILL_SKIP_GUARD == (step_id, num_reqs, int(query.shape[0]))
    ):
        return False
    shared = _PREFILL_STEP_SHARED if (step_id is not None and step_id == _PREFILL_STEP_GUARD) else None
    if shared is not None and (
        shared["num_reqs"] != num_reqs or shared["sum_qo"] != int(query.shape[0])
    ):
        shared = None  # belt-and-braces: a guard hit must still describe *this* step

    if shared is None:
        # First layer of this step (or a library-level caller without step identity):
        # build the layer-independent payload once.
        group = max(1, num_heads // num_kv_heads)
        qo_host = (query_start_loc[1:] - query_start_loc[:-1]).to(torch.int64).cpu().tolist()
        if sum(qo_host) != int(query.shape[0]):
            # 常见原因:runner 对 query 做了 token 填充(短批尤甚)。padding 行的输出会被 vLLM
            # 裁掉,因此只要 sum(qo) <= num_tokens 就可以安全接管,多余行不读不写。
            if sum(qo_host) > int(query.shape[0]):
                _refuse("qo sum exceeds tokens", sum(qo_host), int(query.shape[0]))
                return False
        max_tiles = sum((q * group + 127) // 128 for q in qo_host)
        if max_tiles == 0 or max_tiles > PREFILL_MAX_TILES:
            _refuse("tile bound", max_tiles, PREFILL_MAX_TILES)
            return False

        from tileinfer.metadata import ragged_from_page_table_device
        from tileinfer.plan import plan_query_tiles_from_host
        from tileinfer.kernels.attention.paged_decode_ascend950 import declared_pool_cap

        device = query.device
        sub = _sub_pages(block_size)
        table_exp = _expand_page_table(block_tables[:num_reqs], sub)
        # Kernel-page metadata: the same masking the decode path does, but eager (masked_select's
        # implicit sync is fine outside a capture).
        meta = ragged_from_page_table_device(table_exp, seq_lens[:num_reqs], KERNEL_PAGE_SIZE)

        dtype_name = _TORCH_DTYPE_NAME.get(query.dtype)
        if dtype_name is None:
            _refuse("dtype", str(query.dtype), "bfloat16/float16")
            return False
        if num_reqs > BATCH_BUCKETS[-1]:
            _refuse("batch over largest bucket", num_reqs, BATCH_BUCKETS[-1])
            return False
        pool_pages = declared_pool_cap(int(_pool_sub_view(key_cache, sub).shape[0]))
        # P3 canonical variant quantization: the compiled shape is not the step's exact
        # ``(batch, num_tiles)`` but ``(_bucket_for(batch_cap) + 1 trash rows, next pow2 tiles)``.
        # The trash row owns zero pages (padded tiles pointing at it run zero KV iterations) and
        # padded tiles carry q_len == 0 (fully masked, output unread), so the padding is invisible
        # semantically — and ``prewarm()`` can enumerate the whole canonical ladder (≤ 32 variants),
        # which is what removes the ~110 s first-seen lazy compile from serving.
        cap = int(batch_cap) if batch_cap and batch_cap >= num_reqs else num_reqs
        kernel_batch = _bucket_for(cap) + 1
        # Host-side page counts for the schedule: one extra tiny D2H, still once per step (was
        # per layer, and went through plan_query_tiles' per-row ``.item()`` storm on top).
        pages_host = (meta.kv_indptr[1:] - meta.kv_indptr[:-1]).to(torch.int64).cpu().tolist()

        # D57-①(W1 收尾):长上下文 prefill 步分流回退 FIA。决策在共享面构建之后、
        # split 规划/形状登记之前——同一步其余层走顶部短路,一步只付一次页表展开。
        if _prefill_ctx_shunt(pages_host, _PREFILL_MAX_CTX):
            _PREFILL_STATS["prefill_long_skips"] = (
                _PREFILL_STATS.get("prefill_long_skips", 0) + 1
            )
            if step_id is not None:
                _PREFILL_SKIP_GUARD = (step_id, num_reqs, int(query.shape[0]))
            _refuse("ctx shunt", max(pages_host) * KERNEL_PAGE_SIZE, _PREFILL_MAX_CTX)
            return False

        # H19: split-KV prefill.  ``TILEINFER_PREFILL_KV_TILE_PAGES`` splits every q-block tile
        # along the KV axis so the per-tile serial page walks (the W1-C8 pathology's dominant
        # device cost, E46/E47) run concurrently; the merge combines the partials.  The whole
        # step context (schedule, masks, pack/merge indices) is host math, built once per step.
        ktp = _PREFILL_KV_TILE_PAGES
        split_ctx = None
        max_splits = 0
        split_ms = 0
        if ktp > 0:
            lpl_host = meta.kv_last_page_len.to(torch.int64).cpu().tolist()
            # H20 split counts: head chunks of the visible pages + one frontier split — the
            # +1 keeps this *upper* bound estimate safe (it only sizes the tile budget and
            # the slot layout's power-of-two ``max_splits``; the ctx computes the real split
            # counts per q-block).
            splits_per_req = [max(1, -(-p // ktp) + 1) for p in pages_host]
            split_tiles = sum(
                ((q * group + 127) // 128) * s for q, s in zip(qo_host, splits_per_req)
            )
            if split_tiles <= PREFILL_MAX_TILES:
                kernel_tiles_split = _next_pow2(split_tiles)
                max_splits = _next_pow2(max(splits_per_req))
                indices_cap = meta.kv_indices.to(torch.int32)
                if indices_cap.numel() < pool_pages:
                    indices_cap = torch.cat(
                        [indices_cap, indices_cap.new_zeros(pool_pages - indices_cap.numel())]
                    )
                from tileinfer.kernels.attention.paged_decode_ascend950 import (
                    SPLIT_MS_COMPILE,
                    plan_prefill_split_ctx,
                )

                split_ms = SPLIT_MS_COMPILE  # cache-key material only (data-independent)

                split_ctx = plan_prefill_split_ctx(
                    qo_host,
                    pages_host,
                    lpl_host,
                    group=group,
                    kv_tile_pages=ktp,
                    kernel_batch=kernel_batch,
                    kernel_tiles=kernel_tiles_split,
                    kv_heads=num_kv_heads,
                    block_q=128,
                    max_qo_tokens=max(qo_host),
                    page_size=KERNEL_PAGE_SIZE,
                    indices=indices_cap,
                    device=device,
                )
                kernel_tiles = kernel_tiles_split
            else:
                # pathological split blow-up (huge batch × huge context): serve this step
                # through the unsplit kernel instead of refusing it entirely
                ktp = 0
        if split_ctx is None:
            kernel_tiles = _next_pow2(max_tiles)
        spec = (
            kernel_batch,
            num_kv_heads,
            group,
            head_size,
            KERNEL_PAGE_SIZE,
            pool_pages,
            kernel_tiles,
            128,
            dtype_name,
        ) + ((split_ms,) if split_ctx is not None else ())
        if not PrefillShapeManager.usable(spec):
            return False

        if split_ctx is None:
            schedule = plan_query_tiles_from_host(
                [q * group for q in qo_host],
                pages_host,
                device=device,
                block_q=128,
                load_balance=False,
            )
            if kernel_tiles != schedule.num_tiles:
                schedule = _pad_tile_schedule(schedule, kernel_tiles, kernel_batch - 1)
        qo_dev = torch.tensor(qo_host, dtype=torch.int32, device=device)
        shared = {
            "num_reqs": num_reqs,
            "sum_qo": int(query.shape[0]),
            "group": group,
            "qo_host": qo_host,
            "max_qo": max(qo_host),
            "kernel_batch": kernel_batch,
            "meta": meta,
            "schedule": None if split_ctx is not None else schedule,
            "split_ctx": split_ctx,
            "qo_dev": qo_dev,
        }
        if step_id is not None:
            _PREFILL_STEP_GUARD = step_id
            _PREFILL_STEP_OBJ = step_obj  # strong ref: keeps the id unique while it guards
            _PREFILL_STEP_SHARED.clear()
            _PREFILL_STEP_SHARED.update(shared)
            cls_stats = TileInferDecodeAttention._stats
            cls_stats["prefill_refreshes"] = cls_stats.get("prefill_refreshes", 0) + 1
            if cls_stats["prefill_refreshes"] <= 12:
                logger.info(
                    "TileInfer prefill step build #%d: metadata id=%s batch=%d tiles=%d",
                    cls_stats["prefill_refreshes"],
                    step_id,
                    kernel_batch,
                    kernel_tiles,
                )
    else:
        group = shared["group"]

    qo_host = shared["qo_host"]
    max_qo = shared["max_qo"]
    # [num_tokens, H, D] -> [batch, H, max_qo, D]: one permuted copy per request.  Rows past
    # qo_lens[b] stay zero; the kernel's row thresholds mask them out of the softmax.
    q_pad = torch.zeros(
        (num_reqs, num_heads, max_qo, head_size), dtype=query.dtype, device=query.device
    )
    offset = 0
    for b, qo in enumerate(qo_host):
        if qo:
            q_pad[b, :, :qo].copy_(query[offset : offset + qo].permute(1, 0, 2))
        offset += qo

    from tileinfer.kernels.attention.paged_decode_ascend950 import forward_prefill

    if shared["split_ctx"] is not None:
        from tileinfer.kernels.attention.paged_decode_ascend950 import forward_prefill_split

        out4 = forward_prefill_split(
            q_pad,
            _pool_sub_view(key_cache, _sub_pages(block_size)),
            _pool_sub_view(value_cache, _sub_pages(block_size)),
            shared["split_ctx"],
            group=group,
        )
    else:
        out4 = forward_prefill(
            q_pad,
            _pool_sub_view(key_cache, _sub_pages(block_size)),
            _pool_sub_view(value_cache, _sub_pages(block_size)),
            shared["meta"],
            shared["schedule"],
            shared["qo_dev"],
            group=group,
            block_q=128,
            kernel_batch=shared["kernel_batch"],
        )
    # Unpack [batch, H, max_qo, D] -> [num_tokens, H, D] with the engine's layout.
    offset = 0
    for b, qo in enumerate(qo_host):
        if qo:
            output[offset : offset + qo].copy_(out4[b, :, :qo].permute(1, 0, 2))
        offset += qo
    cls_stats = TileInferDecodeAttention._stats
    cls_stats["prefill_calls"] = cls_stats.get("prefill_calls", 0) + 1
    return True


_TORCH_DTYPE_NAME = {
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
}


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

    # The plan cache, its in-flight/failed sets and the serving counters belong to the
    # *configuration*, not to the layer: vLLM instantiates one of these per layer (28 for this
    # model), and building a plan per layer multiplied a once-only cost by the layer count.
    # Class-level (not lazily created by the first instance): the prefill path bumps the
    # counters without ever constructing an instance.
    _plans: Dict[Any, Any] = {}
    _building: set = set()
    _failed: set = set()
    _lock = threading.Lock()
    _attn = None
    _marker = None
    # Per-step metadata-refresh dedup (B3): every full-attention layer calls run() once per
    # decode step with the *same* page table / seq_lens, so the refresh chain would run
    # ~10x redundantly (E12: ~0.4ms per small NPU eager op on this stack).  Keying on the
    # identity of the step's attn_metadata object (a fresh AscendMetadata is built by
    # vllm-ascend's builder every step, capture included) makes the refresh run exactly
    # once per step - the plan-owned buffers it writes are shared by all layers.  A strong
    # reference to the object is kept so its id cannot be recycled while it is the guard.
    _step_guard: tuple = ()
    _step_obj: Any = None
    # H21: the active per-bucket split policy as built (metrics dumper reads this).
    _policy_view: Optional[Dict[str, int]] = None
    _stats: Dict[str, int] = {
        "capture_calls": 0,
        "eager_calls": 0,
        "fallbacks": 0,
        "exceptions": 0,
        "gdn_gating_calls": 0,
        "gdn_wu_calls": 0,
        "gdn_wu_fallbacks": 0,
        # 采样 top-k 阈值替换(BACKLOG 采样条目):命中/回退计数,在役探针断言用
        "sampler_topk_calls": 0,
        "sampler_topk_fallbacks": 0,
        # H32:GDN in_proj_ba→in_proj_qkvz 权重融合(S-GDN 投影族),命中/回退计数
        "gdnproj_calls": 0,
        "gdnproj_fallbacks": 0,
        # H34:GDN fused 权重 load 期行重排(终序单 GEMM+零拷贝切片),命中/回退计数
        "gdnreorder_calls": 0,
        "gdnreorder_fallbacks": 0,
        # H35:GDN 小核链削减(rearrange cat-free/conv1d 转置缓存/l2norm 配对融合)
        "gdnsmall_rearrange_calls": 0,
        "gdnsmall_conv_hits": 0,
        "gdnsmall_l2norm_fused": 0,
        "gdnsmall_fallbacks": 0,
    }

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
        max_num_batched_tokens: int = 0,
        max_model_len: int = 0,
    ) -> None:
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.scale = scale
        self.device = device
        self.dtype = dtype
        self.kv_tile_pages = kv_tile_pages
        self.max_num_reqs = max_num_reqs or 0
        self.max_num_batched_tokens = max_num_batched_tokens or 0
        self.max_model_len = max_model_len or 0
        self._warned = False
        # H21: per-bucket split policy ("1:16,2:32,4:64,8:128") — the wave-quantised tax makes
        # a single global ktp either starve small batches or tax concurrency (E61); buckets not
        # listed fall back to the global ``kv_tile_pages``.
        self._ktp_policy: Dict[int, int] = _parse_ktp_policy(
            os.environ.get("TILEINFER_KV_TILE_PAGES_POLICY", "")
        )
        if any(ktp > 0 for ktp in self._ktp_policy.values()):
            # The policy entry *is* the split opt-in (two envs for one intent is a footgun:
            # the backend's gate would fail every split plan and silently fall back to FIA).
            os.environ.setdefault("TILEINFER_ALLOW_SPLIT_KV", "1")
        # Metrics view for the in-service probe: what this process will actually build.
        type(self)._policy_view = {"global": int(kv_tile_pages), **self._ktp_policy}

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

    def ktp_for(self, bucket: int) -> int:
        """This bucket's split factor: the H21 policy entry, else the global ktp."""
        ktp = self._ktp_policy.get(bucket)
        return self.kv_tile_pages if ktp is None else ktp

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

    def _plan_key(self, bucket: int, width: int, block_size: int, pool_cap: int) -> tuple:
        """Plan cache key.

        ``width`` is the request's *actual* block-table width in kernel pages (a constant buffer
        width in vLLM: ``ceil(max_model_len / engine_block)`` sub-pages), so one plan covers a
        whole run and the plan-owned staging buffers can be sized to it exactly.

        ``pool_cap`` is the kernel ABI's *declared* capacity (:func:`declared_pool_cap`), not the
        exact pool: vLLM derives the pool from a memory snapshot that drifts run to run, and an
        exact-pool key silently invalidated out-of-process pre-warm (L2).

        The bucket's *own* ktp (H21 policy) is part of the key: buckets with different split
        factors must not share a plan (their schedules and staging buffers differ).
        """
        return (self._config_key, int(self.ktp_for(bucket)), bucket, width, block_size, pool_cap)

    def prewarm(self, width: int, block_size: int, key_cache: torch.Tensor) -> None:
        """Synchronously build a plan for *every* bucket up to ``max_num_reqs``.

        Called from the first non-captured decode step — vLLM's pre-capture warmup run — so the
        whole plan set exists before ACLGraph capture begins.  A compile takes ~100 s cold and
        hits the TileLang disk cache when warm, which is acceptable exactly once at startup.

        ``width`` counts kernel pages (engine table width × sub-pages).  Plans are built on
        kernel pages (see KERNEL_PAGE_SIZE) and the pool is presented as its 128-token view.
        """
        from tileinfer.kernels.attention.paged_decode_ascend950 import declared_pool_cap

        cls = TileInferDecodeAttention
        pool_eff = declared_pool_cap(int(key_cache.shape[0]))
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
                "before graph capture%s",
                built,
                limit,
                width,
                f"; split policy {self._ktp_policy}" if self._ktp_policy else "",
            )
        built_prefill = self.prewarm_prefill(block_size, key_cache)
        cls._stats["prefill_variants"] = len(PrefillShapeManager._ready)
        if built_prefill >= 0:
            logger.info(
                "TileInfer: prefill variant ladder prewarm finished (%d newly compiled, "
                "%d ready, cap %d)",
                built_prefill,
                len(PrefillShapeManager._ready),
                PrefillShapeManager._cap,
            )

    def prewarm_prefill(self, block_size: int, key_cache: torch.Tensor) -> int:
        """Synchronously compile the canonical prefill variant ladder (P3).

        The canonical variant of a step is ``(bucket(max_num_reqs) + 1 trash rows,
        next-pow2 tiles)``; the tile ladder top comes from the scheduler token budget
        (``max_num_batched_tokens``, which bounds every prefill/chunk step's tile count), so the
        ladder covers *every* prefill shape the engine can schedule — a first-seen prompt length
        is served by a ready variant instead of triggering the ~110 s lazy compile (D22).  The
        ladder is |log2(top)| + 1 entries, far below the variant cap.  Returns how many variants
        were newly compiled here (0 when all were ready/covered elsewhere; -1 when the
        configuration is unknown and prewarm was skipped).
        """
        limit = int(self.max_num_reqs) if self.max_num_reqs else BATCH_BUCKETS[-1]
        group = self.group_size
        # A mixed step carries up to ``budget`` tokens: a chunk of ``budget - n`` prefill tokens
        # (group rows each → /128 per tile) plus ``n`` one-row decode appends (one tile each).
        budget = int(self.max_num_batched_tokens) or 8192
        tiles_top = _next_pow2(-(-budget * group // 128) + limit)
        tiles_top = min(tiles_top, PREFILL_MAX_TILES)
        kernel_batch = _bucket_for(limit) + 1
        from tileinfer.kernels.attention.paged_decode_ascend950 import declared_pool_cap

        sub_view = _pool_sub_view(key_cache, _sub_pages(block_size))
        pool_pages = declared_pool_cap(int(sub_view.shape[0]))
        # Split-prefill ladder top: a chunk step's tile count is the unsplit count times the
        # per-request split factor at the model's longest context (the +1 keeps the runtime
        # estimate an upper bound).  Without this, long-context steps ask for tiles past the
        # ladder top and race a ~3.5 min lazy compile on the serving path (the D56-era rerun:
        # tiles=512 missed a top of 256 → 3760 fallback steps).
        tiles_top_split = tiles_top
        if _PREFILL_KV_TILE_PAGES > 0:
            factor = 1
            if self.max_model_len:
                max_pages = -(-int(self.max_model_len) // KERNEL_PAGE_SIZE)
                factor = -(-max_pages // _PREFILL_KV_TILE_PAGES) + 1
            tiles_top_split = min(_next_pow2(tiles_top * factor), PREFILL_MAX_TILES)
        return self._prewarm_prefill_ladder(kernel_batch, tiles_top, tiles_top_split, pool_pages)

    def _prewarm_prefill_ladder(
        self, kernel_batch: int, tiles_top: int, tiles_top_split: int, pool_pages
    ) -> int:
        from tileinfer.kernels.attention.paged_decode_ascend950 import SPLIT_MS_COMPILE

        dtype_name = _TORCH_DTYPE_NAME.get(self.dtype)
        if dtype_name is None:
            return -1
        built = 0
        tiles = 1
        while tiles <= max(tiles_top, tiles_top_split):
            specs = [
                (
                    kernel_batch,
                    self.num_kv_heads,
                    self.group_size,
                    self.head_size,
                    KERNEL_PAGE_SIZE,
                    pool_pages,
                    tiles,
                    128,
                    dtype_name,
                )
            ]
            if _PREFILL_KV_TILE_PAGES > 0 and tiles <= tiles_top_split:
                # The split-prefill canonical variant: the runtime always launches
                # ``(num_tiles=kernel_tiles, num_slots=kernel_tiles+1, max_splits=
                # SPLIT_MS_COMPILE)`` (each schedule half is dead-padded to kernel_tiles),
                # so the ladder compiles exactly those keys — per-step data-shaped keys
                # racing a background compile is what fell back 4400/4400 steps to FIA in
                # the first engine A/B.  The plain ladder stays for the pathological
                # split-blowup steps (ktp forced 0 there).
                specs.append(specs[0] + (SPLIT_MS_COMPILE,))
            for spec in specs:
                if spec not in PrefillShapeManager._ready:
                    if PrefillShapeManager.prewarm_sync(spec):
                        built += 1
            tiles *= 2
        return built

    def plan_if_ready(self, bucket: int, max_pages: int, block_size: int, key_cache: torch.Tensor):
        """Return the plan for this shape, or ``None`` while it is being compiled.

        Kept as the *fallback* path for a step that needs a shape ``prewarm`` did not cover (for
        example a wider table after a config change): such a step compiles in the background and
        serves through FIA until the plan shows up.  The graph-capture path never comes here.
        """
        sub = _sub_pages(block_size)
        width = max_pages * sub
        from tileinfer.kernels.attention.paged_decode_ascend950 import declared_pool_cap

        pool_eff = declared_pool_cap(int(key_cache.shape[0]))
        key = self._plan_key(bucket, width, block_size, pool_eff)
        plan = self._plans.get(key)
        if plan is not None or key in self._failed:
            return plan

        with self._lock:
            if key in self._building:
                return None
            self._building.add(key)

        logger.info(
            "TileInfer: compiling the decode kernel for bucket=%d pages<=%d block_size=%d "
            "declared_pool_cap=%d pages in the background; serving through FIA until it is ready",
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
        # refreshed every step by the caller.  With KV splitting the plan must be *full*: graph
        # capture pins one plan per bucket, while replay-time sequences grow one page per step, so
        # the schedule is built as if every request spanned the whole table width (max_splits =
        # ceil(width / kv_tile_pages)) and the kernel clamps each tile to the pages that actually
        # exist at run time (see paged_decode_split).  Plans are always built on kernel pages
        # (128 tokens) and the pool is presented as its 128-token sub-page view.
        # (128 tokens) and the pool is presented as its 128-token sub-page view.  NOTE: the
        # page-table lens are in *tokens* (`PageTable` derives ceil(len/128) pages), so the
        # full-plan fill is width pages × the kernel page size.
        sub = _sub_pages(block_size)
        pool = _pool_sub_view(key_cache, sub)
        ktp = self.ktp_for(bucket)
        dummy_len = width * KERNEL_PAGE_SIZE if ktp > 0 else 1
        dummy_lens = torch.full((bucket,), dummy_len, dtype=torch.int32, device=self.device)
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
            kv_tile_pages=ktp,
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
        step_obj: Any = None,
    ) -> torch.Tensor:
        """One decode step: ``query`` is ``[num_reqs, num_heads, head_size]`` (the layout vLLM-Ascend
        passes here), writes into ``output`` with the same layout.

        ``step_obj`` is the step's attention-metadata object (vLLM passes the same instance to
        every layer of a step and builds a fresh one per step); when provided, the shared
        metadata-refresh chain runs only for the first layer of the step (B3 dedup).
        """
        cls = TileInferDecodeAttention
        capturing = torch.npu.is_current_stream_capturing()
        num_reqs = int(seq_lens.numel())
        bucket = _bucket_for(num_reqs)
        sub = _sub_pages(block_size)
        width = int(block_tables.shape[1]) * sub
        from tileinfer.kernels.attention.paged_decode_ascend950 import declared_pool_cap

        pool_eff = declared_pool_cap(int(key_cache.shape[0]))
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
        # Buffers the kernel call needs regardless of whether this layer performed the
        # step's refresh (plan-owned, stable addresses).
        lpl = state["last_page_len_buf"]
        idx_buf: torch.Tensor = state["kv_indices_buf"]
        # Engine pages -> kernel pages: expand the page table to 128-token sub-page ids (zero-copy;
        # identity for block_size == 128) and stage everything into plan-owned buffers.  Every op
        # below is shape-static and device-side, so the whole step is capturable.
        step_id = id(step_obj) if step_obj is not None else None
        if step_id is None or (step_id, key) != cls._step_guard:
            # First layer of this step (or a library-level caller without step identity):
            # refresh the shared metadata.  Layers 2..N skip - the values are identical.
            if step_id is not None:
                cls._step_guard = (step_id, key)
                cls._step_obj = step_obj  # strong ref: keeps the id unique while it guards
            cls._stats["refreshes"] = cls._stats.get("refreshes", 0) + 1
            if cls._stats["refreshes"] <= 12:
                logger.info(
                    "TileInfer step refresh #%d: metadata id=%s key_bucket=%d",
                    cls._stats["refreshes"],
                    step_id,
                    bucket,
                )
            table_exp = _expand_page_table(block_tables[:num_reqs], sub)
            table_buf: torch.Tensor = state["table_buf"]
            table_buf[:num_reqs].copy_(table_exp)
            # Zero the padding rows' columns so a padding row (pages=1 from its stale len=1) can
            # only ever reach page 0 - never a leftover id from some earlier longer batch.
            table_buf[num_reqs:].zero_()
            lens_buf: torch.Tensor = state["lens_buf"]
            lens_buf[:num_reqs].copy_(seq_lens[:num_reqs].to(torch.int32))
            # last-page-len from the padded lens, in place: ((lens - 1) % page) + 1
            torch.sub(lens_buf, 1, out=lpl)
            lpl.remainder_(KERNEL_PAGE_SIZE)
            lpl.add_(1)
            # Rebuild the ragged triple on the device (H4 fix): fixed shapes, in-place, capturable.
            # pages_b = ceil(lens_b / 128); kv_indptr = exclusive cumsum; request b's pages end up
            # at ``kv_indices[indptr[b] : indptr[b] + pages_b]`` - exactly the span the kernel
            # derives from indptr diffs.  Invalid columns are parked in the trash slot, outside
            # every span.
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
                    # P3: token budget of one prefill/chunk step — bounds the prefill variant
                    # ladder's tile top (see prewarm_prefill).
                    max_batched = int(self.vllm_config.scheduler_config.max_num_batched_tokens)
                except Exception:  # noqa: BLE001 - fall back to the largest bucket
                    max_num_reqs = 0
                    max_batched = 0
                try:
                    max_model_len = int(self.vllm_config.model_config.max_model_len)
                except Exception:  # noqa: BLE001 - tests without a model config
                    max_model_len = 0
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
                    max_num_batched_tokens=int(max_batched or 0),
                    max_model_len=int(max_model_len or 0),
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

            state = attn_metadata.attn_state
            if state in (
                AscendAttentionState.PrefillNoCache,
                AscendAttentionState.PrefillCacheHit,
                AscendAttentionState.ChunkedPrefill,
            ):
                # B1: prefill/append through the TileLang prefill kernel; every refusal
                # (op switch off, shape not compiled yet, layout drift, exception) falls
                # back to the parent FIA implementation for this step.
                if not ops_enabled("prefill"):
                    return super().forward_impl(query, key, value, kv_cache, attn_metadata, output)
                out = self._forward_prefill_impl(query, attn_metadata, output)
                if out is None:
                    return super().forward_impl(query, key, value, kv_cache, attn_metadata, output)
                return out

            if state != AscendAttentionState.DecodeOnly:
                return super().forward_impl(query, key, value, kv_cache, attn_metadata, output)

            if not ops_enabled("decode"):
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
                    # Step identity for the refresh-once-per-step dedup (B3): vLLM hands the
                    # same metadata instance to every attention layer of a step.
                    step_obj=attn_metadata,
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

        def _forward_prefill_impl(self, query, attn_metadata, output):
            """Prefill/append through TileInfer; False/None means FIA (never raises)."""
            try:
                # Prefill always runs eager; if a capture is somehow active, do not touch it.
                if torch.npu.is_current_stream_capturing():
                    return None
                device_lens = getattr(attn_metadata, "seq_lens_device", None)
                if device_lens is None:
                    device_lens = attn_metadata.seq_lens
                # P3: the batch half of the canonical variant must be prewarm-stable, so it is
                # derived from the scheduler's max_num_seqs (the same number prewarm uses).
                try:
                    batch_cap = int(self.vllm_config.scheduler_config.max_num_seqs)
                except Exception:  # noqa: BLE001 - library callers without a config
                    batch_cap = 0
                served = prefill_step(
                    query=query,
                    key_cache=self.key_cache,
                    value_cache=self.value_cache,
                    block_tables=attn_metadata.block_tables,
                    seq_lens=device_lens,
                    query_start_loc=attn_metadata.query_start_loc,
                    block_size=int(self.key_cache.shape[1]),
                    output=output,
                    num_heads=self.num_heads,
                    num_kv_heads=self.num_kv_heads,
                    head_size=self.head_size,
                    scale=self.scale,
                    batch_cap=batch_cap,
                    step_obj=attn_metadata,
                )
                if not served:
                    TileInferDecodeAttention._stats["prefill_fallbacks"] = (
                        TileInferDecodeAttention._stats.get("prefill_fallbacks", 0) + 1
                    )
                    return None
                return output
            except Exception:  # noqa: BLE001 - a serving process must survive
                logger.exception("TileInfer prefill failed; falling back to FIA for this step")
                TileInferDecodeAttention._stats["prefill_exceptions"] = (
                    TileInferDecodeAttention._stats.get("prefill_exceptions", 0) + 1
                )
                return None

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


_GDN_GATING_INSTALLED = False


def install_gdn_gating() -> bool:
    """Take over vLLM-Ascend's GDN gating (BACKLOG M-GDN, ``TILEINFER_OPS=gdn``).

    On A5 the in-service path is ``vllm_ascend.device.device_op.A5DeviceAdaptor.
    fused_gdn_gating`` — a triton patch whose single kernel costs ~2.6 ms of device time
    per prefill call at T=128 (E20: 78 ms = 11% of the prefill step) for what is an
    elementwise ``(T, H)`` softplus/sigmoid.  We replace it with the torch-composed
    implementation in :mod:`tileinfer.kernels.gdn` (numerically identical to the triton
    kernel, capture-safe fixed-shape ops).  The dispatch honours ``ops_enabled("gdn")``
    at every call, so flipping ``TILEINFER_OPS`` takes the original back without a code
    change, and each invocation bumps the serving-probe ``gdn_gating_calls`` counter.
    """
    global _GDN_GATING_INSTALLED
    if _GDN_GATING_INSTALLED:
        return True
    from vllm_ascend.device import device_op as _device_op

    from tileinfer.kernels.gdn import fused_gdn_gating as _gdn_impl

    adaptor = _device_op.A5DeviceAdaptor
    if getattr(adaptor, "_tileinfer_gdn_installed", False):
        _GDN_GATING_INSTALLED = True
        return True

    cls_stats = TileInferDecodeAttention._stats

    def _fused_gdn_gating(A_log, a, b, dt_bias):
        if ops_enabled("gdn"):
            g, beta = _gdn_impl(A_log, a, b, dt_bias)
            cls_stats["gdn_gating_calls"] = cls_stats.get("gdn_gating_calls", 0) + 1
            return g, beta
        return _original(A_log, a, b, dt_bias)

    _original = adaptor.fused_gdn_gating
    adaptor.fused_gdn_gating = staticmethod(_fused_gdn_gating)
    adaptor._tileinfer_gdn_installed = True
    _GDN_GATING_INSTALLED = True
    logger.info("TileInfer GDN gating dispatch installed (TILEINFER_OPS governs per call)")
    return True


_GDN_WU_INSTALLED = False

#: 长度自适应分流阈值(P1/H11):prefill 总 token 数 ≤ 阈值才走融合核,否则回退原 triton 链。
#: E26/E26b 微基准(在役形状 HK16/HV32/D128,npu Event 预热后中位):T=128 融合 0.527ms vs
#: 基线 0.724ms(0.73x,唯一融合占优区间);T=256 起反向(1.22x/2.40x@512/4.41x@1024/14.4x@8192,
#: 机理=E24 的标量 Neumann/合并积链随 chunk 数线性劣化,基线近常数的启动主导)。交叉点严格
#: 落在 (128, 256) 区间,保守取上界 128。可用 ``TILEINFER_GDNWU_MAX_TOKENS`` 覆盖。
_GDN_WU_MAX_TOKENS_DEFAULT = 128


def _gdnwu_max_tokens() -> int:
    """Length-adaptive dispatch threshold for the chunk-wu fused kernel (P1/H11)."""
    raw = os.environ.get("TILEINFER_GDNWU_MAX_TOKENS")
    if raw is None or not raw.strip():
        return _GDN_WU_MAX_TOKENS_DEFAULT
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning("invalid TILEINFER_GDNWU_MAX_TOKENS=%r, using %d", raw,
                       _GDN_WU_MAX_TOKENS_DEFAULT)
        return _GDN_WU_MAX_TOKENS_DEFAULT


def _build_gdn_chunk_meta(cu_seqlens_host, device, hv_heads: int) -> "object":
    """库级/测试路径的 prebuilt_meta 现场构建(serving 恒由在役 builder 提供)。

    字段与 ``vllm_ascend.ops.gdn_attn_builder._build_non_spec_chunked_prefill_metadata``
    同构:chunk 64 / large block 1216 / cumsum 块 = next_pow2(2^18 // (HV·64))——最后一项
    必须与 ``chunk_local_cumsum`` 内部按同一公式推导的 OPTIM_BLOCK_SIZE 一致,否则 grid 数
    与 128-token 块寻址错位会直接设备越界(507015)。cu_seqlens.tolist() 的宿主同步仅发生
    在这条非在役路径。没有它,回退原链在 meta=None 时 fwd_o 会 507015 设备崩溃(E22 坑位:
    在役恒带 host tuple,None 分支从未被在役走过)。
    """
    import triton
    from vllm_ascend.ops.gdn_attn_builder import GDNChunkedPrefillMetadata
    from vllm_ascend.ops.triton.fla.utils import (
        prepare_chunk_indices,
        prepare_chunk_offsets,
        prepare_final_chunk_indices,
        prepare_update_chunk_offsets,
    )

    cu_cpu = torch.tensor([int(x) for x in cu_seqlens_host], dtype=torch.int64)
    ci = prepare_chunk_indices(cu_cpu, 64)
    cumsum_block = triton.next_power_of_2((2**18) // (max(1, hv_heads) * 64))
    return GDNChunkedPrefillMetadata(
        cu_seqlens_host=tuple(int(x) for x in cu_seqlens_host),
        chunk_indices_chunk64_host=tuple(ci.to(torch.int64).reshape(-1).tolist()),
        chunk_indices_chunk64=ci.to(device, non_blocking=True),
        chunk_offsets_chunk64=prepare_chunk_offsets(cu_cpu, 64).to(device, non_blocking=True),
        update_chunk_offsets_chunk64=prepare_update_chunk_offsets(cu_cpu, 64).to(device, non_blocking=True),
        final_chunk_indices_chunk64=prepare_final_chunk_indices(cu_cpu, 64).to(device, non_blocking=True),
        chunk_indices_large_block=prepare_chunk_indices(cu_cpu, 1216).to(device, non_blocking=True),
        block_indices_cumsum=prepare_chunk_indices(cu_cpu, cumsum_block).to(device, non_blocking=True),
        num_decodes=0,
    )


def install_gdn_chunk_wu() -> bool:
    """接管 vLLM-Ascend GDN prefill chunk 链的中段两核(BACKLOG M-GDN 9c,H10;阶段 3 P1)。

    在役链(E20/E22):cumsum → kkt → **solve_tril + recompute_w_u**(四核 triton 中的两大
    单核)→ AscendC fwd_h → fwd_o。本钩子把 ``chunk_gated_delta_rule_fwd`` 替换为 TileInfer
    版本:中段换成 tileinfer 的融合内核 ``gdn_chunk_wu``(块逆 + 乘 k/v 一步完成),链首尾
    (cumsum/kkt/fwd_h/fwd_o 与 AscendC staging)保持原实现不变。

    **长度自适应分流(P1/H11,E26/E26b)**:仅当 prefill 总 token 数 ``q.shape[1]`` ≤ 阈值
    (缺省 128,``TILEINFER_GDNWU_MAX_TOKENS`` 可覆盖)才走融合核——该区间是微基准扫描中
    融合唯一占优的形状带;更长上下文(E26:T256 起 1.22x、T8192 14.4x)回退原 triton 链,
    保证长上下文零劣化。判定只读宿主侧 shape,零设备同步,图捕获安全。PCP(world_size>1)
    与 cu_seqlens=None(死代码路径,见 E22)时回退原实现;``TILEINFER_OPS`` 每步读取
    ``gdnwu`` 项,秒级回退;计数(gdn_wu_calls/gdn_wu_long_skips/gdn_wu_fallbacks)入在役探针。
    """
    global _GDN_WU_INSTALLED
    if _GDN_WU_INSTALLED:
        return True
    import vllm_ascend.ops.triton.fla.chunk as _chunk_mod
    from vllm.distributed import get_pcp_group

    from tileinfer.kernels.gdn.chunk_wu_ascend950 import gdn_chunk_wu

    if getattr(_chunk_mod, "_tileinfer_gdn_wu_installed", False):
        _GDN_WU_INSTALLED = True
        return True

    cls_stats = TileInferDecodeAttention._stats
    _original_fwd = _chunk_mod.chunk_gated_delta_rule_fwd
    _cumsum = _chunk_mod.chunk_local_cumsum
    _kkt = _chunk_mod.chunk_scaled_dot_kkt_fwd

    def _ti_chunk_gated_delta_rule_fwd(q, k, v, g, beta, scale, initial_state, output_final_state,
                                        cu_seqlens=None, prebuilt_meta=None):
        # 长度自适应分流(P1/H11):短上下文走融合核,长上下文回退原链(E26/E26b 阈值依据)。
        # q 为 varlen 布局 [1, T_total, H, D],shape 读取纯宿主侧、零同步。
        # 库级/测试调用(serving 恒带在役 builder 的 prebuilt_meta)现场补一份同构 meta:
        # 原链在 meta=None 时 fwd_o 会 507015 设备崩溃(E22 坑位),构建仅发生在宿主同步
        # 一次的库级路径上,在役路径不受影响。
        enabled = ops_enabled("gdnwu")
        if enabled and cu_seqlens is not None and prebuilt_meta is None:
            prebuilt_meta = _build_gdn_chunk_meta(
                tuple(int(x) for x in cu_seqlens.to(torch.int64).tolist()),
                cu_seqlens.device, g.shape[2])
        if (not enabled) or cu_seqlens is None or get_pcp_group().world_size > 1 \
                or q.shape[1] > _gdnwu_max_tokens():
            if enabled and cu_seqlens is not None and get_pcp_group().world_size <= 1:
                # 开关开着但形状超阈值:计一次"长度分流跳过"(在役探针可见分流真的发生)
                cls_stats["gdn_wu_long_skips"] = cls_stats.get("gdn_wu_long_skips", 0) + 1
            return _original_fwd(q, k, v, g, beta, scale, initial_state, output_final_state,
                                 cu_seqlens=cu_seqlens, prebuilt_meta=prebuilt_meta)
        try:
            meta = prebuilt_meta
            block_indices_cumsum = meta.block_indices_cumsum
            cu_seqlens_host = meta.cu_seqlens_host
            ci64 = meta.chunk_indices_chunk64
            ci_host = meta.chunk_indices_chunk64_host
            chunk_indices_large_block = meta.chunk_indices_large_block
            chunk_indices_bt = meta.chunk_indices_chunk64

            # ---- 链首:与原实现一致 ----
            g64 = _cumsum(g, chunk_size=64, cu_seqlens=cu_seqlens, block_indices=block_indices_cumsum)
            a_kkt = _kkt(k=k, beta=beta, g_cumsum=g64, cu_seqlens=cu_seqlens,
                         chunk_indices=chunk_indices_bt, output_dtype=torch.float32)
            # ---- 中段:融合内核(solve_tril + recompute_w_u)----
            # 服务端 cu_seqlens / meta.chunk_indices_chunk64 均为 int32(引擎 metadata),
            # 内核签名要求 int64 → 统一转换;triton 链首两核保持原 dtype(与原实现一致)
            cu64 = cu_seqlens.to(torch.int64)
            ci64_64 = ci64.to(torch.int64) if ci64.dtype != torch.int64 else ci64
            w, u = gdn_chunk_wu(a_kkt, k, v, beta, g64, cu64, ci64_64)
            cls_stats["gdn_wu_calls"] = cls_stats.get("gdn_wu_calls", 0) + 1

            # ---- 链尾:与原实现一致(AscendC staging + fwd_h + fwd_o)----
            k_ascendc = k.to(torch.bfloat16).transpose(1, 2).contiguous()
            w_ascendc = w.to(torch.bfloat16).transpose(1, 2).contiguous()
            u_ascendc = u.to(torch.bfloat16).transpose(1, 2).contiguous()
            g_ascendc = g64.transpose(1, 2).contiguous()
            q_ascendc = q.to(torch.bfloat16).transpose(1, 2).contiguous()

            if cu_seqlens_host is None:
                cu_seqlens_host = tuple(cu64.tolist())
            if ci_host is None:
                ci_host = tuple(ci64.flatten().tolist())
            from vllm_ascend.ops.gdn_attn_builder import _compact_empty_segments

            cu_seqlens_kern, initial_state_kern, keep_meta = _compact_empty_segments(
                cu_seqlens_host, initial_state,
                device=initial_state.device if initial_state is not None else None,
            )
            h, v_new, final_state = torch.ops._C_ascend.chunk_gated_delta_rule_fwd_h(
                k_ascendc, w_ascendc, u_ascendc, g=g_ascendc, gk=None,
                initial_state=initial_state_kern, output_final_state=True, chunk_size=64,
                save_new_value=True, cu_seqlens=cu_seqlens_kern, chunk_indices=ci_host,
                use_exp2=False, transpose_state_layout=False,
            )
            if keep_meta is not None:
                _fs_full = initial_state.clone()
                _fs_full[keep_meta] = final_state
                final_state = _fs_full

            o_ascendc = torch.ops._C_ascend.chunk_fwd_o(
                q_ascendc, k_ascendc, v_new, h, scale, g=g_ascendc, g_gamma=None,
                cu_seqlens=cu_seqlens_host, chunk_indices=ci_host, chunk_size=64,
                transpose_state_layout=False,
            )
            o = o_ascendc.to(torch.bfloat16).transpose(1, 2).contiguous()
            v_new = v_new.to(torch.bfloat16).transpose(1, 2).contiguous()
            h = h.to(torch.bfloat16).transpose(1, 2).contiguous()
            from vllm.model_executor.layers.fla.ops.utils import SUPPRESS_LEVEL

            if SUPPRESS_LEVEL < 3:
                return g64, o, None, final_state, None, None, None
            return g64, o, None, final_state, w, h, v_new
        except Exception as exc:  # noqa: BLE001 - 接管失败不得打断推理(与 B5b 同语义)
            cls_stats["gdn_wu_fallbacks"] = cls_stats.get("gdn_wu_fallbacks", 0) + 1
            logger.warning("TileInfer gdn_chunk_wu fallback to vllm-ascend chain: %s", exc)
            return _original_fwd(q, k, v, g, beta, scale, initial_state, output_final_state,
                                 cu_seqlens=cu_seqlens, prebuilt_meta=prebuilt_meta)

    _chunk_mod.chunk_gated_delta_rule_fwd = _ti_chunk_gated_delta_rule_fwd
    _chunk_mod._tileinfer_gdn_wu_installed = True
    _GDN_WU_INSTALLED = True
    logger.info("TileInfer GDN chunk wu dispatch installed (TILEINFER_OPS governs per call)")
    return True


_GDN_PROJ_INSTALLED = False


def install_gdn_proj_fusion() -> bool:
    """GDN in_proj_ba→in_proj_qkvz 权重融合(H32/S-GDN 投影族,``TILEINFER_OPS=gdnproj``)。

    decode 每步每 GDN 层的 ``in_proj_ba`` 是一次 N=64/0.26MB 的纯启动延迟税 GEMM
    (in-engine 3.8µs/次,30 层 ≈ 0.67% TPOT,E97/E100)。本钩子在
    ``QwenGatedDeltaNetAttention.forward_cuda`` 类级包装:接管时把 ba 权重 cat 进
    qkvz 权重(首次 forward 懒构建,预热期、图捕获前)成单 GEMM,输出零拷贝 strided
    切片喂原 ``fix_query_key_value_ordering``(E101:view 纯维度拆分对 strided 视图
    合法);数值与分离路径逐位等价(E100 maxdiff=0.0;上游同构先例 vLLM #41457)。

    守卫(TP=1/无量化/interleaved 布局/普通权重)不满足或运行期异常均回退原实现;
    ``ops_enabled("gdnproj")`` 每步读取,翻转 ``TILEINFER_OPS`` + 重启即回退。
    须在引擎构建前安装(install() 时序,同 sampler 先例)。
    """
    global _GDN_PROJ_INSTALLED
    if _GDN_PROJ_INSTALLED:
        return True
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention as _gdn_cls,
    )

    if getattr(_gdn_cls, "_tileinfer_gdnproj_installed", False):
        _GDN_PROJ_INSTALLED = True
        return True

    from tileinfer.kernels.gdn.proj_fusion import fused_forward_cuda

    cls_stats = TileInferDecodeAttention._stats
    _original = _gdn_cls.forward_cuda

    def _forward_cuda(self, hidden_states, output):
        if ops_enabled("gdnproj"):
            try:
                if fused_forward_cuda(self, hidden_states, output):
                    cls_stats["gdnproj_calls"] = cls_stats.get("gdnproj_calls", 0) + 1
                    return
                cls_stats["gdnproj_fallbacks"] = (
                    cls_stats.get("gdnproj_fallbacks", 0) + 1
                )
            except Exception:  # noqa: BLE001 - 融合路径任何异常都回退原实现
                cls_stats["gdnproj_fallbacks"] = (
                    cls_stats.get("gdnproj_fallbacks", 0) + 1
                )
        return _original(self, hidden_states, output)

    _gdn_cls.forward_cuda = _forward_cuda
    _gdn_cls._tileinfer_gdnproj_installed = True
    _GDN_PROJ_INSTALLED = True
    logger.info("TileInfer GDN in_proj fusion dispatch installed (H32, TILEINFER_OPS governs)")
    return True


_GDN_REORDER_INSTALLED = False


def install_gdn_proj_reorder() -> bool:
    """GDN fused(qkvz+ba)权重 load 期行重排(H34/S-GDN 投影族,``TILEINFER_OPS=gdnreorder``)。

    H32 融合后 forward 仍走 ``fix_query_key_value_ordering``(view+split+reshape)→
    3×rearrange → cat,E105 实证 Cat [32768+32768+65536] ×7650 定距命中(真实拷贝)。
    本钩子把 fused 权重行序在懒构建期重排为消费终序 ``[q|k|v|z|b|a]``,单 GEMM 输出
    即终态,forward 仅零拷贝切片(E102:vLLM #55260 行重排 bit-identical 先例;设备
    tiling 差异由 TOL 断言覆盖,与 H32 同口径)。包装叠于 gdnproj 之外:接管成功时
    gdnproj 包装不触达(计数可归因),懒构建失败缓存属性独立命名,与 H32 互不可见。

    守卫(继承 H32 build 守卫/布局校验)不满足或运行期异常均回退原实现;
    ``ops_enabled("gdnreorder")`` 每步读取,翻转 ``TILEINFER_OPS`` + 重启即回退。
    须在引擎构建前安装(install() 时序,同 gdnproj 先例)。
    """
    global _GDN_REORDER_INSTALLED
    if _GDN_REORDER_INSTALLED:
        return True
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention as _gdn_cls,
    )

    if getattr(_gdn_cls, "_tileinfer_gdnreorder_installed", False):
        _GDN_REORDER_INSTALLED = True
        return True

    from tileinfer.kernels.gdn.reorder import fused_forward_reordered

    cls_stats = TileInferDecodeAttention._stats
    # _original 是安装时刻的 forward_cuda:gdnproj 先装时即 H32 包装(回落链
    # H34→H32→原实现);单独安装时即原实现。
    _original = _gdn_cls.forward_cuda

    def _forward_cuda(self, hidden_states, output):
        if ops_enabled("gdnreorder"):
            try:
                if fused_forward_reordered(self, hidden_states, output):
                    cls_stats["gdnreorder_calls"] = cls_stats.get("gdnreorder_calls", 0) + 1
                    return
                cls_stats["gdnreorder_fallbacks"] = (
                    cls_stats.get("gdnreorder_fallbacks", 0) + 1
                )
            except Exception:  # noqa: BLE001 - 终序路径任何异常都回退原实现
                cls_stats["gdnreorder_fallbacks"] = (
                    cls_stats.get("gdnreorder_fallbacks", 0) + 1
                )
        return _original(self, hidden_states, output)

    _gdn_cls.forward_cuda = _forward_cuda
    _gdn_cls._tileinfer_gdnreorder_installed = True
    _GDN_REORDER_INSTALLED = True
    logger.info(
        "TileInfer GDN in_proj reorder dispatch installed (H34, TILEINFER_OPS governs)"
    )
    return True


_GDN_SMALL_INSTALLED = False


def install_gdn_small_fusion() -> bool:
    """GDN 小核链削减(H35/S-GDN 域,``TILEINFER_OPS=gdnsmall``)。

    三个独立包装(E105 decode 窗定距命中,#19033 蓝本):
    1. ``QwenGatedDeltaNetAttention.rearrange_mixed_qkv`` cat-free 覆写(活类,
       与 H32 同一 patch 目标;split+reshape 视图替代 reshape(-1)+cat 材料化);
    2. ``torch.ops._C_ascend.npu_causal_conv1d_custom`` 权重参数连续缓存包装,
       消每层每步 aclnnContiguous_Transpose(storage 弱引用校验防陈旧);
    3. ``vllm_ascend.ops.gdn.l2norm_fwd`` 配对融合包装,q/k 为同一连续缓冲相邻区
       (seq_len==1)时一次内核,与两次调用数值等价(逐行归一与分组无关)。

    全部按 ``ops_enabled("gdnsmall")`` 每步读取,异常回退原实现;三处全部就绪才
    安装(部分失败整体放弃,不产生半接管态);须在引擎构建前安装(install() 时序,
    同 gdnproj 先例)。
    """
    global _GDN_SMALL_INSTALLED
    if _GDN_SMALL_INSTALLED:
        return True
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention as _gdn_cls,
    )

    if getattr(_gdn_cls, "_tileinfer_gdnsmall_installed", False):
        _GDN_SMALL_INSTALLED = True
        return True

    if "rearrange_mixed_qkv" not in _gdn_cls.__dict__:
        logger.warning(
            "TileInfer GDN small fusion not installed: rearrange_mixed_qkv not on "
            "QwenGatedDeltaNetAttention (upstream layout changed?)"
        )
        return False
    _ns = getattr(torch.ops, "_C_ascend", None)
    _orig_conv = getattr(_ns, "npu_causal_conv1d_custom", None) if _ns is not None else None
    import vllm_ascend.ops.gdn as _gdn_mod

    _orig_l2 = getattr(_gdn_mod, "l2norm_fwd", None)
    if _orig_conv is None or _orig_l2 is None:
        logger.warning(
            "TileInfer GDN small fusion not installed: npu_causal_conv1d_custom="
            f"{_orig_conv is not None}, l2norm_fwd={_orig_l2 is not None}"
        )
        return False

    from tileinfer.kernels.gdn import small_ops

    cls_stats = TileInferDecodeAttention._stats
    small_ops._stats_sink = lambda name: cls_stats.__setitem__(
        name, cls_stats.get(name, 0) + 1
    )
    small_ops._orig_conv = _orig_conv
    small_ops._orig_l2norm = _orig_l2
    small_ops._conv_ns = _ns
    _gdn_cls._tileinfer_rearrange_orig = _gdn_cls.__dict__["rearrange_mixed_qkv"]
    _gdn_cls.rearrange_mixed_qkv = small_ops.cat_free_rearrange
    setattr(_ns, "npu_causal_conv1d_custom", small_ops.make_conv1d_wrapper(_orig_conv))
    _gdn_mod.l2norm_fwd = small_ops.make_l2norm_wrapper(_orig_l2)
    _gdn_cls._tileinfer_gdnsmall_installed = True
    _GDN_SMALL_INSTALLED = True
    logger.info(
        "TileInfer GDN small fusion dispatch installed (H35, TILEINFER_OPS governs)"
    )
    return True


_SAMPLER_TOPK_INSTALLED = False


def install_sampler_topk() -> bool:
    """接管 vLLM-Ascend 采样阶段的 top-k 阈值选择(BACKLOG 采样条目,``TILEINFER_OPS=sampling``)。

    原生 A5 对整张词表(fp32)做 ``probs.sort``(O(B·V log V))求第 k 大概率;本钩子替换为
    DeepSelect ``topk``(bf16, ``sorted=False``, int32)在 logits 上求第 k 大值作阈值
    (softmax 严格单调,两者等价,保留并列项语义一致)。设备耗时包络内实测降至
    0.12~0.17x,L1 wall 0.87~0.88x,在线 TPOT +16~22%(npu_flash 侧证据,见
    tileinfer_topk_poc/PERFORMANCE.md)。

    与 ``install_gdn_gating`` 同构:替换 ``vllm_ascend.sample.sampler`` 的**模块级**
    ``apply_top_k_top_p`` 绑定;``AscendTopKTopPSampler.__init__`` 在构造时以
    ``self.apply_top_k_top_p = apply_top_k_top_p`` 捕获该模块级名字,故须在**构造采样器
    之前**安装(即 ``install()``,早于引擎/采样器构建)。逐调用 honours
    ``ops_enabled("sampling")``(翻 ``TILEINFER_OPS`` + 重启即秒级回退);包络外
    (词表 < 16384 / batch < 8 / top-p 混合)或异常一律回退原实现,绝不抛出;每次命中
    bump 在役计数 ``sampler_topk_calls``(供 ``start_metrics_dumper`` 探针断言)。

    k 标量按张量对象身份跨步缓存(稳态零 D2H)——这是与 TileInfer「plan/run 分离、
    run 零 host 同步」设计相冲突的唯一必需例外,理由见
    ``tileinfer.kernels.sampling.topk`` 模块注释(图模式 ACLGraph 硬要求)。
    """
    global _SAMPLER_TOPK_INSTALLED
    if _SAMPLER_TOPK_INSTALLED:
        return True
    from vllm_ascend.sample import sampler as _sampler_mod

    from tileinfer.kernels.sampling.topk import (
        apply_top_k_top_p as _impl,
        reset_k_cache as _reset_k_cache,
    )

    if getattr(_sampler_mod, "_tileinfer_sampler_installed", False):
        _SAMPLER_TOPK_INSTALLED = True
        return True

    cls_stats = TileInferDecodeAttention._stats
    _original = _sampler_mod.apply_top_k_top_p

    def _apply_top_k_top_p(logits, k, p, *args, **kwargs):
        # 顺序敏感:调用序对齐本仓 vllm_ascend(sample/sampler.py L162/L175):
        # ``self.apply_top_k_top_p(logits, k, p[, self.top_k])``——**k 在前、p 在后**
        # (npu_flash 原实现 `_npu_flash_apply_top_k_top_p(logits, k, p, top_k)` 同序;
        # vLLM 上游 topk_topp_sampler.py 的 ``apply_top_k_top_p(logits, k, p)`` 亦同序)。
        # 门控/包络判定(零 D2H)全部在实现内取 k 之前完成,回退路径零同步;
        # 实现自检窗口后返回 None 即回退(与原生签名解耦:多余位置参数原样透传)。
        if ops_enabled("sampling"):
            try:
                out = _impl(logits, p, k)
                if out is not None:
                    cls_stats["sampler_topk_calls"] = cls_stats.get("sampler_topk_calls", 0) + 1
                    return out
            except Exception:  # noqa: BLE001 - 接管失败必须静默回退,绝不破坏服务
                cls_stats["sampler_topk_fallbacks"] = cls_stats.get("sampler_topk_fallbacks", 0) + 1
        return _original(logits, k, p, *args, **kwargs)

    _sampler_mod.apply_top_k_top_p = _apply_top_k_top_p
    _sampler_mod._tileinfer_sampler_installed = True
    _reset_k_cache()
    _SAMPLER_TOPK_INSTALLED = True
    logger.info("TileInfer sampler top-k dispatch installed (TILEINFER_OPS governs per call)")
    return True


_GREEDY_FAST_INSTALLED = False


def install_greedy_fast_sampler() -> bool:
    """Sampler greedy 快路径(H36/采样域,``TILEINFER_OPS=greedy``)。

    #19033 d-1 蓝本(E106):全批 greedy(temperature=0,权威口径即此场景)时
    跳过基类 ``Sampler.forward`` 的 bf16→fp32 全词表 cast 与贪心不敏感处理器链,
    直接原 dtype argmax 构造 SamplerOutput。bf16→fp32 精确单调 → argmax 逐位
    一致;九重严格守卫(见 ``tileinfer.kernels.sampling.greedy``)任一不满足
    或异常均回退原实现。0.23.0 的 ``AscendSampler`` 未覆写 forward(原实现即
    基类),类级覆写在首次调用即生效;``ops_enabled("greedy")`` 每步读取,
    翻转 ``TILEINFER_OPS`` + 重启即回退。计数器 greedy_fast_calls/fallbacks
    供在役探针断言。
    """
    global _GREEDY_FAST_INSTALLED
    if _GREEDY_FAST_INSTALLED:
        return True
    from vllm.v1.sample.sampler import Sampler as _base
    from vllm_ascend.sample.sampler import AscendSampler as _cls

    if getattr(_cls, "_tileinfer_greedy_fast_installed", False):
        _GREEDY_FAST_INSTALLED = True
        return True

    from tileinfer.kernels.sampling import greedy as _impl

    cls_stats = TileInferDecodeAttention._stats
    _impl._stats_sink = lambda name: cls_stats.__setitem__(
        name, cls_stats.get(name, 0) + 1
    )
    # 0.23.0 AscendSampler 自身无 forward;若上游将来覆写则取其类属性为原实现
    orig = _cls.__dict__.get("forward", _base.forward)
    _cls._tileinfer_greedy_forward_orig = orig
    _cls.forward = _impl.make_greedy_forward(orig)
    _cls._tileinfer_greedy_fast_installed = True
    _GREEDY_FAST_INSTALLED = True
    logger.info(
        "TileInfer greedy fast sampler dispatch installed (H36, TILEINFER_OPS governs)"
    )
    return True


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
                # D57-①:prefill 侧计数(长上下文分流)独立于类状态,合并进同一视图。
                stats.update(_PREFILL_STATS)
                stats["ready_plans"] = len(getattr(cls, "_plans", {}))
                marker = getattr(cls, "_marker", None)
                stats["kernel_markers"] = int(marker.item()) if marker is not None else -1
                stats["ts"] = time.time()
                # Only a process that actually hosts TileInfer state (the EngineCore) may write:
                # the API server imports the plugin too, and its empty view would keep overwriting
                # the real numbers.  Empty processes stay silent instead.
                # 采样 top-k 接管(BACKLOG 采样条目)纳入非空判据:仅承载 sampling 的进程
                # (无 attention 计数)不得被误判为空进程而永远不写探针。
                if (
                    stats["ready_plans"] == 0
                    and stats["capture_calls"] == 0
                    and stats["eager_calls"] == 0
                    and stats["fallbacks"] == 0
                    and stats["kernel_markers"] < 0
                    and stats.get("sampler_topk_calls", 0) == 0
                    and stats.get("sampler_topk_fallbacks", 0) == 0
                    and stats.get("greedy_fast_calls", 0) == 0
                    and stats.get("greedy_fast_fallbacks", 0) == 0
                    and stats.get("moeprefetch_active", 0) == 0
                    and stats.get("moeprefetch_fallbacks", 0) == 0
                ):
                    time.sleep(_METRICS_INTERVAL_S)
                    continue
                # H21 在役探针:A/B 侧断言"策略确实在役"用(策略缺失/为空即配置未生效)。
                # 放在静默守卫之后:空进程不得因它而开始写文件。
                stats["ktp_policy"] = getattr(cls, "_policy_view", None)
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

    # 显式 ``== "1"`` 判断:``"0"`` 等任意非空字符串曾因真值判断被误认为启用,导致
    # TILEINFER_VLLM=0 的 FIA 对照侧(权威口径 P0 基线)仍加载本插件。只有显式
    # ``TILEINFER_VLLM=1`` 才启用;未设置或任何其他值一律关闭。
    if (__import__("os").environ.get("TILEINFER_VLLM") or "").strip() != "1":
        logger.warning(
            "TileInfer's vLLM integration is opt-in: set TILEINFER_VLLM=1 to enable it "
            "(any other value, including 0, keeps it off)."
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
        # GDN gating 接管为**显式 opt-in**(E21/D29:引擎内 A/B 无收益、TTFT +2~9%,不进默认
        # 路径;TILEINFER_OPS 显式含 gdn 时才安装)。decode/prefill 维持缺省全开;moe 同为
        # 显式 opt-in(H27/D70:内核线 ≥1.0x 门槛通过前不进默认路径)。
        _spec = (os.environ.get("TILEINFER_OPS") or "").strip().lower()
        _parts = {s.strip() for s in _spec.split(",") if s.strip()}
        if "gdn" in _parts:
            try:
                install_gdn_gating()
            except Exception as exc:  # noqa: BLE001 - GDN takeover must not break the attention slot
                logger.warning("TileInfer GDN gating dispatch not installed: %s", exc)
        else:
            logger.info(
                "TileInfer GDN gating takeover left off (TILEINFER_OPS does not list gdn; "
                "evidence E21: no engine-level gain, kept as explicit opt-in)"
            )
        if "gdnwu" in _parts:
            # chunk 链中段融合(H10)单独开关,便于与 gating 解耦做同侧 A/B 归因
            try:
                install_gdn_chunk_wu()
            except Exception as exc:  # noqa: BLE001
                logger.warning("TileInfer GDN chunk wu dispatch not installed: %s", exc)
        else:
            logger.info("TileInfer GDN chunk wu takeover left off (TILEINFER_OPS does not list gdnwu)")
        if "moe" in _parts:
            # MoE preNZ 直载接入(H27/D70):显式 opt-in——内核线 ≥1.0x aclnn 门槛
            # (H27 bench)通过前不得进默认路径(验收纪律:接入以微基准门槛为前置)
            try:
                from .moe_prenz import install_moe_prenz

                install_moe_prenz()
            except Exception as exc:  # noqa: BLE001
                logger.warning("TileInfer MoE preNZ dispatch not installed: %s", exc)
        else:
            logger.info("TileInfer MoE preNZ takeover left off (TILEINFER_OPS does not list moe; "
                        "explicit opt-in until the H27 kernel gate passes)")
        if "gdnproj" in _parts:
            # H32(S-GDN 投影族):ba→qkvz 权重融合,显式 opt-in——权威口径引擎 A/B
            # (E100 微基准门槛已过)通过前不进默认路径
            try:
                install_gdn_proj_fusion()
            except Exception as exc:  # noqa: BLE001
                logger.warning("TileInfer GDN in_proj fusion not installed: %s", exc)
        else:
            logger.info(
                "TileInfer GDN in_proj fusion left off (TILEINFER_OPS does not list "
                "gdnproj; H32 engine A/B pending)"
            )
        if "gdnreorder" in _parts:
            # H34(S-GDN 投影族):fused 权重 load 期行重排(终序单 GEMM+零拷贝切片),
            # 显式 opt-in;包装叠于 gdnproj 之外,接管成功时 gdnproj 计数不增(A/B 归因)
            try:
                install_gdn_proj_reorder()
            except Exception as exc:  # noqa: BLE001
                logger.warning("TileInfer GDN in_proj reorder not installed: %s", exc)
        else:
            logger.info(
                "TileInfer GDN in_proj reorder left off (TILEINFER_OPS does not list "
                "gdnreorder; H34 engine A/B pending)"
            )
        if "gdnsmall" in _parts:
            # H35(S-GDN 小核链):cat-free rearrange + conv1d 转置缓存 + l2norm
            # 配对融合,显式 opt-in——引擎 A/B(在役探针)通过前不进默认路径
            try:
                install_gdn_small_fusion()
            except Exception as exc:  # noqa: BLE001
                logger.warning("TileInfer GDN small fusion not installed: %s", exc)
        else:
            logger.info(
                "TileInfer GDN small fusion left off (TILEINFER_OPS does not list "
                "gdnsmall; H35 engine A/B pending)"
            )
        if "sampling" in _parts:
            # 采样 top-k 阈值替换(BACKLOG 采样条目澄清):显式 opt-in,性质同 GDN gating。
            # 安装须早于采样器构造(AscendTopKTopPSampler.__init__ 捕获模块级绑定)——
            # install() 本身先于引擎构建,满足时序契约。
            try:
                install_sampler_topk()
            except Exception as exc:  # noqa: BLE001
                logger.warning("TileInfer sampler top-k dispatch not installed: %s", exc)
        else:
            logger.info(
                "TileInfer sampler top-k takeover left off (TILEINFER_OPS does not list sampling)"
            )
        if "greedy" in _parts:
            # H36(采样域):greedy 全批快路径(跳过 fp32 全词表 cast + 处理器链),
            # 显式 opt-in——权威口径引擎 A/B(在役探针 greedy_fast_calls)通过前
            # 不进默认路径。
            try:
                install_greedy_fast_sampler()
            except Exception as exc:  # noqa: BLE001
                logger.warning("TileInfer greedy fast sampler not installed: %s", exc)
        else:
            logger.info(
                "TileInfer greedy fast sampler left off (TILEINFER_OPS does not list "
                "greedy; H36 engine A/B pending)"
            )
        if "moeprefetch" in _parts:
            # H37(S-MoE 域):MoE 权重 prefetch 阈值对齐(96→1 + 惰性武装兜底),
            # 显式 opt-in——上游 main 已移除该功能(仅 0.23.0 分支自持)且 950PR
            # 小 M GEMM L2-bypass 可能架空预取假设(E109),权威口径引擎 A/B
            # (在役探针 moeprefetch_active)通过前不进默认路径。
            try:
                from ..kernels.moe import prefetch as _moepf_impl
                from ..kernels.moe.prefetch import install_moe_weight_prefetch

                cls_stats = TileInferDecodeAttention._stats
                _moepf_impl._stats_sink = lambda name: cls_stats.__setitem__(
                    name, cls_stats.get(name, 0) + 1
                )
                install_moe_weight_prefetch()
            except Exception as exc:  # noqa: BLE001
                logger.warning("TileInfer MoE weight prefetch not installed: %s", exc)
        else:
            logger.info(
                "TileInfer MoE weight prefetch left off (TILEINFER_OPS does not list "
                "moeprefetch; H37 engine A/B pending)"
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
