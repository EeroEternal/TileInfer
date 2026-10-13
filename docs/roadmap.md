# Roadmap

> The scoreboard - what is actually done, with evidence - is [`status.md`](status.md).

> The concrete, ordered task list lives in [`next-tasks.md`](next-tasks.md); this document is the
> phased plan it serves.

Status legend: ✅ done · 🟡 in progress · ⏳ not started · ⏸ deferred(延后待授权)

> 2026-10-13 刷新:阶段一(attention 内核 + 引擎集成 + 图模式双源 A/B)全部完成;
> 阶段二融合算子层(M-MoE / M-GDN)已按 TEMPORARY/关闭收口(见 [`status.md`](status.md) §0);
> 阶段三(长上下文包)⏸ 延后。当前权威数字:`.agent_loop/PERFORMANCE_REPORT.md`。

## Phase 0 — environment and baseline ✅

| Item | Status |
|---|---|
| TileLang-Ascend toolchain installed and importing (`ascendc_pto` wheel) | ✅ |
| `torch_npu` seeing the Ascend950PR, dev user in the device group | ✅ |
| Metadata model (`indptr` / `indices` / `page_last_len` / page table) + validation | ✅ |
| Scheduler: query tiling, KV split, LPT ordering, workspace pool | ✅ |
| Reference backend (torch, CPU + NPU) as oracle | ✅ |
| Micro-benchmark harness with correctness checking and FIA hook | ✅ |
| Test suite runnable on CPU, NPU tests auto-skip | ✅ |

## Phase 1 — minimal attention ✅(2026-10-13 收口:Qwen3.6-35B-A3B 图模式在役,短上下文 TI≈FIA)

| Item | Status |
|---|---|
| Paged **decode** kernel, GQA/MQA, NHD layout, tail-page masking | ✅ **runs and matches the reference on Ascend950PR** (`tests/test_tilelang_ascend950_decode.py`, 5 cases: full page, partial tail, single token, ragged batch, group==M tile) |
| Correctness vs reference on the 950 | ✅ for decode (bf16, max abs diff ~1e-3 vs the fp32 oracle); prefill/append still to come |
| Fix the toolchain | ✅ done twice over: source build against local CANN for the fork (cube/vector correct, cube→vector hangs on three CANN versions), then the **official 0.1.15 Ascend backend + CANN 9.3.0**, which runs CV kernels correctly |
| Port the kernels to `tilelang.ascend` (`T.alloc_l1/l0c/shared`, `T.gemm`, `T.dual_copy`, `T.SimtVF` + `T.alloc_reducer`, `T.serial`) | ✅ done for decode — see `kernels/attention/paged_decode_ascend950.py` and the `tilelang-ascend950` backend |
| Cover what upstream's FA explicitly refuses: paged KV ✅, ragged lengths ✅, decode ✅, 128-dim path ✅ — remaining: causal/padding masks for prefill, `head_dim != 128`, fp16 | 🟡 decode half done |
| Report the fork's CV hand-off failure with the minimal reproducer | 🟡 draft ready in [`upstream-issue-pto-cv-hang.md`](upstream-issue-pto-cv-hang.md) (superseded for us, still valid for the fork) |
| Decode benchmark across (batch, kv_len) including long context | ✅ first baseline in [`performance.md`](performance.md) (147 GB/s peak at b16/kv4096; 51 GB/s at batch 1 — core-starved) |
| Close the batch-1 / long-context gap: split-KV kernel (consume `kv_tile_pages`) | ✅ split kernel landed and measured: **51.7 → 153.2 GB/s (3.0x)** at b1/32768; merge contract validated on host (2e-4 vs the oracle) — see [`performance.md`](performance.md) |
| Device **merge kernel** + wiring `kv_tile_pages` through the backend | ✅ done and validated; **opt-in** (`TILEINFER_ALLOW_SPLIT_KV=1`) because of KI-1 |
| Investigate KI-1 (vector-core exception when several split shapes share a process) | ⏳ open, with a reproducer (`--preset repro`) and 7 ruled-out hypotheses - see [`known-issues.md`](known-issues.md#ki-1--split-kv-wedges-the-device-when-several-shapes-share-one-process-open) |
| FIA baseline on the same shapes for the tile-level comparison | ⏳ needs a working FIA call on this stack |
| Split-KV merge kernel (`kv_tile_pages > 0`) | ⏳ plan + reference implementation + tests done |
| **Prefill / append** kernel (causal, `qo_len > 1`, chunked prefill) | ✅ validated on device (prefill, append, partial tail page, ragged batch) — tiles are planned at `block_q // 2` rows because the causal mask needs the global row index (PM-2) |
| Prefill: recover the wasted half of the M tile via explicit `T.Cube()` / `T.Vector(2) as sid:` blocks (~2x on prefill) | ⏳ **next** |
| Perf pass: `T.Pipelined` over pages, L0 staging, tile-size sweep, and a fair comparison against upstream's 320–362 TFLOPS dense FA | ⏳ |

Exit criteria: decode matches the reference on device for the full shape grid; prefill matches for
causal and append; published numbers against FIA on the same shapes.

## Phase 2 — engine integration ✅(Qwen3.6 图模式在役 + 双源 A/B;融合算子层收口结论如下)

| Item | Status |
|---|---|
| vLLM-Ascend TileInfer backend(metadata builder + impl + 算子开关) | ✅ `TILEINFER_VLLM=1` 插件接管 decode+prefill;`TILEINFER_OPS` 粒度开关(B6) |
| End-to-end Qwen3.6-35B-A3B serving(85B 权重,混合 GDN+d256) | ✅ eager 与图模式均端到端可用 |
| ACLGraph capture/replay parity | ✅ B5:6 plan READY、capture_calls=40、fallbacks=0、NaN 修复、logprobs 逐位一致 |
| Prefill fallback strategy | ✅ B1:prefill_step 统一接管,惰性编译+变体上限(D22),prefill_calls=949 实证 |
| 图模式双源 A/B(自研 9 场景 + 官方 bench 27 case) | ✅ PERFORMANCE_REPORT §5:短上下文比值 0.91-1.09;长上下文 2-6x(结构性,KI-1+编译窗) |
| M-MoE grouped GEMM 融合算子 | ⏸ **TEMPORARY**:设备时间 0.86x vs aclnn(3 轮优化轨迹 0.691→0.869→0.968x 墙钟),按 1.0x 契约不接入,交用户决策(E19/H7) |
| M-GDN 线性注意力算子(9a/9b/9c/gating) | ⏸ **收口**:9a/9b 按证据关闭(decode 侧无收益,E20);gating 替换 disproven(引擎 A/B 无收益,显式 opt-in 保留,E21/D29);9c chunk 链融合内核数值 6/6 绿但微基准 0.718x@T128/14.28x@T8192 未过 ×0.6 门槛,不接入(E23/E24/D31) |

Exit criteria: `--attention-backend TILEINFER`(插件等价机制)serves a model with output equal to the
FIA path within fp16 tolerance ✅,and TTFT/TPOT within 5% of FIA on standard shapes ✅(短上下文;
长上下文受 KI-1 门禁,属 Phase 3)。

## Phase 3 — long-context package(solid-KV/长上下文)⏸ deferred(阶段二完成后待授权)

| Item | Status |
|---|---|
| **L1** KI-1 诊断(split-KV 多形状同进程 wedge,复现:`benchmarks/bench_attention.py --preset serving --kv-tile-pages 16`) | ⏸ deferred |
| **L2** 池无关内核 ABI(池移出内核签名,预烧免精确池匹配;签名变更需重跑引擎内回归与预烧) | ⏸ deferred(L1 前置/伴生) |
| **L3** split-KV 真正启用(按 shape 预烧 ≈140s/shape + 长上下文 ≥8K 对比数据,兑现 2.8x) | ⏸ deferred(依赖 L1/L2) |

## Phase 4 — optimisation and extension ⏳(未立项;MoE/GDN 已按 TEMPORARY 收口交决策)

| Item | Status |
|---|---|
| Load-balanced split-KV wired through the kernel (not just the plan) | ⏳(并入 Phase 3 L3) |
| MLA (DeepSeek-V3/V4), sparse attention (`tile-ai` sparse FA examples as reference) | ⏳ |
| Low-precision paths (FP8/FP4 KV, int8 quantised decode) | ⏳(已裁决移除:FP8/量化 KV 不做,见 BACKLOG) |
| Cascade / shared-prefix attention, sliding window, attention sink | ⏳(已裁决移除:cascade/前缀缓存不做) |
| Grouped GEMM / MoE expert kernels, sampling | ⏸ MoE 已 TEMPORARY(0.86x,E19/H7);sampling 已裁决移除 |

## Phase 5 — ecosystem ⏳

| Item | Status |
|---|---|
| MindIE and one home-grown engine adapter example | ⏳ |
| Kernel authoring guide ("adding a kernel in 30 minutes") | ⏳ |
| Upstream contributions back to `tile-ai/tilelang-ascend` (perf fixes, examples) | ⏳ |

## Known gaps / deliberate omissions

These are conscious trade-offs of the current code, each with the place where it will be addressed:

* **Decode only in the TileLang backend.**  Prefill/append goes through `reference`.  Phase 1.
* **Split-KV is planned but not executed by a kernel.**  `kv_tile_pages > 0` raises for the
  TileLang backend; the reference backend executes it, which is how the merge contract is pinned
  down.  Phase 1.
* **MHA (group == 1) is rejected by the TileLang kernel** — the vector-unit split needs at least
  two heads per KV head.  Either pad the group or use `reference`.  Phase 1.
* **Page sizes below 16 are rejected** by the paged kernel; token-granularity caches should be
  repacked.  Phase 3.
* **Batch shape changes recompile.**  Batch bucketing (pad to a bucket, e.g. multiples of 16) is
  the standard serving answer and is a Phase 2/3 item.
* **No prefill-on-device causal mask yet.**  The mask machinery (`T.tile.compare` +
  `T.tile.select`) is already used for the decode tail, so the prefill kernel reuses it.
