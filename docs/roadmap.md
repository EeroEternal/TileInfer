# Roadmap

Status legend: ✅ done · 🟡 in progress · ⏳ not started

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

## Phase 1 — minimal attention 🟡

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
| Device **merge kernel** (partial_out/partial_lse over the split axis) + wire `kv_tile_pages` through the backend | ⏳ **next** |
| FIA baseline on the same shapes for the tile-level comparison | ⏳ needs a working FIA call on this stack |
| Split-KV merge kernel (`kv_tile_pages > 0`) | ⏳ plan + reference implementation + tests done |
| **Prefill / append** kernel (causal, `qo_len > 1`, chunked prefill) | ⏳ |
| Perf pass: `T.Pipelined` over pages, L0 staging, tile-size sweep, and a fair comparison against upstream's 320–362 TFLOPS dense FA | ⏳ |

Exit criteria: decode matches the reference on device for the full shape grid; prefill matches for
causal and append; published numbers against FIA on the same shapes.

## Phase 2 — engine integration ⏳

| Item | Status |
|---|---|
| vLLM-Ascend `TILEINFER` backend (metadata builder + impl) | ⏳ sketch in `examples/` |
| End-to-end Qwen/Llama serving, eager mode | ⏳ |
| ACLGraph capture/replay parity (eager vs graph, bit-exact tolerance) | ⏳ |
| Prefill fallback strategy while the prefill kernel lands | ⏳ |

Exit criteria: `--attention-backend TILEINFER` serves a model with output equal to the FIA path
within fp16 tolerance, and TTFT/TPOT within 5% of FIA on standard shapes.

## Phase 3 — optimisation and extension ⏳

| Item | Status |
|---|---|
| Load-balanced split-KV wired through the kernel (not just the plan) | ⏳ |
| MLA (DeepSeek-V3/V4), sparse attention (`tile-ai` sparse FA examples as reference) | ⏳ |
| Low-precision paths (FP8/FP4 KV, int8 quantised decode) | ⏳ |
| Cascade / shared-prefix attention, sliding window, attention sink | ⏳ |
| Grouped GEMM / MoE expert kernels, sampling | ⏳ |

## Phase 4 — ecosystem ⏳

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
