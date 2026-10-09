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
| Paged **decode** kernel, GQA/MQA, NHD layout, tail-page masking | 🟡 written against the fork's API; to be **ported to the `tilelang.ascend` dialect** (row 4) |
| Correctness vs reference on the 950 (rtol/atol 5e-2, fp16) | ⏳ unblocked — the working stack is CANN 9.3.0 + official `tilelang==0.1.15`, verified on `Ascend950PR_9579` by running upstream's GEMM+ReLU Quick Start (it contains a real cube→vector hand-off) |
| Fix the toolchain | ✅ done twice over: source build against local CANN for the fork (cube/vector correct, cube→vector hangs on three CANN versions), then the **official 0.1.15 Ascend backend + CANN 9.3.0**, which runs CV kernels correctly |
| **Port the kernels to `tilelang.ascend`** (`target="ascend"`, `T.const` symbolic shapes, `T.alloc_l1/l0c/shared`, `T.gemm`, `T.dual_copy`, `T.SimtVF` + `T.Parallel`, `T.Pipelined`), with upstream `examples/ascend/flash_attention/core.py` as the shape/tiling reference | ⏳ **next** |
| Cover what upstream's FA explicitly refuses — which is our whole feature list: paged KV, ragged/variable lengths, decode (`q_len == 1`), causal + padding masks, `head_dim != 128`, fp16 | ⏳ next |
| Report the fork's CV hand-off failure with the minimal reproducer | 🟡 draft ready in [`upstream-issue-pto-cv-hang.md`](upstream-issue-pto-cv-hang.md) (superseded for us, still valid for the fork) |
| Decode benchmark across (batch, kv_len) including 128k context, and vs. FIA | ⏳ |
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
