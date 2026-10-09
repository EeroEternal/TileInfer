# TileInfer

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

**TileInfer** is an engine-agnostic, serving-oriented kernel library for LLM inference, with
first-class support for Huawei Ascend (starting from **Ascend 950 / Ascend950PR**).

It is intentionally *not* a serving engine. It is the layer between engines (vLLM-Ascend,
MindIE, home-grown runtimes) and the hardware: a small set of highly optimized, graph-capture
friendly kernels driven by standard metadata (`indptr` / `indices` / page table).

```
┌─────────────────────────────────────────┐
│     inference engine (vLLM-Ascend ...)  │
└─────────────────┬───────────────────────┘
                  │ standard metadata + plan/run API
┌─────────────────▼───────────────────────┐
│   TileInfer  (this repo)                │
│   BatchAttention · plan/run · workspace │
├─────────────────────────────────────────┤
│   TileLang kernels  │  Ascend C / PTO   │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│        Ascend NPU + CANN                │
└─────────────────────────────────────────┘
```

## Why

The Ascend ecosystem has no FlashInfer equivalent:

* MindIE is built on ATB and is bound to the official stack.
* vLLM-Ascend drives CANN FIA plus a handful of internal kernels; the kernel layer is not
  reusable on its own.
* Serving-specific problems — dynamic batch load balancing, custom KV layouts, JIT variants,
  engine decoupling — are not addressed by either.

TileInfer targets exactly that gap, using [TileLang](https://github.com/tile-ai/tilelang) as the
primary development language (`ascendc_pto` backend) and dropping to Ascend C / PTO only on the
hot paths that need it.

## Status

| Component | State |
|---|---|
| `plan` / `run` API, metadata, workspace, backend registry | ✅ implemented |
| `reference` backend (torch, CPU + NPU) — correctness oracle | ✅ implemented |
| Load-balanced scheduler (query tiling, KV split, LPT ordering) | ✅ implemented |
| Paged decode TileLang kernel (GQA, NHD, tail-page masking) | ✅ **runs on Ascend950PR and matches the torch reference** (~1e-3, bf16) — `tilelang-ascend950` backend |
| Prefill / append kernel | ⏳ Phase 1 |
| Split-KV merge kernel | ⏳ Phase 1 (plan + reference implementation done) |
| Micro-benchmark harness (vs. torch reference / FIA) | ✅ implemented |
| vLLM-Ascend backend integration | 🚧 Phase 2 (working sketch in `examples/`) |
| MLA / cascade / sparse / FP8 · MoE · sampling | ⏳ Phase 3+ |

> **Dev-environment note.** The device path is the **official** `tilelang==0.1.15` wheel
> (its `tilelang.ascend` dialect, `target="ascend"`) on **CANN 9.3.0**.  That combination is
> verified on the reference `Ascend950PR_9579`: upstream's Ascend 950 Quick Start — a GEMM whose
> ReLU epilogue goes through a real cube→vector hand-off — prints `All check passed`.  The
> `tile-ai/tilelang-ascend` fork was a dead end for this device (its PTO cube→vector path hangs on
> three different CANN versions); the measurement, the minimal reproducer and the upstream issue
> draft are kept in [`docs/architecture.md`](docs/architecture.md) and
> [`docs/upstream-issue-pto-cv-hang.md`](docs/upstream-issue-pto-cv-hang.md) so nobody repeats it.
> The next step is porting the kernels to `tilelang.ascend` and extending it where upstream stops
> — paged KV, ragged lengths, decode, and causal/padding masks, which upstream's FlashAttention
> explicitly does not support.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
