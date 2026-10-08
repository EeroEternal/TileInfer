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
| Paged decode TileLang kernel (GQA, NHD, tail-page masking) | 🟡 compiles + launches on Ascend950PR; device numerics blocked on the toolchain (see below) |
| Prefill / append kernel | ⏳ Phase 1 |
| Split-KV merge kernel | ⏳ Phase 1 (plan + reference implementation done) |
| Micro-benchmark harness (vs. torch reference / FIA) | ✅ implemented |
| vLLM-Ascend backend integration | 🚧 Phase 2 (working sketch in `examples/`) |
| MLA / cascade / sparse / FP8 · MoE · sampling | ⏳ Phase 3+ |

> **Dev-environment note.** The TileLang Ascend toolchain on the reference machine is a
> `+linux.cann910` wheel running against CANN 9.1.1 / driver 25.7.rc1. With that combination,
> *upstream's own* `paged_flash_attn_bhsd.py` example also returns wrong numbers, so device
> numerics are currently blocked by the toolchain rather than by TileInfer. Everything that can be
> verified without trusting that toolchain is verified: the whole CPU suite passes, and the kernel
> itself compiles and launches. See
> [`docs/architecture.md`](docs/architecture.md#known-environment-traps-ascend-950pr--cann-911).

## Install

```bash
# host side (Python interface + reference backend)
pip install -e .

# device side (Ascend): install the TileLang Ascend toolchain first
source /usr/local/Ascend/ascend-toolkit/set_env.sh
pip install tilelang-<ver>+linux.cann910-cp312-cp312-linux_x86_64.whl   # tile-ai/tilelang-ascend release
pip install -e .
```

## Quick start

```python
import torch
from tileinfer import BatchAttention

attn = BatchAttention(backend="tilelang", device="npu", dtype=torch.float16)

# ---- plan stage: host side, may run once per batch shape change ----
plan = attn.plan(
    mode="decode",
    kv_indptr=kv_indptr,                # [B+1] int32, cumulative page counts
    kv_indices=kv_indices,              # [num_pages] int32, physical page ids
    kv_last_page_len=kv_last_page_len,  # [B] int32
    num_qo_heads=32, num_kv_heads=8, head_dim=128,
    page_size=128, load_balance=True,   # bucket skewed batches into equal-work tiles
)

# ---- run stage: pure compute, ACLGraph-capturable ----
out = attn.run(q, kv_cache, plan=plan)
```

See `examples/quickstart.py` and `benchmarks/bench_attention.py`.

## Repository layout

```
tileinfer/                 Python interface layer
├── attention/             BatchAttention, plan/run, backend registry
├── kernels/attention/     TileLang kernel sources
├── testing/               reference implementation, cache builders
├── metadata.py            engine-agnostic metadata (indptr/indices/page table)
└── plan.py                load-balanced partitioning
benchmarks/                micro-benchmarks (TileInfer vs reference vs FIA)
examples/                  runnable examples, incl. the vLLM-Ascend backend sketch
tests/                     pytest suite (CPU-runnable; NPU tests auto-skip)
docs/                      design, architecture, integration, roadmap
```

## Documentation

* [`docs/design.md`](docs/design.md) — design document (v0.1), goals and architecture.
* [`docs/architecture.md`](docs/architecture.md) — plan/run, metadata, load balancing, workspaces.
* [`docs/roadmap.md`](docs/roadmap.md) — phased plan and acceptance criteria.
* [`docs/integration-vllm-ascend.md`](docs/integration-vllm-ascend.md) — how an engine plugs in.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
