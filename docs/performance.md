# Performance

Numbers from the reference machine (single `Ascend950PR_9579`, CANN 9.3.0, official
`tilelang==0.1.15`, bf16).  Nothing here is a promise — it is a baseline with the methodology and
the known limits written down, so the next person can tell an improvement from a regression.

Reproduce with:

```bash
source <CANN 9.3.x>/set_env.sh
python benchmarks/bench_attention.py --preset serving --backend tilelang-ascend950 --dtype bf16
```

## Decode baseline (paged, GQA 32q/8kv, head_dim 128, page 128)

| batch | kv_len | ms | GFLOP/s | GB/s | max abs err vs fp32 oracle |
|---|---|---|---|---|---|
| 64 | 512 | 0.982 | 547 | **137** | 2.0e-3 |
| 32 | 2048 | 1.847 | 581 | **145** | 9.8e-4 |
| 16 | 4096 | 1.831 | 587 | **147** | 4.9e-4 |
| 8 | 8192 | 2.144 | 501 | **125** | 4.9e-4 |
| 4 | 16384 | 2.785 | 386 | **96** | 2.4e-4 |
| 1 | 32768 | 2.652 | 203 | **51** | 2.4e-4 |
| 1 | 65536 | 5.230 | 205 | **51** | 1.2e-4 |

*Correctness holds across the whole grid* (bf16 vs the fp32 torch reference, max abs error ≤ 2e-3),
which is the first thing to check before believing any of the timings.

### How to read it

**Decode is memory bound, so `GB/s` is the metric, not `GFLOP/s`.**  One query row per request
reads K and V exactly once, so the floor is `2 · batch · kv_heads · kv_len · head_dim · 2` bytes
plus the page table; the harness computes that and divides by the median of 10 runs.

* The best point (147 GB/s at b16/kv4096) is where the grid is wide enough to use the cores:
  the kernel launches `batch × kv_heads` blocks, so b16 gives 128 blocks.
* The worst points are `batch == 1`: 8 blocks for 32 cores, one core streaming 32k–64k tokens of
  K and V alone.  That is a *scheduling* limit, not a bandwidth limit — and it is exactly what the
  planner's split-KV (`kv_tile_pages > 0`) was built for: the reference implementation and its
  tests already exercise the split-and-merge contract, the kernel has yet to consume it.
* For context on the other axis: upstream's dense **prefill** FA reaches 320–362 TFLOP/s on this
  device (compute bound, 4096-row tiles).  A decode kernel is never going to look like that, which
  is why the column above is GB/s.

## Known limits of this baseline

| Limit | Effect | Where the fix belongs |
|---|---|---|
| One block per `(request, kv_head)` | small batches and long contexts starve the cores (51 GB/s at batch 1) | split-KV along pages (planner support exists, `kv_tile_pages`), then a merge kernel; and/or `T.Persistent` over a fixed core count |
| M tile padded to 32 rows | GQA group 8 wastes 3/4 of the QK^T M dimension | pack several requests per tile when they share a page, or accept it — cube time is not the bottleneck at these shapes |
| SIMT softmax | a few `µs` of vector time per page, not overlapped with the next page's load | move to the upstream `T.simd.*` packed-softmax path once correctness is locked (that is what upstream's FA does) |
| ~75 s compile per new shape | first request after a shape change is slow | warm up the shapes a deployment uses; the plan/run split already keeps compilation out of `run` |
| Numbers measured with another vLLM instance holding ~113 GB of the 128 GB HBM | possible interference, no exclusive HBM | re-measure on an idle device before publishing |

## Split-KV: 3x on the starved shapes (measured, merge validated)

The `51 GB/s at batch 1` row above is a *scheduling* limit: the kernel launches
`batch × kv_heads` blocks, so one long request occupies 8 of 32 cores.  Splitting the KV axis —
which the planner has supported all along (`kv_tile_pages`, with the merge contract implemented and
tested in the torch reference) — fills the grid instead:

| shape | tiles / blocks | ms | GB/s |
|---|---|---|---|
| b1 / 32768 tok, no split | 1 / 8 | 2.596 | **51.7** |
| b1 / 32768 tok, 16 pages per tile | 16 / 128 | 0.876 | **153.2** |
| b1 / 32768 tok, 8 pages per tile | 32 / 256 | 0.900 | 149.2 |
| b1 / 65536 tok, 16 pages per tile | 32 / 256 | 1.741 | **154.2** |
| b64 / 512 tok (wide control) | 64 / 512 | 0.886 | 151.4 |

Through the **public API** (`BatchAttention.plan(..., kv_tile_pages=16)` with
`TILEINFER_ALLOW_SPLIT_KV=1`, one shape per process — see KI-1):

| shape | splits | ms | GB/s | vs unsplit |
|---|---|---|---|---|
| b1 / 65536 tok | 32 | 1.866 | **143.9** | 51 -> **2.8x** |
| b4 / 16384 tok | 8 | 1.853 | **144.9** | 97 -> +49% |
| b8 / 8192 tok | 4 | 1.859 | **144.4** | 125 -> +15% |

All three match the CPU oracle to <= 2e-4.  The 2.8x at the long-context single-request case is the
whole point: that is the shape a coding agent or a long-document request produces.

**3.0x** on the long-context single-request case, and the whole grid now sits at ~150 GB/s, i.e. the
split kernel *at the wide shape* is already as fast as the unsplit one — the extra partial/lse
traffic (~3% of the K/V traffic) is not visible.

Correctness of the split path: a 4096-token request split into 4 tiles, partials merged on the host
with the reference's weighting (`out = Σ exp(lse_t − lse_all) · partial_t`, weights summing to 1),
matches the dense oracle to **2e-4**.

Reproduce: `python benchmarks/probes/split_kv_decode.py` (host-side merge, used to validate the
contract) — for the end-to-end path use the API with the opt-in:

```bash
TILEINFER_ALLOW_SPLIT_KV=1 python -c "..."   # BatchAttention.plan(..., kv_tile_pages=16)
```

Status: **split kernel + device merge kernel are landed, wired through the backend, and validated**
by `tests/test_tilelang_ascend950_decode.py` (single request with 4 splits; ragged 2-request batch
with different split counts).  Split-KV is **opt-in** (`TILEINFER_ALLOW_SPLIT_KV=1`) because it has
twice wedged the device with a vector-core exception when several shapes are compiled and run in one
process — not reproducible for a single shape, nor across 20 consecutive launches of one plan; see
[`known-issues.md`](known-issues.md#ki-1--split-kv-wedges-the-device-when-several-shapes-share-one-process-open).
Measurements are therefore taken one shape per process.

## What is *not* measured yet

* prefill / append (the kernel does not exist yet — decode only);
* end-to-end serving (TTFT / TPOT / throughput inside vLLM-Ascend);
* the CANN FIA baseline on the same shapes (the harness has the hook, it needs a working FIA call
  on this stack — see `run_fia`).

## End-to-end: TileInfer against the stock FIA operator inside vLLM

Qwen2.5-1.5B-Instruct (hidden 1536, 12 heads, 2 KV heads, head_dim 128, GQA group 6) served by
vLLM 0.23.0 + vLLM-Ascend 0.23.0, `--enforce-eager`, `--max-model-len 2048`, KV pool pinned with
`--num-gpu-blocks-override 8192`, one prompt, greedy.  TileInfer's decode plan was built **inside the
EngineCore** (`TileInfer plan READY ... backend=tilelang-ascend950`) and the measured window had
**zero fallbacks to FIA**; the baseline is the same server started with `TILEINFER_DISABLE=1`, so
vLLM-Ascend serves through the stock FIA operator.

| workload | TileInfer | stock FIA | ratio |
|---|---|---|---|
| 1-token request (TTFT + one step), median of 5 | 23 ms | 22 ms | 1.05x |
| 128-token greedy completion, 4 runs | 75.6 / 82.9 / 81.7 / 89.0 tok/s | 87.3 / 88.8 / 85.9 / 91.7 tok/s | 0.94x |
| 4 concurrent requests, 64 tokens each | 223.9 tok/s | 319.6 tok/s | 0.70x |

Reading it honestly: for a single stream our decode path is within ~6 % of the production operator,
and at four concurrent requests it is ~30 % behind.  Both sides run eager (no ACLGraph) and the
context never exceeds 2048 tokens, so this is the *short-context* regime — the one where our kernel
has no structural advantage:

* the split-KV path (2.8x standalone at 32k-64k tokens) only pays off from roughly 8k tokens up and is
  still off by default because of KI-1, so nothing here exercises it;
* at batch 4 the difference is scheduling, not numerics: our plan gives each request one work tile per
  step, while FIA's operator packs the batch into its own tiling.  That is what the batch-bucketing
  work (T6) is for;
* the prefill M-tile fix and the SIMD softmax (T4/T5) are the two known ~2x items still on the table.

Method notes that mattered: the kernels must be pre-warmed **out of process** (a compile inside the
engine stalls it, see `architecture.md`), the KV pool must be pinned (`--num-gpu-blocks-override`) so
the pre-warm key is reproducible, and a bare `wait` in the harness waits for the server process too -
which is what made three earlier "engine hangs" look like engine hangs when the engine was idle.
