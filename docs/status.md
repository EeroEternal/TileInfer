# Status

One page that answers two questions: **what has been done** (with the evidence) and **what is left**.
Details live in the linked documents; this file is the index and the honest scoreboard.

Last updated after the vLLM end-to-end milestone (commit `722e607`).

---

## 1. What works today

### The library

| piece | state | evidence |
|---|---|---|
| Decode kernel, paged + GQA + NHD + partial tail + ragged batch | ✅ validated on device | `tests/test_tilelang_ascend950_decode.py` (7 cases, bf16 vs fp32 torch oracle, max abs diff ~1e-3) |
| Split-KV decode (split kernel + device-side merge kernel) | ✅ validated, **opt-in** | 3.0x at batch 1 / 32k-64k tokens (51 -> 143.9 GB/s), within 2e-4 of the unsplit path |
| Prefill / append kernel (causal, ragged, paged) | ✅ validated | `tests/test_tilelang_ascend950_prefill.py` (4 cases incl. the append causal offset) |
| Plan / scheduler / metadata / workspace management | ✅ CPU-tested | `pytest tests -m "not npu"` green; device-side dense→ragged and batch padding helpers |
| Decode throughput, standalone | ✅ measured | 138-147 GB/s at wide shapes; 51 GB/s at batch 1 (why split-KV exists) |

All numbers and the methodology: [`performance.md`](performance.md).

### The vLLM-Ascend integration

Decode is **served by TileInfer inside a real engine**: the EngineCore compiles our kernel, builds the
plan (`TileInfer plan READY ... backend=tilelang-ascend950`) and serves steps through it, with the
first decode step's top-5 logprobs **identical to the stock FIA operator to four decimals**
(Qwen2.5-1.5B, head_dim 128, GQA group 6).

End-to-end, against the same server run with `TILEINFER_DISABLE=1` (stock FIA):

| workload | TileInfer | stock FIA | ratio |
|---|---|---|---|
| 1-token request (TTFT + one step) | 23 ms | 22 ms | 1.05x |
| 128-token greedy completion | 75.6 / 82.9 / 81.7 / 89.0 tok/s | 87.3 / 88.8 / 85.9 / 91.7 tok/s | 0.94x |
| 4 concurrent requests x 64 tokens | 223.9 tok/s | 319.6 tok/s | 0.70x |

Honest reading: this is the short-context regime (<= 2048 tokens, eager, no ACLGraph), which is
precisely where our kernel has no structural advantage — the split-KV path only pays off from ~8k
tokens and is still off by default (KI-1), and the batch-4 gap is scheduling rather than numerics.

Mechanism, fallbacks and the deployment recipe: [`integration-vllm-ascend.md`](integration-vllm-ascend.md);
details: [`performance.md`](performance.md).

### Environment and tooling

* Working stack pinned and documented: **CANN 9.3.0 + PyPI `tilelang==0.1.15`** (the `tile-ai/tilelang-ascend`
  fork is a documented dead end on this box) — [`architecture.md`](architecture.md).
* `scripts/patch-tilelang-ascend-bitcast.py`: the wheel's `numeric_limits.h` uses `std::bit_cast`
  without `<bit>`, which fails **every** kernel compile on this CANN release; the script rewrites the
  nine call sites to `__builtin_bit_cast`.
* `scripts/prewarm-tileinfer-kernels.py`: compiles the declared buckets **out of process** before a
  server starts, so the engine only ever sees cache hits (an in-engine compile blocks the engine).
* Probes that carry their weight: `benchmarks/probes/ascend950_paged_decode.py` (5/5 PASS canary) and
  the KI-1 probes.

---

## 2. What is broken or gated

| issue | impact | status |
|---|---|---|
| **KI-1** split path faults the device (`507035`) when several shapes are compiled and run in one process | keeps split-KV off by default, so long-context decode leaves 2.8x on the table | open; 8 hypotheses refuted (see below) |
| **PM-2** the causal mask needs a *global* row index, which a vector region cannot see, so prefill tiles are planned at `block_q // 2` | ~2x wasted M tile in prefill | open, cause understood, fix designed (`T.Cube()` / `T.Vector(2) as sid`) |
| integration serves **decode only** | prefill, chunked prefill and speculative decoding fall back to FIA | by explicit policy, not by accident |
| `head_size != 128` is declined (e.g. Qwen2.5-0.5B) | that model is served by FIA | deliberate: the 950 path is tuned for 128 |
| short-context single-stream is ~6 % behind FIA, batch 4 is ~30 % behind | this configuration is FIA's home turf | expected; see the items in section 3 |

Post-mortems and refuted hypotheses: [`known-issues.md`](known-issues.md). KI-1 has been probed for
caller-buffer alignment, KV pool size, stream ordering, the shapes themselves, both caches, the
harness' reference, the reference-vs-kernel ordering, and an exact mirror of the harness' case
construction — all refuted, each with a probe in `benchmarks/probes/`.

---

## 3. What is left

The ordered, actionable list is [`next-tasks.md`](next-tasks.md) (what / why / first step / done when).
Summary of it, with status:

| # | task | value | status |
|---|---|---|---|
| T1 | recover the box, collect interrupted logs | — | done |
| T2 | clean TileInfer-vs-FIA A/B | the number everyone asks for | **done for short context**; long context remains |
| T3 | **KI-1**: split path with several shapes in one process | unblocks 2.8x long-context decode | open — next |
| T4 | prefill: recover the wasted half of the M tile | ~2x on prefill (cube-bound) | open — designed |
| T5 | SIMD softmax (replace the SIMT one) | vector time off the critical path | open |
| T6 | warm-up and batch bucketing as first-class features | what makes the integration deployable | partly done (pre-warm script exists) |
| T7 | split-KV on by default | serving's main shape | blocked on T3 |
| T8 | explicit prefill / fallback policy | correctness of the integration surface | open |
| T9 | ACLGraph capture verification | the entire plan/run split exists for this | untested |
| T10 | FIA baseline inside the benchmark harness | "we are fast" needs the operator as yardstick | open |
| T11 | three ready-to-file upstream reports | highest-value contribution we already hold | open |
| T12 | device checklist (or CI) | every device run is manual today | open |
| T13 | Phase 3 breadth: MLA, sparse, FP8/FP4, MoE, sampling | scope | not started |

The single highest-value item is **T3**: it is the only thing standing between the current numbers and
the 2.8-3.0x that the split path already delivers standalone.

---

## 4. Reproducing any of this

```bash
# CPU suite (no device needed)
pytest tests -m "not npu"

# device: the two kernel suites and the canary probe
pytest tests/test_tilelang_ascend950_decode.py tests/test_tilelang_ascend950_prefill.py
python benchmarks/probes/ascend950_paged_decode.py          # expect 5/5 PASS

# standalone decode throughput (one shape per process; --kv-tile-pages + TILEINFER_ALLOW_SPLIT_KV=1
# for the split path)
python benchmarks/bench_attention.py --preset serving --backend tilelang-ascend950 --dtype bf16

# end to end in vLLM (see docs/integration-vllm-ascend.md for the full recipe)
TILELANG_CACHE_DIR=/tmp/tl_cache_vllm python scripts/prewarm-tileinfer-kernels.py \
    --kv-heads 2 --group 6 --dim 128 --page-size 128 --num-pages-cap 8192 --buckets 1 2 4
TILEINFER_VLLM=1 python scripts/vllm_tileinfer_launch.py serve <model> --attention-backend CUSTOM \
    --max-model-len 2048 --enforce-eager --gpu-memory-utilization 0.3 --num-gpu-blocks-override 8192
```

Environment variables that change behaviour: `TILEINFER_ALLOW_SPLIT_KV=1` (enable the split path),
`TILEINFER_VLLM=1` (install the vLLM plugin), `TILEINFER_DISABLE=1` (the A/B baseline),
`TILELANG_CACHE_DIR` (kernel cache; shared by the pre-warm and the server).

---

## 5. Lessons that cost time (so they do not cost it twice)

1. **A compile inside the engine stops the engine.** A TileLang compile holds the interpreter, so a
   "background" compile still blocks the whole process; requests fall back to FIA but they do not get
   served. Pre-warm out of process. This looked like the *machine* hanging for several rounds.
2. **The KV pool is not reproducible.** vLLM derives it from a memory snapshot (9425 pages in one run,
   9899 in the next), which silently invalidates a pre-warm and pushes compilation back inside the
   engine. Pin it with `--num-gpu-blocks-override`.
3. **A bare `wait` in bash waits for the server too.** Three "engine hangs" were my harness waiting on
   the server process launched with `&`, with the engine idle the whole time (`Running: 0`).
4. **Registry initialisation races in real engines.** vLLM builds one backend per attention layer and
   every layer asks the registry on the first decode step; the "loaded" flag was set before the imports
   and unsynchronised, producing `unknown backend ...; known: ['reference']`. It needed a lock, a
   regression test, and it is why a single-threaded test suite never saw it.
5. **Match processes inside a server-side script, never in an `ssh` command line**: a pattern in the
   command line matches the command line itself (`pkill -f`) and kills the session.
