# Next tasks

> Where each of these stands, and what is already finished, is in [`status.md`](status.md).

A working list, ordered by value ÷ cost.  Each entry says **what**, **why**, the **first step** (a file
or a command, not a plan) and **done when**.  Everything here is grounded in what the repo already
contains; nothing is speculative.

## Where the project stands

| | state |
|---|---|
| Decode kernel (paged, GQA, NHD, tail masking) | ✅ validated on device, ~1e-3 vs the fp32 oracle |
| Split-KV kernel + device merge kernel | ✅ written and validated (3.0x at b1/32k–64k), **opt-in** because of KI-1 |
| Prefill / append kernel (causal, ragged, paged) | ✅ validated (incl. the append causal offset), tiles planned at `block_q // 2` — half the M tile is wasted (PM-2) |
| Plan / scheduler / metadata / workspaces | ✅ implemented, 37 CPU tests |
| vLLM-Ascend integration | 🟡 **reaches the kernel**: the EngineCore builds the plan and serves decode through TileLang, first-step logprobs match FIA to 4 decimals.  No performance numbers yet |
| Performance baseline (decode, unsplit) | ✅ 138–147 GB/s at wide shapes, 51 GB/s at batch 1 (the reason split-KV exists) |
| Device CI | ⏳ none yet |

Open bugs and post-mortems live in [`known-issues.md`](known-issues.md) (KI-1 split-path instability,
PM-1/PM-2 resolved) and the environment traps in [`architecture.md`](architecture.md).

---

## P0 — close the loops that are already open

### T1 · Recover the machine and collect the interrupted A/B

*Why*: the box stopped answering during the clean A/B run; its logs are still on disk and may already
contain the numbers, so nothing should be re-run before they are read.

*First step*: wait / arrange console access, then

```bash
ssh tileinfer 'grep -avE "^INFO|Warning" /home/lipi/logs/ab_v4.log | tail -40'
ssh tileinfer 'grep -acE "plan READY|falling back" /home/lipi/logs/v4_ours.log /home/lipi/logs/v4_fia.log'
```

*Done when*: either the numbers are extracted from those logs, or the run is repeated
(`bash /home/lipi/ab_v4.sh`) on a box whose NPU is not shared with another user's test run.

### T2 · A clean TileInfer-vs-FIA comparison

*Why*: the milestone (our kernel serving inside vLLM) is verified numerically but the only timing we
have is from a run with 27 compiles failing in the background, so it measures a starved engine.

*First step*: `bash /home/lipi/ab_v4.sh` — it warms buckets 1/2/4, prints the `plan READY` and
`falling back` counters after each round, then measures 1-token latency and 128-token generation for
TileInfer and for the stock FIA path.  Add a batch-2/4 case (concurrent `curl`s) and an `npu-smi` line
next to each measurement so contention is visible in the numbers.

*Status*: done for short contexts - see the table in `docs/performance.md` (ours within ~6 % of FIA
single-stream, ~30 % behind at batch 4).  What remains: a long-context run (`--max-model-len` beyond 8k)
where the split-KV path is what should show up, and a repeat on a settled kernel.
*Done when*: a table with TTFT-proxy, tokens/s and (if possible) a decode-step latency for both
backends on the same prompts, with the fallback counter at zero for the TileInfer rows, and the
`docs/performance.md` section updated.  Also: explain why the earlier baseline run died
(`TILEINFER_VLLM=1` was still set while `TILEINFER_DISABLE=1`, so the "baseline" had the plugin
active — already fixed in the harness, worth re-verifying).

### T3 · KI-1: the split path faults the device with several shapes in one process

*Why*: the 2.8–3.0x on long-context decode is gated behind `TILEINFER_ALLOW_SPLIT_KV=1` because a
device fault takes the whole serving process with it.

*Already ruled out* (each with a probe in `benchmarks/probes/`): caller-buffer alignment, KV pools above
2 GiB, caller stream ordering, the shapes themselves, the disk/memory cache, the harness' reference,
the reference-vs-kernel ordering, and an exact mirror of the harness' case construction.

*First step*: the smallest fragment of the sweep that still faults — compile two split kernels back to
back **and launch each once**, then add shapes one at a time; `benchmarks/probes/ki1_pool_size.py` is the
closest existing template.  Run it under `ASCEND_LAUNCH_BLOCKING=1` so the fault is attributed to the
launch that causes it instead of the next torch op.

*Done when*: a reproducer short enough to send upstream, a root cause, and split-KV enabled by default
with the existing device tests covering multi-shape processes.

---

## P1 — finish the kernel library

### T4 · Prefill: recover the wasted half of the M tile (~2x)

*Why*: the causal mask needs a **global** row index, but a vector region only sees its own AIV's rows
(`dual_copy` splits M over two AIVs).  The workaround planes tiles at `block_q // 2`, so half of every
M tile is padding (PM-2 in `known-issues.md`).  Prefill is cube-bound, so this is a real 2x.

*First step*: a two-tile unit case with `group=1, block_q=32, qo_len=64` in
`tests/test_tilelang_ascend950_prefill.py`, then restructure the kernel into explicit
`with T.Cube():` / `with T.Vector(vector=2) as sid:` regions and mask with `sid * ROWS + r`
(`T.Vector` is the only way to obtain the AIV index — see `architecture.md`).

*Done when*: the prefill tests pass with full `block_q` tiles and the prefill benchmark improves
towards 2x.

### T5 · SIMD softmax

*Why*: the SIMT softmax was the right first trade (a few lines instead of a hundred), but it keeps
vector time on the critical path; upstream's FlashAttention uses the packed `T.simd.*` path.

*First step*: measure the gap first — `benchmarks/probes/ascend950_paged_decode.py` with a shape that
makes softmax time visible (long context), then port one softmax step at a time (max, exp, sum) using
`T.simd.*` on the existing buffers.

*Done when*: the probe shows a measurable gain and the device tests still match the oracle.

### T6 · Warm-up and batch bucketing as first-class features (partly done)

*Why*: a TileLang kernel is compiled per shape (~2 min standalone), a serving batch changes size every
step (hence bucket padding), and a compile must never sit in a request (hence the background thread +
FIA fallback).  All three mechanisms exist but are spread across the plugin.

*Done so far*: `scripts/prewarm-tileinfer-kernels.py` compiles the declared buckets out of process
into `TILELANG_CACHE_DIR`, which is what keeps the engine from stalling (see `architecture.md`).
*First step*: tie it to the backend (`warmup(buckets)`), and make the pool capacity reproducible so
the pre-warm can be derived from the server config instead of copied from a log.

*Done when*: a served model reaches "zero compiles after start-up" and the docs say how.

### T7 · Split-KV by default

*Why*: it is the main performance feature for the shape serving actually sees (single long request).

*First step*: T3, then flip the default and extend the device tests to cover `kv_tile_pages > 0` in a
multi-shape process.

---

## P2 — engine integration

### T8 · vLLM-Ascend: the prefill path and the compatibility surface

*Why*: TileInfer currently serves **decode only**; prefill, chunked prefill and speculative decoding
fall through to vLLM-Ascend.  That is a fine first policy but it should be a *chosen* one.

*First step*: dispatch on `attn_metadata.attn_state` (`DecodeOnly` → TileInfer, the rest → FIA) and
record the policy in `docs/integration-vllm-ascend.md`; then try the prefill kernel behind an env flag
once T4 has landed.

*Done when*: the policy is explicit, the SupportedConfig paths are exercised by a real model run
(Llama/Qwen), and the fallback is covered by a test or a documented checklist item.

### T9 · ACLGraph capture verification

*Why*: the whole plan/run split exists so that captured graphs contain no allocation, no compile and
no host sync — but that has never been tested with `--enforce-eager` off.

*First step*: run the same prompt with and without ACLGraph (`cudagraph_mode`), compare the outputs and
the log for unexpected allocations/compiles inside capture.

*Done when*: eager and captured runs agree (within fp16 tolerance) and the logs show no compile during
capture.

### T10 · FIA baseline inside the benchmark harness

*Why*: "we are fast" needs the CANN operator as the yardstick, not torch.

*First step*: finish `run_fia` in `benchmarks/bench_attention.py` for the CANN FIA signature used by
this stack (`torch_npu.npu_fused_infer_attention_score` / the vLLM-Ascend call), and print it next to
the TileInfer rows.

---

## P3 — upstream and ecosystem

### T11 · Report upstream (highest-value contributions we already hold)

1. **`tilelang-ascend` wheel header bug**: `src/tl_templates/ascend/numeric_limits.h` calls
   `std::bit_cast` without including `<bit>`, so *every* kernel compile fails on CANN 9.3; the fix is
   `scripts/patch-tilelang-ascend-bitcast.py`.  One-line diff, blocks every new user of the wheel.
2. **Four platform gotchas for the vLLM integration**: `get_attn_backend_cls` ignores
   `--attention-backend`; the EngineCore is a separate process (entry points, not import hooks);
   `vllm_ascend.ops` must be imported before `attention_v1` (circular import); the usage-reporting
   thread kills EngineCore on this box (`VLLM_NO_USAGE_STATS=1`).
3. **The fork's PR**: `benchmarks/probes/pto_cv_handoff.py` (the 25-line cube→vector reproducer) plus
   the three-CANN-version table — already written up in `upstream-issue-pto-cv-hang.md`.

### T12 · A device checklist (or CI)

*Why*: the CPU suite runs in CI; everything on the device is manual.  Two canaries already have
exit-code semantics.

*First step*: `docs/device-checklist.md` with the exact commands and expected outputs
(`benchmarks/probes/ascend950_paged_decode.py` → 5/5; device tests → 7 passed; the vLLM A/B), and — if a
runner with an Ascend card becomes available — a workflow that runs them.

### T13 · Phase 3 breadth

MLA (DeepSeek-V3/V4), cascade / shared prefix, sparse attention, FP8/FP4 KV, fp16 (`head_dim != 128`),
MoE grouped GEMM, sampling.  Each starts as a new kernel module plus a reference implementation in
`tileinfer/testing/`, mirroring how decode and prefill were built.

---

## Operational notes (state of the reference box)

* The pre-warm key needs `--num-gpu-blocks-override` (or an equally deterministic KV pool): vLLM
  computes the pool from a memory snapshot, so it came out as 9425 in one run and 9899 in the next,
  which silently invalidated a pre-warm and pushed compilation back inside the engine.
* Harness trap worth remembering: a bare `wait` in bash also waits for the server process launched
  with `&`, so "warming up" stalled forever while the engine sat idle - use `wait $pids`.
* **The box stopped answering** during the T2 run (`ssh` handshakes but sends no banner, the vLLM port
  is silent).  See the note in `architecture.md`; it needs console access if it does not recover.
  Detached runs keep writing logs, so **collect `/home/lipi/logs/ab_v4.log`, `v4_ours.log`,
  `v4_fia.log` before re-running anything**.
* Two vLLM servers that belonged to another user were stopped on request to free HBM (they do not
  restart by themselves).  If that work matters, coordinate before the next heavy run: two hangs
  happened while someone else's Ascend test was running next to a TileLang compile.
* Useful paths on the box: repo `/home/lipi/TileInfer`, environment `source /home/lipi/env.sh`,
  working stack CANN 9.3.0 + `tilelang==0.1.15` (see `scripts/env-ascend950.sh.example`), the vLLM
  venv `/home/lipi/venv_vllm`, a real `head_dim=128` model `/home/lipi/models/qwen2.5-1.5b`, and the
  A/B driver `/home/lipi/ab_v4.sh`.
