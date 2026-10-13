# Known issues and post-mortems

## PM-1 — "kernel memory corruption" that was a bad gather index (resolved)

**Status:** resolved · **Lesson:** on Ascend, an out-of-range *gather* index is not a fault — it
returns whatever is at that address, which is indistinguishable from a kernel scribbling over
memory.

### What it looked like

A device test that builds a paged KV cache reported that a tensor the kernels only ever *read* came
back corrupted:

```
before: [15, 11, 3, 6, ..., 36, 32, 35, 34, 33, 37, 39, 38]
after : [15, 11, 3, 6, ..., 10573, 0, 42304, 0, 0, 8, 0, 42304]
```

request 0's page ids were intact, request 1's were replaced by garbage — and the *same* garbage
every run.  That fingerprint (deterministic "corruption", starting exactly at a tile boundary) sent
us looking for an out-of-bounds write in the split/merge kernels, and a gate was added that refused
`kv_tile_pages > 0` with `batch > 1`.

### What it actually was

The test's cache builder permuted each request's page list like this:

```python
perm = torch.randperm(end - start, generator=gen) + start   # wrong: global offset ...
indices[start:end] = indices[start:end][perm.to(device)]    # ... applied to a local slice
```

The `+ start` turned a 0-based permutation into global page ids, so the gather indexed a 32-element
slice with values 32..39.  On Ascend that silently returns unrelated device memory instead of
raising (the same mistake raised `IndexError` on the CPU path, which is why the *reference*
computation failed later with `index 42304 is out of bounds for dimension 0 with size 40`).

Fixing the permutation to be 0-based makes every case pass, including multi-request split
schedules, and the "corruption" disappears with no kernel change.

### Why it took so long to see

* The garbage was deterministic, so it looked like real data being written somewhere it should not
  be — rather than like uninitialised memory.
* Our debug hook (`.cpu()` between the two launches) reported the tensor as intact: torch copies
  are ordered on *torch's* stream, while the TileLang kernels are launched on their own, so the
  check ran before the kernels had finished.  `torch.npu.synchronize()` in the test then made the
  writes visible.
* Interleaving guard tensors around the plan's buffers made the symptom vanish — because the
  canaries changed the allocator layout — which "confirmed" a layout-dependent overrun.

### Rules that came out of it

1. **Always bounds-check indices on the host before they reach a device gather** (the test now
   asserts `kv_indices` integrity right after the run, so a future repeat fails loudly).
2. When a device test disagrees with a CPU reference about *values*, first check for
   `IndexError`-class mistakes that Ascend does not report: gather indices, out-of-range slices.
3. Synchronise the device (`synchronize`) before trusting a debug peek, not just the current stream.

Two changes made while chasing this are worth keeping and are now permanent: the kernels read
**plan-owned copies** of the page table (stable addresses, caller tensors can never be touched), and
the merge kernel stores a **full padded tile** (`BR` rows) rather than only the live `group` rows, so
every write is tile-aligned and in-bounds by construction.

## KI-2 — bundled backends did not register inside a vLLM EngineCore (resolved)

**Status:** resolved, and the first diagnosis was incomplete.  Two separate defects, both in
`_ensure_builtin_backends()`:

1. it returned early once `reference` was registered, so a failure while importing a later backend left
   the registry permanently incomplete (fixed first: each backend is now imported independently and its
   failure is collected in `_IMPORT_ERRORS` and quoted in the error message);
2. the "loaded" flag was set **before** the imports and nothing was synchronised.  Engines call this
   from several threads at once — vLLM builds one backend object per attention layer and the first
   decode step of every layer asks the registry simultaneously — so one thread returned immediately,
   looked up `tilelang-ascend950`, and raised `unknown backend ...; known: ['reference']` while another
   thread was still importing.  A single-threaded test could never see it, which is why the earlier
   "resolved" was wrong.  The import now runs under a lock and the flag is set after it, and
   `tests/test_backend_registry.py` races eight threads at the registry (verified to fail against the
   old ordering).

Both defects are the same user-visible message, which is what made this expensive to untangle: it took
one log line per import (the `_diag` calls, now at debug level) to see `registry=[]` and
`registry=['reference']` on consecutive lines.

The original note follows.

**History:** affects the vLLM integration only (the standalone library is fine).

In the vLLM plugin's own process every bundled backend registers
(`['reference', 'tilelang', 'tilelang-ascend950']`); inside the EngineCore `list_backends()` returns
`['reference']`, and asking for `tilelang-ascend950` raises `unknown backend ... known: ['reference']`
**without** any import error being recorded - which `_ensure_builtin_backends()` (independent imports,
failures collected in `_IMPORT_ERRORS` and quoted in the message) cannot produce as written, so
something about the engine process is *undone* rather than failing.

Eliminated already: a duplicate module identity from having TileInfer both editable-installed and on
`PYTHONPATH` (removed, no change).

Next: one probe that logs `list_backends()`, `_REGISTRY.keys()` and `_IMPORT_ERRORS` at plugin load
time and again inside `_build_plan()`.  That distinguishes "imported but unregistered" from "never
imported" and points at the line to fix.  Meanwhile the plugin falls back to FIA, which is why the
integration is still opt-in.

## KI-1 — split-KV wedged the device: merge-kernel staging fault for group 4 (RESOLVED 2026-10-14)

**Status:** resolved · **Root cause:** the merge kernel staged `PartLse` GM→UB at *group* granularity;
with the serving GQA config (32 qo / 8 kv heads → group 4) that is a 16-byte descriptor, and the
copy engine rejects sub-32-byte descriptors with `ACL_ERROR_RT_VECTOR_CORE_EXCEPTION` (507035).
**Fix:** stage the lse row at the slot stride `BR` (one line + docstring; the `PartOut` staging
stays at group granularity - its rows are 256 B). ·
**Affected:** `tilelang-ascend950` with `kv_tile_pages > 0` and group 4 (group 8 was always fine) ·
**Default:** the backend still **refuses** split plans unless `TILEINFER_ALLOW_SPLIT_KV=1` (policy,
not a workaround - see below).

### Symptom (historical)

The serving sweep (`benchmarks/bench_attention.py --preset serving --kv-tile-pages 16`) died on the
third shape - the first one whose plan splits:

```
b64/kv512   -> ok          b16/kv4096 -> ACL_ERROR_RT_VECTOR_CORE_EXCEPTION (507035)
b32/kv2048  -> ok          b8/kv8192  -> every later op fails (device wedged)
```

The exception surfaced asynchronously, in whatever torch op ran next, which (with the misleading
"several shapes in one process" framing) sent the investigation towards allocator layout and process
state.

### Root cause (found 2026-10-14, evidence E37)

Launch-level canary instrumentation (per-launch sync + full pointer dumps, `.agent_loop/run/p6/`)
attributed the fault to the **merge kernel's very first launch**; the split kernel completed cleanly.
Then the "multi-shape process" framing collapsed:

1. a *single-shape* fresh process (`b16/kv4096` only, split+merge) faults identically - no history,
   no reuse, no other kernel;
2. a minimal merge-only probe with **identical addresses and sizes** passes for group 8 and faults
   for group 4 - only the `group` constant differs;
3. the merge kernel copies `PartLse[base*BR : base*BR + group]` - 8 fp32 = 32 B for group 8
   (exactly one sector, fine) but 4 fp32 = **16 B for group 4** (sub-sector → fault).

Every earlier "clean" experiment had used group-8 probe configs (`kv_heads=2, group=8`, as do the
device tests to this day) - the serving group-4 merge had never been launched in isolation.  That is
why "20 consecutive launches clean" and the three-shape mirror probe never saw it, and why the bug
looked like exotic process state.

### Fix

`paged_decode_merge` now stages the lse row at the full slot stride `BR` (32 fp32 = 128 B); the
padding rows are real memory (the slot stride of `PartLse` *is* `BR`, so the copy never leaves the
tensor) and are never read by the reductions.  `PartOut` staging keeps group granularity: each row
is 256 B and a BR-padded `out_g` would overflow UB at the long-context split counts (32 splits ×
32 rows × dim = 256 KB).  Device tests now cover split-KV with group 4 (`split-KV, serving group 4`,
`split-KV, group 4 ragged`).

### Verification (all green, E37 / .agent_loop/run/p6/chain.out)

* merge-only: group ∈ {4, 8} × max_splits 2, and group 4 × max_splits 32 (UB pressure) - SYNC OK;
* backend path, single shape `b16/kv4096`: 20 consecutive launches clean;
* `--preset repro` (b64/b32/b16-split): all three rows ok, split maxerr 4.86e-04;
* `--preset serving` full sweep with `--kv-tile-pages 16`: all seven shapes, no wedge.

### Why split-KV stays opt-in

The fault is fixed, but the 2026-10-13 decision stands on its own merits: per-shape kernel burns
~70 s of compile on first sight, and the engine rollout (P8) needs per-shape prewarm before the flag
flips.  Enabling is now an engineering decision, not a safety one.

### Retracted earlier claims (kept for the record)

* "the vLLM EngineCore fault" - withdrawn even before this resolution: the same server crashes
  identically with TileInfer off (hand-made tiny Qwen2 config fault regardless of backend).
* allocator layout / process state hypotheses (H14): disproven by the single-shape and merge-only
  reproductions above.

## PM-2 — multi-tile prefill was off by a row offset (resolved)

**Status:** resolved (with a documented cost) · `tilelang-ascend950` prefill path.

### Symptom

Prefill/append matched the reference for a single query tile, and was wrong by ~3.0 - the magnitude
of a *misplaced row*, not of a precision problem - as soon as a request spanned more than one tile.
The single-tile case that passed included the append offset `kv_len - qo_len`, so the mask, the
paging and the softmax were all fine; only the row mapping could be wrong.

### Root cause

The causal mask needs the row's *global* query position, and the number the kernel can compute inside
the vector region is the *per-AIV* row index: `dual_copy` splits the M tile over the two AIVs and each
one sees `ROWS = block_q // 2` rows **starting at 0**.  For a tile whose live rows all fall in the
first half, local == global (hence the single-tile pass); otherwise the second AIV computes its mask as
if it held the first half's rows - and its output is *not* discarded, it is written into the upper half
of the packed tile.

The dialect does not expose the AIV index to a `T.SimtVF` region.  It is available only through the
explicit mixed-kernel structure: `with T.Vector(vector=2) as sid:` (documented in
`tilelang/ascend/language/frame.py`, where `sid = asc_get_sub_block_id()`), which also implies
wrapping the cube work in `with T.Cube():`.

### Resolution

For now: **the caller plans its query tiles with `block_q // 2` rows**, so every live row is in the
first AIV half.  `forward_prefill` enforces it and explains why; the tests cover multi-tile prefill,
append, a partial tail page and a ragged batch.  The cost is that half of the M tile is padding, i.e.
~2x the cube work a prefill could need.

The proper fix - worth ~2x on prefill - is to restructure the kernel into explicit `T.Cube()` /
`T.Vector(vector=2) as sid:` regions and mask with `sid * ROWS + r`.  That is a bigger change (it also
moves the `dual_copy`s into the vector region) and it needs device iteration, so it is a standalone
task rather than a footnote to this one.

## KI-4 — offline `LLM()` and serve(HTTP) numbers are not comparable (diagnosed, by design)

**Status:** diagnosed · **Impact:** measurement methodology, not a product bug.

### Symptom

The same decode workload (batch 1 x 128 tokens, enforce-eager, same server flags) measures
~136.7 ms/step through the offline `LLM()` API but ~45.9 ms/step (TPOT) through a `vllm serve`
HTTP benchmark — the *offline* number is 3x **worse**, which is the opposite of the usual
intuition that a benchmark harness should be leaner than a full server (EVIDENCE E14;
`.agent_loop/run/b3_ti_b1.log` vs the eager matrix in `PERFORMANCE_REPORT.md` §3).

### Attribution

1. **EngineCore process topology.** `vllm serve` runs the EngineCore in a dedicated subprocess:
   scheduler, sampler-side CPU work, detokenization and HTTP streaming overlap the engine's
   dispatch loop across processes. The offline `LLM()` (with `VLLM_ENABLE_V1_MULTIPROCESSING=0`,
   which is what in-process profiling probes need) runs everything in **one GIL**: the engine
   loop's Python (sampling prep, detokenization, stream assembly) serializes against every
   `aten::`/CANN op dispatch, inflating wall time per step.
2. **What is actually comparable:** component-level attribution inside `execute_model`.
   Clean in-process timers (perf_counter, no profiler — profiler instrumentation inflates
   in-region timings by ~2x, see EVIDENCE E12) give: FIA `execute_model` 42.2 ms/step at
   batch 4, TileInfer 97.0 ms/step at batch 1 (of which the TileInfer kernel-call path is
   ~4.4 ms x 10 full-attn layers after the per-step refresh dedup, D23/E13). Everything
   outside `execute_model` is engine-loop/GIL territory and differs by topology.
3. **Graph mode dissolves most of the gap:** replay executes the captured kernels without
   per-step Python, so host-path differences (TileInfer's included) mostly vanish — which is
   why the production A/B (graph mode) is the authoritative comparison.

### Rules that came out of it

1. Performance verdicts are **serve-based only** (the A/B harness); offline `LLM()` numbers
   are for component attribution and must never be compared 1:1 with serve TPOT/TTFT.
2. In-process timers must be taken **outside** `torch.profiler` regions; use the profiler
   only for op-mix structure (E12).
3. When probing the offline engine, keep `VLLM_ENABLE_V1_MULTIPROCESSING=0` in mind: it is
   required for monkeypatch instrumentation but changes the very overhead being measured.

## KI-5 — CANN teardown double-free after loading `vllm_ascend_C` (diagnosed, mitigated; upstream)

**Status:** diagnosed 2026-10-13 (O6 排查) · **Impact:** 进程退出路径 only,零运行期影响。

### Symptom

任何 Python 进程在载入 `vllm_ascend_C`(`TORCH_LIBRARY` 的 `_C_ascend` .so)后于解释器
终结期退出时,CANN runtime teardown 触发 double-free abort(exit code 134)。最小复证:
`python -c "import <该 .so>"` 直接 134。serve 进程从未暴露,因为编排脚本以 SIGTERM kill
收尾,引擎进程不经历 Python 终结期 teardown;只有 pytest 这类正常退出通道会踩中
(iter-018 前设备套件数值绿但退出码被 abort 污染)。

### Root cause / attribution

`_C_ascend` 的静态析构序与 CANN runtime 的进程级 teardown 顺序冲突,属上游
(vllm_ascend 构建产物 / CANN)缺陷,与 TileInfer 代码无关(证据:D14、E27 最小复证;
同源现象早在阶段一即观察为"仅 teardown 打印,不影响服务")。

### Mitigation(已落地,业务测试基础设施)

`tests/conftest.py` 在 `pytest_unconfigure` 注册 atexit 钩子:汇总输出写完后以
`os._exit(exitstatus)` 硬退出,跳过解释器终结期析构——pytest 真实退出码保住,abort 消失。
CPU 套件与不载该 .so 的设备套件零影响。裸进程场景(手动 python -i 载 vllm 后退出)仍会
踩中,属上游修复范畴;若上游修复,可移除该钩子复验。
