# Known issues and post-mortems

## KI-6 — `q_pad` staging: target slice missed `[:num_reqs]` (resolved 2026-10-20)

**Status:** resolved · **Lesson:** when a padded buffer's *source* is bounded but its *target*
is not, the bug hides until `num_reqs != bucket` — i.e. exactly at the first non-bucket-aligned
batch, which in practice is graph capture.

### Symptom

EngineCore died silently right after graph capture, with only a CANN log line (no Python
traceback):

```
[aclnn_copy.cpp:62] The size of tensor self [32, 2, 8, 256] must match
                    the size of tensor src [24, 2, 8, 256]
[aclnn_copy.cpp:117] errno[561000] ... EZ1007 / 161002
```

### Root cause

`TileInferDecodeAttention.run` staged the padded query like this:

```python
q_pad.view(bucket, kv_heads, br, head)[:, :, :group].copy_(   # target: bucket rows
    query[:num_reqs].reshape(num_reqs, ...)                   # source: num_reqs rows
)
```

Only the **source** was bounded. `copy_` cannot broadcast 24 rows onto 32, so any step whose
`num_reqs` is not exactly a bucket value (24 vs 32) raised. It had never been exercised because
earlier runs only hit bucket boundaries (8 / 32).

Note the backend's own staging (`attention/backends/tilelang_ascend950.py`) is self-consistent
— both sides use `batch` — and is *skipped* when the integration already staged into the same
buffer, so the integration copy is the only writer and the bug surfaced there.

### Fix

Bound the target as well:

```python
q_pad.view(bucket, kv_heads, br, head)[:num_reqs, :, :group].copy_(...)
```

Safe because `q_pad` is `torch.zeros`-allocated (padding rows stay 0), padding rows' table
columns are zeroed every step, their `len=1` points at page 0, and the read-back only takes
`[:num_reqs]`.

**Regression lock:** `tests/test_vllm_ascend_subpage.py::test_query_staging_slice_must_bound_live_rows`
reproduces the shape failure on CPU and scans the source for the bounded target slice.


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

**Status:** resolved (full M tile, no host restriction, 5730b66) · `tilelang-ascend950` prefill path.

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

### Resolution (full M tile, 5730b66)

Rather than obtain the AIV index, the **host precomputes each row's causal threshold** ``q_pos +
shift`` into a ``RowThr`` array (rows past the live count get ``-1``) and the kernel ``dual_copy``s it
exactly like ``s_ub`` — so each AIV half receives its own half of the thresholds and the mask is
correct on both halves without the kernel ever needing ``sid``.  The ``block_q // 2`` planning
restriction and the ``forward_prefill`` guard are **gone**; the full ``block_q`` tile is used end to
end (``tests/test_tilelang_ascend950_prefill.py`` covers multi-tile prefill, append, a partial tail
page and a ragged batch at ``block_q == 128``).  The earlier ``block_q // 2`` workaround — half the M
tile wasted, ~2x the cube work — is no longer in effect.

### Related: KV-rescan truncation (2026-10-14)

The same causal frontier that drives the mask also bounds how many KV pages a tile must read: a tile
at query position ``q_off`` only needs pages up to ``max(thresholds) = q_pos + shift``.  The prefill
kernel used to scan the request's **whole** page list per tile (``for p in T.serial(req_pages)``), so
the trailing masked pages were re-read and re-GEMMed for every tile — an O(tiles × pages) redundancy
that dominated pure-prefill long-context TTFT.  The kernel now loops ``for p in T.serial(plen)`` with
``plen`` the host-computed per-tile page count, clamped to the real page count (empty/padding tiles
run zero iterations).  Skipped pages were masked to ``-1e30`` before ``exp``, so the result is
bit-for-bit identical; measured **1.14x (8K) / 1.23x (16K) / 1.26x (32K)** on prefill.  See
[`performance.md`](performance.md#prefill-kv-重扫截断prefill-kernel-kv-rescan-truncation2026-10-14).

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

## KI-?(E295,iter-214)tilelang 0.1.15 split 内核 pass 病态 + 设备脚本 CANN 环境依赖

### Symptom

1. **裸 shell(系统 CANN 9.1.1)下任何 tilelang Ascend 内核编译必败**:`bisheng` 报
   `use of undeclared identifier 'asc_pack_to_high'/'asc_pack_to_low'/
   'asc_storealign_pack_quarter'` 与 `asc_set_gm2l1_nz_para`/`asc_load_l2_cache_mode`
   不匹配(tilelang 0.1.15 头文件按 9.2+/930 API 写)。在役全链经
   `.agent_loop/run/env.sh` 走 `.work/cann930`,不受影响;任何**独立设备脚本**
   (微基准/探针)必须 `source .agent_loop/run/env.sh`,否则必败。
2. **split-KV 内核(含 token 窗口,H67)在 dim=256 真实形状编译病态**:卡
   `tilelang/ascend/pipeline.py:83 AscendPassPipelineBody`(TVM pass pipeline)
   17min+ 不出 bisheng 子进程(100% 单线程 CPU);非 SIMD 变体(`TILEINFER_SIMD_SOFTMAX=0`)
   300s+ 同挂(faulthandler 栈证)。对照:dim=128 split 编译 ~3min 可完成;
   decode 内核 dim=256 仅 1-2min ⇒ 仅 split+token 窗口内核触发,vendor 编译器域。
3. ABI 呈现注意:独立调用 `build_decode_kernel` 后必须按在役 `forward()` 口径传参
   (`pool_view` declared-cap 视图 + `kv_indices` 补零到 cap),直接传实际页数张量报
   `violates packed ABI constraint`。

### Root cause / attribution

vendor tilelang 0.1.15(`.venv` 内)Ascend pass pipeline 对 split 内核特定形状的
编译期病态(疑似死循环/指数爆炸 pass),与 TileInfer 内核逻辑无关(token-split 数值
oracle 在 dim=128(SIMD)PASS);CANN 9.1.1 头文件不匹配为环境事实。

### Mitigation

- H67 线阻断性关闭(HYPOTHESES H67 判决,E295);复测脚本
  `.agent_loop/run/h67_toksplit_bench.py` 已留档,tilelang 升级后一键复跑。
- npu 标记套件临时轮换 `-k "not nosimd"`(D159-3),vendor 修复后应去除。
- 设备级脚本一律走 `.agent_loop/run/*_launch.sh` 启动器(内 source env.sh)。

### E315 增补(iter-231,2026-10-21)——病态面扩展与测试守卫

1. **病态面扩展**:dim=128 plain split(`paged_decode_split`)的测试/gate 规格同样挂死——
   两个独立进程同时复现:CPU 套件 `test_offline_lower_probes.py::test_decode_simd_softmax_gate`
   的 compile-only(L123)3.9h+ 零进展;NPU 套件 `test_tilelang_ascend950_decode.py::
   test_paged_decode_matches_reference` 经 `finalize_plan→build_split_kernel`(paged_decode_
   ascend950.py:1031)2h25m+ 零进展;py-spy 双栈同卡 `AscendPassPipelineBody`(pipeline.py:83)。
   上文"dim=128 split ~3min 可完成"的结论仅对当时间测的那一规格成立,不是规格族通则。
   **在役不受影响**:tl_cache_950 编译时间线显示 10-20 17:00-22:00(h75/gen4 A/B 窗)
   40+ 产物(含 split 域)正常产出,E313 权威 A/B 正常完成——病态仅限测试侧规格。
2. **测试守卫**:触发真实 split 编译的用例默认跳过(env `TILEINFER_SKIP_SPLIT_COMPILE_HANG=1`
   默认,置 0 强制跑):decode 测试 9 个 split 参数(matches_reference×4/declared_cap×1/
   full_split_plan×4)+ offline probes 的 `test_decode_split_simd_softmax_gate`(自
   `test_decode_simd_softmax_gate` 拆出,其余 9 项编译断言保留覆盖)+ token-split 设备测试
   的两个 nosimd 参数(74d6f53/后续提交,verify 全量口径实测踩实)。翻案条件=tilelang
   升级后置 0 复跑(h67_toksplit_bench.py 一键复测)。
3. **套件恢复链**(run/recover_i231.sh):h77 gate(oracle+四轴微基准)→h33v3 探针→
   h33_l2hint 基准→重挂双套件→条件挂 h77_ab.sh。

## KI-8 H77 dense NN 预排副本的权重更新流限制(H77 opt-in 域,iter-223 源码审查登记)

### 现象/限制

`tileinfer/integrations/dense_nn.py` 的 NN 预排副本 `w_nn`(加载期 `weight.t().contiguous()`,
dense_nn.py:122)是独立 tensor:若权重在 serve 生命周期内被外部原位更新
(RL 训练循环 `weight.data.copy_` 类流程),副本不会同步,apply 走 `torch.matmul(x, w_nn)`
(dense_nn.py:129)将使用陈旧权重。

### 适用域

推理 serve 场景权重仅加载一次,不受影响;仅 RL/在线权重更新流 + `TILEINFER_DENSE_NN=1`
同时成立时触发。默认关闭(opt-in),在役路径零影响。

### Mitigation

启用 RL 更新流的部署不得开启 `TILEINFER_DENSE_NN`;若需共存,须在权重更新回调后重建
`w_nn`(重跑 `_shim_layer` 前先还原 quant_method.apply)。

## KI-9 torch.profiler with_stack 在 torch_npu 下不产出 python 调用栈(iter-235 实测)

### 现象

`torch.profiler.profile(with_stack=True, record_shapes=True)` 在 torch 2.10+torch_npu
2.10.post4(CANN 9.3)组合下,导出的 chrome trace 不含 `Call stack` 字段
(23GB trace 实测 grep 计数=0),`key_averages(group_by_stack_n=8)` 的 stack 恒为空。
capture 期算子到 torch 源的栈级归因因此不可用,E327 的 H47 归因改走静态源码路径。

### 附带事实

- vLLM 0.23 V1 默认多进程架构下,主进程 profiler 完全看不到 EngineCore 子进程算子
  (top-30 全为客户端侧个位数计数微操作);`VLLM_ENABLE_V1_MULTIPROCESSING=0`
  单进程模式可让 capture 期执行留在本进程,arg_utils.py:1556 为官方支持路径。
- 单进程探针收尾可能 rc=134(SIGABRT,profiler 大 trace 析构段),数据已完整落盘,
  不影响取证产物有效性。

### 影响域

仅影响进程内画像类探针的栈级归因;在役 serve/内核/集成路径零影响。
翻案条件:torch_npu 升级修复 with_stack 后,需要栈级归因的探针一键重跑即可。
