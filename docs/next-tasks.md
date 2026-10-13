# Next tasks

> Where each of these stands, and what is already finished, is in [`status.md`](status.md).

A working list, ordered by value ÷ cost.  Each entry says **what**, **why**, the **first step** (a file
or a command, not a plan) and **done when**.  Everything here is grounded in what the repo already
contains; nothing is speculative.

## Where the project stands(2026-10-13 刷新)

| | state |
|---|---|
| Decode kernel (paged, GQA, NHD, tail masking, **head_dim 128+256**) | ✅ validated on device(d256 经 dsplit-v2,24/24 绿)|
| Split-KV kernel + device merge kernel | ✅ written and validated (3.0x at b1/32k–64k), **opt-in** because of KI-1(启用=阶段三 L3,延后) |
| Prefill / append kernel (causal, ragged, paged) | ✅ validated + **已接线进引擎**(B1,prefill_calls=949 实证) |
| Plan / scheduler / metadata / workspaces | ✅ implemented, CPU 套件全绿 |
| vLLM-Ascend integration | ✅ **Qwen3.6-35B-A3B 图模式真实在役**:6 plan READY、capture_calls=40、fallbacks=0、logprobs 逐位一致;双源 A/B 数字落盘(PERFORMANCE_REPORT §5) |
| Performance baseline (decode, unsplit) | ✅ 138–147 GB/s at wide shapes, 51 GB/s at batch 1(启用 split-KV 可 3x,受 KI-1 门禁) |
| 融合算子层(M-MoE / M-GDN) | ⏸ **收口**:M-MoE TEMPORARY(0.86x 交决策)、M-GDN 9a/9b 关闭、gating disproven(opt-in)、9c chunk-wu TEMPORARY(0.718x 未过门槛)——详见 [`status.md`](status.md) §0 |
| Device CI | ⏳ none yet |

Open bugs and post-mortems live in [`known-issues.md`](known-issues.md) (KI-1 split-path instability,
KI-4 离线/serve 口径,PM-1/PM-2 resolved) and the environment traps in [`architecture.md`](architecture.md).

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

*Status*: **done(2026-10-13,图模式双源)** — 自研 9 场景×3 + 官方 bench 27 case 全落盘,在役探针
通过为前提。结论:短上下文+并发 TI≈FIA(比值 0.91-1.09);长上下文 8K/16K 落后 2-6x(结构性:
KI-1 split-KV 未启用 + prefill 编译窗,属阶段三)。数字与口径:`.agent_loop/PERFORMANCE_REPORT.md`
§5 与 [`performance.md`](performance.md)。环境退化期按双侧比值口径(TEMPORARY),绝对值待环境
恢复后建议重跑一次。

### T3 · KI-1: the split path faults the device with several shapes in one process(**阶段三 L1,延后待授权**)

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

### T7 · Split-KV by default(**= 阶段三 L3,延后**;依赖 T3/L1 与 L2 池无关 ABI)

*Why*: it is the main performance feature for the shape serving actually sees (single long request).

*First step*: T3, then flip the default and extend the device tests to cover `kv_tile_pages > 0` in a
multi-shape process.

---

## P2 — engine integration(核心项已完成)

### T8 · vLLM-Ascend: the prefill path and the compatibility surface ✅(**done,B1/B6,2026-10-13**)

*Done*: prefill_step 统一 PrefillNoCache/PrefillCacheHit/ChunkedPrefill 为 append 语义(K/V 经
reshape_and_cache 入池);惰性后台编译+变体上限 32+blocks 上限(D22);`TILEINFER_OPS=decode,prefill,gdn,gdnwu,moe`
算子粒度开关(每步读取,秒级回退,B6);设备测试 3/3 + 引擎内 probe PASS(prefill_calls=18/949)。

### T9 · ACLGraph capture verification ✅(**done,B5,2026-10-13**)

*Done*: 图模式 Qwen3.6 真实在役——plan 预构建提前到 warmup(B5a)、捕获态优雅回退(B5b)、
在役探针(capture_calls/markers/fallbacks,B5c)、稠密→ragged 设备侧重建修复图模式 NaN(B5d/D21);
设备测试 tests/test_vllm_ascend_graph_stage.py 3/3;图模式端到端生成正常文本 + top-5 logprobs 与
FIA 逐位一致。

### T10 · FIA baseline inside the benchmark harness ✅(**done**)

*Done*: 基线 = vllm-ascend 原生 FIA 算子路径(同 serve 脚本 `TILEINFER` 开关翻转);自研
`benchmarks/ab_qwen36_full.py`(预热+3 次中位+在役探针断言)与官方 `vllm bench serve` 双源产出;
官方客户端三连坑已修(D25:本地路径+served-model-name+chat endpoint+离线)。

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

**进展(2026-10-13)**:MoE grouped GEMM 与 GDN 线性注意力已按 TEMPORARY/关闭收口
(`tileinfer/kernels/{moe,gdn}/` 保留,opt-in 机制就绪);FP8/量化 KV、cascade/前缀缓存、投机解码、
采样融合、RMSNorm 小算子、autotune 框架等已裁决移除(BACKLOG「已裁决移除」节),勿再实现。

---

## P-DEF — 阶段三长上下文包(⏸ 明确延后,待用户授权后按 L1→L2→L3 序开工)

### L1 · KI-1 诊断(split-KV 多形状同进程 wedge)

*What*: 定位 split-KV 内核多形状同进程 507015/507035 wedge 的根因(8 个假设已排除,探针齐全)。
*First step*: `ASCEND_LAUNCH_BLOCKING=1` 下最小碎片 sweep(`benchmarks/probes/ki1_pool_size.py` 模板)。
*Done when*: 可上游提交的复现器 + 根因 + 多形状进程设备测试绿。

### L2 · 池无关内核 ABI —— ✅ DONE(2026-10-14)

*What*: KV 池指针移出内核签名(plan 携带),使预烧免精确池匹配——vLLM 池页数不可复现是预烧失效
的已知根因(status.md §5.2)。**内核签名变更,落地后必须重跑引擎内回归与预烧链路**。
*Done when*: 预烧 key 与池解耦,引擎内 prewarm→READY 全绿。
*Result*: `declared_pool_cap`(pow2 上取整/`TILEINFER_POOL_CAP` 钉死)进入内核 K/V 首维、
kv_indices 槽与 plan key;池以 `Tensor.set_` 容量视图零拷贝呈现(as_strided 拒超界形状,
extent=1 占位被否证——拷贝描述符对 size-1 维忽略动态 pid)。全量回归 EXIT=0(125 dots,含
cap>pool 设备绑定用例);probe_b5 引擎探针 PASS(6 plan + 9 prefill 变体 READY,
capture_calls=40、fallbacks=0);跨进程漂移池(9425→9899/9600)预烧命中同一产物(10s vs
83s 首编)。证据 E38;一次性成本=键变更冷启动重编译 ~25min(已付)。

### L3 · split-KV 真正启用

*What*: 按 shape 预烧(≈140s/shape)+ `TILEINFER_ALLOW_SPLIT_KV` 默认化 + 长上下文(≥8K)引擎内
A/B,兑现 standalone 已证的 2.8x。依赖 L1/L2。
*Done when*: 图模式 serve 下 8K/16K 单流 TTFT/吞吐对比 FIA 落盘且无回退。

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
