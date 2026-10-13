# Status

One page that answers two questions: **what has been done** (with the evidence) and **what is left**.
Details live in the linked documents; this file is the index and the honest scoreboard.

Last updated: **2026-10-13** — Qwen3.6-35B-A3B on Ascend950PR, stage-2 closure (M-MoE / M-GDN verdicts in, stage-3 deferred).

---

## 0. Current state (2026-10-13, authoritative)

Target scenario: **Qwen3.6-35B-A3B** (40 层:10 层 full-attention head_dim=256 + 30 层 GDN 线性注意力,
256 专家 MoE),1× Ascend950PR,vLLM 0.23.0 + vllm-ascend 0.23.0 + CANN 9.3.0 + TileLang 0.1.15。
工程事实与 A/B 数字以 `.agent_loop/PERFORMANCE_REPORT.md` 为准(第 5 节图模式双源全指标)。

**已达成(阶段一,全部引擎内验证)**:

| piece | state | evidence |
|---|---|---|
| decode d256 内核(dsplit-v2)+ 128-token 子页视图 | ✅ 设备 24/24 绿,图模式在役 | `tests/test_tilelang_ascend950_decode.py`;PERFORMANCE_REPORT §2 |
| 图捕获兼容(B5):prewarm plan + 捕获态回退 + 零同步捕获 | ✅ 图模式真实服役 | probe:6 plan READY / capture_calls=40 / fallbacks=0 / markers 随 replay 增长 |
| prefill 接线(B1):TileInfer prefill 进引擎(惰性编译+变体上限) | ✅ prefill_calls=949 | probe 与图模式 A/B 在役探针 |
| 图模式 A/B(双源:自研 9 场景 + 官方 bench 27 case) | ✅ 短上下文 TI≈FIA(比值 0.91-1.09);长上下文落后 2-6x(结构性:KI-1 split-KV 未启用 + prefill 编译窗) | PERFORMANCE_REPORT §5(TEMPORARY_PASS,环境退化期双侧比值口径) |
| 数值一致性 | ✅ top-5 logprobs 与 FIA 逐位一致 | 同报告 §1.5 |

**融合算子层扩展(阶段二)结论**:

| 方向 | 结论 | 数据 |
|---|---|---|
| M-MoE grouped GEMM 融合 | **TEMPORARY(不接入)**:内核数值 7/7 绿,但设备时间 0.86x vs aclnn(墙钟 0.968x),3 轮优化后仍低于 1.0x 契约线 | E19/H7;`tileinfer/kernels/moe/` 保留 |
| M-GDN 9a causal_conv1d / 9b decode 递推核 | **按证据关闭**:decode 侧 GDN 全族 ~0.38ms/步,无可实现收益 | E20/D28 |
| M-GDN gating 替换 | **disproven(显式 opt-in)**:引擎同侧 A/B 无收益,短上下文 TTFT +2~9%(归因 JIT 窗伪影);`TILEINFER_OPS` 显式含 `gdn` 才安装 | E21/H9/D29 |
| M-GDN 9c chunk 链融合(solve_tril+recompute_w_u) | **TEMPORARY(不接入)**:TileLang 融合核数值 6/6 绿,但微基准 0.718x@T128(门槛 ≤0.6x)、14.28x@T8192(标量块逆链不可扩展) | E22/E23/E24/H10/D31;`tileinfer/kernels/gdn/chunk_wu_ascend950.py` 保留 |

**能力边界(当前默认路径)**:split-KV 关闭(KI-1,长上下文 decode 2.8x 潜力未兑现)、
M-MoE 未接入(0.86x)、gdn gating/chunk-wu 未接入(opt-in 机制就绪,`TILEINFER_OPS`);prefill 由
TileInfer 接管但长上下文受编译窗与 KV 重扫主导(B1 价值需 L3 后兑现)。环境退化(E15 外部扫描)
期间的图模式数字均为双侧比值口径,绝对值待环境恢复后重测。

**阶段三(L1 KI-1 诊断 → L2 池无关 ABI → L3 split-KV 启用)明确延后**,未开工;任务清单见
[`next-tasks.md`](next-tasks.md)。

---

## 1. What works today(历史基线:Qwen2.5-1.5B,head_dim=128,已被上方 Qwen3.6 状态取代)

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
| **KI-1** split path faults the device (`507035`) when several shapes are compiled and run in one process | keeps split-KV off by default, so long-context decode leaves 2.8x on the table | open; 8 hypotheses refuted (see below); **诊断+启用列入阶段三 L1/L3,当前延后** |
| **KI-4** 离线 LLM() 与 serve(HTTP) 吞吐不对称(同负载 batch1 离线 136.7ms/步 vs serve eager 45.9ms) | 性能结论口径混淆 | 已定性:EngineCore 进程拓扑+GIL 串行 vs 跨进程重叠(E14);**性能结论一律以 serve 口径为准**(2026-10-13) |
| **PM-2** the causal mask needs a *global* row index, which a vector region cannot see, so prefill tiles are planned at `block_q // 2` | ~2x wasted M tile in prefill | open, cause understood, fix designed (`T.Cube()` / `T.Vector(2) as sid`) |
| 长上下文(8K/16K)图模式 TTFT 落后 FIA 2-6x | 长上下文可用性 | 结构性:KI-1(split-KV 未启用)+ D22 prefill 编译窗 + chunked prefill KV 重扫;属阶段三范围 |
| integration 的 prefill/混合步回退 FIA | 设计内降级(不是事故) | prefill 接管已落地(B1,prefill_calls>0 实证);投机解码等仍回退 |
| `head_size != 128` is declined (e.g. Qwen2.5-0.5B) | that model is served by FIA | deliberate; head_size=256 已支持(Qwen3.6 在役) |

Post-mortems and refuted hypotheses: [`known-issues.md`](known-issues.md). KI-1 has been probed for
caller-buffer alignment, KV pool size, stream ordering, the shapes themselves, both caches, the
harness' reference, the reference-vs-kernel ordering, and an exact mirror of the harness' case
construction — all refuted, each with a probe in `benchmarks/probes/`.

### 运行形态评估与状态审计(2026-10-13,阶段 3 P5)

- **O4 · eager 残余内核调用路径 —— skip(部署形态评估)**:TileInfer 服役形态 = vLLM-Ascend
  **图模式在役**(全部历史 A/B serve 探针 `capture_calls=40>0`、`fallbacks=0`、6 plan READY;
  ab 基准在采数前即断言这些前提)。图捕获把 decode 步内核编入 ACLGraph,replay 期不执行任何
  Python/主机路径,残余的 TI eager 主机调用路径(~4.4ms×10 层,D23 瘦身后,B3/E13)只在回退窗口
  (首见形状编译窗/显式回退)或 `--enforce-eager` 部署下执行——在役数字从未受其影响。结论:
  图模式部署下无收益可图,skip;若未来以 eager 模式部署,优化方向与 P4 的 FFI 主机批处理同向,
  不另行立项。
- **O7 · 状态文件 commit hash 引用审计 —— 无漂移,无需校正**:对 `docs/*.md` 与
  `.agent_loop/state/*.md` 中全部 16 个 commit hash 引用(79a9ea5×6、4370b5c×5、8fe3a26×4、
  7e1308f×3、13fffc4×2 及 86d926f/d72f31f/e76089e/49e7a11/49e6140/17260c4/bf016fb/339f4f2 各 1)
  逐一 `git cat-file -e` 核对:全部可达,且引用处的主题描述与实际 commit message 一致;文档中
  其余 7-9 位数字串均为 PID(2580095/3965161 等)非 hash。未发现历史重写漂移,无需改写。

---

## 3. What is left

The ordered, actionable list is [`next-tasks.md`](next-tasks.md) (what / why / first step / done when).
Summary of it, with status:

| # | task | value | status |
|---|---|---|---|
| T1 | recover the box, collect interrupted logs | — | done |
| T2 | clean TileInfer-vs-FIA A/B | the number everyone asks for | **done(图模式双源,2026-10-13)**:短上下文持平/长上下文 2-6x 落后(结构性),见 PERFORMANCE_REPORT §5 |
| T3 | **KI-1**: split path with several shapes in one process | unblocks 2.8x long-context decode | open — **阶段三 L1,延后** |
| T4 | prefill: recover the wasted half of the M tile | ~2x on prefill (cube-bound) | open, designed |
| T5 | SIMD softmax (replace the SIMT one) | vector time off the critical path | open |
| T6 | warm-up and batch bucketing as first-class features | what makes the integration deployable | **partly done**:prewarm 进引擎首步(B5a)+ 惰性 prefill 编译+变体上限(D22) |
| T7 | split-KV on by default | serving's main shape | blocked on T3(阶段三 L3,延后) |
| T8 | explicit prefill / fallback policy | correctness of the integration surface | **done(B1/B6)**:prefill_step 统一三态 append 语义;TILEINFER_OPS 算子粒度开关(decode/prefill/gdn/gdnwu/moe) |
| T9 | ACLGraph capture verification | the entire plan/run split exists for this | **done(B5)**:图模式在役 + NaN 修复 + 设备测试 3/3 + 引擎探针 PASS |
| T10 | FIA baseline inside the benchmark harness | "we are fast" needs the operator as yardstick | **done**:vllm-ascend FIA 即在役基线,双源 A/B 已产出 |
| T11 | three ready-to-file upstream reports | highest-value contribution we already hold | open |
| T12 | device checklist (or CI) | every device run is manual today | open |
| T13 | Phase 3 breadth: MLA, sparse, FP8/FP4, MoE, sampling | scope | MoE/GDN 已按 TEMPORARY/关闭收口(见 §0);其余未立项 |

阶段二融合算子层(M-MoE/M-GDN)已按 TEMPORARY/关闭收口,交用户决策;**当前无进行中的优化项**,
下一阶段为阶段三 L1-L3(延后待授权)。

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
