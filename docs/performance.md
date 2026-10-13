# Performance

Numbers from the reference machine (single `Ascend950PR_9579`, CANN 9.3.0, official
`tilelang==0.1.15`, bf16).  Nothing here is a promise — it is a baseline with the methodology and
the known limits written down, so the next person can tell an improvement from a regression.

> **2026-10-13(权威数字)**:当前主力场景 **Qwen3.6-35B-A3B**(混合 GDN + d256 full-attn)图模式
> 双源 A/B 见下方「图模式双源全指标」节;完整流程、逐场景 P50/P99 与原始 JSON 索引在
> `.agent_loop/PERFORMANCE_REPORT.md`(第 5 节,TEMPORARY_PASS 比值口径)。本文件其余章节为
> Qwen2.5-1.5B/d128 时代的库级基线,方法论仍有效,绝对值不可与 Qwen3.6 数字混读。

## 图模式双源全指标(Qwen3.6-35B-A3B,自研源+官方源,TI/FIA 比值)

**口径注记(读数前必读)**:①测量期存在外部全盘扫描干扰(E15/H6,禁 kill 的他人进程),图模式
每步主机路径整体退化(~130ms 量级),**绝对值失真,以下均为双侧同条件 TI/FIA 比值**(TEMPORARY
口径,环境恢复后应按正常口径重跑);②split-KV 关闭(KI-1)、M-MoE 未接入(0.86x)、gdn gating/
chunk-wu 未接入(opt-in 机制就绪)——即对比的是"TileInfer decode+prefill 接管 vs FIA 全原生",
比值不含上述未启用项的贡献;③比值 = TileInfer/FIA,吞吐类 <1 为落后、TTFT/TPOT 类 >1 为落后;
④在役探针通过是数字有效前提(6 plan READY、capture_calls=40、fallbacks=0、markers 随 replay
增长、logprobs 逐位一致)。

自研源(中位;场景 1tok/128tok/512tok/8K/16K/并发 4/16/32):

| 场景 | 指标 | FIA | TileInfer | TI/FIA |
|---|---|---|---|---|
| 单 token(33 in) | TTFT ms | 563.4 | 628.9 | 1.12 |
| 128 tok 生成 | TPOT ms | 130.8 | 137.9 | 1.05 |
| 128 tok 生成 | 吞吐 tok/s | 7.64 | 7.25 | 0.95 |
| 512 tok 生成 | TPOT ms | 131.0 | 142.5 | 1.09 |
| 8K in+128 gen | TTFT / TPOT | 5470 / 136.7 | 15917 / 269.2 | **2.91 / 1.97** |
| 16K in+128 gen | TTFT / TPOT | 11298 / 142.9 | 46910 / 402.4 | **4.15 / 2.82** |
| 16K 纯 prefill | TTFT ms | 11298 | 46863 | **4.15** |
| 并发 4 / 16 / 32 | 吞吐 tok/s | 24.21 / 78.69 / 126.55 | 23.01 / 72.69 / 115.33 | 0.95 / 0.92 / 0.91 |

官方源(vllm bench serve,9 场景×3 中位;输出 tok/s、请求 req/s、TTFT/TPOT/ITL P50/P99 完整表
见 PERFORMANCE_REPORT §5.4 与 §8 终版汇总)关键行:1tok TTFT P50 1.05;128/512tok 吞吐
0.94-0.95、TPOT P50 1.06;并发 4/16/32 吞吐 0.96/0.95/0.94;8K 吞吐 **0.33**/TTFT P50 **4.04**;
16K 吞吐 **0.21**/TTFT P50 **5.76**。

解读:短上下文与并发(≤512 输入、≤32 并发)双源一致 TI≈FIA;长上下文 2-6x 为结构性
(KI-1 split-KV 未启用 + D22 prefill 编译窗 + chunked prefill KV 重扫),属阶段三 L1-L3 范围。

## P1/H10 短/长上下文长度自适应分流(gdnwu 融合核,引擎同侧 A/B,2026-10-14)

**方案**:prefill 总 token 数 T ≤ 阈值(缺省 128,微基准交叉点 E26)走 `gdn_chunk_wu` 融合核,
超阈值回退原 chunk 链;`TILEINFER_OPS` 显式含 `gdnwu` 才安装(opt-in,默认路径零改动),
`TILEINFER_GDNWU_MAX_TOKENS` 可覆盖阈值。机制经五轮在役探针验证零异常
(gdn_wu_calls/long_skips/fallbacks 语义全部符合设计,如健康成对轮 900/2040/0)。

**验收数字(健康成对,环境对齐判据:双侧 decode 同代码,8K TPOT 比值 1.0000;
wuoff=原链 vs wuon=融合核,背靠背)**:

| 场景 | wuoff | wuon | wuon/wuoff |
|---|---|---|---|
| 1tok(33 in)TTFT | 110.6 ms | 109.4 ms | 0.989 |
| 128tok(33 in)TTFT / TPOT | 108.2 / 9.26 | 109.1 / 9.28 | 1.008 / 1.002 |
| 512tok TTFT / TPOT | 108.6 / 9.55 | 108.6 / 9.56 | 1.000 / 1.001 |
| 8K TTFT / TPOT | 1487.6 / 17.16 | 1474.9 / 17.16 | 0.991 / 1.000 |
| 16K TTFT / TPOT | 3870.6 / 25.19 | 3868.8 / 25.20 | 1.000 / 1.000 |
| 并发 4/16/32 TTFT | 232.8/315.9/461.2 | 226.3/307.0/455.6 | 0.972/0.972/0.988 |

**目标形状补测(138-token prefill 进融合核,`TILEINFER_GDNWU_MAX_TOKENS=192`;同轮内用
原链场景做暖机漂移内校正)**:138tok 融合核 0.926 vs 同轮 522tok 原链 0.926 → 校正后
**收益 = 1.00x(零)**。微基准预期(E24:T128 省 ~6ms/步≈6%)未在引擎端到端兑现。

**结论**:①长上下文零劣化 proven(全部 |Δ|≤1.5%);②短上下文引擎级收益 ≈0(健康成对
−1.1%~+0.8%;目标形状校正后 1.00x),未达 ≥5% 验收判据——微基准收益 ≠ 引擎收益
(与 M-MoE E19/gating E21 同型教训,在役 A/B 为最终裁决);③处置:**不默认接入**,机制/
内核/测试保留为 opt-in(`TILEINFER_OPS=...,gdnwu`),交用户裁决是否重启(路线=E24 判读 3
平方链全 cube 化)。VERDICT: TEMPORARY_PASS。

方法注记:①健康判定看门狗(`.agent_loop/run/run_gdnwu_ab_health.sh`)以环境对齐比值
(8K TPOT |Δ|≤3%)+ 机制探针前提自动产出成对数据;②背靠背冷启动 TTFT 探针存在暖机漂移
(后跑侧长 prefill 快 8-18%),成对结论必须同轮内校验;③早期"健康带"判据(8K TTFT<700ms)
系误判——drift 轮 462ms 实为 prefill_fallbacks=980(全回退 FIA)的病态数字,TI 接管稳态即
~1700ms(E30 修正)。

## P3 prefill 预热覆盖:canonical 变体量化 + ladder 预热(2026-10-14)

**问题**:prefill 内核按 `(batch, kv_heads, group, dim, page, pool_pages, num_tiles, block_q,
dtype)` 编译变体,真实流量形状空间近连续(任意 prompt 长度×任意并发),首见形状惰性编译
~110s(D22)形成 TTFT 编译窗,是图模式 A/B 长上下文 2-6x 落差的结构性成因之一。

**方案(D39)**:变体量化归一——批维 padding 到 `bucket(max_num_seqs)+1` 行(末行为 trash 行:
0 页、整页 last_page_len→kv_len=0;padding tile 指向它→内核页循环零迭代、输出不读);
tile 维向上量化到 2 的幂(padding tile `q_len=0`:打包/解包跳过、RowThr=-1 全掩码)。
canonical 变体 = `(B, 2^⌈log2 t⌉)`,prewarm 按 token 预算(`max_num_batched_tokens`)推导
ladder:顶 = `next_pow2(ceil(B·group/128)+max_reqs)` 与 `PREFILL_MAX_TILES` 取小 →
变体数 = log2(top)+1 ≈ 11 ≤ 上限 32,覆盖引擎可调度的**全部** prefill 形状。

**验收数字(E34/E34b)**:

| 项 | 数字 | 判据 |
|---|---|---|
| 设备测试(量化路径数值 vs reference_attention,TOL 3e-2) | 4/4 绿(非 2 幂 t=19→32、批 1→5 trash 行、append、decode 行混批) | 通过 |
| ladder 预热后任意 ladder 内形状可用性 | qo∈{1,17,97,257,397,512} 全部即时 usable | 通过 |
| 首见形状首步(API 级,qo=397) | **0.008s** | <5s ✓ |
| 引擎首发首见形状 ~700/~1500/~2600 tok TTFT | **1.43 / 2.59 / 1.54 s** | <5s ✓ |
| 引擎在役探针 | prefill_calls=30(3 请求×10 层,首发即接管)、prefill_fallbacks=0、prefill_exceptions=0 | 通过 |
| decode 侧不受影响 | fallbacks=0、capture_calls=40、ready_plans=6、markers 增长、exceptions=0 | 通过 |

**成本注记**:预热单价 ~109s/变体(与 D22 惰性编译同价)× ladder ≈11 → 冷启动一次性
~16-20min(实测 ladder 9 新编译 16min;T=512/1024 由启动 dummy 的后台编译并行完成),
磁盘缓存持久化后热启 0.2s;量化 waste ≤2x tile 数但 padding tile 指零页行,实际零 KV
计算;超 ladder 形状(配置变更)仍走惰性编译回退,行为同 D22。VERDICT: PASS。

Reproduce with:

```bash
.venv/bin/python -m pytest tests/test_vllm_ascend_prefill_prewarm.py -q -s
bash .agent_loop/run/probe_p3_prefill_warm.sh
```


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
