# 性能：口径、基线、当前数字

> 本文只回答三件事：**怎么测**、**基线是多少**、**现在差多少**。
> 优化方案与优先级在 [`acceptance.md`](acceptance.md) 与 [`architecture.md`](architecture.md)；
> 每个候选算子的离线比值在 [`notes/perf-candidates.md`](notes/perf-candidates.md)。

---

## 0. 怎么读这份文档

- **v1** = vllm-ascend 托管时期（2026-10）。那时 TileInfer 是 vllm-ascend 里的一个
  attention backend，走官方图模式。**v1 的数字不是当前后端的数字**，但它是"不低于现网"
  这句话的分母，必须留着。
- **v2** = 当前的原生 vLLM 后端（`tileinfer/vllm/`）。仍在追赶。
- **v3** = 兼容适配层形态（2026-11-12 起）：平台/worker/KV/图捕获归 vllm-ascend，TileInfer
  只做选择与计数。v3 的 M1 数字见 §0.1，**不要和 v1/v2 的表混算**。
- 数字后面的 **[已核实]** 表示有落盘日志与命令可复现；**[待复核]** 表示只在某次窗口测过，
  换窗口可能变，别当结论引用。

### 0.1 v3 M1（同一 vllm-ascend 栈 + bridge，2026-11-12）

口径：单卡 Ascend950PR，CANN 9.3，`max-model-len 9216`，图模式
`FULL_AND_PIECEWISE`，A/B 交替 A1→B→A2，每场景 3–5 次中位数；并发行
`max-num-seqs == 客户端并发`。完整表与命令见 [`notes/m1-v3-bridge.md`](notes/m1-v3-bridge.md) §4。

| 场景 | 基线 → v3 比值（TTFT / TPOT / 吞吐） |
|---|---|
| 单流 1 token | 0.907 / – / – |
| 单流 decode 128 | 0.970 / 1.000 / 1.000 |
| 单流 decode 256 | 1.016 / 1.001 / 0.999 |
| 4K prompt（3939 tok） | 1.000 / 0.999 / – |
| conc_4（上限 4） | 1.034 / 0.999 / 0.997 |
| conc_16（上限 16） | 0.997 / 0.997 / 1.003 |

结论：在这个窗口内，v3 与基线同速（门是 1.10× / 0.95×）；这是"把平台还给厂商"的结果，
不是 TileInfer 内核的加速。

---

## 1. 口径纪律

这几条是血的教训，违反其中任何一条得出的数字都是废的：

1. **只有同一时间窗内成对跑出来的比值可以引用。** 跨窗口漂移实测可达 2×。
2. **`--max-num-seqs` 必须等于客户端并发。** 它默认是 256；配成 4 而客户端打 16 路，
   数字反映的是排队，不是性能。曾经的"吞吐只有 18.5 tok/s"就是这么来的。
3. **offline `LLM()` 与 serve(HTTP) 的数字不可比**（见 `known-issues.md` KI-4）。
4. **算子级加速不等于端到端加速。** 融合卷积算子级 42.5×，端到端 0.995——启动与接线成本吃掉全部收益。
5. **报数必须带**：命令、serve 参数、模型路径、是否图模式、测量日期。
6. 详细约束见 [`notes/perf-comparison-rules.md`](notes/perf-comparison-rules.md)。

---

## 2. v1 基线（vllm-ascend 图模式，历史对照原点）

**口径**：`vllm bench serve --backend openai --endpoint /v1/completions --dataset-name random
--random-range-ratio 0 --num-prompts 32 --ignore-eos --temperature 0 --metric-percentiles 50,90,99`；
serve 参数 `--max-model-len 32768 --max-num-seqs 32 --gpu-memory-utilization 0.90
--language-model-only`，图模式 `FULL_AND_PIECEWISE`，bucket `[1,4,16,32]`，chunked prefill 默认。
**off 侧** = 官方 FIA 路径；**on 侧** = TileInfer 接管 decode+prefill 并开 split-KV。
单卡独占，双侧背靠背（2026-10-14 16:25–16:41）。

吞吐（输出 tok/s，>1 表示 TileInfer 更优）：

| 场景 | FIA | TileInfer | TI/FIA |
|---|---|---|---|
| 128/128/C1 | 105.8 | 93.9 | 0.887 |
| 128/128/C8 | 412.4 | 452.3 | 1.097 ✓ |
| 128/512/C8 | 427.0 | 487.7 | 1.142 ✓ |
| 8K/128/C8 | 159.7 | 34.0 | **0.213** |
| 16K/128/C8 | 95.0 | 11.5 | **0.121** |

TPOT P50（ms）：

| 场景 | FIA | TileInfer | TI/FIA |
|---|---|---|---|
| 128/128/C1 | 8.77 | 9.88 | 1.126 |
| 128/128/C8 | 18.5 | 15.2 | 0.823 ✓ |
| 128/512/C8 | 18.8 | 15.8 | 0.840 ✓ |
| 8K/128/C8 | 41.1 | 192.9 | **4.696** |
| 16K/128/C8 | 69.9 | 589.3 | **8.429** |

TTFT P50（ms）：128/128/C1 93.7 → 118.5（1.264）；128/128/C8 188.3 → 328.2（1.743）；
128/512/C8 190.6 → 336.1（1.764）；8K 1059.8 → 5836.9（5.507）；16K 1720.5 → 14450.2（8.399）。

**v1 的结论**：短上下文并发档净正（吞吐 +9.7~14.2%，TPOT/ITL 0.82–0.85×），
单流略落后，**长上下文 × 并发是引擎级病态**（4.7–8.4×）。这个病态在 v2 里换了形态但没消失。

> 这张表是 v1 的，不要和下面 v2 的数字混读。v2 现在的任务就是把这几个 0.2× 追回来。

---

## 3. v2 当前数字（原生后端，2026-11-03 起）

**服务命令**：`--attention-backend CUSTOM --enforce-eager --max-model-len 9216
--max-num-batched-tokens 9216 --max-num-seqs 32 --gpu-memory-utilization 0.75`。
**关键差别**：`--enforce-eager`，即**没有图模式**——这是当前最大的一处系统性落后。

| 指标 | v2 TileInfer | vllm-ascend 基线 | 倍数 | 门 | 状态 |
|---|---|---|---|---|---|
| TPOT | 95.8 ms | 37.93 ms | **2.53×** | ≤1.10× | 未达 [已核实] |
| TTFT @ prompt 512 | 1.215 s | 0.103 s | **11.8×** | — | 未达 [已核实] |
| 吞吐 @ 并发 16 | 37.69 tok/s | 待同窗重测 | — | ≥26 tok/s | 达标 [待复核] |
| 吞吐 @ 并发 32 | 38.64 tok/s | 待同窗重测 | — | ≥26 tok/s | 达标 [待复核] |
| prefill 内核占比 | 1.000 | — | — | 自检项 | 达标 [已核实] |

一步 102 ms 的粗分账（[已核实] 的设备端分段）：

| 段 | 时间 |
|---|---|
| GDN（30 层线性注意力） | 19–30 ms |
| MoE（256 专家，routed） | 32 ms |
| attention + 其余 | 剩余 |
| **合计** | ≈102 ms |

prefill 侧：固定 314 ms + 1.88 ms/token；每请求另有约 310 ms 的启动成本。

**读法**：TPOT 2.53× 里有相当一部分是"每步固定开销"（提交/同步/空转），不是算力不够——
这正是图捕获要解决的；另一部分是 GDN 与 MoE 的实现效率。

---

## 4. 已经定位的差距与归因

| 差距 | 归因 | 证据强度 |
|---|---|---|
| 每步固定开销高 | 无图模式，每步重新提交全部 kernel | 强：eager/capture 对照实验 |
| GDN 段 19–30 ms | 多次小 kernel + 段内同步 | 中：分段计时 |
| MoE 段 32 ms | routed 路径未达理想带宽 | 中：离线比值 2.569×（sorted）已进默认 |
| TTFT 11.8× | prefill 有固定 314 ms + 每 token 1.88 ms，且首见编译窗 | 强：分相计时 |

**已经翻过的错误结论**（留在文档里防止重犯）：

- "图捕获在 950 上不可达，`EE1016` 是平台限制" —— 错。v1 在服务里跑过图模式，
  `capture_calls=40`、`fallbacks=0`、logprobs 逐位一致。v2 的失败是我们自己的捕获路径没接完。
- "融合卷积能显著提升端到端" —— 错。算子级 42.5×，端到端 0.995。
- "MoE 设备端路径更快" —— 错。实测 0.818×，已否决。

---

## 5. 怎么测

**唯一入口**是 `scripts/ti_trace.py`（不要再往 `benchmarks/` 里散落新脚本）：

```bash
python3 scripts/ti_trace.py config
python3 scripts/ti_trace.py presets
python3 scripts/ti_trace.py run --preset smoke --dry-run
python3 scripts/ti_trace.py run --preset smoke
```

它负责：查显存门（被占用就拒绝跑，不会去抢别人的卡）、加设备锁、起服务、按预设打流、
采 TTFT/TPOT/吞吐、把 `plan.json`/`meta.json`/`summary.md`/`stages.csv`/`raw/serve.log`
落到 `trace_out/<时间戳>-<预设>/`。

分段计时的环境变量（`tileinfer/serving/phases.py`）：

| 变量 | 作用 |
|---|---|
| `TILEINFER_PHASE_TIMING` | `0`/`1`/`host`/`device`/`both`，双时钟（host 用 `perf_counter`，device 用 `torch.npu.Event`） |
| `TILEINFER_PHASE_OPS=detail` | 展开到算子级 |
| `TILEINFER_PHASE_LEAF_LIMIT` | 叶子节点数量上限 |
| `TILEINFER_PHASE_RESET_AFTER_S` | 定时重置窗口，避免长时间累积 |
| `TILEINFER_PREFILL_TIMING` | 单独打 prefill 分段 |

**统计口径**：一律用中位数与 MAD 判异常（`|x − med| > 6 × 1.4826 × MAD`），**不要用均值与标准差**——
均值会把"偶发一次 3 秒的编译"平摊成"一切正常"。
