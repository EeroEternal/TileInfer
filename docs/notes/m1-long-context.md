# Qwen3.6 长上下文 L1–L3 + G2（2026-11-12）

**结论：L1 / L2 / L3 通过，G2（32K 图模式）通过。** L4（128K）/ L5（262K）未验证，
当前可用 `max-model-len` 只到 33,792；40,960 的配置在首个 8K 请求就 OOM，不能把
"模型上限 262,144"写成"单卡已支持"。

## 1. 环境与参数

| 项 | 值 |
|---|---|
| 时间 | 2026-11-12 17:22–17:37（Asia/Shanghai） |
| 卡 | 1 × Ascend950PR，131,072 MB HBM；`npu-smi` 无他人进程（使用者已确认可用） |
| 栈 | vLLM 0.23.0 / vllm-ascend 0.23.0-950 / CANN 9.3.0 |
| 服务参数 | `--max-model-len 33792 --max-num-seqs 4 --gpu-memory-utilization 0.82`，端口 8322 |
| 两臂 | v3：`TILEINFER_PLAN=/tmp/tileinfer-plan.json`；基线：`.venv_baseline`，无 TileInfer |
| 图 | `FULL_AND_PIECEWISE`，capture sizes `[1, 2, 4]` |
| 调度 | `enable_chunked_prefill=True`，`max_num_batched_tokens=2048`（默认；未显式改） |
| 启动 | v3 62 s 到 `/health`；基线 71 s |

启动内存（两臂一致）：权重 65.52 GiB、峰值激活 2.85 GiB、非 torch 0.12 GiB、
图 0.13 GiB、KV 32.39 GiB = 1,383,782 tokens；33,792 上下文最大并发 40.95×。

## 2. C1/C4：TTFT / TPOT / 吞吐

每次 1 次预热 + 3 次测量，取中位数；prompt tokens 以服务 `usage` 为准。

| N | prompt tokens | v3 TTFT ms | 基线 TTFT ms | TTFT 比 | v3 TPOT ms | 基线 TPOT ms | TPOT 比 | 吞吐比 |
|---|---|---|---|---|---|---|---|---|
| 8K | 8,202 | 646.02 | 640.90 | **1.0080** | 9.169 | 9.180 | **0.9988** | 1.0012 |
| 16K | 16,394 | 1,256.79 | 1,243.40 | **1.0108** | 9.758 | 9.740 | **1.0018** | 0.9982 |
| 32K | 32,778 | 2,612.56 | 2,592.68 | **1.0077** | 10.828 | 10.808 | **1.0019** | 0.9982 |

门：TTFT/TPOT ≤ 1.10×，吞吐 ≥ 0.95×。三档全部通过，且每档单独判，不跨档拼比值。

TTFT 随长度近似线性：8K→32K 斜率约 **0.080 ms/token**。32K prompt 被
`max_num_batched_tokens=2048` 切成 17 个 chunk（8K/16K 分别为 5/9 个），
这是当前 TTFT 的一部分，不是异常。

## 3. C2：needle 正确性（10% / 50% / 90% 三个深度）

每个长度 × 每个深度各一次，`temperature=0`，`max_tokens=256`；v3 与基线同一 prompt。
判定是"回答中含 `TIGER-7421`"，不要求模型真的只输出 code（Qwen3.6 会先 reasoning；
32 token 会把答案截断，这是第一次跑出 9/9 false 的原因，已改为 256）。

| N | prompt tokens | 深度 | v3 ok | 基线 ok | 文本/ token / logprob |
|---|---|---|---|---|---|
| 8K | 8,203 | 10% / 50% / 90% | 3/3 | 3/3 | 全同，max logprob diff = 0.0 |
| 16K | 16,395 | 10% / 50% / 90% | 3/3 | 3/3 | 全同，max logprob diff = 0.0 |
| 32K | 32,779 | 10% / 50% / 90% | 3/3 | 3/3 | 全同，max logprob diff = 0.0 |

9 对响应由 `benchmarks/compare_responses.py` 比较：`same=true`、`text_equal=true`、
`tokens_equal=true`、`max_abs_logprob_diff=0.0`。原始响应在
`/tmp/v3-$N-d$D-needle.json`、`/tmp/base-$N-d$D-needle.json`，比较结果在
`/tmp/cmp-$N-d$D.json`（`/tmp` 会被清，本页已固化关键数字）。

## 4. C3 / G2：图捕获

`kill -USR1 <EngineCore pid>` 写出 `/tmp/v3-33k-metrics.json`：

| 字段 | 值 |
|---|---|
| `plan_id` | `7d00064d487af3f6` |
| `capture_calls` | 1 |
| `captured_graphs` | **6**（PIECEWISE 3 + FULL 3） |
| `cudagraph_mode` | `FULL_AND_PIECEWISE` |
| `capture_done` | true |
| `decode_calls` / `decode_graph_hits` | 2,956 / 2,956 |
| `decode_fallbacks` | **0** |
| `errors` | 0 |
| `attention_backend` | `vllm_ascend.attention.attention_v1.AscendAttentionBackend` |

`tileinfer doctor` 输出 `PASS graph_metrics captured=6 decode=2956 fallbacks=0`。
G2 的"32K 长上下文 + 图"因此成立；G1 的 4K 8 张图是上一轮证据，不混用。

## 5. HBM 峰值 / 40960 的失败边界

- 启动时 EngineCore 进程 HBM **86,833 MB**（`npu-smi`）。
- 32K 请求期间连续采样 12 次，峰值 **107,380 MB**；请求结束后进程 101,243 MB。
  8K/16K 未单独采样峰值，32K 是这三档的上界。
- 峰值高于 0.82 预算（100.88 GiB）但仍低于 123 GiB 总容量；这就是 32K 能过、
  40K 不能过的原因之一。

40,960 / `max-num-seqs 8` / 0.88 的第一次尝试：

- 启动可用 KV 18.67 GiB，33K 时为 32.39 GiB；
- 实际占用：权重 65.52 + 激活 2.85 + 非 torch **21.22** + 图 **9.56** GiB；
- 首个 8K 请求 OOM：尝试分配 48 MiB，84.20 GiB 已分配、925.26 MiB 空闲；
- 同时把 `max-num-seqs` 从 8 降到 4、`max-model-len` 从 40,960 降到 33,792 后稳定。

因此不能把"33,792 能跑"外推成"128K 只要把 cap 调大"。128K/262K 的峰值激活、
非 torch 内存和图内存需要单独量。

### 5.1 L4 128K 单流（2026-11-12 19:18–19:27）

40,960/`max-num-seqs 8` 的 OOM 主要来自 8 seq 放大的图/非 torch 内存，不是 128K
单流不可行。改用 `--max-model-len 131328 --max-num-seqs 1 --gpu-memory-utilization 0.82`
后 128K 可服务：

| 项 | v3 | 基线 | 比 |
|---|---|---|---|
| prompt tokens（TTFT） | 131,082 | 131,082 | — |
| TTFT ms | 15,981.92 | 16,942.15 | **0.9433** |
| TPOT ms | 17.194 | 17.395 | **0.9884** |
| 吞吐（1/TPOT） | — | — | **1.0117** |

- needle 130,816 tokens × 10/50/90%，两臂各 **3/3**，逐 token 同、`max_abs_logprob_diff=0.0`。
- 图 2 张（FULL 1 / PIECEWISE 1），`decode 858/858`、`fallbacks=0`。
- 峰值 HBM **107,374 MB**（请求期间 22 次采样）；启动占用 101.01 GiB
  （权重 65.52 + 激活 2.85 + 非 torch 0.12 + 图 0.13 + KV 32.39）。
- 128K 的峰值与 32K 基本相同，KV 分配仍是主项；TTFT 16 s 约为 32K 的 6.1×，
  超线性，但 C4 仍过（比值是两臂同条件比）。

**L4 判过 C1–C4。** 128K 需要单流，不能沿用 4 seq 的 cap。

> 262K 单流已单独起过：TTFT 116.0 s、TPOT 69.1 ms，needle 尚在进行；在 3/3 完成前
> 不写 L5 通过。

## 6. 判定

| 矩阵项 | 结果 |
|---|---|
| L1 8K | ✅ C1–C4 |
| L2 16K | ✅ C1–C4 |
| L3 32K | ✅ C1–C4；含图回退 = 0 |
| G2 32K 图 | ✅ 6 张，0 回退 |
| L4 128K | ✅ C1–C4；单流；峰值 HBM 107,374 MB；图 2/0 回退 |
| L5 262K | ⏳ 262K 单流已起（TTFT 116.0 s、TPOT 69.1 ms），needle 进行中；未完成前不判过 |

未测项不得写成"支持"；长上下文声明写 **8K/16K/32K/128K 已验证**，262K 待完成。

## 7. 可复现命令

```bash
for N in 8192 16384 32768; do
  .venv_native213/bin/python benchmarks/make_long_prompt.py --tokens $N --out /tmp/prompt-$N.txt
  for D in 10 50 90; do
    .venv_native213/bin/python benchmarks/needle_probe.py --tokens $N --depth 0.$D \
      --out-prompt /tmp/needle-$N-d$D.txt
  done
done

# v3 臂（基线臂同参数，去掉 TILEINFER_*，换 .venv_baseline）
TILEINFER_PLAN=/tmp/tileinfer-plan.json \
TILEINFER_METRICS_PATH=/tmp/v3-33k-metrics.json \
setsid .venv_v3/bin/python -m vllm.entrypoints.cli.main serve /home/models/Qwen3.6-35B-A3B \
  --served-model-name qwen36 --port 8322 --trust-remote-code \
  --max-model-len 33792 --max-num-seqs 4 --gpu-memory-utilization 0.82 \
  > /tmp/v3-serve-33k.log 2>&1 &

for N in 8192 16384 32768; do
  TI_PROMPT=/tmp/prompt-$N.txt .venv_v3/bin/python benchmarks/v3_long_ttft.py \
    8322 v3-$N /tmp/v3-$N.json 3
  for D in 10 50 90; do
    .venv_v3/bin/python benchmarks/needle_probe.py --tokens $N --depth 0.$D \
      --port 8322 --logprobs --out-response /tmp/v3-$N-d$D-needle.json
  done
done
```
