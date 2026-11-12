# Qwen3.6 投机质量门 S-Q 证据（2026-11-12）

**结论：S2（4K MTP）按质量等价口径通过；S3 图机制通过；S4 的 32K 只算部分通过
（Q1/图/TTFT 过，接受率 81–88% 低于 4K 的 90% 门，128K 未测）。** 这是"任务级
floor + needle"，不是全量评测；不能写成"逐位一致"。

## 1. 环境

| 项 | 值 |
|---|---|
| 卡/栈 | 1 × Ascend950PR；vLLM 0.23.0 / vllm-ascend 0.23.0-950 / CANN 9.3.0 |
| 服务 | 端口 8322，`--max-model-len 33792 --max-num-seqs 4 --gpu-memory-utilization 0.82` |
| MTP | `--speculative-config '{"method":"mtp","num_speculative_tokens":1}'` |
| MTP 权重/KV | 68.04 GiB / 29.86 GiB；图 0.16 GiB |
| 非投机 | 同 cap；权重 65.52 GiB、KV 32.39 GiB |
| 图 | 6 张（PIECEWISE 3、FULL 0），`decode_calls=6000`、`hits=6000`、`fallbacks=0` |
| 安全 | 起停同一命令；端口 8322 已释放，无残留进程 |

## 2. Q1 任务正确

### 2.1 固定小套件（8 项）

`benchmarks/spec_quality_probe.py`：算术、字母计数、代码、JSON、精确输出、逻辑、
颜色列表、`len`。两臂都 **8/8**；检查是包含式/可解析，不是逐字格式门。

原始：`/tmp/spec-quality-nospec.json`、`/tmp/spec-quality-mtp.json`。

### 2.2 needle（8K/16K/32K × 10/50/90%）

MTP 全部命中 `TIGER-7421`，9/9：

| 长度 | 深度 | prompt tokens | completion tokens | ok |
|---|---|---|---|---|
| 8K | 10/50/90% | 8,203 / 8,203 / 8,202 | 213 / 256 / 214 | 3/3 |
| 16K | 10/50/90% | 16,395 / 16,394 / 16,394 | 120 / 156 / 256 | 3/3 |
| 32K | 10/50/90% | 32,779 / 32,779 / 32,779 | 199 / 256 / 146 | 3/3 |

原始：`/tmp/v3-mtp-33k-$N-d$D-needle.json`。

**Q1 判过**（套件不低于非投机 8/8；needle 9/9）。
注意：套件只有 8 个短任务，是 floor，不是 MMLU/GSM8K 这类全量评测；Q1 通过不等于
所有任务质量不变。

## 3. Q2 确定性

8K d50 同请求两次（`/tmp/v3-mtp-33k-8192-d50-needle.json` 与 `-2.json`）：

- `same=true`，`text_equal=true`，`tokens_equal=true`，`max_abs_logprob_diff=0.0`。
- 之前 16K 的两次 MTP 也完全一致。

**Q2 判过。**

## 4. Q3 性能（4K）

| 项 | 非投机 | MTP | 比 |
|---|---|---|---|
| 4K 非流式 128 token wall | 1.520 s | 1.225 s | 1.24× |
| 近似 TPOT | 9.03 ms | 6.67 ms | **1.35×** |
| 4K TTFT | 364.8 ms | 371.5 ms | 1.018× |
| 接受率 | — | 93.9% | ≥90% |

"近似 TPOT"= (wall - 同臂 4K TTFT) / 128；不是服务端直方图。**4K Q3 判过。**

## 5. 32K 长上下文（S4 / Q5 的一部分）

| 项 | 非投机 | MTP |
|---|---|---|
| 32K TTFT | 2,612.56 ms | 2,708.79 ms（1.037×） |
| 32K 非流式 128 token wall | 未同轮测 | 3.726 s |
| 接受率（33K 日志 19:13） | — | **81.4–88.2%** |
| 图 | 6 张 / 0 回退 | 6 张（PIECEWISE）/ 0 回退 |

MTP 32K 的 needle 9/9、图 0 回退、TTFT 1.037× 都过；但接受率低于 4K 的 90% 门，
且 32K 没有同轮非流式 TPOT。**S4 只算 32K 部分通过，128K 未测。**

## 6. Q4 披露

- 非逐位：8K 的 256 token 中 220 个相同（85.9%），首分叉在第 58 token；
  4K 在 char 71 分叉。top-1 一致率与 token 一致率只作诊断，不作门。
- 已在 `docs/acceptance.md` §2.3 和模型声明里写明"质量等价，非逐位一致"。
- 不得把本页的 8/8、9/9 写成"输出与基线完全一致"。

## 7. 判定

| 项 | 结果 |
|---|---|
| S2（4K MTP） | ✅ Q1/Q2/Q3 过；明确非逐位 |
| S3（MTP + 图） | ✅ 6 图、0 回退；FULL=0 |
| S4（MTP + 32K/128K） | ⚠️ 32K 部分过（Q1/图/TTFT）；接受率 81–88%、128K 未测 |
| S5（口径写进声明） | ✅ `acceptance.md` §2.3 |
| 全量质量评测 | ⬜ 未做；本页只是 floor |

## 8. 原始文件

- 套件：`/tmp/spec-quality-nospec.json`、`/tmp/spec-quality-mtp.json`
- needle：`/tmp/v3-mtp-33k-*-needle.json`（9 个）
- 确定性：`/tmp/cmp-mtp-33k-det.json`
- 32K 速度：`/tmp/v3-mtp-32k-ttft.json`、`/tmp/v3-mtp-32k-nonstream.json`
- 图：`/tmp/v3-mtp-quality-metrics.json`
- 非投机 4K 对照：`/tmp/v3-nospec-4096-nonstream.json`
