# Qwen3.6 投机解码冒烟：ngram / MTP k=1（2026-11-12）

**结论：投机路径能起、能出结果，MTP 有约 1.35× 的非流式加速；但严格 C2（与不启用
投机逐位一致）未过。** 在验收口径定下来之前，S2 不能写成"通过"。

## 1. 跑了什么

| 项 | 配置 |
|---|---|
| 卡/栈 | 1 × Ascend950PR；vLLM 0.23.0 / vllm-ascend 0.23.0-950 / CANN 9.3.0 |
| 服务 | 端口 8322，`--max-num-seqs 4 --gpu-memory-utilization 0.82` |
| ngram | `--max-model-len 8192`（4K）与 `16384`（8K）；`{"method":"ngram","num_speculative_tokens":1,"prompt_lookup_max":4,"prompt_lookup_min":2}` |
| MTP | `--max-model-len 16384`；`{"method":"mtp","num_speculative_tokens":1}` |
| 对照 | 同 cap 的 v3 非投机服务；4K/8K prompt 与 needle 都复用长上下文那套 |
| 安全 | 每步结束即停；端口 8322 已释放，无残留进程 |

MTP 启动日志确认：`Resolved architecture: Qwen3_5MoeMTP`，draft 与 target 共享
embedding，权重 **68.04 GiB**（比非投机多 2.52 GiB），16K cap 下 KV 29.86 GiB；
ngram 的 capture sizes 为 `[1,2,4,8]`，图内存 0.16 GiB。

## 2. S1：ngram 冒烟

- 4K 与 8K 都能起来并返回完整回答；4K needle `ok=true`（含 `TIGER-7421`）。
- `capture sizes` 从 `[1,2,4]` 变为 `[1,2,4,8]`，说明投机槽位被计入图捕获。
- **未采集到 ngram 的投机 token 统计**（服务已停，没抓 `/metrics`）；S1 的
  "投机 token 统计"证据缺一块，因此只算 **C1 通过**，不算完整 S1。

## 3. S2：MTP k=1 的速度

MTP 的流式探针会把多个 token 合在一个 delta 里，`out_toks=17`，直接拿它算 TPOT 会
高估（第一次算出 12.2 ms 是错的）。改用**非流式、`ignore_eos=true`、固定 128 token**，
并用同臂 4K TTFT 扣掉预填充：

| 臂 | wall (s) | completion tokens | 4K TTFT (ms) | 近似 TPOT (ms) |
|---|---|---|---|---|
| 非投机 | 1.520 | 128 | 364.8 | 9.03 |
| MTP k=1 | 1.225 | 128 | 371.5 | **6.67** |

近似加速 **1.35×**（TPOT 比 0.739）。这个数字是"同 prompt、同 128 输出、扣预填充"的
口径，不是 vLLM 的 `time_per_output_token` 直方图；后续要用服务端指标复核。

MTP 接受率（日志）：

- `Mean acceptance length: 1.94`
- `Accepted: 124 / Drafted: 132`，`Avg Draft acceptance rate: 93.9%`
- 位置接受率 0.939

## 4. S3：MTP + 图

`/tmp/v3-mtp-metrics.json`：

| 字段 | 值 |
|---|---|
| `captured_graphs` | 6（PIECEWISE 3 + **FULL 0**） |
| `cudagraph_mode` | `PIECEWISE`（启动配置写的是 FULL_AND_PIECEWISE） |
| `capture_sizes` | `[1,2,4,8]`，max capture 8 = 4 × (1+1) |
| `decode_calls` / `hits` | 600 / 600 |
| `decode_fallbacks` | **0** |
| `errors` | 0 |

图机制本身能接上、稳态零回退；但 FULL 图在 MTP 下没有捕获，S3 不能脱离 S2 单独判过。

## 5. C2：严格逐位失败（关键）

同一 8K needle、同 cap 16384、temperature 0：

| 对比 | 首个分叉 | 最大 logprob 差 |
|---|---|---|
| MTP vs 非投机，8K needle | token 58：`' specific'` vs `' hidden'` | 0.764 |
| ngram vs 非投机，8K needle | token 10：`'1'` vs `' **'` | 1.415 |
| MTP vs ngram，8K needle | token 10：`' **'` vs `'1'` | 1.460 |
| MTP vs 非投机，4K 非流式 128 token | char 71：`'The user pasted'` vs `'The user has pasted'` | 未请求 logprobs |
| MTP 两次自比 | 无 | 0.0 |

4K 也分叉，说明这不是"长上下文/cascade 只在长 prompt 才触发"的问题；投机路径在短
prompt 上就已经改变输出。MTP 自身是确定的（两次完全一致），所以不是采样噪声。

候选原因验证（2026-11-12 加跑）：

- 用 `--no-async-scheduling` 重跑 MTP：启动日志不再出现 `Disabling cascade attention`，
  但输出与异步 MTP **完全一致**（`same=true`），仍与非投机在第 58 token 分叉；
  4K 也仍在 char 71 分叉。**async scheduling / cascade 假设被排除（至少不是唯一原因）。**
- 该次 metrics 仍是 PIECEWISE、FULL=0，说明 MTP 下 FULL 图没有捕获，和 async 开关无关。

### 5.1 图路径也排除（2026-11-12 eager/PIECEWISE 对照）

| 对比 | 8K | 4K |
|---|---|---|
| 非投机 FULL vs 非投机 PIECEWISE | same，max 0.0 | same，max 0.0 |
| MTP eager vs MTP PIECEWISE/sync | same，max 0.0 | same，max 0.0 |
| MTP eager vs 非投机 eager | **分叉 token 58**，max 0.764 | **分叉 char 71** |
| MTP PIECEWISE vs 非投机 PIECEWISE | **分叉 token 58**，max 0.764 | **分叉 char 71** |

图模式（FULL/PIECEWISE）、eager/compile、async scheduling、cascade attention 都已排除：
非投机在两种图模式下输出完全相同，MTP 在 eager/PIECEWISE 下也完全相同；**只要走
spec（MTP 或 ngram）就与非投机分叉**。分叉在投机执行/验证本身，最可能在 target
logits 的取位、draft 接受/拒绝，或 GDN 的 spec state 更新；尚未定位到具体一行。

### 5.2 数值漂移的起点

| 生成步 | MTP vs 非投机（logprob 差） | ngram vs 非投机 |
|---|---|---|
| 0 | 0.0 | 0.0 |
| 1 | 5.36e-6 | 0.0 |
| 2 | **4.36e-4** | 0.0 |
| 3 | 5.25e-6 | 0.0 |
| 4 | 3.61e-5 | 0.0 |
| 5 | 8.3e-7 | 0.0 |
| 首个 token 分叉 | **58** | **10** |

ngram 在前 6 步的 logprob 与非投机**完全相同**，直到第 10 个 token 才分叉；MTP 从
第 2 个 token 起就有微小差异。两者都指向"投机多 token forward 改变了 target 的数值
路径"，而不是采样噪声：MTP 的 draft token 从第一步就进入 batch，ngram 在首个被接受
的 draft 之前保持精确。这与 GDN spec state / 多 token 前向的数值差一致；尚未证明是
具体哪一行。

在这台 MoE/GDN 模型上，早期 token 的分叉就足以翻转 top-1，所以不能拿"答案看起来对"
代替逐位。矩阵原口径是"先按逐位一致跑，不一致就查，不调容差"；现在就是"不一致"。

## 6. 判定

| 项 | 结果 |
|---|---|
| S1 ngram | ⚠️ C1 通过；投机 token 统计未采集 |
| S2 MTP k=1 | ⚠️ 可跑、93.9% 接受率、约 1.35×；严格 C2 未过 |
| S3 MTP + 图 | ⚠️ 6 图/0 回退，但依赖 S2 |
| S4 MTP + 长上下文 | ⬜ 未跑 |
| S5 口径写进声明 | ⬜ 未定，正是当前阻塞 |

**不能写"MTP 已支持/已通过"**；最多写"厂商路径可加载、可出结果、有加速，但输出等价性
未达标，验收口径待定"。

## 7. 下一步（需要选口径）

1. **严格逐位**：当前 MTP/ngram 都不合格。cascade/FULL/async 已排除；MTP 从第 2 个
   token 起漂移、ngram 到第 10 个 token 才漂移，指向 GDN spec 多 token 前向。继续需要
   插桩 target logits / 改厂商代码，或报 issue；不是启动参数能修的。
2. **定义容差**：例如前 32 token 逐位、之后 top-1 一致率 ≥ X%、或与"同 attention 路径的
   非投机"比。这必须写进 `manifest`，不能临时放宽。
3. **先不判投机**：转去做并发 P2/P3、冷启动 K1/K2，把 S2 留作阻塞项。

## 8. 原始文件

- `/tmp/v3-mtp-8k.log`、`/tmp/v3-mtp-metrics.json`
- `/tmp/v3-mtp-8192-d50-needle.json`、`/tmp/v3-mtp-8192-d50-needle-2.json`
- `/tmp/v3-nospec16k-8192-d50-needle.json`
- `/tmp/v3-ngram-8192-d50-needle.json`
- `/tmp/v3-mtp-4096-nonstream.json`、`/tmp/v3-nospec-4096-nonstream.json`
- `/tmp/cmp-mtp-nospec16k.json`、`/tmp/cmp-mtp-self.json`
