# P2 基线：流式 TTFT 与 decode 实测（vLLM 路径）

> 服务：`--max-model-len 2048 --max-num-seqs 1 --gpu-memory-utilization 0.3 --no-enable-prefix-caching`
> 启动耗时：权重 86.43 s，卡上 10.51 GiB。脚本：`tools/dsv41/perf_ttft.py`（流式，取第一个 SSE chunk）。

## 1. 实测

| prompt | TTFT P50 | P90 | 极差 |
|---|---|---|---|
| 「The capital of France is」(≈10 tok) | **4.34 s** | 4.47 s | 3% |
| 同一句撑到 ≈100 tok | **7.32 s** | 7.45 s | 2% |
| 撑到 ≈400 tok | **10.80 s** | 10.88 s | 1% |
| 撑到 ≈1500 tok | **15.16 s** | 15.27 s | 2% |

（另有一组中文 prompt，因我的 token 粗略换算不准，数值不可直接比较，仅作定性参考：6.09 / 11.28 / 16.48 s。）

## 2. 两个结论

**结论 A：P2 的验收目标未达成。** 目标写的是"2K prompt 的 TTFT ≤ 5 s"，
实测 1.5K prompt 已需 15.2 s，差约 3 倍。

**结论 B（更重要）：存在约 4.3 s 的固定开销。** 10 个 token 的 prompt 也要 4.3 s，
而这条路径的 decode 是 ~700-800 ms/token，说明 4.3 s 不可能只是 10 个 token 的 prefill 成本。
按 40 层倒推，相当于每层约 100 ms 的一次性开销。

嫌疑（都在我们自己的代码里，不是 vLLM）：
1. `ModelState` 每个请求重建（`forward` 里 `pos == 0` 就新建），其中 `LayerRuntime.build()` × 40 层；
2. `_runtime_moe()` 首次调用要建 `ExpertArena`（384 槽、pinned 内存）；
3. 每请求都要重新做一次"专家页入"的冷启动（6 专家 × 40 层）。

## 3. 下一步（先量后改）

在 `forward` 里加计时，把一次请求拆成：`ModelState 构造` / `arena 构建` / `transformer_forward` 三段，
确认 4.3 s 落在哪一段。若落在前三段，则做"跨请求复用"（state 池 + arena 常驻），
这属于纯 Python 改动，收益可能远超 P1 的 kernel 优化。

## 4. 口径说明

`perf_ttft.py` 目前用 SSE **chunk 数**估算 decode 速率，而一个 chunk 可能含多个 token，
所以表里的 "decode ms/token" 实际是 "ms/chunk"。下一轮改用 `stream_options.include_usage` 取
`usage.completion_tokens` 做分母，才符合差分法口径。

## 5. 三段计时：4.3 s 不在 state/arena（假设被推翻）

在 `tileinfer/vllm/models/dsv41.py::forward` 加了 `DSV41_TIMING` 探针（`state` / `moe_setup` / `fwd`），
一次短请求的实测：

```
DSV41_TIMING state=740ms moe_setup=631ms fwd=914ms   ids=1  pos=0    <- warm-up dummy
DSV41_TIMING state=0ms   moe_setup=0ms   fwd=2027ms  ids=5  pos=0    <- 真实 prefill
DSV41_TIMING state=0ms   moe_setup=0ms   fwd=831ms   ids=1  pos=5    <- decode
DSV41_TIMING state=0ms   moe_setup=0ms   fwd=713ms   ids=1  pos=6
DSV41_TIMING state=0ms   moe_setup=0ms   fwd=761ms   ids=1  pos=7
```

**结论 1：`state`（740 ms）与 `moe_setup`（631 ms）只在第一次调用发生**，真实请求里都是 0 ms。
所以"每请求重建 ModelState / 重建 Arena"**不是**固定开销的来源（§2 的假设 1、2 被推翻）。

**结论 2：prefill 5 个 token = 2027 ms，约 405 ms/token**；decode 是 713–831 ms/token。
即 prefill 的批处理收益只有约 1.8 倍——这本身是 MoE 逐专家路径的固有特性（同层专家只页入一次，
但计算仍随 token 数线性增长）。

**结论 3：4.3 s 的 TTFT 里，我们的 `forward` 只占 2027 ms**，剩下约 2.3 s 在
**HTTP 层到 `forward` 之间**（vLLM 的 tokenize / 调度 / 采样 / SSE 首块，或 chat 模板渲染）。

## 6. 下一轮：定位那 2.3 s

用两条对照（都不改代码）：
1. `/v1/completions` 与 `/v1/chat/completions` 各测一次流式 TTFT —— 差值即 chat 模板渲染成本；
2. `max_tokens=1` 的最小请求 —— 排除采样/解码干扰。

若两者都在 4 s 量级，则固定开销在 vLLM 的请求路径上（我们可控性低），优化重点应转向
**prefill 的 405 ms/token**（这才是长 prompt TTFT 的主导项：1500 token × ~10 ms 已说明长 prompt 时
单 token 成本会大幅下降，说明 4 s 更像是"首请求/首块"固定成本）。

## 7. 更正：TTFT 的构成（不是 "prefill + 2.3 s 黑箱"）

上一节把 TTFT 与 `fwd=2027ms` 直接相减得出 "2.3 s 在 HTTP 层"，**这个算法是错的**：
TTFT 是"到第一个生成 token 可读"的时间，它包含
**prefill + 第一个 decode 步 + 服务层开销**，而不是只有 prefill。

用本机实测数字重算（短 prompt，5 token）：

| 组成 | 实测 | 说明 |
|---|---|---|
| prefill | **2027 ms** | `DSV41_TIMING fwd ids=5 pos=0` |
| 第一个 decode 步 | **831 ms** | `DSV41_TIMING fwd ids=1 pos=5` |
| 小计 | **2858 ms** | 我们代码内的部分 |
| 服务层（HTTP/调度/采样/SSE 首块） | **≈ 940 ms** | TTFT 3797 − 2858 |

所以"2.3 s 黑箱"实际是 **~0.94 s**，且它包含了本应不可避免的首个 decode 步——真正可优化的
服务层开销更小。

### 对 P2 目标的意义

P2 要求"2K prompt TTFT ≤ 5 s"。按实测斜率（1.5K prompt 时 TTFT 15.2 s，扣掉 decode 与开销后
prefill ≈ 13.5 s ⇒ **约 9 ms/token**），2K 的 prefill 约 18 s，加 decode/开销约 1.7 s，
**总 TTFT 约 20 s**，距 5 s 差 4 倍。

要达成 5 s，需要 prefill 从 9 ms/token 降到约 **1.6 ms/token**（约 5.5 倍）。这条路径的成本主体是
MoE（每层 6 专家 × 40 层），所以 **P2 的达成依赖 P1 的 MoE 优化**，二者不是独立的。

## 8. 结论与建议

- **测量口径必须先定死**：TTFT = prefill + 首个 decode + 服务开销；报告中三者要分开给，
  否则会像我上一节那样把首个 decode 步误算成"HTTP 开销"。
- **优先做 prefill 的 MoE 优化**（对 P1 与 P2 同时有效），而不是去抠那 ~0.94 s 的服务层。
- 下一轮仍建议补完 chat vs completions 对照（用于确认模板成本），但它的优先级低于 MoE。
