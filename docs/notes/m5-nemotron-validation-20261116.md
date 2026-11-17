# Nemotron 3.5 Lightning 昇腾实机验证（2026-11-16）

目标：在 `192.168.235.16` 的 8×Ascend950PR 上，用 vllm-ascend 0.23.0 拉起
`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`，并留下可复核的 C1/C2/C3、长上下文与 MTP 证据。
本文件记录**实际结果**，不是支持声明。

## 0. 代码与栈

- 远端 `/home/zxs/TileInfer`，MTP 修复提交 `a368a04`
  （`fix(nemotron): make MTP target verification lossless`，见 §5）。
  之前的 conv/state 提交：`ac567ca`（`fix(nemotron): keep causal-conv update graph-capture safe`）、
  `40e46b3`（speculative state update）、`bb4f06b`（speculative causal-conv update）。
- vLLM 0.23.0 / vllm-ascend 0.23.0 / CANN 9.3.0 / Python 3.12.14 / torch_npu 2.10.0.post4。
- 模型权重 `/home/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`（62 GB，14 shards）。
- 目标 plan 仍为 `servable=false`；以下全部是 `TILEINFER_ALLOW_UNSERVABLE=1` 的 bring-up，
  不是“vllm-ascend 已支持该模型”。

## 1. C1：真权重服务

单卡 TP=1，NPU 2，`--max-model-len 8192 --max-num-seqs 1 --gpu-memory-utilization 0.90`，
bridge log 确认：

```text
selections=7 (tileinfer.mamba2_state=1, tileinfer.nemotron_moe=1,
vllm_ascend.attention=1, vllm_ascend.layernorm=1, vllm_ascend.quantization=1,
vllm_ascend.rotary=1, vllm_ascend.sample=1)
access_points=5 fallback=forbid | seams=config_gate,graph_hook,attn_select,impl_table
TILEINFER_ALLOW_UNSERVABLE=1
```

健康检查 200；短 prompt 完成并返回非零 token。代表性输出（`/v1/completions`，greedy）：

| prompt | completion |
|---|---|
| `The capital of France is` | ` Paris.  \nThe capital of Germany is Berlin.  \nThe capital of` |
| `Say hello in one sentence. I` | ` am a language model. I can help with various tasks, from answering questions to` |
| `def add(a, b):\n    return` | ` a + b\n\ndef subtract(a, b):\n    return a - b\n\n` |

显存：权重 58.833 GB；启动日志 `Free memory 79.29/79.81 GiB`；HBM 实测约 80.68 GB
（`npu-smi`，服务 + 图 + KV）。KV cache 9.95 GiB，`573,440` tokens（8192 配置）。

首个厂商缺口：Ascend 的 `causal_conv1d_fn` 在这个几何上返回全零，导致后续 SSD/state 全零；
v3 provider `tileinfer/providers/nemotron/mamba2.py` 补齐 prefill conv 后，短 prompt 才恢复。
这不是“厂商不支持所以阻塞”，而是本任务要补的算子。

## 2. C2：HF 参考一致性

用本地 `tools/nemotron/reference_probe.py` 重新生成 CPU/bf16 HF 参考
（transformers 5.12.1，`/tmp/m5-hf-capital.json`），prompt 取首 token 不并列的
`The capital of France is`：

| 项 | 结果 |
|---|---|
| HF generated token ids | `[6993, 1046, 1256, 1010, 1784, 8961, 1307, 10787, 1395, 10874, 1046, 1256, 1010, 1784, 8961, 1307]` |
| service generated token ids | 完全相同 |
| HF text | ` Paris.  \nThe capital of Germany is Berlin.  \nThe capital of` |
| service text | 完全相同 |
| 首 token top-5 集合 | 4/5 相同（ref 有 ` _`，service 有 ` London`） |
| 最大 logprob 差 | `1.636`（严格 atol `1e-4` 失败；atol `2.0` 通过） |

**结论：C2 的 token/text 级一致；logprob 级不一致，因此不能算完整 C2 通过。**
另跑 `TILEINFER_MOE_FP32=1` 排除“只是 MoE bf16 舍入”：token ids 仍完全一致，但首 token top-5
变成 `Paris / ** / Berlin / a / Brussels`，最大差约 1.18，仍不满足严格 logprob 判据；所以问题不在单一 MoE 精度开关。
再把 `TILEINFER_MOE_SORTED=0 TILEINFER_MOE_BMM=0` 打开（纯 per-expert loop），top-5 与 grouped+fp32
完全相同，说明也不是 grouped matmul 路径。
`Say hello in one sentence.` 的 HF 参考首 token 本身是 ` I` / ` The` 的 bf16 并列（logit 都是
12.25），greedy 任选一个就会分叉；该 prompt 不能作为 C2 判据。

## 3. C3：图捕获 / 回退

不带 `--enforce-eager` 启动，NPU 2，`capture` 成功；`max-num-seqs=1` 捕获 2 张，
`max-num-seqs=4` 捕获 6 张（PIECEWISE 3 + FULL 3）。

把 plan 的 `instrumentation.dump_every_decode_steps` 设为 1（只改落盘频率，不改实现选择），
32-token 请求后 metrics 直接给出真实 decode 计数：

```json
{"capture_calls":1,"captured_graphs":2,"capture_keys":{"PIECEWISE":1,"FULL":1},
 "cudagraph_mode":"FULL_AND_PIECEWISE","capture_done":true,
 "decode_calls":31,"decode_graph_hits":31,"decode_fallbacks":0,
 "none_after_capture":1,"attention_backend":"AscendAttentionBackend","errors":0}
```

`decode_calls=31` 全部命中图，`decode_fallbacks=0`；`none_after_capture=1` 是 prefill/非 uniform
诊断，不是被判定为回退。**8K 长 prefill 复测**：8001 prompt tokens / 8 output，3.01 s；
`decode_calls=7, decode_graph_hits=7, decode_fallbacks=0, none_after_capture=4`（4 个 prefill chunk
诊断，不是回退）。**128K 长 prefill**：130001 prompt tokens / 4 output，29.98 s；
`decode_calls=3, decode_graph_hits=3, decode_fallbacks=0, none_after_capture=64`。
**4 路并发（max-num-seqs=4）**：6 张图，4 个请求各 16 tokens，`decode_calls=15, hits=15,
fallbacks=0, none_after_capture=2`，四路输出相同。**结论：短稳态、8K/128K 长 prefill、4 路并发的
C3 都通过（capture + replay + 0 fallback）。**
默认 `dump_every_decode_steps=500` 时文件只在捕获时落盘（`decode_calls` 仍显示 0），
不能据此判 C3 失败；本次用 1 的频率取到了真实计数。长期压测和并发下的回退率仍未覆盖。

另外，加入 speculative `causal_conv1d_update` 的 host 循环后图捕获曾失败
（`Synchronizing a stream failed ... stream is captured`）；改为单 token 向量化快路径后捕获取回，
MTP 的 varlen 仍走逐 token 循环。

## 4. 长上下文

同一服务，`--max-model-len 262144`，单流：

| 档 | prompt_tokens | max_tokens | 结果 | 用时 |
|---|---:|---:|---|---:|
| 8K | 8192 | 8 | 完成 | 3.31 s |
| 16K | 16384 | 8 | 完成 | 4.80 s |
| 32K | 32000 | 16 | 完成 | 未记录 |
| 128K | 131072 | 4 | 完成 | 32.22 s |
| 261K | 261000 | 4 | 完成 | 68.84 s |

32K 用 32000 prompt 而不是 32768，是为了给生成 token 留出 `max_model_len` 空间；
各档 `usage.prompt_tokens` 与生成 prompt 完全一致。长上下文是**长度/可运行性**证据，
不是“256K 内容理解正确”的证据（seed 是 README 重复文本）。

### 4.1 并发（max-num-seqs=4，短输出）

单流/2 路/4 路都用 greedy、32 output tokens；每档 2 轮：

| 客户端 | 吞吐（tok/s，两轮） | TTFT（s） | 完成 token |
|---:|---|---|---|
| 2 | 11.20 / 11.45 | 0.23–0.36 | 32/32 |
| 4 | 21.60 / 22.57 | 0.26–0.48 | 32/32 |

这是“4 路都能完成且有并发收益”的证据，不是 SLA 或厂商基线结论。

**编译缓存注意**：默认 `VLLM_CACHE_ROOT` 在跑过 fp32 服务和一次失败编译后，`max-num-seqs=4`
启动会在 profile_run 报 `KeyError: 'AscendRMSNorm'`；换一个**全新的 `/tmp` cache** 后同一代码
正常捕获 6 张图并完成 4 路请求。不要删共享 cache，按进程设置 `VLLM_CACHE_ROOT=/tmp/...` 即可。
4 路 fresh-cache 复测每个请求 5.86–6.04 s、32/32 tokens 完成。

### 4.2 冷启动（全新 VLLM_CACHE_ROOT）

在 `/tmp` 新建空 cache 后启动（不删共享 cache）：

- 启动到 `/health=200`：**42 s**；
- 首个请求首 token：**0.33 s**；
- 16 output tokens 完成：2.896 s；
- 日志显示权重加载 58.8333 GB，捕获 1 张 PIECEWISE + 1 张 FULL。

### 4.3 算子审计

`TILEINFER_AUDIT` + `SIGUSR1` EngineCore：

- 4 路并发后：`classes=17, instances=390, vendor=229, torch=92, calls=513, suspicious=1`；
- 冷启动首请求后：`calls=214`，其余相同；
- 唯一 suspicious 是 `AscendUnquantizedFusedMoEMethod`（非门控 MoE 的 torch 替换路径，
  审计器看不到 `apply` 的 forward 计数，属于解释项，不是静默回退）；
- `ReLUSquaredActivation` 在并发轮有 437 次调用，`Mixer2RMSNormGated` 23 次。

## 5. MTP：无损性已修复并通过（2026-11-17）

真权重 MTP 服务可以起，`NemotronHMTPModel`、drafter 加载、`/metrics` 均有记录；
修复前同样 prompt、temperature 0 的 MTP 输出与非 MTP greedy 输出不一致
（` Paris.</think>Paris is the capital of France.` vs
` Paris.  \nThe capital of Germany is Berlin.  \nThe capital of`），因此当时 MTP 质量门判失败。
本节记录 2026-11-17 的根因定位与修复后证据。旧失败现场保留在 git 历史中，不再作为当前结论。

### 5.1 根因

两个缺口叠加，都在 **我们自己的 bridge/provider**，没有修改 vLLM/vllm-ascend 源码：

1. **Ascend V1 runner 只把 speculative metadata 转发给 GDN builder。**
   `NPUModelRunner` 用 `isinstance(builder, GDNAttentionMetadataBuilder)` 决定是否传入
   `num_accepted_tokens` / `num_decode_draft_tokens_cpu`；Mamba2 的
   `Mamba2AttentionMetadataBuilder` 不是该类型，1+draft 的 target verify 被当成 prefill，
   SSM/conv state 按“两个 draft 都接受”推进。bridge 把 Mamba2 builder 加入 runner 的 builder
   元组（`tileinfer/bridge/impl_table.py` 的 `ascend_spec_metadata`），不改上游源码。
2. **varlen selective update 的逐 token 公式与 single-token decode 公式不同形。**
   target verify 的 `dt` 是 `[tokens, 64, 64]`（heads×head_dim），`dt_bias` 是 `[64, 64]`；
   旧 varlen 路径在逐 token 循环里压缩/广播这两者，layer0 的 `ssu` 输出范数
   `2143.231934`，而同一 state/同一 token 的 single-token 路径是 `1996.194458`（约 7%），
   足以让 `.` 与 `.\n` 这种接近并列的 target logits 翻转。修复为**每个 verify token 都使用
   与 ordinary decode 相同的 batch-one 公式**：保留 head_dim 轴、加完整 `[heads, head_dim]`
   bias、用 `state[accepted-1]` 作为起点、逐 token 写回 `dst_state_batch_indices`。
   `tileinfer/providers/nemotron/mamba2.py` 的 varlen 路径即此实现。

关键定位证据（同一 prompt、同一 layer0）：

| 量 | 非 MTP single decode | 修复前 MTP varlen |
|---|---:|---:|
| conv 输出范数 | 56.158230 | 56.177052 |
| selective 输出范数 | 1996.194458 | 2143.231934 |
| state sum | 12820.72265625 | 12820.72265625 |
| `dtf0` / B norm / C norm | 1.2961817 / 6.713455 / 6.799973 | 相同 |

state、B/C、`dtf0` 都相同而 selective 输出差 7%，把问题收敛到 varlen 的逐 token 公式；
改成 batch-one 公式后 selective 范数恢复为 `1996.194458`，首个 token 即与非 MTP 一致。

### 5.2 验证（token ids 逐 token 比较，不只看文本）

服务：`--enforce-eager --mamba-cache-mode align --enable-prefix-caching`，
`TILEINFER_PLAN=/tmp/m5-nemotron-plan-20261117.json TILEINFER_ALLOW_UNSERVABLE=1`；
不再需要任何 `TILEINFER_DEBUG_*` / `TILEINFER_SPEC_*` 环境变量，bridge 默认安装 runner 补丁。

| 场景 | 结果 |
|---|---|
| 6 个 prompt（英文/代码/中文/数字续写），16–32 tokens | token ids 与文本全部逐 token 相同 |
| 同 prompt 128 tokens | token ids 全部相同；重复请求结果相同 |
| `all`（无 prefix caching）+ 6 prompt | token ids 全部相同 |
| `align` + prefix caching + 6 prompt | token ids 全部相同 |
| `max-num-seqs=2` 两路并发、24 tokens | 两路 token ids 均与非 MTP 相同 |

因此：**MTP 的 greedy 输出已经逐 token 复现非 MTP greedy 输出**。这不等于模型/厂商支持声明：
目标 plan 仍是 `servable=false`，默认不开启 MTP；需要显式 bring-up 开关和上述验证过的
配置才能用于证据采集。

## 6. TP=2/4/8 与共享环境

早先 TP=8 曾在 HCCL 初始化报：

```text
Failed to parse the ranktable file /etc/hccl_rootinfo.json.
Reason: The JSON format is incorrect.
```

当时 `/etc/hccl_rootinfo.json` 是**空目录**，所以没有改系统路径、没有删目录，先退回 TP=1。
后来 8 张卡全部空闲、使用全新的 `VLLM_CACHE_ROOT` 重试，**TP=2、TP=4、TP=8 都启动成功**：

- TP=8：`ASCEND_RT_VISIBLE_DEVICES=0..7`，max-model-len 2048，eager；
  每卡权重 7.39 GiB，KV 55.13 GiB；同 prompt 16 tokens 输出与 TP=1 完全一致
  ` Paris.  \nThe capital of Germany is Berlin.  \nThe capital of`；
- TP=2 同样输出完全一致；TP=4 也 health 通过。
- 只启动自己的 PID；停止后 8 张卡都回到 `No running processes`。

结论：TP=8 **不再是环境阻塞**；之前是共享环境/rootinfo 的瞬时状态，重试已通过。
TP=8 下尚未做 C3 图和 MTP。

## 7. 当前结论

| 项 | 状态 |
|---|---|
| C1 真权重可服务 | 通过（bring-up，`servable=false`） |
| C2 text/token | 通过（单 prompt） |
| C2 logprob top-5 | 未通过（最大差 1.636） |
| C3 捕获 | 通过（2 / 6 graphs） |
| C3 稳态零回退 | 通过（32-token 31/31/0；8K 7/7/0；128K 3/3/0；4 路并发 15/15/0） |
| 长上下文 8K–261K | 可完成（长度证据） |
| 并发 2/4 路 | 可完成，4 路约 22 tok/s（短输出） |
| 冷启动 | 42 s 到 health，首 token 0.33 s（全新 cache） |
| 算子审计 | 390 实例；1 个有解释的 suspicious；无静默回退证据 |
| MTP 无损性 | 通过（6 prompt / 32–128 tokens，token ids 逐 token 相同；align/all、2 路并发） |
| TP=2/4/8 | 通过（TP8 输出与 TP1 一致；未做 TP8 图/MTP） |

因此：**不能对外说“vllm-ascend 已支持 Nemotron 3.5 Lightning”。** 可以说的是：
在单张 Ascend950PR 上，通过 v3 provider 补齐 Mamba2 conv/SSD/state 与非门控 relu2 MoE 后，
真权重短文本、C3 短稳态、2/4 路并发、冷启动、TP=2/4/8 和 261K 长度可以跑通；
C2 logprob 仍未闭合；MTP 无损性已闭合，但 plan 仍为 `servable=false`，不对外声明支持。
