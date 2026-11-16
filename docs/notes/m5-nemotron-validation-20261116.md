# Nemotron 3.5 Lightning 昇腾实机验证（2026-11-16）

目标：在 `192.168.235.16` 的 8×Ascend950PR 上，用 vllm-ascend 0.23.0 拉起
`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`，并留下可复核的 C1/C2/C3、长上下文与 MTP 证据。
本文件记录**实际结果**，不是支持声明。

## 0. 代码与栈

- 远端 `/home/zxs/TileInfer`，提交 `ac567ca`（`fix(nemotron): keep causal-conv update graph-capture safe`）。
  之前同一轮提交：`40e46b3`（speculative state update）、`bb4f06b`（speculative causal-conv update）。
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
`Say hello in one sentence.` 的 HF 参考首 token 本身是 ` I` / ` The` 的 bf16 并列（logit 都是
12.25），greedy 任选一个就会分叉；该 prompt 不能作为 C2 判据。

## 3. C3：图捕获 / 回退

不带 `--enforce-eager` 启动，NPU 2，`capture` 成功：

```json
{"capture_calls":1,"captured_graphs":2,"capture_keys":{"PIECEWISE":1,"FULL":1},
 "cudagraph_mode":"FULL_AND_PIECEWISE","capture_done":true,
 "decode_calls":0,"decode_graph_hits":0,"decode_fallbacks":0,
 "none_after_capture":0,"attention_backend":"AscendAttentionBackend","errors":0}
```

日志有 `Replaying aclgraph`，但 `decode_calls=0`：dispatcher 计数没有记录到真实请求的 decode，
因此**只能证明捕获成功，不能证明稳态回退为 0**。这是观测缺口，不要把它写成完整 C3 通过。
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

## 5. MTP：失败，不要开启

真权重 MTP 服务可以起，`NemotronHMTPModel`、drafter 加载、`/metrics` 均有记录：

- `spec_decode_num_drafts_total=14`，`spec_decode_num_accepted_tokens_total=8`；
- 同样 prompt、temperature 0、max_tokens 16，MTP 输出
  ` Paris.</think>Paris is the capital of France.`
  非 MTP 输出 ` Paris.  \nThe capital of Germany is Berlin.  \nThe capital of`；
- 两次 MTP 结果确定，但不是非 MTP 的 greedy 结果。speculative decoding 应当无损；
  当前 MTP 路径不满足这一条。

已补的 `selective_state_update` / `causal_conv1d_update` 多 token 重放没有修复该差异，
因此**不是我们的 state/conv 补丁单独能解释的**；在查清 MTP 验证链之前，MTP 质量门判失败，
不能用 MTP 输出作为模型正确性证据，也不要默认开启。

## 6. TP=8 与共享环境

TP=8 启动在 HCCL 初始化被环境挡住：

```text
Failed to parse the ranktable file /etc/hccl_rootinfo.json.
Reason: The JSON format is incorrect.
```

`/etc/hccl_rootinfo.json` 在宿主上是**空目录**，不是 JSON 文件。没有改系统路径、没有删目录；
本轮只做 TP=1。卡 0/1 有他人的 `VLLM::EngineCore` / `VLLMWorker_TP`，全程只使用空闲的 NPU 2，
只停止自己 PID 启动的进程。

## 7. 当前结论

| 项 | 状态 |
|---|---|
| C1 真权重可服务 | 通过（bring-up，`servable=false`） |
| C2 text/token | 通过（单 prompt） |
| C2 logprob top-5 | 未通过（最大差 1.636） |
| C3 捕获 | 通过（2 graphs） |
| C3 稳态零回退 | 未证明（decode 计数为 0） |
| 长上下文 8K–261K | 可完成（长度证据） |
| MTP 无损性 | 失败 |
| TP=8 | 环境阻塞（HCCL rootinfo 是空目录） |

因此：**不能对外说“vllm-ascend 已支持 Nemotron 3.5 Lightning”。** 可以说的是：
在单张 Ascend950PR 上，通过 v3 provider 补齐 Mamba2 conv/SSD/state 与非门控 relu2 MoE 后，
真权重短文本和 261K 长度可以跑通；C2 logprob、C3 稳态计数、MTP 仍未闭合。
