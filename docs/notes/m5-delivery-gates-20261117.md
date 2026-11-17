# M5 Nemotron 交付门禁复核（2026-11-17）

环境：`192.168.235.16`，8 × Ascend950PR，CANN 9.3.0，vLLM 0.23.0 /
vllm-ascend 0.23.0。目标 plan `1fa2827ab06e73fb`，`servable=false`，运行时显式
`TILEINFER_ALLOW_UNSERVABLE=1`；MTP 默认关闭。本文只记录本次实测，不改变 plan 的
服务声明。

## 1. 结论矩阵

| 门 | 结果 | 说明 |
|---|---|---|
| C1 可启动 | ✅ | TP=1 服务 `/v1/models`、`/health` 200；本轮图/MTP/长上下文服务均正常启动 |
| C2 logprob 严格门（token 全同 + 1e-4） | ❌ | 4 个 prompt 首 token top-1 全同，但 top-5 / logprob 不是 1e-4 级；见 §2 |
| C2 参考容差门（tie/tolerance，声明口径） | ✅ | top-1 等价、top-5 ≥ 4/5、首 token `|Δlogprob| ≤ 0.5`；工具 `check_c2_logprobs.py` |
| C3 图门 | ✅ | 所有本轮图运行 `capture_done=true`、`decode_graph_hits==decode_calls`、`decode_fallbacks=0`、`errors=0` |
| MTP greedy 逐 token 等价（图模式 512） | ✅ | 512/512 token 与文本全同；786 decode calls 全命中 |
| MTP 4 路并发 | ✅ | 4/4 prompt 各 32 token 全同；57/57 图命中、0 回退 |
| MTP 32K / 130K prompt | ✅ | 两个长度各 8 token 全同；24/24 图命中、0 回退 |
| TP=8 | ❌（环境阻塞） | 8 卡空闲也卡在 HCCL rootinfo / ACL stream；不是 TileInfer 代码路径；见 §4 |

## 2. C2：严格不过，参考容差过

参考是 CPU Transformers bf16 的 HF checkpoint（`/home/ernest/TileInfer/.venv`，
torch 2.10.0+cpu，transformers 5.5.4）；服务是 Ascend bf16、TP=1、eager、无 prefix
caching、temperature 0、`logprobs=5`、`max_tokens=32`。请求同一字符串、同一 seed。

| prompt | HF 首 token | 服务首 token | `|Δlogprob|` | top-5 交集 |
|---|---|---:|---:|---:|
| `The capital of France is` | ` Paris` (-0.1405) | ` Paris` (-0.1947) | 0.0542 | 4/5 |
| `def fibonacci(n):` | `\` (-1.3441) | `\` (-1.7103) | 0.3662 | 5/5 |
| `Explain in one sentence why the sky is blue:` | ` The` (-2.0784) | ` The` (-1.8330) | 0.2453 | 4/5 |
| `北京是中国的` | `首` (-1.6734) | `首` (-1.7303) | 0.0568 | 4/5 |

**严格门**（首 token 与 token 序列全同、logprob 绝对误差 ≤ 1e-4）失败，最大首 token
logprob 误差 **0.3662**；top-5 最少 4/5。原因符合 bf16 CPU/GPU 数值差，但不能据此放宽
成“严格通过”。

后续 token 轨迹也不能宣称与 HF 全同：

- `def fibonacci(n):` 32/32 token 相同；
- `北京是中国的` 第 3 个生成 token：HF 的 `北京` 与 `也是` **logit 同为 16.125**，是真正的
  top-1 tie；HF argmax 取 `北京`，服务取 `也是`。两者都是等价 top-1，属于 tie-breaking，
  不是模型错；
- `Explain ... sky ...` 第 18 个 token 有 HF tie，服务取 tie 的另一个 token；
- `The capital of France is` 第 31 个 token：HF `n`（logit 19.625）对服务 `The`，**不是 tie**，
  是后段累计 bf16 漂移。因此 C2 结果只按“首 token 分布 + 显式容差”声明，不声明 32-token
  与 HF 逐 token 相同。

参考容差门的明确定义（工具默认值）：

```text
top-1 相同，或服务 top-1 是 HF top-1 的 logit tie（|Δlogit| ≤ 0.125）
top-5 交集 ≥ 4
首 token |Δlogprob| ≤ 0.5
```

复现：

```bash
python tools/nemotron/check_c2_logprobs.py \
  --hf /tmp/m5-hf-c2-multi.json \
  --service /tmp/m5-service-c2-logprobs.json \
  --hf-steps /tmp/m5-hf-c2-all-steps.json
# exit 0；--require-sequence 或 --logprob-tol 1e-4 会 exit 1
```

`--require-sequence` 用于需要逐 token 的场景；C2 默认不要求整条采样轨迹（MTP 门才是
逐 token 门）。

## 3. MTP + C3 图证据

| 场景 | token 等价 | 墙钟（同窗口） | C3 metrics |
|---|---|---|---|
| 512 token，baseline 图 | — | 87.0 s | `511/511/0`，2 图（PIECEWISE 1 / FULL 1） |
| 512 token，MTP 图 | 512/512 全同 | 48.46 s | `786/786/0`，2 图（PIECEWISE 1 / FULL 0） |
| 4 路 × 32 token，baseline 图 | — | 6.09 s | `32/32/0`，6 图 |
| 4 路 × 32 token，MTP 图 | 4/4 全同 | 4.34 s | `57/57/0`，6 图（PIECEWISE 3 / FULL 0） |
| 32K prompt + 8 token，baseline | — | 6.89 s | `14/14/0`，2 图 |
| 32K prompt + 8 token，MTP | 8/8 全同 | 5.76 s | `24/24/0`，2 图（PIECEWISE 1 / FULL 0） |
| 130K prompt + 8 token，baseline | — | 18.52 s | `14/14/0`，2 图 |
| 130K prompt + 8 token，MTP | 8/8 全同 | 19.32 s | `24/24/0`，2 图（PIECEWISE 1 / FULL 0） |

“全同”是比较服务返回的 `token_ids` 与文本；证据文件：

- `/tmp/m5-graph-base-512.json`、`/tmp/m5-graph-mtp-512.json`
- `/tmp/m5-4way-base.json`、`/tmp/m5-4way-mtp.json`
- `/tmp/m5-long-base.json`、`/tmp/m5-long-mtp-32000.json`、`/tmp/m5-long-mtp-130000.json`
- 对应 `*-metrics.json`

所有 metrics 均通过：

```bash
python tools/nemotron/check_c3_metrics.py /tmp/m5-*-metrics.json
```

MTP 图是 `PIECEWISE`、`FULL=0`，这是 MTP 的预期形态；baseline 是
`FULL_AND_PIECEWISE`。`capture_calls=1`、`capture_done=true`、`decode_calls>0`、
`decode_graph_hits==decode_calls`、`decode_fallbacks=0`、`errors=0` 全部满足。

### 3.1 这次修的捕获路径

MTP 图启动时曾在 ACL graph capture 内做 host D2H 读取：

```text
rtMemcpy execution failed, reason=operation not permitted when a stream is capturing
```

根因是 `tileinfer/providers/nemotron/mamba2.py` 的推测解码 conv/SSM 路径调用了
`_as_int_list(query_start_loc)` 等 host 读。修复不是放宽图门，而是新增仅用 shape/device
tensor 的捕获路径：

- `_causal_conv1d_update_capture`：只按每序列 uniform token 数构造窗口，`F.conv1d`，
  按 token scatter 写回 conv cache line；保留 null block 的原值；
- `_selective_state_update_capture`：用 `accepted-1`、全部 scheduled token 和逐 token
  的 destination 做批量 SSM 更新，算术与 batch-one 路径一致；
- `_write_conv_lines` / `_write_state_slots`：tensor-only 写回；
- `_is_stream_capturing()` 守卫只在捕获时走上述路径；eager 路径不变。

单测：

```bash
.venv_native213/bin/python -m pytest -q \
  tests/test_mamba2_ops.py tests/test_v3_impl_table.py tests/test_c2_gate.py
# 26 passed
```

## 4. TP=8：未通过，环境阻塞

8 卡全部空闲时重试 TP=8（baseline，eager，`max-model-len=2048`）：

```text
hcclCommInitRootInfoConfig(numRanks, &rootInfo, rank, config, &(comm->hcclComm_)),
error code is 4
Failed to parse the ranktable file /etc/hccl_rootinfo.json.
Reason: The JSON format is incorrect.
```

现场事实：

- `/etc/hccl_rootinfo.json` 是**空目录**（不是文件），mtime 2026-11-16 19:20；
  系统里另有一个 `hccl_rootinfo.json.dirbak.20261116` 空目录；
- 把空目录临时移走、让 HCCL 自己找根信息，TP=2 能过 HCCL 建链，但随后
  `HcclAllreduce` / `RunAicpuCommInit` 失败；TP=8 仍在 ACL stream 同步失败；
- 用 `libtopoaddrinfo.so` 的格式（`version` / `topo_file_path` / `rank_count` /
  `rank_list`）手工生成 rank_count=8 的 `/etc/hccl_rootinfo.json` 后，HCCL 不再报
  JSON parse，但模型 warmup 时 `ACL stream synchronize failed, error code:507018`
  （`_warmup_ssd_kernels` 的 `torch.tensor(..., device=...)`），TP=2/TP=8 都一样；
- 这些尝试后已把 `/etc/hccl_rootinfo.json` 恢复为空目录，未留系统改动；
- 共享机器上还观察到别的用户的 Qwen TP=2 占用 NPU 2/3；但在**全 8 卡空闲**时重试
  仍复现同一 rootinfo/ACL 失败，所以不能把 TP=8 归因于“卡被占”。

另一个操作坑：CANN 9.3 的 `HCCL_CONNECT_TIMEOUT` 合法范围是 **[120, 7200]**；
早先按 9.1 的习惯设 60 会得到
`Value 60 ... is out of range[120, 7200]`，这不是模型代码问题。后续重试用 600。

结论：**TP=8 本轮没有拿到可用结果，MTP TP=8 也未验证**；这是共享环境/HCCL 根信息
问题，不是 TileInfer 算子路径。TP=2/4/8 的旧记录来自本文件之外的更早窗口，不能拿来
补这次的 TP=8 结论。

## 5. 仍未关闭 / 不能声明

| 项 | 状态 |
|---|---|
| C2 严格 1e-4 | ❌；只有 C2-ref（tie + 0.5 logprob 容差）通过 |
| TP=8 MTP / 图 | ❌；环境阻塞，未测 |
| plan `servable` | 仍为 `false`；MTP 默认关闭，必须显式 bring-up |
| 厂商支持声明 | 不声明；本轮只是 TileInfer provider 的证据采集 |
| 128K + MTP + 4 路组合 | 未测；已测 130K prompt 单流 8 token |

证据文件在 `/tmp`，是本轮会话产物；仓库里保留了可复现命令和判定工具。最终是否把
`servable` 改为 true，需要产品/交付负责人基于上述 C2 容差和 TP8 缺口决定，不应由
验证脚本自动改 plan。
