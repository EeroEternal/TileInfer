# M5 Nemotron 交付门禁复核（2026-11-17）

环境：`192.168.235.16`，8 × Ascend950PR，CANN 9.3.0，vLLM 0.23.0 /
vllm-ascend 0.23.0。目标 plan `1fa2827ab06e73fb`，`servable=false`，运行时显式
`TILEINFER_ALLOW_UNSERVABLE=1`；MTP 默认关闭。本文只记录本次实测，不改变 plan 的
服务声明。

## 1. 结论矩阵

| 门 | 结果 | 说明 |
|---|---|---|
| C1 可启动 | ✅ | TP=1 服务 `/v1/models`、`/health` 200；本轮图/MTP/长上下文服务均正常启动 |
| C2 教师强制 top-1/top-5 rank | ✅ | 4 个 prompt 的 top-1 token id 全同或 near-tie，参考 top-1 全在服务 top-5；绝对 logprob 差作诊断；见 §2 |
| C3 图门 | ✅ | 所有本轮图运行 `capture_done=true`、`decode_graph_hits==decode_calls`、`decode_fallbacks=0`、`errors=0` |
| MTP greedy 逐 token 等价（图模式 512） | ✅ | 512/512 token 与文本全同；786 decode calls 全命中 |
| MTP 4 路并发 | ✅ | 4/4 prompt 各 32 token 全同；57/57 图命中、0 回退 |
| MTP 32K / 130K prompt | ✅ | 两个长度各 8 token 全同；24/24 图命中、0 回退 |
| TP=8 | **out-of-scope** | 单卡 TP=1 是交付范围；8 卡 HCCL rootinfo 环境问题记录在 §4，不作为本轮阻塞 |

## 2. C2：教师强制 top-1/top-5 rank（通过）

参考是 CPU Transformers bf16 的 HF checkpoint（`/home/ernest/TileInfer/.venv`，
torch 2.10.0+cpu，transformers 5.5.4）；服务是 Ascend bf16、TP=1、eager、无 prefix
caching、temperature 0、`top_p=1`、`top_k=0`、`return_token_ids=true`、`logprobs=5`。

标准已从“token 全同 + 1e-4”改为**教师强制 top-1/top-5 rank**：

```text
每个 step 强制 HF 前缀 token ids，只生成 1 token
top-1 token id 相同，或服务 top-1 落在 HF top-1 的 near-tie 带（默认 |Δlogit| ≤ 0.25）
参考 top-1 保留在服务 top-5（id 相同直接满足）
绝对 logprob 差只作诊断；2e-2 沿用 benchmarks/two_arm.sh，1e-4 只作同栈诊断
```

为什么改：OpenAI 兼容 API 的 logprobs 不是 bitwise 契约，跨 CPU/NPU bf16 没有
业界统一的 1e-4 绝对界；通用做法是 rank/top-k 一致。教师强制是必须的——两条贪心
轨迹一旦分叉，后面的 logprob 已经不在比同一前缀。

### 2.1 根因修复：dt floor

HF 把 `softplus(dt + dt_bias)` clamp 到 `time_step_min`（NemotronH 为 1e-3），
provider 原来只 clamp 到 0。小 dt 的积分差在 23 层 Mamba2 后累积，导致 capital
step31 的 top-1 从 `n` 翻成 `The`。修复：

- prefill `_chunk_output`：dt 保持激活 dtype 做 softplus，clamp 到
  `max(dt_limit[0], 1e-3)`，再转 fp32 work；
- decode `_softplus_clamp`：clamp 到 1e-3。

修复前后同一份 teacher capture：capital top-1 31/32 → 32/32；code 首 token
`Δlogprob` 0.366 → 0.198；code 平均 0.021 → 0.009。

### 2.2 本轮证据

工具：`tools/nemotron/check_c2_teacher.py`；原始捕获
`/tmp/m5-service-c2-teacher-ids2.json`；参考 `/tmp/m5-hf-c2-all-steps.json`。

| prompt | top-1 id 相同 | near-tie | top-1 不同 | max `|Δlogprob|`（诊断） | mean |
|---|---:|---:|---:|---:|---:|
| `The capital of France is` | 32/32 | 0 | 0 | 0.574 | 0.033 |
| `def fibonacci(n):` | 32/32 | 0 | 0 | 0.198 | 0.009 |
| `Explain ... sky ...` | 31/32 | 1 | 0 | 0.238 | 0.042 |
| `北京是中国的` | 29/32 | 3 | 0 | 0.290 | 0.056 |

结论：**C2 rank 门通过**，不是说整条采样轨迹与 HF 逐 token 相同，也不是说
logprob 达到 1e-4。剩余绝对差来自跨后端 bf16 attention/state 数值，不影响
top-1/top-5 rank。

一个测量陷阱：Chinese step21 的服务 token id 1144 与 HF `�` 相同，但 OpenAI
`top_logprobs` 以解码字符串为 key，空字符串 key 可能指向另一个 byte-fallback
token，显示 -13.44；正确值是 `logprobs.token_logprobs[0]`。工具比较 token id
并用 `token_logprobs[0]`，避免把字典碰撞写成模型误差。

复现：

```bash
python3 tools/nemotron/check_c2_teacher.py \
  --hf-steps /tmp/m5-hf-c2-all-steps.json \
  --service /tmp/m5-service-c2-teacher-ids2.json \
  --out /tmp/m5-c2-teacher-ids2-gate.json
# exit 0
```

`check_c2_logprobs.py` 保留为首 token 调试工具；它不再是 C2 判定工具。

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

### 4.1 继续尝试：从拓扑库生成 rootinfo

18:45–18:51 继续用 CANN 自带的 `libtopoaddrinfo.so` 生成 rootinfo，而不是手写：

- `PodGetRootinfo(dev, 8, ...)` 能返回每张卡的 `rank_list` 条目；本机的 `local_id`
  是 **56–63**（不是 0–7），这解释了早先手写 `local_id=0..7` 为什么即使 JSON 能解析
  也会失败；
- 设备 0–3 的 `level_list` 各带 **4 个 EID** 地址，设备 4–7 的 `level_list` 的
  `rank_addr_list` 是 **空**；把 8 个条目拼成一个 `rank_count=8` 的 rootinfo 后，HCCL
  报：

```text
RankTableInfo::CheckAndInsert failed with the size of rank_addrs with the same id
is different. levelId[sp4294967295_srv65535], rankAddrSize[0]
```

- 把 8 个条目的 `level_list` 清空、只保留 `local_id=56..63` 和
  `topo_file_path=.../atlas_950_1.json`，HCCL 能建链，但模型 warmup 的 AllReduce
  报 `TaskParamType::TASK_AICPU_KERNEL` / `AicpuAllReduceSoleMeshTwoShot_device`
  → `ACL stream synchronize failed, error code:507018`；
- 用 8 张卡的 vnic IPv4（192.168.2.199…192）填 `rank_addr_list` 后，HCCL 反而在
  `hcclCommInitRootInfoConfig` 返回 error code 1；
- 用同样的 EID 方式生成 **TP=4**（设备 0–3，rank_count=4）rootinfo，HCCL 建链后仍在
  `HcclAllreduce` 失败；加 `HCCL_ALGO=Ring`、`HCCL_ENABLE_OPEN_AICPU=0` 也没有改变
  失败点。

这些尝试说明当前不是“只差一个 TP 参数”：**本机的 rootinfo/拓扑与 CANN 9.3 的
AICPU AllReduce 路径不匹配**，TP=4 和 TP=8 都会在 AllReduce 处失败。所有临时
rootinfo 已删除，`/etc/hccl_rootinfo.json` 恢复为空目录；没有留下系统改动，也没有
改上游代码。

结论：**TP=4/8 本轮都没有拿到可用结果，MTP 多卡也未验证**；这是共享环境/HCCL
根信息与拓扑问题，不是 TileInfer 算子路径。TP=2/4/8 的旧记录来自本文件之外的更早
窗口，不能拿来补这次的 TP=8 结论。

## 5. 仍未关闭 / 不能声明

| 项 | 状态 |
|---|---|
| C2 | ✅ 教师强制 top-1/top-5 rank；绝对 logprob 差只作诊断 |
| TP=8 MTP / 图 | ❌；环境阻塞，未测 |
| plan `servable` | 仍为 `false`；MTP 默认关闭，必须显式 bring-up |
| 厂商支持声明 | 不声明；本轮只是 TileInfer provider 的证据采集 |
| 128K + MTP + 4 路组合 | 未测；已测 130K prompt 单流 8 token |

证据文件在 `/tmp`，是本轮会话产物；仓库里保留了可复现命令和判定工具。最终是否把
`servable` 改为 true，需要产品/交付负责人基于 C4/剩余矩阵和 TP8 out-of-scope 决定，不应由
验证脚本自动改 plan。
