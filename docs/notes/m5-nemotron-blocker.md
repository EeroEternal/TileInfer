# Nemotron 30B 真权重验证的当前阻塞

## 事实

目标目录：`/home/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`

`model.safetensors.index.json` 声明 14 个分片；磁盘上只有 5 个：

```
model-00001-of-00014.safetensors
model-00005-of-00014.safetensors
model-00006-of-00014.safetensors
model-00013-of-00014.safetensors
model-00014-of-00014.safetensors
```

缺失：

```
model-00002-of-00014.safetensors
model-00003-of-00014.safetensors
model-00004-of-00014.safetensors
model-00007-of-00014.safetensors
model-00008-of-00014.safetensors
model-00009-of-00014.safetensors
model-00010-of-00014.safetensors
model-00011-of-00014.safetensors
model-00012-of-00014.safetensors
```

`tools/nemotron/preflight_model.py` 因而是 `ready_to_load=false`、`ready_for_dummy=true`。
`--load-format dummy` 能启动，但权重是随机的，不能用于 C1 正确性、C2 参考一致性或 MTP 接受率。

## 已验证（不依赖真权重）

- 静态 gap/plan：`state.linear` 是唯一 vendor gap；候选 `tileinfer.mamba2_state` 已实现并出现在
  gap 的 candidates 中，但未注册为默认，plan 仍 `servable=false`。
- CPU：Mamba2 目标几何（64 头 × 64 head_dim × 128 state × 8 组，37/53 varlen）与 scalar recurrence
  一致；非门控 relu2 MoE 与逐 token 循环一致。
- NPU：grouped-matmul 非门控 MoE 与 CPU 参考一致（fp32 max abs 5.96e-8，bf16 max rel 8.9e-3）。
- 目标 dummy：eager 和 graph 两种服务都启动；graph 捕获 PIECEWISE+FULL、2.68 GiB、日志出现
  `Replaying aclgraph`；MTP drafter 执行（7 drafts/0 accepted，随机权重）；8K 上下文 6001 token；
  4 路并发 4/4 完成。全部是机械证据。

## 真权重到位后要做的

1. `preflight_model.py` 必须 `ready_to_load=true`。
2. C1：启动真实 14-shard 服务，跑 1-token 和 161-token prompt，记录 token、日志、plan id。
3. C2：同一 prompt 与 HF/transformers 参考实现逐 token/容差对齐；只有对齐后才能把候选
   `tileinfer.mamba2_state` 注册为默认并改 plan 的 `servable`。
4. C3：真实权重下确认 graph replay 与稳态 `decode_fallbacks=0`；当前 bridge 的 `decode_calls`
   计数在图 replay 路径上没有推进，需要先修计数器或改用 replay 计数。
5. MTP：真实权重接受率；长上下文/并发/冷启动/绝对地板。

## 不做什么

不下载缺失分片、不 kill 他人进程、不改上游源码；需要用户明确提供分片或授权下载后才能继续。
