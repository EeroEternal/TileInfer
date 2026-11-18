# Nemotron 30B 真权重验证的当前阻塞

## 事实

目标目录：`/home/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`

2026-11-15 复核：`model.safetensors.index.json` 声明的 14 个分片**全部到位**，没有缺失、没有空文件；
`tools/nemotron/preflight_model.py` 返回 `ready_to_load=true`。证据：
[`m5-nemotron-real-weights.json`](m5-nemotron-real-weights.json)。

## 已验证（真权重，CPU/算子级）

- **全部 6513 个张量的形状与 config 逐项核对：0 unknown、0 mismatch**；14 个分片都被索引使用；
  52 层类型为 23 mamba / 23 moe / 6 attention。这能排除“分片齐但 layout 不对”的常见问题。
- MoE layer 1 的真实 128 个 expert 权重 `up_proj [128,1856,2688]` / `down_proj [128,2688,1856]`
  已加载；TileInfer 非门控 relu2 `fused_experts` 与逐 expert 直接参考**完全一致**（max abs 0.0）。
  注意 checkpoint 存的是分离的 `up_proj`/`down_proj`，不是 fused `w13`；调用方必须 stack 成
  `[E,I,H]` / `[E,H,I]`，不能按门控 SwiGLU 解释。
- Mamba2 layer 0 的真实 `A_log`/`D`/`dt_bias`（A 范围 -1752.41..-5.36e-4）已在目标几何
  64 头 × 64 head_dim × 128 state × 8 组、varlen 37/53 下与 scalar recurrence 对齐：
  chunk 输出 max abs 9.16e-4（相对 4.0e-7），final state max abs 1.14e-4。
- 检查中发现 `initial_states` 在已是 fp32 时会被原地写；已改为 clone 输入，并加回归断言
  （`tests/test_mamba2_ops.py`）。

## 历史阻塞：宿主重启后 NPU 设备不可见（已恢复）

2026-11-15 15:48 曾出现 `/dev/davinci*` 不存在、`npu-smi` 报 `dcmi module initialize failed`。
2026-11-16 之后设备已恢复，C1/C3/TP1 MTP 已在真权重上跑过；本节只作为历史记录。

## 当前阻塞：HCCL rootinfo 与多卡 AllReduce

2026-11-17 复核时 `/etc/hccl_rootinfo.json` 是一个**空目录**（另一个
`hccl_rootinfo.json.dirbak.20261116` 也是空目录）。直接用空目录启动 TP=8 报：

```text
hcclCommInitRootInfoConfig(...), error code is 4
Failed to parse the ranktable file /etc/hccl_rootinfo.json.
Reason: The JSON format is incorrect.
```

用 CANN 的 `libtopoaddrinfo.so::PodGetRootinfo` 读出真实拓扑后确认：

- 本机 8 张卡的 `local_id` 是 **56–63**，不是 0–7；
- 设备 0–3 的 level 0 各有 4 个 EID，设备 4–7 的 `rank_addr_list` 为空；
- 把 8 个条目合并后，HCCL 报 `RankTableInfo::CheckAndInsert ... rankAddrSize[0]`；
- 清空 `level_list`、只保留真实 `local_id` 和 `topo_file_path` 后，HCCL 能建链，但
  warmup 的 AllReduce 报 AICPU `AllReduceSoleMeshTwoShot` → `ACL stream synchronize
  failed:507018`；
- 用 vnic IPv4 填 `rank_addr_list`，HCCL init 返回 error code 1；
- 用真实 EID 生成 **TP=4** rootinfo，也在 `HcclAllreduce` 失败；`HCCL_ALGO=Ring` /
  `HCCL_ENABLE_OPEN_AICPU=0` 没有改变失败点。

共享环境里的 Qwen TP=2 也遇到同一 `RootInfoDetect failed`，见
[`multicard-tp2-verification-20261117.md`](multicard-tp2-verification-20261117.md)。
因此这是宿主/HCCL 根信息与拓扑问题，不是 Nemotron provider 的算子路径；所有临时
rootinfo 已删除，`/etc/hccl_rootinfo.json` 已恢复为空目录。

## 恢复后要做的

1. 环境侧提供一份与本机 8 卡拓扑匹配的 `/etc/hccl_rootinfo.json`，或修复生成/挂载方式；
   `npu-smi info` 能看到 8 卡且 HBM 空闲。
2. 先跑 TP=1 回归（C1/C2/C3/MTP），确认 rootinfo 修复没有改变单卡行为。
3. 再跑 TP=8 baseline 与 MTP：512 token、4 路并发、32K/130K prompt、图模式；
   记录 `decode_calls`/`decode_graph_hits`/`decode_fallbacks` 和逐 token 等价结果。
4. TP=8 通过后再决定是否更新 `servable`；在此之前保持 `servable=false`、MTP 默认关闭，
   不对外声明支持。

C2 参考与逐 step 统计已落地在
[`m5-delivery-gates-20261117.md`](m5-delivery-gates-20261117.md)；C2 教师强制 top-1/top-5 rank
已通过；本 blocker 只保留 TP=8 共享环境问题，TP=8 为 out-of-scope，不阻塞单卡交付。

## 不做什么

不下载/不删除权重；不 kill 他人进程；不改上游源码；没有明确同意不启动服务/占端口/占卡；没有证据
不写“支持”。
