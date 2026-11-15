# Nemotron 30B 真权重验证的当前阻塞

## 事实

目标目录：`/home/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`

2026-11-15 复核：`model.safetensors.index.json` 声明的 14 个分片**全部到位**，没有缺失、没有空文件；
`tools/nemotron/preflight_model.py` 返回 `ready_to_load=true`。证据：
[`m5-nemotron-real-weights.json`](m5-nemotron-real-weights.json)。

## 已验证（真权重，CPU/算子级）

- MoE layer 1 的真实 128 个 expert 权重 `up_proj [128,1856,2688]` / `down_proj [128,2688,1856]`
  已加载；TileInfer 非门控 relu2 `fused_experts` 与逐 expert 直接参考**完全一致**（max abs 0.0）。
  注意 checkpoint 存的是分离的 `up_proj`/`down_proj`，不是 fused `w13`；调用方必须 stack 成
  `[E,I,H]` / `[E,H,I]`，不能按门控 SwiGLU 解释。
- Mamba2 layer 0 的真实 `A_log`/`D`/`dt_bias`（A 范围 -1752.41..-5.36e-4）已在目标几何
  64 头 × 64 head_dim × 128 state × 8 组、varlen 37/53 下与 scalar recurrence 对齐：
  chunk 输出 max abs 9.16e-4（相对 4.0e-7），final state max abs 1.14e-4。
- 检查中发现 `initial_states` 在已是 fp32 时会被原地写；已改为 clone 输入，并加回归断言
  （`tests/test_mamba2_ops.py`）。

## 新阻塞：宿主重启后 NPU 设备不可见

2026-11-15 15:48（Asia/Shanghai）检查：

- `uptime` 显示宿主在 2026-11-13 21:27 左右重启；重启前目标 dummy 服务还能跑。
- `/dev/davinci*` 设备节点全部不存在；`/dev/davinci_manager`、`/dev/hisi_hdc` 也不存在。
- `npu-smi info` 失败：`dcmi module initialize failed. ret is -8005`。
- PCI 上没有 Huawei 加速卡（只有 62:00.0 的 NVMe 控制器）；`/sys/class/davinci*` 不存在。
- 当前没有测试服务/端口/NPU 进程；没有 kill 任何人。

结论：**真权重已就绪，但本机当前没有可用的 NPU 设备/驱动，无法启动真权重 C1 服务、图捕获或
MTP/长上下文/并发/冷启动验证。** 这是宿主/驱动层问题，不是代码或权重问题。

## 恢复设备后要做的

1. 宿主恢复 Ascend 驱动/设备节点，`npu-smi info` 能看到卡，且 HBM 空闲。
2. 取得明确同意后，用指定端口/卡启动真权重 C1：1-token 与 161-token prompt，记录 token、日志、plan id。
3. C2：同一 prompt 与参考实现逐 token/容差对齐。当前 `transformers 4.57.6` 没有 `NemotronH`
   类，不能直接 `AutoModel`；需要可用参考（升级/带 remote code 的 checkpoint，或用 vLLM 模型代码在
   CPU/第二设备上做参考），只有对齐后才能把 `tileinfer.mamba2_state` 注册为默认并改 plan 的 `servable`。
4. C3：真实权重下确认 graph replay 与稳态 `decode_fallbacks=0`；bridge 的 `decode_calls` 计数在图
   replay 路径上没有推进，需要先修计数器或改用 replay 计数。
5. MTP、8K/16K/32K/128K、并发、冷启动、绝对性能地板。

## 不做什么

不下载/不删除权重；不 kill 他人进程；不改上游源码；没有明确同意不启动服务/占端口/占卡；没有证据
不写“支持”。
