# 原生 vllm-ascend 对比尝试记录

> 2026-11-08。目标：在独立环境跑 DeepSeek-V4.1-Flash 原生 vllm-ascend，与 TileInfer 臂同口径
> 对比 TTFT/TPOT/ITL。**当前没有任何原生性能数字**，下面是逐版本的实际错误。

## 环境隔离

- 原 `.venv`（TileInfer）不做任何改动；曾误装 `vllm-ascend`，已 `pip uninstall` 清理。
- `.venv_ascend`：复制 `.venv` + 改写 shebang + `vllm-ascend 0.23.0`。
- `.venv_ascend30`：复制 `.venv` + `vllm 0.30.0 + vllm-ascend 0.30.0 + transformers 5.14.1`
  + `torch-npu 2.10.0.post7`（后换回 A5 的 post4）+ `triton-ascend 3.6/3.2`。

## 逐版本结果

| 版本 | 结果 |
|---|---|
| `vllm-ascend 0.23.0` | 可 import，但只认 `deepseek_v4`；`deepseek_v41` 在 transformers/vLLM config 表里不存在 |
| `vllm-ascend 0.23.0.post1` | 源码仍只含 `deepseek_v4`，无 v41 |
| 0.23 + config 改成 `deepseek_v4` | vLLM 报 `text_config ... does not have num_attention_heads` |
| `vllm-ascend 0.30.0` | **有 `vllm/models/deepseek_v41`、`DeepseekV41Config`，能正确解析 v41 config** |
| 0.30 + `vllm 0.30 + transformers 5.14.1 + torch-npu post7` | import 通过；启动时报 **A5/A2 设备类型不匹配**：`Current device type: AscendDeviceType.A5 does not match the installed version's device type: AscendDeviceType.A2` |
| 0.30 + 换回 A5 的 `torch-npu 2.10.0.post4` + triton 3.2 | 仍报同一 A5/A2 错误；另有 `triton.language.target_info` 缺失 |

## 判断

0.30 已经支持 DSV4.1，缺的是 **A5 对应的 `torch-npu 2.10.0.post7` / `triton-ascend 3.6` 构建**；
公开索引上的 post7 是 A2 构建，当前卡是 A5（950PR）。仅换 venv 解决不了，需要：

1. 提供 A5 版 `torch-npu 2.10.0.post7` 和 `triton-ascend 3.6.0.post2` 的 wheel/安装源；或
2. 使用官方 vllm-ascend 0.30 的 A5 镜像/容器；或
3. 退回 0.23 + 让 vllm-ascend 支持 `deepseek_v41`（需要改模型注册/config，不是调参）。

在此之前，原生臂无 TTFT/TPOT/ITL 可报；不把 0.23 的 `deepseek_v4` 冒充 v4.1。
