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
| 0.30 + `vllm 0.30 + transformers 5.14.1 + torch-npu post7` | import 通过；启动报 **A5/A2 设备类型不匹配** |
| 0.30 + 强制 `_build_info.__device_type__='A5'` + `VLLM_USE_V2_MODEL_RUNNER=1` | 继续推进到 engine 初始化；带 `--cpu-offload-gb` 报 `get_accelerator_view_from_cpu_tensor is currently not supported in: npu` |
| 0.30 + 去掉 CPU offload | 报 **`AssertionError: DeepseekV4 attention requires a CUDA device`**：V2 runner 的 `load_model` 仍加载上游 `DeepseekV41LLMForCausalLM` |
| 0.30 + `VLLM_USE_V2_MODEL_RUNNER=0` | 被平台校验直接拒绝：`DeepSeek V4.1 on Ascend A5 requires Model Runner V2` |
| 0.30 + 手动 `vllm_ascend.register_model()` + V2 | 绕过上游 CUDA 注意力，进入模型构造；`block_size=16` 报 `KeyError: 16` |
| 0.30 + `--block-size 64` | 继续到权重加载；**475 GB 模型在 123 GB HBM 上 OOM**（119 GB allocated，申请 8.44 GB 失败） |
| 0.30 + `--cpu-offload-gb 300` | `get_accelerator_view_from_cpu_tensor is currently not supported in: npu` |
| 0.30 + prefetch offload（`--offload-group-size 5 --offload-num-in-group 3`） | 仍先加载 119 GB 权重后 OOM；该 offload 未覆盖初始权重载入 |

## 判断

0.30 已包含 DSV4.1 模型代码，且手动注册模型 + `block-size=64` 可以绕过 CUDA 注意力和 block size
问题；当前真正卡住的是 **A5 版 torch-npu/triton + 单卡 HBM 不足以载入 475 GB 模型，而
vLLM 0.30 的 CPU/prefetch offload 在这套 V2+A5 组合上不可用**。

需要的条件（任一）：

1. A5 版 `torch-npu 2.10.0.post7` / `triton-ascend 3.6` + 官方配套 vLLM 0.30；
2. 官方 vllm-ascend A5 镜像（含可用的权重 offload）；
3. 多卡/更大 HBM（原生不做 host-offload 时，单卡 123 GB 放不下 475 GB 模型）。

在此之前，原生臂没有可报的 TTFT/TPOT/ITL。
