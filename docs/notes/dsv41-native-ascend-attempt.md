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

## 判断

0.30 已包含 DSV4.1 模型代码，但当前组合仍不可运行，缺三层：

1. **A5 构建的 `torch-npu 2.10.0.post7` / `triton-ascend 3.6`**：公开索引的 post7 是 A2；
2. **与 vllm-ascend 0.30 匹配的 vLLM 0.30 构建**：V2 runner 仍走上游
   `DeepseekV41LLMForCausalLM`，其注意力 assert 要求 CUDA；V1 又被平台禁止；
3. **真实 `fla_npu`**：公开索引无此包，当前用 import-only stub 才走到 engine 初始化。

仅换 venv 解决不了；需要官方 A5 镜像/配套 vllm 轮子，或把 V2 runner 切到
`vllm_ascend.models.deepseek_v41` 的 Ascend 实现。没有这些，原生臂没有可报的 TTFT/TPOT/ITL。
