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
| 0.30 + `--offload-backend prefetch --offload-params experts` | 走到 offload 阶段；主机 pinned 分配 OOM：`aclrtMallocHostWithCfg ... 207001` |
| 释放主机后重试（host free 456 GB） | 原生进程把 host used 推到 612 GB，随后仍 `207001`；kill 后 host 回到 free 525 GB |

## 为什么不是“单卡 HBM 被别的 v4.1 占了”

- 重试时 NPU 只有 5.4 GB 基线占用，`npu-smi` 无进程，HBM **空闲**；
- `ps` 没有其他 v4.1-Flash 进程，无需 kill；
- 真正的限制是 **原生 vLLM 0.30 没有 DeepSeek-V4.1 的专家 host-offload**：
  - 模型 475.2 GB，HBM 123 GB；
  - 原生 prefetch offload 按“整层”把权重复制到 pinned host，475 GB 权重 + pinned 副本
    超过主机 628 GB 总量；即使 host free 525 GB，进程仍涨到 612 GB 后 `aclrtMallocHost` OOM；
- 我们之前的 TileInfer 部署能跑，是因为它用 **mmap 权重 + 自管 expert arena 按专家换入**，
  不是原生 vLLM 的整层 offload。

因此“之前验证可以部署”验证的是 TileInfer 路径，不是原生 vllm-ascend；原生在当前
128 GB HBM + 628 GB host 上无法容纳 475 GB 模型。

## 判断与恢复条件

0.30 已包含 DSV4.1 模型代码，且手动注册模型 + `block-size=64` 可以绕过 CUDA 注意力和 block size
问题；当前真正卡住的是 **单卡容量**：

- 模型 475.2 GB，HBM 123 GB；
- 原生 prefetch offload 需要约 350 GB 主机侧副本；进程实际涨到 612 GB host used；
- kill 后 host free 525 GB，仍不足以容纳“475 GB 权重 + 350 GB pinned 副本”；
- vLLM 0.30 的 UVA offload 在 NPU V2 上不支持，prefetch offload 又需要先固定住整段权重。

需要的条件（任一）：

1. 官方 vllm-ascend A5 镜像（含 **专家级 host-offload**，而不是整层 pinned 拷贝）；
2. 多卡/更大 HBM：原生不做 host-offload 时，单卡 123 GB 放不下 475 GB 模型；
3. 若坚持用 vLLM prefetch 整层 offload：主机需能同时容纳 475 GB 权重和约 350 GB pinned
   副本（约 900 GB 可用），当前 628 GB 总量做不到。

在此之前，原生臂没有可报的 TTFT/TPOT/ITL。
