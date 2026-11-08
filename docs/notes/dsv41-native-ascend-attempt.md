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

## 已有专家级卸载分支（关键）

`/home/dlb/vllm-ascend` 的 `feat/ds4-expert-offload`（commit `358be06`）确实实现了
**单卡专家权重 offload**，文档 `docs/source/tutorials/models/DeepSeek-V4-Flash.md` 第 11 节给的是：

```bash
--offload-backend prefetch \
--offload-group-size 2 \
--offload-num-in-group 1 \
--offload-params '["w13_weight","w2_weight"]'
# pinned 受限时：
VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1
```

这说明 vLLM 侧可以只卸载 `w13_weight` / `w2_weight`，而不是整层权重；之前我用
`--offload-params experts` 是**错的匹配段**，且 PyPI 0.30 的 offloader 在初始载入时仍先把
119 GB 权重放进 HBM，所以出现“900 GB / 初始 OOM”。

该分支的限制：

- `.github/vllm-release-tag.commit` 是 **v0.23.0**；
- 分支源码里只有 `deepseek_v4`，**没有 `deepseek_v41`**；
- 我们的 checkpoint 是 `deepseek_v41`，直接换分支不能加载 V4.1。

所以正确路径不是继续试 PyPI 0.30 的 offload 参数，而是二选一：

1. 把该分支的 `model_executor/offloader/prefetch.py` +
   `patch_process_weights_after_loading.py` 的**专家级载入补丁**移植到 0.30 的 v41 栈；
2. 让该分支支持 `deepseek_v41`（模型/config 注册），再按第 11 节 recipe 启动。

在此之前，原生 v41 仍不能单卡跑；但这证明“900 GB 不是模型本身要求”，而是 offload 粒度/载入顺序问题。

### 移植尝试（2026-11-08 晚）

已在 `.venv_ascend30` 内做了隔离移植：

- 用分支的 `vllm_ascend/model_executor/offloader/prefetch.py` 替换 0.30 版本；
- 补 `ACL_FORMAT_FRACTAL_NZ_C0_16=50` 和 `AscendPrefetchOffloader` 别名；
- 在入口把 `vllm.v1.worker.gpu.model_runner.create_offloader` 指向 vllm-ascend 的工厂；
- 启动参数改为 `--offload-params w13_weight w2_weight`（不再用 `experts`），
  并设置 `VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1`。

结果：offloader 已正确选成 `NPUPrefetchOffloader`，参数也解析为 `['w13_weight','w2_weight']`，
但仍在 `wrap_modules` 的模型构造阶段 OOM：

```
NPU out of memory. Tried to allocate 4.22 GiB
119.42 GiB already allocated; 2.96 GiB free
```

说明 0.30 V4.1 的 V2 模型构造/载入顺序仍是“先把权重放 HBM，再 wrap”，
分支的专家级释放没有在构造阶段生效；需要继续改 V2 的 model construction/load hook，
或把分支的 V4 模型路径整体接到 0.30。移植尚未跑通。

后续又试了两条：

- 放行 V1（patch `platform.py` 的 V2 强制校验）后，V1 仍在同一 `wrap_modules` 处
  `119.44 GiB already allocated`；
- 在 `wrap_modules` 内对每个 `_NPUModuleOffloader` 立即 `post_init()`，仍 OOM。

**2026-11-08 21:33 的关键进展**：把 `--offload-group-size` 从 4 改成 **1**、
`--offload-num-in-group 1`（每层都卸载），专家权重 offload 生效，HBM 稳定在 **24.6 GiB**；
whitelist 已正确匹配：

```
mlp.experts.routed_experts.w13_weight
mlp.experts.routed_experts.w2_weight
```

新的阻塞点变成 **Engram 的 98.3 GB host 分配**，不是专家：

```
vllm_ascend/models/deepseek_v41/engram/npu.py:157
HostUvaBuffer: aclrtMallocHost failed: rc=207001 size=98305579008
```

即专家 offload 已经成功，剩下的 98 GB engram UVA 主机内存在当前 host/pinned 限制下分配失败。
下一步是：
1. 将 `HostUvaBuffer` 拆成多个较小的 host 段（`ptrs` 已按 `CHUNK_ROWS` 分块，具备拆分条件）；
2. 或关闭 engram host offload（但 189 GB engram 放 HBM 不可行，需要另想办法）；
3. 或使用 pageable + 分段 `aclrtHostRegisterV2`。

**2026-11-08 21:38 起**：已按方案 3 改造 `HostUvaBuffer`：用一个 `mmap` 连续虚拟区，
按 `CHUNK_ROWS` 分段 `aclrtHostRegisterV2`，`ptrs` 每个 chunk 一个 device pointer。
这样绕过了单次 98 GB `aclrtMallocHost` 的 207001。

结果：engram 注册不再报错，但**主机 RSS 涨到 543 GB**；host 总 628 GB，叠加其他租户后
只剩 4 GB available，模型载入在 module 39 之后停住（12+ 分钟无进展），只能 kill。

根因从“98 GB 单块分配”变成“**475 GB 模型 + 98 GB engram 匿名 host + 运行时**”超过
host 容量。下一步必须把 engram 改成**文件-backed mmap**（页面可回收）或换更大 host/多卡，
而不是继续拆 chunk。

**进一步核实**：engram 的 mmap 会走 `aclrtHostRegisterV2`，UVA 注册会把页面
**pin 住**，因此即使改成文件-backed mmap，98 GB 仍然不可回收，host RSS 不会下降。
当前 628 GB host 上有约 100 GB 是 VSCode/Node 等常驻进程；模型 475 + engram 98 +
运行时已经接近/超过剩余 526 GB。要单卡跑原生，实际需要：

- host ≥ 700–750 GB 可用（且不为其他进程占用）；或
- 原生 engram 改用 **pageable + 按需 H2D gather**（不再 UVA pin 98 GB）；或
- 多卡/更大 HBM，把 engram 放 HBM。

这不是继续改 `offload-params` 能解决的；瓶颈已经从专家转移到 engram 的 UVA 常驻内存模型。

**swap 尝试（2026-11-08 22:00）**：在 `/home` 建了 128 GB swapfile 并 `swapon`，再启动原生；
结果 host 仍为 624 GB used / 4 GB free，swap 只用了 3 GB，模型载入继续停住。
说明这些 CPU 权重/engram 页面是 **pinned**，内核不能换出；swap 不能解决问题。
已 `swapoff` 并删除 swapfile，host 回到 free 525 GB。

**pageable engram 尝试（2026-11-08 22:10）**：给 `HostUvaBuffer` 加了
`VLLM_ASCEND_ENGRAM_PAGEABLE=1` 分支：不再 `aclrtHostRegisterV2`，engram 表为 pageable
CPU tensor，`gather_dequantize_host_uva` 改为 CPU 按 ids 取行、反量化后 H2D；
同时建 64 GB swap。结果 engine 被 OOM killer 杀掉（`EngineCore: -9`），64 GB swap 被用满。
说明即使 engram 可换出，模型 475 GB + 其他 100 GB 仍把 host 压满，pageable 方案在当前
628 GB host 上也不够；需要更大 host / 多卡，或把模型 CPU 权重也改成可回收 mmap。

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
