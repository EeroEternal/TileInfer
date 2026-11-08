# 原生 vllm-ascend V4.1-Flash：可回收 mmap 权重 + 逐专家 NZ 方案

## 1. 为什么需要

原生 vLLM 0.30 的 `PrefetchOffloader` 在 `_CpuParamOffloader` 里把每个被
offload 的参数复制到 **anonymous**（可选 pinned）host 内存：

- 未压缩的 w13/w2 在 host 上是 **543 GB**（w13 9.06 GB/layer，w2 4.52
  GB/layer，40 层）；
- Engram 两张表共 **196 GB**（`HostUvaBuffer`，`aclrtHostRegisterV2` 钉住）；
- 单卡 host 只有 628 GB，另有 VSCode/DSH 等常驻进程。

即使先把 `/tmp` 下 78 GB 历史 tensor dump 清掉，543 + 196 = 739 GB 仍然
超过 628 GB。之前的失败点因此不是设备显存，而是 host：

1. `/tmp` 旧 dump 占 87 GB；
2. 543 GB 匿名页 + 196 GB pinned Engram 已经 739 GB；
3. host-NZ 模式还要再复制一份 543 GB（`prepare_host_nz`），必然 OOM。

## 2. 设计

### 2.1 可回收 mmap 模型 CPU 权重

`tools/dsv41/native_mmap_store.py` 提供 `MmapParamStore`：为每个 offload 参数
创建 scratch 文件，`ftruncate` 后用 `MAP_SHARED` 映射，再通过
`torch.frombuffer` 暴露成 CPU tensor。`VLLM_CPU_WEIGHT_MMAP_DIR` 未设置时
行为与原生完全一致。

`tools/dsv41/native_mmap_offload_patch.py` 修改：

- `vllm/model_executor/offloader/prefetch.py`
  - `_CpuParamOffloader._offload_to_cpu_internal`：优先创建 mmap store；
  - `_update_cpu_storage_from_param`：`process_weights_after_loading` 之后把
    新权重写回同一 store，而不是新分配 anonymous 内存；
  - `_cpu_store` 生命周期跟随参数；host-NZ 丢弃 ND 时 `release()`。
- `vllm/model_executor/layers/fused_moe/runner/moe_runner.py`
  - `select_experts` 之后调用 `offloader.maybe_load_selected_experts(topk_ids)`。
- `vllm_ascend/model_executor/offloader/base.py`
  - `VLLM_ASCEND_OFFLOAD_SELECTIVE_TOKENS` 控制选择性加载阈值。

文件页是 clean page cache，内核可在压力下直接回收，不会丢数据。

### 2.2 不再复制第二份 host NZ：逐专家 cast

host-NZ 模式需要 543 GB 的第二份 pinned/匿名副本，host 放不下。探针
`tools/dsv41/probe_nz_layout.py` 验证：

- NZ_C0_16 cast 后物理字节 = ND 字节（无 4-bit 压缩）；
- 对单专家 `[1, M, K]` 做 cast，与完整 `[E, M, K]` cast 的对应专家块
  **逐字节相同**（e = 0/1/7/383 全部相等）；
- 单专家 cast 约 **0.19 ms**（121 GB/s）。

因此 decode 时只需把 router 选中的 6 个专家逐个 H2D、cast、D2D 写入共享
NZ buffer 的专家偏移，不再需要 host NZ 副本。

`tools/dsv41/native_per_expert_patch.py` 增加
`VLLM_ASCEND_OFFLOAD_PER_EXPERT_NZ=1`：

- `post_init` 只调用 `create_nz_buffers()`，不调用 `prepare_host_nz()`；
- `load_selected_experts()` 走 `_load_selected_experts_expert_nz()`：
  `staging.copy_(cpu[e:e+1])` → `_cast_nd_to_nz(staging)` →
  `aclrtMemcpyAsync(nz_ptr + e*per, ...)`；
- prefill 仍走整层 ND->NZ（`start_onload_to_static`）。

### 2.3 Engram 改为 pageable + file-backed

`VLLM_ASCEND_ENGRAM_PAGEABLE=1` 时，`HostUvaBuffer` 不做
`aclrtHostRegisterV2`，而是用 `gather_dequantize_host_uva` 的 CPU fallback：
按 token 取行、CPU 反量化、再 H2D。`native_per_expert_patch.py` 把该
fallback 的 `codes/scales` 改为 `VLLM_CPU_WEIGHT_MMAP_DIR` 下的 mmap 文件，
196 GB 变成可回收 page cache。

Engram 每 token 只访问少量行，磁盘 miss 的代价远小于权重。

## 3. 启动

```bash
VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1 \
VLLM_ASCEND_OFFLOAD_HOST_NZ=0 \
VLLM_ASCEND_OFFLOAD_PER_EXPERT_NZ=1 \
VLLM_ASCEND_OFFLOAD_SELECTIVE_TOKENS=64 \
VLLM_ASCEND_ENGRAM_PAGEABLE=1 \
VLLM_CPU_WEIGHT_MMAP_DIR=$PWD/.work/native30_mmap \
bash tools/dsv41/serve_native_v41_mmap.sh
```

## 4. 测量

`tools/dsv41/perf_native_matrix.sh` 覆盖单/4 并发 × 127/2K/4K/16K/32K，
TTFT、TPOT、ITL P50/P90、实际输出 token 数，并采样 `npu-smi` HBM 峰值。

## 5. 2026-11-09 实跑结果：启动仍被厂商 kernel 卡住

### 5.1 已经打通的部分

按上面的 mmap + 逐专家方案，原生栈已经能做到：

- **host 容量**：权重 272 GB（`deepseek_v4_fp8` 的 packed MXFP4，w13
  `[384,4608,2560] uint8` 4.53 GB/layer、w2 2.26 GB/layer）+ Engram 196 GB
  file-backed page cache，主机 628 GB 足够，不再 OOM；
- **权重载入**：48/48 shards，`Loading weights took 231–235 s`，
  `Model loading took 27.85 GiB memory`；
- **offload**：40 个 decoder layer、HBM 24.7 GiB、per-expert NZ buffer
  6.79 GB、`host_nz=False`、`selective_token_threshold=64`；
- **关键修复**（缺一不可）：
  1. `--quantization deepseek_v4_fp8`：否则走通用 `fp8`，专家被当成 9 GB/layer
     的 float8，host 543 GB，且 checkpoint 的 MXFP8 group scale 与 block scale
     对不上；
  2. `deepseek_v4_fp8` 下 dense scale 名为 `weight_scale`、专家 scale 名为
     `w13_weight_scale`/`w2_weight_scale`，恢复原生 `.scale -> .weight_scale`
     映射即可；不要用 `weight_scale_inv`；
  3. `device_loading_context` 退出前调用
     `record_offloaded_param_formats()`，把 NZ 参数还原为 ND，否则
     `copy_` 内部格式报错；
  4. 自定义算子：`ASCEND_CUSTOM_OPP_PATH` + `LD_PRELOAD=libcust_opapi.so`，
     并让 `LD_LIBRARY_PATH` 优先使用 CANN 9.3.0 的 x86_64 kirin
     `libstdc++.so.6`（系统 6.0.28 缺 `GLIBCXX_3.4.29`）。

### 5.2 阻塞点：`aclnnHcPre` 的 950 kernel 缺 tilingKey 1003

profile run 在第一个 `DeepseekV41DecoderLayer.hc_pre` 处失败：

```
call aclnnHcPre failed ... Cannot find tilingKey[1003] in kernel json file of HcPre.
```

已确认：

- CANN 9.1.1/9.3.0 的 `libopapi.so` 只有 `aclnnMhcPre`，没有 `aclnnHcPre`；
  `aclnnHcPre` 在 vllm-ascend 的 `libcust_opapi.so` 里；
- wheel 自带 `_cann_ops_custom` 只有 **ascend910b** kernel；仓库
  `/home/dlb/vllm-ascend` 的 950 kernel 是 10 月旧构建，`hc_pre.json`
  只有 `tilingKey 1000/1001`，而 0.30 的 host tiling 请求 **1003**
  （对应 `HcPreMSplitCorePremixArch35`，源码里有 1003 分支但预编译 bin 没有）；
- 换用 910b、换 libstdc++、换 opapi 路径都只能解决“找不到符号/SoC”，
  无法补出 1003 的 950 device binary。

因此原生 vllm-ascend 在 Ascend950PR 上**仍无法完成 profile/起服**，本轮
没有可测的 TTFT/TPOT/ITL。要闭环需要厂商提供带 950 tilingKey 1003 的
`cann-ops-transformer-custom` 包，或用同版本源码重新编译 `HcPre` 950 kernel；
这不是 host 内存或本仓库 offload 代码能解决的。

复现命令（会加载约 468 GB host mmap，profile 在 HcPre 处失败）：

```bash
bash tools/dsv41/serve_native_v41_mmap.sh   # 见该脚本头部环境变量
```

TileInfer 自身的 P1/P3/P4/block-cache 结果不受影响。

