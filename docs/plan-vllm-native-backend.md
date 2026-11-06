# TileInfer 作为 vLLM 原生昇腾后端：Qwen3.6 单卡方案

> 状态：施工方案 v2（2026-10-27），取代 v1 的通用/多阶段版本。
> 关联：[`integration-vllm-native.md`](integration-vllm-native.md)（耦合点盘点）、
> [`status.md`](status.md)（现有能力证据）、[`architecture.md`](architecture.md)（内核契约）。

---

## 0. 范围

**目标**：`pip install vllm tileinfer` 之后，`vllm serve <Qwen3.6-35B-A3B>` 在 **1 张 950** 上跑起来，
不安装 `vllm_ascend`，性能不低于现网（vllm-ascend 0.23 + 图模式）。

**定位**：单模型后端。只保证 Qwen3.6 这条路又快又对，不做通用模型兼容。

**明确不做**（砍掉即减少工作量，不是延后）：

| 砍掉 | 原因 |
|---|---|
| TP / EP / PP / DP、HCCL 通信器、custom allreduce | 只有 1 张卡 |
| W8A8 / W4A8 / FP8-KV 等量化 | 现网是 bf16 |
| MLA / 稀疏注意力 / 滑窗 / 多模态 / mrope | Qwen3.6 不用 |
| 投机解码（MTP/DSpark） | 现网未启用（`performance.md` §2211） |
| 其他模型（Qwen2/2.5/3 dense、DeepSeek、Llama…） | 单模型定位 |
| CUDA / ROCm / CPU 推理路径 | 只做 NPU |

**保留**：平台层、Worker、attention 后端、算子层、混合 KV、ACLGraph、内核库（L1/L2 仍引擎无关）。

---

## 1. 目标模型：精确结构

全部来自本仓既有证据推导，标注了依据与待核对项：

| 项 | 值 | 依据 | 状态 |
|---|---|---|---|
| 层数 / 类型 | 40 层 = **30 层 GDN + 10 层 full-attention**（3:1 交替） | `status.md` §0 | 待 config 核对 |
| hidden_size | **2048** | `performance.md`（`shared_expert_gate=Linear(2048,1)`、`qkvz[2048→12288]`） | 待核对 |
| full-attn | **16 q heads / 2 kv heads / head_dim 256**，**output gate**（q 侧输出双倍），qk-norm | `o_proj[4096→2048]` ⇒ 16×256；`attn_qkv[2048→9216]` = 16×256×2 + 2×256×2 ⇒ 带 output gate | 待核对 |
| GQA group | **8** | 16/2；`vllm_ascend.py:629` 同口径 | 待核对 |
| GDN | **16 k heads × 128**、**32 v heads × 128**、conv kernel = 4 | `qkvz[2048→12288]` = 16×128×2 + 32×128×2 | 待核对 |
| MoE | **256 专家**，router **无 bias**（非 internal router），shared expert `intermediate=512` | `performance.md` E310 | 待核对 |
| engine block_size | **2048**（与 mamba state 页对齐） | `performance.md` §554/558 | 待核对 |
| 内核页 | **128**（`KERNEL_PAGE_SIZE`），2048 = 16 × 128 子页 | `vllm_ascend.py:62` | 已实现 |
| KV | 10 层 NHD attention 池 + 30 层 mamba state（conv_state + ssm_state） | 模型结构 | 待核对 |

依赖的 vLLM 上游模型代码（**不修改，只供我们适配**）：

```text
vllm/model_executor/models/qwen3_5.py                          # 模型定义（40 层，hybrid）
vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py    # GDN 层，调用 _C 算子
vllm/model_executor/layers/fused_moe/*                          # MoE
vllm/model_executor/layers/attention/*                          # Attention 层封装
```

> 关键利好：**模型层代码在上游 vLLM 里，不在 vllm-ascend 里**。vllm-ascend 只是替换了 GDN 的
> `forward`（`patch_qwen3_5.py:207` → `AscendGatedDeltaNetAttention.forward`）。我们不需要替换
> 模型代码，只需要提供它调用的算子。

---

## 2. 架构与依赖规则

```
L4  vllm（PyPI wheel，不改一行源码）
L3  tileinfer.vllm     唯一允许 import vllm 的层
    platform / worker / mem / graphs
    attn/{backend,metadata,impl,kv_cache,selector}
    ops/{dense,gdn,moe,attention,sampling}
    compat/{patches,audit,cuda_bridge}
    plugin.py（install 入口）
L2  tileinfer.serving  会话语义，不 import vllm
    buckets / planner / step / prefill / counters
L1  tileinfer.core     现有资产：metadata / plan / attention / kernels / testing
```

硬规则（CI 断言，见 §7）：

- L1/L2 禁止 `import vllm` / `import torch_npu` / `import vllm_ascend`；
- L3 禁止 `import vllm_ascend`（任何形式）；
- L3 对 vLLM 内部符号的每处引用都必须登记在 `compat/patches.py`。

`pyproject.toml`：

```toml
[project.entry-points."vllm.platform_plugins"]
tileinfer = "tileinfer.vllm.platform:register"

[project.entry-points."vllm.general_plugins"]
tileinfer = "tileinfer.vllm.plugin:install"
```

---

## 3. 工作包

每个包都有明确的**产出文件**和**验收判据**，按依赖排序；W1 必须先做，其余可并行。

### W1 · 事实核准（先做）

| 产出 | 内容 |
|---|---|
| `docs/notes/vllm-api-0.23.md` | 从**源码**（sdist，不是记忆）抄回的：`Platform` 抽象方法全集、`AttentionBackend`/`Impl`/`CommonAttentionMetadata`、`CustomOp.dispatch_forward` 规则、graph wrapper 的设备假设、`vllm.ops._C` 在 GDN/MoE 路径上的**全部**调用点 |
| `docs/notes/qwen36-config.md` | 模型 `config.json` 实值（§1 表逐行填） |
| `docs/op-gaps.json` | 算子缺口首版（由 `compat/audit.py` 生成） |

> 这一包决定后面所有代码的签名正确性。**不核准就写，等于写两遍。**

### W2 · 包骨架与入口点

产出：`tileinfer/serving/__init__.py`、`tileinfer/vllm/__init__.py`、`pyproject.toml` 入口点、
`tests/test_no_vllm_ascend.py`、`tests/test_layer_deps.py`。

验收：`pip install -e .` 后 `import tileinfer` 不拉 vllm；`pytest tests -m "not npu"` 全绿。

### W3 · `serving/` 抽取（把已验证的会话语义搬出 vllm_ascend.py）

| 文件 | 内容 | 来源 |
|---|---|---|
| `buckets.py` | `KERNEL_PAGE_SIZE`、`sub_pages`、`bucket_for`、`next_pow2`、`expand_page_table`、`pool_sub_view`、`pad_lens` | `vllm_ascend.py:62–152` |
| `planner.py` | plan 缓存、`declared_pool_cap` 键、后台编译、`prewarm`、失败集合、策略视图 | `:889–1330` |
| `step.py` | step guard + 设备侧 ragged 重建（H4）、子页表展开、`last_page_len` 原地算 | `:1334–1440` |
| `prefill.py` | `prefill_step`（三态统一 append）、`PrefillShapeManager` | `:286–890` |
| `counters.py` | 探针计数（纯 dict） | `:927–953` |

验收：CPU 单测（无需设备）覆盖桶选择、子页换算、ragged 重建定值；行为与抽前逐位一致。

### W4 · attention 后端

| 文件 | 内容 |
|---|---|
| `attn/backend.py` | `TileInferAttentionBackend`：`get_name/get_impl_cls/get_builder_cls/get_kv_cache_shape/get_supported_kernel_block_sizes/validate_configuration/accept_output_buffer` |
| `attn/metadata.py` | `TileInferMetadata` + `Builder.build()`：只取引用 + 算 `max_query_len` + 补桶 |
| `attn/impl.py` | `forward(layer, q, k, v, kv_cache, meta, output)`：先写 KV → 调 `serving` → 写 output |
| `attn/kv_cache.py` | `slot_mapping` → NHD 池的扁平散射（decode/prefill 共用） |
| `attn/selector.py` | 层选择：full-attn → 我们；GDN → 上游 |

验收：CPU 契约测试（mock vllm 类型）+ 设备上 top-5 logprobs 对齐冻结基线。

### W5 · platform / worker / mem

| 文件 | 内容 |
|---|---|
| `platform.py` | `TileInferPlatform`（device_type=npu、OOT）、`get_attn_backend_cls`、worker 选择（写 `parallel_config.worker_cls`；0.23 无 `Platform.get_worker_cls`）、`check_and_update_config`（block_size 对齐 2048、禁止不支持项）、显存接口、能力位 |
| `worker.py` | `NPUWorker(GPUWorker)`：`init_device`、`determine_available_memory`、KV 池初始化、`compile_or_warm_up_model` |
| `mem.py` | 显存探测、KV 池切分（按 spec）、profiling |

验收：`vllm serve` 能加载权重并分配 KV 池；`npu-smi` 可见占用。

### W6 · 算子层（14 个，见 §4）

验收：每个算子有 CPU oracle 测试；设备上与 torch 参考 `max|diff| < 1e-3`（bf16）。

### W7 · 混合 KV 与 mamba state

验收：40 层里 10 层拿到 NHD 池、30 层拿到 mamba state；长上下文不串请求（前缀缓存/换出场景）。

### W8 · 图模式

验收：`capture_calls > 0 && fallbacks == 0`，replay 期零 Python 路径，生成文本与 eager 一致。

### W9 · 单模型性能专项（与 W6/W8 并行推进）

| 项 | 现状 | 目标 |
|---|---|---|
| attention d256 decode（dsplit-v2） | 已验证 | 保持 |
| split-KV 默认化 | 可用未默认（KI-1 已修） | 8K+ 上下文默认开 |
| GDN 9a/9b/9c/gating | 有内核，部分未接入 | 按 profile 逐一接 |
| MoE grouped GEMM | 0.86x 未接入 | ≥1.0x 或保持 aclnn 等价路径 |
| shared expert 折叠/overlap | 有实现 | 按 profile 接 |
| 小算子（l2norm/cta/rearrange） | `kernels/gdn/small_ops.py` | 接入 |

### W10 · 验收

数值：与冻结基线（top-5 logprobs）逐位一致；性能：TTFT/TPOT/吞吐 ≥ 现网；长稳：连续 24h 无 crash。

---

## 4. 算子清单（Qwen3.6 全集）

| # | 算子 | 用于 | 上游调用点 | native 兜底 | 优先级 |
|---|---|---|---|---|---|
| 1 | `rms_norm` / `fused_add_rms_norm` | 全部层 | `torch.ops._C` | 有（torch） | P0（热） |
| 2 | `rotary_embedding` | 10 层 attention | `torch.ops._C` | 有 | P0 |
| 3 | `silu_and_mul` | MoE/共享专家 | `torch.ops._C` | 有 | P0 |
| 4 | **KV 写入** | 10 层 attention | 我们自己 | — | P0 |
| 5 | **TileInfer attention** | 10 层 attention | 我们自己 | — | P0 |
| 6 | `causal_conv1d_fn` | GDN prefill | `torch.ops._C` | **无** | P0（硬） |
| 7 | `causal_conv1d_update` | GDN decode | `torch.ops._C` | **无** | P0（硬） |
| 8 | `chunk_gated_delta_rule` | GDN prefill | `torch.ops._C` | **无** | P0（硬） |
| 9 | `fused_recurrent_gated_delta_rule` | GDN decode | `torch.ops._C` | **无** | P0（硬） |
| 10 | `gated_delta_rule_update` / `fused_post_conv_prep` | GDN 状态更新 | `torch.ops._C` | **无** | P0（硬） |
| 11 | `l2norm`（q/k 归一） | GDN | `torch.ops._C` | 有（`F.normalize`） | P1 |
| 12 | `fused_gdn_gating` | GDN gate | `torch.ops._C` | 有（torch 组合） | P1 |
| 13 | `topk_softmax` | MoE 路由 | `torch.ops._C` | 有（`torch.topk`） | P0 |
| 14 | `moe_align_block_size` + `fused_experts` | MoE 前向 | `torch.ops._C` | 部分（需分组 GEMM） | P0 |
| 15 | `moe_sum` / shared expert | MoE 合并 | `torch.ops._C` / 上游 Python | 有 | P1 |
| 16 | sampler（top-k/p） | 采样 | 纯 torch | 有 | P2 |

> 判据：**6–10 是唯一没有 native 兜底的族**，它们决定"能不能跑"；1–3、13–15 决定"跑多快"。
> 实现顺序：6–10（正确性）→ 1–5（attention 链路）→ 13–15（MoE）→ 性能替换。

---

## 5. 混合 KV（最容易踩的坑）

```
10 × full-attention 层 → NHD 池 [2, blocks, 2048, kv_heads, 256]
                         内核视图: reshape → [2, blocks*16, 128, kv_heads, 256]（零拷贝）
30 × GDN 层            → mamba state（conv_state + ssm_state），页与 attention 页对齐(2048)
```

要点：

1. `get_kv_cache_shape()` 只对 attention 层生效；GDN 层的 spec 由**模型代码**给出，我们只声明
   `support_hybrid_kv_cache() -> True`；
2. 池是连续大块，`view(-1, kv_heads, dim)` 与 128 子页视图**同一块内存**，所以 KV 写入只有一条路径；
3. mamba state 的换出/换入（prefix caching、抢占）由 vLLM 的 block manager 负责，我们只保证
   state tensor 的地址与形状稳定；
4. `slot_mapping` 是**逐 token** 的引擎坐标，decode 与 chunked prefill 完全一致，不区分。

---

## 6. 图模式

- 捕获前：warmup 期把所有 bucket 的 plan 建好（`serving/planner.py`）；
- 捕获中：`run()` 只查缓存，命中不了就**走降级并计数**，绝不编译/分配/同步；
- 断言化：`assert plan is not None`，不让"忘了预热"变成静默 NaN；
- 验收沿用现有探针口径：`capture_calls / fallbacks / markers`。

---

## 7. 验收体系

| 层 | 环境 | 内容 |
|---|---|---|
| L0 | CI（无卡） | 现有 145 例 + 新增契约测试 |
| L1 | CI（无卡） | 算子 CPU oracle、KV 写入逐元素、桶/子页换算 |
| L2 | 设备 | 内核套件（现有）+ 每个新算子 |
| L3 | 设备 | 引擎冒烟：1 请求，logprobs 对齐冻结基线 |
| L4 | 设备 | 端到端：长度 × batch × 图/eager |
| L5 | 设备 | 性能：TTFT/TPOT/吞吐 vs 现网 |

守卫测试：

```text
tests/test_no_vllm_ascend.py   仓库内无 vllm_ascend 引用（legacy/ 除外）
tests/test_layer_deps.py       L1/L2 不 import vllm / torch_npu（AST 检查）
tests/test_kv_cache_write.py   slot_mapping 写入 vs 参考逐元素相等
tests/test_serving_buckets.py  桶/子页/ragged 重建定值
tests/test_op_cpu_oracle.py    每个算子的 CPU 参考实现对比
```

---

## 8. 风险

| # | 风险 | 缓解 |
|---|---|---|
| R1 | `torch.cuda` 硬编码面广，兼容层失控 | W1 先量化（grep 计数），patch 表设上限（40 条）作为熔断 |
| R2 | GDN 算子族无 native 兜底 | 最优先做，先用 torch 组合版跑通正确性，再换内核 |
| R3 | 上游 vLLM 接口漂移 | 钉 `vllm>=0.23,<0.24` + 契约测试 |
| R4 | 单卡被占用，设备窗口不足 | 设备测试做短小可重入；CPU 侧用 mock 前置 |
| R5 | 失去 FIA 后无降级路径 | 三级：启动期拒绝 / plan 未就绪降级 / 运行期熔断，均计数 |

---

## 9. 施工顺序（依赖图）

```
W1 事实核准 ──┬─→ W2 骨架 ─→ W3 serving 抽取 ─→ W4 attention ─┐
              │                                                ├─→ W7 混合KV ─→ W8 图模式 ─→ W10 验收
              └─→ W5 platform/worker ─→ W6 算子(6-10 先) ──────┘
                                                        W9 性能专项（贯穿）
```

第一批文件（可直接开工）：

1. `docs/notes/vllm-api-0.23.md`（W1）
2. `pyproject.toml` 入口点 + `tileinfer/serving/{__init__,buckets,counters}.py`（W2/W3）
3. `tileinfer/vllm/{__init__,plugin}.py` + `attn/kv_cache.py`（W2/W4）
4. `tests/test_no_vllm_ascend.py` + `tests/test_layer_deps.py`（W2 守卫）
5. `tests/test_serving_buckets.py` + `tests/test_kv_cache_write.py`（W3/W4 验收）

---

## 10. 施工看板（随实现更新）

状态：✅ 已实现且有 CPU 测试 · 🟡 部分 · ⬜ 未开始

| 部件 | 文件 | 状态 | 备注 |
|---|---|---|---|
| 子页/桶/池容量算术 | `serving/buckets.py` | ✅ | 含 `padded_rows`、`declared_pool_cap` |
| 探针计数 | `serving/counters.py` | ✅ | 线程安全 |
| 设备侧 ragged 重建 | `serving/step.py` | ✅ | 零分配改写（index_select + 常量偏移） |
| plan 缓存 / 单飞 / 预烧 | `serving/planner.py` | ✅ | 失败记忆 + 幂等预烧 |
| KV 写入（slot_mapping） | `vllm/attn/kv_cache.py` | ✅ | decode/prefill 同一路径 |
| patch 注册表 | `vllm/compat/patches.py` | ✅ | why/since/remove_when + 容错安装 |
| GDN `l2norm_fwd` | `vllm/ops/gdn.py` | ✅ | eps 在平方和内（已核对 kernel） |
| GDN `chunk_local_cumsum` | `vllm/ops/gdn.py` | ✅ | 含 reverse 与 cu_seqlens |
| GDN `solve_tril` | `vllm/ops/fla/solve_tril.py` | ✅ | **`(I+A)⁻¹`**、整块 BT×BT（非块对角）；30 例对拍 fp64 + 3 个错误版本反向对照 |
| GDN `causal_conv1d`（fn/update） | `vllm/ops/fla/causal_conv1d.py` | ✅ | 两入口布局不同（fn=`[dim,T]`，update=token-major）；32 例，含「逐 token == 一次 prefill」 |
| GDN 门控 + 状态更新 | `vllm/ops/fla/gating.py` | ✅ | softplus/exp/sigmoid 顺序已核；状态原地更新 |
| GDN `chunk_scaled_dot_kkt` / `chunk_delta_h` / `chunk_o` / `wy_fast` | — | ⬜ | 四个相互耦合（共享 A/g/beta 约定），**串行做**，不适合并行 |
| GDN `fused_recurrent_*`（decode） | — | ⬜ | 与 prefill 不同入口，`packed_decode` 单独写 |
| MoE：`topk_softmax` / `fused_experts` / `moe_align_block_size` | — | ⬜ | |
| attention 后端（backend/metadata/impl/selector） | `vllm/attn/` | ⬜ | 契约待从 `v1/attention/backend.py` 转录 |
| platform / worker / mem | `vllm/` | ⬜ | `Platform` ~70 方法，含 `update_block_size_for_backend` / `_align_hybrid_block_size` |
| 入口点 + plugin | `pyproject.toml`, `vllm/plugin.py` | ⬜ | 依赖上面两个模块存在后再加，避免装完即坏 |
| 图模式 | `vllm/graphs.py` | ⬜ | `vllm/compilation/cuda_graph.py` 已下载待读 |

### 下一步（严格按序）

1. 读 `fla_solve_tril.py` 的 `solve_tril` 主函数与三个 kernel，确认「A 的三角形式」与分块合并规则；
2. 实现 `solve_tril` + CPU oracle（对照 `torch.linalg.inv`），再实现 `chunk_scaled_dot_kkt`；
3. 读 `chunk_delta_h` / `chunk_o` / `wy_fast`，补齐 prefill 链；
4. `fused_recurrent_gated_delta_rule_packed_decode`（decode 热路径）；
5. 转到 attention 契约转录 + `platform.py`。
