# 迁到 vLLM 原生集成：现状盘点与待实现清单

> 目标：去掉对 `vllm_ascend` 内部结构的依赖，让 TileInfer 通过 **vLLM 官方扩展点**接入。
> 本文只做两件事：说清现在卡在哪，列出还需要实现什么（含验收口径）。
> 现有集成的行为与证据见 [`integration-vllm-ascend.md`](integration-vllm-ascend.md)、
> [`status.md`](status.md)。

---

## 0. 先对齐一个前提

`vllm` 上游**没有 Ascend 平台**。昇腾上的设备发现、Worker、ModelRunner、显存/KV 池分配、
ACLGraph 捕获、HCCL、以及 RMSNorm/Rotary/MoE/采样/GDN 这些算子，全部由 `vllm_ascend` 提供
（它本身就是通过 `vllm.platform_plugins` 入口注册进去的 out-of-tree 平台插件）。

所以"移除 vllm-ascend"有三种口径，工作量差一个数量级，**先选口径再谈实现**：

| 口径 | 含义 | 还要不要装 `vllm_ascend` | 量级 |
|---|---|---|---|
| **A. attention 层原生**（推荐） | TileInfer 只实现 vLLM 的 `AttentionBackend` 契约，不再继承/猴补 `vllm_ascend.attention.*`、不再改 `NPUPlatform` 以外的任何东西 | 要装，仅当"平台 + 算子底座" | 2–3 周 |
| **B. 只留平台** | A + 把 GDN/MoE/采样那些猴补全删掉 | 要装，但不再被 patch | A + 1 周 |
| **C. 彻底不装** | TileInfer 自带 platform plugin / worker / 算子面 | 不装 | 6–12 人月，风险高 |

本文 §4 是 A 口径的清单，§5 是 C 口径额外要做的。

> **决策更新（2026-10-27）**：口径已定为 **C —— 彻底不装 `vllm_ascend`**，TileInfer 自身作为
> vLLM 的昇腾后端（platform + attention + ops），定位同步从"引擎无关算子库"改为"vLLM 原生后端"。
> 执行方案见 [`plan-vllm-native-backend.md`](plan-vllm-native-backend.md)。本文保留作为
> **耦合点盘点与缺口清单**（§2/§4 的条目在 C 口径下全部适用，只是不再有"只做 attention 层"
> 这个中间态），§0 的推荐阅读时请以新文档为准。

---

## 1. 当前架构（按实际代码）

```
┌──────────────────────────────────────────────────────────────────────┐
│ 引擎层        vLLM 0.23  →  vllm_ascend 0.23 (NPUPlatform / Worker /   │
│                              ModelRunner / aclnn ops / FIA 算子)      │
└───────────────┬──────────────────────────────────────────────────────┘
                │ ① 注册: vllm.general_plugins → install()             │
                │ ② 选择: 猴补 NPUPlatform.get_attn_backend_cls        │
                │ ③ 元数据: AscendMetadata(block_tables/seq_lens/       │
                │           seq_lens_device/query_start_loc/attn_state)│
┌───────────────▼──────────────────────────────────────────────────────┐
│ integrations/vllm_ascend.py  (2785 行, 一个文件两层东西)              │
│  ├─ 会话语义(可移植): bucket / plan cache / prewarm / 子页视图 /       │
│  │                    step guard / prefill_step / 探针计数器          │
│  └─ 平台耦合(要剥离): AscendAttentionBackendImpl 子类 + FIA 兜底 +     │
│                       AscendAttentionState 分流 + seq_lens_device     │
│                       ！私有补丁 + 8 个 install_* 猴补                │
└───────────────┬──────────────────────────────────────────────────────┘
                │ plan/run 契约（引擎无关）
┌───────────────▼──────────────────────────────────────────────────────┐
│ tileinfer 核心（已经是引擎无关的，迁移时基本不动）                     │
│  metadata.py  PageTable / RaggedMetadata / 设备侧 dense→ragged        │
│  plan.py      AttentionPlan / TileSchedule / LPT / WorkspaceManager   │
│  attention/   BatchAttention(plan/run) + 后端注册表                   │
│  kernels/     TileLang Ascend950: decode / split+merge / prefill      │
└──────────────────────────────────────────────────────────────────────┘
```

要点：**核心层已经干净**，脏的全部集中在 `integrations/` 的两个部分——一个 2785 行的
`vllm_ascend.py`，以及 8 个 `install_*` 补丁模块（另有 2 个补丁落在 `kernels/gdn/small_ops.py`
与 `kernels/moe/prefetch.py` 里）。迁移是"拆适配层"，不是重写内核。

### 各模块对 `vllm_ascend` 的依赖

| 模块 | 行数级 | 依赖 `vllm_ascend` | 纯 vLLM 下的处置 |
|---|---|---|---|
| `metadata.py` / `plan.py` / `attention/*` / `kernels/attention/*` | — | ❌ 无 | 不动 |
| `integrations/vllm_ascend.py` 的会话语义段（~1400 行） | 大 | ⚠️ 只经由 `AscendMetadata` 字段 | **搬到 `tileinfer/serving/`，参数化** |
| 同上，`_build_classes` / shim / prefill 兜底（~300 行） | 中 | ✅ 强耦合 | **按 vLLM 契约重写** |
| `prefill_step` + `PrefillShapeManager`（~350 行） | 中 | ⚠️ 弱耦合 | 搬到 `serving/`，兜底改成原生算子 |
| `prewarm.py` | 300 | ❌ 无（`next_pow2`/`bucket_for` 是副本，靠 `test_prewarm_specs.py` 防漂移） | 不动，补一个"从 vLLM config 派生"的入口 |
| `step_diag.py` / `prep_fastpath.py` / `gdn_meta_memo.py` | 中 | ✅ 强耦合 | B 口径删除；A 口径保留但降级为 `vllm_ascend` 存在时启用 |
| `dense_nn.py` / `shared_gate_fuse.py` / `shared_expert_fold.py` / `moe_prenz.py` / `fused_sample.py` | 中 | ✅ 强耦合 | 同上 |
| `kernels/gdn/*` / `kernels/moe/prefetch.py` | 中 | ✅ 强耦合 | 同上 |

---

## 2. 耦合点清单（逐条对照）

| # | 现在依赖的东西 | 位置 | 纯 vLLM 下从哪来 |
|---|---|---|---|
| 1 | `AscendAttentionBackendImpl` 的 `forward()`：负责把本步 K/V 写进 paged cache | 隐式继承 | **要自己实现**（用 `slot_mapping`），见 N4 |
| 2 | `forward_impl(query, key, value, kv_cache, attn_metadata, output)` 签名与路由 | `vllm_ascend.py:1577` | vLLM 的 `AttentionImpl.forward(layer, q, k, v, kv_cache, attn_metadata, output, ...)`，见 N3 |
| 3 | `AscendAttentionMetadataBuilder` + `AscendMetadata`（`attn_state` 区分 prefill/decode） | `:1513`、`:1593` | 自己做 metadata + builder，prefill/decode 用 `max_query_len`/`query_start_loc` 判定，见 N2 |
| 4 | `attn_metadata.seq_lens_device` —— **本项目给 vllm-ascend 打的私有补丁**（D17） | `:1648`、`:1679` | `CommonAttentionMetadata.seq_lens` 本来就是 device tensor，**迁移后这一项反而消失** |
| 5 | `NPUPlatform.get_attn_backend_cls` 猴补（`--attention-backend` 被忽略） | `:1731` | 口径 C 自己实现 Platform 就没这坑；口径 A **仍需保留这 1 个猴补** |
| 6 | `torch.npu.is_current_stream_capturing()` 判捕获态 | 多处 | 不变（torch_npu） |
| 7 | `vllm_ascend.compilation.acl_graph.ACLGraphWrapper` | `step_diag.py` | 见 N6：vLLM 的 graph wrapper 在 NPU 上是否可用要先验证 |
| 8 | 导入顺序 hack：`import vllm_ascend.ops` 必须先于 `attention_v1` | `:1489` | 消失 |
| 9 | `AscendSampler` / `AscendTopKTopPSampler` | 采样补丁 | 与 attention 无关，A 口径不动；C 口径要自己接 `vllm` 的 `Sampler` |
| 10 | GDN（`vllm_ascend.ops.gdn`、`device_op.A5DeviceAdaptor`）、MoE（`fused_moe`）、
`AscendUnquantizedLinearMethod`、`model_runner_v1`、`block_table` | 各 `install_*` | A 口径保留；C 口径是最大一座山，见 §5 |

---

## 3. 目标形态（A 口径）

```
tileinfer/
├── serving/                     # 新增：从 vllm_ascend.py 抽出的"引擎无关会话语义"
│   ├── buckets.py               #   bucket 选择 / pad_batch
│   ├── planner.py               #   plan 缓存、后台编译、prewarm、失败集合
│   ├── step.py                  #   step guard、设备侧 ragged 重建、子页视图
│   ├── prefill.py               #   prefill_step（统一 append 语义）
│   └── counters.py              #   探针计数（不依赖 vLLM）
├── integrations/
│   ├── vllm/                    # 新增：vLLM 原生适配（只 import vllm.* 与 torch/torch_npu）
│   │   ├── __init__.py          #   install()
│   │   ├── backend.py           #   TileInferAttentionBackend
│   │   ├── metadata.py          #   TileInferMetadata + builder
│   │   ├── impl.py              #   TileInferAttentionImpl（forward）
│   │   ├── kv_cache.py          #   slot_mapping → NHD 池写入（新功能 N4）
│   │   └── fallback.py          #   无 FIA 时的回退策略（N7）
│   ├── prewarm.py               # 保留（单一事实源）
│   └── vllm_ascend.py           # 降级为 vllm_ascend 专属补丁集合，A 口径下不参与主路径
└── ...
```

`pyproject.toml` 的入口点保留 `vllm.general_plugins`；`TILEINFER_VLLM=1` 保留；
新增 `TILEINFER_VLLM_NATIVE=1` 做新旧双跑 A/B，验证完再删旧路径。

---

## 4. 待实现功能清单（A 口径）

优先级：**N1–N5 是"不写就跑不起来"**，N6–N9 是"能跑但不达现有水准"，N10–N12 是工程面。

### N1 · backend 类：`TileInferAttentionBackend`

- 现状：`TileInferBackend`（`vllm_ascend.py:1497`）已实现 `get_name/get_impl_cls/get_builder_cls/
  get_kv_cache_shape/get_supported_kernel_block_sizes`，但 builder 是 vllm-ascend 的、impl 是它的子类。
- 要做：按 vLLM 0.23 的 `AttentionBackend` ABC 补齐：`validate_configuration()`（拒绝
  `head_size ∉ {128,256}`、`group < 2`、`block_size % 128 != 0`、非 NHD、非 bf16/fp16）、
  `accept_output_buffer`、`get_kv_cache_shape()`（保持 `(2, blocks, block, kv_heads, dim)`）、
  `get_supported_kernel_block_sizes()`、以及 KV spec 相关钩子。
- 验收：`AttentionBackendEnum.CUSTOM.get_class()` 解析到本类；不支持的配置在**启动期**被拒
  （而不是第一个 decode step 才回退）。

### N2 · metadata 与 builder

- 现状：完全复用 vllm-ascend 的 builder，靠 `attn_state` 三分支。
- 要做：`TileInferMetadata`（持有 `block_table_tensor`、`seq_lens`(device)、`query_start_loc`、
  `num_reqs`、`num_actual_tokens`、`max_query_len`、`slot_mapping`、`causal`），
  `build(common_prefix_len, common_attn_metadata, fast_build=False)` 里只做**引用 + 形状推导**，
  不做拷贝；声明 graph 复用能力（`_cudagraph_support` / `AttentionCGSupport`）。
  prefill/decode 的判定改为 `max_query_len > 1`（不再依赖 `AscendAttentionState`）。
- 验收：CPU 单测覆盖"同一 builder 连续 build 10 步，block_table 原地更新后 run 结果随动"；
  捕获态下 builder 不分配、不同步。

### N3 · impl 与 forward 签名

- 现状：`forward_impl` 是 vllm-ascend 内部的单点分派，签名是它的。
- 要做：实现 `AttentionImpl.__init__(num_heads, head_size, scale, num_kv_heads,
  alibi_slopes, sliding_window, kv_cache_dtype, ...)` 与
  `forward(layer, query, key, value, kv_cache, attn_metadata, output, output_scale=None,
  output_block_scale=None)`；处理 `layer` 上的 scale（`_q_scale/_k_scale/_v_scale/_o_scale`）
  或显式拒绝量化；布局 `[num_tokens, num_heads, dim]`；返回或写入 `output`。
- 验收：Qwen2.5-1.5B 单步 top-5 logprobs 与现路径逐位一致（沿用现有 A/B 脚本）。

### N4 · KV cache 写入（**现在完全由 vllm-ascend 代劳，必须新写**）

- 现状：TileInfer 只读池——`forward_impl` 进来时 K/V 已经被父类的 `forward()` 用
  `reshape_and_cache` 写好了。
- 要做：`write_kv_cache(key, value, k_cache, v_cache, slot_mapping)`：按 `slot_mapping`
  把本步 K/V 散射进 `[blocks, block_size, kv_heads, dim]`；decode（每请求 1 token）与
  chunked prefill（每请求多 token）共用；子页视图（`block_size = k*128`）下 `slot_mapping`
  是引擎页内偏移，需要与 N5 的页视图一致。
- 验收：与 `torch_npu` 原生 `reshape_and_cache` 逐元素对齐（`torch.equal`），
  并覆盖 `block_size=128/2048`、partial page、chunked prefill。

### N5 · block size / KV spec 对齐

- 现状：靠 `_sub_pages` / `_expand_page_table` / `_pool_sub_view` 把 `block_size = k*128`
  的引擎页拆成 128-token 内核页（`vllm_ascend.py:120/136`），每步 `repeat_interleave` 展开表。
- 要做：把这套逻辑从"引擎恰好允许"升级为"契约"：`get_supported_kernel_block_sizes()` 与
  `get_kv_cache_shape()` 一致声明；hybrid 模型（GDN）的 mamba page 与 attention page 对齐；
  展开表改成**固定形状、原地**的写法（`block_size > 128` 时当前 `repeat_interleave`
  每步分配一次，且发生在捕获区内，与"run 零分配"的契约冲突）。
- 验收：`block_size ∈ {128, 256, 2048}` × decode/prefill 设备测试全绿；捕获区内零分配
  （用 `torch.npu.memory` 或断言钩子）。

### N6 · graph capture 契约

- 现状：依赖 vllm-ascend 的 `ACLGraphWrapper`；TileInfer 侧的约束是"plan 必须在捕获前
  ready、捕获态只查缓存不编译"（`TileInferDecodeAttention.run`，`:1364`）。
- 要做：确认 vLLM 0.23 的 graph wrapper 在 NPU 上走哪条路（是否可用 `torch.npu.graph`）；
  不可用则自己包一层 `TileInferGraphWrapper`；把"捕获态禁止 compile/alloc/sync"变成
  显式断言 + 探针（现有 `capture_calls`/`fallbacks` 计数器可复用）。
- 验收：`capture_calls > 0 && fallbacks == 0`，replay 期无 Python 路径（现有探针口径）。

### N7 · 回退策略（**去掉 vllm-ascend 引用后就没有 FIA 了**）

- 现状：所有拒绝路径都是 `super().forward_impl(...)` → FIA 算子。
- 要做：分三级——
  1. 形状不支持（head_size/group/block_size）→ **启动期拒绝**，让 vLLM 选别的 backend；
  2. plan 未就绪（首次形状）→ 延迟：先跑一次通用路径（`torch_npu` 原生 paged attention，
     不经过 vllm_ascend 的类）或 `reference` 后端，同时后台编译；
  3. 运行期异常 → 熔断到通用路径 + 计数，进程存活。
- 验收：拔掉 `vllm_ascend` 的 attention 模块后仍能出字（哪怕慢），日志有明确归因。

### N8 · prefill 归属（**本次迁移最硬的一条**）

- 事实：`_PREFILL_MAX_CTX` 默认 **1**（`vllm_ascend.py:327`），即当前默认**完全把 prefill
  交还 FIA**；依据是 H61 终审（TI 自研 prefill 在全部实测场景都慢于 FIA，TTFT −1.2%~−71.3%）。
- 含义：一旦移除 vllm-ascend，**prefill 没有兜底**，必须由 TileInfer 自己扛。
- 要做（三选一，需决策）：
  - (a) 先接受短上下文回归，只保证正确性，长上下文限流；
  - (b) 优化 prefill 内核（`docs/next-tasks.md` T4/T5 方向：M tile 全用、SIMD softmax、
        分页流水），目标 ≤1.0x FIA；
  - (c) 保留一条原生算子 prefill 路径（`torch_npu` FIA）作为显式降级，承认它仍是"别人的算子"。
- 验收：`pytest tests/test_tilelang_ascend950_prefill*.py` + serve 下 TTFT 对比表落盘。

### N9 · 混合模型 / 非 attention 层的归属

- 事实：目标模型 Qwen3.6-35B-A3B 是 40 层里 30 层 GDN + 10 层 full-attention。
- 要做：`AttentionBackend` 只在 MHA/GQA 层生效；GDN 层需要 `MambaAttentionBackend`
  或平台自带实现。口径 A 下这部分仍由 vllm-ascend 提供，**必须在选择逻辑里排除**
  （现 shim 里的 `use_mla/use_sparse/use_compress` 排除逻辑要扩展到线性注意力（GDN/mamba）
  层的判定，具体字段名以本版本 `AttentionSelectorConfig` 为准，不要凭记忆写）。
- 验收：Qwen3.6 图模式启动后，10 个 attention 层走 TileInfer，30 个 GDN 层走原路径，
  探针计数可区分。

### N10 · 注册与选择入口

- 现状：`vllm.general_plugins` 注册 + 猴补 `get_attn_backend_cls`。
- 要做：A 口径保留猴补（1 处，注释写清上游行为）；同时支持 `--attention-backend CUSTOM`
  与显式 `TILEINFER`；把 `install()` 拆成 `install_backend()` / `install_patches()` 两部分，
  后者在 B 口径下直接不调用。
- 验收：`list_backends()` 在 EngineCore 内可见（KI-2 已修，回归一次即可）。

### N11 · prewarm / bucket 与配置派生

- 现状：`integrations/prewarm.py` 已是"要编译什么"的单一事实源，但入口是脚本 + 手填
  `TILEINFER_PREWARM_PAGES`。
- 要做：从 `VllmConfig`（`max_num_seqs`、`max_num_batched_tokens`、`max_model_len`、
  `block_size`、KV 池页数）直接派生，init 期在 warmup 里同步预烧；`plan_if_ready` 的
  "后台编译 + 降级"从主路径降为兜底。
- 验收：`TILEINFER_METRICS_PATH` 里 `ready_plans == 预期桶数`，且 serve 首步无编译日志。

### N12 · 观测与回归

- 要做：保留现有探针 JSON（`capture_calls/eager_calls/fallbacks/refreshes/ready_plans`），
  新增 `backend=native|legacy` 字段；补一套"新旧双跑"的 CPU 契约测试 + 一份 serve 级 A/B 脚本。
- 验收：同一 prompt 在两条路径下 logprobs 一致；性能差 ≤1%。

---

## 5. 只有 C 口径（彻底不装 vllm-ascend）才要做

这一节是"如果真要连平台一起自持"，列全是为了让决策有依据——不是建议做。

| # | 模块 | 内容 | 参考体量 |
|---|---|---|---|
| C1 | `vllm.platform_plugins` 平台插件 | `Platform` 子类：device 探测/名称/显存、worker 选择（`parallel_config.worker_cls`，`parallel.py:248`；0.23 无 `Platform.get_worker_cls`）、`get_attn_backend_cls`、HCCL、dtype、pin memory、async output、`support_*` 能力位 | ~1.5k 行 |
| C2 | Worker / ModelRunner | input batch、block table、`slot_mapping` 生成、memory profiling、piecewise + full graph 捕获、多卡并行 | ~3–5k 行 |
| C3 | KV 池与 spec | hybrid（mamba/attention 双页）、`KVCacheSpec` 构造、page 分配器对接 | ~1k 行 |
| C4 | 算子面 | RMSNorm / fused_add_rmsnorm / rotary(+mrope) / silu_and_mul / W8A8 量化线性 / fused_experts(MoE) / GDN（causal_conv1d、chunk_gated_delta_rule、gated_delta_rule）/ sampler / top-k-top-p | ~5k 行 + 大量调优 |
| C5 | 模型覆盖 | Qwen2/2.5/3、Qwen3.6(GDN hybrid)、DeepSeek(MLA)、量化模型… 每个都要验证 | 持续 |
| C6 | 打包与版本矩阵 | 入口点、`vllm` 版本区间、CANN 版本矩阵、与上游 API 漂移跟踪 | 持续 |

结论：C4 + C2 两项就超过 TileInfer 现有全部代码量，且**没有任何性能收益**——它只是把
vllm-ascend 的活重做一遍。除非目标是"完全自持昇腾栈"，否则不建议。

---

## 6. 风险与必须先拍板的 5 件事

1. **口径**：A / B / C。建议 A，并把 B 作为 A 之后的清理动作。
2. **prefill 谁扛**（N8）。这是移除 vllm-ascend 后唯一**能力回退**的地方，不是工作量问题。
3. **是否接受"仍要装 vllm-ascend"**。A 口径下它还在，只是不再被 patch；对外说法要一致。
4. **split-KV 是否随之默认化**。KI-1 根因已定位并修复（`known-issues.md`，2026-10-14），
   但 `docs/status.md` / `next-tasks.md` 仍写"open / 延后"——**文档漂移**，
   迁移前先对齐，否则会按错误前提排期。
5. **私有补丁清理**：`seq_lens_device` 是对 vllm-ascend 的改动（D17）。A 口径下它变成
   "不需要"，要确保新路径不引用；否则会逼着继续维护那个补丁。

---

## 7. 建议里程碑

| 阶段 | 内容 | 出口标准 |
|---|---|---|
| M1（周 1） | 抽 `tileinfer/serving/`，参数化掉 `AscendMetadata`；旧集成改为调用它，行为不变 | CPU 套件全绿；serve 数字与迁移前一致 |
| M2（周 1–2） | N1–N5：backend/builder/impl/KV 写入/页视图，`TILEINFER_VLLM_NATIVE=1` 双跑 | 同 prompt logprobs 一致；capture 探针 PASS |
| M3（周 2） | N6/N7/N9/N10：捕获契约、回退策略、层选择、入口拆分 | Qwen2.5 与 Qwen3.6 图模式在役，fallbacks=0 |
| M4（周 3） | N8/N11/N12：prefill 决策落地、prewarm 自派生、双跑 A/B | 性能表落盘，短上下文 ≤1.0x，长上下文不劣化 |
| M5 | B 口径清理：删/降级 `install_*` 补丁 | `grep -r vllm_ascend tileinfer/` 只剩回退与平台探测 |

---

## 8. 迁移前要用 grep 核对的 vLLM API（0.23）

这些是按记忆写的契约，落地前在目标版本上逐个确认（版本一旦漂移，N1/N3/N6 都要改）：

```text
vllm/v1/attention/backend.py            AttentionBackend 的抽象方法全集、accept_output_buffer
                                        AttentionImpl.forward 的完整签名
                                        CommonAttentionMetadata 的字段
                                        AttentionCGSupport / _cudagraph_support
vllm/v1/attention/backends/registry.py  AttentionBackendEnum、register_backend
vllm/v1/attention/backends/utils.py     PAD_SLOT_ID、split_attn_metadata 等工具
vllm/platforms/__init__.py              platform_plugins 入口协议(register() 返回值)
vllm/config/vllm.py                     VllmConfig 里 KV 池页数 / block_size 的可得性
vllm/v1/worker/gpu/*                    graph wrapper 是否只用 torch.cuda(→ NPU 上是否可用)
vllm/v1/attention/ops/*                 slot_mapping 的语义与 dtype 约定
```
