# TileInfer 设计文档

**版本 v0.1（草案，已按实机调研更新）** · 状态：Phase 1 进行中

## 1. 项目定位

**TileInfer** 是面向 LLM Serving 的、**引擎无关**的高性能内核库，优先支持华为昇腾（从
Ascend 950 起步）。定位对标 FlashInfer：

- 不绑定任何推理引擎（vLLM-Ascend、MindIE、自研引擎均可接入）；
- 提供统一的 Attention / GEMM / MoE / Sampling 算子族；
- 针对 Serving 场景优化：Paged/Ragged KV-Cache、动态 batch、负载均衡、图模式兼容；
- 以 **TileLang 为主力开发语言**，关键路径可下沉到 Ascend C / PTO。

## 2. 背景与机会

昇腾侧缺少 FlashInfer 这类独立内核层：MindIE 依赖 ATB 并深度绑定官方栈；vLLM-Ascend 主要走
CANN FIA 加少量内部自定义 kernel，内核层不可独立复用。官方栈在标准路径上已经很强，但在
**动态 batch 负载均衡、自定义 KV 布局、JIT 变体、与开源引擎解耦** 等方面仍有明显空白。

实机调研（2026-10，Ascend950PR / CANN 9.1.1 / driver 25.7.rc1）确认了这些空白确实存在：

- `vllm_ascend 0.23.0`（950 专用 wheel）自带的 attention backend 只有
  `attention_v1`（FIA）、`fa3_v1`（外部 `flash_attn_npu_v3`）、`mla_v1`、`dsa_v1`、`sfa_v1`，
  没有任何可独立复用的通用内核层；
- PyPI 上的 `tilelang 0.1.8` **没有 Ascend 后端**（`SUPPORTED_TARGETS` 只有
  cuda/hip/metal/llvm/webgpu/c/cutedsl）；昇腾支持在 **`tile-ai/tilelang-ascend`**，含两条后端路线
  （`ascendc_pto` 默认分支 / `npuir` MLIR 分支），需从 GitHub release 安装 wheel；
- 公开的 TileLang FA 实现（如 `platelett/fa`）在 Atlas A3 上达到 Cube 峰值的 ~50%，
  手写 CCE 可到 ~97% —— 说明**「用 TileLang 写好昇腾 attention」本身仍有大量工程空间**，
  这正是 TileInfer 要做的事（而不是重复造框架）。

## 3. 设计目标

| 目标 | 说明 | 优先级 | 当前状态 |
|------|------|--------|----------|
| 引擎无关 | 通过标准元数据（indptr/indices/page table）接入，不依赖引擎内部结构 | P0 | ✅ 已实现 |
| Serving 友好 | Paged/Ragged、prefill/decode/append、动态 batch | P0 | 🟡 decode 内核完成，prefill 待做 |
| 图模式兼容 | plan/run 分离，run 阶段零分配、零编译、零同步 | P0 | ✅ 契约已实现 |
| 负载均衡 | plan 阶段按 KV 长度切分与 LPT 调度 | P0 | ✅ 调度器 + 参考实现完成，内核侧待接 |
| 高性能 | 目标场景接近或超过 FIA | P0 | ⏳ 待实测 |
| 易开发 | 主力 TileLang，降低手写 Ascend C 成本 | P0 | ✅ 已跑通工具链 |
| 可扩展 | MLA、Cascade、Sparse、低精度、MoE | P1 | ⏳ Phase 3 |

## 4. 总体架构

```
┌─────────────────────────────────────────┐
│           推理引擎层                     │
│  vLLM-Ascend / MindIE / 自研引擎        │
└─────────────────┬───────────────────────┘
                  │ 标准元数据（indptr/indices/page table）+ plan/run
┌─────────────────▼───────────────────────┐
│           TileInfer 接口层              │
│  BatchAttention · plan/run 分离         │
│  元数据转换 · Workspace 管理 · 后端注册  │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│           Kernel 实现层                 │
│  ┌─────────────┐  ┌─────────────────┐   │
│  │ TileLang    │  │ Ascend C / PTO  │   │
│  │ (主力)      │  │ (关键路径)      │   │
│  └─────────────┘  └─────────────────┘   │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│         Ascend 950 硬件 + CANN          │
└─────────────────────────────────────────┘
```

分层细节见 [`architecture.md`](architecture.md)。

## 5. 核心模块

### 5.1 Attention（第一优先级）

- 支持模式：Prefill / Decode / Append（chunked prefill、speculative step）；
- 支持布局：Paged KV-Cache（page table）、Ragged 变长、GQA/MQA（含连续分组约定）；
- 关键优化：
  - **plan 阶段处理变长，run 阶段形状全静态**（图捕获友好）；
  - **LPT 负载均衡**：按 KV 长度切片 + 最长作业优先派发，缓解「一条 100k 上下文 + 一批短请求」
    造成的 core 空转；
  - 后续：MLA、Cascade（共享前缀）、Sparse、FP8/FP4、attention sink。

### 5.2 接口

```python
attn = BatchAttention(backend="tilelang", device="npu", dtype=torch.float16)

plan = attn.plan(                       # 每个 batch 形状一次（捕获前）
    kv_indptr=kv_indptr,                # [B+1] int32
    kv_indices=kv_indices,              # [num_pages] int32
    kv_last_page_len=kv_last_page_len,  # [B] int32
    page_size=128,
    num_qo_heads=32, num_kv_heads=8, head_dim=128,
    kv_tile_pages=0,                    # >0 时按 KV 切片做负载均衡
)

out = attn.run(q, k_cache, v_cache, plan=plan)   # 纯计算，可被 ACLGraph 捕获
```

引擎若只有稠密 `block_tables`，用 `plan_from_page_table(block_table, seq_lens, ...)` 一行接入
（稠密表在 plan 阶段被压缩成 rag 形式，padding 不再进入 kernel）。

### 5.3 元数据与调度

- 统一输入：`qo_indptr` / `kv_indptr` / `kv_indices` / `kv_last_page_len` / page table；
- plan 阶段职责：模式判定 → 工作切分（切 KV / 切 query）→ LPT 排序 → workspace 分配 →
  kernel 变体选择与 JIT（**只在这里编译**）；
- run 阶段职责：把变长信息已经「静态化」的形状喂给 kernel，零分配、零编译、零 host 同步。

## 6. 与 vLLM-Ascend 集成

与现有 `fa3_v1.py` 同构，新增 `vllm_ascend/attention/tileinfer_v1.py`：

1. `get_name()` → `"TILEINFER"`；`get_kv_cache_shape()` 返回 **NHD**（与 FIA 路径一致，切后端不重分 KV）；
2. Metadata Builder：`block_tables` / `seq_lens` / `query_start_loc` →
   `plan_from_page_table(...)`；
3. Forward：调用 `run`，plan 在捕获前构建完毕；
4. 注册到平台选择逻辑，支持 `--attention-backend TILEINFER`。

可运行草图见 [`../examples/vllm_ascend_tileinfer_backend.py`](../examples/vllm_ascend_tileinfer_backend.py)。

## 7. 路线图（含实机状态）

| 阶段 | 内容 | 产出 | 状态 |
|------|------|------|------|
| Phase 0 | 环境与基线 | TileLang-Ascend 环境、metadata/plan/workspace、参考实现、micro-bench | ✅ 完成 |
| Phase 1 | 最小 Attention | **Paged Decode（GQA）TileLang 内核**、独立 benchmark、与参考/FIA 对齐；Prefill/Append 内核 | 🟡 decode 完成，prefill 进行中 |
| Phase 2 | 引擎接入 | vLLM-Ascend 可选 backend，端到端跑通 Qwen/Llama | ⏳ |
| Phase 3 | 优化与扩展 | Split-KV 归并、LPT 落到内核、MLA、低精度、性能调优 | ⏳ |
| Phase 4 | 生态化 | 文档、示例、其他引擎适配、上游 PR | ⏳ |

详细的验收标准见 [`roadmap.md`](roadmap.md)。

## 8. 性能与验证目标

- Micro-benchmark：固定 (batch, kv_len, heads, dim) 网格下与 FIA 对比（`benchmarks/bench_attention.py
  --sweep --fia`），同时报告与 torch 参考的 max abs error，避免「快但错」；
- 端到端：TTFT / TPOT / 吞吐（vLLM-Ascend 中，Phase 2 之后）；
- 目标：标准场景不落后于 FIA；**dynamic batch / 长上下文 / 极偏斜 batch 上要有明确优势**；
- 正确性：与 torch 参考实现对齐（fp16 下 rtol/atol ~1e-2），并与 FIA 输出对齐。

## 9. 风险与应对

| 风险 | 影响 | 应对 |
|------|------|------|
| TileLang-Ascend 后端成熟度 | 部分算子性能/功能不足 | 关键路径下沉 Ascend C / PTO；紧跟上游 PR |
| 硬件单卡且被占用 | 调试窗口有限 | 小 shape 快速验证；benchmark 与业务错峰；逻辑先在参考实现里验证 |
| 元数据对接复杂 | 集成周期长 | 先最小可用集（decode），再逐步对齐 FlashInfer 习惯 |
| 官方栈演进 | 被官方能力追上 | 聚焦 serving 特有场景（动态负载、自定义布局、引擎解耦） |
| 生态碎片化 | 维护成本高 | 严格引擎无关；接口与文档优先 |

## 10. 命名与开源

- 名称：**TileInfer**；License：Apache 2.0；
- 仓库结构：`tileinfer/`（Python 接口层）、`tileinfer/kernels/`（TileLang kernel 源码，
  放进包内以保证可安装、可导入；未来 Ascend C / PTO 源码另开 `csrc/`）、
  `examples/`、`benchmarks/`、`tests/`、`docs/`；
- 初期作为独立库维护，同时提供 vLLM-Ascend 集成示例 PR。
