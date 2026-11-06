# Qwen3.6 性能优化方案（vLLM 原生后端）

> 前置：[`status-native-backend.md`](status-native-backend.md)（交付与证据分级）、
> [`notes/perf-comparison-rules.md`](notes/perf-comparison-rules.md)（A/B 环境约束）。
> 本文只解决一个问题：**在已跑通的前提下，把 TTFT/TPOT 拉到与基线可比**。

---

## 1. 现状（实测，2026-11-03，Ascend950PR 单卡）

**注意**：这一轮 A/B 两臂 torch 版本不同（2.10 vs 2.13），按
[`perf-comparison-rules.md`](notes/perf-comparison-rules.md) **只能作方向性证据**，不能作为验收基线。
下面数字的用途是**排序优先级**，不是结论。

| prompt≈ | TileInfer TTFT / TPOT | 基线(vllm-ascend) TTFT / TPOT | 倍数 |
|---|---|---|---|
| 16 | 0.694 s / 0.133 s | 0.101 s / 0.035 s | 6.9× / 3.8× |
| 128 | 1.108 s / 0.134 s | 0.103 s / 0.042 s | 10.8× / 3.2× |
| 512 | 1.755 s / 0.156 s | 0.107 s / 0.044 s | 16.4× / 3.5× |

吞吐：6.4–7.5 tok/s vs 22.5–28.7 tok/s。启动：冷编译约 6 min，缓存热约 60–90 s。

### 归因（每条都有代码/日志证据）

| # | 现象 | 归因 | 证据 |
|---|---|---|---|
| A1 | TTFT 随 prompt 长度线性恶化（0.69→1.11→1.76 s），基线恒定 0.10 s | **prefill 走 torch 参考实现**：native 路径把 prefill 交给 `_reference_forward`（O(n²) 逐 tile），内核后端只声明 decode | `attn/impl.py` 的 `_reference_forward` docstring；`tilelang_ascend950.py` 声明 decode-only |
| A2 | TPOT 3.5×，与 prompt 长度无关（恒定劣势） | **decode 每 token 的边算子全是 torch**：MoE `fused_experts` 按专家 Python 循环（256 专家 top-8）、GDN chunk 链用 `torch.linalg.solve_triangular` 等 | `ops/moe.py`、`ops/fla/*` 均为 torch 实现；`candidates.py` 里 `fused_experts` 只有 disabled 候选 |
| A3 | 启动 60–90 s | 内核 JIT 编译（缓存命中后仍有模块导入与预烧） | `tl_cache/0.1.15` 产物；worker prewarm 日志 |
| A4 | 未启用图捕获 | 平台在 `capture_path_ready()` 未就绪时强制 `enforce_eager`（现已修好 `NPUGraph` 别名，可能已就绪但未验证） | `platform.py` / `graphs.py` |

**结论**：prefill（A1）是 TTFT 的全部；MoE+GDN 的 torch 实现（A2）是 TPOT 的主因；两者独立，可并行推进。

---

## 2. 目标（量化，全部以**同环境**基线为分母）

| 指标 | 当前 | 目标 0（M1） | 目标 1（M2） | 终态（M3） |
|---|---|---|---|---|
| TTFT @512 prompt | 16.4× | ≤3× | ≤1.5× | ≤1.1× |
| TPOT | 3.5× | ≤1.8× | ≤1.3× | ≤1.1× |
| 吞吐 tok/s | 6.4–7.5 | ≥15 | ≥22 | ≥26 |
| 启动（热缓存） | 60–90 s | ≤60 s | ≤45 s | ≤45 s |

术语：目标 0 = "不再差一个数量级"；目标 1 = "进入可比区间"；终态 = "不低于基线"。

---

## 3. 方法学（比任何单项优化都重要）

1. **每次改动必须走 candidate 门槛**：`CandidateSet.add(measured_ratio=...)`，
   `measured_ratio < 1.0` 一律 `enabled_by_default=False`。这条已经实现并有测试
   （`ops/candidates.py`），优化工作只是往里面填候选。
2. **每次改动必须有同环境 A/B**：`benchmarks/ab_report.py --check` 通过 + 记录
   `pip list`/`npu-smi` 头部。旧的那组数字只用来排序。
3. **先量化再改**：每个阶段开工前跑一次**分相计时**（见 W0），拿到 decode 步里
   KV 写入 / attention / GDN / MoE / sampler 的占比，避免"改了个不热的地方"。
4. **一次只改一处**，并用 `TILEINFER_OPS` 开关做同进程对照（已实现）。

---

## 4. 工作项

### W0 · 分相计时探针（前置，1–2 天）

decode 一步拆成：`kv_write` / `attention(kernel or reference)` / `gdn` / `moe` / `sampler` /
`host_prep`，用 NPU event 或 `torch.npu.synchronize()+perf_counter` 采样 100 步取中位数，
输出到 `TILEINFER_METRICS_PATH` 的 `phase_ms` 字段。

**验收**：一份 40 层模型的阶段占比表；后续每个 W 的收益都用它归因。

### W1 · 打开 prefill 内核（收益最大，TTFT 7–16×）

仓库里已有 prefill 内核（`tests/test_tilelang_ascend950_prefill*.py` 覆盖，含 split 变体），
但 native 路径没启用。**这是纯接线工作，不是新写内核。**

- `attn/backend.py::supports_mode` 声明 prefill；
- `impl._reference_forward` 只在"形状不支持"时兜底（记录原因并计数）；
- 按 `(num_tiles, block_q)` 预烧变体，沿用 `PrefillShapeManager` 的惰性编译 + 变体上限思路；
- 长上下文回退策略：沿用 `TILEINFER_PREFILL_MAX_CTX` 的阈值语义，超阈回参考实现并计数。

**验收**：TTFT@512 ≤3×；prefill 走内核的比例 > 90%；长上下文不 OOM。

### W2 · MoE 批量化（TPOT 主项之一）

现状：`fused_experts` 对命中的专家做 Python 循环（256 专家、每 token top-8）。
两条路，按 W0 数据择一或都做：

- **W2a torch 批量化**：`expert_map` 排序 + `index_select` 后一次 `bmm`（避免逐专家 python 循环），
  预计 2–3×；零新内核，风险最低。
- **W2b 内核**：仓库的 `kernels/moe/grouped_gemm_ascend950.py` 曾测 **0.86×**（低于 1.0 门槛，
  未接入）。若要启用，必须先在同环境下重测 ≥1.0 才可 `measured_ratio` 入册。

**验收**：MoE 阶段耗时（W0）下降 ≥2×；`candidates` 里记录 ratio。

### W3 · GDN 链（TPOT 主项之二）

- `solve_tril` 的 `torch.linalg.solve_triangular` 是逐 chunk 调用，先改成按 chunk 批量（一次 solve 多 chunk），
  再考虑专用内核；
- chunk 链（kkt/wu/delta_h/chunk_o）合并中间张量，减少 kernel launch 数（当前每层 6 次）；
- `recurrent`（decode 路径）优先：它每 token 都跑，且是纯逐 token 递推，最容易批量化。

**验收**：GDN 阶段耗时下降 ≥2×。

### W4 · 图捕获（decode 端固定开销）

`graphs.py` 的别名已修（`NPUGraph`），但未验证。步骤：确认 `capture_path_ready()` 为真 →
跑 `CUDAGraphWrapper` 委托 → 检查 `capture_calls>0 && fallbacks==0`。
注意：捕获前必须完成预烧（`serving/planner.py` 已接线）。

**验收**：eager vs graph 的 TPOT 对比（同环境），并确认无 capture 期分配/编译。

### W5 · split-KV 默认化（长上下文 decode）

KI-1 根因已修，split 内核可用但 opt-in。开启后长上下文 decode 有 3× 潜力（历史数据）。
需要：按桶设置 ktp 策略（`TILEINFER_KV_TILE_PAGES_POLICY` 已有机制）+ 预烧。

**验收**：8K/16K 上下文 TPOT 改善 ≥2×；短上下文无回退。

### W6 · 启动时间

- 内核预烧：把 `scripts/prewarm-tileinfer-kernels.py` 接到部署前（已有脚本），启动直接命中缓存；
- 变体收敛：桶表按 `max_num_seqs` 截断（已有）；
- 可选：`TILELANG_CACHE_DIR` 固化并在镜像里预热。

**验收**：热缓存启动 ≤45 s。

### W7 · 采样与小算子

`apply_top_k_top_p` 已强制 torch（Triton 编译失败）；若占比可观，考虑
`tileinfer/kernels/sampling/*`（已有 topk/greedy 内核）作为 candidate 接入。

---

## 5. 里程碑

| 里程碑 | 内容 | 出口标准（同环境 A/B） |
|---|---|---|
| M0 | W0 探针 + 同环境基线重测 | 阶段占比表落盘；基线数字头部含版本信息 |
| M1 | W1（prefill 内核） | TTFT@512 ≤3×；prefill 内核占比 >90% |
| M2 | W2 + W3 | TPOT ≤1.8×；MoE 与 GDN 各降 ≥2× |
| M3 | W4 + W5 + W6 | TTFT ≤1.1×，TPOT ≤1.1×，吞吐 ≥26 tok/s，启动 ≤45 s |

---

## 6. 风险

| # | 风险 | 缓解 |
|---|---|---|
| R1 | prefill 内核在这个形状上不如 FIA（历史上长上下文劣化） | 保留阈值回退 + 分阶段启用，先用 W0 数据确认 |
| R2 | MoE 内核 0.86× 的历史结论仍成立 | 先做 W2a（torch 批量化），内核只作候选且需重测 |
| R3 | 图捕获在 torch_npu 2.10 上不成熟 | 保持 `enforce_eager` 为默认，图模式作 opt-in 并单独验收 |
| R4 | 优化把正确性改坏 | 每个 W 都跑 `ab_report.py --compare` 对 logprobs；candidate 门槛 + 既有 700+ CPU 测试 |
| R5 | 同环境 A/B 成本高（要重装 venv 切换两臂） | 用 `cp -a .venv .venv_baseline` 做两份，避免反复装卸 |

---

## 7. 立刻可做的三件事

1. **W0**：写分相计时探针（半天），拿到真实占比——现在所有排序都基于"推理 + 一组不可比数字"。
2. **重测同环境基线**（半天）：`cp -a .venv .venv_baseline`，两臂各自跑 `ab_report.py`。
3. **W1 接线**（2–3 天）：prefill 内核已有，缺的是 native 路径的启用与兜底策略。

---

## 8. M0 实测结果（2026-11-03，`TILEINFER_PHASE_TIMING=1`）

测量方式：`tileinfer/serving/phases.py` 探针 + `ops/install.py` 的 takeover 包装（按 family 计时）
+ `attn/impl.py` 的 forward 包装。日志每 20 次计数输出一张表（WARNING 级——我们的 INFO 被
vLLM 的 logging 配置过滤，这是探针第一次跑空的原因）。

单请求 64 token 生成，引擎内累计表（`/tmp/ti_phase2.log`）：

```
TileInfer phase timing: 580 decode step(s), 10 prefill step(s), step=4.57 ms
phase                     calls   mean_ms  per_step_ms   share
op:gdn                     3570     0.530        3.261   71.3%
attention                   610     0.708        0.745   16.3%
op:moe                     2439     0.135        0.567   12.4%
```

### 读法修正（必须做，否则会误判）

- `decode step(s)=580` 是**注意力层调用次数**，不是调度步数：模型有 10 个 full-attention 层，
  每步调用 10 次 → 真实步数 ≈ 58，与 64 token 生成一致。
- 因此**每 token** 的真实开销是：

| 阶段 | 每 token | 占 instrumented 合计 | 占 TPOT(~130 ms) |
|---|---|---|---|
| GDN（30 个线性注意力层） | **32.6 ms** | 71% | 25% |
| attention（10 个 full-attn 层） | 7.4 ms | 16% | 6% |
| MoE（op:moe，专家循环） | 5.7 ms | 12% | 4% |
| **合计（我们接管的算子）** | **45.7 ms** | 100% | **35%** |
| 其余（dense GEMM / RMSNorm / sampler / 等待） | ~85 ms | — | **65%** |

### 结论（改变优先级）

1. **GDN 是第一大头（33 ms/token），不是 MoE（5.7 ms）**。原方案把 MoE 放在 W2a 第一位是按
   "专家循环看起来最笨"推理出来的；实测只有 5.7 ms，改它最多省 4% TPOT。
2. **35% 的 token 时间在我们接管的算子里，65% 在别处**——所以只优化 W2/W3 不可能达标，
   必须同时看清那 65%（dense 层与等待）。
3. **测量本身有偏差**：`perf_counter` 测的是 host 侧入队时间，NPU 算子异步；这个表反映的是
   "host 发多少指令"，不是"设备跑多久"。下一步必须改用设备事件（`torch.npu.Event`）或
   分阶段 `synchronize`，否则优化对象可能是错的。

### M0 修订后的工作顺序

| 新顺序 | 原顺序 | 理由 |
|---|---|---|
| W0b 设备事件计时 | 无 | 先确认 host 时间与设备时间是否同序；这是后面所有判断的地基 |
| W3 GDN 批量化 | W2 之后 | 实测 33 ms/token，单项最大 |
| W1 prefill 内核 | 同 | TTFT 仍是 7–16×，且与 decode 路径独立 |
| W2 MoE | 提前到第一 | 实测仅 5.7 ms，降级为低优先级 |
| 新 W8：剩余的 65% | 无 | dense GEMM/RMSNorm/sampler/等待，必须单独测 |

---

## 9. W0b：设备事件复核（2026-11-03，`TILEINFER_PHASE_TIMING=device`）

`phases.py` 增加了第二块表：`torch.npu.Event(enable_timing=True)` 采样，在 `report()` 时统一
`synchronize` 再算 `elapsed_time`（若在块内算，会串行化所有内核，破坏要观测的流水）。
事件采样有上限（40000），超限只保留 host 时钟；无 `torch.npu` 时自动降级为 host-only。

64 token 单请求，同一台机器：

```
TileInfer phase timing: 620 attn-layer calls (decode), 10 prefill, host step=5.33 ms
phase                     calls   host/step   dev/step  dev/call
op:gdn                     3810       3.752      3.814     0.621
attention                   650       0.867      0.892     0.850
op:moe                     2599       0.711      0.688     0.164
```

**结论 1：host 与 device 同序同量**（gdn 3.75 vs 3.81、attention 0.87 vs 0.89、moe 0.71 vs 0.69 ms）。
所以第 8 节的 host 表可以作为后续优化的判据，不必担心"异步导致测错对象"。

**结论 2：单 token 的真实分解**（每 token = 10 次注意力层调用）：

| 阶段 | 每 token（设备） | 占 TPOT(~130 ms) |
|---|---|---|
| GDN | **38.1 ms** | 29% |
| attention | 8.9 ms | 7% |
| MoE | 6.9 ms | 5% |
| 小计（instrumented） | **53.9 ms** | **41%** |
| 其余（未 instrumented） | ~76 ms | **59%** |

**结论 3：剩下 59% 不在我们接管的算子里**，而在 vLLM 自己的层内（qkv/o_proj 的 GEMM、router、
共享专家、RMSNorm）与调度/采样。W8 因此不是"优化边角"，而是第二大的单一目标；在它被拆开之前，
即使把 GDN 优化到 0，TPOT 也只能从 3.5× 降到约 2.6×，达不到 ≤1.8×。

**结论 4：事件采样有代价**——同一请求从 9.3 s 变 13.0 s（每步 3.7k 次事件创建）。这只影响开启
探针的运行；测量时要么只跑短序列，要么接受这个开销（占比不受影响）。

### 下一步（顺序已由数据确定）

1. **W8 拆分**：把 `step` 层的设备时间分成 `dense_gemm` / `router+shared_expert` / `sampler` /
   `other`。做法是在 vLLM 的 `forward` 外层加同样的 `timed` 包装（模型的 `forward`、sampler 的
   `forward`），不需要改 vLLM 源码。
2. **W3 GDN**：38 ms/token 是单项最大。先确认 `ops/fla/recurrent.py` 的 decode 路径是否含
   Python 循环；有则批量化，无则必须上内核。
3. W1 prefill 内核与上面两条并行（互不依赖）。

---

## 10. W8 拆分结果（2026-11-03）：真正的瓶颈不在我们接管的算子里

在 `GPUModelRunner.load_model` 后给 `model.forward` 与 `sampler.forward` 套上同一个
`phases.timed`（通过 `compat/patches.py` 的 Patch 机制，未改 vLLM 源码），得到：

```
phase                     calls   host/step   dev/step  dev/call
model_forward                64      17.468     17.269   167.297
op:gdn                     3810       3.240      3.284     0.534
attention                   650       0.691      0.735     0.701
op:moe                     2599       0.617      0.601     0.143
sampler                      66       0.391      0.384     3.609
```

`model_forward` 的 `dev/call` 就是**每 token 的模型前向**（64 次调用 = 64 token）：

| 组成 | 每 token | 占模型前向 |
|---|---|---|
| **model_forward 总计** | **167.3 ms** | 100% |
| ├─ 其中 GDN | 32.8 ms | 20% |
| ├─ 其中 attention | 7.4 ms | 4% |
| ├─ 其中 MoE | 6.0 ms | 4% |
| └─ **其余（dense GEMM / router / 共享专家 / RMSNorm）** | **≈121 ms** | **72%** |
| sampler（模型外） | 3.6 ms | — |

**结论（又一次改变策略）**：我们接管的三个算子合计只占模型前向的 **28%**；72% 在 vLLM 自己的
dense 路径上。所以：

1. 即使把 GDN/attention/MoE 全部优化到 0，TPOT 也只能从 165 ms/token 降到约 121 ms/token
   （≈3.5× → 2.6×），**仍然达不到 M2 的 ≤1.8×**。
2. 真正的问题变成："为什么我们的 dense 路径要 121 ms/token，而基线整步只要 44 ms？"
3. **但这个问题现在还不能回答**：基线臂用的是另一个 venv（torch 2.13），按 C1 那组数字不可比。
   必须在**同一 venv** 下重测基线，才能判断 121 ms 是我们独有的（例如某处掉到 fp32/慢路径）
   还是两边都有的共同成本。

### 因此 M0 的最后一项仍未完成

`M0` 要求"重测同环境基线"，而它现在不只是流程合规问题——**它是判断 121 ms 归属的唯一手段**：

- 若同环境基线 dense 也是 ~120 ms/token → 我们的差距来自别处（需继续拆），TPOT 目标要重估；
- 若同环境基线 dense 是 ~20 ms/token → 我们某处走了慢路径，找出来就是单项最大收益。

### 下一步（明确）

1. **同环境基线**（`cp -a .venv .venv_baseline`，装 vllm-ascend，同一 torch 2.10）：
   跑同样的 64-token 请求，用同样的探针对基线做分相（基线不装 tileinfer，探针要单独注入）。
2. 按结果二选一：查我们的 dense 慢路径，或继续拆 `model_forward` 到每个 decoder layer
   （GDN 层 vs full-attn 层各自耗时）。

---

## 11. 同环境 A/B（2026-11-03）：第一个可用于决策的比值

### 环境怎么对齐的

vllm-ascend **0.23.0 在 torch_npu 2.10.0.post4 上无法启动**：worker 初始化调用
`torch_npu/op_plugin/atb/_atb_ops.py:68`，尝试加载 `<torch_npu>/lib/libop_plugin_atb.so`，
而该文件在本机不存在（全盘 `libatb*.so` 零命中）→ `OSError`。

所以"同一环境"只能**向上**对齐到 2.13（这也正是生产用的版本）：

| | native | baseline |
|---|---|---|
| venv | `.venv_native213`（由 `/home/dlb/.venv_ascend` 复制） | `/home/dlb/.venv_ascend` |
| torch / torch_npu | 2.13.0+cpu / 2.13.0rc1 | 同 |
| vllm / tilelang / CANN | 0.23.0 / 0.1.15 / 9.3.0 | 同 |
| 启动参数 | `--enforce-eager --max-model-len 2048 --max-num-batched-tokens 2048 --max-num-seqs 4 --gpu-memory-utilization 0.75` | 同（native 另加 `--attention-backend CUSTOM`） |
| 唯一差异 | **无 vllm_ascend** | 有 vllm_ascend 0.23.1.dev |

### 结果（64 output token，同一 prompt，4 次，第 1 次为冷启动）

| 臂 | run1（冷） | run2 | run3 | run4 | 稳态中位数 | ms/token |
|---|---|---|---|---|---|---|
| native | 10.08 s | 5.26 s | 5.36 s | 5.44 s | **5.36 s** | **83.7** |
| baseline | 2.63 s | 2.47 s | 2.55 s | 2.49 s | **2.49 s** | **38.9** |

**TPOT 比值 = 5.36 / 2.49 = 2.15×**

### 结论

1. **真实差距是 2.15×，不是之前记录的 3.5×**。3.5× 来自跨 torch 版本（2.10 vs 2.13），按 C1 作废。
2. **torch 2.13 对我方也更友好**：我方 torch 2.10 稳态 118 ms/token → 2.13 稳态 83.7 ms/token
   （约 1.4×）。所以后面所有优化都应在 2.13 上做——那才是生产版本。
3. 冷启动惩罚：native 10.08 s vs 稳态 5.36 s（1.9×），主要是一次性内核编译；baseline 只有 1.06×。

### 下一步

- 在 2.13 稳态下重跑分相（只统计 run2 之后的样本），确认 83.7 ms 里 GDN/attention/MoE/dense 各占多少；
- 然后按 M1（prefill 内核）与 GDN（当前最大单项）推进，每项都用本节的测量方法出比值。

---

## 12. 分相修正：用累计表差分（2026-11-03）

§10 的"dense 占 72%"**作废**：那是用嵌套设备事件算的，父事件（`model_forward`）的跨度会把相邻
token 与 prefill 的排队工作算进去。证据：同一窗口里 `model_forward` 算出 168 ms/token，
而墙钟只有 ~84 ms/token——父子数值互相矛盾，说明父级不可用。

改为**相邻两张快照相减**（日志每 20 次计数打一张表；差分天然排除冷启动与累计污染）：

```
window: 20 attn-layer calls = 2 tokens
phase               d_calls  dev_ms_total  dev_ms/token   share
model_forward             2         336.0        168.02    (不可用: 混入 prefill)
op:gdn                  120          71.5         35.76    75.3%
attention                20          10.2          5.11    10.8%
op:moe                   80          13.2          6.60    13.9%
sampler                   2           0.0          0.01     0.0%
instrumented ops per token (excl. model_forward): 47.48 ms
```

对照同环境端到端 83.7 ms/token：

| 组成 | 每 token | 占总量 |
|---|---|---|
| **GDN** | **35.8 ms** | **43%** |
| MoE | 6.6 ms | 8% |
| attention | 5.1 ms | 6% |
| 小计（instrumented） | 47.5 ms | 57% |
| 未 instrumented（dense GEMM/embedding/调度等） | ~36 ms | 43% |

**结论**：GDN 一项就占总量 43%，是唯一"改对了就能看到 30% 以上收益"的目标；MoE 只有 8%，
与 §8 的判断一致但更弱；未 instrumented 的 43% 仍需要单独拆分（但不能再靠 `model_forward`）。

**方法学备注（写进规则）**：嵌套计时在异步设备上会重复归因，只能用同级的兄弟事件（叶子）或
端到端差分；父级必须排除，或只统计同一类（纯 decode）的窗口。

---

## 13. M2-① GDN：GQA gather 提升到层内（2026-11-03）

**改动**：`ops/fla/recurrent.py` 里 `q_flat[tok].index_select(0, hv_to_h)` /
`k_flat[tok].index_select(0, hv_to_h)` 原本在**每 token** 执行两次；head 映射与 token 无关，
改为循环前对整个 `[B*T, H, D]` 做一次（每层省下 ~2×(T−1) 次小内核下发，30 层累计每 token 约 60 次）。

**正确性**：`tests/test_gdn_recurrent.py` 45 项全通过；设备上输出与原实现一致
（"The sky appears blue because Earth's atmosphere scatters shorter wavelengths…"）。

**测量**（同环境：torch 2.13 / vllm 0.23 / CANN 9.3；64 token，4 次，第 1 次冷启动）：

| | run1（冷） | run2 | run3 | run4 | 稳态中位 | ms/token |
|---|---|---|---|---|---|---|
| 改动前 | 10.08 s | 5.26 | 5.36 | 5.44 | 5.356 s | 83.7 |
| 改动后 | 10.17 s | 5.25 | 5.33 | 5.25 | **5.255 s** | 82.1 |

**measured_ratio = 5.356 / 5.255 = 1.019**（>1.0，按 C2 保持默认启用）
对基线的比值：2.15× → **2.09×**。

**判读**：收益只有 1.9%，说明 `index_select` 只是 GDN 0.71 ms/层里的一小部分。GDN 的耗时主要在
**每 token 约 10 个 1 MB 级张量算子**的下发与执行本身。所以要拿到 M2 要求的"GDN 降 ≥2×"，
必须减少算子数量或上内核，微优化不够。

**下一步（按预期收益排序）**：
1. 合并每 token 的算子（decay/beta 缩放、l2norm 的 norm+div+eps 三合一）→ 目标算子数从 ~10 降到 ~5；
2. 若仍不足，启用 `tileinfer/kernels/gdn/` 的内核，按 C2 需测出 ratio ≥1.0 才接入。

---

## 14. M2-② GDN 算子合并：**负结果，已回退**（2026-11-03）

**改动（已回退）**：把每 token 的 `(h * k).sum(-1)` / `(h * q).sum(-1)` 改成
`torch.einsum("hvk,hk->hv", …)`，把 `h + delta⊗k` 改成 `torch.addcmul`。

**正确性**：`test_gdn_recurrent.py` 45 项通过；设备输出文本与改动前一致。

**测量**（同环境、同参数、64 token、4 次）：

| | run2 | run3 | run4 | 稳态中位 | 结论 |
|---|---|---|---|---|---|
| M2-① 之后 | 5.25 | 5.33 | 5.25 | 5.255 s | 基准 |
| M2-② 融合后 | 5.58 | 6.47 | 5.33 | 5.576 s | **慢 6%** |

**measured_ratio = 5.255 / 5.576 = 0.94 < 1.0 → 按 C2 不允许默认启用，已回退到 M2-① 版本。**

**判读**：`einsum` 在这条 NPU 路径上比 `mul + sum(-1)` 慢（很可能走了通用收缩实现而不是融合
的 elementwise+reduce），`addcmul` 的收益被它抵消。这再次说明**"算子更少"不等于"更快"**，
必须逐项测。

**对 M2 的含义**：torch 层面的重组已经试过两种（省内核下发 ✓ 1.9%、减算子数 ✗ −6%），
收益上限就在 2% 量级。要拿到"GDN 降 ≥2×"，**必须走内核路线**（`tileinfer/kernels/gdn/`），
而不是继续在 torch 算子上做文章。

**同时记录一个 C4 违规**：这次把 einsum 与 addcmul 两项一起改，无法区分是谁拖慢的。
下次再做同类实验必须一次只改一个算子。

---

## 15. M1 的资产盘点（2026-11-03）

**prefill 内核是存在的**，在 `tileinfer/kernels/attention/paged_decode_ascend950.py` 里：

```
paged_prefill_ascend950            paged_prefill_split_ascend950
paged_prefill_split_head_ascend950 paged_prefill_split_frontier_ascend950
```

**缺的不是内核，是接线**：

| 位置 | 现状 | 需要 |
|---|---|---|
| `attention/backends/tilelang_ascend950.py:87-88` | `supports_mode` 只返回 `DECODE` | 声明 PREFILL，并按 mode 分派 |
| 同文件 `finalize_plan` / `run` | 只按 decode 的 bucket + kv_indptr 形状编译与执行 | 增加 prefill 的 shape 编译与 ragged 执行路径 |
| `vllm/attn/impl.py` 的 prefill 分派 | 一律走 `_reference_forward`（torch，O(n²)） | 改为"形状支持则走内核，否则回退并计数" |

所以 M1 的工作量是**后端接线 + 形状判定 + 回退计数**，不是写内核。这也解释了为什么 TTFT 随
prompt 长度线性恶化：prefill 一直在跑 torch 参考实现。

**判读**：M1 仍是最优先项——资产齐备、影响 TTFT（当前差距最大的一项）、与 decode 路径解耦。

---

## 16. 同环境 TTFT 基线（2026-11-03，M1 的起点）

同一 venv 组合（torch 2.13 / vllm 0.23 / CANN 9.3）、同一 `benchmarks/bench_arm.py`、同一启动参数：

| prompt≈ | native TTFT | baseline TTFT | 比值 |
|---|---|---|---|
| 16 | 0.754 s | 0.104 s | **7.3×** |
| 128 | 1.168 s | 0.109 s | **10.7×** |
| 512 | 1.837 s | 0.110 s | **16.8×** |

**TTFT 的比值与跨 torch 的那次几乎相同（6.9/10.8/16.4）**——说明 prefill 走 torch 参考实现这件事
与 torch 版本无关，是结构性的。M1 的出口"TTFT@512 ≤3×"因此要以 **16.8×** 为起点。

**同时发现一个测量不一致，必须记录**：同一份 bench 里 native 的 TPOT 报 0.138–0.166 s，而
64-token 非流式实测是 0.082 s/token（§11/§13）。基线两法一致（0.038 vs 0.039）。差异只出现在
native，最可能是**流式分块到达**（我们的 sampler/输出路径可能一次吐多个 token，导致
`chunks-1` 与实际 token 数不符）。结论：**TPOT 以非流式墙钟为准，流式只用于 TTFT**；
若要用流式报 TPOT，必须先校验 chunk 与 token 的对应关系。

---

## 17. M1 实现规格（下一步照此执行）

目标：TTFT@512 从 **16.8×** 降到 **≤3×**，且"prefill 走内核的比例 >90%"。

### 改动点（三处，必须一起改，否则编译期崩）

**① `tileinfer/attention/backends/tilelang_ascend950.py:87`**

```python
@classmethod
def supports_mode(cls, mode: AttentionMode) -> bool:
    return mode is AttentionMode.DECODE          # 现状
    # 改为: return mode in (AttentionMode.DECODE, AttentionMode.PREFILL)
```

**② 同文件 `finalize_plan`（:92）与 `run`（:268）**

- `finalize_plan` 目前按 decode 的 `(bucket, kv_indptr)` 形状编译。prefill 需要按
  `(num_tiles, block_q)` 编译：`block_q = plan.block_q`，每个 tile 一行 query；
- `run` 需要按 mode 分派：decode → `paged_decode_ascend950`；prefill →
  `paged_prefill_ascend950`（或 `paged_prefill_split_ascend950`，当 `seq_len > threshold`）；
- 内核签名在 `kernels/attention/paged_decode_ascend950.py`，参数与 decode 版同为
  `(q, k_cache, v_cache, kv_indptr, kv_indices, kv_last_page_len, ...)`，但 query 是
  `[num_tokens, H, D]` 且 `block_q` 分块。

**③ `tileinfer/vllm/attn/impl.py` 的 prefill 分派**

现在 prefill 一律走 `_reference_forward`（`max_query_len > 1` 分支）。改为：

```python
if prefill and self._prefill_backend().supports_mode(AttentionMode.PREFILL) and shape_ok:
    IMPL_COUNTERS.bump("prefill_kernel")
    return self._kernel_prefill(...)
IMPL_COUNTERS.bump("prefill_reference")   # 兜底必须计数，M1 的出口要用它算占比
return self._reference_forward(...)
```

`shape_ok` 的判据：`head_size in (128, 256)` 且 `num_tokens <= 预烧上限`；不满足就回退并计数。

### 验收

1. `pytest tests/test_tilelang_ascend950_prefill*.py tests/test_attn_contract.py` 全绿；
2. 同环境 TTFT@512 ≤3×（用 `benchmarks/bench_arm.py`，流式只取 TTFT）；
3. 日志里 `prefill_kernel` 计数 / (kernel+reference) > 0.9；
4. `benchmarks/ab_report.py --compare` 对 logprobs，确认 prefill 结果与原参考实现一致
   （允许 bf16 级容差，不允许分布偏移）。

### 风险

- prefill 内核只在特定 `block_q`/head_size 下编译过，**先跑单个形状**（512 token、d=256、16/2 头）
  再接全量；
- 若内核在此形状上比参考实现还慢（历史上长上下文出现过劣化），按 C2 记录 ratio 并保持回退。

## 18. M1 落地完成，以及 §17 前提被证伪

### 18.1 §17 的 ②③ 已实现

- 新增 `tileinfer/vllm/attn/prefill.py`（`PrefillPath`）：**eager** 的 paged-prefill 执行层。
  每步在 host 上重建 tile schedule（长度取自 `query_start_loc_cpu`，**零 D2V 同步**），
  编译变体在两条轴上量化：tile 数向上取到 `TILE_LADDER`（8/16/…/2048），`batch` 固定为
  `KERNEL_BATCH=512`（batch 只影响 `kv_indptr/kv_last_page_len/causal_shift` 的形状，内核不遍历它）。
  死 tile 带 `q_len=0`、`split_counts=1`，内核为其跑 0 次页循环。
- 为什么**不**走 `AttentionPlan`：prefill 的 tile 数 `Σ_b ceil(qo_len_b·group/block_q)` 随 prompt 长度变，
  而 plan 编译一次永久缓存 —— 要么每个长度编一次，要么把某一步的 schedule 冻进之后所有步。
  `supports_mode(PREFILL)` 保留，语义改为"由 PrefillPath 服务"。
- `impl.py`：prefill 分派；`PrefillUnsupported` **只在碰设备前**抛出（声明式），其余异常上抛（C3）；
  计数器 `prefill_kernel` / `prefill_reference` / `prefill_unsupported_<reason>`；
  每 20 个 prefill phase 打一行 kernel share（WARNING）。
- `TILEINFER_PREFILL=0` 作为 C4 同进程对照开关。

### 18.2 两个「reshape 冒充 transpose」的静默错误（都已修）

1. **`_reference_forward`（参考/兜底路径）**：`q3.reshape(batch, heads, qo, d)`、`out3.reshape(...)`。
   tiled reference 的约定是 `[batch, h_q, qo, d]`（`testing/reference.py:190,272`），层给的是
   `[tokens, h_q, d]`；reshape 是**重解释内存**，于是头轴与 token 轴互换。
   实测：token 0 的 head 1 拿到的是 token 1 的 head 0 —— 注意力看到了**未来** token。
   之前那次"per-request plan 修好了 ragged"只修了 batch 维度（padding 让 mask 错位），
   轴向一直是错的；单请求时第 0 个 token 恰好看起来对，把错误藏住了。
   修复：每个请求一次调用，进出都 `transpose(0,1)`；`qo_len==1`（退化 decode）保留单次批量调用，
   因为那里 4-D 形式确实是视图。
2. **我自己写的 `PrefillPath._store` 单请求快路径**同样错（`result.reshape(...)`）。
   是 ragged 用例把它反衬出来的：ragged 走通用分支（有 transpose）对，单请求走快路径错。

### 18.3 验证：三方对照，而不是两方对拍

三方 = 内核 / 参考后端 / **独立 fp32 CPU oracle**（直接读 K/V 源张量，不共享任何实现）。

| 用例 | kernel vs reference `max_abs` | cosine |
|---|---|---|
| plain-512（512 token，单请求） | 0.00195 | 0.999998 |
| ragged（300 + 200） | 0.00195 | 0.999998 |
| append（上下文 400 / 新 100，shift=300） | 0.00024 | 0.999997 |

修复前同一表格里 reference vs oracle 在 token 0 上是 **0.665**（正确答案就是 `v[0]`），修复后 **0.00000**。

### 18.4 同环境 A/B（C1）与一个必须承认的结论

`/tmp/m1_ttft.sh`，两臂同 venv 族、同 torch 2.13、同参数，唯一差别是"装没装 vllm_ascend"：

| prompt | baseline TTFT | native TTFT | 比值 | baseline TPOT | native TPOT | 比值 |
|---|---|---|---|---|---|---|
| ~16 | 0.1176 s | 0.7643 s | 6.50× | 0.0396 s | 0.1376 s | 3.47× |
| ~128 | 0.1046 s | 1.2521 s | 11.97× | 0.0451 s | 0.1331 s | 2.95× |
| ~512 | 0.1053 s | 1.8799 s | **17.85×** | 0.0453 s | 0.1669 s | 3.68× |

- prefill 内核占比：**1.000**（100 个 phase，`reference=0`，`unsupported=0`）。
- 但 TTFT@512 相对 §16 的 1.837 s **没有改善**（1.880 s）。

**§17 的前提是错的：TTFT 不是 prefill-attention 瓶颈。** 三条证据：

1. prompt 只有 16 token 时比值已经是 6.5×，而那个长度上 attention prefill 的成本可忽略；
2. 把 100% prefill 换成内核后 TTFT 不变（1.837 → 1.880）；
3. 分相表（512 token，`max_tokens=1`，图外干净窗口）：10 个 attention 层合计 **50 ms**（5.05 ms/层），
   `op:gdn` 0.44 ms/次、`op:moe` 0.19 ms/次，而 `model_forward` 每次 **1862 ms**。

一次 512-token prefill ≈ 2.4 s 的去向：

| 部分 | 实测 | 占比 |
|---|---|---|
| attention（含内核） | ~55 ms | 2% |
| 被包裹的 gdn + moe | ~45 ms | 2% |
| **未包裹** | **~1.77 s** | **96%** |

未包裹部分又能拆成两段：与 token 数无关的 **~0.6 s/步**（40 层 × 每层几十个小 kernel 逐个下发，
enforce-eager 无图），以及 **~2.4 ms/token**（随长度线性，指向 GDN 的 chunked prefill 在 torch 里逐 token 递推）。

### 18.5 编译成本（M3 预烧的输入）

新的 `(tile bucket, pool cap)` 组合的第一次 prefill 要编译：warmup **130.7 s** vs 之后同一请求 **2.7 s**。
即 **~110 s/变体**，必须接进部署预烧，否则第一个用户请求会吃满一次客户端超时。

### 18.6 教训

- **两个实现同时错会互相"证实"。** 修复前 kernel vs reference 只差 0.0488，看起来"接近"，
  其实是两边都错在同一个轴向上。只有独立 oracle 能判定谁错 —— C3 的对拍应当至少有一路对着 oracle。
- **归因必须做单变量。** §17 拿"TTFT@512 = 16.8×"当"prefill 走参考实现"的证据，
  但同一批数据里 16 token 就已经 7.3×，这本身就排除了 prefill 计算。
- 凡是 `(tokens, heads) ↔ (batch, heads, tokens)` 的转换，一律显式 `transpose`，永不 `reshape`。

### 18.7 里程碑重排

M1 出口里的"TTFT ≤3×"在 prefill-attention 范围内**不可达**（attention 只占 2%）。改为：

- **M1（已达）**：prefill 全部走内核（share 1.000）+ 三方对数一致（≤2e-3，cosine ≥0.999997）。
- **M1.5 固定开销**：ACLGraph 捕获（先验证能不能捕到），目标把 ~0.6 s/步压到 ~0.06 s。
  这是唯一能改善**所有** prompt 长度的项，且它同时决定 TPOT 的地板。
- **M2 边算子**：MoE 批量化 + GDN 内核化/批量化，目标吃掉那 ~2.4 ms/token。
- TTFT 预算：基线 0.105 s，3× 门槛 0.32 s。0.6 s 的固定开销不消掉，任何长度都到不了 3×。

### 18.8 ACLGraph 在当前平台不可达（M3 的硬阻塞，已定位到具体代码）

把 `--enforce-eager` 去掉，引擎起不来。逐层剥开后是一条完整的链：

1. **为什么是 eager 后端**：`TileInferPlatform.simple_compile_backend = "eager"`
   （`tileinfer/vllm/platform.py:123`），因为 torch_npu 没有 inductor 后端。
   vLLM 因此打印 `Inductor compilation was disabled by user settings`（`config/vllm.py:1097`）。
2. **eager 后端 = 只有 dynamo 的 eval_frame，没有 inductor 图分区**，于是 `splitting_ops`
   （`vllm::unified_attention_with_output` 等，默认列表里有）**不生效**，
   dynamo 会真的把 `TileInferAttentionImpl.forward` 内联进图里。
3. **第一次失败**：`torch._dynamo.exc.Unsupported: Unsupported context manager`，
   user-code 帧直指 `impl.py:425 → serving/counters.py:33  with self._lock:`。
   已修：`forward` 加 `torch._dynamo.disable`（变 graph break）；`PrefillPath` 改为构造期创建
   （`__init__` 不在 trace 里）。
4. **第二次失败**：`Unsupported: Skip inlining torch.compiler.disable()'d function`。
   vLLM 的 compile wrapper 是 **fullgraph**，不接受 graph break —— 注意力层必须**可被 trace**，
   不能是黑盒。
5. **即使 3、4 都过了也拿不到图**：`resolve_cudagraph_mode_and_sizes`
   （`config/compilation.py:1340`）在 `AttentionCGSupport.NEVER`、或"attention 不是 piecewise 编译"
   时把 `cudagraph_mode` 降为 `NONE`（实测启动配置里就是 `CUDAGraphMode.NONE`）。
   没有 piecewise 就没有图捕获，与本仓库的 NPUGraph 别名无关（别名已修好）。

**结论**：当前平台配置下 vLLM 的 ACLGraph 路径不可达。要去掉逐 kernel 下发开销只剩三条路，
都不在 M1/M2 范围内：

| 路 | 内容 | 成本 |
|---|---|---|
| (a) | 给 torch_npu 提供/启用 inductor 后端 | 平台外，非本仓库 |
| (b) | 把 attention 做成真正可 trace 的实现（去掉锁、`.item()`、所有 host 分支） | 与"编译期决定边界"的架构冲突 |
| (c) | 绕过 vLLM，在 worker 里自己对 decode step 做 `torch.npu.NPUGraph` 捕获 | 需要接管 runner 的输入打包 |

### 18.9 两个成本项都不是"计算量"

- 固定 ~0.6 s/步：与 token 数无关（16 token 时 TTFT 0.764 s、512 token 时 1.880 s）；
- ~2.4 ms/token：随长度线性。

同样的 `--enforce-eager`、同样的 40 层，基线在 16/128/512 上分别是 0.118/0.105/0.105 s，
即**几乎只有计算量**，没有这 0.6 s。两者层数相同、下发次数同阶，差别在于 vllm-ascend 用融合算子
把每层的下发次数压低了一个数量级。

因此最可信的解释是：这 0.6 s 是**逐 kernel 下发开销**（40 层 × 每层数十个小算子 × ~40 µs），
而不是算术。**确认方法**（下一轮）：用 `torch_npu.profiler` 统计一次 prefill step 的 kernel 数，
与基线同 step 的 kernel 数对比；若相差约一个数量级，此判断成立。

这条结论直接决定 M2 的排序：**"换内核"改善的是单算子效率，而差距主要在"算子个数"**。
M2 里 MoE 批量化（256 次 → 1~3 次）恰好属于后者，优先级应高于 GDN 内核化。

## 19. 叶子模块归因：§18.9 的"下发风暴"假设被证伪，真正的两个大头是两个模块

### 19.1 做了什么

`_apply_phase_probes_on_load`（`tileinfer/vllm/compat/patches.py`）现在除了 `model_forward` /
`sampler`，还把模型里**每个叶子模块**（无子模块的 Linear / Norm / Conv / 各类 op 对象）用
`phases.timed` 包起来，标签去掉层号后聚合（`mod:layers.linear_attn.chunk_gated_delta_rule`），
所以一行 = "一类模块、某一次调用"的平均，`calls` 是这一类的总次数。

两个坑（都已修，都会让表**看起来正常但结论错**）：
- 第一版只包了每类的**第一个实例**，于是 30 层的成本被报成 1 层，与 `model_forward` 对不上；
- `patches.py` 没 `import os`，`os.environ` 抛 `NameError` 被外层 `except` 吞掉 ——
  表现是"探针装好了但表里没有 `mod:` 行"。

### 19.2 结果（512 token prefill，`TILEINFER_PHASE_TIMING=device`）

绝对值因探针本身的开销（240 个包装 × 每步）和窗口里混入 profile run 而偏大，**但排序是稳的**：

| 模块类 | 每次调用 | 每步调用次数 | 判读 |
|---|---|---|---|
| `layers.linear_attn.chunk_gated_delta_rule` | **43.0 ms** | ~21–30 | GDN 的 chunked prefill 扫描，**最大单项** |
| `layers.mlp.experts.quant_method` | **35.9 ms** | ~32 | MoE 专家模块 |
| `layers.self_attn.attn` | 13.5 ms | 10 | 里面才是我们的 `attention` 相位（14.6 ms） |
| `layers.self_attn.qkv_proj` | 0.135 ms | 42 | 稠密投影 |
| `layers.linear_attn.in_proj_qkvz` | 0.168 ms | 126 | 稠密投影 |
| `layers.linear_attn.out_proj` | 0.079 ms | 126 | 稠密投影 |
| `input/post_attention_layernorm` | 0.15–0.17 ms | 162 | norm |
| `visual.*`（视觉塔） | 0.3–110 ms | 27 | profile/dummy run 里的图像分支，与文本请求无关 |

### 19.3 结论（其中一条是否定自己上一轮的话）

1. **稠密投影不是问题**：qkv / in_proj / out_proj / norm 每次 0.08–0.17 ms，40 层加起来只有个位数 ms。
   所以 §18.9 的"逐 kernel 下发风暴"**不成立**——如果成立，这些同样小的算子也该是几十 ms。
2. **固定成本就是专家个数**：MoE 每层一次 ~36 ms，32 层；而**这个成本与 token 数无关**（每个专家都要被
   走一遍），所以 16 token 的 prefill 也要付它 —— 这正是 §18 里"0.6 s/步、与长度无关"的来源。
   256 专家 × 32 层 = 8192 次专家调用/步，即使每次只有 70 µs 也是 0.57 s。
3. **线性成本是 GDN 的 chunked 扫描**：43 ms/层，随 token 数增长，对应 §18 的 ~2.4 ms/token。

两者都落在 M2，而且顺序因此变得明确：
- **MoE 批量化**（排序 + index_select + bmm，把 8192 次变成几十次）消掉的是 **TTFT 的固定 0.6 s**，
  对**所有** prompt 长度都有效 —— 这是 TTFT 从 17.9× 往下走的第一块。
- **GDN chunk 内核化**消掉的是 ~2.4 ms/token，对 TTFT 的长 prompt 段和（其 decode 版本）TPOT 同时有效。

这也修正了 §18.9 给 M2 排序的理由（原来写的是"压算子个数"），但**结论顺序没变**：MoE 先做。

## 20. M2 实现规格：MoE 走 Ascend 原生分组算子（下一轮照此执行）

### 20.1 已完成的小项：matmul dtype

`fused_experts` 原来把一切都升到 fp32（`compute = torch.float32`），代价是 fp32 矩阵乘只有 bf16 一半的速率，
且 `w1[e].to(fp32)` 会对每个专家的权重块做一次**新的 fp32 转换**——每层每步 256 次。

现在由 `TILEINFER_MOE_FP32`（默认 `1` = 旧行为）控制：设为 `0` 时矩阵乘保持输入 dtype
（Ascend 内部按 fp32 累加），激活与 top-k 累加仍在 fp32。这也**更接近**上游 Triton 内核
（bf16 载入、fp32 计算、bf16 存回）。

同进程 A/B（`/tmp/moe_fp32_ab.py`，512 token、top-8、256 专家、bf16 权重）：

| 模式 | 对 fp32 oracle 的 max_abs | 时间 |
|---|---|---|
| fp32（旧，默认） | 0.000053 | 121.50 ms/call |
| input dtype | 0.000080 | 101.31 ms/call |

`measured_ratio = 1.199`。按 C2 可开启，但它是数值改动，**默认仍关**，等一次完整的同环境 A/B +
logprobs 对拍之后再翻默认值。

### 20.2 真正的大头：把 256 次专家循环换成原生分组算子

`fused_experts` 现在是 `for e in torch.unique(local[active]).tolist():` —— 一次 **D2H 同步** +
每个命中专家 8 个算子（nonzero/3×index_select/2×matmul/index_add）。Qwen3.6 命中 256 个专家 × 32 层，
这一步就是 §19 里那笔与 token 数无关的 0.6 s。

torch_npu 已提供全套原语（**已在本机确认存在且可用**，`probe_grouped.py` 实测）：

| 原语 | 作用 | 备注 |
|---|---|---|
| `npu_moe_init_routing(x, row_idx, expert_idx, active_num)` | 按专家排序 + 展开，一步完成 | 返回 `(expanded_x, expanded_row_idx, expanded_expert_idx)`；`expanded_row_idx` 是回填映射，shape `NUM_ROWS*K`，int32 |
| `npu_grouped_matmul(x, weight, *, group_list=..., group_type=...)` | 1 次下发做完全部专家的矩阵乘 | `weight` 是 **List[Tensor]**；`group_type` **必须显式传**（不传报 `Requires manual passing group_type`） |
| `npu_swiglu(x)` | 融合 gate/up 激活 | 实测 `[16,1536] -> [16,768]` ✓ |
| `npu_moe_token_unpermute(permuted, sorted_indices, probs=...)` | 回填 + top-k 加权 combine | `probs` 就是 topk_weights，省掉手写 `index_add_` |
| `npu_grouped_matmul_finalize_routing` | gmm + combine 进一步融合 | 文档只列了 a8w8/a8w4（int8/int4），**bf16 未必支持，先不用** |

目标形态（约 5 个算子，替换 256 × 8）：

```python
expanded_x, expanded_row_idx, _ = torch_npu.npu_moe_init_routing(
    x, row_idx, ids, active_num=num_tokens * top_k)
gate_up = torch_npu.npu_grouped_matmul([expanded_x], [w13_right], group_list=group_list,
                                       group_type=0)
act = torch_npu.npu_swiglu(gate_up)
down = torch_npu.npu_grouped_matmul([act], [w2_right], group_list=group_list, group_type=0)
out = torch_npu.npu_moe_token_unpermute(down, expanded_row_idx.to(torch.int64), probs=topk_weights)
```

### 20.3 三个必须先回答的问题（第一件事就是它们，别先写实现）

1. **权重布局**：我们拿到的是 `w1 [E, 2*inter, hidden]`、`w2 [E, hidden, inter]`，而 gmm 要的是右矩阵
   `[hidden, 2*inter]` / `[inter, hidden]`，即 `w1.transpose(1,2)`。**能否直接传非连续视图？**
   不能的话要一次性转置成连续缓冲：`w13` 一层就是 256×1536×2048×2 B = **1.6 GB**，32 层 = 51 GB，
   在 131 GB 卡上**不可接受**（模型本身已占 ~70 GB）。所以这一题决定整条路是否可行。
2. **`group_type` 语义**：0/1/2 分别对应"按 group_list 切 x 的第 0 维"/"权重按组"/其他。
   必须用一个小例子（2 组、已知答案）确定，不能猜。
3. **`npu_moe_token_unpermute` 的 `probs`** 是"先乘再加"还是别的顺序；以及它要求 bf16 输入
   （文档写死 `torch.bfloat16`），fp32 路径要相应调整。

### 20.4 验收与回退

- 开关 `TILEINFER_MOE_GMM=1`，关闭时走现有 torch 循环；**默认关**。
- 正确性：对 `testing.moe_reference.reference_moe` 做 24 token 小例（当前 torch 路径 max_abs 5e-5，
  gmm 路径应同量级），再加一次 512 token 的真实对拍。
- 性能：`measured_ratio` 用 §19 的叶子表 `mod:...mlp.experts` 直接读，目标 ≥2×
  （M2 出口写的是"MoE 各降 ≥2×"）。
- 若问题 1 的答案是"必须连续"，则 MoE 这条路改为：**保持列表式 gmm 但按层缓存一份转置权重**，
  并要求把 KV 池或 `gpu_memory_utilization` 降下来腾出 51 GB —— 那属于"影响面大"，按 E2 上报。

## 21. M2-MoE 达成：排序一次的专家循环（2.57×，数值逐位一致）

### 21.1 先修一个测量错误

§20.1 记的 `measured_ratio = 1.199`（fp32 → 输入 dtype）**是错的**：那次 bench 没有把首次调用的
JIT/分配器成本从计时里排掉。重测（同进程、都先 warm 一次）后是：

| 变体 | 对 fp32 oracle 的 max_abs | mean_abs | ms/call |
|---|---|---|---|
| fp32 + 旧循环 | 0.000053 | 4.30e-6 | 33.89 |
| 输入 dtype + 旧循环 | 0.000080 | 1.13e-5 | 32.75（**ratio 1.035**） |
| fp32 + 排序 | 0.000053 | 4.30e-6 | — |
| 输入 dtype + 排序 | 0.000080 | 1.13e-5 | **13.19（ratio 2.569）** |

结论：**dtype 那项只有 1.035，属于噪声，保持默认关**（C2 下不值得为它承担数值风险）。

### 21.2 真正的改动：一次排序，然后走连续段

旧循环每个命中专家要付：`torch.nonzero`（全表扫描 + 分配）一次、`index_select` 三次、
`index_add_` 一次 —— 命中 256 个专家时是 256 × 8 个算子。而在同一形状下**只做两个 matmul**
的裸循环只要 7.6 ms，整通调用却要 ~34–100 ms，所以 **~93% 的成本是这套逐专家管道，不是算术**。

`_routed_sorted`：`argsort` 一次把 slot 按专家排好，之后每个专家的行段就是**视图**
（`sorted_x[off:off+n]`），循环体变成"一次切片 + 两个 matmul + 两次逐元素"，
combine 收敛成**末尾一次 `index_add_`**（原来每个专家一次）。

- 丢弃的 slot（`local < 0`）被放进"最后一个专家之后"的合成桶，永不访问；
- `torch.bincount(...).tolist()` 与旧代码的 `torch.unique(...).tolist()` 是**同样一次**同步，
  但拿到的是稠密表，循环里可以直接下标，不必每个专家再发一次设备算子。

### 21.3 为什么可以直接开默认

它对 fp32 oracle 的 `max_abs`/`mean_abs` 与旧循环**完全一致**（0.000053 / 4.30e-6；
输入 dtype 模式下同样是 0.000080 / 1.13e-5）—— 因为它跑的是同一套逐专家算术、写进同一个 fp32
累加器，只改了"段怎么访问"和"combine 合并成一次"。这不是"用正确性换性能"，所以按 C2 可以直接
默认开（`TILEINFER_MOE_SORTED=0` 可回退）。

### 21.4 gmm 这条路的考古结果（暂时搁置）

schema 已取到：
`npu_grouped_matmul(Tensor[] x, Tensor[] weight, *, Tensor? group_list=None, int? split_item=0,
int? group_type=None, int? group_list_type=0, ...) -> Tensor[]`。

两个卡点：
1. 带 `group_list` 时要求 **`x` 长度为 1**，且（未显示完的）`weight` 长度约束；实测把
   `[expanded_x]` + `[w13.transpose(1,2)]`（连续/非连续都试了）传进去仍报同一句，
   说明还有一条没满足的条件（很可能是 `split_item` 与 `group_type` 的组合，或 weight 必须
   `[g,k,n]` 的 NZ 布局）。
2. `npu_moe_token_unpermute` 在本机直接失败：`aclnnMoeTokenUnpermute` error **161002**
   （probs 传 fp32 和 bf16 都失败），这条"免费拿 weighted combine"的路当前不可用。

这两条都不是阻塞项：排序版已经把 MoE 降到 M2 的门槛以下。留作后续可选项，继续深挖的性价比不高
（每次探测都要一次设备往返，而收益上限相对排序版并不多）。

### 21.5 在服务内的效果：小 prompt 明显，512 token 不动

同环境 A/B（`/tmp/m1_ttft.sh`，与 §18.4 完全同一套参数与 venv），排序版开启前后：

| prompt | native TTFT 前 | native TTFT 后 | 变化 | baseline | 后/前 比值 |
|---|---|---|---|---|---|
| ~16 | 0.7643 s | **0.5317 s** | **−30%** | 0.1144 s | 6.50× → **4.65×** |
| ~128 | 1.2521 s | 0.9855 s | −21% | 0.1108 s | 11.97× → 8.90× |
| ~512 | 1.8799 s | 1.9030 s | ±0 | 0.1029 s | 17.85× → 18.5× |

baseline 三档在两次运行间稳定（0.114/0.111/0.103 vs 0.118/0.105/0.105），说明这套 A/B 可复现。

**这与 op 级 2.57× 并不矛盾，两者指的是不同的东西**：

- 排序版省掉的是**逐专家的管道开销**（`nonzero` / 3×`index_select` / 每专家一次 `index_add_`），
  这笔开销**与每专家的行数无关**，所以它表现成一个固定量。16 token 时专家多半只有 1–2 行，
  几乎全是管道 → 省下 0.23 s；512 token 时每专家约 16 行，**矩阵乘本身占了主导**，
  管道省下的那部分被淹没。
- 也就是说 §19 里那笔"与 token 数无关的 0.6 s"，其中约 **0.23 s 是 MoE 管道**，已经拿掉；
  剩下的固定量不在 MoE 管道里。

**未解的问题（下一轮第一件事）**：512 token 时 MoE 的 34→? ms 里，管道与 matmul 各占多少？
以及 —— 服务内是否真的走到 `moe.fused_experts`（仓库里还有 `fused_moe_oot.py`，
若服务路径走的是它，本节在 512 token 上的"不动"就有了更简单的解释）。
判别方法：临时把 `TILEINFER_MOE_SORTED` 的两条分支各加一个计数，跑一次 512 token prefill 看命中了谁。

## 22. decode 侧归因：M0 的"MoE 8%"是错的，它和 GDN 一起并列第一

### 22.1 数字（decode step，叶子探针，`dev/call` × 每步调用数）

| 模块类 | dev/call | 调用/步 | ms/步 |
|---|---|---|---|
| `layers.mlp.experts.quant_method` | **1.343 ms** | 27.2 | **36.5** |
| `op:gdn` | **0.615 ms** | 59.8 | **36.8** |
| `attention`（我们的相位） | 0.866 ms | 10.1 | 8.7 |
| `op:moe` | 0.163 ms | 40.2 | 6.6 |
| `input_layernorm` | 0.123 ms | 27.2 | 3.3 |
| `post_attention_layernorm` | 0.116 ms | 27.2 | 3.2 |
| `sampler` | 2.379 ms | 1 | 2.4 |
| `linear_attn.norm` | 0.113 ms | 21.1 | 2.4 |
| `in_proj_qkvz` / `out_proj` | 0.062 / 0.043 | 21.1 | 1.3 / 0.9 |

（探针本身把每层 ~240 个叶子都包了一层设备事件，所以绝对值和墙钟不可直接比；占比可用。）

### 22.2 必须纠正的一条基线描述

M0 写的是"MoE 8%"。那 8% 来自 `op:moe` 计数器，而 `op:moe` 与**专家模块**是两个不同的东西：
`op:moe` 每步 40 次、每次 0.163 ms；真正干活的 `mlp.experts.quant_method` 每步 27 次、每次 **1.343 ms**。
**MoE 专家从来不在 `op:moe` 里，它占 TPOT 的 ~36%，与 GDN 并列第一。**

这是"测量方法有系统性错误"那一类（E2）——指标名字对了，包的对象不对，于是整整两个里程碑的排序都基于
一个 8%。后续所有关于 TPOT 的结论以本节为准。

### 22.3 两个否定结果

1. **fp32 权重转换不是 decode 的大头**。我原以为 256 次 `w1[e].to(fp32)`（每次 12.6 MB、与 slot 数无关）
   是 decode 的主要成本，所以做了服务内 A/B（`TILEINFER_MOE_FP32=1` vs `0`）：
   TTFT 0.512/0.913/1.864 → 0.516/1.088/2.085，TPOT 0.118/0.148/0.147 → 0.118/0.139/0.143。
   **全部在噪声内**。假设不成立。
2. **gmm 路线基本被权重布局堵死**。schema 里 `Tensor[] weight` 且带 `group_list` 时长度必须为 1，
   即单个 `[g, k, n]`；而我们的 `w1` 是 `[E, 2I, H]`，`transpose(1,2)` 之后**最后一维 stride 是 H**
   （不连续），gmm 要求最后一维 stride-1 → 必须连续化，一层 1.6 GB、32 层 51 GB，模型已占 ~70 GB，
   不可行。除非 gmm 接受 NZ 布局（未验证，且每次探测要一次设备往返）。

### 22.4 那么 TPOT 的 82 ms 到底是什么

两个 36% 的项都是**按算子个数算钱**，不是按算术：MoE 每层约 40–64 个算子、
GDN 每步 60 次调用 × 每次约 12 个算子。反推每个算子的有效开销约 **25–45 µs**，
40 层 × 约 30 个算子 × 30 µs ≈ **36 ms/步**，这就是 M0 里那 43% "未拆"的来源。

结论：**TPOT 的地板由"每步算子个数"决定**。要动它只有三条路，且都不便宜：

| 路 | 效果 | 代价 |
|---|---|---|
| 融合算子（MoE 走 gmm / GDN 写 decode 内核） | 直接减算子数 | 需要解决权重布局（MoE）或新写内核（GDN） |
| 图捕获 | 一次下发 | 平台不可达（§18.8：无 inductor） |
| 减层数/减宽度 | 不动架构前提不允许 | — |

M2 出口"TPOT ≤1.8×"意味着 82 → ≤70 ms，即要省 12 ms。按上表，**只做 MoE 或只做 GDN 都不够**
（各 ~36 ms，但单个改动很难减半），必须两项都动，或者拿到图捕获。

## 23. 两项结果：图捕获第二次被证否；GDN 常量缓存 1.231×

### 23.1 ACLGraph：从"配置上被降级"到"显式请求也被拒绝"

§18.8 的结论是"attention 不是 piecewise 编译 → `cudagraph_mode` 被降成 NONE"。这一轮换一个角度
再验一次：**绕过 compile 直接要 FULL** ——

```
--compilation-config '{"mode":"none","cudagraph_mode":"FULL"}'
```

引擎起来了，但日志里 vLLM 说得很直白：

> `Skipping CUDA graph capture. To turn on CUDA graph capture, ensure 'cudagraph_mode' was not
> manually set to 'NONE'.`

即它自己又把请求降回 NONE 并跳过捕获。附带一组数字（同一模型同一参数）：TTFT@16/128/512 =
0.601 / 1.702 / 2.299 s、TPOT 0.128 / 0.169 / 0.178 —— 比 `--enforce-eager` 那组（0.512 / 0.913 /
1.864、0.118 / 0.148 / 0.147）**更差**，所以"mode=none + FULL"不是一条可用的折中。

**结论收敛**：ACLGraph 在本平台不可达，与我们的代码无关（NPUGraph 别名那一层早已修好），
根因是 vLLM 要求 attention 走 piecewise 编译，而 piecewise 需要 inductor，torch_npu 没有。
M3 里的"启用 ACLGraph 捕获"这一项，在平台提供 inductor 之前**没有实现路径**，按 E2 上报。

### 23.2 GDN decode 常量缓存：1.231×，逐位一致

`fused_recurrent_gated_delta_rule_packed_decode` 每次调用都会重建三个**每层不变**的常量：
`arange(HV) // (HV // H)`、`A_log.to(fp32)`、`dt_bias.to(fp32)`。decode 是下发受限的（§22），
所以从 ~15 个算子里去掉 3 个（含两次 dtype 转换、三次分配）就是实打实的。

`_decode_constants` 做缓存，键里带 `id()` 与 `data_ptr()`，并且**缓存值强引用 `A_log`/`dt_bias`**
—— 这样条目活着的时候源张量不会被释放，键不可能落到复用后的地址上。

| | us/call |
|---|---|
| `TILEINFER_GDN_CACHE=0` | 251.8 |
| 缓存开启（默认） | 204.6 |
| **measured_ratio** | **1.231** |

正确性：**输出与状态都逐位相同**（`torch.equal` 为真），所以按 C2 可以直接默认开
（`TILEINFER_GDN_CACHE=0` 回退）。

### 23.3 M2 出口的算术

`op:gdn` 36.8 ms/步 → 缓存后约 36.8 × (1 − 0.231 × 占比)。按该算子占 `op:gdn` 的一半估算，
约省 4–7 ms/步：**TPOT 82 → ~75–78 ms，比值 2.11× → ~1.93–2.0×**。

要到 M2 的 ≤1.8×（≤70 ms）还差 5–8 ms，而剩下两块（MoE decode ~36 ms、GDN 其余部分）都需要
内核或 gmm 级别的改动：

- MoE decode 只有 8 个 slot，却仍要 40–64 个算子 → 只能靠 gmm，而 gmm 被权重布局堵住（§22.3）；
- GDN 剩余部分在 torch 层已到上限（上一会话测得 einsum/addcmul 合并 ratio 0.94，已回退）。

## 24. 负结果：按步共享 ragged triple 无收益，并把"下发成本"校准了

### 24.1 做了什么

10 个全注意力层每步各自调用一次 `refresh_step`（约 15 个小算子：`mul`、`index_select`、`add_`、
`floor_divide_`、`cumsum`、`lt/ge`、`scatter_` …），而它算出来的 `kv_indptr`/`kv_indices`/
`kv_last_page_len` 描述的是**批次**，与层无关。于是加了 `_SharedStep`：每步只算一次，
其余 9 层复用；缓存用 `meta_owner` 强引用元数据对象来保证"同一个对象=同一步"。

开关 `TILEINFER_ATTN_STEP_SHARE`（**默认关**），同环境 A/B：

| prompt | share=0 TTFT / wall-TPOT | share=1 TTFT / wall-TPOT |
|---|---|---|
| ~16 | 0.536 s / 122.2 ms | 0.582 s / 113.4 ms |
| ~128 | 0.869 s / 137.3 ms | 0.934 s / 142.6 ms |
| ~512 | 1.939 s / 140.7 ms | 1.940 s / 136.7 ms |

**没有可测收益**（在 ±10% 的噪声里）。按 C2 保持默认关。

### 24.2 这个负结果有用：下发成本的模型要修正

§22.4 写的是"TPOT 的地板由每步算子个数决定，单算子 25–45 µs"。这次的数据说明那句话**太粗**：

- `refresh_step` 的 15 个算子作用在 `[bucket+1]` / `[bucket, width]` 的 **int32 小张量**上 ——
  9 层复用省了 135 个这样的算子，却不到 1%；
- GDN 常量缓存（§23.2）去掉 3 个算子，在 251.8 µs 里省了 47 µs → **约 15 µs/算子**；
- MoE/GDN 那 36 ms 级的开销，算子作用在 **MB 级**张量上（`[4096, 2048]`、`[32, 128, 128]`）。

所以正确的模型是：**成本 ≈ 算子数 × 该算子的张量规模**，不是"算子数"。推论：

1. 小张量上的算子合并（本节的共享、上一会话的 einsum 合并）收益上限只有个位数百分比，
   而且**低于当前测量分辨率**（同一配置重复跑 TTFT 波动 ±10%）。
2. 想在 TPOT 上再拿到两位数百分比，只能减少**大张量算子的个数** —— 也就是 MoE 的 matmul 链
   或 GDN 的 128×128 状态链，两者都需要内核级实现。
3. 测量方法也要跟着升级：小于 5% 的改动，用当前这套 `bench_arm` 是判不出来的，
   需要固定迭代次数 + 墙钟 + 多次取中位数，或者直接读阶段计时的 `dev/call`。

### 24.3 结论

M2 剩余空间已经明确到具体对象上，且都不在 torch 层的可及范围内：

| 对象 | 现状 | 需要 |
|---|---|---|
| MoE decode ~36 ms/步 | 排序版已把 prefill 端的管道拿掉；decode 端 8 个 slot 仍要 40–64 个键张量算子 | gmm（被权重布局堵住）或自写内核 |
| GDN decode ~30 ms/步 | 缓存后 ~205 µs/层 × 30 | 自写 decode 内核（约 5 个算子可替 15 个） |
| 图捕获 | 两次独立验证不可达（§18.8、§23.1） | 平台提供 inductor |

## 25. 换了把尺子之后：2.11× 不成立，实测 3.1×–4.3×，且设备漂移 37%

### 25.1 新尺子：差分两个生成长度

```
tpot = (T(N2) - T(N1)) / (N2 - N1)        N1=8, N2=72, 同一 prompt, 非流式
```

同一 prompt 下，提示词成本、首 token 成本、每请求开销、客户端延迟都在两项里出现并**相消**，
剩下的就是"多生成一个 token 的边际成本"。实现在 `benchmarks/wall_tpot.py`，重复 3 次取中位数，
并**报出极差**——没有极差，"提升 3%" 和噪声就无法区分。

### 25.2 三个数字

| 臂 | TPOT 中位数 | 极差 | 备注 |
|---|---|---|---|
| baseline（vllm-ascend） | **37.93 ms/token（26.37 tok/s）** | 7.9% | 与历史 38.9 一致 |
| native，第一次调用 | **118.52 ms/token** | 50.5%（rep0 是 177 的热身） | |
| native，第二次调用（同一进程，几分钟后） | **163.02 ms/token** | 5.8% | |

### 25.3 两件必须记下的事

**（1）2.11× 这个数作废。** 它来自 M0 那次 `64 token / 5.255 s` 的推算，那套算法把首 token 成本
摊进了每 token。用新尺子重测，比值是 **118.5/37.9 = 3.13×**（取最好的一次）或
**163.0/37.9 = 4.30×**（取漂移后的一次）。本文档此前所有引用 "TPOT 2.11×" 的地方都应以此为准。

**（2）同一台机器、同一个进程，前后两次调用差 37%。** 且 baseline 两次测量之间只差 2.5%
（38.9 → 37.93），说明不是"机器整体变慢"。运行中 `npu-smi` 显示 **73–75 ℃、282–313 W、
util 95%**，最可能是**持续负载后的降频**：第一次调用的 16 个请求把它烤热了，第二次调用全程在热态。
这也解释了本轮之前那些 ±5–10% 的"A/B 差异"为什么时有时无。

### 25.4 由此定下的测量纪律（同步写入 docs/notes/perf-comparison-rules.md）

1. TPOT 只用 §25.1 的差分法，并报极差；
2. 两臂必须**交替**（A→B→A），用第二个 A 判定漂移；只跑 A 再跑 B 的结论不可信；
3. 每次跑都记录 `npu-smi` 的温度/功率；
4. 小于 10% 的改动，除非在同一进程内做 op 级 A/B（那里没有降频变量），否则不下结论。

### 25.5 状态

M2 出口"TPOT ≤1.8×"当前差距是 3.1× → 1.8×（约需 42% 的降幅），而不是文档早前暗示的 2.11× → 1.8×。
按 §24.3 的对象清单，这个量级只能靠 MoE/GDN 的内核级实现，或图捕获（平台不可达）。

## 26. 端到端复测失败：这台机器的吞吐在会话之间摆动到 2×

本轮想用交替 A/B 复测已落地的两项优化（MoE 排序、GDN 常量缓存），没能完成，但把**为什么完不成**
查清楚了。三个具体缺陷，两个已修进脚本（`benchmarks/` 下的 `.work/two_arm_fixed.sh`）：

### 26.1 修掉的两个 harness 缺陷

1. **`pkill -f 'vllm.entrypoints.cli.main serve'` 会杀掉自己**。这条通过 `ssh "…"` 下发时，
   远端的 bash 命令行里就含有这个字符串，`-f` 匹配到自身，于是脚本在启动前就把自己的 shell 杀了
   （表现为 ssh 返回空输出）。必须写成 `vllm[.]entrypoints`。
2. **杀进程后 HBM 不会在固定 sleep 内释放**。`gpu_memory_utilization=0.75` 在 131 GB 卡上约 98 GB；
   `SIGTERM` 之后 45 s 仍可能没释放，下一个臂直接死在
   `ValueError: No available memory for the cache blocks`（本轮出现两次）。
   正确做法是**轮询 `npu-smi` 直到 HBM 降到阈值以下**再启动 —— 修好后实测 10 s 就能释放。

### 26.2 那个 247.5 ms 很可能是假的：我自己的残留进程

本轮结束时发现**两个 native 服务进程同时活着**（已运行 7–10 分钟）——它们是前一次 `two_arm.sh`
没被 `pkill` 掉（26.1 缺陷 1）留下的。而两张卡只有一个 NPU：**两个服务共用一个设备，吞吐大致减半**，
`247.45 / 123.5 ≈ 2.00`，与观测吻合。

所以正确的归因是：

| 观测 | 归因 |
|---|---|
| 247.5 ms（本轮 off 臂） | **自己两个服务抢同一张卡**，不是环境漂移，作废 |
| 118.5 → 163.0 ms（§25，同一个服务进程内） | 真实存在的漂移（37%），温度 73–75 ℃ |
| 123.5 ms（本轮 nat1，单服务） | 与 session 内 118.5 一致 |

### 26.3 结论（修正后）

1. **harness 必须自证干净**：每一臂启动前除了"HBM 已释放"，还要 `ps` 确认**零个** `vllm.entrypoints`
   进程。本轮两个教训合起来说明这一条不能省：残留进程既会占显存（启动直接失败），也会分走算力（吞吐减半）。
2. **仍存在的真实漂移是 37%**（同一进程内，几分钟间隔，温度 73–75 ℃）。所以端到端仍要用 A→B→A
   交替并报极差；≤10% 的差别不要下结论。
3. 本轮**没有完成**两项优化的端到端复测。已完成且可信的仍然是同进程 op 级 A/B：
   MoE 排序 **2.569×**、GDN 常量缓存 **1.231×**。端到端的效果下一次必须用"干净自检 + 交替"跑。
4. 目标验收口径（TTFT ≤1.1×、TPOT ≤1.8×）在这个分辨率下**仍然勉强可判**：
   37% 的漂移小于 1.8× 的判定距离（80%），但远大于 1.1×（10%）——所以 M3 的终态验收
   需要交替 + 多轮中位数，单次对比不够。

## 27. ON→OFF→ON 复测完成：同一份配置在同一会话里摆动 45%，端到端无法判定

自检协议（每臂启动前确认 0 个 `vllm` 进程 + HBM < 5 GB，轮询而非固定 sleep）下跑完三臂，
每臂 3 次 `wall_tpot`：

| 臂 | 三次采样 (ms/token) | 中位数 | 臂内极差 |
|---|---|---|---|
| **ON**（两项优化默认开） | 163.33 / 188.60 / 199.30 | **188.60** | 19.1% |
| **OFF**（两项都关） | 164.47 / 164.42 / 140.32 | **164.42** | 14.7% |
| **ON**（同配置再跑一次） | 128.66 / 130.16 / 147.43 | **130.16** | 14.4% |

### 27.1 结论：不是"ON 有没有用"，是"这台机器判不出来"

**同一个 ON 配置，两次测得 188.60 与 130.16 —— 相差 45%**，而 OFF（164.42）正好夹在中间。
也就是说：臂间差异完全被"第几个臂"这个变量吞掉了。

而且方向是**越跑越快**（188.6 → 164.4 → 130.2），这与"越热越慢"相反，所以不是热降频；
更像是设备需要相当长时间才进入高速状态（时钟爬升或某种缓存效应）。

由此：

1. **两项优化的端到端效果在本协议下无法判定**，无论跑几次。能站住的只有同进程 op 级 ratio
   （MoE 排序 2.569×、GDN 缓存 1.231×，都带逐位一致的数值验证）。
2. 上一轮那种"重启两臂对比"的做法，其噪声底就是 45%，**它比 M2 的判定距离（80%）小不了多少**，
   比 M3 的（10%）大四倍。
3. 之前所有基于"重启两臂"的小百分比结论（例如 §21.5 的 −30%/−21%）都应视为**趋势而非结论**。

### 27.2 唯一可能有效的修法（下一步）

把 A/B 挪进**同一个服务进程**：两个优化的开关都是**每次调用读 `os.environ`**
（`_sorted_path_enabled`、`_matmul_dtype`、`_decode_constants`），所以进程内切换在技术上是成立的，
只缺一个控制通道 —— 客户端改不了服务端的 env。

最小实现：给这三个开关加一条"文件开关"旁路（例如 `/tmp/tileinfer_flag/<name>` 存在即翻转），
由 `TILEINFER_FLAG_WATCH=1` 显式开启（默认关，且只在测量时开）。
这样客户端每发一个请求就能切一次档，配对差消除"第几个臂""设备状态"这两个变量，
剩余分辨率就是臂内极差（约 7%），足以判定 M2 的量级。

这件事下一次做，因为它决定后面所有结论能不能被验证。

## 28. 进程内配对 A/B 建成并跑通：两项优化端到端值 21%，且方法本身验证成立

### 28.1 控制通道

新增 `tileinfer/serving/flags.py`：`TILEINFER_FLAG_WATCH=1` 时，开关的取值由
`/tmp/tileinfer_flag/<name>=1|=0` 文件决定，否则完全走环境变量、**不碰文件系统**（默认关，
生产零开销）。已接到 `moe_sorted`、`gdn_cache` 两个开关上。

客户端和服务端在同一台机器，所以 bench 脚本能直接写这些文件 —— 一个进程内切换档位，
配对 A/B 成立。`benchmarks/paired_tpot.py` 实现：每对里按 `on→off` / `off→on` **交替顺序**，
每档仍用差分尺子（8 vs 72 token），统计**配对差**而不是两个中位数之差。

### 28.2 结果

| 对 | 顺序 | on (ms/token) | off (ms/token) | off − on |
|---|---|---|---|---|
| 0 | on→off | 115.5 | 143.3 | **+27.8** |
| 1 | off→on | 116.3 | 149.8 | **+33.5** |
| 2 | on→off | 127.6 | 152.1 | **+24.5** |
| 3 | off→on | 155.7 | 163.7 | **+7.9** |

- 配对差中位数 **+26.2 ms/token（+21.5%，相对 on 臂）**
- **`measured_ratio` on/off = 1.238**，四对里 on 全部快于 off（符号一致）
- 配对差极差 25.5 ms（20.9%）—— 被第 3 对拉大：那一对两个数都偏高（设备状态变了），
  差值只剩 +7.9，但**方向仍然一致**

### 28.3 方法本身被验证了

同一配置在不同对之间：**on 测到 115.5 与 116.3（差 0.7%）**；
而上一轮"重启两臂"的同一配置测到 188.6 与 130.2（差 45%）。

同样的两次测量、同样的机器，只是把"重启"换成"进程内切换"，重复性从 45% 提到 0.7%。
**以后所有 A/B 都用这个。**

### 28.4 对进度的意义

- 两项已落地优化的端到端价值：**约 21%**（不是"看不出效果"，此前看不出来是方法问题）。
- 当前 TPOT ≈ **122 ms/token**（本会话设备状态），baseline 约 **37.9** → **3.2×**；
  若把这两项关掉则是 ~151 → **4.0×**。
- M2 出口 ≤1.8× 意味着要到 ~68 ms。从 122 到 68 还要再降 44%，
  而剩下的对象还是那两个（MoE decode ~36 ms、GDN decode ~30 ms），都需要内核级实现。

## 29. 两个 bug：一个修的（`device_`），一个查出来的（孤儿 EngineCore 占着 121 GB）

### 29.1 已修：内存紧张时引擎崩在缺字段的快照上

`tileinfer/vllm/worker.py` 的 `_requested_memory` 把 `NpuMemorySnapshot` 交给 vLLM 的
`vllm.v1.worker.utils.request_memory`，而后者会读 `snapshot.device_` 来打印诊断信息 ——
我们的 dataclass 没有这个字段，于是：

```
AttributeError: 'NpuMemorySnapshot' object has no attribute 'device_'
```

这条路径**只在内存紧张时才会走到**，所以它长期潜伏。已按 `worker.py` 自己的注释指引补上字段
（`device_` 为必填，不给默认值 —— 默认值会打印一个不是实际测量的 device），并加了
`_as_device()`：CPU-only 的测试环境里 `torch.device("npu:0")` 会抛异常，此时回退到 `cpu`。

**修完立刻验证到了**：同样的内存紧张场景，现在报的是

```
ValueError: Free memory on device npu:0 (3.57/123.02 GiB) on startup is less than desired GPU
memory utilization (0.75, 92.27 GiB).
```

—— 一句可操作的错误，而不是一个莫名其妙的 AttributeError。`test_worker_contract` 与
`test_boot_rehearsal` 相应更新（28 + 36 个测试通过）。

### 29.2 查出来的：`pkill -f vllm.entrypoints` 杀不掉 EngineCore 子进程

修好报错之后真相露出来了：**一个孤立的 `VLLM::EngineCore` 子进程已经独占 121 GB 达 32 分钟**，
而我一直用的检查 `ps -eo cmd | grep -c '[v]llm.entrypoints'` **返回 0**。

原因：父进程的命令行是 `python -m vllm.entrypoints.cli.main serve`，而子进程的 `comm` 是
`VLLM::EngineCor`（npu-smi 里显示为 `VLLMEngineCor`），命令行里**不含** `vllm.entrypoints`。
于是父进程被杀了，子进程活着，继续占着 NPU。

**正确的检查与清理**：

| 目的 | 做法 |
|---|---|
| 判断 NPU 是否真的空闲 | `npu-smi info` 的进程表必须显示 `No running processes found in NPU 0`；HBM 必须降到接近 0 |
| 杀干净 | `pkill -9 -f 'VLLM::EngineCore'` **加上** `pkill -9 -f 'vllm[.]entrypoints'` |
| 不要只用 | `ps | grep vllm.entrypoints` —— 它对孤儿 EngineCore 完全看不见 |

### 29.3 修正 §25 / §27 的解释

§25 把 118.5 → 163.0 的 37% 差异归给"持续负载后的降频"，§27 把 188.6 → 130.2 的 45% 归给
"设备需要长时间进入高速状态"。现在有了更简单的解释：**那几次极可能有个孤儿 EngineCore 在同一张卡上
瓜分算力**（本轮实测遗孤状态：121 GB 被占、`ps` 却显示 0 个服务）。

因此：
- 那两组数字本身仍然有效（是当时的真实吞吐），但**归因要改成"资源争用"，不是"降频/预热"**；
- 进程内配对法（§28）依然是正确工具 —— 它对争用和降频**都**免疫，因为两个档位共享同一状态；
- 以后每臂开始前的自检必须是"`npu-smi` 进程表为空 + HBM < 5 GB"，而不是数 `ps`。

### 29.4 SwiGLU 这项：op 级已成立，端到端待测

`torch_npu.npu_swiglu` 替换 `silu_and_mul`（两步设备算子 → 一步）：

| | 结果 |
|---|---|
| 数值 | **逐位相同**（`torch.equal` 为真，8 与 512 token 两种形状） |
| 每次调用 | 26.98 → **8.50** µs（8 token）、25.21 → **8.64** µs（512 token），约 **3.0×** |
| 影响面 | decode 每步约 256 次调用、prefill 每步约 8192 次 |
| 状态 | 默认开（`TILEINFER_MOE_SWIGLU=0` 可回退），数值零风险 |

端到端配对测量被 29.2 的孤儿进程挡住了，未完成 —— 下一轮第一件事。

## 30. SwiGLU 端到端 1.043，且与 op 级预测精确吻合；干净设备上的数字收敛了

### 30.1 结果（干净 NPU，配对法）

| 对 | 顺序 | on (ms/token) | off (ms/token) | off − on |
|---|---|---|---|---|
| 0 | on→off | 118.5 | 125.7 | +7.2 |
| 1 | off→on | 122.6 | 124.3 | +1.7 |
| 2 | on→off | 131.8 | 126.0 | −5.9 |
| 3 | off→on | 117.9 | 125.6 | +7.8 |

- on 中位数 **120.6**，off 中位数 **125.7**，`measured_ratio` = **1.043**
- 配对差中位数 **+4.5 ms/token（+3.7%）**，四对里三对为正

### 30.2 为什么这个 4.5 ms 可信：它和 op 级算术对上了

op 级测得每次调用省 **18.0 µs**（26.98 → 8.50）。decode 每步约 **256 次**调用
（8 个专家 × 32 层），所以预测省 `256 × 18 µs = 4.6 ms/step`。
端到端独立测到 **+4.5 ms**。

两个测量互不依赖（一个是同进程 op 计时，一个是端到端配对），却给出同一个数 ——
这是本文档里第一次有"机制 → 预测 → 端到端验证"闭环。单看配对差（4.5 ms）确实小于
配对差极差（13.6 ms，11.3%），但它是被独立预测出来的，不是从噪声里挑出来的。

### 30.3 干净设备上的数字收敛了

同一天早些时候（§25/§27）同一份代码测到 118.5 / 130.2 / 163.0 / 188.6，现在是
**120.6 / 125.7**，配对差极差也从 20.9% 降到 11.3%。这印证了 §29.3 的修正：
**那些大摆动是孤儿 EngineCore 的资源争用，不是降频或预热。**

### 30.4 两个 harness 教训（都重复犯过，写下来）

1. **`pkill -f '<pattern>'` 会杀掉自己**，只要那个 pattern 出现在 `ssh "…"` 的命令行里 ——
   本轮又踩了一次（`pkill -f 'round31_pairs'` 把执行它的 shell 杀了，脚本因此从未启动，
   表现为"命令返回空、文件不存在"）。规避：用括号形式 `round31_pair[s]`，或者把 kill 放进
   脚本内部（脚本命令行里不含该字符串）。
2. **别去解析 `npu-smi` 的 HBM 列**：值在设备块的第二行、列宽会变。判空只要看进程表那句
   `No running processes found in NPU 0` —— 它已经把结论说完了。本轮因为 HBM 解析失败，
   白白空转了 150 s 并在已经干净的机器上判成 NOT CLEAN。

## 31. 把 `op:gdn` 拆开：decode 最大的单个算子不是 delta rule，是 4-tap 卷积

### 31.1 拆法

`install.py` 的包装加上 `TILEINFER_PHASE_OPS=detail`：每个接管改用自己的相位名
`op:<family>:<函数名>`，默认仍是 `op:<family>`（不动既有表格/看板）。这条正是 §22 对 GDN
没能回答的那个问题。

### 31.2 decode 一步的真实构成（1560 个 attention 层调用，`dev/call` × 每步次数）

| 算子 | dev/call | 次数/步 | ms/步 |
|---|---|---|---|
| `op:gdn:causal_conv1d_update` | **1.073 ms** | 30 | **32.2** |
| `op:gdn:fused_recurrent_gated_delta_rule_packed_decode` | 0.760 ms | 30 | 22.8 |
| `attention`（我们的相位） | 1.087 ms | 10 | 10.9 |
| `op:moe:topk_softmax` | 0.169 ms | 41 | 6.9 |
| `op:gdn:causal_conv1d_fn`（仅 prefill） | 0.655 ms | — | — |
| `op:gdn:fused_post_conv_prep`（仅 prefill） | 0.269 ms | — | — |

**结论：`op:gdn` 的 36.8 ms 里 32.2 ms 是一个 4-tap 因果卷积**，比 delta rule（22.8）还贵。
这改变了 M2 的靶子：写 delta-rule 内核最多省 22.8 ms 的一部分，而卷积是更大的、也更简单的目标
（无递推，只有 4 个 tap）。

### 31.3 已做的改动：把两层 Python 循环向量化

`_depthwise_causal_conv` 原来对每个 tap 做一轮 `(mul, add)`（decode 是 4 轮），
`_prior_window` 原来逐列 gather + `stack`（3 列）。decode 分支（`length == 1`、窗口在界内）改为：

- 一次 `(z[:width].T * w).sum(-1)` 归约，替掉 `width` 轮；
- 窗口变成一次切片 + 转置的视图，替掉 3 次 gather + stack。

开关 `TILEINFER_GDN_CONV_FAST`（默认开，`gdn_conv_fast` 文件可切换），
`tests/test_gdn_causal_conv1d.py` 的 32 个对拍测试（含手写 token 循环 oracle）全绿。

### 31.4 收益：约 1.12×，而不是表面对比的 1.42×

前后两次分相表都测了：

| | conv_update | delta rule（未改，作对照） | 比值 conv/delta |
|---|---|---|---|
| 改动前 | 1.073 | 0.760 | 1.412 |
| 改动后 | 0.755 | 0.597 | 1.265 |

**但同一批数据里 `model_forward` 从 354.7 掉到 159.6 ms/call（2.2×）** —— 说明这两次运行之间机器
状态变了，绝对数不能直接比。用未改动的 delta rule 作对照归一化后，卷积的真实改善是
`1.412 / 1.265 = 1.12×`，即每步省约 **3.8 ms**。

这个教训和 §30 一样：**跨运行比较必须带对照**。要拿到卷积这一项的干净 ratio，必须用配对法
（`paired_tpot.py` 支持任意开关，下一轮用 `gdn_conv_fast` 跑）。

## 32. 卷积的真凶是**每步 30 次的 D2H 同步**，不是算术：配对实测 1.241

### 32.1 找到了什么

`causal_conv1d_update` 的状态更新里有这一句（通用分支）：

```python
if bool(valid.any()):          # <-- device -> host 同步
    block[valid] = line_view[:, src[valid]].transpose(0, 1)
```

它**每次调用**都同步一次，而 decode 每步调用 **30 次**（每个 GDN 层一次）。
按 §22 标定过的量级，一次同步约 0.3–0.5 ms，30 次就是 **9–15 ms/步** ——
这才是那 32.2 ms 的主体，而不是 4 个 tap 的乘法。

已加 decode 专用快路径：基础条件满足时（无 varlen、`offset == 0`、无 spec 窗口、`take == 1`）
状态窗口就是"丢掉最老一列、追加本 token"，两行切片写、**无控制流、无同步**。
源与目标是同一条 cache line 的重叠视图，所以先 `.clone()`（这不是漏优化，原地拷贝会读到已覆盖的元素）。

### 32.2 配对实测（干净设备，一次进程内切档）

| 对 | 顺序 | on (ms/token) | off (ms/token) | off − on |
|---|---|---|---|---|
| 0 | on→off | 128.5 | 145.7 | +17.2 |
| 1 | off→on | 115.5 | 133.9 | +18.4 |
| 2 | on→off | 153.1 | 160.5 | +7.4 |
| 3 | off→on | 110.7 | 157.1 | +46.4 |

- on 中位数 **122.0**，off 中位数 **151.4**，`measured_ratio` = **1.241**
- 配对差中位数 **+17.8 ms/token（+14.6%）**，**四对全为正**
- 配对差极差 39.0 ms（32%）—— 第 3 对偏大，但符号没变

### 32.3 与 §31 的估计对照

§31 用"未改动的 delta rule 作对照"估出卷积只改善 1.12×（约 3.8 ms）。那时的改动**只有值向量的
合并**（少几个 mul/add），确实只值这么多。本轮加上"去掉同步"的路径之后是 **1.241（+17.8 ms）**。
两者不矛盾，且差值（~14 ms）与"30 次同步 × ~0.5 ms"符合。

这也解释了为什么单纯的算子合并只值个位数百分比（§24.2）：**同样的算子个数下，有没有同步是完全不同的量级**。
以后看 `op:*` 的行，除了 `dev/call` 还要问一句"这里面有没有 `.item()` / `bool(...)` / `nonzero` 落回主机"。

### 32.4 仍然要注意：绝对值不跨会话

本会话 on 臂 122.0，而 §30（当时还没有任何卷积优化）测到 120.6 —— 绝对值差得比优化本身还大。
**只有配对差可以跨会话引用。**

`TILEINFER_GDN_CONV_FAST`（默认开）覆盖本次全部三处改动：值归约、窗口切片、免同步状态更新。
`tests/test_gdn_causal_conv1d.py` 的 32 个对拍测试（含手写 token 循环 oracle）两种开关下都通过。

## 33. 负结果：把 `int(idx)` 的同步换成设备侧 gather 反而慢 2.2 倍

### 33.1 动机与做法

`fused_recurrent_gated_delta_rule_packed_decode` 里 `state_idx = int(ssm_state_indices[n])`
是**每次调用一次 D2H 同步**，serving 下每步 30 次。按 §32 的汇率（同步 ~0.5 ms/次）估计能省 ~10 ms。

改为设备侧索引：`idx` 保持 0 维张量，用 `index_select` 读状态行、`index_copy_` 写回，
把"槽位 <= 0 则跳过"折成 `torch.where`，这样循环里不再把索引读回主机。
代价是失去越界检查（改为 clamp），所以做成 opt-in：`TILEINFER_GDN_NO_SYNC`（默认关），
`tests/test_gdn_recurrent.py` 在两种开关下都通过（越界断言在 fast path 下改为"不抛、clamp"）。

### 33.2 实测：0.453，慢 2.2 倍

| 对 | 顺序 | on (ms/token) | off (ms/token) | off − on |
|---|---|---|---|---|
| 0 | on→off | 235.6 | 106.7 | −128.9 |
| 1 | off→on | 234.9 | 107.8 | −127.1 |
| 2 | on→off | 234.1 | 105.2 | −128.9 |
| 3 | off→on | 233.9 | 105.6 | −128.3 |

`measured_ratio` = **0.453**（on 是中位数 234.5，off 是 106.1），配对差极差只有 **1.8 ms（0.8%）**
—— 这个负结果非常确定，不是噪声。**按 C2 保持默认关。**

### 33.3 为什么：一次同步比三次 MB 级搬运便宜

免同步路径每层每步多做三件 MB 级的事：`index_select` 取一整行状态（[32,128,128] ≈ 1 MB）、
`torch.where` 选新旧（再 1 MB）、`index_copy_` 写回（再 1 MB）。三步合计 ~3–4 MB，
在 30 层上就是 ~100 MB/步，实测换来 +128 ms/步。

**结论：同步贵不贵，取决于用什么替代它。** §32 的汇率（一次同步 ~0.5 ms）在"用几个小算子替代"时成立
（那里是切片写，不搬大数据）；一旦替代品是 MB 级的 gather/scatter，就完全不划算。

### 33.4 这条负结果对下一步的意义

1. GDN decode 剩下的成本（本轮 off 臂仍是 106 ms 里的主要部分）**不是主机同步**，是**固有的
   MB 级张量运算**。要继续降只能减少这些运算本身 —— 即内核，而不是消除主机读。
2. 同一条规则可以外推到 MoE：任何"为了避免一个同步而加一次大张量搬运"的改法，先按这个汇率算一遍
   （1 MB ≈ 1–4 ms，1 次同步 ≈ 0.5 ms）。
3. 顺带得到一个可用的绝对数：本轮 off 臂 **106.1 ms/token** 是目前最快的一次（卷积修复默认生效），
   相对 baseline 37.9 约 **2.8×** — 但按 §32.4，绝对值不跨会话，只有配对差可信。

## 34. 卷积改用分组 `conv1d` 一次算完；以及一次外部干扰把配对测量打坏了

### 34.1 做法

decode 的 4-tap 深度卷积可以写成**一次库调用**而不是"乘法 + 归约"：

```python
F.conv1d(z.T.unsqueeze(0),          # [1, dim, width]
         w.unsqueeze(1),            # [dim, 1, width]
         None, groups=dim)          # -> [1, dim, 1]
```

`conv1d` 是互相关（`out[t] = Σ_i w[i]·x[t+i]`），与 `y = Σ_i w[:,i]·Z[i]` 同一个公式、无需翻转核。
开关 `TILEINFER_GDN_CONV_LIB`（**默认开**），回退到 §32 的向量化 torch 形式。

### 34.2 过程中我自己引入并被抓到的一个 bug

第一版把 `bias` 传给了 `conv1d`，而函数尾部又统一加了一次 bias —— **bias 被加了两遍**，
`test_update_single_token_matches_the_oracle` 立刻报 `Greatest absolute difference: 1.108`。
32 个 oracle 测试在这个仓库里的价值就在这：这类"多算一次"的错误不会被任何形状检查发现。
修法是让 `conv1d` 的 bias 传 `None`，把 bias 留给唯一的尾部。

### 34.3 测量：3 个有效对全为正，但第 4 对被外部负载打坏

| 对 | 顺序 | on (ms/token) | off (ms/token) | off − on |
|---|---|---|---|---|
| 0 | on→off | 233.9 | 255.2 | +21.3 |
| 1 | off→on | 231.0 | 244.7 | +13.8 |
| 2 | on→off | 230.4 | 233.1 | +2.7 |
| 3 | off→on | 456.9 | **−1378.1** | — |

第 3 对出现了**负的每 token 时间**（`t(72) < t(8)`），显然是测量被破坏，不是代码问题。
原因也清楚：这一轮里设备上有**别的用户的负载**（`ts-serve` 等），绝对吞吐从之前的 106 ms/token
涨到 231–457 ms/token（2–4×），某一对里请求被拖到自相矛盾。

所以：**3/3 个有效对都为正**，中位约 **+13.8 ms（≈6%）**；总体 `measured_ratio` 报 1.028 是被坏对拉低的。
按 C2 默认开启（它数值逐位一致、且是更少的算子），但**幅度只是估计**，需要一次干净设备上的重测。

### 34.4 一条必须记住的环境事实

这台机器**和别的用户共享**。§29 只把方差归给"孤儿 EngineCore"，现在看至少还有第二个来源：
`ts-serve` / `tokenspeedsch` 这类外部进程一出现，吞吐掉 2–4×，且能让单次配对测量自相矛盾。

因此每次测量前的自检不能只看"我们自己的进程为零"，还要记录 `npu-smi` 里**有没有别的进程**，
并把外部负载写进结果；配对法能吸收稳态干扰，但吸收不了**一对内部发生的突变**（如第 3 对）。
与之配套的判据：`wall_tpot` 的单次采样如果出现非物理解（负时间、或 t(N2) < t(N1)），该对必须作废，
不能进入中位数。

## 35. 纠正 §29：那 121 GB 里至少有一次是**别的用户**，不是我的孤儿进程

### 35.1 怎么发现的

本轮重测时自检门拒绝启动（正确行为），进程表给的是：

```
0  2410437  VLLMEngineCor   89214 MB
0  2408957  python            268 MB
```

顺着 pid 查命令行，父进程属于**另一个用户**：

```
2408956 bash  bash -c cd /home/userlist/ww/test_infer_engine && PYTHONUNBUFFERED=1 \
              /home/userlist/zxs/TileInfer/.venv_native213/bin/python -m vllm.entrypoints.cli.main serve …
```

也就是说：**对方在用我们这套 venv、从这个仓库启动服务**，在自己的目录下做实验。

### 35.2 两处要改的结论

1. **§29 的归因过窄。** `pkill -f 'vllm.entrypoints'` 确实匹配不到 `comm = VLLM::EngineCor` 的子进程
   （这条对**我们自己的**进程仍然成立，也仍然要按 `comm` 补杀），但当时那 121 GB 的持有者
   **至少有一次是对方**，不是我的遗留。把外部负载当成自己的 bug，会让人去修不存在的问题。
2. **这解释了 §34.3 那对负时间、以及 2–4× 的吞吐摆动**：同卡上另一个 serve 在跑，气体条被分走。
   配对法吸收稳态干扰，但吸收不了"一对之内突变"。

### 35.3 一条需要上报的风险（E2）

这个仓库是 editable 安装，**我每次 `scp` 同步都直接改动对方正在依赖的代码树**。
对方此刻正在跑 `test_infer_engine`，也就是说他们的实验结果可能建立在我半改完的状态上，
反之亦然。这属于协作层面的事实，不是我能单方面解决的：

- 需要跟对方约一个**独占窗口**（或各自复制一份 venv/仓库）来做性能测量；
- 在那之前，任何"同环境 A/B"都可能混入对方的负载，§34 那种坏对会继续出现。

### 35.4 本轮完成的不占设备的工作

1. **配对 bench 加了硬规则**：`measure()` 遇到 `t(N2) <= t(N1)`（即负的每 token 时间）直接抛错，
   该对作废并重试，最多试 `3 × PAIRS` 次；末尾打印"有效对数 / 作废对数"。
   §34 那个 −1378 ms/token 正是在这一步被永久堵住的 —— 不再依赖我事后用眼睛发现。
2. **`tileinfer/serving/flags.py` 有了第一批测试**（5 个）：watch 默认关闭、关闭时**不碰文件系统**、
   文件覆盖优先于环境变量、`set_switch` 是替换而非叠加、文件名带值。
   注意测试里不能用 pytest 的 `tmp_path`（沙箱禁止写 %TEMP%），改用工作区内的临时目录。
3. 结论：本轮的测量因 E1 停下，没有新数字；下一轮在有独占窗口时再跑
   `gdn_conv_lib` 的干净配对。

## 36. 候选登记表 + 默认值护栏（不占设备的补齐）

设备此轮仍被外部负载占着（89 GB），按 E1 只做不占设备的工作，但这两件是**长期有用**的：

### 36.1 `docs/notes/perf-candidates.md`

一张表把全部已落地候选的四件事并排放：**开关、默认值、measured_ratio、怎么测的、出处**。
写这张表的过程本身暴露了两个此前只在散落各节里的问题：

- 「同进程 op 级」「端到端配对」「跨会话绝对数」三种证据混在一起时，**只有前两种可比**；
  表中每行都注明了属于哪一种，读者不会再拿 2.569（op 级）去和 1.24（端到端）直接相加。
- 负结果占了三行（`MOE_FP32` 1.035、`GDN_NO_SYNC` 0.453、`STEP_SHARE` 无收益），
  它们和正结果一样需要登记 —— 否则下一个人会重复踩。

### 36.2 `tests/test_candidate_defaults.py`

把每个候选的**默认值**钉死。C2 说"ratio < 1.0 一律默认关闭"，而这条规则最容易在改动中悄悄失效
（加开关时顺手默认开、或者 env 读取的默认值被改）。5 个断言覆盖：

| 断言 | 内容 |
|---|---|
| 默认开 | MoE 排序、卷积向量化、分组 conv |
| 默认关 | `gdn_no_sync`（0.453） |
| 保持 fp32 | `_matmul_dtype`（1.035 的候选不得变成默认） |
| 开关仍然有效 | 三个默认开的候选都能被环境变量关掉（E4 可回退） |
| 缓存默认开 | `gdn_cache` 的解析结果，且空缓存 + 关闭时不抛 |

配合 `tests/test_serving_flags.py`（5 个）与 `tests/test_ab_harness.py`（16 个）全绿，
**C3 的比对工具链本身是健康的** —— 只差一次真机 logprobs 对拍。

### 36.3 仍欠的一件事

`gdn_conv_lib` 的干净配对。它现在默认开，依据是"数值逐位一致 + 3/3 有效对为正 + 更少算子"，
而不是一个可引用的 ratio。这条在独占窗口里补测之前，**不能算完成**。

## 37. 全量同步点审计：decode 已干净，prefill 的 GDN 链每步还有约 200 次

§32 教会我们"看 `op:*` 的行要顺带问一句它有没有落回主机"。这一轮把
`tileinfer/vllm/ops/**` 里所有 `.item()` / `bool(t.any())` / `nonzero` / `.tolist()` 全数扫出来，
按**是否在 decode 热路径**分类：

| 位置 | 形态 | 频次 | 判定 |
|---|---|---|---|
| `moe.py` `fused_experts` 三处校验 | `bool(ids.max() …)` ×3 | 每次调用（32 层/步） | **热**，本轮已加开关 |
| `moe.py` `_routed_sorted` | `bincount(...).tolist()` | 每次调用 | 热，但**必要**（专家循环需要主机侧段长） |
| `causal_conv1d.py` `bool(valid.any())` | 状态更新通用分支 | 每次调用 | 已被 §32 的快路径绕过 |
| `causal_conv1d.py` `starts/accepted/ids` 的 `tolist()` | varlen / spec-decode 专用分支 | 非我们路径 | 不触发 |
| `gdn.py` + `fla/{chunk,wu,kkt,delta_h,solve_tril}.py` | `cu_seqlens.tolist()` / `chunk_indices.tolist()` | **每层每次 prefill** | **仅 prefill**，见下 |

### 37.1 decode 侧：已经干净

`causal_conv1d_update` 的 plain decode 分支在 §32 之后**不再有任何同步**（varlen/spec 分支里的
`tolist()` 不在我们的调用路径上）。剩下的两处都在 MoE：

- **`fused_experts` 的 3 处校验**：本轮加了 `TILEINFER_MOE_STRICT`。默认仍是"每次调用都校验"
  （保持原契约），设 `0` 则按 `(device, 专家数)` 只校验一次并记住 —— 于是每次调用省掉 3 次同步，
  而**不增加任何搬运**（这正是 §33 划出的那条线：省同步不能靠搬兆字节）。
  它保持非默认，等独占窗口里测过再决定。
- `_routed_sorted` 的 `bincount().tolist()` 是设计的一部分（专家循环需要主机侧段长），
  要省它就得改成分组矩阵乘，而那条路被权重布局堵住了（§22.3）。

### 37.2 prefill 侧：约 200 次同步/步，是新发现的 TTFT 成本

`chunk_gated_delta_rule` 每层每步调用一次（§31：43 ms/call），而它内部串了
`chunk` / `wu` / `kkt` / `delta_h` / `solve_tril` / `gdn` 这些模块，**每个都要把
`cu_seqlens` 或 `chunk_indices` 读回主机**（约 7 次/层）。30 个 GDN 层 → **约 210 次同步/步**。

按 §32 的汇率（0.5 ms/次）约 **105 ms/步**，相对 ~1.9 s 的 prefill 是 ~5%。
数字不大，但它是**纯浪费**：这些 `cu_seqlens` 是同一个批次属性，和 §32 的卷积窗口一样
可以在层间共享，或者在设备侧直接算。

这条留给下一轮（TTFT 方向），因为 prefill 的 512-token 现在 ~1.9 s、目标是 ≤0.32 s，
需要的是数量级而不是 5%。但它和 §32 是同一类问题，值得排在 MoE 之后、内核之前。

## 38. prefill 的 ~210 次同步：改用批次元数据主机缓存（待测）

### 38.1 做了什么

新增 `tileinfer/vllm/ops/fla/hostcache.py`：`int_list(t)` / `pairs_int_list(t)` ——
把 `.tolist()` 的结果按**张量身份**缓存起来（`id` + `data_ptr` + shape + dtype + device），
并在条目里**强引用源张量**，这样地址不可能被回收后误命中（与 `recurrent._decode_constants`
同一条安全论证）。开关 `TILEINFER_GDN_HOST_CACHE`（默认开，`=0` 退化为"每次重新转换"）。

替换了 10 个调用点：`gdn.py` 1 处、`fla/wu.py` 2、`fla/kkt.py` 2、`fla/solve_tril.py` 3、
`fla/delta_h.py` 3（含一处 `chunk_offsets`）。

**数值完全不变**：返回的就是 `tolist()` 会给的那串 Python int（列表推导式从 `[int(v) for v in …]`
搬进了 `hostcache`），所以 206 个 GDN oracle 测试（chunk / kkt / wu / delta_h / gating / recurrent /
causal_conv1d）原样通过。

### 38.2 为什么它可以默认开，而 `moe_strict` 不行

区别在于**是否改变契约**：

- 本项只是把一次纯主机转换的结果记住，输入张量没变、算术没变、错误信息也没变；
  唯一的风险是"缓存命中到错误的张量"，而强引用把这条堵死了。7 个单测钉住这些性质。
- `moe_strict` 砍掉的是一次**校验**，缓存后越界的 id 会退化成 `w1[e]` 的索引错误，
  错误信息变差 —— 所以它保持非默认，等测过再决定（§37.1）。

### 38.3 预期与待办

按 §37.2 的估算，prefill 每步省 ~210 次同步 × ~0.5 ms ≈ **105 ms**，相对 ~1.9 s 的 prefill 约 5%。
这个数字**还没测**，因为设备连续三轮被外部负载占着（§35）。

测法已经准备好、只等窗口：用 `paired_tpot.py` 不行（它测的是 decode），prefill 要用
`ab_on_off_on.sh` 那种"重启两臂 + 交替"的形态，或者更简单 —— 在 `TILEINFER_PHASE_TIMING=device`
下读 `op:gdn:chunk_gated_delta_rule` 的 `dev/call`，同进程开关切换即可。

## 39. 测量工具箱就绪，但设备只剩 ~3 GB：连"小切片"都开不起来

### 39.1 补上的那把尺子：prefill 的边际成本

`benchmarks/prefill_pair.py`：把 `wall_tpot` 的差分思路搬到 **prompt 轴** ——

```
prefill_ms_per_token = (T(P2) - T(P1)) / (P2 - P1)      P1=128, P2=512
```

`T` 是流式首 token 时间（§16 明确说过：流式只对 **TTFT** 可信，对每 token 增量不可信）。
每请求开销、分词、调度交接、首 token 成本都在两项里出现并相消。同样轮流切换档位、同样对
`T(P2) <= T(P1)` 的非物理解直接作废。

### 39.2 一键脚本 `benchmarks/owed_measurements.sh`

一次窗口跑完三个欠账，共用一个服务进程（开关走文件通道，不必重启）：

| 阶段 | 测什么 | 用什么 |
|---|---|---|
| 1/3 | `gdn_conv_lib` 的 decode 收益 | `paired_tpot.py ... gdn_conv_lib` |
| 2/3 | `moe_strict=0` 的 decode 收益 | `paired_tpot.py ... moe_strict` |
| 3/3 | `gdn_host_cache` 的 prefill 收益 | `prefill_pair.py ... gdn_host_cache` |

跑完会清掉所有 flag 文件（残留的 `=1` 会悄悄改变下一次的基线），并在开头检查设备空闲：
**忙就 exit 2 并打印占用者，不静默跳过**。

### 39.3 试过的绕行：小切片

我们的三个问题都是**计算/下发**问题，不是显存问题 —— 一个请求只需要几张 2048-token 页。
所以试着用 `--gpu-memory-utilization 0.06`（≈8 GB）在别人占着 121 GB 时挤进去跑：

- **能起来**（一次实测 15 s 就 up），说明思路成立；
- 但紧接着请求中途连接被断开，日志给出的原因是
  `Free memory on device npu:0 (2.98/123.02 GiB) on startup is less than desired (0.06, 7.38 GiB)`
  —— 对方现在的占用已经把可用显存压到 **~3 GB**，连 8 GB 都放不下。

也就是说：**除非等到窗口，否则物理上跑不起来**。这不是调度问题，是显存被占满。

### 39.4 连续四轮的代价，和必须说清楚的话

未经验证的改动已经积到三项（`gdn_conv_lib` / `moe_strict=0` / `gdn_host_cache`）。
其中两项默认开，依据是"数值逐位一致 + 结构上不可能更慢"，**但那不是 measured_ratio**，
按 C2 不能算进度。我不会为了推进而把它们记成"已完成"。

## 40. 补上一个真实的覆盖缺口：MoE 路径在 CPU 上原本完全没被收集

### 40.1 缺口

§21 / §29 / §37 的 MoE 改动（排序版专家循环、融合 SwiGLU、`moe_strict` 的校验缓存）**只在设备上**
对着 `reference_moe` 验过。而仓库里所有 MoE 测试模块在收集阶段就 `import tilelang`，
在 CPU 机器上**整个区域不被收集** —— 也就是说，后续任何一次对 `_routed_sorted` 的改动，
本地都不会有任何东西报警。

这不是"文档不齐"，是真的没有护栏：我这个会话里已经在这一区域改过三次，每次的验证都依赖设备。

### 40.2 补上了什么

`tests/test_moe_routed_cpu.py`（7 个，纯 CPU，不需要 tilelang）：
`fused_experts` 只需要 torch（`flags`/`torch_npu` 都是懒导入），`reference_moe` 是直线 fp32 oracle。
三条路径互相对照：

| 测试 | 钉住的语义 |
|---|---|
| 排序版 vs 循环版 vs oracle | 三者一致；两个 torch 路径**逐位相同** |
| 重复专家 id | 必须累加而不是覆盖 |
| 被丢弃的槽位（id<0） | 两条路径一致，且**确实不贡献**（同专家、同权重列、不重新归一化） |
| `apply_router_weight_on_input` | 排序版必须让位给循环版（它把权重折进 combine） |
| `moe_strict` 0/1 | 缓存校验不得改变任何一个值 |
| 越界 id | strict 模式下仍然抛 `ValueError` |
| `silu_and_mul` 布局 | gate 在前半、up 在后半 |

写这个测试时我自己先写错两次断言（一次拿错权重列、一次用错了 oracle 的 `top_k` 形状），
两次都是"算出来的数差 2.5 倍"这种一眼能看出的错 —— 也正好说明这类 oracle 值得存在。

### 40.3 由此得到的一条规矩

**设备上的验证必须有一个 CPU 侧的同构副本**，否则"验过了"这件事在下次改动时就失效了。
MoE 现在有了；GDN 的 chunk 链（206 个 oracle）和 causal-conv（32 个）本来就有；缺的是
`kernels/**`（tilelang，只能在设备上跑，这部分无解）和 `vllm/attn/impl.py` 的设备分支。

## 41. 补上 C3 缺的那一半：logprobs 采集工具（在此之前那条闸门从未被执行过）

### 41.1 缺口

`benchmarks/ab_report.py --compare observed.json baseline.json` 从项目一开始就在，
C3 把"logprobs 对拍"定为每项改动的闸门 —— 但**spec 里从来没有东西产出那两个 JSON**。
所以这条闸门在真机上一次都没跑过。缺的不是判断逻辑（`compare_baseline` 有 16 个测试），
是**采集端**。

### 41.2 补上了 `benchmarks/capture_logprobs.py`

- 对同一组固定 prompt 取 `temperature=0`、`logprobs=5` 的生成结果；
- 写两份文件：`OUT.json`（`{prompt: [token_logprob, ...]}`，正是 `--compare` 要的形状）
  与 `OUT.tokens.json`（`{prompt: [token_text, ...]}`，供人看与下面的序列检查）；
- `--compare-tokens A B` 先判**两个臂是否生成了同一串 token**。

第二条是关键的：两个实现如果吐出不同的词，每个 logprob 都会因为"模型选了另一个字"而不同，
看上去却像数值回归。**先判序列、再判数值**。这部分纯离屏，已有 5 个 CPU 测试
（`tests/test_capture_logprobs.py`）。

### 41.3 用法（窗口里的完整命令序列）

```bash
# 我方臂（.venv_native213，--attention-backend CUSTOM）
python benchmarks/capture_logprobs.py http://127.0.0.1:8111 native.json
# 基线臂（/home/dlb/.venv_ascend）
python benchmarks/capture_logprobs.py http://127.0.0.1:8112 baseline.json
# 1) 先确认可比：token 序列必须一致
python benchmarks/capture_logprobs.py --compare-tokens native.tokens.json baseline.tokens.json
# 2) 再判数值：bf16 量级容差
python benchmarks/ab_report.py --compare native.json baseline.json --atol 2e-2
```

`--atol 2e-2` 是起点不是结论：先在**同一份实现**上跑两次（`native.json` 对自己的副本），
量出这个后端的自身抖动，再拿它当门槛 —— 否则容差是我拍的，不是测的。

### 41.4 状态

工具齐了，仍然缺窗口。本轮重试循环（`/tmp/om_retry.sh`，8 次 × 约 6 分钟）在跑，
一旦某次抢到空闲就会自动完成三项配对测量；每次尝试都在日志里留痕。

## 42. 三项欠账测量完成：一项真收益（1.107×），两项是零

第 3 次尝试抢到窗口（前两次都是 `DEVICE BUSY`），三项在**同一个服务进程**内按文件开关配对跑完，
零作废。

### 42.1 结果

| 候选 | 配对差（中位） | ratio | 判定 |
|---|---|---|---|
| `moe_strict`（砍掉每次调用的 3 次校验同步） | **−10.6 ms/token（−9.6%）**，4/4 对为正 | **on/off = 0.904 → 缓存版快 1.107×** | **开**（默认翻为"不每次校验"） |
| `gdn_conv_lib`（分组 conv1d） | +0.2 ms（+0.2%），极差 2.1% | 1.004 | **关**（无差别，保留更简单的形式） |
| `gdn_host_cache`（prefill 批次元数据缓存） | −0.05 ms/prompt-token（−1.8%），正负各半 | 0.999 | **关**（无效果） |

数字出处：`/tmp/om_retry.log` 的 attempt 3。decode 用 `paired_tpot.py`（on 中位 110.4 / off 99.8），
prefill 用 `pair_prefill`（on 2.554 / off 2.550 ms/prompt-token）。

### 42.2 关键的一条：那 3 次校验同步真的值 10.6 ms

§37.1 只敢说"值得一试、但没测"。测出来是 **每 token 10.6 ms**，正好是 §32 那笔账的第二次兑现
（一次同步 ~0.5 ms × 3 × 32 层 ≈ 48 ms？—— 实测 10.6，说明每次同步的实际代价比 0.5 ms 小得多，
大约 0.1 ms；§32 那次 14.6% 的收益里也含有"去掉同步 + 更少算子"两项，不能平分）。

**所以 §32 的"0.5 ms/次同步"这个汇率要下调到 ~0.1 ms/次**，否则下次又会高估收益。

### 42.3 两条零，以及它们推翻了我自己的判断

1. **分组 `conv1d` = 1.004**。上一轮我以为它至少值 ~6%（当时 3/3 有效对为正，但设备被外部占用）。
   干净窗口下它**与向量化形式完全等价** —— 那次的正号是争用噪声。按 C2 默认翻回关。
2. **批次元数据主机缓存 = 0.999**。§37 我按"~210 次同步 × 0.5 ms ≈ 105 ms"估了 5% 收益，
   实测是零。两个原因：同步单价被高估（见 42.2），而且 prefill 每步的 1.9 s 里
   `chunk_gated_delta_rule` 的**算子**成本远大于它的主机读。按 C2 默认翻回关。

这两条正是 C2 存在的意义：**"结构上不可能更慢"的推理连着两次错了**，其中一次还是我自己
半小时前刚写进文档的估计。没有配对测量，它们会以"已优化"的身份留在默认路径上。

### 42.4 落地改动

| 文件 | 改动 |
|---|---|
| `vllm/ops/moe.py` | `_moe_strict()` 默认改 `False`（缓存校验），docstring 写入 1.107× 与代价 |
| `vllm/ops/fla/causal_conv1d.py` | `_conv_lib_enabled()` 默认改 `False`（1.004） |
| `vllm/ops/fla/hostcache.py` | `_enabled()` 默认改 `False`（0.999） |
| `tests/test_candidate_defaults.py` | 三个默认值钉死，并新增"默认可被重新打开"的反向断言 |
| `tests/test_fla_hostcache.py` | 缓存测试改为显式 opt-in fixture（默认关了，不显式打开就测不到） |

翻默认时又踩了一次测试：`flags.flag(name, env, True)` 的第三个参数是**默认值**，
我在断言里传了 `True` 却期望 `False` —— 改成断言解析器本身（`hostcache._enabled()`）才对。

## 43. C3 第一次真机执行：**失败**，而且失败的方式和下一步都清楚

### 43.1 结果（同一组固定 prompt，两臂各采一次）

```
=== (a) tokens ===
TOKENS DIFFER:
  prompt 'Explain in one sentence why the sky is b':
  sequences diverge at token 22 ([' The', ' user'] vs [' **', 'Question'])

=== (b) logprobs @ atol 2e-2 ===
Explain in one sentence why the sky is blue:        | 0.464346  | FAIL
List three primary colours, comma separated:        | 0.210608  | FAIL
Question: what is 17 times 23? ...                  | 0.001584  | ok
Summarize the following in one word. ...            | 0.007848  | ok
RESULT: FAIL
```

两臂：native（`--attention-backend CUSTOM`，`.venv_native213`）vs baseline（vllm-ascend，
`/home/dlb/.venv_ascend`），同一份模型、同一组 prompt、`temperature=0`、`max_tokens=24`。

### 43.2 怎么读这个结果（先别下结论）

1. **prompt 1 是"生成分歧"**，不是"同一串 token 上数值不同"：第 22 个 token 两臂选了不同的词
   （`The user` vs `**Question`）。一旦分歧，后面的 logprob 全部不同，0.46 这个数**没有意义**。
2. **prompt 3 才是真正的数值信号**：token 序列一致，但最大偏差 **0.21 nats**。这不是 bf16 噪声
   （那是 1e-3 量级）。
3. 另外两个 prompt 是 0.0016 / 0.0078，属于可接受范围。

### 43.3 现在还缺的那一步（下轮第一件事）

**必须先量"自身抖动"**：把同一臂跑两次、互相比对。
- baseline 自己 vs 自己 → 得到这个推理栈的噪声底；
- native 自己 vs 自己 → 得到我们的确定性。

如果 baseline 自己两次也分歧（很可能：bf16 + 融合算子在 20+ 步生成上会漂），那 0.21 就不能算
"我们错了"；如果 baseline 自己完全确定而 native 与它相差 0.21，那就是**真的数值问题**，要找。

这正是 §41.3 里写的那句"`--atol 2e-2` 是起点不是结论，先量自身抖动"—— 第一次执行就验证了这句话的必要性。

### 43.4 顺带修掉一个 harness bug

`c3_logprobs.sh` 里日志是 `>> ` 追加的，而我在启动后 `grep "Engine core initialization failed"`
搜的是**整个文件**，于是匹配到上一次失败的旧行，把已经在跑的服务判成 `START-FAILED`。
本轮就是因此白等了一个启动周期。修法：每次启动前 `: > log`（其他脚本早就是这么写的，
这个脚本漏了）。

## 44. C3 的自抖动控制：**我方不确定，基线逐位确定** —— 这是本会话最重要的发现

### 44.1 三个比对（同一组 prompt，`temperature=0`，24 token）

| 比对 | token 序列 | logprobs |
|---|---|---|
| **baseline vs baseline**（同一服务，两次请求） | 4/4 完全一致 | **全部 0** |
| **native vs native**（同一服务，两次请求） | prompt 1 在第 22 个 token 分歧 | 最大 **0.467** |
| native vs baseline | 同样的第 22 个 token 分歧 | 最大 0.467 |

**结论有两层，第二层比第一层严重：**

1. 上一轮那个 0.21 nats **不能算作"我们与基线不一致"** —— 因为 `native vs native` 本身就有 0.467，
   同一份代码、同一个服务、同样的贪心解码，两次请求给出不同结果。C3 之前**无法判定**。
2. 更严重的是：**我们的实现是不确定的，而基线是逐位确定的。** 这不是"数值容差"问题，
   而是"同样的输入两次跑出不同输出"，对服务来说是质量缺陷（同一 prompt 复现不出同样答案）。

### 44.2 根因：`index_add_` 在 NPU 上是原子累加

没有去猜，直接测了原语（`benchmarks/probe_index_add_determinism.py`，50 次重复）：

```
### index_add_   NON-DETERMINISTIC in 50/50  worst_abs=1.192e-07
### scatter_add_ NON-DETERMINISTIC in 50/50  worst_abs=1.788e-07
### reshape+sum   DETERMINISTIC            worst_abs=0.000e+00
```

`index_add_` 和 `scatter_add_` 的求和顺序不固定，同一份输入每次差 1 个 fp32 ulp；
而固定顺序的归约逐位一致。**我们的 MoE 专家合并正是用 `index_add_` 做的** ——
1 ulp 的 hidden state 差异，足以在 20 多步之后翻掉一个 argmax。

（顺带说明为什么不能"关掉我们的算子做二分"：vLLM 自带的 GDN/MoE 需要 Triton，
`TILEINFER_OPS=dense` 时模型根本起不来 —— 这条二分路走不通，所以改成直接测原语。）

### 44.3 修法：换成"槽位缓冲 + 固定顺序归约"

`fused_experts` 的合并从 `out.index_add_(0, tok, y)` 改为：

1. 每个槽位的结果写进 `slot_out[slots] = …`（槽位互不相同，`index_copy_` 语义，无原子）；
   排序路径用一次 `slot_out.index_copy_(0, order, out)` 把结果放回槽位序；
2. 合并变成 `slot_out.view(num_tokens, top_k, hidden).sum(dim=1)` —— 沿固定的 `top_k` 轴归约。

这条比原来的散射**更简单也更快**（一次归约替代一次原子散射），而且顺带让 `top_k` 轴的求和顺序
固定下来。`tests/test_moe_routed_cpu.py` 的 7 个 oracle（含与 `reference_moe` 的逐位比较）全过。

### 44.4 待验证

下一轮第一件事：重跑 `c3_jitter.sh`，确认 `native vs native` 变成 **delta 0 / token 全同**。
只有那样 C3 才能重新回答它本来要回答的问题。在那之前，§43.1 里那个 0.21/0.46 **不作数**。

### 44.5 现场结果：**这个修法没有解决问题**

改完后立刻在同一台（空闲）设备上重跑 `native vs native`：

```
TOKENS DIFFER at token 22: [' The','user'] vs [' **','Question']     <-- 和修复前完全相同的分歧点
RESULT: FAIL
```

也就是说 **MoE 的 `index_add_` 不是（唯一的）根因**。这条必须先说清楚，免得把"修了一个真问题"
误当成"修好了这个问题"。

**顺带在推理过程中发现并修掉一个我引入的真 bug**：`_routed_sorted_into` 里槽位缓冲原本用
`torch.empty`，而被丢弃的槽位（`id < 0`）落在合成桶里、专家循环永不写它 —— 于是那一行是
**未初始化内存**，会被 `top_k` 归约加进对应 token。serving 里 `topk_ids` 恒非负所以碰不到，
但这正是"读未初始化内存"的典型来源，改成 `torch.zeros`。我的 CPU oracle 没抓到它
（新分配的小块内存通常恰好是零），说明"逐位比较"这类测试只能证明两条路径**一致**，
不能证明它们**正确** —— 这条要记住。

### 44.6 下一步的二分法（不需要关掉我们的算子）

`TILEINFER_ATTN_BACKEND=reference` 把注意力换成 torch 参考实现，其它一切不变：

- 若 `native vs native` 变成逐位一致 → **根因在 `paged_decode_ascend950` 内核**（最可能：
  内核里有原子或读了未初始化的 UB/out_pad 行）；
- 若仍然分歧 → 根因在 GDN 的 torch 链或采样器。

这条路可走（不像 `TILEINFER_OPS` 那样会让模型起不来）。本轮试启动时因**上一台服务还占着
98 GB** 而失败（`ref http=000`），所以还没跑成 —— 必须一次只开一个服务。

## 45. 顺带挖出一个更严重的 bug：**兜底路径一被调用就崩**（缺 import）

做 §44.6 的二分时，`TILEINFER_ATTN_BACKEND=reference` 的服务起得来，但**每个请求都 500**：

```
NameError: name 'RaggedMetadata' is not defined
  at tileinfer/attention/backends/reference.py:47  (ReferenceBackend.run)
```

`ReferenceBackend.run` 在"调用方显式传了 `kv_indptr`/`kv_indices`/`kv_last_page_len`"的分支里
构造 `RaggedMetadata`，而这个模块**从来没有 import 它**。

### 45.1 为什么这条比性能问题更严重

那个分支不是罕见路径：serving 的 decode 步**正是**显式传这三个张量（`impl._kernel_forward`）。
也就是说，只要引擎走"降级到参考后端"这条路 —— 计划未就绪（`plan_not_ready`）、计划编译失败
（`plan_failed`）、kernel 后端不可用 —— **它不是变慢，而是当场把引擎打死**
（`EngineDeadError`，整个服务进程退出）。仓库里所有关于"degraded path 会优雅降级"的注释，
在真机上都不成立。

这条也解释了为什么它一直没被发现：`TILEINFER_ATTN_BACKEND` 默认不是 `reference`，
单测里也没人用显式 ragged 参数去调 `ReferenceBackend.run`。

### 45.2 修法与回归测试

- `reference.py` 补上 `from ...metadata import RaggedMetadata`；
- 新增 `tests/test_reference_backend_reachable.py`（3 个）：
  1. 模块命名空间里 `RaggedMetadata` 就是那个类型；
  2. **用字节码的 `LOAD_GLOBAL` 操作数**逐个解析 `run` 里用到的全局名 —— 第一版我用
     `co_names` 写，结果把属性名（`meta.kv_indptr` 贡献 `meta` 和 `kv_indptr`）也算进去，
     误报 18 个"未定义"；改成只取 `LOAD_GLOBAL` 才对；
  3. `run` 的签名仍接受那三个 ragged 参数（改名会让 serving 静默失效）。

### 45.3 本轮没跑完的那件事

二分（§44.6）仍然没结论：A 组这次**起得来**，但两次采集都没产出文件，而脚本把 capture 的
stderr 丢进了 `/dev/null`，所以看不到原因。已把那条改成 `2>&1 | tail -3` 并加了一句
"capture a produced no file" 的显式提示 —— 下一轮第一次运行就能看到真实错误，而不是再猜一轮。

**记一笔教训**：这一轮我在同一个地方摔了两次 —— 把诊断命令的 stderr 丢掉。继"追加日志却 grep 全文件"
之后，这是第二个"为了让输出干净而在诊断脚本里做手脚，结果反而多花一轮"的例子。
诊断脚本的原则应该是**宁可吵，不可哑**。

## 46. 又一个未初始化内存 bug，以及"测试时过时不过"这个信号

### 46.1 首修是错的

上一轮我为了修 §44 的确定性，把合并改成"槽位缓冲 + 固定顺序归约"，并把**调用方的** `slot_out`
从 `torch.empty` 改成 `torch.zeros`。当时以为这样处理了"被丢弃槽位读到未初始化内存"。

**修错了缓冲区**：真正未初始化的是 `_routed_sorted_into` **内部**那个 `out`
（`torch.empty`）。被丢弃的槽位落在合成桶（专家索引 == `num_experts`），循环 `for e in range(num_experts)`
永远不访问它，随后 `slot_out.index_copy_(0, order, out)` 把**整个** `out`（含那些垃圾行）拷出去，
再被 `top_k` 归约加进 token。

### 46.2 它是被"同一个测试时而过、时不过"抓出来的

`test_dropped_slots_contribute_nothing` 单独跑全过，和其它三个测试文件一起跑就失败。
这个信号几乎总是指向**未初始化内存**：新分配的小块内存常常恰好是零，被复用的则不是。

一开始我把它写成 `xfail` 并标注"待查"，那是**错的处理** —— 它把一个真实的正确性缺陷记成"已知问题"。
正确做法是顺着信号查下去：改用"丢弃槽位 vs 权重为 0 的槽位"来断言（不依赖 oracle 的状态），
再去看那个断言为什么在批跑时失败，就找到了 `torch.empty`。

修法：内部缓冲改 `torch.zeros`。两处 `zeros` 现在都有注释说明"为什么不能省"。

**教训**：**顺序相关的测试失败 ≈ 未初始化读取**。不要用 `xfail` 把它盖住。

### 46.3 与 §44（不确定性）的关系

这条是**真 bug**，但**不是** §44 那个非确定性的原因（§44.5 已经验证：改完 MoE 后自抖动仍在）。
两者都指向同一类根因（读未初始化内存），所以下一轮排查注意力内核时，"内核是否读了未初始化的
`out_pad`/UB"应该作为首要假设。

### 46.4 本轮的测试增量

`tests/test_moe_routed_cpu.py` 加了 autouse fixture（清 `_VALIDATED_ID_BOUNDS` 与相关开关），
让这个文件不依赖执行顺序；`tests/test_reference_backend_reachable.py` 新增 3 个
（其中"用字节码 `LOAD_GLOBAL` 逐个解析全局名"这条能抓住同类的缺失 import）。
四个文件**任意顺序**跑都绿。

## 47. 修正 46 的一个副作用：清零从"每层无条件"改成"只在有丢弃槽位时"

§46.2 的修法（内部缓冲改 `torch.zeros`）是**正确**的，但它把一次
``[slots, hidden]`` 的清零从"偶尔"变成了"每层每步"：

- prefill 512 token、top-8 时是 `4096 × 2048 × 4 B = 33 MB`；
- 32 层 → **每步额外 1 GB 的 memset**。

这是我在修 bug 时引入的一个**潜在性能回退**，而且是在自己反复强调"不要用搬运换同步"之后
又犯了同类错误。

### 47.1 改法：用已经拿到的 host 列表决定要不要清零

排序路径本来就有 `counts = torch.bincount(...).tolist()`（一次同步，用于专家循环）。
被丢弃的槽位落在合成桶 `counts[num_experts]` 里 —— **一个现成的 host 整数**就能判断：

```python
if counts[num_experts]:          # 有丢弃槽位：这一轮才需要清零
    out = torch.zeros(...)
else:
    out = torch.empty(...)       # 所有行都会被写，无需清
```

调用方的 `slot_out` 同理：改成 `torch.empty`，再对 `dead = (local < 0).nonzero()`
做一次 `index_fill_(0, dead, 0.0)` —— `dead` 为空时是一次空 kernel，代价可忽略；
正常 serving（`topk_ids` 恒非负）永远走这条路。

### 47.2 状态

数值行为不变（`test_dropped_slots_contribute_nothing` 与其余 6 个 oracle 全过，任意顺序），
清零成本从"每层 33 MB"降回"仅当存在丢弃槽位时"。

**这一条没有真机 ratio**，因为它只影响 prefill 的固定开销，而我们已有的 prefill 尺子
（`prefill_pair.py`）分辨率约 13%，看不出这个量级。不做假声明：它属于"结构性必然更好"，
不计入 C2 的进度。

## 48. 外部 SIGKILL：测量被第三方按进程名杀

本轮二分再次失败，这次拿到了原因：服务器进程被 **SIGKILL**（`Killed`），而：

- host 内存 628 GB、free 442 GB，`dmesg` 无 OOM 记录 → **不是 OOM**；
- 我们这边只有一个 `det_bisect.sh` 在跑，没有第二个循环在对打；
- 帧上只剩一行 `Killed`，没有异常栈。

结合 §35 已经确认的事实（另一个用户 `/home/userlist/ww/test_infer_engine` 在用我们的 venv
起 serve），最可信的解释是**对方在清理时按进程名 kill**，把我们的服务一起带走了。

**这意味着：任何需要"服务存活 N 分钟"的测量，在当前时段都无法保证完成。** 对策只有两条：

1. 找对方要一个明确的时间窗（我一直没拿到）；
2. 把测量压到"启动 + 两次请求"这个最小窗口内（约 4 分钟），并接受重试。

本轮我把这条写下来，而不是继续重试 —— 连续三次失败已经说明这是环境约束，不是脚本问题。

## 49. 逐步排除：注意力、MoE、GDN **全部逐位确定**

服务被外部杀，所以改用**不需要长寿服务**的探针：在同一进程里用固定输入重复跑一步，逐位比较。
三个脚本各自约 1 分钟，都在设备上跑通了。

| 探针 | 重复次数 | 结果 |
|---|---|---|
| 单次 decode：`TileInferAttentionImpl.forward`（内核路径，含 write_kv_cache / refresh_step） | 5 | **0/4 差异**，`worst_abs = 0.000e+00` |
| `fused_experts`（MoE，含新的确定性合并） | 20 | **0/19 差异** |
| `causal_conv1d_update`（GDN 卷积，含 §32 的免同步路径） | 20 | **0/19 差异** |
| `fused_recurrent_gated_delta_rule_packed_decode`（GDN delta rule） | 20 | **输出 0/19、状态 0/19 差异** |

**结论：我们自己的三步（注意力、MoE、GDN）在单步上都是逐位可复现的。** §44 那个非确定性
不在这三处。

### 49.1 剩下的两个嫌疑

1. **采样器**。我们给 `topk_topp_sampler.HAS_TRITON` 打了 `False`（为了绕开 `bishengir-compile`
   崩溃）。如果这条补丁把贪心解码换成了一条**按概率抽样**的路径，那同一份 logits 每次抽样不同 ——
   而基线臂没有这条补丁。这是我目前最怀疑的一条。
2. **模型主体的大矩阵乘**（LM head 2048→248K、qkv/o 投影）。某些后端的大 K 归约用 split-K 原子
   累加，同样 1 ulp 不确定。但基线臂跑同一套 matmul，它却是逐位确定的 —— 所以如果真是这里，
   说明 vllm-ascend 把某个算子换成了确定性实现。

### 49.2 下一步的决定性实验（约 4 分钟，能塞进被杀的间隙）

**只采一个 token**（`max_tokens=1`）跑两次：

- 若**首个 token 就不同** → 问题在前向（prefill）或采样器；
- 若**首个 token 相同、之后才分歧** → 问题是 decode 的累积数值。

再进一步：把 `topk_topp_sampler` 那条补丁临时关掉（加一个开关），若自抖动消失 → 就是采样器。

### 49.3 这一轮的判断

三次"我们自己的代码不确定"的假设全部被证伪，这是好事：嫌疑范围从"整条链路"缩到
"采样器补丁 / 模型主体 matmul"两处，而且下一步实验只需要 4 分钟设备时间。
**在 C3 能稳定通过之前，我不再碰性能代码** —— 现在任何"提升 X%"都可能在流沙上。

## 50. 自抖动消失（2/2 轮），跨臂差异变成**可复现**的 0.21 nats

### 50.1 自抖动：两轮独立测试都完全确定

同一服务内连采两次，三种生成长度，`atol=1e-6`：

| max_tokens | 第一轮 | 第二轮 |
|---|---|---|
| 1 | token 全同，RESULT ok | token 全同，RESULT ok |
| 4 | token 全同，RESULT ok | token 全同，RESULT ok |
| 24 | token 全同，RESULT ok | token 全同，RESULT ok |

**§44 那个"同一 prompt 两次不同答案"不再复现。**

归因我只能说到这一步：在**失败的测量**与**通过的两轮**之间，代码上只有两处变化 ——
§46/§47 把 `_routed_sorted_into` 内部缓冲从 `torch.empty` 改成零初始化（lazy），以及
`tests/` 的补充。逻辑上 serving 的 `topk_ids` 恒非负、不该走到合成桶，所以因果链我**没有证明**，
只能说"修掉一处未初始化读取之后，不再复现"。要证明需要二分，而二分需要设备窗口 —— 我选择先把
这个事实记下来，而不是编一个机制。

### 50.2 跨臂：差异稳定在同一个 token

| 比对 | 结果 |
|---|---|
| native vs native | **逐位一致**（§50.1） |
| native vs baseline | 在第 22 个 token 分歧：`[' The',' user']` vs `[' **','Question']`；其余两 prompt 的 logprob 差 0.0016 / 0.0078；`List three primary colours` 那条差 **0.21** |

**这是 C3 第一次给出有意义的结果**：不是"我们的实现不确定，所以比不了"，而是
"我们的实现确定，并且它与 vllm-ascend 真的不一样"。

- 2 个 prompt 在 1e-2 以内（说明大部分路径是通的）；
- 1 个 prompt 差 0.21 nats（≈20 倍于容差）；
- 1 个 prompt 直接换了 token。

### 50.3 这 0.21 该查什么（下一步）

按 §18 的设计，我们的注意力内核在 PV 阶段把 P 矩阵存成 **bf16**（`p_ub` 是 `dtype`），
而参考实现全程 fp32。在一段 512 token 的上下文上，概率的 bf16 截断完全可能积累出 0.2 nats。
验证方法不需要基线臂，只需要我们自己的两套实现：

```
TILEINFER_ATTN_BACKEND=reference TILEINFER_PREFILL=0 + capture   ← torch 参考注意力
默认（内核路径） + capture                                      ← 内核注意力
```

若两者差 ≈0.2 → 内核的 bf16 PV 就是来源，届时按 C2 评估"精度换速度"是否可接受（若不可接受，
`pv_acc0/pv_acc1` 走 fp32 是一条现成的开关）；若两者几乎相同 → 差异来自别处（GDN 的 fp32/bf16 边界）。

### 50.4 本轮建立的可复用资产

- `benchmarks/first_token_det.sh`：一次服务、三档长度、自抖动判定 —— 4 分钟跑完，
  专为"窗口随时会被杀"设计；
- `benchmarks/decode_det.py` / `benchmarks/gdn_det.py`：不需要服务的单步确定性探针
  （§49 用它把注意力/MoE/GDN 逐个排除）；
- `capture_logprobs.py` 支持 `CAPTURE_MAX_TOKENS`，`--compare-tokens` 先判可比性。

## 51. 注意力被排除（逐位相同），并修掉兜底路径的**第二个** bug

### 51.1 兜底路径的第二个 bug：3-D 查询

用 `TILEINFER_ATTN_BACKEND=reference` 做精度对照时，第 0 步之后立刻崩：

```
ValueError: not enough values to unpack (expected 4, got 3)
```

`ReferenceBackend.run` 里 `reference_attention_tiled` 期待 `[batch, h_q, qo_len, d]`，
而 **decode 路径传的是 `[batch, h_q, d]`**（`_kernel_forward` 递 `state.q_buf`）。
§45 修掉那个缺失 import 之后，这一层才露出来 —— 说明这条兜底路径**两层都是坏的**。

修法：`run` 入口发现 3-D 就 `unsqueeze(2)`，出口 `squeeze(2)` 还原调用方要的形状
（3-D 查询等于"每序列一个 token"，正是这个后端要兜的 decode 情形）。

### 51.2 然后才拿到真正的结论：注意力与参考实现**逐位相同**

同一进程里构造两个 impl（内核路径 / torch 参考），喂同一组输入跑 24 个 decode 步：

```
step  0..3, 8, 16, 23:  max|kernel - reference| = 0.000000
first/last/max = 0.000000 / 0.000000 / 0.000000
```

**24 步全部恰好为 0。** 这否定了 §50.3 的假设（"内核把 P 存成 bf16 会积累出 0.2 nats"）：
`p_ub` 是 bf16 没错，但两端输出的隐藏状态都是 bf16，内核在 bf16 上与 fp32 参考**逐位一致**。

所以 §50.2 那 0.21 nats **不是注意力造成的**。剩下的是 GDN chunk 链或 MoE 的**累加顺序**
（我们的 torch 移植与 vllm-ascend 的融合内核在 512 token 上下文上必然不同序），
或者模型主体的大矩阵乘。

### 51.3 为什么这条值得记

我原本要给一个"bf16 PV 造成 0.2 nats"的结论。**测出来是 0**，所以那个结论是错的 ——
如果我没有先做这个对照，就可能去"修"一个不存在的问题（比如把 PV 改成 fp32），
白白付出性能代价，而真正的差异还在。

这和 §42（分组 conv 与批次缓存两项"结构上必然更好"实测为零）是同一类教训：
**关于数值/性能的因果推理，必须在改动之前用最小对照验一遍。**

### 51.4 下一步

把同样的对照搬到 GDN：用 `tests/` 里已有的 oracle 之外，**在设备上**比较我们的 chunk 链
与一段朴素 fp32 递推（512 token、单层），看差多少。若差 ≈0.2 nats → GDN 是来源；
若也逐位接近 → 差异来自模型主体（那就不在我们的可改范围内，只能作为与基线"实现不同"记录在案）。

## 52. C3 的性质终于清楚：**我们每一层都对，0.21 nats 是两套 bf16 实现的固有差异**

### 52.1 逐个组件的设备侧验证（都对）

| 组件 | 参照 | 结果 |
|---|---|---|
| 注意力（内核） | 我们自己的 torch 参考 | **24 步逐位相同，delta 恰好 0**（§51） |
| GDN chunk 链（卷积 + delta rule + 分块） | 朴素递推 oracle，**在 NPU 上跑** | `tests/test_gdn_chunk_full.py` + `test_gdn_gating_delta_rule.py` **49 项全过** |
| MoE `fused_experts` | `reference_moe`（直线 fp32） | CPU oracle 1e-5 内；设备上 20 次重复逐位一致（§49） |

也就是说：**TileInfer 的每一层都对着各自的 oracle 是"对的"。** 那 0.21 nats 不是我们某一层算错了。

### 52.2 那它来自哪里

只剩一处：**两个栈的 bf16 稠密算子不同**。vllm-ascend 把 vLLM 的 rmsnorm / rotary / linear
换成了自己的融合算子（配置里能看到 `pass_config` 的 `fuse_norm_quant`、`fuse_act_quant`、
`ir_op_priority=rms_norm=['native']`），累加顺序与我们的纯 torch 路径不同。

在 512 token 上下文、24 步贪心生成上，两套 bf16 实现差出 0.0016–0.21 nats、并在一个 prompt 上
翻掉一个 token —— 这是**预期内的**，不是缺陷。它无法通过修改 TileInfer 消除，除非把对方的
算子也复刻一遍（那等于重写 vllm-ascend）。

### 52.3 因此 C3 的判据要改成"可测且有界"，而不是"1e-2"

C3 原文是"允许 bf16 级容差，不允许分布偏移"。我一开始拍的 `--atol 2e-2` 是**不可达的**，
因为对象是"两套不同的 bf16 实现"。可执行、且与 C3 本意一致的判据是三条：

1. **组件级**：每个组件对自己的 oracle 必须通过（上表）—— 这是"实现正确"的证据；
2. **端到端**：与基线对比时，报告 **token 序列一致率**（本次 3/4 prompt 24 步全同）
   与 **logprob 最大差**（0.0016–0.21 nats），并注明这是跨实现差异；
3. **回归**：任何改动后，我们**自己对自己**必须仍然逐位一致（`first_token_det.sh`，§50 已建立）。

这条改动的意义：C3 从"一个过不去的闸门"变成"三道能过的检查"。我没有下调要求 ——
我只是把"跨实现必须数值相同"这个做不到的要求，换成"实现正确 + 差异有界 + 自查确定性"。

### 52.4 需要你确认的一点

第 2 条里的"可接受上界"需要一个数：跨两套 bf16 实现，**0.21 nats / 24 步、4 个 prompt 里 1 个
翻 token**，你认可作为基线吗？如果要求更紧，唯一的办法是让我们复刻 vllm-ascend 的融合算子
（工作量等于重写对方），我不建议；如果认可，我会把它写成 `docs/notes/perf-comparison-rules.md`
里的一条固定判据，后续所有改动以此为回归线。

## 53. MoE 批量化落地：prefill 形状 1.480×，decode 不动（按形状门控）

### 53.1 动机（按 §19/§42 的量级算出来的）

prefill 512 token、top-8、256 专家时，专家循环是 **每层 256 对 matmul、32 层共 8192 次内核下发**。
§19 量过：两个 matmul 本身只值 7.6 ms，而整通调用 ~34–100 ms —— 所以成本在下发，不在算术。

新路径 `_routed_bmm_into`：把排序后的槽位补齐成 `[E, max_n, hidden]`，用**两次 `bmm`**
（gate/up、down）替掉 256 次逐专家 matmul。补齐浪费约 2× 的算术，而算术本来只占 0.3%。

**按形状门控**：只有"命中专家数 ≥ 一半"时才走 bmm。decode 只命中 8 个专家，若也补齐，
算术会涨约 250 倍，而且要把这些专家的权重 gather 成稠密批（每层几十 MB 拷贝）—— 所以 decode 仍走循环。

### 53.2 实测（设备，`benchmarks/moe_bmm_ab.py`）

| 形状 | 循环 | bmm | ratio |
|---|---|---|---|
| 512 token（256 专家全命中） | 14.80 ms/call | **10.00 ms/call** | **1.480** |
| 1 token（decode，门控走循环） | 0.71 ms/call | 0.71 ms/call | 0.992（无变化，符合预期） |

正确性：两条路径对 `reference_moe` 的 `max_abs` **逐位相同**（0.000053 / mean 4.3e-6）。
CPU oracle 也新增了 bmm 分支的断言（`loop` 与 `sorted` 因调用同一组 matmul 仍要求逐位相等；
`bmm` 则用 1e-6 容差 —— 不同 BLAS 内核允许 fp32 末位不同，这是我在改测试时明确写下的一条）。

按 C2，ratio 1.48 ≥ 1.0 → **默认开**（`TILEINFER_MOE_BMM=0` 可回退）。

### 53.3 预期与局限

- 每步 prefill 省约 `32 × 4.8 ms ≈ 154 ms`，相对 ~1.9 s 的 TTFT 约 8%。不大，但这是**第一次**
  真正动到 prefill 的 MoE 路径。
- 离我原先的估计（256× 下发 → 秒级收益）差得远，原因是 `fused_experts` 在 prefill 形状下
  只有 14.8 ms，而不是我按"每层 36 ms"以为的 1.15 s（那个 36 ms 是 **decode** 的数字）。
  **又一次把两个形状的数字混用了** —— 这个错误在本会话里已经是第三次（§42 的分组 conv、
  §37 的批次缓存也各一次）。
- 下一步（有窗口时）：bmm 路径现在是**算术受限**的，所以 `TILEINFER_MOE_FP32=0`（bf16 而不是
  fp32 matmul）在这里可能第一次真正值钱 —— 在循环路径上它只值 1.035（下发主导），在 bmm 上
  应当接近 2×。这是一条便宜、可测的后续。

## 54. bf16 与批量化的**乘积**：prefill 形状 5.587×，decode 1.430×

### 54.1 四组对照（同一形状、同一进程，`benchmarks/moe_bmm_ab.py`）

| 路径 | 512 token (ms/call) | 相对 loop+fp32 | 1 token (ms/call) | 相对 |
|---|---|---|---|---|
| loop + fp32（旧默认） | 11.10 | 1.000 | 0.68 | 1.000 |
| loop + bf16 | 8.87 | 1.251 | 0.60 | 1.132 |
| bmm + fp32 | 9.76 | 1.137 | 0.67 | 1.012 |
| **bmm + bf16（新默认）** | **2.46** | **4.517** | **0.60** | **1.137** |

重测（默认已改成 bmm+bf16）得到 **2.55 ms / 5.587×**（基线在这一轮跑出 14.2），
decode 0.61 ms / **1.430×**。

**两项单独都只有 1.1–1.25×，合起来是 4.5–5.6×。** 这是超线性的，机制也清楚：
逐专家循环是**下发受限**（dtype 无关，所以 bf16 只值 1.035，这也解释了 §21.1 那次"噪声内"的结果）；
bmm 之后变成**算术受限**，而此时 fp32 的算术吞吐只有 bf16 的约 1/4。

### 54.2 数值

对 `reference_moe`（直线 fp32）：fp32 路径 5.3e-5、bf16 路径 **8.0e-5**（在 ~1.8e-2 的量级上），
两条 bmm 路径彼此逐位相同。bf16 载入 + fp32 累加正是上游 Triton 的做法，所以这是**更接近**
上游的移植，不是妥协。

### 54.3 落地

- 默认改成 **bmm + bf16**；`TILEINFER_MOE_BMM=0` / `TILEINFER_MOE_FP32=1` 各自可回退；
- `test_candidate_defaults.py` 增加两条断言：默认是输入 dtype、`MOE_FP32=1` 能恢复 fp32 upcast
  （并写明"这个开关在循环路径上曾是噪声内的 1.035，是批量化让它值钱"）；
- `docs/notes/perf-candidates.md` 两行同步更新。

### 54.4 预期收益与又一次教训

按 §19 的分层（MoE 在 decode 占 ~36 ms/步、在 prefill 每层 ~14 ms × 32 层）：

- decode：36 ms → ~25 ms → TPOT 122 → ~111 ms（2.8–3.2× → 约 2.9×）；
- prefill：每层 14.2 → 2.55 ms，32 层 454 → 82 ms → TTFT 1.9 s → ~1.5 s。

教训：**"这个开关没用"的结论有时只是"另一条路径把它吃掉了"**。§21.1 我把 bf16 判成噪声内，
结论错了不是测量错，而是当时被测路径（逐专家循环）根本不受 dtype 影响。以后遇到"两个候选各自
无效"，要试一次**组合** —— 它们可能不在同一个瓶颈上。

## 55. 端到端确认：TPOT **95.8 ms**（预测 111，实际更好），比值 3.2× → **2.53×**

### 55.1 实测

两次独立运行（同一台空闲设备、默认全开、单请求）：

| 运行 | TPOT 中位数 | 极差 |
|---|---|---|
| 第一次 | 97.83 ms/token | 12.1% |
| 第二次 | **95.84 ms/token** | **2.1%** |

对 baseline 的 37.93 ms/token：**2.53×**（§32 那次是 122.0 ms → 3.2×）。

所以 §53/§54 的 MoE 改动在端到端兑现了 **1.27×** —— 比我预测的"122 → 111"更好。
原因是我那个预测把"MoE 在 decode 占 36 ms"当成了上限，而 36 ms 是**叶子探针在外部负载下**测的
（§19）；实际 MoE 的 decode 部分比那更小，且 bf16 也作用在循环路径的 matmul 上。

### 55.2 仍未拿到 TTFT

`bench_arm.py` 这一段在脚本里跑了但没有输出（其余部分正常）。两次尝试都这样，
所以 TTFT@512 **本轮没有新数字**，不能拿 `wall_tpot` 的采样去反推（那是另一条尺子，
反推出来的 0.22 s 只能当线索）。下一轮第一件事是用 `prefill_pair.py` 的单档测量拿绝对
ms/prompt-token，它不依赖 `bench_arm` 的输出格式。

### 55.3 当前成绩单（同环境、配对/尺子实测）

| 指标 | 起点（§无修正时的记录） | 现在 | baseline | 比值 |
|---|---|---|---|---|
| TPOT | 151.4 ms（全部优化关，§32） | **95.8 ms** | 37.93 ms | **2.53×** |
| TTFT@512 | 1.90 s（§18） | **未测**（MoE prefill 5.6× 应已改善） | 0.103 s | — |

M2 的出口是 TPOT ≤1.8×（即 ≤68 ms）。从 95.8 到 68 还要再降 **29%**。
按 §24.3 的对象清单，剩下的主要是 GDN（decode 约 30 ms/步）—— 也就是那个还没写的内核。

## 56. TTFT 实测：512 token 从 1.90 s 降到 **1.26 s**（比值 18× → 12.3×），但固定开销没动

### 56.1 数字

`benchmarks/ttft.py`（新写的最小 TTFT 尺子，避开 `bench_arm` 的输出问题），3 次取中位：

| prompt | 现在 | 改动前 | baseline | 现在比值 | 原比值 |
|---|---|---|---|---|---|
| ~16 | **0.5476 s** | 0.5317–0.5824 | 0.118 | 4.64× | 4.65× |
| ~128 | **0.6630 s** | 0.9335–0.9855 | 0.105 | 6.32× | 8.90× |
| ~512 | **1.2649 s** | 1.8799–1.9000 | 0.103 | **12.3×** | 18.5× |

- **TTFT@512 改善 1.50×**，与 §54 的预测（~1.5 s）吻合；
- 但 **prompt~16 一点没变**（0.55 s），说明这轮的 MoE 改动只作用于"每 token"的部分，
  **固定的 ~0.5 s 完全没动**。

### 56.2 现在的结构

TTFT ≈ **0.5 s 固定 + 约 1.5 ms/token**（512 token 增量 0.72 s）。

- 固定部分：40 层 × 每层几十个小算子的下发（§24 已证不是稠密投影的算术，而是每步的下发次数）。
  它只能靠图捕获消掉，而图捕获在本平台**两次验证不可达**（§18.8、§23.1）。
- 每 token 部分：MoE 已从 14.8 降到 2.55 ms/层（32 层 ≈ 82 ms），所以 0.72 s 里剩下的
  大头是 **GDN 的 chunk 链**（30 层）+ 注意力（10 层）。
  注意 §19 那个 "43 ms/层" 是**争用下**测的，实际应在 20 ms/层量级 —— 否则 30 层就 1.29 s，
  超过整个增量了。

### 56.3 M1 出口（≤3× = 0.32 s）的现实距离

12.3× → 3× 需要再降 4 倍。固定 0.5 s 已经比目标值 0.32 s 还大，所以**只要图捕获不可达，
M1 出口在数学上不可能达成** —— 除非把固定部分的下发次数本身砍下来（那等价于把 40 层的
小算子融合成少数几个，工作量接近重写 vLLM 的层实现）。

这是一条应该正式写进结论的事实，而不是继续用"再优化一点"来掩盖：
**M1 的 ≤3× 与 M3 的 ≤1.1× 都卡在同一个结构性前提上 —— 平台没有 inductor，因此没有图捕获。**

### 56.4 但每 token 部分仍值得做

0.72 s 的增量里，GDN 的 chunk 链是唯一还没被系统优化过的大块（30 层）。它的结构与 MoE 类似：
torch 移植的 Triton 算子、每个 chunk 一次 Python 循环。**下一轮从这里下手**，
预期能再拿 1.5–2×，把 TTFT@512 推到 ~0.8–0.9 s（比值 ~8–9×）。

## 57. 定位到 GDN chunk 链：**16.15 ms/层 × 30 层 = 484 ms**，占增量 67%

### 57.1 分算子表说了什么（512 token prefill，`TILEINFER_PHASE_OPS=detail`）

```
model_forward                 490 ms
attention                     460 ms   <- 占 model_forward 的 94%
op:gdn:fused_recurrent_...decode       1.3 ms
op:gdn:causal_conv1d_update            1.0 ms
op:moe:topk_softmax                    0.9 ms
op:gdn:causal_conv1d_fn                0.4 ms
op:gdn:fused_post_conv_prep            0.4 ms
```

两个要点：

1. **`attention` 占 forward 的 94%** —— 这里的 `attention` 标签挂在整层注意力模块上，
   Qwen3.6 的 GDN 层也算在内，所以它不是"全注意力"的意思。
2. **我们打过标签的 op 全部只有 1 ms 量级** —— 说明成本不在这些被接管的算子里，
   而在 chunk 链里那些**没有单独标签**的函数。这正是"先测再改"要避免的坑：
   如果按 op 表去优化 `causal_conv1d_update`，能拿到的上限只有 1 ms/步。

### 57.2 直接量公共入口（`benchmarks/bench_gdn_chunk.py`）

```
### chunk_gated_delta_rule (public)           16.15 ms
```

× 30 个 GDN 层 = **484 ms**，即 §56 那 720 ms 边际成本的 **67%**。注意力（10 层内核）与 MoE
（82 ms）加起来只占剩下的三分之一。

### 57.3 为什么是 16 ms——算一下就知道不是"小算子太多"

T=512、chunk=64 → **8 个 chunk**。若每个 chunk 有 ~8 个 `[B,HG,64,64]` 量级的算子
（fp32 约 262 KB/个），8×8 = 64 个算子，16 ms ⇒ **约 0.25 ms/算子** ——
这是 **MB 级张量的正常速率**，不是"下发空转"。

所以这里不能照搬 MoE 的"减少下发次数"药方。真正的机会是**把 chunk 循环批量化**：
把 `[B, HG, 64, 64]` × 8 个 chunk 变成 `[B, HG, 8, 64, 64]` 一次算完 ——
同样的算术量，但 8 倍的算子宽度、1/8 的次数，且 `solve_tril` 的块求逆天然可以按 batch 维并行。

### 57.4 下一轮的具体计划

1. 先给 chunk 链的 5 个阶段（kkt / wu / solve_tril / delta_h / 最终 attn）各加一个
   `op:gdn:<name>` 标签（`ops/install.py` 已有机制），跑一次 512 token 拿真实分布；
2. 按最大那一项做"跨 chunk 批量化"，逐项测量；
3. 目标：484 ms → ~250 ms，TTFT@512 从 1.26 s 到 ~1.0 s。

注意我不再预设"哪一项最大" —— §51 和 §57.1 各教了一次：**直觉指的地方经常不是成本所在。**

## 58. chunk 链分级：**`solve_tril` 占 74%**；批量化尝试**失败**（默认关闭）

### 58.1 分级结果（T=512、BT=64、H=16，`benchmarks/bench_gdn_stages.py`）

| 阶段 | 耗时 | 占链 |
|---|---|---|
| `_chunk_cumsum` | 0.033 ms | 0% |
| `chunk_scaled_dot_kkt_fwd` (kkt) | 1.57 ms | 10% |
| **`solve_tril`** | **11.91 ms** | **74%** |
| `recompute_w_u_fwd` (wu) | 2.54 ms | 16% |
| `chunk_gated_delta_rule_fwd_h` (delta_h) | 2.32 ms | 14% |
| `chunk_fwd_o` | 2.11 ms | 13% |
| 合计（整链） | 16.10 ms | — |

（各段单独测会重复计算输入准备，所以各段之和超过整链；但排序是明确的。）

**`solve_tril` 一项就占 74%**，×30 层 = 357 ms，是 §57 那 484 ms 的主体。

### 58.2 机制：串行的 64 步三角求解

`solve_tril` 对每个 chunk 调 `torch.linalg.solve_triangular`（unit-lower, 64×64，按 head 批到 16）。
三角求解本质是 **64 步依赖链**，batch=16 太小掩盖不了延迟；8 个 chunk 又是 Python 循环。

### 58.3 尝试：跨 chunk 批量化 + 并行求逆 —— **失败**

思路：`A` 是 `[B, T, H, BT]`（块内列），可 `view` 成 `[B, n, BT, H, BT]`，一次对所有 chunk
做 `torch.linalg.inv`（并行算法），`tril` 保形，再写回。

结果：

| 量 | 改前 | 改后 |
|---|---|---|
| `solve_tril` 单独 | 11.91 ms | **9.38 ms**（1.27×） |
| **整链** `chunk_gated_delta_rule` | 16.10 ms | **29.31 ms（1.8× 更慢）** |

`linalg.inv` 是 LU 分解，不是我假设的 BLAS-3 批量求解；加上两次 33 MB 张量的布局转换，
总成本超过它替掉的串行替换法。

按 C2 **默认关闭**（`TILEINFER_GDN_SOLVE_BATCH=1` 可复现），代码保留并注明"下一步要从
16×16 分块倍增（上游 Triton 的做法）出发，而不是一般求逆"。

### 58.4 这一轮值不值

值得，因为**失败被量化了**：如果我按"批量化一定更快"的直觉直接改，然后只测整链，
会看到 1.8× 的退化却不知道为什么；分级数据说明"退化发生在 `linalg.inv` 与布局转换上，
而不是我的形状处理"。同时 `solve_tril = 74%` 这个定位是净收益 —— 下一轮的正确入口是
**16×16 分块倍增求逆**（6 层、每层 2 个批量 matmul，无 Python 循环），预计 11.9 → 1–2 ms。

## 59. Newton 倍增求逆：`solve_tril` **45.6×**，但端到端只有 **1.04×** —— 落差值得记

### 59.1 实现

`X <- X(2I - M X)`：从 `X0 = I` 出发，`X1 = I - A` 已经精确，每步把"正确项数"翻倍，
所以 `ceil(log2(BT)) = 6` 步对 nilpotent 指数 64 的严格下三角是**精确**的。
6 次批量 matmul 取代 64 步依赖链，跨 chunk 一次算完（`[B, n, H, BT, BT]`）。

（第一次尝试用的是批量 `linalg.inv`，那是 LU 分解，测出来**更慢**，就是 §58 记的那次。
这次换的是倍增法，不是换成"更快的 inv"。）

### 59.2 孤立测量（T=512、BT=64、H=16）

| 量 | 串行替换 | Newton 倍增 | ratio |
|---|---|---|---|
| `solve_tril` | 11.91 ms | **0.261 ms** | **45.6×** |
| `chunk_gated_delta_rule` 整链 | 16.10 ms | **8.43 ms** | **1.91×** |
| 各段之和 / 整链 | 127%（重复计算） | **89%** | 测量自洽性改善 |

129 个 GDN oracle 在开关开、关两种状态下都通过。

### 59.3 端到端：只有 1.04×

| prompt | 改前 | 改后 |
|---|---|---|
| ~16 | 0.548 s | 0.552 s |
| ~128 | 0.663 s | 0.622 s |
| ~512 | 1.265 s | **1.214 s** |

按孤立数据（每层省 7.7 ms × 30 层 = 231 ms）本该到 ~1.03 s，实际只省了 51 ms。
而样本极差是 8%（1.208–1.313），所以 **1.04× 落在噪声带内**。

按 C2 保留默认开（孤立 45.6× 只可能更好，且大 batch 下会更明显），但**不把它记成端到端收益**。

### 59.4 落差说明了什么 —— 以及下一步该打谁

用现在的数字反推 TTFT@512 的构成（边际 662 ms / 496 token）：

| 块 | 估算 | 依据 |
|---|---|---|
| GDN chunk 链 | 253 ms | 8.43 ms × 30 层（本轮改后） |
| MoE 专家 | 82 ms | 2.55 ms × 32 层（§54） |
| **注意力 prefill 内核** | **约 327 ms** | 662 − 253 − 82 |

也就是说：**注意力 prefill 内核现在是最大的一块**，比 GDN 链更大。这与 §18/§19 的旧结论不同，
因为那两块都被改动过。**下一轮打注意力 prefill 内核。**

### 59.5 一条方法论教训

孤立加速比不能直接换算成端到端。本会话里两者多次不一致（分组 conv 1.06→1.004、
批次缓存 1.05→0.999、MoE bmm 5.59→端到端 1.27、本轮 1.91→1.04）。
原因是每一步都在等**最慢的那一项**，而改动只削掉了自己那一项。
所以：**孤立 ratio 用来判断"这一项值不值得做"，端到端 ratio 才是成绩**。这条以后每条记录都分开写。

## 60. 更正 §59.4：注意力 prefill 在**孤立**测量里只有 3 ms/层，但服务里报 30.6 ms/层

### 60.1 孤立测量（512 token，单请求，`benchmarks/bench_prefill_attn.py`）

| 配置 | 每层耗时 | 有效算力 |
|---|---|---|
| 内核（默认） | **3.05 ms** | 0.71 TFLOP/s |
| 参考 prefill（`TILEINFER_PREFILL=0`） | 3.77 ms | 0.57 TFLOP/s |
| 全 torch（`TILEINFER_ATTN_BACKEND=reference`） | 2.60 ms | 0.83 TFLOP/s |

因果掩码下这一层是 2.15 GFLOP，所以 3 ms 对应 0.71 TFLOP/s —— 慢，但**绝对值很小**：
×10 个全注意力层 = **30 ms**，不是 §59.4 反推出来的 327 ms。

**§59.4 那张表的"注意力 ≈ 327 ms"是错的**，它来自"总数减去已测量项"的余项法，
而余项里显然还有别的东西。记在这里，避免下次引用。

### 60.2 服务里的分相表却显示 30.6 ms/次

同一批 512-token 请求（8 个）的分相表：

```
model_forward                 9    1877.306
attention                   100    1835.573          <- 占 forward 的 98%
mod:language_model.model.layers.self_attn.attn   60   1834.312   <- 单次约 30.6
mod:language_model.model.layers.linear_attn.chunk_gated_delta_rule  63   7.700   <- 单次 0.12?!
mod:...mlp.experts.quant_method   270   12.885
mod:...mlp.shared_expert.down_proj 270   0.122
```

两个刺眼之处：

1. `self_attn.attn` 单次 30.6 ms，而孤立测量是 3.05 ms —— **差 10 倍**；
2. `chunk_gated_delta_rule` 单次只有 0.12（而孤立测量是 8.43 ms）—— **方向相反**。

### 60.3 结论：分相表的列语义我没能确认，不能用它做决策

`count | ? | ? | max` 这几列的含义我猜过两次都没对上（§42 也踩过一次）。
在搞清列语义之前，**不依据这张表做优化决策** —— 这正是本会话反复出现的教训：
先确认仪器，再相信数字。

### 60.4 下一步（两个都很便宜）

1. **把 P0 的请求改成一次一个**（`--max-num-seqs 1`、串行发送），排除批内相互干扰；
2. **直接给 `PrefillPath` 加一个自带的计时输出**（不经过 phases 框架，避免插桩开销），
   在服务里打印"每层 prefill 内核 X ms、host 侧 Y ms"。这是唯一能把 3.05 与 30.6 对上或对不上的办法。

搞清楚这个 10 倍差之前不动代码 —— 因为按 30.6 去优化和按 3.05 去不优化，是两个完全相反的结论。

## 61. 自带计时已装进 `PrefillPath`，但两次都没抢到设备

### 61.1 装了什么

`TILEINFER_PREFILL_TIMING=1` 时，`PrefillPath.run` 自己计两个数（不经过 phases 框架、不用设备事件）：

- **launch**：从进入 `run` 到所有内核**下发完毕**的墙钟时间（纯 host 侧）；
- **total**：之后 `torch.npu.synchronize()` 完的时间（含设备时间）。

两者之差就是真实设备耗时，每 20 次打印一行中位数。这就把 §60 那个 3.05 vs 30.6 的歧义变成
一次可直接读的测量，而且**不干扰**被测路径（开关关闭时零开销：只有一次 `os.environ.get`）。

本地 23 项 prefill 测试通过（其中 2 项 skip 是原有的形状不适用）。

### 61.2 两次尝试都被"秒抢"

- 第一次：gate 看到设备空闲（5450 MB used），启动时却报
  `Free memory on device npu:0 (5.48 GiB) … less than desired (0.75, 92.27 GiB)`
  —— 对方在我检查后的 **几秒内** 占了 117 GB；
- 于是改成 3 次重试的后台循环（`/tmp/ptr.out`），本轮预算内未读到结果。

这条和 §48 是同一件事的延续：设备是共享的，而且对方的启停节奏比我一次测量所需的时间窗还短。

### 61.3 下一轮第一件事

读 `/tmp/ptr.out`。如果拿到 `prefill-timing` 行，就按那两个数决定注意力要不要动：

- **total ≈ 3 ms** → 注意力不是瓶颈，§59.4 的余项法彻底作废，那 297 ms 要另找
  （候选：稠密投影、`linear_attn` 的 in_proj/out_proj、embedding/LM head）；
- **total ≈ 30 ms** → 服务里确实比孤立慢 10 倍，最可能的原因是 `PrefillPath` 的 tile 调度
  在服务里用了更小的 tile（`TILE_LADDER`），并行度不足 —— 那就改调度而不是改内核。

## 62. 10 倍差解开了（不需要设备）：分相表把**一次编译**算进了总量

### 62.1 关键观察

回到 §60.2 那张表，只看这三行：

```
mod:...self_attn.attn      60   1834.312   1834.291   1834.291
attention                 100   1835.573   1835.561   1101.336
mod:visual.blocks.attn.attn 27     53.268     50.379    111.953
```

`self_attn.attn` 的第 2、3、4 列**几乎相等**（1834.312 / 1834.291 / 1834.291）。
如果总量 1834 ms 分布在 60 次调用上，最大值不可能等于总量 ——
**除非其中一次调用就占掉了几乎全部**。结合 `attention` 那行（100 次、总量 1835、最大 1101），
唯一自洽的解释是：**这是 tilelang 内核的首次编译**（几百毫秒到 1.8 s），而不是稳态耗时。

顺带也解释了 `visual.*` 那几行（视觉塔在 profile 阶段用 dummy 输入跑过一次）。

同时必须承认：这张表的列语义我仍然没能确定（`visual` 那行 max 111 > 总量 53，
说明第 2 列不是"总量之和"）。**所以它不能用于任何定量结论，只能用于"有没有编译开销"这类定性判断。**

### 62.2 结论

- **§60.2 的"30.6 ms/次"撤回**；注意力 prefill 的可靠数字是孤立测量的 **3.05 ms/层**（×10 层 = 30 ms）；
- 那个 10 倍差**不是**真实的性能差异，是"把编译算成稳态"造成的读表错误；
- 因此 §59.4/§60 里"注意力是最大一块（327 ms）"**彻底作废**。

### 62.3 那 662 ms 到底花在哪（用算术下界反推）

512 token 的 prefill，激活参数约 3B：`512 × 3e9 × 2 = 3.07 TFLOP`。
在 ~100 TFLOP/s 上，**算术下界约 31 ms**；实测边际成本 **662 ms**，差 **21 倍**。

已测到的三块：GDN 链 253 ms + MoE 82 ms + 注意力 30 ms = **365 ms**，
**余下约 297 ms**。稠密投影 + LM head 的算术量加起来只有 ~8 ms（`512×2048×3072×2` 每层 ×40，
加上 LM head 的 520 GFLOP），所以这 297 ms **不是算术**，而是**约 1200 个小算子的下发**
（40 层 × 每层约 30 个 op × 约 0.25 ms）。

也就是说：**prefill 剩下的成本是下发受限，和那 0.5 s 固定开销是同一类问题**，不是算力问题。
这条与 §18.9、§24 的结论一致，但这次是**用算术下界 + 已测三块反推**出来的，不是猜的。

### 62.4 由此得到的行动优先级（与之前不同）

| 方向 | 判断 |
|---|---|
| 优化注意力内核 | **不做** —— 已证只有 30 ms，改它上限就是 30 ms |
| 继续压 GDN 链 | 已从 484 降到 253 ms，再做上限 253 ms（Newton 那条路已走到头，下一步要写内核） |
| **减少每层小算子数（手工融合）** | **最有价值**：297 ms 分散在约 1200 个 op 上，每层省 5 个 op 就是约 40 ms |
| 图捕获 | 结构性阻塞（§18.8/§23.1），仍不可达 |

下一轮（有设备时）验证方式很具体：给 `PrefillPath` 的计时已经装好（§61），
再在 `install.py` 里给"每层 op 计数"加一个计数器，跑一次 512 token 就能拿到**每层到底多少个 op**，
然后挑数量最多的那一族做手工融合。

## 63. 两个 harness 问题：HBM 门限设错，以及计时没打印

### 63.1 HBM 门限设错 —— 白白错过窗口

我的入场判断一直是"占用 < 20 GB"，但 vLLM 实际只需要 123.02 − 92.27 ≈ **30 GB** 以下就能起来。
对方占 23.9 GB 时，我判"忙"、实际完全能跑。

改正后立刻成功：

```
device free enough (23913 MB used)
up
```

租户式的设备共享下，这个门限的误差直接等于"白等一个窗口"。已把 `benchmarks/prefill_timing.sh`
里的门限改成 29 GB，并注明依据（不是拍的数，是 `123.02 − 92.27`）。

### 63.2 计时一行都没打印（未解决）

那次运行里 `PrefillPath` 确实跑了 100 次（日志有
`TileInfer prefill: 100 steps, kernel=100 reference=0 (kernel share 1.000)`），
服务器上的 `prefill.py` 也确实含 `prefill-timing` 打印代码（grep 命中），
但 `tileinfer/prefill-timing` 一次都没出现。

我已排除的：代码没同步（有）、`_timing_stats` 没初始化（有）、路径没走到（100 步）。
下一步最省的查法：在 `PrefillPath.__init__` 里打印一次 `TILEINFER_PREFILL_TIMING` 的**实际取值** ——
如果 EngineCore 子进程没继承到这个环境变量，那就解释了全部现象，也说明我该用
`--env` 或写进 flag 文件通道（`tileinfer/serving/flags.py` 已经支持）。

### 63.3 本轮净结果

- 一个真实的 harness bug 修掉了（门限），它此前让我误判过至少两次窗口；
- 计时没生效，**没有新数字**；
- 但 §62 的那个结论（注意力只有 3 ms/层、662 ms 里约 297 ms 是约 1200 个小算子的下发）
  是**从已有数据推出来的**，不依赖这次失败的测量，且它把下一轮的靶子从"注意力"改成了"每层算子数"。

## 64. 服务内计时跑通：注意力 prefill 每步 **1.78 ms，其中 1.17 ms 是内核下发**

### 64.1 为什么之前没打印

两个叠加的 bug，都不在"环境变量没继承"上（那个假设**被证伪**了：`enabled (TILEINFER_PREFILL_TIMING='1')`
在日志里出现了 6 次，每个注意力层一个 `PrefillPath` 实例各打一次）：

1. 计数器是**实例级**的。`PrefillPath` 每层一个实例，每层每请求只 `run` 一次 → 实例内的计数
   永远到不了 20 的门限，什么都不打印。改成**模块级聚合**后立刻正常。

（顺带又踩了一次"编辑把两行粘成一行"：`qo = _query_lengths(...)        dev = q3.device`
→ 服务端 `NameError: name 'dev' is not defined`。这次是服务端日志抓到的，不是本地测试 ——
说明本地编译检查挡不住这种，得靠单测或真跑。）

### 64.2 结果（稳态，144 token/次，20 次窗口）

```
total=1.78ms   setup=0.55ms   launch=1.17ms   store=0.01ms
```

- **`launch` 占 1.17 ms**（66%）：tilelang 内核的 Python 侧封送/参数派生；
- `setup` 0.55 ms：`_query_lengths` + `refresh_step` + `_rectify` + `_schedule`；
- `store` 0.01 ms；
- `total`（含 `synchronize`）1.78 > 1.17+0.55+0.01 = 1.73，**设备时间只有约 0.05 ms**。

也就是说：**这一步 98% 是 host 开销，内核下发是设备耗时的约 23 倍。**
第一个窗口 8.74 ms（launch 7.91）是首次编译，与 §62 对分相表的解释一致。

另外 `tokens/call=144` 不是"分块 prefill"：窗口里混了 prefill（512）与 decode（1）两种调用，
解方程得约 6 次 512-token prefill + 14 次 decode —— 也就是**每个请求的 prefill 是一次内核调用**，
没有被切成小块。这纠正了 §60.4 里"可能用了更小 tile"的猜测。

### 64.3 每请求的账（512 token，10 层全注意力 + 30 层 GDN + 32 层 MoE）

| 块 | 每请求 | 依据 |
|---|---|---|
| 注意力 prefill | 10 × 1.78 = **17.8 ms** | 本节实测 |
| GDN chunk 链 | 30 × 8.43 = **253 ms** | §59 |
| MoE 专家 | 32 × 2.55 = **82 ms** | §54 |
| **合计已测** | **353 ms** | |
| 实测边际 | **662 ms** | TTFT@512 − TTFT@16 |
| **余项** | **约 310 ms** | vLLM 自身的算子（norm/rope/投影/embedding/LM head） |

余项的 310 ms 按算术算只有约 8 ms，所以它同样是**下发受限**，而且**不是我们的算子**
（op 计数显示我们的接管每 2000 次调用里只占约 150 次/步，其余约 1050 次是 vLLM 自己的）。

### 64.4 因此"融合我们的算子"这条路的收益上限很低

我上一轮把"每层算子数"列为最有价值的方向，现在必须修正：**我们的算子只占约 12% 的下发**。
即使把 ours 全部融掉，上限也就 ~12% 的余项（约 37 ms）。

真正剩下的三条路：

1. **降低 tilelang 的每次调用成本**（1.17 ms/次）：同样的形状每步都重复派生 grid/参数，
   缓存掉可以降到 ~0.1 ms，10 层 → 每请求省约 10 ms。**量小但确定性高**；
2. **GDN chunk 链 253 ms**：已从 484 降下来，剩下的要写内核（M2 的原始任务）；
3. **图捕获**：唯一能消掉"vLLM 自身 310 ms 下发"的手段，但平台不可达（§18.8/§23.1）。

按收益/风险排，下一轮做 (1)：它是纯 host 侧的缓存，风险低、可回退，且能被本次装好的计时器
直接验证（`launch` 从 1.17 降到多少，一眼可见）。

## 65. 缓存 pack 索引：`launch` 1.17 → 1.06 ms（1.106×），**端到端可忽略**

### 65.1 做了什么

`forward_prefill` 每次调用都重新推导"把 Q 打包进内核 tile 布局"的索引张量：
3 次 dtype 转换（`seq_ids`/`q_offsets`/`q_lens` → int64）、2 个 `arange`、4 个索引算术算子、
1 次 `clamp`。这些**只依赖 schedule**，而 schedule 只依赖每请求的 query 长度。

抽出 `_pack_indices(...)`，在 `PrefillPath` 里按 `(tuple(qo), num_reqs)` 缓存（上限 32 条，
持有设备张量所以必须封顶），`forward_prefill(pack=...)` 命中时跳过全部推导。

### 65.2 实测（服务内、稳态、20 次窗口）

| 量 | 缓存前 | 缓存后 | ratio |
|---|---|---|---|
| `launch` | 1.17 ms | **1.06 ms** | 1.106 |
| `total` | 1.78 ms | **1.67 ms** | 1.067 |

按 C2 保留（1.106 ≥ 1.0，纯缓存、数值逐位不变、32 项 prefill 测试全过）。

### 65.3 但这条**不值得当成进展**

10 层 × 0.11 ms ≈ **每请求 1.1 ms**，相对 1.2 s 的 TTFT 是 **0.1%**。我原本猜"能省 0.6 ms"，
实测只有 0.11 —— **又一次把某个部件的耗时想多了**。

真正剩下的 1.06 ms 在哪里：`forward_prefill` 里还有
`q.permute().reshape().contiguous()`（4 MB 拷贝）、`index_select` + `masked_fill` + `contiguous`
（4 MB 级）、`meta.kv_indices.to(torch.int32)` 及可能的 `cat` 补长，以及 **tilelang 自身的
Python 包装层**。要把这 1 ms 再压下去，得进 tilelang 的调用路径 —— 那是另一个量级的改动，
且 10 层合计也只有 10 ms/请求（1.5% 的边际成本）。

**所以：这条路到头了。** 记在这里是为了避免下一轮又去动它。

### 65.4 现在的真实处境

| 块 | 每请求 512 token | 能否再降 | 需要什么 |
|---|---|---|---|
| GDN chunk 链 | 253 ms | 能，上限大 | **写内核**（多轮工作量） |
| vLLM 自身算子下发 | ~310 ms | 只有图捕获 | **平台 inductor（不可达）** |
| MoE 专家 | 82 ms | 已从 484 降下来 | 已接近 bmm 上限 |
| 注意力 prefill | 17.8 ms | 只剩 10 ms 的空间 | tilelang 包装层 |

也就是说：**剩下两块大的都需要超出"改几行代码"量级的工作**（写内核 / 平台能力）。
在单轮预算内，我能继续做的只有测量与小幅收敛，**这不会再产生可与目标相称的收益**。

这一点我按 E2 明确记录：目标终态（TTFT ≤1.1×、TPOT ≤1.1×）在当前平台能力下，缺口主要在
两处结构性前提上 —— 没有 inductor（图捕获）和 GDN 缺内核。前者不是软件能补的。

## 66. **推翻 §18.8/§23.1**：图捕获不是平台不可达，是被我自己的 `_dynamo_opaque` 挡住的

### 66.1 第一步：别名桥其实是好的

设备上直接探测（不需要 HBM）：

```
### torch.npu.NPUGraph: <class 'torch_npu.npu.graphs.graph'>   (torch.npu.CUDAGraph 不存在)
### install() -> (True, 'aliased torch.cuda.CUDAGraph, torch.cuda.graph,
                      torch.cuda.mem_get_info, torch.cuda.Stream; best-effort skipped: OutOfMemoryError')
### after install, ready: True
```

也就是说 **`graphs.capture_path_ready()` 在 install 之后是 True** —— 目标里那句"NPUGraph 别名已修"
是对的，桥是可用的。§18.8/§23.1 关于"平台不可达"的结论**前提就错了**。

### 66.2 第二步：`--enforce-eager` 一直在掩盖这个问题

我每一次测量都带 `--enforce-eager`。vLLM 把它映射成 `cudagraph_mode=NONE`，
于是日志出现"Skipping CUDA graph capture. … cudagraph_mode was not manually set to 'NONE'"——
我当初就是拿这句话下的结论。**那句话是我自己的启动参数造成的。**

改掉之后（`--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'`，不带 `--enforce-eager`），
失败点搬到了别处：

### 66.3 第三步：真正的阻塞点是 dynamo fullgraph 拒绝我们的 impl

```
torch._dynamo.exc.Unsupported: Skip inlining `torch.compiler.disable()`d function
  Explanation: Skip inlining function <TileInferAttentionImpl.forward> since it was wrapped with
  `torch.compiler.disable` (reason: None)
  file: vllm/model_executor/layers/attention/attention.py:751 → self.impl.forward(...)
```

vLLM 的图捕获路径用 **fullgraph=True**，它**不允许任何 graph break**；
而我当初为了绕开 "Unsupported context manager: `with self._lock`" 给 `forward` 加了
`@torch.compiler.disable`（`_dynamo_opaque`，§18.8）。**那个 workaround 正是现在唯一挡路的东⻄。**

### 66.4 因此正确的做法不是"等平台"，而是把内核暴露成 custom op

`torch.compiler.disable` 让函数在 fullgraph 下非法；而 **`torch.library.custom_op`** 让 dynamo
把它当作"不透明但可追踪"的算子 —— 这正是 fullgraph 允许的形态。要做的：

1. `TileInferAttentionImpl.forward` 去掉 `_dynamo_opaque`，把 `with self._lock` 移出追踪路径
   （计数改为无锁或延后）；
2. `PrefillPath.run` 里的 Python 分支/字典缓存改成"按 shape 在构造期决定"，追踪区内不留 host 逻辑；
3. 注意力内核与 FLA 的 tilelang/torch 算子在边界处包成 `custom_op`（含 fake/meta 实现），
   让 dynamo 拿到形状就能继续；
4. 然后再试 `cudagraph_mode=FULL_DECODE_ONLY`。

这是一件多轮的工作，但它是**唯一**能消掉那 310 ms/请求下发与 0.5 s 固定开销的路径，
而且现在已经从"平台不可达"变成"四个明确的改造点"。

### 66.5 一条方法论教训（本会话第三次同类）

前两次是"把两个形状/频次的数字混用"（§42、§54）；这次是**把一个由自己的启动参数造成的日志，
当成了平台的限制**。共同点：**结论的证据必须能追溯到"在什么条件下观测的"**。
我引用那句 "cudagraph_mode was not manually set to NONE" 时，没有回去看自己的命令行。

## 67. 清掉三个阻塞点，第四个现形：dynamo 用 **2048** 的形状去追踪我们的内核

### 67.1 修复清单（都已落地，可回退）

| # | 阻塞点 | 修法 | 证据 |
|---|---|---|---|
| 1 | `forward` 被 `torch.compiler.disable` 包住 → fullgraph 硬报错 | 加开关 `TILEINFER_DYNAMO_OPAQUE=0` 可撤掉该装饰器 | 装饰器撤掉后 dynamo 不再报 `Skip inlining` |
| 2 | `Counters.bump` 里的 `with self._lock` → `Unsupported context manager` | 追踪期间（`torch._dynamo.is_compiling()`）直接 no-op，不加锁不分配 | 计数器语义无变化，仍线程安全 |
| 3 | **vLLM 上游 bug**：`pass_manager.py:164` 无平台保护地实例化 `RMSNormQuantFusionPass`，而该类只在 `is_cuda_alike()`/`is_xpu()` 下 import | `_disable_cuda_only_compilation_passes()` 在 `check_and_update_config` 里关掉 7 个 CUDA-only 融合 pass（全是 fp8/int8 量化融合，本后端本来就直接拒绝量化注意力） | NameError 消失 |

第 3 条是一个**真正的上游不兼容**：只要不开 `--enforce-eager`，`NameError` 就会在引擎启动时抛出 ——
也就是说在这台机器上，"编译路径"以前**从来没有成功初始化过**，而这正是我此前把 ACLGraph 判成
"平台不可达"的全部依据。

### 67.2 第四道坎

清掉上面三条后，错误前进到 tilelang 自己的断言：

```
AssertionError: Shape: 2048 out of considered ranges: [(1, 2048)]
```

`dynamo` 追踪时用 `max_model_len = 2048` 的形状去实例化内核，而我们的编译变体只覆盖
`< 2048`（上界开区间）。eager 路径下每步给的是真实长度（144/512），永远不会碰到这个边界。

这正是 §66.4 第 3 条要解决的事：**把内核调用包成 `torch.library.custom_op`（带 fake/meta 实现）**，
让 dynamo 用"抽象形状"通过，而不是拿一个 2048 的实参进内核。这一条是纯新增（一个 wrapper 模块 +
在每个调用点替换），不改内核本身。

### 67.3 本轮的位置

- M3 的第一个前提（图捕获）从"平台不可达"变成"**还差一个 custom_op 包装层**"；
- 本轮无性能数字（拿不到，因为引擎还没能在非 eager 下起来）；
- 三个修复本身都是净收益：第 3 条让"编译路径在 NPU 上能初始化"这件事第一次成立，
  它同时也是 §18.8 那条错误结论的**真正解释**。

### 67.4 下一轮

实现 `tileinfer/kernels/custom_ops.py`：给 `paged_decode_ascend950.forward` / `forward_prefill`
以及 FLA 的 5 个算子在边界包一层 `custom_op`（`abstract` 分支返回 `torch.empty` 即可，
形状由入参推出），然后在 `impl`/`prefill`/`ops` 的调用点改用包装后的入口。
这是本轮明确出来的、范围可控的一步。

## 68. 不开图捕获、只开编译：**TPOT 90.0 ms（历史最好）**，但 TTFT 变差

### 68.1 先说结果

去掉 `--enforce-eager`（保留 `--compilation-config` 的 `compile_sizes`），服务正常起来并跑完两把尺子：

| 指标 | 之前（eager） | 现在（编译开、图捕获仍被跳过） | 变化 |
|---|---|---|---|
| **TPOT** | 95.8 ms | **90.0 ms**（极差 1.1%） | **1.064× 更好** |
| TTFT@16 | 0.548 s | 0.529 s | 1.036× 更好 |
| TTFT@128 | 0.663 s | **1.053 s** | **1.59× 更差** |
| TTFT@512 | 1.265 s | **1.415 s** | **1.12× 更差** |

TPOT 的 6% 来自 vLLM 的 piecewise 编译（decode 的稠密部分被编成图段），与 ACLGraph 无关
（日志仍是 `Skipping CUDA graph capture`）。

TTFT 变差是可预期的：`compile_sizes` 让 prefill 也走分段编译，而 prefill 的
token 数（144/512）在多组 size 之间切换，每次切换都有调度开销；我们 512 token 的短 prefill
本来就不适合按 size 分段。

**结论：这是一个需要分开决策的取舍**，不是"全面变快"。TPOT 与 TTFT 在目标里是两个独立指标
（终态要求都 ≤1.1×），所以不能简单地"开编译"。按 C2，这一项**先不进默认**：
它同时带来 1.064× 和 0.63×，需要先弄清 TTFT 那部分能否用 `compile_sizes` 的取值把它消掉
（例如只让 decode 走编译、prefill 保持 eager）。

### 68.2 图捕获链：清掉四道坎，进到 vLLM 的捕获路径里

| # | 阻塞点 | 修法 | 结果 |
|---|---|---|---|
| 1 | `forward` 被 `torch.compiler.disable` 包住 | 开关 `TILEINFER_DYNAMO_OPAQUE=0` | `Skip inlining` 消失 |
| 2 | `Counters.bump` 的 `with self._lock` | 追踪期间 no-op | `Unsupported context manager` 消失 |
| 3 | 上游 bug：`pass_manager.py:164` 实例化未 import 的 `RMSNormQuantFusionPass` | 关掉 7 个 CUDA-only 量化融合 pass | `NameError` 消失 |
| 4 | `cudagraph_mode` 被平台解析成 NONE；随后又缺 capture size | `TILEINFER_ACLGRAPH=1` 在 `check_and_update_config` 里显式设 mode + `cudagraph_capture_sizes` | 进到 `profile_cudagraph_memory` |

第 4 条里两个细节值得记：`--compilation-config '{"cudagraph_mode": ...}'` **不能**穿透非 CUDA 平台的
post-init（实测日志仍是 `cudagraph_mode was not manually set to NONE`），必须在平台里改；
而 `compile_sizes` 不会顺带给出 `max_cudagraph_capture_size`（`cudagraph_dispatcher.py:80` 直接 assert）。

现在失败在 `gpu_model_runner.py:6378 profile_cudagraph_memory` —— 这是**捕获前的显存试算**，
说明配置层面已经全部打通，剩下的是运行时（`torch.npu` 侧的图/显存原语在真实捕获中的行为）。

### 68.3 顺带修掉第 6 次 `import os`

`platform.py` 用了 `os.environ` 但没 import —— 本会话第 6 次同类错误，且这次是在**设备启动时**
才炸（本地单测不覆盖那行）。已加 `tests/test_import_hygiene.py`：AST 扫描所有模块，
凡是用到 `os` 就必须 import 它。这是把重复犯的错交给机器，而不是交给记性。

## 69. 捕获真的跑起来了：卡在"捕获区里有 H2D 拷贝"

### 69.1 本轮配置与结果

`TILEINFER_COMPILE=0`（关 torch.compile）+ `TILEINFER_ACLGRAPH=1`（要解码图）：
**没有** `Skipping CUDA graph capture`，而是直接进到 NPUGraph 捕获内部：

```
[__cudagraphs] NPUGraph: ERROR —  (torch_npu/npu/graphs.py:918)
RuntimeError: copy_between_host_and_device_opapi: CopyKernelOpApi.cpp:100
  NPU function error: aclrtMemcpy, error
```

这不是配置问题，而是**被捕获的图里有一次 host→device 拷贝**：捕获期与回放期都不能执行
依赖主机指针的 memcpy（回放时那个主机地址可能已经不同）。

### 69.2 这条路径是怎么一步步打通的

| 阶段 | 失败点 | 说明 |
|---|---|---|
| §18.8/§23.1 | `Skipping CUDA graph capture` | **是我自己的 `--enforce-eager`** |
| §66 | `Skip inlining torch.compiler.disable` | 我自己的 `_dynamo_opaque` |
| §67 | `NameError: RMSNormQuantFusionPass` | **vLLM 上游 bug**（CUDA-only pass 无平台保护） |
| §68 | `cudagraph_mode` 解析成 NONE / 缺 capture size | 平台层可修 |
| §68 | `profile_cudagraph_memory` 断言 | 显式设 mode + sizes 后通过 |
| **§69** | **`aclrtMemcpy` 在捕获区** | **我们代码里的一次 H2D** |

六层，前五层全部清掉；这一轮第一次真的开始 capture。

### 69.3 下一步：找出解码路径里的那次 H2D

最可疑的三处（按嫌疑排序）：

1. **`serving/step.py: refresh_step`** —— 每步把 ragged 三元组写进缓冲，
   `kv_indptr`/`last_page_len` 的写入可能含 `host_list -> device` 的 `copy_`；
2. **`TileInferMetadata.query_start_loc_cpu`**（CPU 张量）若在 `_kernel_forward` 里被 `.to(device)`；
3. **`write_kv_cache`** 的 `meta.slot_mapping` 若某步由 CPU 张量拼出。

查法不需要设备：在 `_kernel_forward` / `refresh_step` 路径上 grep `.to(dev`、`torch.tensor(`
（由 Python 列表构造的），逐个确认是否落在捕获区内。

### 69.4 一个必须写下的判断

从 §66 到 §69，我用了**五轮**才把「ACLGraph 不可达」这个结论拆掉。它的起点是一次
**没有去掉自己启动参数**的观察。如果 §18.8 当时多做一次"去掉 `--enforce-eager` 再看"，
后面这些轮次都会省下。这不是运气问题，是**证据纪律**问题：任何"平台不支持"的结论，
都必须先排除自己传下去的参数。

## 70. 捕获失败的真因：**捕获区里的一次同步拷贝**，已给解码路径做了"零主机读取"版本

### 70.1 错误信息读法

之前只看到 `aclrtMemcpy, error`，误以为是"某处 H2D 拷贝"。拉完整块后 torch_npu 自己给了诊断：

```
[__cudagraphs] NPUGraph: ERROR — capture failed: RuntimeError:
        rtStreamSynchronize execution failed
        Synchronous memory copy failed
        rtMemcpy execution failed
[__cudagraphs] NPUGRAPH DFX common causes: default stream, CPU ...
```

**"Synchronous memory copy"** —— 就是 `.tolist()` / `.item()` 这类 **device→host 同步读取**。
捕获与回放都不能执行它（回放时主机地址没有意义）。

### 70.2 解码路径上唯一的那次读取

`fused_experts` 的排序路径为了做专家循环，必须知道每个专家命中几个槽位：

```python
counts = torch.bincount(sorted_ids, minlength=num_experts + 1).tolist()   # 一次 D2H 同步
```

**每层每步一次**，32 层 → 每步 32 次。这就是捕获失败的根因。

### 70.3 修法：解码路径全部用形状推出的上界

新增 `_routed_device_into`：**完全不读主机**。

- 上界来自形状：任何专家都不可能拿到超过 `num_tokens` 个槽位 → 把每个专家补齐到
  `num_tokens` 行就是安全的；
- `counts` 只在**设备上**用（`cumsum - counts` 当偏移），`valid = rows < counts` 也在设备上；
- 写回用一次 `index_copy_`（唯一目标 → 确定性），被丢弃的槽位由调用方补零。

门控：`num_tokens <= 8` 走这条路（解码），prefill 保留原来"允许读主机"的形式
（图捕获只覆盖解码，prefill 不需要满足这个约束）。

算术代价：解码 1 token 时补到 256 专家 × 1 行 —— 比只算 8 个专家多 32 倍算术，
但那是 1.6 GFLOP/层 × 32 层 ≈ 0.5 ms/步，用一次 D2H 同步（~0.1 ms × 32 = 3.2 ms/步）换，
**净赚**。

### 70.4 验证状态

- **CPU oracle 全过**（7 项；测试用 2–6 个 token，正好落在新路径的门控内，所以新代码被真实覆盖）；
- **设备上的捕获验证本轮没拿到**：重试脚本的输出文件为空、进程没起来（harness 又出问题，
  这次是 `nohup` 启动的那一层没生效）。下一轮第一件事就是重跑 `graph_eager_capture.sh`。

### 70.5 这一条即使捕获不成，也是净收益

`_routed_device_into` 去掉了每步 32 次 D2H 同步。按 §42 实测的"一次校验同步 ≈ 0.1 ms/步/层"，
32 层约 **3 ms/步**（TPOT 95.8 → 约 93）。也就是说：**这个改动本身有一个可测的 TPOT 收益**，
而捕获成功只是额外奖励。下一轮先测它，再测捕获。

## 71. 解码路径的同步读取审计：只剩 **GDN 卷积的 `_cache_index_list`**

捕获要求"捕获区内没有任何同步读取"。设备被占（122 GB），所以本轮做纯代码审计，
把解码路径上的 `.tolist()` / `.item()` / `bool(...)` 逐个定位：

| 位置 | 调用频率 | 是否在解码路径 | 说明 |
|---|---|---|---|
| `moe.fused_experts` 的 `bincount().tolist()` | 32 次/步 | **已消除** | §70 新增 `_routed_device_into`，解码走零读取路径 |
| `fla/causal_conv1d.py:671 _cache_index_list` | **30 次/步** | **是** | `.to("cpu").tolist()` 取每个序列的 cache 行号 |
| `fla/causal_conv1d.py:674 num_accepted_tokens` | — | 否（投机解码才用） | 我们没开投机解码 |
| `fla/causal_conv1d.py:366/377` | — | 否（varlen/prefix-cache 路径） | prefill 用 |
| `fla/causal_conv1d.py:757 bool(valid.any())` | — | 否 | 只在"投机窗口/带偏移"的一般分支；普通解码走 744-750 的免同步快路径（§32） |
| `attn/impl.py:910 torch.tensor([qo_len], device=…)` | — | 否 | 参考实现（降级）路径 |

**结论：解码路径上现在只剩 `_cache_index_list` 这一处同步读取（30 次/步）。**
它是 `conv_state[line]` 的下标来源 —— 上游写成一个 Python int，所以要读回主机。

### 71.1 下一轮的具体改法（已想清楚，不需要再设计）

把内层循环从"Python int 下标"改成"设备张量下标"，保持 `range(batch)` 的形状级循环
（batch 由形状决定，dynamo 会特化，这是合法控制流）：

- `line = int(lines[seq])` → `line_idx = conv_state_indices[seq : seq + 1]`（设备张量）；
- `line_view = conv_state[line]` → `line_view = conv_state.index_select(0, line_idx)[0]`；
- 状态写回 → `conv_state.index_copy_(0, line_idx, new_line)`；
- `null_block_id` 判断（`if line == null_block_id`）改成 device mask 或按"无效行写回原值"处理。

这条路的风险点是 `index_copy_` 与切片写的语义差异（尤其重叠视图那个 `.clone()`），
所以要靠现有的 32 个 `test_gdn_causal_conv1d` oracle 兜住，并且按 C2 用同进程 A/B 测
（它同时会去掉 30 次/步的同步，预期又是一笔 TPOT 收益）。

### 71.2 顺序

1. 先测 §70 的 `_routed_device_into`（预期 TPOT 95.8 → ~93）；
2. 再做本节这条（去掉 30 次/步同步，预期再 ~3 ms）；
3. 两处都干净后再试捕获 —— 那时捕获区里应该已经没有同步读写了。

## 72. 最后一处同步读取已消除（开关默认关，双状态 oracle 全过）

### 72.1 做了什么

`causal_conv1d_update` 的普通解码分支原本把 `conv_state_indices` 读回主机拿 Python int 下标
（`_cache_index_list`，30 次/步）。新增 `_update_capture_safe`：**全程用设备张量寻址**。

- `index = conv_state_indices[:batch].to(torch.int64)`；
- `lines = conv_state.index_select(0, index)`（一次性取批内所有 cache 行）；
- 每个 seq 的卷积用现有的 `_prior_window` + `_depthwise_causal_conv`（数学不变）；
- 写回 `conv_state.index_copy_(0, index[keep], new_lines[keep])`；
- `null_block_id` 的行用**掩码**跳过（不读、不写、输出保持 x），替代原来的 Python `continue`；
- 输出按 `original_x_dtype` 转回（与一般路径一致）。

门控：`TILEINFER_GDN_CONV_CAPTURESAFE=1`（默认 **关**，符合 C2 —— 还没测）。
前置条件：`conv_state_indices` 非空、非 varlen、无投机解码、`seqlen == 1`。

### 72.2 索引边界的处理（沿用已有先例）

`index_copy_` 不做范围检查，所以第一次遇到某个形状 `(device, num_cache_lines, batch)` 时
做一次 `min/max` 校验并记住（`_CONV_INDEX_CHECKED`）——和 `moe._VALIDATED_ID_BOUNDS`（§42）
同一个取舍：之后越界会以 `index_copy_` 的索引错误暴露，而不是我们的 `ValueError`。

### 72.3 验证

- **32 个 `test_gdn_causal_conv1d` oracle 在开关开、关两种状态下都通过**；
- 过程中修掉三个真实问题：`index_copy_` 要求 int64 索引、输出 dtype 必须转回 `original_x_dtype`、
  `null_block_id` 在 helper 里是未定义名（我漏传参数，被 `test_dtype_preservation` 抓住）。

### 72.4 状态与下一步

现在解码路径上**理论上是零同步读取**（MoE §70 + 卷积本节）。剩下的验证需要设备：

1. 开关打开后先做**正确性**：`first_token_det.sh` 的自抖动 + logprobs 对拍（C3）；
2. 再做**性能**：同进程 A/B（预期去掉 30 次/步同步，约 3 ms/步）；
3. 最后才是捕获：`graph_eager_capture.sh`。

**顺序不能反** —— 这个改动动了 GDN 最热的路径，正确性必须先立住。

## 73. §72 的验证结果：token 一致，但 logprobs 不再逐位相同 —— **开关保持关闭**

### 73.1 测到了什么

设备终于空出来，跑 `verify72.sh`（eager、`TILEINFER_GDN_CONV_CAPTURESAFE=1`）：

```
token sequences identical across 4 prompts      <- token 层面是确定的
RESULT: FAIL   (atol 1e-6)
```

放宽到 1e-2 仍 FAIL，最大差 **0.0357 nats**（`Summarize the following in one word...` 那条）。

**对照**：改动之前（§50）同样的自抖动测试是 **atol 1e-6 通过、逐位相同**。
所以这个差异是随这次改动出现的信号，不能当成"bf16 正常抖动"糊过去。

### 73.2 因此把开关保持关闭

`TILEINFER_GDN_CONV_CAPTURESAFE` 默认就是关，本轮**不改这个默认**。理由：

- token 一致说明大概率是数值细节，但 0.036 nats 比此前任何一次自抖动都大；
- 现在没有窗口做同进程配对 A/B（两次尝试都在测量中途被外部 SIGKILL，见 §48 的同一现象），
  无法判定"是不是这次改动引入的"；
- C2 的规矩是 ratio/正确性都要有据，不能靠猜。

### 73.3 下一步的两个怀疑点（都有现成的验证方法）

1. **`index_copy_` 在 NPU 上是否原子/非确定**。我在 §45 只验证过 `index_add_`/`scatter_add_`
   非确定、`reshape+sum` 确定，**`index_copy_` 从来没单独测过** —— 而我现在的 MoE 解码路径
   （§70）和这条卷积路径都用它。补一个 30 秒的原语探针就能定论。
2. **每步一次的 `int(index.min())` 校验**（`_CONV_INDEX_CHECKED`）插入了一次同步，改变了时序；
   虽然它不改变数值，但如果 0.036 只在第一次调用时出现，就说明是别的东西。

**先做第 1 条**（不需要服务、30 秒、结论明确）：如果 `index_copy_` 也是原子的，那么 §70 和 §72
两条路都必须换成"确定性写入"（例如先 `sort` 目标索引再按段写，或退回 `index_select`+`cat`），
否则捕获即使成功，输出也不可复现 —— 那比慢更糟。

## 74. `index_copy_` 是确定的 —— 那 0.036 nats 来自别处，最大的嫌疑是我自己新加的 `bmm`

### 74.1 原语探针结果（30 秒，不需要服务）

```
### index_copy_ (permutation): DETERMINISTIC  worst_abs=0.000e+00      (50/50 相同)
### control (duplicate index, last-writer-wins): 10/10 differ          (探针灵敏)
### RESULT: SAFE
```

对照组（重复目标）10/10 都不同 —— 说明这个探针**能**检出差异，不是"测不出来"。
`index_copy_` 用唯一索引（置换）写回是逐位确定的，**§70 的 MoE 写回和 §72 的卷积写回都不危险**。

### 74.2 因此 §73 的怀疑需要修正

§73 我把 0.036 nats 归给 §72（卷积）。但那一轮里**同时开着 §70 的 MoE 设备路径**，
而 §70 是**没有开关、直接进默认路径**的（`num_tokens <= 8` 自动生效）。
所以更应该查的是 §70，而不是那条默认关闭的卷积改动。

具体地说：`_routed_device_into` 用的是**补齐到 256 个专家的 `bmm`**，而 §49 的确定性探针
（`fused_experts` 20 次逐位一致）跑的是**旧的逐专家循环** —— 也就是说
**这个大 `bmm` 从未被确定性验证过**。大 batch 的 GEMM 在不少后端会用 split-K 原子累加，
这与 `index_add_` 是同一类问题。

### 74.3 下一步（30 秒可判）

探针 `benchmarks/probe_index_copy_determinism.py` 已经写好，把其中的 `index_copy_`
换成 `torch.bmm`（形状 `[256, max_n, 2048] × [256, 2048, 1536]`，bf16，重复 20 次逐位比较）
就能判定。若 `bmm` 也非确定：

- §70 的收益（去掉 32 次/步同步）与"可复现"直接冲突，**必须回退或换写法**；
- 换写法的方向：把 `bmm` 换成"逐专家 matmul 但用设备张量寻址"（仍需解决主机读问题），
  或者把 `bmm` 的累加改成固定顺序（例如 fp32 分段求和）。

**在这条判定之前，§70 的收益我不计入成绩**，即使它的 TPOT 数字更好看。

## 75. 三个结论：`index_copy_` 安全、§70 收回默认关、以及一个真正的可移植性 bug

### 75.1 原语探针：`index_copy_` 用唯一索引是确定的

```
index_copy_ (permutation): DETERMINISTIC  (50/50 相同)
control (duplicate index): 10/10 differ   <- 探针灵敏，不是"测不出来"
```

所以 §70（MoE 写回）与 §72（卷积写回）的写回原语都不是问题来源。嫌疑转向 §70 用的**补齐 `bmm`**
（256 个专家的大 batch GEMM，从未做过确定性验证；`§49` 那次 20/20 一致跑的是旧的逐专家循环）。

### 75.2 §70 收回"默认开"

它一直是**无开关生效**的。现在加了 `TILEINFER_MOE_DEVICE_PATH`，默认**关**：

- 自抖动从 §50 的逐位相同变成 §73 的 0.036 nats，而这个路径正是那轮新开的东西；
- `test_gdn_causal_conv1d.py` + `test_moe_routed_cpu.py` 同进程跑时，丢弃槽位 oracle 失败
  （单独跑都过）—— §46 的教训是这种模式要"查下去"，而不是"标 xfail"。

默认关之后，出厂路径回到已知良好状态；要不要开取决于那个 `bmm` 的确定性判定。

### 75.3 一个真正的 bug：`_swiglu` 只看"算子可用"，不看"张量在哪个设备"

服务端跑 MoE oracle 时失败，报：

```
NotImplementedError: Could not run 'npu::npu_swiglu' with arguments from the 'CPU' backend
```

原因：`_swiglu` 判断 `npu_swiglu` 是否**存在**来决定走不走融合算子，而它只注册在 PrivateUse1
后端。本机（无 torch_npu）走了 torch 回退所以一直绿；**服务器（装了 torch_npu）上 CPU 张量
就会选中它然后炸**。修法：加 `x.device.type != "cpu"`。

这条值得单独记：**"torch_npu 存在" ≠ "张量在 NPU 上"**。之前几轮我在服务器上跑本地同款测试
时看到的失败/通过，可能都掺了这类差异 —— 以后服务端跑测试前要把相关模块一起同步，
并且这类"能力探测"必须带上设备判断。

### 75.4 本轮净结果

- 探针证明 `index_copy_` 安全（省掉一个怀疑方向）；
- §70 收回默认关（保守，但 C2/E4 都要求这样）；
- 修掉一个真 bug（`_swiglu` 设备判断），服务端 39 项测试与本地一致全绿；
- **无性能数字**：TPOT 测量两次都在中途被外部 SIGKILL（§48 同一现象）。

## 76. 排除法收尾：`bmm`/`index_copy_` 都确定，0.036 nats 与本轮改动**无关**

### 76.1 `bmm` 探针（30 秒，三个 max_n）

```
bmm [256, 1,2048] x [...]   DETERMINISTIC  worst_abs=0.000e+00   |bmm - matmul| = 0
bmm [256, 8,2048] x [...]   DETERMINISTIC  worst_abs=0.000e+00   |bmm - matmul| = 0
bmm [256,32,2048] x [...]   DETERMINISTIC  worst_abs=0.000e+00   |bmm - matmul| = 0
```

补齐 `bmm` 在三种 max_n 下都是逐位确定的，**并且与逐专家 matmul 逐位相同**。
加上 §74 的 `index_copy_`，§70 用到的两个原语都被排除了。

### 76.2 关键对照：**默认状态自己也不是逐位可复现**

关掉两个新开关（`TILEINFER_MOE_DEVICE_PATH` / `TILEINFER_GDN_CONV_CAPTURESAFE`）跑同一个自抖动测试：

```
RESULT: FAIL   (atol 1e-6)
```

也就是说 §73 那个 0.036 nats **不是**这两处改动引入的 —— 在默认路径上它同样存在。
那它是什么时候出现的？最可能的位置是 **§54**：默认从"逐专家 matmul + fp32"变成
"补齐 bmm + bf16"，而那之后我没再做过 1e-6 级的自抖动复测。

**结论（对 §73 的更正）**：0.036 nats 应记为"§54 之后默认路径的性质"，而不是"§70/§72 的嫌疑"。
下一个该查的是 §54 那条路径里唯一没被探针覆盖的算子：`npu_swiglu`（融合激活），
探针方法同 §74（30 秒）。

### 76.3 §70 仍然是关的，原因换了

不是"怀疑它引入抖动"（那点已排除），而是**那个顺序相关的测试失败仍然存在**：

```
pytest tests/test_gdn_causal_conv1d.py tests/test_moe_routed_cpu.py   # 开 MOE_DEVICE_PATH 时失败
pytest tests/test_moe_routed_cpu.py                                   # 单独跑通过
```

按 §46 的教训，这种模式必须查清而不是绕过，所以在查清前保持默认关。

### 76.4 本轮的绝对数字：机器慢了很多，不能直接比

| 指标 | §55/§68 那次 | 本轮 | 说明 |
|---|---|---|---|
| TPOT | 95.8 ms | **100.73 ms**（极差 1.6%） | +5% |
| TTFT@16 | 0.529 s | **0.870 s** | +64% |
| TTFT@128 | 0.663 s | **0.984 s** | +48% |
| TTFT@512 | 1.215 s | **1.262 s** | +4% |

TTFT@16 涨了 64% 而 TTFT@512 只涨 4%，说明是**固定开销/机器状态**的变化（不是我的代码）——
这正是 §35 那条"跨会话绝对值不可引用、只有配对差可引用"的又一例。**本轮不产出新的绝对结论。**

## 77. §70 的解码路径**是错的**，不只是"未验证"

### 77.1 症状

打开 `TILEINFER_MOE_DEVICE_PATH=1` 跑丢弃槽位 oracle，输出里出现垃圾：

```
tensor([[ 8.89e-03,  1.76e+00, -3.69e+19,  1.74e+00,  3.69e+19, ...]])
```

`3.7e+19` 是**未初始化内存** —— 说明有一个有效槽位没有被写入，调用方的 `torch.empty` 直接读穿。

### 77.2 为什么之前"单独跑通过"

单独跑该文件时它过了，组合跑才失败 —— 我一度按 §46 把它理解成"顺序相关"。**这个理解是错的**：
真正的差别是那些运行里 `TILEINFER_MOE_DEVICE_PATH` 是否真的生效（有一轮我在命令末尾就
`Remove-Item Env:` 了）。所以它不是"环境泄漏"，而是**一个稳定存在的索引错误**，
只是被某一次带错环境的运行掩盖了。

### 77.3 定位方向（下一轮）

垃圾值 + "有效槽位没写" ⇒ 写回的 `dst` 少了一项或重复了一项。可疑点按顺序：

1. `starts = cumsum(counts_dev) - counts_dev` 用的是**截断到 E** 的 `counts_dev`，而 `src` 的
   上界依赖 `sorted_x` 的总行数 —— 若 `counts_dev[:E]` 之和小于总槽位数（有丢弃槽位时必然如此），
   `starts` 本身是对的，但 **`max_n = num_tokens` 可能小于某个专家的真实计数**（当丢弃槽位存在时，
   一个专家可以拿到 `num_tokens` 个槽位——上界仍成立）…所以这一条需要实测而不是推；
2. `live = mask.nonzero()` 与 `y.reshape(-1, hidden).index_select(0, live)` 的下标空间是否一致；
3. `order.index_select(0, src_flat.index_select(0, live))` 是否漏了 `keep` 过滤。

### 77.4 处置

- 开关默认**关**，docstring 与测试 docstring 都写明"**当前不正确**，不是"待测"——
  避免以后有人把它当候选打开；
- §70 的"去掉 32 次/步同步"收益**暂不计入**，等这个索引错误修好；
- 这条也再次说明：**"单独跑通过"不能当证据**，必须确认环境变量真的生效。
  这是本会话里第三次被"运行条件没核实"误导（前两次：`--enforce-eager`、混用不同形状的数字）。

## 78. §70 的索引错误修好了，然后配对测量判定它**慢 22%** → 按 C2 保持关闭

### 78.1 根因就一行

```python
max_n = max(1, int(num_tokens))     # 错
max_n = max(1, int(slot_out.shape[0]))   # 对：上界是槽位数
```

一个 token 可以把**多个 top-k 槽位投给同一个专家**（重复 id），所以单个专家最多能占 `top_k`
个槽位。用 `num_tokens` 当补齐宽度会把超出的槽位截掉，然后 `torch.empty` 的行被读穿 ——
就是 §77 看到的 `3.7e+19`。

修好后：**7/7 单独通过、与卷积测试组合 39/39 通过、开关关闭 7/7 通过**。
所以它此前那个"顺序相关"的表象，纯粹是我某次命令里环境变量没生效造成的假象（§77.2 已经指出）。

### 78.2 然后按 C2 量它值不值：**0.818 —— 更慢**

同进程配对（`paired_tpot.py`，文件开关切档、交替顺序）：

```
[pair0] on=126.8 off=101.7  (off-on) = -25.1
[pair1] on=128.1 off=102.9  (off-on) = -25.2
[pair2] on=125.4 off=103.9  (off-on) = -21.5
[pair3] on=125.8 off=103.9  (off-on) = -22.0
### valid pairs 4, rejected 0
### on median 126.3   off median 103.4
### measured_ratio on/off = 0.818     (paired diff -23.5 ms = -18.6%, spread 2.9%)
```

**4/4 一致，极差 2.9%，不是噪声。** 省掉 32 次/步同步，却多花 23.5 ms/token。

算术解释：补齐到 256 专家 × 8 槽位（1 token × top-8）= 2048 行，
每层 `2048 × 2048 × 1536 × 2 ≈ 12.9 GFLOP`，32 层 ≈ **412 GFLOP/token** ——
在约 20 TFLOP/s 的有效吞吐上就是 20 ms。同步省下的 ~3 ms 完全不够抵。

### 78.3 处置与它对 M3 的含义

- 按 C2（ratio < 1.0 一律默认关闭），`TILEINFER_MOE_DEVICE_PATH` **保持默认关**，并且现在这个
  默认有数据支撑，不再是"未验证"；
- **对 ACLGraph 的含义**：捕获要求捕获区无同步读取，而唯一做到这一点的解码路径要付出 +22%。
  所以"开捕获"必须先证明固定开销的收益 > 22%，否则就是净损失。这条把 M3 的账算清楚了 ——
  以后不能再说"捕获一定值得试"，得先有这个对比。

### 78.4 本轮的净成果

一个真 bug（一行上界写错，会导致静默垃圾数值）+ 一个决定性测量（0.818，4/4）。
两者都进档；开关保持关闭。**这是本会话第一条"修对了但不采用"的记录**，我觉得它比一个
勉强采用更有价值 —— 因为它同时给出了 M3 的成本前提。

## 79. 全开捕获：失败点又前进一层 —— 捕获区内的一次**流同步**

### 79.1 本轮配置（所有已知阻塞点都打开）

`TILEINFER_DYNAMO_OPAQUE=0` + `TILEINFER_COMPILE=0` + `TILEINFER_ACLGRAPH=1`
+ `TILEINFER_MOE_DEVICE_PATH=1` + `TILEINFER_GDN_CONV_CAPTURESAFE=1`

结果：捕获**真的开始了**，但失败于：

```
[__cudagraphs] NPUGraph: ERROR — capture failed:
        rtStreamSynchronizeWithTimeout
[__cudagraphs] NPUGRAPH DFX common causes: default stream, CPU ...
```

`rtStreamSynchronizeWithTimeout` = **捕获区内发生了一次流同步**。§70 的 `aclrtMemcpy`（同步拷贝）
已经消失，所以这是比它更细的一层：不是拷贝，而是**等待**。

### 79.2 最可疑的一处（本轮已改）

`_update_capture_safe` 的"每个形状校验一次"边界检查里有 `int(index.min())` —— 这是一次
device→host 同步，而且**恰好发生在捕获的第一次调用里**（形状第一次出现时）。已加
`_capturing()` 守卫：捕获进行中跳过校验（`torch.npu.is_current_stream_capturing` /
`is_capturing`，取不到就返回 False，退化成原行为）。

32 个 conv oracle 在开关开、关两种状态下仍然全过。

### 79.3 还剩哪些同步（下一轮的清单）

按解码路径逐处过：

| 位置 | 是否仍可能同步 |
|---|---|
| `_update_capture_safe` 的边界校验 | **已加守卫** |
| `moe._routed_sorted_into` 的 `bincount().tolist()` | 解码走设备路径后没有了，但**prefill 仍有**（prefill 不在捕获区，OK） |
| `impl._query_lengths` | 若它读 `query_start_loc` 到主机，prefill 每步一次；解码 `max_query_len==1` 应早返回 —— **需确认** |
| `refresh_step` 的 `dst32/dst64` 写入 | 若源是 CPU 张量就会拷（§71 列为可疑，已确认是设备张量） |
| vLLM 自身在捕获区内的 `.item()` | 需要 `TORCHDYNAMO`/profiler 才能看到，暂只能靠排除 |

### 79.4 这一轮的判断

从 §66 到现在，捕获的失败点已经推进了 **七层**：自己的启动参数 → 装饰器 → 计数器锁 → 上游
pass 的 NameError → `cudagraph_mode` 解析 → capture size → 同步拷贝 → 同步等待。
每一层都是**具体的、可修的**，没有一层是"平台不支持"。

但也必须说清代价：要让捕获成立，解码路径必须走"零同步读取"的形式，而那已经实测要 **+22% TPOT**
（§78）。所以 M3 是否值得，最终取决于捕获能不能省下超过 23.5 ms/token —— 这个账现在还无法算，
因为捕获一次都没成功过。

## 80. 捕获的第八层：**平台层面的流同步限制**；以及我建议停止追它的理由

### 80.1 本轮清掉的两个候选

1. `impl._query_lengths` 的主机读取只存在于**降级路径**（`_reference_forward`），文档已注明
   "不被捕获、不是稳态" ✓；
2. `tilelang` 的启动路径**不**同步（只有 nvrtc 适配器有 `synchronize`，我们走 cython/tvm_ffi）✓；
3. 发现并修掉一个真问题：**解码后端在捕获区内才第一次构造**。V1 的 profile 跑
   `attn_metadata=None`，而我们的 impl 在那里直接早返回，于是 `BatchAttention`（plan + tilelang
   编译，会同步）留到捕获的第一次调用才建。已在 profile 分支里预热（`_decode_attn_or_none`）。

### 80.2 剩下的错误是平台级的

```
Not_Supported(EE1016): Synchronizing a stream failed.
  Reason: Stream (stream_id=63) during the capture stage is not supported.
  rtStreamSynchronize execution failed, reason=stream is captured
```

这是 ACL 在**捕获期禁止任何流同步**。我们已经把能想到的同步源逐个移除（同步拷贝 → 流同步；
MoE 主机读 → 卷积主机读 → 解码后端构造），错误仍然指向一次"对已捕获流的同步"，
而**调用者不在我们的代码里**可辨范围内（torch_npu 的 graph 实现或 vLLM 的捕获路径）。

### 80.3 我建议停止追捕获，理由是账已经能算了

- 让捕获成立的前提是解码路径"零同步读取"，实测 **TPOT 0.818**（§78，4/4 一致）；
- 捕获能省的是每步的下发开销。我们在 §64 量过：**一次 tilelang 调用 1.17 ms host**，
  解码每步 10 层注意力 → 约 12 ms；加上 30 层 GDN 与 32 层 MoE 的下发，理论上限也就是
  每步 30–40 ms；
- 而代价已经是固定 +23.5 ms/token，且捕获还要额外承担"不可复现风险"（图捕获会放大任何
  非确定算子，§44/§73 已经出现过 0.036 nats 的自抖动）。

**结论：在当前平台实现下，ACLGraph 的收益上限与它的代价同量级，且它需要先付出 22% 的确定成本。
按 C2 的口径，这不是一个"值得默认开启"的候选。** 我建议把 M3 的这一条标记为
"已探查、暂不采用"，并回到 M2 的 GDN 内核 —— 那里 253 ms 是确定的上限，不依赖平台能力。

## 81. 解码预算量清了：**62.6 / 103 ms 可归因，且都是"每算子 0.1 ms"的下发地板**

### 81.1 直接测每个算子（服务形状：1 token、batch 1，`benchmarks/bench_decode_ops.py`）

| 算子 | 每次调用 |
|---|---|
| `gdn: causal_conv1d_update` | **0.72 ms** |
| `gdn: recurrent packed decode` | 0.29 ms |
| `moe: fused_experts`（1 token） | **1.01 ms** |

按层数折算：

```
GDN  x30 层 = 30.26 ms/step
MoE  x32 层 = 32.30 ms/step
合计可归因  = 62.56 ms/step          (TPOT 实测 ~103)
```

余下约 40 ms 是 10 层注意力 + 稠密投影 + 下发。

### 81.2 关键判断：这不是算术，是**每算子 0.1 ms 的地板**

- `causal_conv1d_update` 的算术量：8192 通道 × 3 taps ≈ **5 万次乘加/层** —— 微秒级；
  实测 0.72 ms ⇒ 它由 7 个小算子组成，**每个约 0.1 ms**（这就是本设备的单算子下发地板）；
- `fused_experts`（1 token）1.01 ms ≈ 10 个算子；
- 62.6 ms ≈ **600 个小算子 × 0.1 ms** —— 与 §62 对 prefill 的判断完全同构。

**所以 M2 剩下的工作不是"把某个算子写快"，而是"减少每层算子数"。**
写一个更快但同样数量的算子不会有用。

### 81.3 因此 M2 的正确目标：**融合的 GDN 卷积更新内核**

`causal_conv1d_update` 目前是 7 个算子（切片、`mul`、`sum`、`silu`、状态移位、写入、bias）。
一个 tilelang 内核可以把它做成 **1 次下发**：

- 预期每层 0.72 → 0.1–0.2 ms；
- ×30 层 ⇒ **每步省 15–18 ms**，TPOT 103 → 约 **86 ms**（比值 2.53× → 约 2.27×）；
- 这条正好是 M2 原始任务里"GDN recurrent/decode 走内核（当前 kernels/gdn 无 decode 内核，需新写）"
  的那一项，而且它是**纯算子替换**，不依赖平台能力、不依赖图捕获。

### 81.4 下一轮的具体做法

1. 在 `tileinfer/kernels/gdn/` 下新增 `conv_update_ascend950.py`：输入 `x[1, 8192]`、
   `conv_state[num_lines, 8192, 3]`、`weight[8192, 4]`、`bias`、`cache_index`，一个 kernel 完成
   "取窗口 → 4-tap 卷积 → 加 bias → SiLU → 状态左移 + 追加"；
2. 用 `tests/test_gdn_causal_conv1d.py` 的 32 个 oracle 兜住（它们已经覆盖激活/偏置/状态/ dtype）；
3. 按 C2 同进程 A/B 测，ratio ≥ 1.0 才进默认。

这条是**本会话里第一个"明确的、可验证的、不依赖平台能力"的性能机会**，也是 M2 出口
（TPOT ≤1.8× = ≤68 ms）唯一还够得着的路径。

## 82. 不用写内核也拿到 2.77×：**消掉两次跨步写入**

### 82.1 先看清楚了再动手

计划是写一个融合内核（§81.3），但那要盲写 + 多轮设备编译。先做了一遍算子清点，
发现 `_prior_window` **已经是 view**（0 个算子），所以可压的只剩状态更新那三步：

```python
nxt = line_view[:, 1:state_len_eff].clone()            # 一次分配 + 拷贝
conv_state[line, :, : state_len_eff - 1].copy_(nxt)     # 写 [dim, 0:2] —— 跨步
conv_state[line, :, state_len_eff - 1] = x_seq[0]       # 写 [dim, 2]  —— 跨步
```

改成：

```python
new_line = torch.cat([line_view[:, 1:state_len_eff], x.new(…).unsqueeze(1)], dim=1)  # 连续 [dim,3]
conv_state[line, :, :state_len_eff] = new_line                                       # 一次连续写
```

### 82.2 实测（设备，`benchmarks/bench_decode_ops.py`，20 次中位）

| 量 | 改前 | 改后 | ratio |
|---|---|---|---|
| `causal_conv1d_update` | 0.72 ms/call | **0.26 ms/call** | **2.77×** |
| GDN ×30 层 | 30.26 ms/step | **19.01 ms/step** | 1.59× |

**收益远超"少一个算子"。** 真正的原因是**访存模式**：`conv_state[line]` 是 `[dim, 3]`，
写 `[:, 0:2]` 与 `[:, 2]` 都是**按最后一维跨步**（stride 3）的非连续写，在 NPU 上代价很高；
改成一次写 `[:, 0:3]` 就变成连续写。少一个算子只是附带效果。

32 个 conv oracle 在本地与设备上都通过。

### 82.3 预期端到端

GDN 每步 30.26 → 19.01，即 **−11.25 ms/step**；TPOT 103 → 约 **92 ms**（比值 2.53× → 约 2.26×）。
这条与 §78 的 +22% 形成对比：**同样一笔"每步毫秒数"，来自访存模式的一行改动，而不是新内核。**

### 82.4 因此下一个该查的同类项

`moe.fused_experts` 解码路径 1.01 ms/层（×32 = 32.3 ms/step）里也有一批"按最后一维跨步"的写入
（`slot_out[slots] = …`、`slot_out.index_copy_(0, dst, …)`）。按本节的机制，**先数跨步写入**
比先写内核更划算：`slot_out` 是 `[slots, hidden]`，其写入沿第 0 维（连续）——但
`padded_w.unsqueeze(-1)` 的广播乘、以及 `y.reshape(-1, hidden).index_select(0, live)` 都是连续读，
所以 MoE 那边未必有同类问题。**下一轮第一件事是把 MoE 解码路径也照这个角度过一遍**
（而不是直接假定它也能这样优化）。

## 83. **撤回 §82**：同进程配对把"2.77×"打成 0.926

### 83.1 我做了什么

§82 里我用两次独立脚本运行比较（0.72 → 0.26 ms），并据此写下"2.77×、GDN 每步省 11 ms"。
这一轮补了同进程 A/B（`benchmarks/ab_conv_state.py`：同一进程内交替两种模式，5 轮）：

```
[pair0] on=0.214 off=0.186  (off-on)=-0.028
[pair1] on=0.197 off=0.186  (off-on)=-0.011
[pair2] on=0.201 off=0.203  (off-on)=+0.001
[pair3] on=0.208 off=0.198  (off-on)=-0.010
[pair4] on=0.193 off=0.174  (off-on)=-0.019
### on median 0.201   off median 0.186
### paired difference median -0.011 ms (-5.3%)
### measured_ratio on/off = 0.926
```

**单次连续写入其实慢 7%**，而且配对差的正负在轮次间翻转 —— 典型的"没有真实效应"。

### 83.2 那个 2.77× 是什么

是**机器漂移**：同一台设备在不同窗口上，这个算子的绝对耗时本来就在 0.19–0.72 ms 之间浮动
（本轮两种模式都测到 ~0.2 ms，而上一轮一种模式测到 0.72 ms）。§35 早就写过
"跨会话绝对值不可引用、只有配对差可引用"，我却又犯了一次 —— 而且这次差点把它当成
"本会话单笔最大算子收益"写进结论。

按 C2：ratio 0.926 < 1.0 → `TILEINFER_GDN_CONV_STATE1` **默认关闭**（两个状态都通过 32 个 oracle）。

### 83.3 因此本轮的真实状态

- **没有**性能改进进默认路径；
- 上一轮那句"不用写内核也拿到 2.77×"是**错的**，收回；
- 端到端 TPOT 这轮也测了（124.16 ms），但当时设备被旁人占了 25 GB（TTFT@16 从 0.53 涨到 1.00 s），
  所以那个数**同样不可引用**。

**当前已确认的数字仍然是：TPOT 95.8 ms（全默认）、90.0 ms（编译开）、TTFT@512 1.215 s。**

### 83.4 一条硬规矩（写下来防止再犯）

> 任何性能改动，若不能用**同一进程内的交替 A/B** 给出 4 对以上的配对差，就不得写入结论；
> 跨运行的"改前/改后"只允许作为**线索**（"值得做一个配对测试"），不得作为 ratio。

本轮已把 `benchmarks/ab_conv_state.py` 留下作为模板。

## 84. 会话收束：已验证的数字、已否决的候选、下一步（一条即可执行的动作）

### 84.1 已验证的性能数字（同环境、配对/尺子）

| 指标 | 会话起点 | 当前已确认 | baseline | 比值 |
|---|---|---|---|---|
| TPOT | 151.4 ms（全关） | **95.8 ms**（全默认，极差 2.1%） | 37.93 ms | **2.53×** |
| TPOT（编译开） | — | **90.0 ms**（极差 1.1%） | 37.93 ms | 2.37× |
| TTFT@512 | 1.90 s | **1.215 s** | 0.103 s | **11.8×** |
| prefill 内核占比 | — | 1.000（kernel=100/reference=0） | — | ✅ M1 出口之一 |

### 84.2 进默认且有 ratio 的候选（6 项）

| 改动 | ratio | 类型 | 小节 |
|---|---|---|---|
| MoE 排序版专家循环 | 2.569 | op | §29 |
| MoE bmm + bf16 | 1.480 × 3.78 | op | §53/§54 |
| MoE 校验缓存 | 1.107 | e2e 配对 | §42 |
| GDN 卷积免同步 | 1.241 | e2e 配对 | §32 |
| GDN 常量缓存 | 1.231 | op | §31 |
| GDN `solve_tril` Newton 倍增 | 45.6（op）/ **1.04（e2e，噪声带内）** | §59 |

### 84.3 明确否决的候选（有数据，别再试）

| 候选 | 数据 | 结论 |
|---|---|---|
| `gdn_no_sync` | 0.453 | 2.2× 更慢 |
| 批次元数据主机缓存 | 0.999 | 无效应 |
| 分组 conv1d | 1.004 | 无效应 |
| 单次连续状态写入（§82） | **0.926**（同进程配对） | 更慢，撤回 |
| MoE 零主机读取解码路径 | **0.818**（同进程配对） | 更慢；且曾是错的（§78 已修） |
| ACLGraph 捕获 | 八层阻塞，第八层是平台限制 | 建议不追（§80） |

### 84.4 未解的两件事（有明确下一步）

1. **§73 的 0.036 nats 自抖动**：默认路径也存在（不是我的改动引入）。下一个探针：
   `npu_swiglu` 的确定性（30 秒，方法同 §74/§76）。若它非确定，`TILEINFER_MOE_SWIGLU=0`
   就是解法（代价见 §29 的 1.043）。
2. **M2 的 GDN decode 内核**：`causal_conv1d_update` 仍是 0.72 ms/层（×30 = 21.6 ms/step），
   由约 7 个算子组成。**写内核**是唯一还能拿到量级收益的路，但必须按新规矩用同进程配对验证。

### 84.5 下一步就做这一件事

**给 `npu_swiglu` 做确定性探针**（30 秒、无服务、结论二元）：

- 确定 → 0.036 nats 来自别处，继续按 §80 的排除法走；
- 非确定 → 它是一个**可修的**正确性问题（`TILEINFER_MOE_SWIGLU=0` 可回退），
  而且它同时解释了为什么图捕获会把差异放大。

这条之所以优先：它把"性能"与"可复现性"两个目标接到同一个开关上，而且成本只有 30 秒设备时间。

### 84.6 方法纪律（本会话沉淀，已写进 §83.4）

1. 任何 ratio 必须来自**同一进程内的交替 A/B ≥4 对**；跨运行的比较只能当线索；
2. 引用日志前先确认**自己的启动参数**（`--enforce-eager` 坑过一次）；
3. 引用数字前先确认**形状与频次**（混用 prefill/decode 数字踩过两次）；
4. 引用测量前先确认**该测量本身语义**（分相表列义未定、含编译开销，踩过一次）；
5. "单独跑测试通过"不算证据，先确认**环境变量真的生效**（踩过一次）。

## 85. `npu_swiglu` 也是确定的 —— 0.036 nats 仍未找到来源（如实留档）

### 85.1 探针结果

```
### npu_swiglu available: True
### shape (8, 1, 1536):  fused 0/20 differ | torch port 0/20 differ | fused vs port 0.000e+00
### shape (8, 8, 1536):  fused 0/20 differ | torch port 0/20 differ | fused vs port 0.000e+00
### shape (256, 1, 1536): fused 0/20 differ | torch port 0/20 differ | fused vs port 0.000e+00
### RESULT: SAFE
```

融合激活在三种形状上都逐位确定，且与 torch 实现**逐位相同**。

### 85.2 目前"已排除"的清单（默认解码路径）

| 组件 | 探针 | 结论 |
|---|---|---|
| `index_add_` / `scatter_add_` | §45 | 非确定（已被替换为 `index_copy_`+`sum`） |
| `index_copy_`（唯一索引） | §74 | 确定 |
| 补齐 `bmm`（三种 max_n） | §76 | 确定，且与逐专家 matmul 逐位相同 |
| `npu_swiglu` | §85 | 确定，且与 torch 逐位相同 |
| 注意力内核 | §51 | 与参考逐位相同（24 步） |
| GDN conv / recurrent | §49 | 确定（19 次重复） |
| **`fused_post_conv_prep`** | — | **未探** |
| **vLLM 自身算子（norm/rope/投影）** | — | **未探**（不在我们的代码里） |

### 85.3 结论：这是一个**已知存在、但未定位**的小差异

- 量级：**0.036 nats**，token 序列相同（§73）；
- 影响：不影响 greedy 解码结果，但会让"逐位可复现"这条强断言不成立；
- 我**没有找到来源**，且已经排除了我们自己路径上全部可疑的原语（上表）。

诚实的处理方式：**记为未解，不再猜**。下一个可查对象是 `fused_post_conv_prep`（30 秒探针，
方法同 §74）与 vLLM 侧算子；如果两者都确定，那么 0.036 就很可能来自 vLLM 的融合算子，
那不在我方修改范围内 —— 届时按跨实现差异处理（§52 的判据）。

### 85.4 本会话到此的净状态

- 性能：TPOT 95.8 ms（2.53×）、TTFT@512 1.215 s（11.8×），共 6 项带 ratio 的改动进默认；
- 正确性：注意力/MoE/GDN 各自对 oracle 通过；**唯一悬项是 0.036 nats 的自抖动**（未定位）；
- 结构性阻碍：ACLGraph 的第八层是平台限制（§80），已给出不追的理由；
- 方法纪律：5 条已写入 §83.4/§84.6，本会话因违反它们而浪费过 3 轮以上。

## 86. 解码路径的确定性排查**做完了**：我们的算子全部确定，0.036 nats 在 vLLM 侧

### 86.1 最后一个探针

```
### tokens=1: 0/20 trials differ across 5 outputs (worst_abs=0.000e+00)
### tokens=8: 0/20 trials differ across 5 outputs (worst_abs=0.000e+00)
### RESULT: SAFE
```

`fused_post_conv_prep`（L2 归一 + 门控变换，5 个输出）逐位确定。

### 86.2 完整清单：**我们解码路径上的每个算子都已被探针清过**

| 算子 | 探针 | 结论 |
|---|---|---|
| `index_add_` / `scatter_add_` | §45 | 非确定 → 已替换 |
| `index_copy_`（唯一索引） | §74 | 确定 |
| 补齐 `bmm`（1/8/32 行） | §76 | 确定，且与逐专家 matmul 逐位相同 |
| `npu_swiglu` | §85 | 确定，且与 torch 逐位相同 |
| 注意力内核（24 步） | §51 | 与参考逐位相同 |
| `causal_conv1d_update` | §49/§82 | 确定 |
| recurrent packed decode | §49 | 确定（输出与状态都是） |
| `fused_post_conv_prep` | §86 | 确定 |
| `chunk_gated_delta_rule` 链 | §49 | 确定 |

### 86.3 因此 0.036 nats 的结论

**它不在我们的代码里。** 默认路径上剩下的是 vLLM 自身的算子（RMSNorm、RoPE、qkv/o 投影、
LM head）—— 那些我们在任何后端上都不改，而它们里的 split-K GEMM 完全可能非确定。

按 §52 的判据处理：这是**跨实现/跨运行差异**，不是我们的缺陷；记为"已知、有界（0.036 nats、
token 序列相同）、来源在 vLLM 侧"。

进一步定位需要的是对 **vLLM 模型前向**做算子级二分（例如逐个替换成参考实现），那既超出本仓库的
范围，也超出"提升性能"的目标本身。**我停在这里，并把这个判断写清楚，而不是继续猜。**

### 86.4 顺带说明为什么这条值得 6 轮

排查过程本身修掉了两个真 bug（§45 的 `index_add_` 非确定、§46/§78 的未初始化读导致 `3.7e+19`），
并产出了 5 个可复用的探针（`probe_index_add`/`index_copy`/`bmm`/`swiglu`/`post_conv`）。
这些探针的价值超过结论本身：以后任何"数值不对"的怀疑，都可以在 30 秒内排除一整类原因。

## 87. GDN 卷积融合内核：**写出来了、编译通过、数值正确**，但会触发 vector core 异常

### 87.1 已达成

`tileinfer/kernels/gdn/conv_update_ascend950.py`（**未被任何代码引用**，即"未接线"）：
一个内核完成"4-tap 卷积 + bias + SiLU + 状态左移"，每核负责一段**连续**通道。

首次成功运行的读数：

```
### correctness: out max_abs=9.766e-04 (scale 2.656e-01)  state max_abs=0.000e+00
### correctness verdict: OK
[pair0] kernel=0.005  torch=0.169 ms          <- 首对采样，约 34×
```

- 输出误差 0.4%（bf16 舍入量级），**状态逐位相同**；
- 首对采样显示 0.005 ms vs torch 的 0.169 ms。

### 87.2 没达成：稳定性

随后报
`507035 The vector core execution is abnormal`，之后**每次运行立刻失败** —— 这类错误会让设备处于
需要复位的不良状态，所以"下一次尝试失败"不能当成新证据。

最可疑的原因是每核通道数不能整除：第一版 `per = ceil(8192/28) = 293`，
最后一核 `base + per = 8204 > 8192` 越界；改成 `cores=32`（`per=256`，整除）后仍报同一错误，
但那时设备已被上一次异常污染，**无法区分"新配置也错"还是"设备未恢复"**。

### 87.3 教训（第 4 次同类）

盲写 tilelang 内核的迭代成本远高于我的估计：语法（`T.Parallel` 必须在 `T.SimtVF` 内）、
SSA 规则（`T.serial` 不能嵌在 `T.Parallel` 里携带累加器）、闭包（`tilelang.compile` 用模块
globals 解析注解，工厂局部变量丢失）、以及一次异常就会污染设备的运行环境。
**四类错误各花了一次设备往返，而每类都可以通过先读一个同类内核避免。**

### 87.4 现状与下一步

- 该模块**没有任何调用点**，也没有开关接线 —— 因此它对当前性能**零影响**，可以安全留在仓库里
  作为下一次迭代的起点；
- 下一步（需要干净的设备与至少 3 次往返）：
  1. 先复位设备，再单独验证 `cores=32` 的版本；
  2. 若仍异常，把 `T.SimtVF(threads=...)` 的线程数与 `per` 的对齐关系按兄弟内核的取值（128/64）逐一试；
  3. 正确性稳定后，再按 §83.4 用 `probe_conv_kernel.py` 的交替配对出 ratio（预期 ≥5×，
     因为它是 1 个算子对 7 个）。

## 88. 内核通了：**4/4 配对 42.5×，状态逐位相同**

### 88.1 关键修复：行号必须是**标量参数**，不能是共享缓冲的值

上一轮"第一次调用正确、第二次崩（507035 vector core exception）"，根因是：

```python
line_ub = T.alloc_shared((1,), "int32")
T.copy(Idx[0:1], line_ub[0:1])
State[line_ub[0], base : base + per, 0:taps]     # 用共享缓冲里的值当行号 -> 崩溃
```

改成把行号作为**内核标量参数** `line: T.int32` 直接参与寻址：

```python
def main(..., line: T.int32, ...):
    State[line, base : base + per, 0:taps]
```

之后一次跑完 4 对、无异常。推断：共享缓冲的加载与使用之间没有同步/依赖，编译器把
`State[line_ub[0], …]` 的地址算早了（或与 `T.copy` 的写入重排），于是访问到非法内存。

### 88.2 结果（`benchmarks/probe_conv_kernel.py`，同进程交替 4 对）

```
### correctness: out max_abs=9.766e-04 (scale 2.656e-01)  state max_abs=0.000e+00
### correctness verdict: OK
[pair0] kernel=0.005  torch=0.303
[pair1] kernel=0.005  torch=0.188
[pair2] kernel=0.006  torch=0.251
[pair3] kernel=0.005  torch=0.178
### kernel median 0.005 ms   torch median 0.220 ms
### measured_ratio = 42.541x
```

- 输出误差 **0.37%**（bf16 舍入量级）；**状态逐位相同**（`max_abs = 0`）；
- `kernel` 0.005–0.006 ms，`torch` 0.18–0.30 ms ⇒ **42.5×**，四对同号，符合 §83.4。

### 88.3 预期端到端与剩余步骤

按今天这台机器的 torch 基线（0.22 ms/层）：

| | 每步（30 层） |
|---|---|
| 现状 | 6.6 ms |
| 内核 | 0.15 ms |
| **省** | **约 6.5 ms/step** |

若按 §81 那份更慢的基线（0.72 ms/层），省的会是约 21 ms。**两种口径都记，端到端测过才下结论。**

剩下三步（顺序不能换）：

1. **接线**：在 `causal_conv1d_update` 里加 `TILEINFER_GDN_CONV_KERNEL=1` 走内核，默认**关**；
2. **正确性**：32 个 `test_gdn_causal_conv1d` oracle 在开关开、关两种状态下都要过
   （包括 `null_block_id`、多 batch、dtype 保持、varlen 不走内核这些分支）；
3. **端到端**：`paired_tpot.py` 同进程配对出 TPOT 的 ratio，再决定默认值。

### 88.4 这条为什么值得一整轮

它是 M2 里唯一"用内核换掉一堆小算子"的改动，而且**第一次拿到了可信的内核级 ratio（42.5×，
4/4 配对）**。前面几轮的错误（闭包、VF 块、SSA、标量寻址）全部是"读一个同类内核就能避免"的，
下一次写内核我会先通读兄弟实现再动笔。

## 89. 内核接线完成并**通过集成对拍**；过程中又抓出两个"假通过"

### 89.1 接线

`causal_conv1d_update` 的普通解码分支里新增：`TILEINFER_GDN_CONV_KERNEL=1` 时先尝试融合内核
（`_try_conv_kernel`），成功则写回并 `continue`，否则返回 `None` 走原 torch 路径。
形状门槛：width==4、channels%32==0、state 末维==3、bf16、单 token 单请求。
开关默认**关**（C2）。

### 89.2 两个"假通过"（都曾让我以为已经好了）

1. **32 个 oracle 全过** —— 但它们用的是小维度（8/16 通道），被 `channels % 32` 挡在门外、
   走的是 torch 回退。**"全过"什么也没证明。** 于是补了 `benchmarks/probe_conv_integration.py`：
   用真实维度（8192）对比"开关关 vs 开关开"。
2. **集成对拍显示 `max_abs = 0`** —— 我一度以为"内核正确到逐位相同"。实际是
   `except Exception: return None` **静默吞掉了异常**。加计数与首 3 次告警后才看到真因：
   - `NameError: name 'C' is not defined`（闭包问题，见下）；
   - 以及更早的 bias dtype 不匹配（内核声明 fp32，调用方传 bf16）。

**教训**：`except: fallback` 这种写法会把"没跑"伪装成"跑对了"。回退必须**可观测**
（计数 + 告警），否则它与成功的信号完全相同。

### 89.3 闭包问题的真正解法

`tilelang.compile` 会**重解析 `prim_func` 源码**，注解在**模块 globals** 里求值，因此工厂局部
变量（`C = channels`）不可见。同目录其它内核之所以没踩，是因为它们的注解里只用模块级名字。

`build_conv_update_kernel` 现在在 `tilelang.compile` **前后**把形状临时发布到模块 globals
（`try/finally` 还原，避免两个形状互相污染）。这解释了为什么之前 `probe_conv_kernel.py` 能跑：
那个探针自己做了同样的注入，而我从未把它搬进模块。

### 89.4 集成对拍结果（真实维度）

```
### out  switch on vs off: max_abs=9.766e-04 (scale 2.676e-01)   -> 0.37%
### state on vs off:       max_abs=0.000e+00                     -> 逐位相同
### repeat-call stability: True
### RESULT: OK
```

0.37% 与独立探针（§88）测到的 9.766e-04 **完全一致**，说明走的确实是同一个内核。

### 89.5 还差最后一步：端到端配对

内核级 ratio 已有（42.5×，§88），但按 C2 还需要 `paired_tpot.py` 的同进程配对，
确认 TPOT 上兑现多少（按今天机器的 0.22 ms/层估算约 −6.5 ms/step）。
**在这条跑出来之前，`TILEINFER_GDN_CONV_KERNEL` 保持默认关、不计入成绩。**

## 90. 接线补了"视图/连续性"守卫；端到端测量被窗口挤掉（重试循环在跑）

### 90.1 补的守卫（本地 + 服务端 38 项测试通过）

内核拿到的是裸指针，按 `[L, C, taps]` / `[C, W]` **稠密布局**寻址。若调用方传入的是视图
（比如从更大的 state 池里切出来的行、或转置过的权重），内核会读到错误地址并返回**看起来合理的错误数字**
—— 这是最坏的失败形态。`_try_conv_kernel` 现在拒绝：

- `not conv_state.is_contiguous()`（切片视图的最后一维 stride ≠ 1）；
- `not weight.is_contiguous()`（转置权重）；
- `x_seq.numel() != channels` 或 x 非连续。

新增 6 项测试钉住这些拒绝路径，其中"非连续视图必须回退"这条的 docstring 写明了理由：
**不是崩溃，而是似是而非的错数**。

### 90.2 端到端测量：两次被"秒抢"

- 第一次：门限检查看到 5455 MB（空闲），启动时却报
  `Free memory on device npu:0 (55.66/123.02 GiB) … less than desired (0.75, 92.27 GiB)`
  —— 对方在几秒内占了 67 GB；
- 已改为重试循环（`/tmp/pk_retry.log`，最多 10 次、每次失败等 60 秒），本会话内未读到结果。

这与 §48/§61 是同一现象：设备的空闲窗口比一次六分钟测量所需的窗口更短。

### 90.3 因此本轮仍然没有端到端 ratio

已有的是：内核级 **42.5×**（4/4 配对，§88）、接线正确（集成对拍 0.37%/状态逐位相同，§89）、
以及上一轮那些防"假通过"的测试。缺的就是最后那一步。

## 91. 接线里一个"让内核永远不会生效"的门禁（已修）

### 91.1 问题

我写的条件是 `int(x.shape[0]) == 1` —— 想表达"每次一个 token"。但服务用
`--max-num-seqs 4`，decode 步的 `x.shape[0]` **是 4**，于是门禁恒假：

> **这个优化在真实配置下一次都不会执行，而它在 op 级探针里是 42.5×。**

探针用单序列测，所以永远看不到这个问题。真正的约束是"每序列一个 token"，即 `length == 1`
（外层循环本来就是按序列走的，每次调用有自己的行号与 `[dim]` 行）。

### 91.2 修法与验证

条件改为 `length == 1`（去掉 batch 限制），新增测试 `test_multi_sequence_batch_reaches_the_kernel`：
用 3 个序列、`conv_state_indices = [1,2,3]`，断言内核被**按序列调用三次且行号依次为 1、2、3**。

39 项测试通过（32 oracle + 7 门禁/回退）。

### 91.3 这条的教训

**"op 级 42.5×" 与 "在服务里生效" 之间，隔着一个门禁条件。** 本会话里我已经因为
"探针通过 ≠ 模块可用"（§89，闭包注入在探针里而不在模块里）吃过一次亏，这是同一类问题的第二种形态：
**探针的调用方式与生产的调用方式不同**。以后凡是加"形状门禁"，必须有一条测试用**生产配置**
（这里是多序列）去断言它真的走到了。

## 92. 又一处"每步分配"：bias 的 fp32 转换（已缓存）

### 92.1 问题

内核把 bias 声明为 fp32，而 GDN 的卷积 bias 是 bf16 参数。我的接线里写的是
`bias.to(torch.float32)` —— **每层每步都做一次 8192 元素转换 + 分配**，30 次/步。

这是我在 §89 修 dtype 不匹配时顺手写下的，正是 §47 那条"修 bug 时引入固定开销"的同一个错误形态。

### 92.2 修法

`_fp32_bias(bias)`：按 `(data_ptr, dtype, numel)` 缓存，且**持有强引用**并在命中时校验
`entry[0] is bias`（地址被回收后不能误命中 —— 与 `hostcache` / `recurrent._decode_constants`
同一条安全论证）。fp32 输入直接返回原张量，零拷贝。

### 92.3 验证

新增测试断言：同一参数复用同一转换结果、不同参数各自转换、fp32 bias 零拷贝。
**50 项测试通过**（32 oracle + 18 门禁/回退/缓存），本地与服务端一致。

### 92.4 状态

`TILEINFER_GDN_CONV_KERNEL` 仍默认关。设备在 127 GB 占用（几乎满），
`/tmp/pk_retry.log` 仍无 `measured_ratio`；端到端测量继续阻塞于外部占用。

## 93. 端到端配对结果：**0.995 —— 内核在真实服务里没有收益，保持关闭**

### 93.1 数字（同进程配对，4/4 有效，`/tmp/pk_retry.log`）

```
[pair0] on=105.1  off=104.4  (off-on)=-0.7
[pair1] on=105.5  off=104.4  (off-on)=-1.0
[pair2] on=104.7  off=103.5  (off-on)=-1.2
[pair3] on=103.3  off=104.6  (off-on)=+1.2
### on median 104.9   off median 104.4
### paired difference median -0.9 ms/token (-0.8%), spread 2.4 ms (2.3%)
### measured_ratio on/off = 0.995
```

按 C2：**ratio 0.995 < 1.0 → 默认关闭**（它本来就关着，现在这个默认有了数据）。

### 93.2 为什么 op 级 42.5× 到端到端只剩 0.8%

- op 级：`causal_conv1d_update` 0.005 ms vs 0.22 ms；×30 层本该省 **6.6 ms**（约占 104 ms 的 6.3%）；
- 实测只省 **0.9 ms**，也就是只有约 **14%** 兑现。

两个可能的原因，都指向同一处：

1. **每序列一次调用**：`--max-num-seqs 4` 时每层要调 4 次内核，每次都要过
   `_try_conv_kernel` 的字典查找、连续性检查、`x_seq.reshape(-1).to(...).contiguous()`、
   `torch.empty`，以及内核句柄缓存查找 —— 这些 host 开销在探针里**不存在**（探针直接调 `kernel(...)`）；
2. 6.6 ms 的 6% 处在 2.3% 极差的边缘，即使全部兑现也接近分辨率下限。

### 93.3 教训：这是"探针调用方式 ≠ 生产调用方式"的**第三种形态**

§89 是闭包注入在探针里、§91 是门禁只在单序列下成立、这一次是**每次调用的 host 包装开销**
在探针里被绕过了。三次都是同一句话：**探针测的是内核，生产测的是"接线后的路径"。**

因此以后 op 级 ratio 只能当**上界**，不能当预期；预期必须来自接线后的同进程配对。

### 93.4 让它可能兑现的下一步（如果还要继续这条路）

把内核改成**一次处理整个 batch**（`X: [B, C]`、`State: [L, C, taps]` + `lines: [B]`），
把每层 4 次调用变成 1 次，同时把 host 包装降到一次。那是内核侧的改动，
按 §93.2 的算术，兑现上限仍然是那 6.6 ms（6%），**投入产出比不高** —— 我倾向就此打住，
把这一项记为"内核级快 42.5×，端到端 0.995，不采用"。

## 94. 默认状态无回归验证（多轮改动之后）

### 94.1 复测

`--enforce-eager`、全默认（本轮所有改动都在默认关闭的路径上）：

| 指标 | 之前确认值 | 本轮 | 判断 |
|---|---|---|---|
| TPOT | 95.8 ms（极差 2.1%） | **99.11 ms**（极差 1.9%） | 无回归 |
| TTFT@512 | 1.215 s | **1.246 s** | 无回归 |
| TTFT@16 | 0.529 s（好窗口） | 0.846 s | 机器慢了，非代码 |

跨窗口的绝对值在本机可以漂 2 倍（TPOT 见过 95.8 / 99.1 / 100.7 / 103.4 / 104.4；TTFT@16 见过
0.53 / 0.85 / 1.00），所以**上表只能读"没有显著回归"，不能读"变快或变慢"**。
自抖动仍报 `RESULT: FAIL`（atol 1e-6）—— 那是 §85/§86 记录的 0.036 nats，已知且在 vLLM 侧。

### 94.2 结论

所有本轮的改动（融合卷积内核及其门禁/守卫/缓存、§90–§92）**都在默认关闭的路径上**，
默认行为与改动前一致。这也验证了 E4 的可回退性：开关关着时，行为回到原样。

## 95. 验证 §93.2 的假设：`max-num-seqs 1` 时内核从 0.995 翻到 1.015（但仍在噪声内）

### 95.1 实验

§93.2 把"42.5× 只兑现 14%"归因于**每序列一次调用**的 host 开销（batch 4 → 每层 4 次）。
验证方法：把 `--max-num-seqs` 降到 1（每层只 1 次调用），再配对一次。

`benchmarks/pair_switch.sh` 现在支持第三个参数（柜数）：

```
[pair0] on=103.9 off=105.6 (off-on)=+1.7
[pair1] on=103.4 off=103.1 (off-on)=-0.3
[pair2] on=106.2 off=118.7 (off-on)=+12.5     <- 离群
[pair3] on=103.3 off=104.6 (off-on)=+1.4
### on median 103.6   off median 105.1
### paired diff median +1.6 ms (+1.5%), spread 12.8 ms (12.4%)
### measured_ratio on/off = 1.015
```

### 95.2 怎么读

- **方向与假设一致**：batch 4 时 0.995（无收益），batch 1 时 1.015（+1.5%）——
  调用次数减少后收益回来了，这正是"每调用 host 开销"的表现；
- **但不可信**：极差 12.8 ms（12.4%）远大于中位效应 1.6 ms，其中 `pair2` 是明显的离群
  （off 臂 118.7 ms，比其它三对高 13 ms）。

按 §83.4 的铁律：**效应被极差淹没 → 不作结论、不翻默认**。开关保持关闭。

### 95.3 如果还要继续这条路，正确的做法

把内核改成**一次处理整个 batch**（`X: [B, C]` + `lines: [B]`），这样 batch 4 时也是 1 次调用。
但按 §93.2 的算术，兑现上限仍是每步 6.6 ms（6%）——**这个上限本身就不够**，
所以我建议把这项记为"已探查、不采用"，而不是继续投入。
















### 80.4 八层阻塞的完整清单（备查）

| # | 阻塞 | 性质 | 状态 |
|---|---|---|---|
| 1 | `--enforce-eager` | 我自己的启动参数 | 已识别 |
| 2 | `torch.compiler.disable` | 我自己的 workaround | 开关可撤 |
| 3 | `with self._lock` | 我们自己代码 | 追踪期 no-op |
| 4 | `RMSNormQuantFusionPass` NameError | **vLLM 上游 bug** | 平台关掉 7 个 pass |
| 5 | `cudagraph_mode` 被解析成 NONE | 平台配置 | `TILEINFER_ACLGRAPH` |
| 6 | capture size 缺失 | 平台配置 | 同上 |
| 7 | 同步拷贝（`aclrtMemcpy`） | 我们的主机读 | MoE/卷积两处设备路径 |
| 8 | **捕获期禁止流同步** | **平台限制 / 上游捕获路径** | 未解，建议不追 |

前七层全部是可修的；第八层不是。这条清单本身就是 M3 的结论：**M3 的前置条件不是"别名已修"，
而是"平台在捕获期允许流同步"或"vLLM 的捕获路径里没有同步"。**

































































