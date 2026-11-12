> **SUPERSEDED（2026-11-12）**：本文描述的是 **v2「原生后端」** 定位（不依赖 vllm-ascend、
> 自带平台层），该定位已被 v3 取代。**不要按本文建目录或改代码。**
>
> 现行文档：[`../../architecture.md`](../../architecture.md) ·
> [`../../integration.md`](../../integration.md) · [`../../status.md`](../../status.md)
>
> 保留原因：历史取证与"哪些路走不通"的记录。

# TileInfer 作为 vLLM 原生昇腾后端：Qwen3.6 单卡方案

> 这是**唯一**的方案文档。范围、分工、里程碑、验收口径都以本文为准。
> 现状与实测数字在 [`status.md`](status.md)；性能口径在 [`performance.md`](performance.md)。
> 耦合点盘点在 [`integration-vllm-native.md`](integration-vllm-native.md)。

---

## 0. 范围

**目标**：`pip install vllm tileinfer` 之后，`vllm serve <Qwen3.6-35B-A3B>` 在 **1 张 950** 上跑起来，
**不安装 `vllm_ascend`**，性能不低于现网（vllm-ascend 0.23 + 图模式）。

**定位**：单模型后端。只保证 Qwen3.6 这条路又快又对；通用性靠"能加模型"的扩展点，不靠兼容层。

**明确不做**（砍掉即减少工作量，不是延后）：

| 砍掉 | 原因 |
|---|---|
| 量化（W8A8 / W4A8 / FP8-KV） | 主线先跑 bf16；量化会让每个算子多一条分支，收益要在打平之后再谈 |
| TP / EP 并行 | 单卡先对齐；多卡是第三块的事，且要先有设备抽象 |
| MLA / 稀疏 / cascade | Qwen3.6 用不到 |
| V2 model runner | vLLM 0.23 里它是另一个抽象层，支持它等于重写一遍 worker hook |
| 通用模型兼容 | 一个模型一条路，第二个模型（DeepSeek-V4.1）是"证明扩展点够用"的样例，不是承诺 |

**三条价值**（任何改动至少推进其一，否则不做）：

1. 单卡性能真的与基线可比 —— 这是唯一能证伪的口号。
2. 用自有算子的同时，有能力在同一名字下混用上游/厂商实现，共同让性能最好 —— 这是架构价值。
3. 同一套结构能换卡、能加模型 —— 这是可复用价值。

---

## 1. 三块分工

块之间**可并行**，块内**串行**。每块有独立验收，不靠"整体感觉变好了"。

| 块 | 负责的问题 | 交付物 | 优先级 |
|---|---|---|---|
| **单卡性能** | 一步为什么是 102 ms，哪一段能压短 | 一份可复现的报告 + 更短的步 | 最高 |
| **算子取舍** | 同一个算子名，谁来实现、按什么规则选、选完怎么记账 | 选择层 + 账本 | 高（与上块并行） |
| **平台** | 换卡/加模型时要改哪些文件，哪些是设备事实 | `DeviceProfile` + 第二个模型跑通 | 低（并行） |

### 1.1 单卡性能块的四个子类

同属"让单卡更快"，但**手段不同、验收不同**，不要混为一谈：

| 子类 | 典型问题 | 验收 |
|---|---|---|
| 仪器 | 这 102 ms 花在哪？报告能复现吗？ | 一次前向的完整报告，异常不被平均值掩盖 |
| 算子 | 这个 kernel 本身还能快吗？ | 算子级实测比值 + 端到端比值**同时**改善 |
| 执行模式 | 能不能少提交、少同步、少空转？ | 图捕获后每步 host 时间显著下降且数值一致 |
| 解码算法 | 能不能一步出多个 token？ | 投机解码接受率与端到端加速比 |

### 1.2 块内顺序（为什么是这个顺序）

1. **仪器收尾**（`scripts/ti_trace.py`）。没有可信的分段计时，后面三项都在猜。
2. **三个已定位的算子** + 捕获安全回归门。它们有离线比值、有现成实现，收益确定。
3. **图捕获（ACLGraph）**。它是"每步固定开销"的唯一解，且必须在算子不再引入 host 读之后做，
   否则捕获会一直踩同步。
4. **投机解码（MTP）**。模型侧现成，瓶颈在我们自己的批语义。

### 1.3 已知事实（写方案前已核实，不要再重新论证）

**MTP 可行性**（2026-11-10 在 102 上核实）：

- checkpoint 里有 `text_config.mtp_num_hidden_layers: 1`、`mtp_use_dedicated_embeddings: false`，
  以及 19 个 `mtp.*` 张量（`mtp.fc.weight`、`mtp.layers.0.mlp.experts.*`、`shared_expert`、`gate`）。
- vLLM 自带 `qwen3_5_mtp.py`，**模型侧不需要训练，也不需要改权重**。
- 缺的是我们的后端：`tileinfer/vllm/attn/metadata.py:33` 明确拒绝投机解码批
  （"one query token only"），`tileinfer/vllm/platform.py:37` 把 speculative 列为范围外。
- 要打通的链路：metadata 批语义（1 + k 个 token）→ 多 token kernel → 拒绝采样 → KV 追加。

**图捕获**：v1（vllm-ascend 托管）在服务里**真的跑起来过**图模式（6 plan READY、
`capture_calls=40`、`fallbacks=0`、logprobs 逐位一致）。v2 现在的失败是**我们自己的捕获路径
没接完**，不是平台限制。曾经把 `EE1016`（"stream is captured" 时同步失败）判成平台死路，
这个判断是错的，已作废。

---

## 2. 结构与硬约束

### 2.1 目录边界（谁改哪里）

| 块 | 可以改 | 不许碰 |
|---|---|---|
| 单卡性能 | `tileinfer/kernels/**`、`tileinfer/serving/trace/**`、`tileinfer/vllm/ops/{fla,moe,gdn}.py`、`tileinfer/vllm/attn/{impl,prefill}.py`、`tileinfer/vllm/graphs.py` | 平台层、worker 主体 |
| 算子取舍 | `tileinfer/vllm/ops/{candidates,router,switch,spec}.py`、`ops/providers/**`、`vendor/**`、`testing/**`、`benchmarks/ab_*` | kernel 实现细节 |
| 平台 | `tileinfer/vllm/platform.py`、`worker.py` 主体、`attn/{backend,metadata,kv_cache}.py`、`compat/**`、`doctor.py`、`device/**`、`profiles/**`、`models/**` | 算子内部 |

**联合项**（一方主导，另一方提供接口）：图捕获（性能块主导，平台块给 worker 挂点）、
MTP（性能块主导 kernel，平台块定 metadata 语义）、设备抽象（平台块主导）。

### 2.2 四条硬约束

1. **不 import `vllm_ascend`。** 需要它的算子就抄进自己的代码（放 `vendor/`，带出处与许可证头）。
   CI 用 AST + `sys.modules` 双重检查（`tests/test_no_vllm_ascend.py`）。
2. **没有 `measured_ratio` 和 bench 脚本的算子不许进主干。** 比值是离线手写数字，不是运行时测的。
3. **运行期不许为新算子做计时决策。** 运行时只允许"免费判断"（形状/dtype/连续性）。
4. **任何"提速"都要同时给出算子级和端到端两个数。** 只报前者会重复已经犯过的错
   （融合卷积算子级 42.5×，端到端 0.995）。

### 2.3 自有算子要多少

不设百分比。要求是：**至少有一个完整阶段的核心路径是我们自己写的，且该阶段占一步 ≥15% 时间**；
不许只实现一两个边角算子然后称之为"自有算子"。

---

## 3. 算子的四种状态与三个选择时刻

**四种状态**：`enabled`（默认开）、`armed`（满足条件才开）、`parked`（关掉但保留重开触发条件）、
`deleted`（删掉，连代码一起）。每个候选必须处在其中之一，不允许"没人知道它开没开"。

**三个时刻**：

| 时刻 | 允许做什么 | 不允许 |
|---|---|---|
| T0 离线 | 测比值、写 bench、决定默认值 | 改运行期行为 |
| T1 启动（`install_ops()` / `compile_or_warm_up_model()`，在 `capture_model()` 之前） | 读 `measured_ratio`、按设备/形状选实现、可选自动调优（≤60 s） | 写回主干配置 |
| T2 每次调用 | 免费判断（形状、dtype、contiguity、是否有 flag） | 计时、查表、任何 I/O |

---

## 4. 里程碑

| # | 里程碑 | 轨 | 完成判据 |
|---|---|---|---|
| M0 | 工具收尾 + 基线重建 | 主 | `ti_trace.py` 端到端跑通；按 §5 矩阵重测双臂，数字落盘 |
| M1 | 三个算子 + 捕获安全回归门 | 主 | 三个候选进默认；`fallbacks=0`；端到端比值改善而不是只在算子级 |
| M2 | ACLGraph | 主 | 图模式下 `capture_calls>0`、`fallbacks=0`、logprobs 与 eager 逐位一致、每步 host 时间下降 |
| M3 | MTP 投机解码 | 主 | 接受率与端到端加速比同时为正；关闭时逐位回退到 M0 数字 |
| M4 | 选择层纯重构（行为不变） | 并行 | 重构前后所有数字在噪声内；账本可查 |
| M5 | 择优 + 厂商算子混用 | 并行 | 至少一个名字下有两个 provider 可切换，且切换有数据依据 |
| M6 | 平台抽象 + 第二块卡/第二个模型 | 并行 | 新卡只加一个 profile 文件；dsv41 走同一套扩展点 |

M4 的"纯重构"是刻意的：先把选择层结构做对，再往里灌择优逻辑，避免两者同时改导致无法归因。

---

## 5. 测试矩阵与口径

**固定**：输出 128 token；`--max-num-seqs` 必须等于客户端并发；`--max-model-len 9216`。

| 组 | 场景 | 用途 |
|---|---|---|
| S1–S4 | 128/1、4096/1、128/16、2048/8 | 短/中上下文、单流/并发 |
| R1–R3 | 8192/1、8192/8、2048/32 | 长上下文与高压并发（风险组，不是门） |
| D1–D2 | 1/1、prefill-only 512/2K/8K | 只测 prefill 的纯 TTFT |

**口径纪律**：

- 只有**同一个时间窗内成对跑出来**的比值可以引用。跨窗口漂移可达 2×。
- 8K 是风险组不是门：v1 在这里就崩过（吞吐 0.213×）。
- 引用任何数字必须同时给：命令、serve 参数、模型路径、是否图模式、测量日期。

---

## 6. 与旧方案的关系

本文替代 `docs/plan-vllm-native-backend.md` 的 v1/v2 历史版本、`docs/roadmap.md` 的阶段划分，
以及 `docs/next-tasks.md` 的任务清单。旧文档里凡与本文冲突的，以本文为准。
`docs/notes/dsv41-*.md` 只剩模型事实与结论，过程日志已删除。
