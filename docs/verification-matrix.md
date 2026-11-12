# 模型验证矩阵

> 一个模型"在役"，不是"服务能起"，而是下面每一格都有可复现的证据。
> 本文件是 [`acceptance.md`](acceptance.md) §2.2 的展开，也是
> [`long-term-task.md`](long-term-task.md) §4 第 10 条的执行表。
>
> 当前对象：**Qwen3.6-35B-A3B**（基线模型）。M5 起每个新模型——尤其是厂商部署不了的——
> 沿用同一张表，另加缺口报告与 provider 证据。

---

## 0. 规则

1. **空格不是通过。** 必填格必须有证据；没测过的能力写"未验证"，不能写"支持"。
2. **"厂商已支持"不免测。** 厂商支持只说明能力存在；要的是在本机、本参数、本组合下跑过。
3. **组合格单独跑。** 长上下文 × 图、投机 × 图、投机 × 长上下文，不能由单格推断。
4. **证据五件套**：命令、时间、参数、原始日志/JSON、对照（基线或参考实现）。
5. **基线分支**按 [`acceptance.md`](acceptance.md) §2.1：有基线用比值，无基线用绝对口径 +
   参考实现一致性；禁止跨场景拼比值。
6. **回退非 0 是缺陷**，不是"正常降级"；回退要计数、进日志、能被断言。
7. **未启用的能力要显式声明**（量化 / TP / EP / 视觉），不能留空让人误读成"支持"。
8. **部署受审批约束**：见 §6；任何起服务、重启、绑端口、占卡，先征得使用者同意。

状态列是 2026-11-12 的快照；每次验证后同步 [`status.md`](status.md)，两处冲突以 status 为准。

---

## 1. Qwen3.6 验证矩阵

### 1.1 基础端到端

| # | 场景 | 证据 | 门 | Qwen3.6 |
|---|---|---|---|---|
| E1 | 4K 内单流，温度 0 | 完整输出 + 与基线逐位对比（logprobs 四位） | C1 + C2 | ✅ 2026-11-12 |
| E2 | `doctor` 四道门 | plan / layer_guard / bridge / graph_metrics | 全 PASS | ✅ 4K 下 |
| E3 | 同输入复跑一致 | 两次请求的 token 与 logprobs | 完全一致 | ⬜（每档长上下文/投机都复跑一次） |

### 1.2 长上下文

| # | 场景 | 证据 | 门 | Qwen3.6 |
|---|---|---|---|---|
| L1 | 8K prompt | TTFT/TPOT、HBM 峰值、输出与基线一致 | C1–C4 | ✅ 2026-11-12：TTFT 1.0080×、TPOT 0.9988×；9/9 needle 逐 token 同；[证据](notes/m1-long-context.md) |
| L2 | 16K | 同上 | 同上 | ✅ 2026-11-12：TTFT 1.0108×、TPOT 1.0018×；9/9 needle 同 |
| L3 | 32K | 同上 + 图回退 | 同上 | ✅ 2026-11-12：TTFT 1.0077×、TPOT 1.0019×；6 图/0 回退；32K 峰值 HBM 107,380 MB |
| L4 | 128K | 同上 + 峰值激活 | 同上，可单独放宽并发 | ⬜ 当前 cap 33,792；40,960 已 OOM，需单独内存方案 |
| L5 | 模型上限 262,144 | 同上；记录**实际可用上限** | 同上 | ⬜ 未测；KV 容量 1,383,782 tokens 不等于可服务 |

长上下文必须记录：`max-model-len`、prompt tokens（以 usage 为准）、KV 容量、峰值 HBM、
是否触发 chunked prefill、TTFT 随长度的斜率。模型上限 262,144 不代表单卡一定能服务到；
测出来的可用上限写进声明，不写"理论上支持"。

参考：33K 配置下实测 KV 32.39 GiB / 1,383,782 tokens（约 24.5 KiB/token），
单请求 262K 的 KV 约 6.1 GiB，容量不是最明显的墙；**真正要先量的是 262K prefill
的峰值激活与非 torch 内存**。40,960/8 的首次尝试已在 8K 请求 OOM，说明 cap 不能直接外推。

prompt 用 `benchmarks/make_long_prompt.py --tokens N --out /tmp/prompt-Nk.txt` 生成，
探针用 `TI_PROMPT=/tmp/prompt-Nk.txt benchmarks/v3_long_ttft.py <port> <label> <out> [runs]`；
正确性用 `benchmarks/needle_probe.py`（10% / 50% / 90% 三个深度埋码），操作步骤见
[`notes/long-context-runbook.md`](notes/long-context-runbook.md)。
两者都以 `usage.prompt_tokens` 为准，不拿字符数冒充 token 数。

### 1.3 图模式

| # | 场景 | 证据 | 门 | Qwen3.6 |
|---|---|---|---|---|
| G1 | 4K 单流 + conc16 | `captured_graphs`、`capture_keys`、`decode_hits`、`fallbacks` | 捕获 > 0，稳态回退 = 0 | ✅ 8 张（4 FULL + 4 PIECEWISE），0 回退 |
| G2 | 32K 长上下文 | 同上 | 同上 | ✅ 6 张（3 FULL + 3 PIECEWISE），0 回退（2026-11-12，33K cap） |
| G3 | 投机开启 | 同上 + capture sizes 含 `num_spec_tokens + 1` 的倍数 | 同上 | ⬜ |
| G4 | batch 1/4/16/32/64 | 每个尺寸都命中捕获图 | 回退 = 0 | 部分：1–32 已捕获，64 未验 |
| G5 | 投机 + 长上下文 | 同上 | 同上 | ⬜ |

### 1.4 投机解码

| # | 场景 | 证据 | 门 | Qwen3.6 |
|---|---|---|---|---|
| S1 | ngram 冒烟（无 draft 权重） | 服务起、输出、投机 token 统计 | C1；不要求加速 | ⚠️ 4K/8K 服务与输出通过；token 统计未采集（2026-11-12） |
| S2 | MTP k=1，4K | 输出 vs 非投机基线；接受率；TPOT/吞吐 | Q1 任务正确 + Q3 性能（非逐位） | ⚠️ 可跑、接受率 93.9%、约 1.35×；质量门 Q1/Q2 待跑；严格逐位已按 B 改为质量等价；[证据](notes/m1-mtp-smoke.md) |
| S3 | MTP k=1 + 图 | `captured_graphs`、`fallbacks` | 回退 = 0 | ⚠️ 6 图（PIECEWISE 3 / FULL 0）、0 回退；Q5 组合未跑 |
| S4 | MTP k=1 + 长上下文 32K/128K | 同上 + HBM | Q1/Q3 + 回退 = 0 | ⬜ |
| S5 | 投机验收口径写进声明 | `manifests/` 里的字段 | 见下 | ✅ 2026-11-12：S-Q Q1–Q5 写入 `acceptance.md` §2.3，明确非逐位 |

**投机口径必须先定**（当前未定，属于阻塞项）：

- **正确性**：温度 0 下是否要求与非投机基线逐位一致？若个别 token 因浮点差异不同，
  判失败还是给容差？建议先按"逐位一致"跑，不一致就查，不调容差。
- **性能**：以 TPOT 还是吞吐为主？建议 decode 主导场景看 TPOT，服务吞吐看 tokens/s；
  目标值写进声明（例如 TPOT ≤ 非投机 × 0.85），不写"开了就行"。
- **接受率**：记录 `accepted_tokens / draft_tokens`，低接受率要能解释。

厂商侧已具备的条件（2026-11-12 代码核对）：模型 `mtp_num_hidden_layers=1`、权重里
19 个 `mtp.*` 张量；vLLM 0.23 有 `qwen3_5_mtp.py`；vllm-ascend 把 `qwen3_5_moe` 映射到
`qwen3_5_mtp`，并注册了 GDN / KV 绑定 / 拒绝采样补丁。**这是"能力在"，不是"已验证"。**

2026-11-12 实测补充（[`notes/m1-mtp-smoke.md`](notes/m1-mtp-smoke.md)）：

- MTP 能加载（`Qwen3_5MoeMTP`，权重 +2.52 GiB），接受率 93.9%、mean acceptance 1.94，
  非流式 128 token 约 **1.35×**；图 6 张、0 回退，但 FULL=0。
- **严格逐位未过**：同 cap 下 MTP 与不启用投机在 8K 第 58 token 分叉（max logprob 0.76）；
  ngram 与不启用投机在第 10 token 分叉；MTP 与 ngram 也互不相同；MTP 自身两次一致。
- 图/async/cascade 已排除：非投机 FULL 与 PIECEWISE 输出完全相同；MTP eager 与
  PIECEWISE 输出完全相同；但 MTP 与 ngram 仍与非投机在 8K/4K 分叉。MTP 从第 2 个
  token 起 logprob 漂移，ngram 到第 10 个 token 才分叉，指向 GDN spec 多 token 前向；
  不是启动参数能修的，继续需插桩/厂商修复。
- 2026-11-12 决议 **B（质量等价）**：投机不再要求逐位，按 `acceptance.md` §2.3 的
  S-Q 门。S2 的 Q1/Q2 待跑；Q3 已过（93.9%、1.35×）。严格逐位证据保留为诊断，
  不再作为 S2 的门。

### 1.5 并发

| # | 场景 | 证据 | 门 | Qwen3.6 |
|---|---|---|---|---|
| P1 | conc 1 / 4 / 16 | 吞吐、TTFT、TPOT 与基线比值 | C4 | ✅ |
| P2 | conc 32（当前 `max-num-seqs` 上限） | 同上 + 回退计数 | C4 | ⬜ |
| P3 | conc 64 | 同上；需先抬 `max-num-seqs`，重新捕获图 | C4 + 回退 = 0 | ⬜ |

### 1.6 冷启动

| # | 场景 | 证据 | 门 | Qwen3.6 |
|---|---|---|---|---|
| K1 | 清空 `torch.compile` / AOT 缓存后启动 | 到 `/health` 时间、首请求 TTFT、编译日志 | 启动时间可测量；请求期零编译 | ⬜ |
| K2 | 同一配置热缓存复跑 | 同上 | 两臂条件一致才可比值 | ⬜ |
| K3 | TileLang 缓存清空（涉及我方 kernel 时） | 每核编译时间 | 构建期预编译，运行期只命中缓存 | N/A（当前全厂商 kernel） |

冷缓存指**编译缓存为空**，不是 KV cache 冷、也不是权重没加载。清缓存用"移走"不用"删除"，
两臂要同冷/同热，否则 A/B 作废。当前 v3/Qwen3.6 的冷缓存成本主要是
`torch.compile`（2026-11-12 启动日志：12.21 s，且有一次 AOT 加载失败后重编），
不是 TileLang；TileLang 的 70–100 s/核 要等 M5 引入我方 kernel 才成为主要项。

### 1.7 观测与证据

| # | 项 | 证据 | 门 | Qwen3.6 |
|---|---|---|---|---|
| O1 | 算子级审计 | `TILEINFER_AUDIT` JSON：类、impl、实例数、调用数 | 热路径无非厂商实现，或有解释 | ✅ 23 类 / 903 实例（类级） |
| O2 | 设备侧 profiler | `torch_npu.profiler` / msprof 的 kernel 列表 | 覆盖 decode 稳态 | ⬜ |
| O3 | 回退计数 | `fallbacks`、`errors` | 稳态 = 0 | ✅ 4K；长上下文/投机 ⬜ |

### 1.8 范围声明（不许默认"支持"）

| 项 | Qwen3.6 现状 | 规则 |
|---|---|---|
| 量化 | 只跑 bf16 | 未验证的组合不声明支持 |
| TP / EP / 多卡 | 未做，单卡 | 明确延后；不要用"理论上" |
| 视觉输入 | 模型是多模态架构，只跑文本 | 若目标含图，补 image 请求；否则写"不支持" |
| 前缀缓存 / 其他调度特性 | 未单独验 | 不作为"支持"的依据 |

---

## 2. 新模型附加项（M5+）

0. 先过**版本 gate V0**：确认这台 950 能否拿到更新的 vllm-ascend build；拿不到就冻结
   当前栈，并在 gap report 里写明"pinned-stack gap"。见
   [`notes/version-strategy.md`](notes/version-strategy.md)。
1. 先用 `tileinfer plan build` 得到缺口报告：`servable=False` 与 required gap 列表。
2. 每个 gap 对应一个 provider，声明五项：coverage / graphable / parallel / 证据 / `delete_when`。
3. 跑完 §1 全表；无厂商基线时按 [`acceptance.md`](acceptance.md) §2.1 的绝对口径。
4. 记录"从声明到在役"的人日与缺口条数（边际成本，见 acceptance §4）。
5. 接入点总数仍 ≤ 8；超了写例外说明并重新设计。
6. 矩阵不是可选项：**新模型也必须验证 MTP / 图模式 / 长上下文 / 并发 / 冷启动**，
   不能只验"能跑"。

---

## 3. 判定

- 所有必填格有证据且过门 → 可以写"在役"。
- 任一稳态回退 ≠ 0 → 缺陷，停止加模型，先修根因。
- 未测 / 未启用 → "未验证 / 未声明"，不得写成"支持"。
- 组合格未跑 → 不能由单格推断"组合没问题"。

---

## 4. 证据落盘

- 状态页：`docs/status.md`（唯一权威）。
- 单次验证：`docs/notes/` 下的笔记，带命令、时间戳、参数、原始 JSON。
- 长上下文/投机的原始输出放 `/tmp` 可以，但结论与命令必须进仓库；`/tmp` 会被清。
- 指标 JSON 命名建议：`<date>-<model>-<scenario>-metrics.json`。

---

## 5. 当前游标

Qwen3.6 按以下顺序推进：

1. ✅ 长上下文阶梯（L1 8K → L2 16K → L3 32K，2026-11-12）；L4/L5 待 128K/262K 内存方案。
2. ⚠️ 投机质量门：S-Q 已定；Q1/Q2 待跑，Q3 已过；S3 图机制过、Q5 待跑。
3. 组合（S3 投机 × 图、S4 投机 × 长上下文）。
4. 并发 32 / 64（P2/P3）、冷启动（K1/K2）。
5. 设备侧 profiler（O2）。

---

## 6. 部署审批（共享环境）

环境上有其他同事在用，**部署不是本任务可以自动做的动作**。

- 任何 `vllm serve`、重启、端口绑定、占卡之前，先向使用者说明：命令、端口、预计时长、
  预计 HBM 占用、要验证的场景。
- 先 `npu-smi info` 确认卡空闲；卡被占用就等，不抢。
- 得到明确同意后再执行；未获同意不得部署/重启。
- 不 kill 不属于本任务的进程；已有服务是上一轮遗留时，也要先问再处置。
- 验证结束按约定关闭自己起的服务，释放端口与卡。
