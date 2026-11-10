# 状态

> 一页纸回答两个问题：**建成了什么（凭什么相信）**、**哪里还断着**。
> 方案见 [`plan-vllm-native-backend.md`](plan-vllm-native-backend.md)，数字见 [`performance.md`](performance.md)。
>
> 最后核对：**2026-11-10**，代码 `4cb86c9`，单卡 Ascend950PR，CANN 9.3.0。

---

## 0. 一句话

后端**能起服务、能出正确结果、混合 KV 能跑**；**性能没打平**（TPOT 2.53×、TTFT 11.8×），
**图捕获还没接上**，**投机解码没开始**。这三件事就是当前全部工作。

---

## 1. 已经建成

| 能力 | 状态 | 凭什么相信 |
|---|---|---|
| 平台层（`vllm.platform_plugins` 注册、`AttentionBackendEnum.CUSTOM` 接管） | ✅ | 服务能起，`AttentionBackendEnum.CUSTOM.get_class()` 返回我们的类，无静默回退 |
| worker 四个 hook（`init_device` / `determine_available_memory` / `initialize_from_config` / `compile_or_warm_up_model`） | ✅ | `tileinfer/vllm/worker.py`，`tests/test_worker_contract.py` 锁契约 |
| attention 后端四件套（`backend` / `metadata` / `impl` / `kv_cache`） | ✅ | 设备上出 token，与参考实现对数一致 |
| 混合 KV cache（30 层 GDN 线性 + 10 层 full attn，单池两种页） | ✅ | `support_hybrid_kv_cache()` 为真；设备实测不报池一致性错误 |
| GDN 算子族（causal_conv1d / chunk / kkt / wu / solve_tril / delta_h / gating / recurrent / post_conv） | ✅ | `tests/test_gdn_*.py` + 设备对数；离线比值见 `notes/perf-candidates.md` |
| MoE 算子族（sorted routing / bmm+bf16 / strict cache） | ✅ | 三个候选已进默认，各有离线比值与 bench 脚本 |
| 分相计时（host/device 双时钟） | ✅ | `tileinfer/serving/phases.py`，报告可复现 |
| 一键剖析工具 | 🟡 | `scripts/ti_trace.py` 已能在 102 上列预设、出计划、写报告目录；端到端真跑待验 |
| 不依赖 `vllm_ascend` 的守卫 | ✅ | `tests/test_no_vllm_ascend.py`（AST + `sys.modules` 双重检查） |

---

## 2. 没建成 / 已知缺陷

| 项 | 现状 | 影响 |
|---|---|---|
| **图捕获（ACLGraph）** | 未接上。`--enforce-eager` 是当前唯一能跑稳的形态 | 每步固定开销吃掉大部分 TPOT 差距。这是 P0 |
| **投机解码（MTP）** | 未开始。模型侧现成（checkpoint 有 MTP 头、vLLM 有 `qwen3_5_mtp.py`），卡在我们的批语义只允许一个 query token（`attn/metadata.py:33`） | 单 token 一步的天花板 |
| **TPOT / TTFT** | 2.53× / 11.8× 于基线，门是 1.10× | 交付门槛未过 |
| **长上下文** | v1 在这里崩过（8K 吞吐 0.213×）；v2 的长上下文数字还没系统测 | 风险组，不是当前门 |
| **量化（W8A8/W4A8/FP8-KV）** | 明确不做，bf16 一条路 | 见方案 §0 |
| **TP / EP / 多卡** | 明确延后，优先级最低 | 见方案 §1 |
| **算子选择层** | 有候选机制（`Candidate`/`CandidateSet`）但没有 provider 抽象与账本 | 第二块的主要工作 |

---

## 3. 证据分级（引用数字前先看这个）

| 级别 | 含义 |
|---|---|
| **[已核实]** | 有落盘日志、命令可复现，且知道当时的口径 |
| **[待复核]** | 只在某一个窗口测过。跨窗口漂移可达 2×，不能当结论 |
| **[已作废]** | 曾经写进文档但被后续实验推翻。**保留在案，防止重犯** |

已作废的结论清单在 [`performance.md`](performance.md) §4 与
[`notes/handover-2026-10-28.md`](notes/handover-2026-10-28.md) 的文首提示里。
最典型的一条：**"图捕获在 950 上不可达"是错的**，v1 在服务里跑过图模式。

---

## 4. 上机第一件事

```bash
cd /home/userlist/zxs/TileInfer
source scripts/env-ascend102.sh
npu-smi info | head -20            # 先确认卡是空的（本项目不抢别人的卡）
python3 scripts/ti_trace.py config
python3 scripts/ti_trace.py run --preset smoke
```

`ti_trace.py` 自己会查显存门（>20 GB 占用就拒绝）并加设备锁，不会和别的进程抢卡。
报告落在 `trace_out/<时间戳>-smoke/`，先看 `summary.md` 顶部的异常汇总，再看 `stages.csv`。

---

## 5. 环境

| 项 | 值 |
|---|---|
| 机器 | `tileinfer-102`（10.119.5.102），`/home/userlist/zxs/TileInfer` |
| 卡 | 1 × Ascend950PR，131072 MB HBM |
| CANN | 9.3.0，在 `.work/cann930/install/cann` |
| Python（本题后端） | `.venv_native213`：torch 2.13.0+cpu、vLLM 0.23.0、tilelang 0.1.15 |
| Python（基线 A/B） | `.venv_baseline`：vllm-ascend 0.23，不含 tileinfer |
| TileLang 缓存 | `$HOME/tl_cache_213`（**不要放进仓库**） |
| 统一入口 | `source scripts/env-ascend102.sh` |

`scripts/env-ascend102.sh` 是唯一的环境定义。它原来只存在于 `.agent_loop/run/env.sh`
（未跟踪的临时目录），删掉那个目录等于删掉项目唯一的工具链定义——这是已经修掉的一个坑。

---

## 6. 测试现状（2026-11-10 实测，尚未修）

**CPU 套件在 102 上不是绿的**：`pytest tests -m 'not npu and not tilelang'` 有 80 项失败。
三条已核实的事实，避免误判：

1. **每个文件单独跑是绿的。** 例：`pytest tests/test_kv_cache_write.py` → 30/30 passed，
   但同一文件在整套里会失败。**是跨文件顺序污染，不是算子坏了。**
2. **与仓库清理无关。** 在 `HEAD`（清理前）的纯净工作树上跑同一命令是 89 项失败；
   清理后 80 项，**0 项新增、9 项被修好**（那 9 项原先失败是因为 `reference/vllm-0.23/`
   的转录文件没入库，测试读不到源码）。
3. **诱因是"装了真 vLLM"这个事实本身。** 契约测试用 `tests/fake_vllm.py` 钉住 vLLM 0.23 的
   API **形状**；该 fake 只在没有真 vLLM 时注入自己。于是同一份代码在不同机器、不同文件顺序
   下含义不同。把 vLLM 藏起来再跑反而更差（100 项失败），所以"换个环境跑"不是解。

**结论**：这是**独立的一轮排查任务**，不属于"清理工作区"。在它被修掉之前：

- 判断某个改动有没有破坏东西，用 `pytest tests/<file>.py`（逐文件），不要用整套的红绿。
- 设备侧（`-m npu`）跑的是 `scripts/run_device_ci.sh`，与上面这套是两回事。

---

## 7. 仓库纪律

- 没有 `measured_ratio` 和 bench 脚本的算子不许进主干。
- 环境产物、编译缓存、profiling dump 一律不进仓库（`.gitignore` 已覆盖）。
- `reference/vllm-0.23/` 下的文件**不许改**——它们是行号锚点。
- `legacy/` 只减不增。
- 环境定义必须随仓库走（`scripts/env-ascend102.sh`），不许只活在临时目录里。
