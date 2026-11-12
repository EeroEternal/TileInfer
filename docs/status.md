# 状态

> 一页纸回答两个问题：**建成了什么（凭什么相信）**、**哪里还断着**。
> 目标形态见 [`architecture.md`](architecture.md)，验收口径见 [`acceptance.md`](acceptance.md)，
> 数字见 [`performance.md`](performance.md)。
>
> 最后核对：**2026-11-12 20:10**，代码 `6183c95`（长上下文 L4/L5；v3 主体 `6f41866`），
> 单卡 Ascend950PR，CANN 9.3.0。

---

## 0. 一句话

**v3 形态已经跑通：vllm-ascend 仍是平台/worker/KV/图捕获的主人，TileInfer 只做
"声明 → 决策 → 选实现 → 补缺口"。** Qwen3.6-35B-A3B 端到端四条验收全过：
能跑、输出与基线逐位一致、图捕获 8 张且稳态回退 0、TPOT/首字/吞吐在 1.10×/0.95× 内。

v2 的四项旧账（图捕获没接、TPOT 2.53×、TTFT 11.8×、MTP 未做）仍然成立；它们描述的是
v2 自带平台层的形态，v3 不再走那条路。本页 §1 的 v3 数字与 §1.1 的 v2 资产不要混着比。

| 验收 | 结论 | 凭什么相信 |
|---|---|---|
| **C1 能跑** | ✅ | `:8322` 服务 70 s 就绪，`/health=200`；平台插件只有 `ascend`，TileInfer 只在 general_plugins |
| **C2 跑对** | ✅ | 与 `/tmp/baseline_fia.json` 去掉 `id/created/system_fingerprint` 后 JSON 全等；首 token `Here -0.0302655640989542` |
| **C3 能上图** | ✅ | `captured_graphs=8`（PIECEWISE 4 + FULL 4），`decode_calls=15（门禁轮）/ 2375（压测轮，旧口径）`, `graph_hits` 同步，`fallbacks=0`；`doctor` 绿 |
| **C4 跑得快** | ✅ | 单流 TTFT 0.907–1.016×、TPOT 0.999–1.001×；4K（3939 tok）1.000×/0.999×；conc16 吞吐 1.003×、TTFT 0.997× |

证据页：[`notes/m0-baseline.md`](notes/m0-baseline.md)（v2 冻基线）、
[`notes/m1-v3-bridge.md`](notes/m1-v3-bridge.md)（v3 设计、接入点、A/B、复现命令）。

---

## 1. v3 M1 已建成（2026-11-12）

| 能力 | 状态 | 凭什么相信 |
|---|---|---|
| 声明层（`manifests/qwen3_6.py`） | ✅ | 40 层/30 GDN+10 full、16Q/2KV、256 experts top-8、8 项 required capability，纯数据可校验 |
| 契约层（`contracts/`） | ✅ | 8 个 `Protocol` + plan/manifest/gap schema；未知 key、重复选择、超预算都拒绝 |
| 决策层（`resolve/`） | ✅ | 纯函数；vendor 在→8/8 选 vendor、gap=0；vendor 不在→8 条 required gap、servable=False（这是"会不会服务不了"的判定，不是"要去补 8 个实现"的待办） |
| 接入层（`bridge/`） | ✅ | 6 个具体钩子（plugin entry、dispatch、capture、attn select、config gate、init keys），4/6 类机制，预算 8 |
| 分层禁令 | ✅ | AST 守卫：只有 `tileinfer/bridge/` 可 import `vllm_ascend`；另有测试防止 carve-out 变空 |
| 观测 | ✅ | capture/fallback 计数落 JSON（`SIGUSR1`/500 步/atexit）；API server 的零计数不再覆盖 EngineCore；算子级审计（`TILEINFER_AUDIT`）2026-11-12 跑通：23 类/903 实例，热路径全 vendor，3 个非 vendor 行均有解释，见 `notes/m1-op-audit.md` |
| CLI / doctor | ✅ | `plan build/show/verify`、`tileinfer doctor`；doctor 在最终 metrics 上全 PASS |

### 1.1 v2 资产（冻结，v3 继续用）

| 能力 | 状态 | 凭什么相信 |
|---|---|---|
| 平台层（平台插件、注意力枚举接管）**v2 形态，v3 不用** | ⛔ | 冻结；v3 由 vllm-ascend 拥有平台槽 |
| worker 四个 hook（设备初始化 / 可用显存 / 配置初始化 / 编译预热） | ✅ 冻结 | 有契约测试锁住形状；v3 不安装 |
| attention 后端四件套（后端 / 元数据 / 实现 / KV 写入） | ✅ | 设备上出 token，与参考实现对得上；作为 M2+ provider 候选 |
| 混合 KV cache（30 线性 + 10 full，单池两种页） | ✅ | 能力探测为真；v2 设备实测不报池一致性错误 |
| 线性注意力（GDN）算子族：卷积 / chunk / kkt / wu / solve_tril / delta_h / gating / recurrent / post_conv | ✅ | 单测 + 设备对数；离线比值见 `notes/perf-candidates.md` |
| MoE 算子族（排序路由 / bmm / 严格缓存） | ✅ | 三个候选已进默认，各有离线比值与 bench 脚本 |
| 分相计时（主机/设备双时钟） | ✅ | 报告可复现 |
| 一键剖析工具 | ✅ | 2026-11-10：57 s 就绪、首字 814.7 ms、单步 106.62 ms。2026-11-12 复跑首请求 4963.9 ms，疑似首请求 JIT；见 M0 §3 |
| 分层守卫（原"绝对不依赖 vllm-ascend"） | ✅ 已反转 | 改为"仅 bridge 可 import"；`test_no_vllm_ascend.py` 2 passed |

---

## 2. 没建成 / 已知缺陷

> 完整的模型验证矩阵（长上下文/图/投机/并发/冷启动）见
> [`verification-matrix.md`](verification-matrix.md)；本表只列当前断点。

| 项 | 现状 | 影响 |
|---|---|---|
| **补缺口还是空的** | `providers/` 无 provider——**按规则它就该是空的**：Qwen3.6 的 8 项能力厂商全覆盖，没有缺口可补。gap=0 是"没有缺口"，不是"补完了" | 要验证补缺口，得先选一个**厂商部署不了的模型**；给厂商已覆盖的能力写实现属于重复劳动（[`extending.md`](extending.md) §3.0） |
| **图捕获** | v3 已接上（厂商捕获 + 我们计数）；4K 8 张、32K 6 张、MTP 6 张（PIECEWISE，FULL=0），均 0 回退；MTP+32K 6 图/0 回退；`impl_table`/`model_table` 两个机制仍是空实现 | 非 vendor 实现还没有落点；128K 组合未验 |
| **投机解码 / MTP** | S2/S3 按 B 质量等价过：套件 8/8、needle 8K/16K/32K 9/9、同请求两次 `same`、4K 93.9% / 1.35×；图 6 张、0 回退。严格逐位不成立（256 token 中 85.9% 相同、第 58 token 分叉），明确非逐位。S4 32K 部分：9/9、TTFT 1.037×、6 图/0 回退，acceptance 81–88%、128K 未测。证据见 [`notes/m1-spec-quality.md`](notes/m1-spec-quality.md) | S2/S3 过；S4 仅 32K 部分；非逐位 |
| **TPOT / 首字** | v3 M1：单流 0.907–1.016× / 0.999–1.001×；4K 1.000× / 0.999×；长上下文 8K/16K/32K：TTFT 1.008–1.011×、TPOT 0.999–1.002× | 门已过（本机、本参数） |
| **长上下文** | **L1–L4 已过**（8K/16K/32K/128K；128K 单流，3/3 needle 逐 token 同，峰值 HBM 107,374 MB）；**L5 262K 不过**：d90 两臂 256 token、v3 500 token 均未答出、C4 TPOT 1.1019×/吞吐 0.9076×。当前 33K/4 用于投机/并发，128K 需 `max-num-seqs=1`。证据见 [`notes/m1-long-context.md`](notes/m1-long-context.md) | 实际质量上限 128K；262K 不声明支持 |
| **量化** | 只做 bf16 | 未验证组合一律不声明支持 |
| **TP / EP / 多卡** | 明确延后 | 优先级最低 |
| **冷缓存 / 首次启动** | 两臂共用编译缓存；冷启动 TTFT 未测 | M2 预编译门要单独回答 |
| **128K / 262K** | 128K 已过：TTFT 0.9433×、TPOT 0.9884×、吞吐 1.0117×、峰值 107,374 MB。262K：d10/d50 可答，d90 两臂 256 token、v3 500 token 均未答出；C4 TPOT 1.1019×、吞吐 0.9076×；峰值 107,376 MB、2 图/0 回退。40,960/8 的 OOM 是 8 seq 放大，不代表 128K/1 不可行 | 262K 能起但不过质量/C4；不要当支持 |

---

## 3. 定位演进与已作废结论

### 3.1 三次定位

| | 依赖 vllm-ascend | 结果 | 现在怎么处理 |
|---|---|---|---|
| **v1 寄生** | 是 | 短上下文与基线打平（0.91–1.09），补丁点散落十几处 | 代码冻结在 `legacy/`，作为 v3 接入点的素材 |
| **v2 原生** | 否 | 能跑能对，但 TPOT 2.53×、TTFT 11.8×，图捕获未接 | 形态作废；内核、契约、计时、剖析工具保留 |
| **v3 兼容层** | 是 | M1 跑通：C1–C4 全过，6 个钩子/预算 8 | 目标形态，见 [`architecture.md`](architecture.md)、[`notes/m1-v3-bridge.md`](notes/m1-v3-bridge.md) |

### 3.2 已作废的结论（保留在案，防止重犯）

| 结论 | 状态 | 依据 |
|---|---|---|
| "图捕获在 950 上不可达" | **已作废** | v1 在服务里跑过图模式（捕获次数 > 0、回退 0） |
| "只要重写平台层就能拿回性能" | **已作废** | v2 的代价就是这条假设的反证：TPOT 2.53× |
| "AI 侧能跑起来就算支持" | **已作废** | 四条独立结论（见 acceptance.md），能跑只是第一条 |

**留这条的目的**：这三个念头每一个都曾经看起来很有道理，并且各自花掉过以周计的时间。

---

## 4. 证据分级（引用数字前先看这个）

| 级别 | 含义 |
|---|---|
| **[已核实]** | 有落盘日志、命令可复现，且知道当时的口径 |
| **[待复核]** | 只在某一个窗口测过。跨窗口漂移可达 2×，不能当结论 |
| **[已作废]** | 曾写进文档但被后续实验推翻。**保留在案，防止重犯** |

---

## 5. 上机第一件事

```bash
cd /home/userlist/zxs/TileInfer
source scripts/env-ascend102.sh
npu-smi info | head -20            # 先确认卡是空的（本项目不抢别人的卡）
python3 scripts/ti_trace.py config
python3 scripts/ti_trace.py run --preset smoke
```

剖析工具自己会查显存门（占用 > 20 GB 就拒绝）并加设备锁，不会和别的进程抢卡。
报告落在 `trace_out/<时间戳>-smoke/`，先看 `summary.md` 顶部的异常汇总，再看 `stages.csv`。

---

## 6. 环境

| 项 | 值 |
|---|---|
| 机器 | `tileinfer-102`（10.119.5.102），`/home/userlist/zxs/TileInfer` |
| 卡 | 1 × Ascend950PR，131072 MB HBM |
| CANN | 9.3.0，在 `.work/cann930/install/cann` |
| 版本策略 | 冻结 vLLM 0.23.0 / vllm-ascend 0.23.0-**950**；M5 前做 V0 gate，见 [`notes/version-strategy.md`](notes/version-strategy.md) |
| Python（本题后端） | `.venv_native213`：torch 2.13.0+cpu、vLLM 0.23.0、tilelang 0.1.15 |
| Python（基线 A/B） | `.venv_baseline`：vllm-ascend 0.23，不含 tileinfer |
| TileLang 缓存 | `$HOME/tl_cache_213`（**不要放进仓库**） |
| 统一入口 | `source scripts/env-ascend102.sh` |

`scripts/env-ascend102.sh` 是唯一的环境定义。它曾经只活在未跟踪的临时目录里，
删掉那个目录就等于删掉项目唯一的工具链定义——这是已经修掉的坑。

---

## 7. 测试现状（2026-11-10 实测，尚未修）

**CPU 套件在 102 上不是绿的**：`pytest tests -m 'not npu and not tilelang'` 有 80 项失败。
三条已核实的事实，避免误判：

1. **每个文件单独跑是绿的。** 例：同一个文件单独跑 30/30 passed，在整套里会失败。
   **是跨文件顺序污染，不是算子坏了。**
2. **与仓库清理无关。** 清理前 89 项失败，清理后 80 项，0 项新增、9 项被修好。
3. **诱因是"装了真 vLLM"这个事实本身。** 契约测试用假的 vLLM 钉住 API 形状，
   该 fake 只在没有真 vLLM 时注入自己，于是同一份代码在不同机器、不同文件顺序下含义不同。
   把 vLLM 藏起来再跑反而更差（100 项失败），所以"换个环境跑"不是解。

**结论**：这是独立的一轮排查任务。在它被修掉之前：判断某次改动有没有破坏东西，
用逐文件跑，不要用整套的红绿。设备侧测试是另一套入口。

---

## 8. 仓库纪律

- 没有证据的算子不许进主干：有基线给同一窗口的成对比值，无基线给绝对数 + 与参考实现的
  一致性（口径见 [`acceptance.md`](acceptance.md) §2.1），并且都要有 bench 脚本。
- **厂商已能正确部署的能力不写提供者**；写了就是重复劳动，按红线删除（[`extending.md`](extending.md) §3.0）。
- 环境产物、编译缓存、profiling dump 一律不进仓库。
- `reference/vllm-0.23/` 下的文件不许改——它们是行号锚点。
- `legacy/` 只减不增。
- 环境定义必须随仓库走，不许只活在临时目录里。
- **文档里提到的目录必须真实存在**；不存在就写"计划"，不要写进目录表。
