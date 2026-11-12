# M0 迁移清单：把 v2 代码搬进 v3 分层

> 目标：**只搬不改**。行为一位不变，逻辑一行不动，删除动作全部留到 M1。
> 这份文档是"新会话可以直接照着做"的程度：每一步有输入、输出、验收门。

---

## 0. 三条 M0 纪律

1. **只 move，不 edit。** 允许的编辑只有三类：`git mv`、修 import 路径、加转发用的 `__init__`。
   任何"顺手改一下逻辑"都算越界，会让"行为一位不变"这个门失效。
2. **不删代码。** `platform.py` / `worker.py` 这类 v3 明确不要的东西，M0 只加 `deprecated` 注释 +
   保证没有新目录引用它们，删除留给 M1。
3. **旧路径留 shim。** 搬走的模块在旧路径留一行转发 import，一个发布周期后再删。
   否则外部脚本（`scripts/serve_qwen36.py` 等）会当场断。

---

## 1. 三个必须先处理的坑

| 坑 | 事实 | 不处理会怎样 |
|---|---|---|
| **`tests/test_no_vllm_ascend.py` 是 v2 的守卫** | 它用 AST + `sys.modules` 双重检查，**禁止**任何模块 import `vllm_ascend` | v3 要求 bridge 层 import vllm_ascend。不反转这个测试，新会话第一次写接入点就被 CI 判红 |
| **CPU 套件本来就是红的** | 102 上 `pytest tests -m 'not npu and not tilelang'` 有 80 项失败（跨文件顺序污染，逐文件是绿的） | 用"整套绿"当门会永远过不去。门只能是"逐文件 + 与基线比" |
| **平台层与 worker 是 v2 的核心资产、也是 v3 的删除对象** | `platform.py` 1954 行 + `worker.py` 2059 行 + `platform_dsv41.py` 67 行 ≈ 4.1k 行 | 搬进任何一层都会让"层"立刻失去意义。它们是 M0 的**冻结区**：原地不动、不搬、不引用 |

`test_no_vllm_ascend.py` 的正确改法（M0 第一件事）：把它从"全局禁止"改成"分层禁令"——
`kernels/`、`contracts/`、`resolve/`、`aot/` 不许 import 上游；只有 `bridge/` 可以。

---

## 2. 现状规模（决定搬的优先级）

`tileinfer/vllm/` 约 **21 k 行**，是整块 v2 后端。分布大致是：

| 组 | 行数 | v3 里的归属 |
|---|---|---|
| `attn/`（backend + impl + metadata + kv_cache + prefill） | ~2.9 k | bridge |
| `ops/`（fla 9 个文件 + moe/gdn/extern/fused_moe + candidates/switch/install/families） | ~8.9 k | 大部分 providers，小部分 resolve / bridge |
| `platform.py` + `worker.py` + `platform_dsv41.py` | ~4.1 k | **M1 删除**（M0 冻结） |
| `models/`（dsv41 适配 5 个文件） | ~1.7 k | models + bridge |
| `profiles/`（base + 两个具体 profile） | ~0.7 k | resolve（注册表）+ manifests（具体声明） |
| `plugin.py` / `compat/patches.py` | ~1.1 k | bridge（入口函数 + 接入点） |
| `doctor.py` / `graphs.py` / `config.py` | ~1.8 k | cli（体检）/ bridge（图钩子）/ resolve |

---

## 3. 六步，每步一个门

门的定义（每步都要跑）：

- **G-A**：改动涉及的测试文件**逐个**跑，结果与步骤 0 的基线一致（不要求整套绿）。
- **G-B**：`scripts/ti_trace.py run --preset smoke` 的首字 / 单步耗时与基线在噪声内（±5%）。
- **G-C**：`git diff` 里除了 `git mv`、import 路径、新增转发文件之外，没有其他改动。

| 步 | 动作 | 产出 | 门 |
|---|---|---|---|
| **S0 冻基线** | 记录逐文件测试结果、`ti_trace smoke` 数字、`git rev-parse HEAD` | `docs/notes/m0-baseline.md` | 三步可复现 |
| **S1 立骨架** | 建空目录 + `__init__`；改 `test_no_vllm_ascend.py` 为分层禁令；加 import 规则检查 | 目录存在，检查能跑 | G-A（零改动即零差异） |
| **S2 搬零依赖层** | `kernels/`（已在位）、`contracts/attention/`、`observ/`、`aot/prewarm` | 三处搬完，旧路径留 shim | G-A / G-C |
| **S3 搬 L1 + L2** | `models/dsv41/`、`manifests/`（现 profiles 的两个具体声明）、`resolve/`（profiles.base、candidates、switch、config、flags、buckets、planner） | 声明与决策分开 | G-A / G-B / G-C |
| **S4 搬 L4** | `providers/`：fla、moe、gdn、extern、fused_moe_oot、families、attention 后端（reference / tilelang） | 实现与声明同目录 | G-A / G-B / G-C |
| **S5 搬 L5** | `bridge/`：plugin → 入口函数；compat/patches → 接入点；attn/* → 后端适配；doctor → cli | 接入层收敛到一处 | G-A / G-B / G-C |
| **S6 收尾** | `tileinfer/vllm/` 只留冻结区（platform / worker / platform_dsv41）+ shim；写 M1 删除清单 | 冻结区有注释、无引用 | 全部门 + `grep` 证明冻结区零引用 |

顺序理由：**先搬"零依赖"的（S2），最后搬"碰上游"的（S5）**。这样每一步都能独立回滚，
且越靠后越清楚哪些代码真的需要留在接入层。

> 2026-11-12 的实际推进：S0 基线见 [`notes/m0-baseline.md`](notes/m0-baseline.md)；
> v3 分层骨架 + 决策 + bridge + 端到端验收见 [`notes/m1-v3-bridge.md`](notes/m1-v3-bridge.md)。
> v2 冻结区与删除清单见 [`notes/m1-delete-list.md`](notes/m1-delete-list.md)。
>
> **S4 已修订**：原计划把 fla / moe / gdn / fused_moe 等整族搬进 `providers/`。按 v3 的
> "只补缺口"规则（[`extending.md`](extending.md) §3.0），这些是厂商**已经覆盖**的能力，
> 搬进去等于重做一遍，因此不再搬：它们留在冻结区，按删除清单逐批清理；`providers/`
> 只收厂商确实不能用的实现。所以 S4 的产出从"实现与声明同目录"改成**一张确认清单**：
> 哪条形状厂商真的用不了。当前结论是"没有"，本层为空。

---

## 4. 映射表（现有路径 → 目标 → 动作）

### 4.1 直接搬（move-only）

| 现有 | 目标 | 备注 |
|---|---|---|
| `tileinfer/kernels/**` | `kernels/**` | 目录名不变，只需补 `aot/` |
| `tileinfer/attention/api.py` | `contracts/attention/api.py` | 对外入口 |
| `tileinfer/attention/backends/base.py` | `contracts/attention/backend_base.py` | 接口 + 注册表 |
| `tileinfer/metadata.py` | `contracts/attention/metadata.py` | 元数据模型 |
| `tileinfer/plan.py` | `contracts/attention/plan.py` | 计划与调度 |
| `tileinfer/attention/backends/reference.py` | `providers/reference/attention.py` | 正确性 oracle |
| `tileinfer/attention/backends/tilelang_ascend950.py` | `providers/tileinfer/attention_950.py` | 主力实现 |
| `tileinfer/attention/backends/tilelang_ascend.py` | `providers/tileinfer/attention_legacy.py` | 旧变体，M1 评估是否弃用 |
| `tileinfer/testing/reference.py` | `providers/reference/torch_ref.py` | 参考实现 |
| `tileinfer/testing/{gdn,moe,sampling}_reference.py`、`cache.py` | `testing/` | 测试 oracle，留在测试侧 |
| `tileinfer/integrations/prewarm.py` | `aot/prewarm.py` | 构建期预热 |
| `tileinfer/serving/{phases,counters,step}.py` | `observ/` | 计时与计数 |
| `tileinfer/models/dsv41/**` | `models/dsv41/**` | 路线 O 的样例 |
| `tileinfer/vllm/profiles/{dsv41,qwen3_6}.py` | `manifests/` | 具体声明（先保持 py，后续可转 yaml） |
| `tileinfer/vllm/profiles/base.py` | `resolve/profiles.py` | 注册表 + 匹配规则 |
| `tileinfer/vllm/ops/{candidates,switch}.py` | `resolve/candidates.py`、`resolve/switch.py` | 候选与选择 |
| `tileinfer/vllm/{config.py}`、`serving/flags.py` | `resolve/` | 配置与开关 |
| `tileinfer/serving/{buckets,planner}.py` | `resolve/planning.py` | 桶规划（决策的一部分） |
| `tileinfer/vllm/ops/fla/**` | `providers/tileinfer/fla/**` | 若来源于上游 fla，改放 `providers/vendored/` 并写来源 |
| `tileinfer/vllm/ops/{gdn,moe,extern,fused_moe_oot}.py` | `providers/tileinfer/` 或 `providers/vendored/` | 逐个判来源 |
| `tileinfer/vllm/ops/families/gdn_moe.py` | `providers/tileinfer/families/` | 算子族声明 |
| `tileinfer/vllm/plugin.py` | `bridge/install.py` | **入口函数** |
| `tileinfer/vllm/compat/patches.py` | `bridge/vllm_0_23/seams.py` | 六个接入点的实现 |
| `tileinfer/vllm/attn/{backend,impl,metadata,kv_cache,prefill}.py` | `bridge/vllm_0_23/attn/` | 后端适配 |
| `tileinfer/vllm/doctor.py` | `cli/doctor.py` | 体检四道门 |

### 4.2 拆开搬（一个文件进多处分）

| 现有 | 拆成 | 判据 |
|---|---|---|
| `tileinfer/vllm/ops/install.py`（755） | `providers/*/declare` + `bridge/seams` | "声明这个实现覆盖什么"归 providers；"把它装进引擎"归 bridge |
| `tileinfer/vllm/models/dsv41*.py`（5 个文件，~1.7k） | `models/dsv41/vllm_adapter/` + `bridge/` 的注册片段 | 模型结构归 models；注册到引擎归 bridge |
| `tileinfer/vllm/graphs.py`（929） | `bridge/.../graph_hook.py` | 图捕获归厂商，我们只保留"捕获前桶规划、捕获后零回退确认" |

### 4.3 冻结（M0 不动，M1 删）

| 现有 | 为什么冻结 |
|---|---|
| `tileinfer/vllm/platform.py`（1954） | v2 的平台层。v3 里平台归厂商——这是 v2 付了 2.53× 代价换来的东西，别再搬进新层 |
| `tileinfer/vllm/worker.py`（2059） | 同上（worker 归厂商） |
| `tileinfer/vllm/platform_dsv41.py`（67） | 同上 |

M0 对这些文件只做两件事：在文件头加 `deprecated` 说明；用 `grep` 证明新目录没有引用它们。

---

## 5. M0 完成的定义

- 新目录存在且**各自只 import 该 import 的东西**（分层禁令在 CI 里能跑）。
- `tileinfer/vllm/` 只剩冻结区 + shim，且**没有任何新目录引用它**。
- 逐文件测试结果与 S0 基线一致；`ti_trace smoke` 数字在噪声内。
- `docs/notes/m0-baseline.md` 与 M1 删除清单已写。

**没做到的**：功能一行没加。M0 的价值是"让后面每一次改动都有唯一的落点"。
