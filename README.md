# TileInfer

在 **vllm-ascend 之上**加一层兼容与适配层：让更多模型能在一张昇腾卡上跑起来，跑得对、跑得动、
不比基线慢；**已经跑得好的模型，一行都不碰**。

```
                     部署期（一次）
   tileinfer 启动器：读模型声明 + 探测本机 → 计划：这次注入哪些实现
        ├─ 一项都不注入 → 直接起 vLLM（我们的代码不进热路径）
        └─ 注入 N 项     → 起 vLLM，插件按计划换表

                     运行期（一个进程）
                     vLLM 引擎（上游，不打补丁）
                        │                          │
        A 查表类：用谁的实现                B 平台类：这台机器怎么用
        算子 / 层 / 模型 / 注意力后端        设备 / worker / 显存 / KV / 图捕获
                        │                          │
                        ▼                          │
              ┌───────────────────┐                │
              │ TileInfer 选择性   │                │
              │ 拦截（6 个接入点） │                │
              └────────┬──────────┘                │
                       └──────────────┬─────────────┘
                                      ▼
                            vllm-ascend 平台层（唯一平台）
                                      ▼
                             CANN 9.3 + Ascend 950
```

**两个关键事实**（读任何其它文档前先记住）：

1. **平台槽位始终归 vllm-ascend，我们不注册平台插件。** 一个进程只允许一个平台激活；
   我们再去注册只会和厂商撞车，而收益为零。
2. **B 类请求直连厂商，不经过我们。** 只有"用谁的实现"这类查表请求才经过我们。
   所以这不是"通栏一层"，是**选择性拦截**。

**一句话定位**：我们是**选择层 + 补齐层**，不是另一个推理引擎。
vllm-ascend 已经适配的模型走它；它没适配的模型，我们把缺的那部分补上。

## 现在是什么状态

- **v3 接入层已建成（M1 + P0/P1/P2）**：`tileinfer/bridge/`（6 个接入点 + 版本门 + 指标）、
  `contracts/`、`resolve/`、`manifests/`、`cli/` 均已实现；P0 增加了 provider 生命周期字段、
  config adapter、manifest/plan `support`；P1 增加了 `WeightMapSpec`、`StateLayout`、
  `CompatHook`；P2 把 provider 查找收敛到 `providers/registry.py`、安装结果收敛到
  `InstallResult`、`ModelAdapters` 统一 config/model/parser 流水线、provider `install_actions` 声明化、
  plan 记录实际 `mechanisms`/`hooks`，并在 install 中执行 runtime parallel gate。
  `pyproject.toml` 只注册
  `vllm.general_plugins → tileinfer.bridge.install:install`，不注册平台插件。当前进度与每条断点见
  [`docs/status.md`](docs/status.md)。
- **v2 运行时（`tileinfer/vllm/`）已归档到 `legacy/v2/`**，不在安装包内、不可 import。v3 的
  分层不 import 它；它也不再是打包入口。注意：v3 运行需要 vllm-ascend；没有 vllm-ascend 的
  机器上，v2 只能作为源码参考（要运行请 checkout 归档前的提交）。
- 已有资产：注意力 / GDN / MoE / 采样四族内核、`plan/run` 注意力契约、分相计时的剖析工具、
  v1 时代在役过的 vllm-ascend 插件（冻结在 `legacy/`）。
- 性能数字一律以 [`docs/performance.md`](docs/performance.md) 为准，**不要从 README 或旧文档引用**。

### 已经接入的模型（不要把“C1 跑通”当成“完整支持”）

| 模型 | 路线 | 当前状态 | 证据 |
|---|---|---|---|
| **Qwen3.6-35B-A3B** | vendor-only 控制模型 | 8/8 vendor、gap=0；4K–128K C1–C4 过；262K L5 不过 | [`docs/notes/m1-v3-bridge.md`](docs/notes/m1-v3-bridge.md) |
| **MuseGlimmer-30B** | 路线 O，text-only | C1 通过、C3 8 图/8 命中/0 回退；C2/C4 未闭合，vision 未实现 | [`docs/notes/muse-glimmer-static.md`](docs/notes/muse-glimmer-static.md) |
| **LFM3-1.7B** | 路线 O + short-conv provider | C1/C3 通过；C2/C4 有历史证据，P1 后未重做完整 p50/p90；单卡、无量化/MTP | [`docs/notes/lfm3-adaptation.md`](docs/notes/lfm3-adaptation.md) |
| **Nemotron-3.5-30B-A3B** | vendor 模型 + config + candidate providers | C1/C3 通过；plan `servable=false`，C2 logprob 和 MTP 未过 | [`docs/notes/m5-nemotron-validation-20261116.md`](docs/notes/m5-nemotron-validation-20261116.md) |

### 适配层现在长什么样

```text
manifest（说明书）
  → decide（计划员：vendor / provider / model_impl / config_impl）
  → plan（装配清单，内容寻址）
  → bridge（唯一 import vLLM 的层）
       ├─ plugin_entry / config_gate   入口与配置翻译/注册
       ├─ model_table      注册/model 实现
       ├─ impl_table       注册 provider
       ├─ attn_select      注意力后端选择
       ├─ method_wrap      包装方法
       └─ graph_hook       图捕获计数
```

P0/P1/P2 的新增内容包括 `ProviderSpec` 的 status/scope/install/install_actor/graph_safe/
evidence_level、manifest/plan 的 `config_impl`/`support`/`hooks`/`mechanisms`，以及：

- `tileinfer/providers/registry.py`：所有 provider 的唯一查找入口；
- `tileinfer/bridge/results.py`：所有安装动作的统一 `InstallResult`；
- `tileinfer/bridge/model_adapters.py`：config adapter → register_config → register_models → parser 的顺序流水线；
- `tileinfer/bridge/config_adapter.py`：`config_impl` 的标准入口；
- `tileinfer/bridge/weight_map.py`：权重改名 / 融合 / skip / expert remap；
- `tileinfer/models/state_layout.py`：state 逻辑形状、Ascend 页对齐、padded shape；
- `tileinfer/bridge/compat_hooks.py`：vendor hook / platform guard / rope patch / model-side fix 的登记、禁用和删除条件；
- `tileinfer/bridge/runtime_gate.py`：install 和 doctor 共用的版本/并行/provider 适用性检查；
- `tileinfer/manifests/verify.py`：manifest ↔ checkpoint config 的通用一致性校验。

## 文档地图

**要开工就读 [`docs/long-term-task.md`](docs/long-term-task.md)**——任务书：目标、完成定义、工作顺序、
硬约束、每轮会话的接续协议。

要理解设计，先读三篇：

| | |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | 定位、分层、调用方向、层与目录的对应、生命周期、两条路线、不变量、硬约束 |
| [`docs/acceptance.md`](docs/acceptance.md) | 目标与验收口径、KPI、红线（什么情况必须收缩） |
| [`docs/status.md`](docs/status.md) | 现在到哪一步、凭什么相信、哪里还断着 |

按需：

| | |
|---|---|
| [`docs/integration.md`](docs/integration.md) | 与 vLLM / vllm-ascend 的接入关系（机制层，不含代码） |
| [`docs/migration-m0.md`](docs/migration-m0.md) | 现有代码怎么搬进新分层（M0：只搬不改，每步有门） |
| [`docs/extending.md`](docs/extending.md) | 加一个模型 / 一个实现提供者 / 一个接入点 |
| [`docs/hardware-checklist.md`](docs/hardware-checklist.md) | 上机点亮清单（操作手册，不是架构） |
| [`docs/known-issues.md`](docs/known-issues.md) | 踩过的坑与复盘（读它能省几天） |
| [`docs/performance.md`](docs/performance.md) | 性能口径、基线、测量方法 |
| [`docs/verification-matrix.md`](docs/verification-matrix.md) | 一个模型“在役”的完整矩阵（C1–C4、长上下文、图、投机、并发、冷启动） |
| [`docs/development-rules.md`](docs/development-rules.md) | **规范性**：v3 新代码放哪、provider 怎么装、证据门槛 |
| [`docs/quickstart-muse-950.md`](docs/quickstart-muse-950.md) | 本机跑 MuseGlimmer-30B 的自包含配方（环境、权重校验、启动、排错） |
| `docs/notes/` | 契约转录、事实清单、历史归档 |

> 历史文档已归档到 `docs/notes/archive/`，头部标了 SUPERSEDED，**不要按它们建目录或改代码**。

## 目录

**目标形态的目录与分层对应关系见 [`docs/architecture.md`](docs/architecture.md) §3.3**——那张表是
权威的层↔目录映射，本处只列"现在硬盘上有什么"。

| 路径（当前） | 是什么 |
|---|---|
| `tileinfer/kernels/` | TileLang 算子（attention / GDN / MoE / sampling） |
| `legacy/v2/tileinfer/vllm/` | **v2 形态**的后端本体（平台层、worker hook、attention 后端、图捕获）——源码归档，v3 不 import、不安装、不测试 |
| `tileinfer/bridge/models/` | 路线 O 的 vLLM 模型实现与注册模块（MuseGlimmer、LFM3、Nemotron config hooks） |
| `tileinfer/models/` | engine-free 模型辅助：`state_layout.py`、LFM3 compat 等；**不 import vLLM** |
| `tileinfer/bridge/` | 唯一 import vLLM 的装配层：config/model/impl 表、兼容 hook、指标 |
| `tileinfer/serving/` | serving 侧工具（分相计时、bucket、planner） |
| `legacy/` | v1/v2 冻结归档：不打包、不测试、不再增长，只作 A/B 参照与素材库 |
| `reference/vllm-0.23/` | vLLM 0.23 关键模块逐字副本，只为让 `file:line` 引用可核对，**不许改** |
| `benchmarks/`、`scripts/` | A/B harness、探针、统一剖析入口 `scripts/ti_trace.py` |
| `docs/` | 见上面的文档地图 |

## 环境

- 硬件：1 × Ascend950PR（单卡是当前唯一验证过的形态）
- 本机验证栈（2026-11-17）：CANN 9.3.0、torch 2.10.0+cpu、torch_npu 2.10.0.post4、
  vLLM 0.23.0、vllm-ascend 0.23.0、tilelang 0.1.15。
- LFM3 / Muse / Nemotron 真权重 C1 使用 `.venv_lfm3`（transformers 5.17.0）；还需要
  `PYTHONPATH=/home/zxs/triton-ascend-3.2.2:$PYTHONPATH`，否则 `vllm-ascend` 的 block table
  会走到错误的 Triton 实现。
- 通用 CPU 契约测试可用 zxs 的 4.57.6 环境；LFM3/Muse/Nemotron 的 NPU 运行请用 5.17 环境。
- 统一入口：`source scripts/env-ascend102.sh`。**环境定义必须在仓库里**，不许只活在临时目录。

## License

Apache License 2.0，见 [LICENSE](LICENSE)。`reference/vllm-0.23/` 是 vLLM（Apache-2.0）的未修改副本，
来源与规则见该目录的 `README.md`。
