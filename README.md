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

- **v3 接入层已建成（M1）**：`tileinfer/bridge/`（6 个接入点 + 版本门 + 指标）、`contracts/`、
  `resolve/`、`manifests/`、`cli/` 均已实现；`pyproject.toml` 只注册
  `vllm.general_plugins → tileinfer.bridge.install:install`，不注册平台插件。当前进度与每条断点见
  [`docs/status.md`](docs/status.md)。
- **v2 运行时（`tileinfer/vllm/`）保留为冻结基线**，删除清单未执行。v3 的分层不 import 它；它也不再
  是打包入口（v2 的插件入口只出现在该目录的文档字符串里）。注意：v3 运行需要 vllm-ascend，在未安装
  vllm-ascend 的机器上，v2 是唯一可运行的形态。
- 已有资产：注意力 / GDN / MoE / 采样四族内核、`plan/run` 注意力契约、分相计时的剖析工具、
  v1 时代在役过的 vllm-ascend 插件（冻结在 `legacy/`）。
- 性能数字一律以 [`docs/performance.md`](docs/performance.md) 为准，**不要从 README 或旧文档引用**。

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
| `docs/notes/` | 契约转录、事实清单、历史归档 |

> 历史文档已归档到 `docs/notes/archive/`，头部标了 SUPERSEDED，**不要按它们建目录或改代码**。

## 目录

**目标形态的目录与分层对应关系见 [`docs/architecture.md`](docs/architecture.md) §3.3**——那张表是
权威的层↔目录映射，本处只列"现在硬盘上有什么"。

| 路径（当前） | 是什么 |
|---|---|
| `tileinfer/kernels/` | TileLang 算子（attention / GDN / MoE / sampling） |
| `tileinfer/vllm/` | **v2 形态**的后端本体（平台层、worker hook、attention 后端、图捕获）——冻结基线，v3 分层不 import 它，删除清单见 `docs/status.md` |
| `tileinfer/models/` | 上游没有的模型实现（DeepSeek-V4.1 是样例） |
| `tileinfer/serving/` | serving 侧工具（分相计时、bucket、planner） |
| `legacy/` | 冻结的 vllm-ascend 时代代码：不打包、不测试、不再增长，是 v3 接入层的素材库 |
| `reference/vllm-0.23/` | vLLM 0.23 关键模块逐字副本，只为让 `file:line` 引用可核对，**不许改** |
| `benchmarks/`、`scripts/` | A/B harness、探针、统一剖析入口 `scripts/ti_trace.py` |
| `docs/` | 见上面的文档地图 |

## 环境

- 硬件：1 × Ascend950PR（单卡是当前唯一验证过的形态）
- 栈：CANN 9.3.0、torch 2.13.0、torch_npu、vLLM 0.23.0、tilelang 0.1.15
- 统一入口：`source scripts/env-ascend102.sh`。**环境定义必须在仓库里**，不许只活在临时目录。

## License

Apache License 2.0，见 [LICENSE](LICENSE)。`reference/vllm-0.23/` 是 vLLM（Apache-2.0）的未修改副本，
来源与规则见该目录的 `README.md`。
