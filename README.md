# TileInfer

在 **vllm-ascend 之上**加一层兼容与适配层：让更多模型能在一张昇腾卡上跑起来，跑得对、跑得动、
不比基线慢；**已经跑得好的模型，一行都不碰**。

```
                vLLM 0.23（上游，不打补丁）
                          │
              ┌───────────▼────────────┐
              │   vllm-ascend 0.23     │  平台 / worker / 显存 / KV 池 / 图捕获
              └───────────┬────────────┘
                          │   6 个接入点（注册表替换 + 方法包装，不改源码）
              ┌───────────▼────────────┐
              │   TileInfer（本仓库）   │  模型声明 · 决策 · 契约 · 提供者 · 接入 · 内核
              └───────────┬────────────┘
                          │
                  CANN 9.3 + Ascend 950
```

**一句话定位**：我们是**选择层 + 补齐层**，不是另一个推理引擎。
vllm-ascend 已经适配的模型走它；它没适配的模型，我们把缺的那部分补上。

**边界（重要）**：设备、worker、显存、KV 池、图捕获归 vllm-ascend。
我们不重写平台层——上一个版本（v2）试过，代价是 TPOT 2.53×、TTFT 11.8×。

## 现在是什么状态

- **定位已改为 v3（兼容适配层）**；**代码仍是 v2 形态**（自带平台层、不装 vllm-ascend）。
  重构计划见架构文档的里程碑，当前进度与每条断点见 [`docs/status.md`](docs/status.md)。
- 已有资产：注意力 / GDN / MoE / 采样四族内核、`plan/run` 注意力契约、分相计时的剖析工具、
  v1 时代在役过的 vllm-ascend 插件（冻结在 `legacy/`）。
- 性能数字一律以 [`docs/performance.md`](docs/performance.md) 为准，**不要从 README 或旧文档引用**。

## 文档地图

先读三篇：

| | |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | 定位、分层、生命周期、两条路线、不变量、硬约束 |
| [`docs/acceptance.md`](docs/acceptance.md) | 目标与验收口径、KPI、红线（什么情况必须收缩） |
| [`docs/status.md`](docs/status.md) | 现在到哪一步、凭什么相信、哪里还断着 |

按需：

| | |
|---|---|
| [`docs/integration.md`](docs/integration.md) | 与 vLLM / vllm-ascend 的接入关系（机制层，不含代码） |
| [`docs/extending.md`](docs/extending.md) | 加一个模型 / 一个实现提供者 / 一个接入点 |
| [`docs/hardware-checklist.md`](docs/hardware-checklist.md) | 上机点亮清单（操作手册，不是架构） |
| [`docs/known-issues.md`](docs/known-issues.md) | 踩过的坑与复盘（读它能省几天） |
| [`docs/performance.md`](docs/performance.md) | 性能口径、基线、测量方法 |
| `docs/notes/` | 契约转录、事实清单、历史归档 |

> 历史文档已归档到 `docs/notes/archive/`，头部标了 SUPERSEDED，**不要按它们建目录或改代码**。

## 目录

| 路径 | 是什么 |
|---|---|
| `tileinfer/kernels/` | TileLang 算子（attention / GDN / MoE / sampling） |
| `tileinfer/vllm/` | **v2 形态**的后端本体（平台层、worker hook、attention 后端、图捕获）——v3 会拆进分层 |
| `tileinfer/models/` | 上游没有的模型实现（DeepSeek-V4.1 是样例） |
| `tileinfer/serving/` | serving 侧工具（分相计时、bucket、planner） |
| `legacy/` | 冻结的 vllm-ascend 时代代码：不打包、不测试、不再增长，是 v3 接入层的素材库 |
| `reference/vllm-0.23/` | vLLM 0.23 关键模块逐字副本，只为让 `file:line` 引用可核对 |
| `benchmarks/`、`scripts/` | A/B harness、探针、统一剖析入口 `scripts/ti_trace.py` |
| `docs/` | 见上面的文档地图 |

## 环境

- 硬件：1 × Ascend950PR（单卡是当前唯一验证过的形态）
- 栈：CANN 9.3.0、torch 2.13.0、torch_npu、vLLM 0.23.0、tilelang 0.1.15
- 统一入口：`source scripts/env-ascend102.sh`。**环境定义必须在仓库里**，不许只活在临时目录。

## License

Apache License 2.0，见 [LICENSE](LICENSE)。`reference/vllm-0.23/` 是 vLLM（Apache-2.0）的未修改副本，
来源与规则见该目录的 `README.md`。
