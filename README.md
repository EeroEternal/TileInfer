# TileInfer

把 **TileInfer 做成 vLLM 的原生昇腾后端**：`pip install vllm tileinfer` 之后，
`vllm serve Qwen3.6-35B-A3B` 在**一张 Ascend 950** 上跑起来，**不安装 `vllm_ascend`**，
性能不低于现网（vLLM 0.23 + vllm-ascend 0.23 + 图模式）。

```
      vLLM 0.23 (上游，未打补丁)
              │  平台插件 + 通用插件两个官方扩展点
    ┌─────────▼──────────┐
    │  tileinfer.vllm    │  平台层 / worker hook / attention 后端 / 图捕获
    ├────────────────────┤
    │  tileinfer.kernels │  自有算子（TileLang）+ 可混用的上游与厂商算子
    ├────────────────────┤
    │  CANN 9.3 + 950    │
    └────────────────────┘
```

## 现在是什么状态

后端能起服务、能出正确结果、能跑混合 KV（30 层 GDN 线性注意力 + 10 层 full attention），
但**性能还没有和基线打平**：TPOT 2.53×、TTFT 11.8× 于 vllm-ascend 基线，图捕获还没接上。
确切数字、每条差距的归因、以及哪些结论已经被实验推翻，都在
[`docs/status.md`](docs/status.md)。不要从这里或旧文档里引用性能数字。

## 三条价值 / 三个目标

1. **单卡性能**：把 TTFT/TPOT 拉到与 vllm-ascend 基线可比，并把图捕获、投机解码接上。
2. **算子取舍**：用自有算子，同时有能力在同一个名字下混用上游与厂商实现，共同让性能最好。
3. **平台泛化**：同一套代码结构能换卡（如海光）、能加模型（DeepSeek-V4.1 是第二个样例）。

三块怎么分工、里程碑、验收口径见 [`docs/plan-vllm-native-backend.md`](docs/plan-vllm-native-backend.md)。

## 快速开始（在 102 机器上）

```bash
source scripts/env-ascend102.sh            # CANN 9.3 + TileLang 缓存目录 + 运行期开关
.venv_native213/bin/python -m vllm.entrypoints.cli.main serve \
  /home/models/Qwen3.5-35B-A3B \
  --attention-backend CUSTOM --enforce-eager \
  --max-model-len 9216 --max-num-batched-tokens 9216 --max-num-seqs 32 \
  --gpu-memory-utilization 0.75 --trust-remote-code
```

一键性能剖析（起服务 → 打流 → 采集 → 出报告）：

```bash
python3 scripts/ti_trace.py config                       # 看当前配置
python3 scripts/ti_trace.py presets                      # 看可用预设
python3 scripts/ti_trace.py run --preset smoke --dry-run # 只打印将要执行的计划
python3 scripts/ti_trace.py run --preset smoke           # 真跑，报告落 trace_out/
```

## 目录

| 路径 | 是什么 |
|---|---|
| `tileinfer/vllm/` | 后端本体：平台层、worker hook、attention 后端、图捕获、算子接线 |
| `tileinfer/kernels/` | TileLang 算子（attention / GDN / MoE / sampling） |
| `tileinfer/models/dsv41/` | DeepSeek-V4.1 的实现，作为"加一个模型"的样例保留 |
| `tileinfer/serving/` | 与引擎无关的 serving 侧工具（分相计时、bucket、planner） |
| `reference/vllm-0.23/` | vLLM 0.23 关键模块的逐字副本，**只**用于让 `file:line` 引用可核对 |
| `legacy/` | 冻结的 vllm-ascend 时代代码，不打包、不测试、不再增长；是算子取用的素材库 |
| `tools/dsv41/` | DeepSeek-V4.1 的部署与验证脚本 |
| `benchmarks/` | A/B harness 与探针；`scripts/ti_trace.py` 是统一入口 |
| `docs/` | 见下面的文档地图 |

## 文档地图

先读这两篇，其余按需：

| | |
|---|---|
| [`docs/status.md`](docs/status.md) | 现在到哪一步了，凭什么相信，哪里断了 |
| [`docs/plan-vllm-native-backend.md`](docs/plan-vllm-native-backend.md) | 要做什么、怎么分块、里程碑与验收口径 |

实现细节：

| | |
|---|---|
| [`docs/integration-vllm-native.md`](docs/integration-vllm-native.md) | 用了 vLLM 的哪些官方扩展点，耦合在哪 |
| [`docs/extending.md`](docs/extending.md) | 加一个模型 / 一族算子 / 一个更快的实现 |
| [`docs/hardware-checklist.md`](docs/hardware-checklist.md) | 上机点亮 11 步，每步的失败归因 |
| [`docs/known-issues.md`](docs/known-issues.md) | 踩过的坑与事后复盘（读它能省几天） |
| [`docs/performance.md`](docs/performance.md) | 性能口径、基线、当前数字、测量方法 |
| [`docs/architecture.md`](docs/architecture.md) | plan/run 契约、metadata 模型、调度器 |

契约与转录（写代码时要对着看的）：

`docs/notes/vllm-api-0.23.md` · `docs/notes/vllm-attention-contract.md` ·
`docs/notes/fla-chunk-contract.md` · `docs/notes/op-coverage-qwen36.md` ·
`docs/notes/perf-candidates.md` · `docs/notes/perf-comparison-rules.md`

## 环境

- 硬件：1 × Ascend950PR（单卡是当前唯一验证过的形态）
- 栈：CANN 9.3.0、torch 2.13.0+cpu、torch_npu、vLLM 0.23.0、tilelang 0.1.15
- Python 环境：`.venv_native213`（本题后端）、`.venv_baseline`（vllm-ascend 基线，A/B 用）
- 统一入口：`scripts/env-ascend102.sh`。**环境定义必须在仓库里**，不能只活在某个临时目录

## License

Apache License 2.0 — 见 [LICENSE](LICENSE)。`reference/vllm-0.23/` 为 vLLM（Apache-2.0）的
未修改副本，来源与规则见该目录的 `README.md`。
