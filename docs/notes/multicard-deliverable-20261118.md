# 多卡交付范围（2026-11-18）

> 这份文件回答："多卡现在交付到什么程度、哪些能声称、哪些明确不做"。
> 不是"所有模型都支持多卡"；是**一个可验证、失败即拒绝的交付边界**。

## 1. 交付内容

| # | 能力 | 状态 | 证据 |
|---|---|---|---|
| 1 | 并行维度进入环境/计划：`tp/ep/dp/pp` 探测、`Plan.parallel` 记录、`--tp/--ep/--dp/--pp` | ✅ | `resolve/probe.py`、`contracts/plan.py`、`cli/plan.py`；`tests/test_parallel.py` |
| 2 | Provider 级 fail-closed：单卡-only provider 在多卡下不被选中 | ✅ | `resolve/decide.py`；`tests/test_parallel.py` |
| 3 | **Route-O 模型级 fail-closed**：自带模型类声明 `parallel`，不支持则 `servable=false` | ✅ | `contracts/manifest.py`、`resolve/decide.py`、`cli/doctor.py` |
| 4 | 跨 TP 的 C2 口径：必须由 `benchmark.tolerance_tp` 显式声明 | ✅ | [`acceptance.md`](../acceptance.md) §2.1.1 |
| 5 | `doctor` 的 parallel 门 + **HCCL rootinfo 预检** | ✅ | `cli/doctor.py`；`tests/test_hccl_preflight.py` |
| 6 | MuseGlimmer TP=2：C1/C2/C3/**C4** + 128K×图/needle + 128K×4 | ✅ | [`muse-glimmer-v3-validation-20261117.md`](muse-glimmer-v3-validation-20261117.md) §4.2（C2 1.14e-3 ≤ 2e-3；C4 TPOT 24 ms、8K TTFT 0.62 s、8 路 279 tok/s、冷启动 75 s；128K×图 14/14、0 回退；needle `7391`） |
| 7 | Qwen3.6 TP=2/4：C1/C3（厂商路径） | ✅ | [`qwen36-tp4-tp8-20261118.md`](qwen36-tp4-tp8-20261118.md)（6 图、99/99、0 回退） |
| 8 | TP 预检脚本 | ✅ | `scripts/tp-smoke.sh <model> <tp>`（plan+doctor，不占卡） |

## 2. 可声称 / 不可声称

**可以写：**
- 多卡下不会装错：单卡-only 的 provider 和 route-O 模型会被拒绝（`servable=false`）；
- **MuseGlimmer 在 TP=2 下 C1–C4 通过**（C2 在声明的 `tolerance_tp` 内；C4 为绝对门），128K×图/needle、128K×4 可跑；
- **Qwen3.6 在 TP=2/4 下能跑、能上图、0 回退**（控制模型、厂商路径）；
- `doctor` 能在 TP 运行前指出宿主 HCCL rootinfo 的问题。

**不能写：**
- "Qwen3.6 TP=2 与 TP=1 数值一致"（无 `tolerance_tp`，第 7 token 分叉）；
- "Qwen3.6 TP=8 支持"（首个请求挂起，未验证）；
- "LFM3/Nemotron 支持多卡"（显式单卡、fail-closed）；
- "支持 EP/DP/PP"（未实现/未验证）；
- "多卡已交付生产"（宿主 HCCL 未根治、无设备 CI）。

## 3. 明确不做（本轮交付边界）

| 项 | 原因 |
|---|---|
| LFM3 TP | 1.7B 单卡足够；short-conv 是 depthwise、需要通道切分与 state 布局，收益低 |
| Nemotron provider TP | 候选 provider 单卡；TP1 的 C2/MTP 另线推进，TP 未验 |
| EP / PP | 未实现；EP 只在 MoE 必须专家并行时做 |
| Qwen TP=8 | 首个请求挂起（厂商/HCCL/图路径），先记未验证 |
| 宿主 `/etc/hccl_rootinfo.json` | 环境问题，需环境负责人修；`doctor` 已能提前报出 |

## 4. 复现

```bash
# 预检（不占卡），需要先 source 本机环境（CANN 9.3 + triton-ascend）
TILEINFER_PYTHON=/home/ernest/TileInfer/.venv/bin/python \
  scripts/tp-smoke.sh /home/models/Muse-Glimmer-30B 2

# 设备步骤：先 npu-smi 确认卡空闲，再起服务；TP>1 前确认 rootinfo 已修或临时移走
```

设备运行会占用卡，属于共享环境，必须按 [`verification-matrix.md`](../verification-matrix.md) §6
先说明命令/端口/时长并取得同意。
