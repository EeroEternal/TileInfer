# 多卡 Phase 0：并行成为声明与门禁（2026-11-17）

> 目标不是"已经支持多卡"，而是**先让"不支持"可判定**：环境记录并行维度，计划记录并行
> 配置，选择对单卡-only provider 在多卡下 fail-closed，`doctor` 有一道门。
> 仍未支持：provider 没有做 TP 数值验证，EP 未实现，TP8 的图/MTP/长上下文未验。

## 0. 结论

| 项 | 状态 |
|---|---|
| 并行维度进入 `Environment` | ✅ `tp_size/ep_size/dp_size/pp_size` + `parallel_axes()` |
| 计划记录并行配置 | ✅ `Plan.parallel`，旧计划缺字段时按 `("single",)` 读取 |
| provider 声明并行 | ✅ `ProviderSpec.parallel` 封闭词表（`single/tp/ep/dp/pp`） |
| 选择层 fail-closed | ✅ 单卡-only provider 在任意分片轴下不被选中，写成 gap |
| `doctor` 门 | ✅ 计划并行 ≠ 现场并行 → FAIL；非厂商 selection 覆盖不了现场轴 → FAIL |
| 厂商 provider | ✅ 声明支持全部轴（平台层负责分片，我们不安装请求路径） |
| 我方 provider（LFM3/Nemotron） | ❌ 仍只声明 `("single",)`，多卡下被拒绝 |

## 1. 行为

- `tileinfer plan build --tp 2` 会写出 `parallel=["tp"]` 的计划；
- 同一计划在单卡环境跑 `doctor` 会 FAIL：`plan was built for parallel=['tp'] but this run is ['single']`；
- Qwen3.6 在 `--tp 2` 下仍全选厂商、`servable=true`（厂商平台负责分片）；
- LFM3 在 `--tp 2` 下 `state.linear` 变 gap，因为它的 short-conv provider 只声明单卡；
- Nemotron 的 Mamba2/MoE 候选在 `--ep 2` 下不会被选中，gap 原因写明"no support for parallel"。

## 2. 命令与证据

```bash
# 生成 TP=2 计划
.venv/bin/python -m tileinfer.cli.main plan build \
  --model /home/models/Qwen3.6-35B-A3B --out /tmp/ti-plan-tp2.json --tp 2
# -> plan=... parallel=tp  servable=True gaps=0

# doctor 必须看到相同的并行配置
TILEINFER_PLAN=/tmp/ti-plan-tp2.json TILEINFER_TP_SIZE=2 \
  .venv/bin/python -m tileinfer.cli.main doctor --json
# -> parallel PASS plan=['tp'] live=['tp']

# 不带 TILEINFER_TP_SIZE 必须失败
TILEINFER_PLAN=/tmp/ti-plan-tp2.json \
  .venv/bin/python -m tileinfer.cli.main doctor --json
# -> parallel FAIL
```

CPU 套件（逐文件）：

```bash
.venv/bin/python -m pytest tests/test_parallel.py -q          # 14 passed
.venv/bin/python -m pytest tests/test_v3_impl_table.py -q     # 8 passed
.venv/bin/python -m pytest tests/test_lfm3_manifest.py -q     # 4 passed
```

`tests/test_v3_contracts.py` 的两条 `tileinfer.vllm` 目录断言与
`tests/test_nemotron_offline.py` 的静态审计断言在本机为既有环境性失败（空目录/本机 vLLM
registry 与审计脚本预期不一致），与本次改动无关。

## 3. 改动面

| 文件 | 改动 |
|---|---|
| `tileinfer/contracts/capabilities.py` | `PARALLEL_MODES`、`normalize_parallel`、`parallel_supports`，`ProviderSpec.validate` 校验声明 |
| `tileinfer/contracts/plan.py` | `Plan.parallel`（写入/读取/校验/plan_id/摘要） |
| `tileinfer/resolve/probe.py` | `Environment` 并行字段 + `parallel_axes()`；`probe()` 读 CLI/环境变量 |
| `tileinfer/resolve/decide.py` | 选择层按 `provider.parallel ⊇ env.parallel_axes()` 过滤；计划写入 `parallel` |
| `tileinfer/resolve/vendor.py` | 厂商条目声明全轴 |
| `tileinfer/cli/plan.py` | `--tp/--ep/--dp/--pp` |
| `tileinfer/cli/doctor.py` | `parallel` 门 |

**接入点数量不变（仍是 6 个）**；契约层没有新增机制，只是给已有声明加了一个维度。

## 4. Phase 1 进展（2026-11-18）

Provider 级门禁之外，补上了**模型级**门禁，并把第一个 route-O 模型的 TP 固化：

| 文件 | 改动 |
|---|---|
| `tileinfer/contracts/manifest.py` | `ModelManifest.parallel`（默认 `("single",)`，仅 route-O 生效；非默认才写入 `as_dict`/fingerprint） |
| `tileinfer/resolve/decide.py` | route-O 模型在环境轴不被 `manifest.parallel` 覆盖时，追加 required gap `model.parallel`、`servable=false`；厂商模型不拦 |
| `tileinfer/cli/doctor.py` | 加载 plan 的 manifest，route-O 模型声明不覆盖现场轴 → `parallel` FAIL |
| `tileinfer/manifests/muse_glimmer.py` | `parallel=("single","tp")`（依据 TP=2 的 C1/C2/C3 与 `tolerance_tp`） |
| `tileinfer/manifests/lfm3_1_7.py` | 显式 `parallel=("single",)`（short-conv 未做 TP，fail-closed） |
| `tileinfer/manifests/nemotron_3_5.py` | `support.tp` 改为 `["1"]`，注明候选 provider 单卡、fail-closed |
| `docs/acceptance.md` | 新增 §2.1.1：跨并行配置的 C2 必须由 `benchmark.tolerance_tp` 显式声明 |
| `docs/verification-matrix.md` | Muse 增加 `C2-TP` 行（1.14e-3 ≤ 2e-3，token 集合相同） |

MuseGlimmer 的 TP=2 设备证据（125K 单流 18.4 s/30.6 ms、128K×4 74.2 s max TTFT）见
[`muse-glimmer-v3-validation-20261117.md`](muse-glimmer-v3-validation-20261117.md) §4.2。

**2026-11-18 设备补充**（卡 0-7，HCCL rootinfo 临时移走/恢复）：

- **Muse TP=2 C3**：6 图（PIECEWISE 3 + FULL 3）、`decode_calls=99`、99/99 命中、
  `fallbacks=0`、0 errors；补上之前缺的"TP 下图捕获"证据；
- **Qwen3.6 TP=4**：health 75 s、6 图、99/99 命中、0 回退；
- **Qwen3.6 TP=8**：health 90 s、捕获 6 图，但**首个请求挂起**（`Running: 1 reqs`、
  `No available shared memory broadcast block found in 60 seconds`、`decode_calls=0`）；
  记为未验证，不写成支持。细节见 [`qwen36-tp4-tp8-20261118.md`](qwen36-tp4-tp8-20261118.md)。

工具：`doctor` 新增 `hccl_rootinfo` 预检（空目录/空文件/非法 JSON 会直接说明原因），
`scripts/tp-smoke.sh <model> <tp>` 做 plan+doctor 预检并打印设备命令（不占卡）。

## 5. 仍未做

1. **LFM3 provider TP**：short-conv 的 depthwise 通道切分 + state 布局；验证后再加 `tp`。
2. **Nemotron provider TP**：先闭合 TP1 的 C2/MTP，再做 Mamba2（本地 head/state）与
   MoE TP-shard（`expert_map` 留给 EP）。
3. **Qwen3.6 TP=8** 的挂起（厂商/HCCL/图路径）；TP=2/4 已过 C1/C3。TP 下的性能/并发/
   长上下文仍未做；控制模型，点到为止。
4. **EP / DP / PP**：EP 单独做；DP 主要校验 metrics 按 rank；PP 延后。
5. `prewarm`/AOT 的规格乘上每 rank 的 local heads（只在我们自己的 TileLang 内核进多卡时需要）。

停止条件：让 provider TP 安全需要改/绕过厂商分布式 plumbing → 收缩为"厂商并行可用、我方
provider 拒绝多卡"；两次尝试仍定位不到失败层 → 先补观测。
