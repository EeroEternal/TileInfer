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

## 4. 下一步（Phase 1，未做）

1. 先闭合 TP1 的 Nemotron C2/MTP（否则 TP 没有正确对照）；
2. TP=2 数值正确：Mamba2 head/state 按 TP 切、MoE 先做 TP-shard（`expert_map` 留给 EP）；
3. TP=2 过 C1/C2 后再 4/8；随后补 C3 图、MTP、长上下文、并发；
4. EP 单独做（`providers/moe.py` 的 `expert_map` 骨架接 all-to-all）；
5. `prewarm`/AOT 的规格乘上每 rank 的 local heads（只在我们自己的 TileLang 内核进多卡时需要）。

停止条件：让 provider TP 安全需要改/绕过厂商分布式 plumbing → 收缩为"厂商并行可用、我方
provider 拒绝多卡"；两次尝试仍定位不到失败层 → 先补观测。
