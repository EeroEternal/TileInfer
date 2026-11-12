# Nemotron provider 契约草案（O4，离线）

> 状态：**候选，不是默认路径**。没有实机证据的 provider 只能标 `candidate`；
> 不能写进 plan 的默认选择，也不能声称“已补缺口”。

---

## 1. 规则

按 [`extending.md`](../extending.md) §3.0 与 [`acceptance.md`](../acceptance.md) §2.1：

- 厂商能跑、能跑对、能上图 → 选厂商，**不写 provider**；
- 厂商不能跑 → 才允许 provider；
- 无厂商基线时，`measured_ratio` 留空，但 `evidence` 不能空；
- 每条 provider 必须有 `delete_when`（通常是上游补齐）。

当前 Nemotron 的静态缺口见 [`m5-nemotron-static-gap.md`](m5-nemotron-static-gap.md)；
在实机确认前，G1–G7 都是 `unknown`，不是“厂商确定不支持”。

---

## 2. 候选 provider 表

| # | 缺口 | 候选能力 | 实现形态 | graphable | parallel | evidence | delete_when |
|---|---|---|---|---|---|---|---|
| P1 | Ascend 没有 NemotronH 模型注册/加载 | 模型路由（model table） | 注册/包装 `NemotronHForCausalLM`，不改上游 | 待测 | single | `candidate`（无） | vllm-ascend 原生注册 NemotronH |
| P2 | Mamba2 mixer 的 SSD/chunk-scan 执行 | `state.linear`（`Mamba2Contract` 子契约） | Mamba2 conv/SSM/state provider | 待测 | single | `candidate`（静态未见 Ascend SSD 替换） | 厂商 Mamba2/SSD backend 覆盖 |
| P3 | 混合 full-attention + Mamba state 的 KV/布局 | `attention.full` + `state.linear` | 状态/KV 适配；不重写 KV 池 | 待测 | single | `candidate`（无） | 厂商支持混合 KV/MambaSpec |
| P4 | MTP `NemotronHMTPModel` 的 Ascend 注册/执行 | MTP/模型路由 | 复用 vLLM MTP，仅补注册/包装 | 待测 | single | `candidate`（无） | 厂商注册并验证 NemotronHMTP |
| P5 | Mamba/混合 attention 图捕获 | graph hook（不是 provider） | 捕获前检查 bucket，捕获后计数回退 | 待测 | — | `candidate`（无） | 厂商图路径覆盖 |
| P6 | 权重映射（`mixer.{k,v}_proj` 等） | 加载器 | loader 适配/映射 | — | single | `candidate`（无） | 厂商 loader 覆盖 |
| P7 | 参考实现（HF/transformers） | reference（不进默认路径） | CPU/参考 harness | — | — | 常驻参考 | 不需要删 |

> P1/P4 是“模型路由”，不是自研算子；如果 vLLM 通用路径在 Ascend 上能跑，就不应写
> provider。P2/P3 才是真正可能需要算子的地方，但必须先上机确认 vllm-ascend 的
> Mamba Triton 算子是否已覆盖目标形状。

---

## 3. 五条声明怎么填

以 P2 为例（仅示意，未验证）：

| 字段 | 草案值 | 说明 |
|---|---|---|
| `coverage` | `{"layer": "mamba2", "dtype": ["bf16"], "head_dim": "TODO"}` | 符号表达，不写具体 tensor |
| `graphable` | `false`（先保守） | 未测前不能写 true |
| `parallel` | `("single",)` | 当前只单卡 |
| `evidence` | `candidate` | 没有证据不能进默认路径 |
| `delete_when` | 厂商 Mamba2 backend 覆盖该形状 | 必须可判定 |

---

## 4. 当前不能做的事

- 不能把 P1–P7 直接写成 `ProviderSpec` 默认项：`ProviderSpec.validate()` 要求
  `evidence`，而当前只有 `candidate`；
- 不能为 Mamba2 复用 GDN 的 `LinearAttentionContract`：参数/状态语义不同，会把
  “能跑”写成“跑对”。已加 `Mamba2Contract`（`forward` + `state_shape`）作为
  `state.linear` 的子契约；但**没有实机证据前仍只是 candidate**，不能进默认路径。
- 不能在没有 config 的情况下写死 `coverage` 的数字；
- 不能因为 vllm-ascend 有 `ops/triton/mamba/` 就认定模型级支持已存在。

---

## 5. 完成 O4 的判据

- 每个真实缺口对应一条候选 provider；
- 每条都有 `coverage/graphable/parallel/evidence/delete_when`；
- 没有证据的标 `candidate`，不进 plan；
- 实机后把 `candidate` 改成实测证据，或删除 provider（厂商已覆盖）。
