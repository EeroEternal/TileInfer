# M5 交付模型任务书：Nemotron 3.5 Lightning 30B-A3B

> 状态：**实机 gate 已开始（2026-11-13）**。离线 O1–O7 已收口；目标权重仍为 5/14，
> 目标 30B 尚未加载。已完成目标 config/page 校验和合成同架构 Mamba2 机械探测；
> 目标 C1/C2/C3/C4 仍未执行，不能写“支持”。
>
> 关联：[`long-term-task.md`](../long-term-task.md) M5、[`acceptance.md`](../acceptance.md)、
> [`verification-matrix.md`](../verification-matrix.md)、[`model-bringup-layers.md`](model-bringup-layers.md)。

---

## 0. 任务卡

| 项 | 值 |
|---|---|
| 任务 ID | `M5-NEMOTRON-30B-A3B` |
| 模型 | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` |
| 规模/架构 | 31.6B 总参数；Mamba2 + attention 混合 + MoE（静态调研口径，待 config/权重确认） |
| 厂商状态 | vLLM 0.23 已注册 `NemotronHForCausalLM`、`NemotronHMTPModel`；vllm-ascend 0.23 无 Nemotron 模型注册（待上机确认是否完全不可用） |
| 目标栈 | **冻结** vLLM 0.23.0 / vllm-ascend 0.23.0-950；不升级，除非单独过 V0 gate |
| 硬件 | 单卡 Ascend950PR（131072 MB HBM），共享环境 |
| 实机约束 | 2026-11-13 用户已授权用卡；只在空闲卡/端口上起自己的进程，不 kill 他人；目标卡被占时停 |
| 交付目标 | 端到端推理跑通；MTP、图模式、长上下文、并发、冷启动、算子审计逐项留证；provider 只补真实缺口 |

---

## 1. 任务目标（可直接作为会话 goal 的文本）

```text
完成 TileInfer v3 对 nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 的端到端交付：

先离线完成所有不依赖实机的工作——静态 gap report、模型声明、plan/provider 契约、
参考实现方案、CPU/fake 测试和实机运行手册；今晚不启动服务、不占卡、不下载权重。

经用户明确同意后再上机，按顺序验证：
C1 服务能起并返回完整输出；
C2 与参考实现（HF/transformers 固定样例）一致，或按声明容差一致；
C3 图捕获 >0 且稳态回退 = 0；
C4 无厂商基线，按绝对 TTFT/TPOT/吞吐地板 + 参考实现一致性判定，不造 A/B；
MTP 投机、长上下文（8K/16K/32K/128K；config 声明 262144，262K 为条件档）、并发、冷启动、
算子级回退审计逐项留证。

硬约束：共享环境任何部署/重启/端口/占卡前必须说明并取得明确同意，不 kill 他人进程；
不改上游源码；接入点 ≤8；禁止请求期编译；禁止静默回退；没有证据不写“支持”。
```

---

## 2. 完成定义（DoD）

只有下面全部满足，才能把该模型写进“在役”：

| # | 结论 | 门 | 证据 |
|---|---|---|---|
| D1 | 能起 | 单请求返回完整输出，无异常/截断 | 服务日志 + 完整响应 |
| D2 | 跑对 | 固定短样例与 HF/transformers 参考一致；逐位不可比时按声明容差 | 两臂输出、logits/文本 diff、容差声明 |
| D3 | 能上图 | `captured_graphs > 0`，稳态 `fallbacks = 0` | 捕获/回退计数 |
| D4 | MTP | 若 checkpoint 含 MTP heads：投机开启、质量门过、图捕获/回退留证；若不含：**上报缺口，不得用 ngram 冒充 MTP** | 配置、接受率、质量对比 |
| D5 | 长上下文 | 8K/16K/32K/128K 逐档；config 声明 262144，262K 作为最后条件档（HBM/图/质量通过才留） | TTFT/TPOT、HBM、needle/质量、回退 |
| D6 | 并发 | 1/4/16/32；`max-num-seqs ≥ 客户端并发` | 吞吐、TTFT/TPOT、回退 |
| D7 | 冷启动 | 清编译缓存后启动；请求期零编译 | 到 `/health` 时间、首请求 TTFT、编译日志 |
| D8 | 算子审计 | 热路径无非厂商实现，或有解释 | `TILEINFER_AUDIT` JSON |
| D9 | 性能地板 | 无厂商基线：绝对地板 + 与参考实现一致性 | 绝对数、门槛来源 |
| D10 | 缺口闭环 | 静态 gap report 与实际工作量一致；provider 有 `delete_when` | gap report、provider 证据 |

> D2/D9 不使用 vllm-ascend A/B：厂商路径跑不了，没有可比基线。按
> [`acceptance.md`](../acceptance.md) §2.1 第三行处理。

---

## 3. 已知事实与未知

### 3.1 已确认（本地代码级）

| 事实 | 证据 |
|---|---|
| vLLM 0.23 注册了 `NemotronHForCausalLM` | `.venv_v3/.../vllm/model_executor/models/registry.py` |
| vLLM 0.23 注册了 `NemotronHMTPModel`，有 `nemotron_h_mtp.py` | 同上；`models/nemotron_h_mtp.py` |
| vllm-ascend 的 `patch_speculative_config.py` 会把 `nemotron_h` 映射为 `nemotron_h_mtp` | `.venv_v3/.../vllm_ascend/patch/platform/patch_speculative_config.py` |
| vllm-ascend 的 `models/__init__.py` **没有** Nemotron 注册 | 该文件只注册 DeepSeek V4 / Eagle3 等 |
| vllm-ascend 有 Mamba 相关 Triton 算子文件 | `.venv_v3/.../vllm_ascend/ops/triton/mamba/` |
| vllm-ascend **没有** `MambaMixer2`/`ssd_combined` 替换 | 全树搜索这些符号为空；Mamba 目录只有 conv/lightning/postprocess |
| vllm-ascend 补了 hybrid Mamba cache config 和 `causal_conv1d_update` | `patch_mamba_config.py`、`patch_triton.py` |
| 目标 config：52 层 = 23 Mamba + 23 MoE + 6 attention；上下文 262144；`num_nextn_predict_layers=1` | [`m5-nemotron-config.json`](m5-nemotron-config.json)（仅 config 快照） |
| vLLM 0.23 **直接加载**目标 config 会 `AttributeError: property 'layers_block_type' has no setter`；默认 pattern 也与目标不符 | `.venv_v3/bin/python` 实测 |
| 权重索引含 270 个 `mtp.*` key（`mtp.layers.0/.1`），MTP 权重确实存在 | [`m5-nemotron-mtp.json`](m5-nemotron-mtp.json)；索引 612207 B / 6513 keys |
| `normalize_config()` 转成 `hybrid_override_pattern`/`mtp_hybrid_override_pattern` 后，`NemotronHConfig` 可加载，层型/MTP 与目标一致 | `tools/nemotron/config_to_manifest.py` 测试 |

### 3.2 未知（必须在离线阶段查清，不能猜）

1. ~~checkpoint 是否带 MTP heads/权重~~ → **已确认有**：索引含 270 个 `mtp.*` key，`mtp.layers.0/.1`；Ascend MTP 执行/图仍未知。
2. vllm-ascend 的 Mamba2/混合 attention 是否能接住该模型，还是只在 vLLM 通用路径下能跑。
3. 图捕获对 Mamba state / 混合 KV 是否安全；需要上机计数，不能静态断言。
4. 参考实现路径：HF/transformers 在 CPU 上能否在可接受时间给出固定短样例 logits；
   若太慢，改用分层 oracle + 同输入复跑一致。
5. 权重是否已在本地/可获取；今晚不下载，先记录阻塞。
6. 目标 model 的 tokenizer/chat template 是否随权重提供；参考与 C1 必须用同一模板。

---

## 4. 离线工作（今晚必须完成，不占卡）

按“可并行、互不阻塞”拆成 O1–O7；任何一项都不依赖 NPU。

| # | 工作 | 产出 | 完成判据 |
|---|---|---|---|
| O1 | 静态审计：vLLM/vllm-ascend 的模型、MTP、Mamba、KV、图路径 | `docs/notes/m5-nemotron-static-gap.md` | 每条事实有文件路径/行号；未知项列明 |
| O2 | 模型声明草案：结构、能力清单、KV/state、benchmark 占位 | [`m5-nemotron-declaration.md`](m5-nemotron-declaration.md) + [`nemotron_3_5.py`](../../tileinfer/manifests/nemotron_3_5.py) + [`config_to_manifest.py`](../../tools/nemotron/config_to_manifest.py) | 结构来自真实 config；声明是 draft，benchmark top5 未填，**不进 registry** |
| O3 | 缺口报告（静态先验）：required gaps、候选 provider、`delete_when` | [`m5-nemotron-static-gap.md`](m5-nemotron-static-gap.md) + [`m5-nemotron-static-audit.json`](m5-nemotron-static-audit.json) | 与 O1 一致；不把“未验证”写成 gap=0 |
| O4 | provider 契约草案：coverage/graphable/parallel/evidence/delete_when | [`m5-nemotron-provider.md`](m5-nemotron-provider.md) | 每个 provider 对应一个真实缺口；无证据标 candidate |
| O5 | 参考实现方案：固定样例、tokenizer、logits/文本容差 | [`reference_probe.py`](../../tools/nemotron/reference_probe.py) 骨架 | 能在 CPU 上跑小样例；不依赖 NPU |
| O6 | 实机运行手册：命令、端口、时长、停止方式、回滚 | [`m5-nemotron-runbook.md`](m5-nemotron-runbook.md) | 部署前检查表完整；默认不执行 |
| O7 | CPU/fake 测试：声明校验、gap 序列化、plan 预算 | [`test_nemotron_offline.py`](../../tests/test_nemotron_offline.py) | `pytest` 全过；不 import NPU |

**不要为了等实机而停 O1–O7。** 硬件 gate 只是最后一步；离线产物先合并。

---

## 5. 实机 Gate（必须取得明确同意后才执行）

### 5.1 进入条件

- O1–O7 完成并 review；
- config/索引/tokenizer 已到位；真权重 C1 需要 `preflight_model.py` 返回 `ready_to_load=true`；
  dummy 只需 `ready_for_dummy=true`，但 30B dummy 仍占完整参数/state HBM；
- 已生成目标 plan 并设置 `TILEINFER_PLAN`，否则默认 Qwen plan 会在 config gate 失败；
- 用户明确同意：命令、端口（默认 8322）、预计时长、卡号/显存；
- `npu-smi info` 预检空闲，端口未被占用；只停止自己启动的 PID，不 kill 他人进程。

### 5.2 顺序（每步留证，失败即停）

1. **Dummy/加载**：`--load-format dummy --max-model-len 8192 --max-num-seqs 1 --enforce-eager`，
   确认 tokenizer/config/权重映射/架构能走到 forward。
2. **C1 真权重**：短请求完整输出；记录启动日志、HBM、图捕获。
3. **C2 参考一致**：固定样例 vs HF/transformers；不一致先查 tokenizer/权重映射，不调温度掩盖。
4. **C3 图**：FULL/PIECEWISE 捕获计数；稳态回退必须 0；记录 graph 张数。
5. **D4 MTP**：确认 `num_nextn_predict_layers>0` 且权重有 MTP；开投机，跑质量/图/回退；
   没有 MTP 就停在上报，不拿 ngram 代替。
6. **D5 长上下文**：8K → 16K → 32K → 128K；每档记录 `usage.prompt_tokens`、TTFT/TPOT、
   HBM 峰值、chunked prefill、回退；262K 仅在 config 支持时。
7. **D6 并发**：1/4/16/32，`max-num-seqs` 不小于客户端并发；记录吞吐/排队/回退。
8. **D7 冷启动**：移走编译缓存（不是删除），冷/热两臂条件一致；记录到 `/health`/首请求。
9. **D8 审计**：`TILEINFER_AUDIT` 落盘；热路径 torch 实现要有解释。
10. **D9/D10 收口**：绝对性能地板、gap report 与实际工作量对照、provider `delete_when`。

### 5.3 已执行 / 仍未做

已执行（2026-11-13，授权后）：

- 目标 shim 的 `EngineArgs.create_engine_config()`：通过，block 4096 / Mamba page 4231168；
- 合成 2 层 `M*` 小模型 dummy：厂商 Mamba2 SSD Triton 编译失败；随后 torch 替换使其启动成功，
  `/health` 200，completion 8 token；证据见 [`m5-nemotron-tiny-mamba2.json`](m5-nemotron-tiny-mamba2.json)、
  [`m5-nemotron-mamba2-ops.json`](m5-nemotron-mamba2-ops.json)；
- 合成 3 层 `M*E` 小模型（非门控 relu2 experts）：启动成功，completion 6 token；同一证据文件；
- 目标 dummy 已通过 config / 模型构造 / Mamba2 warmup，后在 vendor 共享专家
  `NemotronHMLP.gate_up_proj` 处失败，已补别名；一次因 HBM 不足（57.85 GiB free）在加载前退出；
- 2026-11-13 19:38 卡空闲后，目标 52 层 dummy 启动成功，加载 58.83 GB、KV 33.5 GiB，
  `/health` 200，1 token 和 161 token prompt 各完成 8 token；**这是 dummy 机械 smoke，
  不是真权重 C1/C2/C3**。证据 [`m5-nemotron-target-dummy.json`](m5-nemotron-target-dummy.json)；
- 卡上已有他人的 `VLLM::EngineCore` 或其他服务（约 103 GB HBM），未终止、未抢占。

仍未做：

- 目标 30B dummy / C1 / C2 / C3 / MTP / 长上下文 / 并发 / 冷启动 / 算子审计；
- 目标权重只有 5/14，不能做 C2；
- 不把合成小模型或静态推断写成目标结论。

---

## 6. 验收矩阵（本模型）

| 项 | 目标 | 备注 |
|---|---|---|
| 短上下文 | 4K/8K 正确输出 | C1/C2 |
| 图 | 捕获 >0，回退 0 | C3；Mamba/混合 attention 重点 |
| MTP | 有则过质量门；无则上报 | 不得用 ngram 冒充 |
| 长上下文 | 8K/16K/32K/128K | 262K 看 config |
| 并发 | 1/4/16/32 | 共享环境，逐档审批 |
| 冷启动 | 请求期零编译 | 编译缓存“移走” |
| 审计 | 热路径无非厂商实现 | O1/O2 |
| 性能 | 绝对地板 + 参考一致性 | 无 A/B |

---

## 7. 约束（违反即停）

1. **共享环境**：任何部署/重启/端口/占卡先说明并取得明确同意；不 kill 他人进程。
2. **实机需审批**：任何服务/端口/占卡先说明并取得明确同意；目标卡被他人占用时停，不 kill。
3. **不阻塞开发**：O1–O7 可并行；硬件未批不影响离线产物合并。
4. **冻结版本**：vLLM 0.23.0 / vllm-ascend 0.23.0-950；升级另过 V0 gate。
5. **不改上游**：只允许安装期注册、启动期换表、包装方法；接入点 ≤8。
6. **不静默回退**：回退计数、日志、断言；稳态必须 0。
7. **不虚报**：没有 MTP 权重不写 MTP；没有 262K config 不写 262K；没有证据不写“支持”。

---

## 8. 当前阻塞与下一步

| 阻塞 | 影响 | 今晚能否绕过 |
|---|---|---|
| 目标权重 5/14 | 目标 C1/C2 不能执行；dummy 可做但占完整 HBM | 补齐 9 个分片后 `ready_to_load=true` |
| 目标卡被他人 `VLLM::EngineCore` / 其他服务占用 | 本次已等到空闲并完成目标 dummy；共享卡随时可能再次被占 | 启动前先看 HBM；不 kill |
| 目标权重 5/14 | 只能做 dummy；真权重 C1/C2 不能做 | 补齐 9 个分片后 `ready_to_load=true` |
| Mamba2 SSD Triton 在合成模型上编译失败 | 已由 torch provider 绕过；合成 `M*`/`M*E` 启动成功；目标形状未测 | 目标 C1 验证数值/图；目标通过前不把 provider 登记为默认 |
| 目标 `nemotron_h` 不在 vLLM AutoConfig 注册表 | 无别名时 `ModelConfig` 直接失败 | bridge 已补别名；不要关 `TILEINFER_V3` |
| 参考实现未跑 | C2 top5 未产生 | 完整权重后 CPU HF probe |

**下一步**：目标 dummy 的 eager 与 graph 两条路径都已跑通；graph 捕获为 PIECEWISE+FULL、
目标图内存 2.68 GiB、日志有 `Replaying aclgraph`。但权重是 dummy，`decode_calls` 计数未推进，
不能据此证明真权重 C3/稳态 0 回退。等 9 个缺失分片补齐后做真权重 C1，再做 CPU HF 参考 C2；
MTP 也已用 dummy 跑通执行路径（`NemotronHMTPModel`、drafter 共享 embedding、7 drafts/0 accepted），
但 0 accepted 是随机权重预期，MTP 图捕获和真权重接受率未测。目标 C1/C2 通过前，`state.linear`
仍保持 gap，不把 provider 写成默认支持。
