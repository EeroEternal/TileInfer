# Nemotron 3.5 Lightning 30B-A3B：静态缺口先验（离线，不上卡）

> 状态：**静态代码审计 + 受限的 config/小模型机械探测**，不是目标 checkpoint 的 C1/C2 结论。
> 目标权重仍只有 5/14 分片，没有加载过目标 tensor；小模型探测只回答“同架构 Mamba2 路径能否编译”。
> 目的：在权重/实机到位前，把“确定的事实、确定的未知、候选 provider”分开。
> 任何 `unknown` 都不能写成 `gap=0`，也不能写成“不支持”。
>
> 原始输出：[`m5-nemotron-static-audit.json`](m5-nemotron-static-audit.json)（11 项检查，9 项确认；
> 失败项是“Ascend 没有 Nemotron 注册”和“没有 Mamba2 mixer/SSD patch”，这是待验证的缺口，不是结论）。
>
> 决策层快照：[`m5-nemotron-gap.json`](m5-nemotron-gap.json)。在“厂商已安装、静态覆盖”
> 的假设下，7 项 required capability 里只有 `state.linear` 是 gap；原因是厂商 GDN state
> 只 `covers={"mixer":"gdn"}`，不会覆盖 Mamba2。该快照不读权重、不碰设备。

---

## 1. 已确认事实（本地 0.23 栈）

| # | 事实 | 证据 |
|---|---|---|
| F1 | vLLM 0.23 注册 `NemotronHForCausalLM` | `.venv_v3/.../vllm/model_executor/models/registry.py:180` |
| F2 | vLLM 0.23 注册 `NemotronHMTPModel` | 同上 `:631`；`models/nemotron_h_mtp.py` |
| F3 | vLLM 有 `NemotronHConfig`，`model_type="nemotron_h"` | `.venv_v3/.../transformers_utils/configs/nemotron_h.py:27,144` |
| F4 | 默认混合层 pattern 含 Mamba/attention/MoE 标记 | 同上 `:154`（`M-M-M-M*-...`；`*`=attention，`E`=MoE，`M`=Mamba） |
| F5 | 模型 forward 含 `MambaMixer2`、`FusedMoE`、`Attention` | `.venv_v3/.../models/nemotron_h.py:378,217,468` |
| F6 | MTP 由 `num_nextn_predict_layers` 触发，vLLM 有 `NemotronHMTPModel` | `vllm/config/speculative.py`；`nemotron_h_mtp.py` |
| F7 | vllm-ascend 的 spec patch 会把 `nemotron_h` 改写成 `nemotron_h_mtp` | `.venv_v3/.../vllm_ascend/patch/platform/patch_speculative_config.py:82-88` |
| F8 | vllm-ascend 的 `models/__init__.py` **没有** Nemotron 注册 | 该文件只注册 DeepSeek V4 / Eagle3 等 |
| F9 | vllm-ascend 有 Mamba 相关 Triton 算子目录 | `.venv_v3/.../vllm_ascend/ops/triton/mamba/`（`causal_conv1d.py`、`lightning_attn.py`、`postprocess.py`） |
| F10 | vllm-ascend 补了 hybrid attention/Mamba 的 cache config，并替换了 `causal_conv1d_update` | `patch/platform/patch_mamba_config.py:149`；`patch/worker/patch_triton.py:10` |
| F11 | **没有**找到 Ascend 对 `MambaMixer2`/`ssd_combined`/`mamba_chunk_scan` 的替换 | `vllm_ascend` 全树搜索这些符号为空；Mamba 目录只有 conv/lightning/postprocess，不含 SSD chunk-scan |
| F12 | 目标 checkpoint 的 config 用 `layers_block_type`，没有 `hybrid_override_pattern`；52 层 = 23 Mamba + 23 MoE + 6 attention；`mtp_layers_block_type=["attention","moe"]`；上下文 262144 | [`m5-nemotron-config.json`](m5-nemotron-config.json)（仅 config 快照，未下载权重；来源见 [`m5-nemotron-config.md`](m5-nemotron-config.md)） |
| F13 | vLLM 0.23 的 `NemotronHConfig` 直接加载该 config 会抛 `AttributeError: property 'layers_block_type' has no setter`；即使绕过，默认 `hybrid_override_pattern` 是 24 M / 24 MLP / 4 attention、无 MoE，与目标 23/23/6 不一致 | `.venv_v3/bin/python` 实测；见 G0 |
| F14 | 用 `normalize_config()` 把 `layers_block_type`/`mtp_layers_block_type` 转成 `hybrid_override_pattern`/`mtp_hybrid_override_pattern` 后，`NemotronHConfig` 能加载，层型和 MTP pattern 与目标一致 | `tools/nemotron/config_to_manifest.py`；测试 `test_normalized_config_loads_in_vllm_without_npu` |
| F15 | 权重索引有 270 个 `mtp.*` key（`mtp.layers.0` attention / `mtp.layers.1` moe），backbone 最大层号 51；MTP 权重确实存在 | [`m5-nemotron-mtp.json`](m5-nemotron-mtp.json)；索引 612207 B / 6513 keys |
| F16 | 权重名与声明的 52 层完全对齐：23 Mamba / 23 MoE / 6 attention；每个 MoE 层 0..127 experts 齐全，shared expert 宽度 3712，MTP 为 attention+moe；`mapping_ok=true` | [`m5-nemotron-weight-map.json`](m5-nemotron-weight-map.json)；只读索引，不读 tensor |
| F17 | 已在盘上的 5 个分片头部全部有效：1968 个 tensor header，文件大小与 header offset 一致，无重复 key、无 index/header 漂移；但只覆盖 5/14 分片，`complete=false` | [`m5-nemotron-shard-headers.json`](m5-nemotron-shard-headers.json)；只读 header，不读 tensor payload |
| F18 | 用真实 checkpoint 生成的 shim config 能被 vLLM 0.23 `NemotronHConfig` 加载：52 层、`MEMEM*...`、MTP `*E`、Mamba 64/64/128/8/4/2、MoE 128/top-6/1856/shared 1/3712 | `.venv_v3/bin/python` 实测；未构造模型、未上卡 |
| F19 | 已在盘上的 5 个分片里，1950 个 tensor 的 header shape 与 config 推导值一致；0 处不匹配；18 个非 mixer/norm 类 key 未做形状检查 | [`m5-nemotron-shapes.json`](m5-nemotron-shapes.json)；只读 header，不读 tensor payload |
| F20 | vLLM 0.23.0 的 `_CONFIG_REGISTRY` 没有 `nemotron_h` 项；Transformers 4.57 下 `ModelConfig` 会在模型代码之前报 “does not recognize this architecture”。TileInfer 的 bridge 在 `ModelConfig` 之前注册 vLLM 自带的 `NemotronHConfig` 后，该错误消失 | `tileinfer/vllm/config_compat.py`；`tileinfer/bridge/install.py`；`.venv_v3` 实测 |
| F21 | 目标 shim 的 `EngineArgs.create_engine_config()` 通过：`NemotronHForCausalLM`、attention block size 4096、Mamba page padded 4231168 B、padding 0.87%；没有加载权重、没有起服务 | [`m5-nemotron-config-load.json`](m5-nemotron-config-load.json) |
| F22 | 合成同架构小模型（`M*`、2 层、dummy）最初在 “Warming up Mamba2 SSD Triton kernels” 阶段失败：`ssd_chunk_scan.py:238` 的 `CompilationError` / `AttributeError('NoneType' object has no attribute 'type')`，Ascend 后端报 `Currently ptr type from different source not supported`。该失败是待实现缺口，不是结论；随后 `tileinfer/vllm/ops/mamba2.py` 的 torch 替换和 op table 绑定使同一 `M*` 合成模型启动成功、`/health` 200、completion 返回 8 token | [`m5-nemotron-tiny-mamba2.json`](m5-nemotron-tiny-mamba2.json)；[`m5-nemotron-mamba2-ops.json`](m5-nemotron-mamba2-ops.json) |
| F23 | 目标 30B 的 52 层几何 + dummy 权重也能启动：加载 58.83 GB、KV 33.5 GiB、`/health` 200；1 token 和 161 token prompt 各完成 8 token；无 ACL/NotImplemented/回退错误。因 `--enforce-eager` 且权重为 dummy，这不是 C1 正确性、C2 或 C3 证据 | [`m5-nemotron-target-dummy.json`](m5-nemotron-target-dummy.json)；`/tmp/m5-target-dummy6.log` |
| F24 | 同一目标 dummy 去掉 `--enforce-eager` 后，图捕获通过：PIECEWISE 1 + FULL 1，图内存 2.68 GiB，日志出现 `Replaying aclgraph`；这依赖非门控 MoE 的 grouped-matmul 路径和 decode state 的 tensor 索引化。仍不是完整 C3：权重 dummy，`decode_calls` 计数未推进 | [`m5-nemotron-target-graph.json`](m5-nemotron-target-graph.json)；`/tmp/m5-target-graph.log` |
| F25 | 目标 dummy 的 MTP 执行路径可用：解析为 `NemotronHMTPModel`，加载 drafter 并共享 target embedding；161-token prompt 完成 8 token，metrics 为 7 drafts / 7 draft tokens / 0 accepted。0 accepted 是随机权重的预期结果；MTP 图捕获与真权重接受率未测 | [`m5-nemotron-target-mtp.json`](m5-nemotron-target-mtp.json)；`/tmp/m5-target-mtp.log` |
| F26 | 目标 dummy 以 `max-num-seqs 4` 并发跑 4 个请求：164/204/244/284 prompt token 各完成 8 token，4/4 `finish_reason=length`，约 5.1–5.2 s；`request_success_total{length}=4`。这是 eager 机械并发，不是真权重正确性、图并发或吞吐结论 | [`m5-nemotron-target-concurrency.json`](m5-nemotron-target-concurrency.json)；`/tmp/m5-target-conc.log` |

---

## 2. 确定的未知（不能猜）

| # | 未知 | 为什么关键 | 离线怎么查 |
|---|---|---|---|
| U1 | 目标 checkpoint 的 `config.json` 字段 | 层数/pattern/MTP/上下文决定声明与验收 | 权重到位后读 config；今晚网络不可达，先留空 |
| U2 | checkpoint 是否带 MTP heads/权重 | 没有就不能承诺 MTP | 查 config `num_nextn_predict_layers` + 权重 key |
| U3 | vllm-ascend 是否能接住 `NemotronHForCausalLM` | vLLM 有模型 ≠ Ascend 能跑；已确认 config/page 层能过，Mamba2 kernel 层在合成模型上失败；目标形状仍未测 | 读 Ascend attention/Mamba/状态路径；目标 dummy/真权重判 |
| U4 | Mamba2/混合 attention 的 Ascend 图捕获 | 不能图捕获则 C3 失败 | 上机捕获计数；静态只能列风险 |
| U5 | 混合 KV/Mamba state 的分配与 page 约束 | 错则 OOM/错数 | 读 `MambaSpec`/Ascend platform；上机验 |
| U6 | 权重名映射（`mixer.{k,v}_proj` 等） | 错则加载失败/静默错 | 读 `weight_utils.py` 与 Ascend loader；上机验 |
| U7 | 参考实现可执行性 | 无厂商基线，C2 只能对 HF/transformers | 权重到位后 CPU 小样例；太慢则分层 oracle |
| U8 | ~~MTP 权重 key 是否真的存在~~ → **已确认有** 270 个 `mtp.*` key | config 声明不等于权重有；现已查索引确认 | 上机验 Ascend 的 MTP 注册/执行/图；失败按 D4 上报 |

---

## 3. 缺口先验（静态，不是最终 gap report）

> 结论：**`servable=False`（先验）**。不是说“永远不能”，而是当前没有证据它能在这套
> Ascend 栈上跑；必须上机或补 provider 后才能改。G0 是本轮新确认的第一道硬门：
> 不先做 config shim，模型在加载 config 阶段就会失败，轮不到算子。

| # | 能力/缺口 | 性质 | 候选 provider | `delete_when` |
|---|---|---|---|---|
| G0 | **目标 config 与 vLLM 0.23 schema 不兼容** | **已实测**：raw config 直接 `AttributeError`；默认 pattern 也不是目标 pattern | config shim（`normalize_config`，不改上游） | vLLM/上游接受 `layers_block_type` 或升级 |
| G1 | NemotronH 模型在 Ascend 的注册/加载 | 未知（可能 vLLM 通用路径可接） | model table / 模型包装 | vllm-ascend 原生注册 NemotronH |
| G2 | Mamba2 mixer 的 SSD/chunk-scan 执行路径 | 厂商 Triton 在 `ssd_chunk_scan.py:238` 编译失败；已用 torch 实现 `mamba_chunk_scan_combined_varlen` / `selective_state_update`，在合成 `M*` 和 `M*E` 模型上启动并完成请求；目标 30B 形状未测，`state.linear` provider 尚未登记 | `state.linear` + `Mamba2Contract` provider | 目标 C1/C2 通过后才把 provider 写进默认表；否则保持 gap |
| G3 | 混合 attention（full + Mamba）的 KV/state 布局 | 未知 | state/KV provider 或 bridge | 厂商支持混合 KV/MambaSpec |
| G4 | MTP（`NemotronHMTPModel`）在 Ascend 的注册/执行 | 权重已确认（270 个 `mtp.*` key）；Ascend 注册/执行/图仍未知 | MTP provider / model table | 厂商注册并验证 NemotronHMTP |
| G5 | 图捕获（Mamba state + 混合 attention） | 未知 | graph hook（不是 provider） | 厂商图路径覆盖 |
| G6 | 权重映射/加载 | 未知 | loader patch/provider | 厂商 loader 覆盖 |
| G7 | 参考实现（HF/transformers） | 阻塞 | reference harness（不进默认路径） | 不需要删；参考实现常驻 |

**禁止**：把 G1–G7 的 `unknown` 写成“厂商不支持”或“provider 已补”；只有上机/代码证据才能改状态。

---

## 4. 对任务书的影响

- O2 模型声明不能凭默认 config 填死；必须先拿到目标 checkpoint 的 `config.json`。
- O3 缺口报告先按上表生成“静态先验”，字段 `servable=False`、`notes` 标 `static-only`。
- O4 provider 只能写**候选**；`evidence` 留空/标 `candidate`，不能进默认路径。
- O5 参考实现先搭 HF/transformers 脚本骨架；权重未到不跑。
- O6 运行手册必须包含“先 dummy、再真权重、先 eager、再图”的顺序。

---

## 5. 复现（只读，不占卡）

```bash
# 只读源码，不 import vLLM、不起服务
grep -n "NemotronHForCausalLM\|NemotronHMTPModel" \
  .venv_v3/lib/python3.12/site-packages/vllm/model_executor/models/registry.py
grep -n "nemotron_h" \
  .venv_v3/lib/python3.12/site-packages/vllm_ascend/patch/platform/patch_speculative_config.py
sed -n '1,20p' .venv_v3/lib/python3.12/site-packages/vllm_ascend/models/__init__.py
ls .venv_v3/lib/python3.12/site-packages/vllm_ascend/ops/triton/mamba/
```

---

## 6. 2026-11-13 受限探测（非目标 C1）

用户授权使用 NPU 后，先检查到卡上已有他人的 `VLLM::EngineCore`（约 103 GB HBM），未终止、
未抢占；因此没有在目标 30B 上起服务。改为：

1. 只用 `EngineArgs.create_engine_config()` 验证目标 shim 的 config/page 层；结果见 F21。
2. 用 2 层、`M*`、hidden 256 的合成同架构模型做 dummy 机械 smoke，结果见 F22；服务未起来，
   自己的 PID 已全部停止。
3. 目标 checkpoint 仍为 5/14 分片，`preflight_model.py` 仍 `ready_to_load=false`；没有读目标
   tensor payload，也没有产生目标 C1/C2/C3/C4 证据。

因此本次把 G2 从“静态缺候选”升级为“同架构合成路径实测失败”，但**不能**写成
“目标 30B 在 Ascend 上确认不支持”；要得到目标结论，仍需完整权重 + 空闲卡 + 目标 dummy。
