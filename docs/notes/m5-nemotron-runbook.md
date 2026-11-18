# Nemotron 实机运行手册（O6，默认不执行）

> **默认状态：不执行。** 任何命令在运行前都必须先向用户说明命令、端口、预计时长、
> 卡号/显存，并取得**明确同意**。`npu-smi` 也只在获准后运行。
>
> 共享环境：只停止自己启动的 PID；不要 `pkill`/`killall`；不要动他人的进程、端口、缓存。

> **2026-11-16 实机结果**：单卡 TP=1、真权重、`servable=false` bring-up 已跑通；
> C2 仅 token/text 通过（logprob top-5 未通过）；C3 短稳态 31/31 命中、0 回退（metrics 文件默认只写捕获快照），
> MTP 无损性失败；TP=2/4/8 后来用全新 cache 重试均通过，早先 TP=8 的 HCCL rootinfo 报错是共享环境瞬时状态。
> 证据与命令见 [`m5-nemotron-validation-20261116.md`](m5-nemotron-validation-20261116.md)。

> **2026-11-17 更新**：当前树已验证 `.venv_lfm3`（transformers 5.17.0）可跑 C1/C3；
> `nemotron_h` config 别名由 plan 的 `config_impl` 驱动，不再硬编码在 `install.py`。
> 文中 `.venv_v3` 命令是历史机器记录，当前 checkout 优先用 `.venv_lfm3`。

> **2026-11-18 更新**：单卡 TP=1、真权重、`servable=false`；图模式 greedy MTP 已逐 token
> 等价（512 token、4 路 32 token、32K/130K prompt 8 token），C3 图门全部命中/0 回退；
> C2 教师强制 top-1/top-5 rank 通过（dt floor 修复 capital step31），绝对 logprob 差只作诊断。
> **本轮交付只要求 TP=1；TP=8 out-of-scope**，历史 TP2/4/8 结果保留但不作为本轮阻塞。
> 证据与命令见 [`m5-delivery-gates-20261117.md`](m5-delivery-gates-20261117.md) 与
> [`m5-nemotron-validation-20261116.md`](m5-nemotron-validation-20261116.md)。
> **不要默认开 MTP，也不要对外说“已支持”或“TP8 已支持”。**

---

## 0. 部署前检查表（全部满足才申请）

- [ ] O1–O5 离线产物已 review；
- [ ] `config.json` 已确认：`architectures/model_type/num_hidden_layers/
      layers_block_type/mtp_layers_block_type/num_nextn_predict_layers/max_position_embeddings`；
- [ ] 已生成 vLLM 0.23 兼容的 config shim（见 §1.1），且 `NemotronHConfig` 能加载；
- [ ] 权重路径存在，磁盘/内存足够；**2026-11-15 已 14/14，`ready_to_load=true`**；但宿主无 NPU 设备，仍未做真权重 C1；
- [ ] 若只做 dummy：`preflight_model.py` 的 `ready_for_dummy=true`（config/tokenizer 齐全）；
- [ ] 30B dummy 仍会分配完整参数/state 的 HBM，不是“小模型 smoke”；
- [ ] MTP 是否存在已确认（没有就不开 MTP，不拿 ngram 冒充）；
- [ ] `npu-smi info` 显示目标卡空闲（获准后执行）；
- [ ] 端口 `8322` 未被占用（`ss -ltnp`）；
- [ ] `npu-smi info` 确认没有他人的 `VLLM::EngineCore` 占着目标卡；**绝不 kill 他人进程**；
- [ ] 已生成目标 plan（§1.2），否则 bridge 会用默认 Qwen plan 并在 config gate 失败；
- [ ] 用户同意：命令、端口、预计时长、卡号；
- [ ] 记录开始时间、PID、命令原文。

---

## 1. 环境与变量

```bash
cd /home/userlist/zxs/TileInfer
source scripts/env-ascend102.sh

MODEL=/home/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16
SHIM=/tmp/m5-nemotron-shim-$(date +%s)
PLAN=/tmp/m5-nemotron-plan.json
PORT=8322
LOG=/tmp/m5-nemotron-serve.log
METRICS=/tmp/m5-nemotron-metrics.json
AUDIT=/tmp/m5-nemotron-opaudit.json
```

> 如果权重不在本地，先停下来确认获取方式；**不要**在本手册里自动下载。

### 1.0 文件完整性预检（不读 tensor）

```bash
python tools/nemotron/preflight_model.py --model "$MODEL"
```

`ready_to_load=false` 时不要做真权重加载；它只查文件是否存在/非空，不会碰 NPU。
`ready_for_dummy=true` 表示 config/tokenizer 已齐，可以做 §2 的随机权重机械 smoke；
它仍然**不能**替代 `ready_to_load`，也不能产生 C1/C2/C3/C4 证据。
2026-11-15 的现状是 14/14 分片、`ready_to_load=true`；宿主重启后无 `/dev/davinci*`，
真权重服务仍无法启动，见 [`m5-nemotron-blocker.md`](m5-nemotron-blocker.md)。

### 1.1 config shim（vLLM 0.23 不认 `layers_block_type`）

目标 config 直接喂给 vLLM 0.23 会在加载 config 时抛
`AttributeError: property 'layers_block_type' has no setter`；必须先用
`normalize_config()` 转成 `hybrid_override_pattern`，再用一个只替换 config 的
shim 目录启动（权重文件用 symlink，不复制、不修改原目录）：

```bash
python tools/nemotron/make_shim.py --model "$MODEL" --shim "$SHIM"
```

它只替换 `config.json`，其余文件全部软链到原目录；不会复制/修改权重。

验证：

```bash
.venv_v3/bin/python - <<PY
import json
from vllm.transformers_utils.configs.nemotron_h import NemotronHConfig
c = NemotronHConfig(**json.load(open("$SHIM/config.json")))
print(c.hybrid_override_pattern)
print(c.mtp_hybrid_override_pattern)
PY
```

期望：52 字符 pattern，含 23 个 `M`、23 个 `E`、6 个 `*`；MTP pattern 为 `*E`。
**这一步只改 config 的 schema 表达，不改权重，不改上游源码。**

### 1.2 vLLM 0.23 的 `nemotron_h` AutoConfig 别名

vLLM 0.23.0 自带 `NemotronHConfig`，但它的 `_CONFIG_REGISTRY` 没有 `nemotron_h` 项；
Transformers 4.57 下 `ModelConfig` 会在模型代码之前直接失败。P0 之后，这个别名不再硬编码在
`install.py`，而是由 manifest 的
`config_impl="tileinfer.bridge.nemotron_config:register_nemotron_h_alias"` 驱动：bridge 在
`ModelConfig` 之前导入 `tileinfer/bridge/nemotron_config.py` 并执行 actor，不修改上游源码。
如果 bridge 被 `TILEINFER_V3=0` 关闭，这个别名不会注册；因此目标服务不要关 bridge。

### 1.3 目标 plan（不注册 draft）

`NEMOTRON_3_5_DRAFT` 仍在 benchmark pending，**不在** `MANIFESTS`；但实机需要目标 plan 才能
过 config gate。用离线工具生成临时 plan：

```bash
# 本机可用 .venv_lfm3（transformers 5.17）；历史机器用 .venv_v3
.venv_lfm3/bin/python tools/nemotron/build_plan.py --allow-candidates --out "$PLAN"
```

输出会明确 `servable=false`、`gaps=["state.linear"]`；这是 bring-up 用的 plan，不是支持声明。
`--allow-candidates` 只是把已声明的 v3 候选（Mamba2 + 非门控 MoE）选进 plan，plan 仍然
`servable=false`；启动时还要
显式设 `TILEINFER_ALLOW_UNSERVABLE=1`，bridge 才会安装候选，否则会拒绝并保留厂商路径。同时设
`TILEINFER_PLAN="$PLAN"`，否则 bridge 会按默认 Qwen 声明生成 plan，config gate 会报
`TileInfer plan/model mismatch`。

---

## 2. Dummy 加载（先判机械链路，不是验收）

> **只用于判断“config/model 构造/服务入口”是否通，不能作为 C1/C2/C3/C4 证据。**
> 随机权重没有语义，不能拿它的输出算 top5、loss 或质量；它的 timing 也不进性能结论。
> 仍然需要完整 tokenizer 和显式实机授权。

```bash
TILEINFER_PLAN="$PLAN" TILEINFER_AUDIT="$AUDIT" \
.venv_v3/bin/python -m vllm.entrypoints.openai.api_server \
  --model "$SHIM" \
  --served-model-name nemotron \
  --load-format dummy \
  --max-model-len 8192 \
  --max-num-seqs 1 \
  --enforce-eager \
  --gpu-memory-utilization 0.82 \
  --host 127.0.0.1 --port "$PORT" >"$LOG" 2>&1 &
echo $! > /tmp/m5-nemotron.pid
```

`vllm` 脚本的 shebang 在这个 venv 里指向不存在的 `.venv/bin/python3.12`，所以手册用
`.venv_v3/bin/python -m ...`，不要直接调用 `vllm`。

判据：

- `/health` 200；
- 日志出现模型类/架构识别、KV/Mamba state 分配；
- 没有 `NotImplementedError` / 缺算子；
- `TILEINFER_AUDIT` 能落盘（若无 TileInfer 注入，至少确认厂商路径）；
- 日志里出现 `TileInfer ... plan=... model=Nemotron-3.5...`，而不是默认 Qwen plan。

结果只记“机械链路通过/失败 + 首个错误”，不要写成“模型可用”。

### 2.1 合成小模型结果与 Torch 算子替换（2026-11-13）

目标卡多次被他人 `VLLM::EngineCore` / 其他服务占用，没有抢占、没有 kill 他人进程。为了把
“厂商 Mamba2 SSD 编译失败”从阻塞变成可执行缺口，我们补了 torch 参考算子并先在合成同架构
模型上验证：

- **厂商失败证据**（保留）：2 层 `M*`、hidden 256，进入
  `Warming up Mamba2 SSD Triton kernels` 后，`ssd_chunk_scan.py:238` 的
  `CompilationError` / `AttributeError('NoneType' object has no attribute 'type')`，Ascend 后端
  报 `Currently ptr type from different source not supported`；服务未监听 `/health`。证据
  [`m5-nemotron-tiny-mamba2.json`](m5-nemotron-tiny-mamba2.json)。
- **替换后**：`tileinfer/providers/nemotron/mamba2.py` 实现 `mamba_chunk_scan_combined_varlen` 和
  `selective_state_update`；op table 在 `ssd_combined`、`mamba_mixer2`、`ssu_dispatch` 三处
  绑定。2 层 `M*` 合成模型启动成功，`/health` 200，chat completion 返回 8 token
  （`chatcmpl-90bbd5d4088b9a52`）。
- **非门控 MoE**：目标专家 MLP 是 `relu2_no_mul`、`w1=[E,inter,hidden]`，不是 SwiGLU。扩展
  `fused_experts` 后，3 层 `M*E`、4 experts、top-2 的合成模型启动成功，completion 返回 6 token
  （`cmpl-b01b37e4b878290f`）。同时绕过 vLLM `FusedMoE` 的
  `is_act_and_mul=False` 平台 guard，只在该构造调用期间生效；NemotronHMLP 的共享专家
  `gate_up_proj` 做 `up_proj` 别名。
- **目标 dummy**：`m5-target-dummy4.log` 已通过 config、模型构造、Mamba2 warmup，随后在
  vendor 共享专家 `NemotronHMLP.gate_up_proj` 处失败；该别名已补。`m5-target-dummy5.log` 在
  加载前因 HBM 只剩 57.85 GiB、请求 0.75×123 GiB 而退出。2026-11-13 19:38 卡空闲后，
  `m5-target-dummy6.log` 以目标 52 层几何 + dummy 权重启动成功：加载 58.83 GB，KV 33.5 GiB，
  `--enforce-eager`，`/health` 200；1 token prompt 和 161 token prompt 各完成 8 token
  （`cmpl-b3d297350912e262`、`cmpl-88a043485639fdba`），没有 ACL/NotImplemented/回退错误。
  这是**目标形状机械 smoke**，不是真权重 C1/C2/C3；服务已停止、端口已释放。证据：
  [`m5-nemotron-target-dummy.json`](m5-nemotron-target-dummy.json)。

证据：[`m5-nemotron-mamba2-ops.json`](m5-nemotron-mamba2-ops.json)。合成形状不等于目标 30B
形状；目标 C1/C2 仍需完整权重 + 空闲卡。

停止：

```bash
kill -TERM "$(cat /tmp/m5-nemotron.pid)"
```

---

## 3. C1：真权重、短请求

```bash
TILEINFER_PLAN="$PLAN" TILEINFER_AUDIT="$AUDIT" \
.venv_v3/bin/python -m vllm.entrypoints.openai.api_server \
  --model "$SHIM" \
  --served-model-name nemotron \
  --max-model-len 8192 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.82 \
  --host 127.0.0.1 --port "$PORT" >"$LOG" 2>&1 &
```

请求：

```bash
curl -sS "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"nemotron","messages":[{"role":"user","content":"Say hello in one sentence."}],"max_tokens":32,"temperature":0}'
```

留证：完整响应、`usage`、启动日志、HBM、图计数。

---

## 4. C2：参考实现一致

参考探针已在本机跑通：CPU + HF transformers 5.12.1（`NemotronHForCausalLM`，naive Mamba 路径，
不加载 torch_npu、不占卡）。默认的 `.venv_v3` 里 transformers 4.57 没有 NemotronH，探针会优先把
`TILEINFER_HF_SITE`（默认 `/home/dlb/sglang/.venv_npu/lib/python3.12/site-packages`）放到
`sys.path` 前面，并用本 venv 的 CPU torch。若该路径不存在，先设置可用的 5.x site-packages。

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 .venv_native213/bin/python tools/nemotron/reference_probe.py \
  --model "$MODEL" \
  --prompt "Say hello in one sentence." \
  --max-tokens 32 \
  --out /tmp/m5-nemotron-ref.json
```

C2 必须用**原始 completion**（不是 chat template）请求服务，并要求 logprobs：

```bash
curl -sS "http://127.0.0.1:$PORT/v1/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"nemotron","prompt":"Say hello in one sentence.","max_tokens":32,"temperature":0,"logprobs":5}' \
  -o /tmp/m5-nemotron-c2-service.json
.venv_native213/bin/python benchmarks/compare_reference.py \
  /tmp/m5-nemotron-ref.json /tmp/m5-nemotron-c2-service.json --atol 1e-4 \
  --out /tmp/m5-nemotron-c2-diff.json
```

`compare_reference.py` 比较生成文本和首个生成 token 的 top-5 logprob；top-k 边界打平时按
`--atol` 容差判定并在 JSON 里单独列出 strict/tie-aware 结果。C2 未通过前不得把候选
`tileinfer.mamba2_state` 登记为默认，也不得把 plan 改成 `servable=true`。

v3 bridge 的安装条件同样是硬性的：plan 必须有 `impl_table` 接入点、非厂商 selection，且
`servable=true`；当前目标的 plan 是 `servable=false`，bridge 会拒绝安装 Mamba2 候选并保留
厂商路径，而不是"先装上再说"。只有显式 `TILEINFER_ALLOW_UNSERVABLE=1` 的 bring-up 运行
可以覆盖，且日志会标注。

> 参考是本机 HF/CPU，不是厂商/vLLM 基线；不要用另一个 vLLM 场景的绝对值当参考。

---

## 4.5 编译缓存（2026-11-16 实测）

`max-num-seqs=4` 在**默认** `VLLM_CACHE_ROOT` 下、跑过 fp32/失败编译后可能报
`KeyError: 'AscendRMSNorm'`（profile_run 的 AOT 编译）。这不是模型/provider 回退；
给本次进程设置全新的 `VLLM_CACHE_ROOT=/tmp/m5-vllm-cache-$(date +%s)` 后，同一代码可正常
捕获 6 张图并完成 4 路请求。**不要删除共享 cache**，也不要 `pkill` 其他服务。

## 5. C3：图捕获

> C3 必须开图跑；eager 输出不能作为 C3 证据。**2026-11-16 真权重已捕获 2 张图**
> （`max-num-seqs=4` 时 6 张），并在 32-token 请求中观测到 31 次 decode/31 次图命中/0 回退。
> 默认 `instrumentation.dump_every_decode_steps=500` 时 metrics 只写捕获快照
> （`decode_calls` 仍可能显示 0），不能据此判 C3 失败；把 plan 的该值设为 1 即可在请求后
> 读到实时计数。本轮 clean run 已得到 32-token 31/31/0、8001-prompt 7/7/0、
> 130001-prompt 3/3/0、4 路并发 15/15/0，均 0 回退。
>
> 合成 `M*E` 的首次图捕获实测（端口 18084，无 `--enforce-eager`）在 PIECEWISE 0/1 阶段失败：
> `aclnnNonzero` 错误码 107027，位置 `tileinfer/providers/moe.py`（原 v2 位置 `legacy/v2/tileinfer/vllm/ops/moe.py:1010`，已迁移到 v3 并归档 v2）
> （`(local < 0).nonzero(...)`）。这说明**非门控 MoE 的数据依赖路径先于 Mamba2 断图**。
>
> 随后把非门控 MoE 改成 `npu_grouped_matmul`（按 local expert 排序，不再 `unique/nonzero/tolist`），
> 并把 `selective_state_update` 的 state 索引改成张量 index_select/index_copy，不再 `.to("cpu")`。
> 合成 `M*E` 与目标 30B dummy 都能捕获 **PIECEWISE 1 + FULL 1**，日志出现 `Replaying aclgraph`；
> 目标 dummy 图内存 2.68 GiB。证据：
> [`m5-nemotron-c3-attempt.json`](m5-nemotron-c3-attempt.json)、
> [`m5-nemotron-target-graph.json`](m5-nemotron-target-graph.json)。
>
> 但这仍不是完整 C3：权重是 dummy。当次 `decode_calls=0` 暴露了计数缺口；现在非 `NONE` replay
> 即使厂商传 `uniform_decode=False` 也会计数，单测见 `tests/test_bridge_metrics.py`。修复后尚未在
> NPU 上重跑，仍不能用它证明稳态 `fallbacks=0`；真权重 C1/C2 和长上下文/并发/MTP 的图行为仍未测。
>
> MTP 单独跑过一遍（仍为 dummy，`--enforce-eager`）：`NemotronHMTPModel` 被解析、drafter 加载并
> 共享 target embedding，161-token prompt 的请求完成 8 token；metrics 有 7 次 draft、7 个 draft
> token、0 个 accepted。0 accepted 是随机 dummy 权重的预期结果，不是质量结论；MTP 图捕获和真权重
> 接受率未测。证据：[`m5-nemotron-target-mtp.json`](m5-nemotron-target-mtp.json)。
>
> 目标 dummy 还跑过 8K 上下文（6001 prompt token，eager，4 s）和 `max-num-seqs 4` 的 4 路并发
> （164–284 prompt token，4/4 完成，约 5.1–5.2 s）。两者都是机械通路，不是真权重正确性、图并发
> 或 SLA 结论。证据：[`m5-nemotron-target-long.json`](m5-nemotron-target-long.json)、
> [`m5-nemotron-target-concurrency.json`](m5-nemotron-target-concurrency.json)。
>
> 离线算子审计见 [`m5-nemotron-op-audit.json`](m5-nemotron-op-audit.json)；Mamba2 还加了目标几何
> 的 CPU 参考一致性测试（64 头 × 64 head_dim × 128 state × 8 组，非 chunk 对齐的 varlen：
> 37/53 token），这仍不是真权重 C2。
>
> `tileinfer/providers/nemotron/__init__.py` 已声明 `tileinfer.mamba2_state` 候选（`covers={"mixer":
> "mamba2"}`、graphable、证据编号 `m5-nemotron-mamba2-ops`），但**没有**注册进决策表；gap 报告会
> 列出这个候选并注明“not default until target C1/C2”。这是“已实现、待真权重证据”，不是阻塞项。

保持服务运行，观察日志/指标：

- `captured_graphs > 0`；
- 稳态 `fallbacks = 0`；
- 记录 FULL/PIECEWISE 张数、bucket、是否因 Mamba/混合 attention 断图。

若捕获失败：先试 `--enforce-eager` 确认是图问题，不要直接改模型。

---

## 6. D4：MTP（仅当 checkpoint 有 MTP）

先查权重索引，不靠 config 声明：

```bash
python tools/nemotron/mtp_probe.py \
  --index "$MODEL/model.safetensors.index.json" --num-layers 52
```

- `mtp_weights_found=false` → 记录“无 MTP”，跳到 D5，**不要**用 ngram 替代；
- `true` → 再启动下面的 MTP 服务。当前目标索引已见 270 个 `mtp.*` key（`mtp.layers.0/.1`），
  所以应走 `true` 分支；见 [`m5-nemotron-mtp.json`](m5-nemotron-mtp.json)。

```bash
vllm serve "$SHIM" \
  --served-model-name nemotron \
  --max-model-len 8192 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.82 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
  --port "$PORT" >"$LOG" 2>&1 &
```

判据：

- 服务能起，MTP 配置被识别；
- 质量门（任务正确/确定性）过；
- 图捕获/回退留证；
- 若 `num_nextn_predict_layers=0` 或权重无 MTP：**停止**，记录“无 MTP”，不用 ngram 代替。

**2026-11-16 真权重结果：MTP 质量门失败。** 隔离过：跳过 `selective_state_update`、跳过
`causal_conv1d_update`、`--no-async-scheduling` 三种情况输出都不变，所以不是这两个补丁单独导致；
问题在 MTP draft/验证链。服务能起（`NemotronHMTPModel`、drafter 加载，
drafts=14 / accepted=8），但同 prompt、greedy 下输出与非 MTP 不同：
MTP ` Paris.</think>Paris is the capital of France.`，非 MTP
` Paris.  \nThe capital of Germany is Berlin.  \nThe capital of`。
speculative decoding 应无损；在查清验证链之前 MTP 输出不能作为正确性证据，默认关闭。

---

## 7. D5：长上下文

逐档，不要一次跳满：

| 档 | `--max-model-len` | prompt tokens | 备注 |
|---|---:|---:|---|
| 8K | 8192 | ~8K | 首档 |
| 16K | 16384 | ~16K | |
| 32K | 32768 | ~32K | |
| 128K | 131072 | ~128K | 单流 |
| 262K | 262144 | ~261K | config 支持 262144；HBM/图/质量通过才留 |

**2026-11-16 真权重实测**：8K/16K/32K/128K/261K 均完成（32K 用 32000 prompt 给生成留空间；
128K 32.22 s，261K 68.84 s，单流 eager）。这是长度/可运行性证据，不是内容理解正确性证据。

每档记录：`usage.prompt_tokens`、TTFT、TPOT、HBM 峰值、chunked prefill、回退、完整输出。

生成 prompt 可复用：

```bash
python benchmarks/make_long_prompt.py --tokens 32768 --out /tmp/m5-nemotron-32k.txt
```

---

## 8. D6：并发

```bash
# 先确认 max-num-seqs >= 客户端并发
python benchmarks/v3_conc.py "$PORT" /tmp/m5-nemotron-conc-16.json 2 16
python benchmarks/v3_conc.py "$PORT" /tmp/m5-nemotron-conc-32.json 2 32
```

每档记录：吞吐、TTFT/TPOT、排队、回退；不要用 4 seq 的配置打 32 路。

---

## 9. D7：冷启动

```bash
# 只“移走”，不删除；先获准
CACHE=/root/.cache/vllm/torch_compile_cache
BACKUP=/tmp/torch_compile_cache.m5.$(date +%s)
mv "$CACHE" "$BACKUP"
# 启动服务，记录到 /health 的时间、首请求 TTFT、请求期是否编译
# 完成后恢复：
# mv "$BACKUP" "$CACHE"
```

冷/热两臂条件必须一致，否则数字作废。

---

## 10. D8：算子审计

```bash
TILEINFER_AUDIT=$AUDIT vllm serve ...   # 与 C1 同配置
```

检查：

- 热路径 `impl=torch` 的实例；
- 是否有静默回退；
- 与静态 gap 对照。

---

## 11. 停止与回滚

```bash
# 只杀自己记录的 PID
kill -TERM "$(cat /tmp/m5-nemotron.pid)"
sleep 5
ps -p "$(cat /tmp/m5-nemotron.pid)" >/dev/null && kill -TERM "$(cat /tmp/m5-nemotron.pid)"
```

- 不 `pkill -f vllm`；
- 不动别人的端口/缓存；
- 失败时保留日志、PID、命令、时间，再回滚到上一版配置。

---

## 12. 证据落盘

| 项 | 路径 |
|---|---|
| 启动日志 | `/tmp/m5-nemotron-serve.log` |
| 指标 | `/tmp/m5-nemotron-metrics.json` |
| 算子审计 | `/tmp/m5-nemotron-opaudit.json` |
| 长上下文 | `/tmp/m5-nemotron-{8,16,32,128,262}k-*.json` |
| 并发 | `/tmp/m5-nemotron-conc-*.json` |
| 冷启动 | `/tmp/m5-nemotron-cold-*.log` |

最终结论写回 [`m5-nemotron-task.md`](m5-nemotron-task.md) 的 D1–D10，不把“跑过”写成“通过”。
