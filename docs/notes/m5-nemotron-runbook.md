# Nemotron 实机运行手册（O6，默认不执行）

> **默认状态：不执行。** 任何命令在运行前都必须先向用户说明命令、端口、预计时长、
> 卡号/显存，并取得**明确同意**。`npu-smi` 也只在获准后运行。
>
> 共享环境：只停止自己启动的 PID；不要 `pkill`/`killall`；不要动他人的进程、端口、缓存。

---

## 0. 部署前检查表（全部满足才申请）

- [ ] O1–O5 离线产物已 review；
- [ ] `config.json` 已确认：`architectures/model_type/num_hidden_layers/
      layers_block_type/mtp_layers_block_type/num_nextn_predict_layers/max_position_embeddings`；
- [ ] 已生成 vLLM 0.23 兼容的 config shim（见 §1.1），且 `NemotronHConfig` 能加载；
- [ ] 权重路径存在，磁盘/内存足够；当前是 **5/14 分片**，`ready_to_load=false`，不要做真权重 C1；
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
2026-11-13 的现状是 5/14 分片，`ready_to_load=false`、`ready_for_dummy=true`。

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
Transformers 4.57 下 `ModelConfig` 会在模型代码之前直接失败。TileInfer 的 bridge 在
`ModelConfig` 之前注册 vLLM 自带的类（`tileinfer/vllm/config_compat.py`），不修改上游源码。
如果 bridge 被 `TILEINFER_V3=0` 关闭，这个别名不会注册；因此目标服务不要关 bridge。

### 1.3 目标 plan（不注册 draft）

`NEMOTRON_3_5_DRAFT` 仍在 benchmark pending，**不在** `MANIFESTS`；但实机需要目标 plan 才能
过 config gate。用离线工具生成临时 plan：

```bash
.venv_v3/bin/python tools/nemotron/build_plan.py --out "$PLAN"
```

输出会明确 `servable=false`、`gaps=["state.linear"]`；这是 bring-up 用的 plan，不是支持声明。
启动时设 `TILEINFER_PLAN="$PLAN"`，否则 bridge 会按默认 Qwen 声明生成 plan，config gate
会报 `TileInfer plan/model mismatch`。

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

### 2.1 已记录的合成小模型结果（2026-11-13）

目标卡当时被他人 `VLLM::EngineCore` 占用（约 103 GB HBM），没有抢占，也没有起目标 30B
dummy。改用 2 层、`M*`、hidden 256 的合成同架构模型做机械 smoke：

- 能构造模型、加载 0.1253 GB dummy 权重；
- 进入 `Warming up Mamba2 SSD Triton kernels` 后失败：
  `ssd_chunk_scan.py:238` 的 `CompilationError` / `AttributeError('NoneType' object has no
  attribute 'type')`，Ascend 后端报 `Currently ptr type from different source not supported`；
- 服务没有监听 `/health`，自己的进程已停止；
- 证据：[`m5-nemotron-tiny-mamba2.json`](m5-nemotron-tiny-mamba2.json)。

这说明当前 vllm-ascend 的 Mamba2 SSD chunk-scan 路径不可用；但合成形状不是目标 30B 形状，
不能直接写成“目标确认不支持”。目标 C1 仍需完整权重 + 空闲卡 + 目标 dummy。

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

```bash
# 参考实现（HF/transformers）固定样例；权重/CPU 时间先估
python tools/nemotron/reference_probe.py \
  --model "$MODEL" \
  --prompt "Say hello in one sentence." \
  --max-tokens 32 \
  --out /tmp/m5-nemotron-ref.json
```

对比：

```bash
python benchmarks/compare_responses.py \
  /tmp/m5-nemotron-ref.json /tmp/m5-nemotron-c1.json
```

> 参考实现未落地前，本步是 `blocked`；不要拿另一个 vLLM 场景的绝对值当参考。

---

## 5. C3：图捕获

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
