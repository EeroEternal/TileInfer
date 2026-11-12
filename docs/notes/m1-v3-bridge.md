# M1：v3 适配层跑通 Qwen3.6-35B-A3B（2026-11-12）

> 一句话：**同一套 vllm-ascend 0.23 / CANN 9.3 / Qwen3.6 权重，外面加一层 TileInfer v3，
> 服务能起、输出逐位一致、图捕获 8 张、稳态回退 0、TPOT/首字/吞吐都在 1.10×/0.95× 内。**
>
> 这不是 v2 的性能翻案：v2 自己实现平台/worker/KV/图捕获，TPOT 2.53×；v3 把那些交还给
> 厂商，只做"选实现 + 补缺口"，所以数字回到 1.0× 是设计结果，不是调优结果。
>
> 基线页：[`m0-baseline.md`](m0-baseline.md)。验收口径：[`../acceptance.md`](../acceptance.md)。

## 1. 形态与边界

| 层 | 目录 | 职责 | 是否 import vllm_ascend |
|---|---|---|---|
| L0 CLI | `tileinfer/cli/` | `plan build/show/verify`、`doctor` | 否（doctor 通过 probe 读 build info） |
| L1 声明 | `tileinfer/manifests/` | 模型需要什么：层序列、头数、KV、量化、能力表、oracle | 否 |
| L2 决策 | `tileinfer/resolve/` | 探测环境 → 纯函数决策 → plan + gap report | 否 |
| L3 契约 | `tileinfer/contracts/` | 8 类能力协议、plan/manifest/gap 的 schema 与校验 | 否 |
| L4 提供者 | `tileinfer/providers/` | 我们自己的实现（M1 为空） | 否 |
| L5 桥 | `tileinfer/bridge/` | **唯一**允许 import 上游的层；安装 6 个钩子、计数 | 是 |
| L6 内核 | `tileinfer/kernels/` | v2 资产，M1 不动 | 否 |

`tests/test_no_vllm_ascend.py` 已从"全仓库绝对禁止"改成"只有 `tileinfer/bridge/` 可以"：
先解析 AST 的 import，再按路径前缀判定；另有一条测试要求 bridge 真的 import 了厂商包，
防止 carve-out 变成空规则。`tests/test_layer_deps.py` 把 `contracts / manifests / resolve /
providers / aot / observ / cli` 加进 engine-free 列表。

## 2. 四个接口产物

1. **声明**：`tileinfer/manifests/qwen3_6.py`。40 层 = 30 GDN + 10 full（`i % 4 == 3`），
   16 Q / 2 KV、head_dim 256、`attn_output_gate`、partial RoPE 0.25；GDN 16 K / 32 V、
   128/128、conv 4、state fp32；256 experts top-8、intermediate 512；bf16；8 项 required
   capability；benchmark 里冻结了首 token `Here` 及其 top-5。
2. **契约**：`tileinfer/contracts/capabilities.py` 的 8 个 `Protocol`（full/linear attention、
   MoE、RMSNorm、RoPE、sampling、state、quant），加 `ProviderSpec` 五项声明
   （coverage / graphable / parallel / 证据（有基线才带 measured_ratio）/ delete_when）。
3. **决策**：`tileinfer/resolve/decide.py` 是纯函数；`vendor_first` 下 8/8 选厂商，
   gap=0、servable=True。没有厂商时输出 8 条 required gap、servable=False
   （这是"这台机器能不能服务"的判定，不是"要去补 8 个实现"的待办——见
   [`../extending.md`](../extending.md) §3.0）。
4. **接入**：`tileinfer/bridge/install.py` + `seams.py`。入口只做四件事：读 plan → 版本门 →
   装钩子 → 记录实际生效的 seam；任何异常不外抛。

Plan 是内容寻址的，本次为 `7d00064d487af3f6`：

```bash
.venv_v3/bin/python -m tileinfer.cli.plan build \
  --model /home/models/Qwen3.6-35B-A3B --out /tmp/tileinfer-plan.json
# plan=7d00064d487af3f6 model=Qwen3.6-35B-A3B selections=8 (...) access_points=4 fallback=forbid
# model=Qwen3.6-35B-A3B servable=True gaps=0 (required=0)
```

## 3. 接入点：6 个具体钩子，4/6 类机制，预算 8

| # | 机制 | 具体位置 | 做什么 | 改数值？ |
|---|---|---|---|---|
| 1 | plugin_entry | `pyproject.toml` → `tileinfer.bridge.install:install` | 唯一注册的 vLLM 入口；**无 platform_plugins** | 否 |
| 2 | method_wrap | `CudagraphDispatcher.initialize_cudagraph_keys` | 记录 capture keys / mode | 否 |
| 3 | graph_hook | `CudagraphDispatcher.dispatch` | 统计 decode 命中/回退 | 否 |
| 4 | graph_hook | `GPUModelRunner.capture_model` | capture 后统计图数量（`get_capture_descs`） | 否 |
| 5 | attn_select | `NPUPlatform.get_attn_backend_cls` | 按 plan 返回实现；vendor 选择原样返回 | 否 |
| 6 | method_wrap | `NPUPlatform.check_and_update_config` | 校验 plan.arch == 实际 checkpoint.arch，不匹配 fail-closed | 否 |

预算 8，用 6；`impl_table` / `model_table` 两个机制留了空实现（非 vendor 选择时才写表），
不占当前钩子数。所有钩子都带 `_tileinfer_*` 幂等标记。

负向门已单独验证：

```text
negative gate ok: TileInfer plan/model mismatch: plan is for 'Qwen3_5MoeForConditionalGeneration'
```

## 4. 验收证据

### C1 能跑

```bash
.venv_v3/bin/python -m vllm.entrypoints.cli.main serve /home/models/Qwen3.6-35B-A3B \
  --served-model-name qwen36 --port 8322 --trust-remote-code \
  --max-model-len 9216 --max-num-seqs 32 --gpu-memory-utilization 0.88
```

启动 70 s 到 `/health=200`；`init engine` 23.35–23.88 s（编译 12.1–12.5 s）；权重 65.52 GiB，
KV 39.77 GiB；`FULL_AND_PIECEWISE`；capture sizes `[1,4,16,32]`。启动时只有
`ascend -> vllm_ascend:register` 一个平台插件；`tileinfer` 只在 `vllm.general_plugins`。

### C2 跑对（逐位一致）

请求 `/tmp/req_qwen36.json`（temperature 0，top_logprobs 5，max_tokens 16）。
基线 `/tmp/baseline_fia.json` vs v3 `/tmp/v3-c2-gated.json`，去掉 `id/created/
system_fingerprint` 三个易变字段后 **JSON 完全相等**：文本、16 个 token、每个 token 的
logprob 与 top-5 全部一致；首 token `Here -0.0302655640989542`，usage 15/16。

### C3 能上图、稳态回退 0

`SIGUSR1` 到 EngineCore 后写 `/tmp/tileinfer-v3-metrics-final.json`：

```json
{
  "plan_id": "7d00064d487af3f6",
  "capture_calls": 1,
  "captured_graphs": 8,
  "capture_keys": {"PIECEWISE": 4, "FULL": 4},
  "cudagraph_mode": "FULL_AND_PIECEWISE",
  "capture_done": true,
  "decode_calls": 15,
  "decode_graph_hits": 19,
  "decode_fallbacks": 0,
  "attention_backend": "vllm_ascend.attention.attention_v1.AscendAttentionBackend",
  "errors": 0
}
```

另一轮压测（同一实现，统计口径尚未加 `capture_done` 过滤）为 `decode_calls=2375, graph_hits=2375, fallbacks=0`；门禁轮用上面的严格口径。回退的定义是：
`capture_done` 之后的 uniform decode 步，且 `valid_modes` 含 `FULL/PIECEWISE`，
`dispatch()` 仍返回 `NONE`。预热/profile 步与超长 prefill 不计入。

指标落盘修正：API server 也装了 bridge 但从不 capture，它的 `atexit` 曾把 EngineCore 的
计数覆盖成 0；现在 `dump()` 对"零 capture 且零 decode"的进程直接跳过，`SIGUSR1` 仍强制写。

### C4 跑得快

单流 / 长上下文（`max-num-seqs=32`，batch=1，上限不影响）：

| 场景 | 基线 TTFT / TPOT | v3 TTFT / TPOT | 比值 TTFT / TPOT |
|---|---|---|---|
| 1 token（33 prompt） | 96.94 / – | 87.92 / – | **0.907** / – |
| decode 128 | 96.49 / 8.745 | 93.62 / 8.745 | **0.970 / 1.000** |
| decode 256 | 95.68 / 8.755 | 97.24 / 8.760 | **1.016 / 1.001** |
| 4K prompt（3939 tok） | 262.16 / 8.911 | 262.24 / 8.902 | **1.000 / 0.999** |

并发（**`max-num-seqs` 与客户端并发严格相等**；两臂各 3–5 次中位数）：

| 场景 | 并发/上限 | 基线 吞吐 / TTFT / TPOT | v3 吞吐 / TTFT / TPOT | 比值 吞吐 / TTFT / TPOT |
|---|---|---|---|---|
| conc_4 | 4 / 4 | 327.10 / 186.25 / 10.85 | 326.02 / 192.55 / 10.84 | **0.997 / 1.034 / 0.999** |
| conc_16 | 16 / 16 | 1085.02 / 198.13 / 13.28 | 1088.48 / 197.49 / 13.24 | **1.003 / 0.997 / 0.997** |

门：TTFT/TPOT ≤1.10×，吞吐 ≥0.95×。全部通过。A1/A2 基线漂移 ≤3%，见 M0 页。

`tileinfer doctor` 在最终 metrics 上：

```text
PASS python / vllm / vllm_ascend / device_type / plan / layer_guard / bridge
PASS graph_metrics  captured=8 decode=15 fallbacks=0
OK
```

## 5. 复现

```bash
cd /home/userlist/zxs/TileInfer
source scripts/env-ascend102.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/nnal/atb/9.1.1/atb/cxx_abi_1/lib:${LD_LIBRARY_PATH}

# 1) 一次性：从基线 venv 复制出 v3 venv，装 editable（--no-deps 保持与基线同 torch/vLLM）
cp -a .venv_baseline .venv_v3
.venv_v3/bin/python -m pip install -e . --no-deps --no-build-isolation

# 2) 部署期决策
.venv_v3/bin/python -m tileinfer.cli.plan build \
  --model /home/models/Qwen3.6-35B-A3B --out /tmp/tileinfer-plan.json

# 3) 起服务（脚本见 /tmp/run_v3.sh）
TILEINFER_PLAN=/tmp/tileinfer-plan.json \
TILEINFER_METRICS_PATH=/tmp/tileinfer-v3-metrics.json \
setsid .venv_v3/bin/python -m vllm.entrypoints.cli.main serve /home/models/Qwen3.6-35B-A3B \
  --served-model-name qwen36 --port 8322 --trust-remote-code \
  --max-model-len 9216 --max-num-seqs 32 --gpu-memory-utilization 0.88 \
  > /tmp/v3-serve.log 2>&1 &

# 4) 验收（A/B 脚本已入库：benchmarks/v3_ab.py、v3_long_ttft.py、v3_conc.py）
curl -s http://127.0.0.1:8322/v1/chat/completions -H 'Content-Type: application/json' \
  -d @/tmp/req_qwen36.json -o /tmp/v3-c2.json
kill -USR1 <EngineCore pid>          # 强制落 metrics
TILEINFER_PLAN=/tmp/tileinfer-plan.json \
TILEINFER_METRICS_PATH=/tmp/tileinfer-v3-metrics.json \
  .venv_v3/bin/python -m tileinfer.cli.main doctor
```

## 6. 这版还没做的（不要把 M1 当 M5）

* **补缺口还是空的**：`tileinfer/providers/` 没有 provider，8/8 都是厂商实现；gap=0 是因为
  厂商全覆盖，不是因为我们补上了。`impl_table` 遇到非 vendor 选择会返回 False 并记录，
  不会静默装错。
* **MTP / 投机解码**：未接；Qwen3.6 的 MTP 头在权重里，但批语义与 vLLM 0.23 的耦合还没做。
* **8K/16K**：M1 只测到 3939 prompt tokens；`max-model-len=9216`，16K 场景会直接被拒。
* **冷缓存 A/B**：两臂共用 `.venv_baseline` 的编译缓存，且 v3 复制同一份；这让 A/B 公平，
  但没有回答"全新机器第一次启动"的 TTFT。M2 的预编译门要单独测冷启动。
* **日志一行**：`install()` 在 API server 配置日志之前执行，`logger.info` 可能被过滤器丢掉；
  现在同时向 stderr 打一行 `[tileinfer] ...`，服务日志里必定可见。结构化证据仍以 `doctor`
  和 metrics 的 EngineCore pid 为准。
* **doctor 的 device_type 检查**在无 CANN 的纯 CPU 机上会 FAIL；这是有意的（v3 只支持
  Ascend），但 CI 上应加 `--skip-device` 一类开关。
