# 长上下文验证 runbook（待批准窗口）

> 对应 [`../verification-matrix.md`](../verification-matrix.md) 的 L1–L3。
> **前置：使用者已明确同意占用卡。** 未获同意不得执行本页任何 `vllm serve`。
> 当前配置：单卡 Ascend950PR、vLLM/vllm-ascend 0.23.0-950、CANN 9.3.0。

---

## 0. 前置检查（不部署，只读）

```bash
cd /home/userlist/zxs/TileInfer
source scripts/env-ascend102.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/nnal/atb/9.1.1/atb/cxx_abi_1/lib:${LD_LIBRARY_PATH:-}

npu-smi info | sed -n '1,20p'      # 必须没有他人的 process
ss -ltnp | grep 8322 || echo "8322 free"
ls -l /tmp/tileinfer-plan.json     # v3 臂需要
```

任何一项不满足就停：**不抢卡、不 kill 他人进程**。把 `npu-smi` 输出记进报告。

---

## 1. 生成 prompt（CPU，不占卡）

```bash
for N in 8192 16384 32768; do
  .venv_native213/bin/python benchmarks/make_long_prompt.py \
    --tokens $N --out /tmp/prompt-$N.txt
  for D in 10 50 90; do
    .venv_native213/bin/python benchmarks/needle_probe.py \
      --tokens $N --depth 0.$D --out-prompt /tmp/needle-$N-d$D.txt
  done
done
```

`make_long_prompt.py` 是测速用的连续文本；`needle_probe.py` 在 **10% / 50% / 90%**
三个深度各埋一次 `The secret access code is TIGER-7421`，末尾问同一个问题。
两者都以服务返回的 `usage.prompt_tokens` 为准，不拿字符数冒充 token。

---

## 2. v3 臂

### 2.1 起服务

`--max-model-len 40960` 是为了容纳 32K prompt + 32 输出 + chat template；
不是 32K 本身，32K 之外的余量留给模板和输出。

```bash
TILEINFER_PLAN=/tmp/tileinfer-plan.json \
TILEINFER_METRICS_PATH=/tmp/v3-40k-metrics.json \
setsid .venv_v3/bin/python -m vllm.entrypoints.cli.main serve /home/models/Qwen3.6-35B-A3B \
  --served-model-name qwen36 --port 8322 --trust-remote-code \
  --max-model-len 40960 --max-num-seqs 8 --gpu-memory-utilization 0.88 \
  > /tmp/v3-serve-40k.log 2>&1 &
SERVER_PID=$!
```

等待就绪（记录耗时；正常约 70–90 s）：

```bash
START=$(date +%s)
for i in $(seq 1 90); do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 http://127.0.0.1:8322/health || true)
  [ "$code" = 200 ] && break
  sleep 2
done
echo "ready in $(( $(date +%s) - START )) s"
grep -E "Available KV cache memory|Maximum concurrency|Capturing|torch.compile took" /tmp/v3-serve-40k.log | tail -10
```

### 2.2 测速（每档 1 次预热 + 3 次测量）

```bash
for N in 8192 16384 32768; do
  TI_PROMPT=/tmp/prompt-$N.txt .venv_v3/bin/python benchmarks/v3_long_ttft.py \
    8322 v3-$N /tmp/v3-$N.json 3
done
```

### 2.3 正确性（needle，10% / 50% / 90% 三个深度）

```bash
for N in 8192 16384 32768; do
  for D in 10 50 90; do
    .venv_v3/bin/python benchmarks/needle_probe.py --tokens $N --depth 0.$D \
      --out-prompt /tmp/needle-$N-d$D.txt --port 8322 --logprobs \
      --out-response /tmp/v3-$N-d$D-needle.json
  done
done
```

每档、每个深度都必须 `ok=true` 且 `answer` 含 `TIGER-7421`；否则是 C2 失败，
不是"长上下文慢"。Qwen3.6 会先输出 reasoning，32 token 会把答案截断；needle 默认
`--max-tokens 256`，判定是"回答里含 code"，不要求它真的只输出 code。
`--logprobs` 让响应里带 token 级 logprob，供 §4 逐 token 对比。

### 2.4 图计数与 doctor

```bash
ENGINE_PID=$(pgrep -f "VLLM::EngineCore" | head -1)
kill -USR1 "$ENGINE_PID"
cat /tmp/v3-40k-metrics.json

TILEINFER_PLAN=/tmp/tileinfer-plan.json \
TILEINFER_METRICS_PATH=/tmp/v3-40k-metrics.json \
  .venv_v3/bin/python -m tileinfer.cli.main doctor
```

看 `captured_graphs > 0`、`decode_fallbacks = 0`。长 prompt 的 prefill 不计入稳态回退，
但 decode 段必须零回退。

### 2.5 停服务

```bash
kill "$SERVER_PID"
for i in $(seq 1 30); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
ss -ltnp | grep 8322 || echo "8322 free"
```

不要用 `pkill -f vllm` 一刀切；只停自己起的 `$SERVER_PID`。若端口仍占用，
先 `ps -eo pid,cmd | grep 8322` 确认是自己的残留进程，再处理。

---

## 3. 基线臂（同参数，无 TileInfer）

```bash
setsid .venv_baseline/bin/python -m vllm.entrypoints.cli.main serve /home/models/Qwen3.6-35B-A3B \
  --served-model-name qwen36 --port 8322 --trust-remote-code \
  --max-model-len 40960 --max-num-seqs 8 --gpu-memory-utilization 0.88 \
  > /tmp/base-serve-40k.log 2>&1 &
BASE_PID=$!
```

重复 §2.2 与 §2.3，结果写 `/tmp/base-$N.json`、`/tmp/base-$N-d$D-needle.json`，
然后按 §2.5 停掉。

---

## 4. 对比与判定

```bash
# 每档：v3 / 基线 的 TTFT、TPOT
python3 - <<'EOF'
import json
for n in (8192, 16384, 32768):
    v=json.load(open(f"/tmp/v3-{n}.json")); b=json.load(open(f"/tmp/base-{n}.json"))
    print(n, "prompt_tokens", v["prompt_tokens"], b["prompt_tokens"],
          "TTFT", round(v["ttft_ms_p50"]/b["ttft_ms_p50"],3),
          "TPOT", round(v["tpot_ms_p50"]/b["tpot_ms_p50"],3))
EOF

# 正确性 / C2：逐 token + logprob 对比（atol 1e-4，四位小数口径）
for N in 8192 16384 32768; do
  for D in 10 50 90; do
    .venv_native213/bin/python benchmarks/compare_responses.py \
      /tmp/v3-$N-d$D-needle.json /tmp/base-$N-d$D-needle.json \
      > /tmp/cmp-$N-d$D.json || echo "C2 DIFF: $N d$D"
  done
done
python3 - <<'EOF'
import glob, json
for p in sorted(glob.glob("/tmp/cmp-*-d*.json")):
    d = json.load(open(p))
    print(p, "same=", d["same"], "max_lp=", d["max_abs_logprob_diff"])
EOF
```

判定（[`../acceptance.md`](../acceptance.md) §2.1，有基线走比值）：

- C1：两臂都能起、都能返回完整输出。
- C2：每个 needle 都 `ok=true`，且 `compare_responses.py` 的 `same=true`（文本、token
  序列、logprob 都在 1e-4 内）。任一不满足先查模板/采样，不调容差。
- C3：v3 的 `captured_graphs > 0` 且 `decode_fallbacks = 0`。
- C4：TTFT/TPOT ≤ 1.10× 基线、吞吐 ≥ 0.95×；每档单独判，不跨档拼比值。

把命令、时间、`prompt_tokens`、HBM、图计数、比值写进
`docs/notes/` 的证据页，再回写 `docs/status.md` 与矩阵。

---

## 5. 时间预算与回滚

| 步骤 | 预计 |
|---|---|
| 前置检查 + 生成 prompt | 2 min |
| v3 臂起服务 | 2 min |
| v3 8K/16K/32K 测速 + needle + metrics | 8–10 min |
| 基线臂起服务 + 同样三档 | 8–10 min |
| 停服务 + 落盘 | 2 min |
| **合计** | **20–25 min** |

失败即停：起不来、needle 错、回退非 0，都先停服务、保留日志，不在同一个窗口里反复重试。
把失败现象写进报告，下一次窗口再修。
