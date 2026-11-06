#!/bin/bash
# vLLM 路径 TPOT 双臂差分法 A/B：默认分派 vs DSV41_DECODE_BATCHED=1（窗口驱动）。
#
# 两臂必须同 venv、同启动参数、同一 prompt、同一 N1/N2（C1）。两次启动是必要的——
# 开关是进程级环境变量，无法在同一服务内切换。
#
# 流程：
#   等窗口(>=515G) → 起服务(默认) → TPOT 差分法 → 记录 → 停服务
#                  → 起服务(DECODE_BATCHED=1) → 先验 Paris → TPOT 差分法 → 记录
# 产出：/tmp/moe_tpot.result（人读） + /tmp/tpot_default.json + /tmp/tpot_batched.json
#
# 用法： bash tools/dsv41/perf_moe_tpot.sh [--need 515] [--port 8122] [--n1 8] [--n2 72] [--reps 3]
set -u

NEED=515
PORT=8122
N1=8
N2=72
REPS=3
while [ $# -gt 0 ]; do
  case "$1" in
    --need) NEED="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --n1) N1="$2"; shift 2 ;;
    --n2) N2="$2"; shift 2 ;;
    --reps) REPS="$2"; shift 2 ;;
    *) shift ;;
  esac
done

REPO=/home/userlist/zxs/TileInfer
PY="$REPO/.venv/bin/python"
RES=/tmp/moe_tpot.result
SVC=/tmp/moe_tpot_svc.log
: > "$RES"

wait_window() {
  echo "== 等窗口（available >= ${NEED} GB）==" | tee -a "$RES"
  for i in $(seq 1 80); do
    A=$(free -g | awk '/^Mem:/{print $7}')
    if [ "$A" -ge "$NEED" ]; then echo "窗口：avail=${A}G" | tee -a "$RES"; return 0; fi
    sleep 30
  done
  echo "窗口未出现（最后 ${A}G）" | tee -a "$RES"
  return 1
}

start_svc() {   # $1 = 额外的环境前缀（可为空）
  pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null
  sleep 5
  : > "$SVC"
  cd "$REPO" || return 1
  export PYTHONDONTWRITEBYTECODE=1 VLLM_PLUGINS=tileinfer VLLM_DSV41=1
  env $1 nohup "$PY" -m vllm.entrypoints.openai.api_server \
    --model /home/models/DeepSeek-V4.1-Flash --served-model-name dsv41 --trust-remote-code \
    --chat-template /home/models/DeepSeek-V4.1-Flash/chat_template.jinja \
    --enforce-eager --max-model-len 2048 --max-num-seqs 1 --gpu-memory-utilization 0.3 \
    --no-enable-prefix-caching --no-enable-chunked-prefill --port "$PORT" >> "$SVC" 2>&1 &
  for i in $(seq 1 80); do
    sleep 10
    if grep -q "Application startup complete" "$SVC" 2>/dev/null; then return 0; fi
    if ! pgrep -f "vllm.entrypoints.openai.api_server" >/dev/null; then
      echo "服务启动失败：" | tee -a "$RES"; tail -n 5 "$SVC" | tee -a "$RES"; return 1
    fi
  done
  echo "服务就绪超时" | tee -a "$RES"
  return 1
}

check_paris() {
  OUT=$(curl -s -m 300 "http://127.0.0.1:${PORT}/v1/completions" -H 'Content-Type: application/json' \
    -d '{"model":"dsv41","prompt":"The capital of France is","max_tokens":4,"temperature":0}')
  if echo "$OUT" | grep -q "Paris"; then
    echo "  首 token 校验 PASS（输出含 Paris）" | tee -a "$RES"; return 0
  fi
  echo "  首 token 校验 FAIL（不测性能，保持默认分派）" | tee -a "$RES"
  echo "$OUT" | head -c 200 | tee -a "$RES"; echo | tee -a "$RES"
  return 1
}

wait_window || exit 1

echo "== 臂 A：默认分派 ==" | tee -a "$RES"
start_svc "" || exit 2
check_paris || exit 3
"$PY" tools/dsv41/perf_vllm_tpot.py --port "$PORT" --n1 "$N1" --n2 "$N2" --reps "$REPS" \
  --label default --out /tmp/tpot_default.json 2>&1 | tee -a "$RES"

echo "== 臂 B：DSV41_DECODE_BATCHED=1 ==" | tee -a "$RES"
start_svc "DSV41_DECODE_BATCHED=1" || exit 4
check_paris || exit 5
"$PY" tools/dsv41/perf_vllm_tpot.py --port "$PORT" --n1 "$N1" --n2 "$N2" --reps "$REPS" \
  --label batched --out /tmp/tpot_batched.json 2>&1 | tee -a "$RES"

echo "== 汇总 ==" | tee -a "$RES"
"$PY" - <<'PY' 2>&1 | tee -a /tmp/moe_tpot.result
import json, os
rows = []
for name, path in (("default", "/tmp/tpot_default.json"), ("batched", "/tmp/tpot_batched.json")):
    if os.path.exists(path):
        d = json.load(open(path, encoding="utf-8"))
        rows.append((name, d["tpot_ms"], d["spread_t1_pct"], d["spread_t2_pct"]))
for name, tpot, s1, s2 in rows:
    print(f"  {name:8s} TPOT={tpot:7.1f} ms/token  极差 T1={s1:.1f}% T2={s2:.1f}%")
if len(rows) == 2:
    a, b = rows[0][1], rows[1][1]
    print(f"  差异：{(b - a) / a * 100:+.1f}%（负值表示 batched 更快）")
PY
echo "== 完成 $(date +%H:%M:%S) ==" | tee -a "$RES"
