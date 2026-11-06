#!/bin/bash
# 等内存窗口 -> 起服务 -> 跑 TTFT 对照（chat vs completions）-> 落盘。
#
# 为什么需要它：本机被其他租户长期占用，服务常在测量中途被 OOM-killer 杀掉
# （实测 `anon-rss 484 GB` 被杀）。把"等窗口 + 起服务 + 测 + 存"串成一个自包含脚本，
# 窗口一开就能一次性拿到数据，不必人工盯着。
#
# 用法：  bash tools/dsv41/perf_ttft_ab.sh [--need 520] [--port 8122]
# 产出：  /tmp/ttft_ab.log（完整输出）  /tmp/ttft_ab.result（关键几行）
set -u

NEED=520
PORT=8122
while [ $# -gt 0 ]; do
  case "$1" in
    --need) NEED="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    *) shift ;;
  esac
done

REPO=/home/userlist/zxs/TileInfer
PY="$REPO/.venv/bin/python"
LOG=/tmp/ttft_ab.log
RES=/tmp/ttft_ab.result
: > "$LOG"
: > "$RES"

echo "== 等待内存窗口（available >= ${NEED} GB，最多 40 分钟）==" | tee -a "$LOG"
ok=0
for i in $(seq 1 80); do
  A=$(free -g | awk '/^Mem:/{print $7}')
  echo "[$i] $(date +%H:%M:%S) avail=${A}G" >> "$LOG"
  if [ "$A" -ge "$NEED" ]; then ok=1; break; fi
  sleep 30
done
if [ "$ok" != "1" ]; then
  echo "窗口未出现，退出（不启动任何服务）" | tee -a "$RES"
  exit 1
fi
echo "窗口出现：avail=${A}G" | tee -a "$RES"

pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null
sleep 5

: > /tmp/ttft_ab_svc.log    # 关键：截断旧日志，否则下面的 startup 匹配会命中上一轮的残留
cd "$REPO" || exit 1
export PYTHONDONTWRITEBYTECODE=1 VLLM_PLUGINS=tileinfer VLLM_DSV41=1
nohup "$PY" -m vllm.entrypoints.openai.api_server \
  --model /home/models/DeepSeek-V4.1-Flash --served-model-name dsv41 --trust-remote-code \
  --chat-template /home/models/DeepSeek-V4.1-Flash/chat_template.jinja \
  --enforce-eager --max-model-len 2048 --max-num-seqs 1 --gpu-memory-utilization 0.3 \
  --no-enable-prefix-caching --no-enable-chunked-prefill --port "$PORT" >> /tmp/ttft_ab_svc.log 2>&1 &

echo "== 等待就绪 ==" >> "$LOG"
for i in $(seq 1 80); do
  sleep 10
  if grep -q "Application startup complete" /tmp/ttft_ab_svc.log 2>/dev/null; then break; fi
  if ! pgrep -f "vllm.entrypoints.openai.api_server" >/dev/null; then
    echo "服务启动失败（可能再被 OOM）:" | tee -a "$RES"
    tail -n 5 /tmp/ttft_ab_svc.log | tee -a "$RES"
    exit 2
  fi
done

echo "== chat 流式 TTFT ==" >> "$LOG"
"$PY" tools/dsv41/perf_ttft.py --port "$PORT" --endpoint chat --reps 3 --max-tokens 4 >> "$LOG" 2>&1
echo "== completions 流式 TTFT ==" >> "$LOG"
"$PY" tools/dsv41/perf_ttft.py --port "$PORT" --endpoint completions --reps 3 --max-tokens 4 >> "$LOG" 2>&1

grep -aE "TTFT|decode" "$LOG" | tail -n 8 >> "$RES"
echo "== 完成 $(date +%H:%M:%S) ==" >> "$RES"
