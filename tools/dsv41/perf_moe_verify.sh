#!/bin/bash
# MoE 优化的数值验收 + vLLM decode 分派验证（窗口驱动）。
#
# 步骤：
#   ① 独立栈 batched 臂：确认 moe_forward_batched 在设备上仍能出正确首 token（Paris）
#      —— 这是启用 vLLM 侧 DSV41_DECODE_BATCHED 的前置条件（数值硬门槛：首 token 11111）
#   ② vLLM + DSV41_DECODE_BATCHED=1：确认 /v1/completions 仍返回 Paris
#   ③ （若 ② 通过）提示继续用 s5_wall_tpot.py 做差分法 TPOT
#
# 为什么窗口驱动：476 GiB 权重要求 available >= 520 GB，否则会被 OOM-killer 杀掉
# （实测 anon-rss 484 GB 被杀）。脚本自己等窗口，窗口一开就跑完。
#
# 用法： bash tools/dsv41/perf_moe_verify.sh [--need 520] [--port 8122]
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
RES=/tmp/moe_verify.result
: > "$RES"

echo "== 等窗口（available >= ${NEED} GB，最多 40 分钟）==" | tee -a "$RES"
ok=0
for i in $(seq 1 80); do
  A=$(free -g | awk '/^Mem:/{print $7}')
  if [ "$A" -ge "$NEED" ]; then ok=1; break; fi
  sleep 30
done
if [ "$ok" != "1" ]; then
  echo "窗口未出现（最后 avail=${A}G），退出" | tee -a "$RES"
  exit 1
fi
echo "窗口：avail=${A}G" | tee -a "$RES"

cd "$REPO" || exit 1

echo "== ① 独立栈 batched 臂（数值验收）==" | tee -a "$RES"
timeout 900 "$PY" tools/dsv41/s4_full_stack_device.py --batched --gen 2 \
  --prompt "The capital of France is" > /tmp/moe_v1.log 2>&1
grep -aE "generated text|token=" /tmp/moe_v1.log | tail -n 3 | tee -a "$RES"
# 区分"被 OOM 杀"与"算错"：前者日志往往为空、且找不到 generated text 行
if grep -qa "Paris" /tmp/moe_v1.log; then
  echo "① PASS：batched 臂仍出 Paris" | tee -a "$RES"
elif grep -qa "generated text" /tmp/moe_v1.log; then
  echo "① FAIL：跑了但输出不含 Paris（数值问题），停止" | tee -a "$RES"
  exit 2
else
  echo "① ABORT：没有产出（大概率被 OOM-killer 杀掉，logs 见 /tmp/moe_v1.log），不启用 vLLM 侧 decode 分派" | tee -a "$RES"
  tail -n 3 /tmp/moe_v1.log | tee -a "$RES"
  exit 5
fi

echo "== ② vLLM + DSV41_DECODE_BATCHED=1 ==" | tee -a "$RES"
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null
sleep 5
: > /tmp/moe_v_svc.log
export PYTHONDONTWRITEBYTECODE=1 VLLM_PLUGINS=tileinfer VLLM_DSV41=1 DSV41_DECODE_BATCHED=1
nohup "$PY" -m vllm.entrypoints.openai.api_server \
  --model /home/models/DeepSeek-V4.1-Flash --served-model-name dsv41 --trust-remote-code \
  --chat-template /home/models/DeepSeek-V4.1-Flash/chat_template.jinja \
  --enforce-eager --max-model-len 2048 --max-num-seqs 1 --gpu-memory-utilization 0.3 \
  --no-enable-prefix-caching --no-enable-chunked-prefill --port "$PORT" >> /tmp/moe_v_svc.log 2>&1 &

for i in $(seq 1 80); do
  sleep 10
  if grep -q "Application startup complete" /tmp/moe_v_svc.log 2>/dev/null; then break; fi
  if ! pgrep -f "vllm.entrypoints.openai.api_server" >/dev/null; then
    echo "② 服务启动失败：" | tee -a "$RES"
    tail -n 5 /tmp/moe_v_svc.log | tee -a "$RES"
    exit 3
  fi
done

OUT=$(curl -s -m 300 "http://127.0.0.1:${PORT}/v1/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"dsv41","prompt":"The capital of France is","max_tokens":4,"temperature":0}')
echo "$OUT" | head -c 300 | tee -a "$RES"
echo | tee -a "$RES"
if echo "$OUT" | grep -q "Paris"; then
  echo "② PASS：vLLM（decode=batched）仍出 Paris" | tee -a "$RES"
  echo "③ 下一步：DSV41_DECODE_BATCHED=1 $PY tools/dsv41/s5_wall_tpot.py --n1 8 --n2 72 --reps 3" | tee -a "$RES"
else
  echo "② FAIL：vLLM 输出不正确，保持默认（streamed）分派" | tee -a "$RES"
  exit 4
fi
echo "== 完成 $(date +%H:%M:%S) ==" | tee -a "$RES"
