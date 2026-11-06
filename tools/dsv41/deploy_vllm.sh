#!/bin/bash
# 部署 DeepSeek-V4.1-Flash（vLLM 0.23 + TileInfer，不装 vllm-ascend）
#
# 这一行组合是 2026-11-04 本机实测跑通并返回 " Paris." 的那一条，未做任何改动。
# 关键点：
#   VLLM_PLUGINS=tileinfer  → 加载本仓库的 Ascend 平台插件
#   VLLM_DSV41=1            → 选 Dsv41Platform（放行 fp8/fp4；默认平台只允许 bf16）
#   --no-enable-prefix-caching → 绕开"0 个 KV cache group"断言（模型自持 KV，不用分页缓存）
#   --max-model-len 64      → 功能验证够用；要用长上下文需调大并重新核算显存
#
# 用法：  bash tools/dsv41/deploy_vllm.sh [port]
set -u

PORT="${1:-8121}"
REPO=/home/userlist/zxs/TileInfer
MODEL=/home/models/DeepSeek-V4.1-Flash
PY="$REPO/.venv/bin/python"
LOG=/tmp/dsv41_serve.log
NEED_GB=560          # 476 GiB 权重 + vLLM 自身缓冲；低于此值实测会被 OOM-killer 强杀

echo "== 1. 前置检查 =="
AVAIL=$(free -g | awk '/^Mem:/{print $7}')
echo "主机可用内存: ${AVAIL} GB (需要 >= ${NEED_GB} GB)"
if [ "$AVAIL" -lt "$NEED_GB" ]; then
  echo "!! 内存不足，停止。请先释放内存（其它租户进程会占用 100+ GB）。"
  exit 1
fi
if pgrep -f "vllm.entrypoints.openai.api_server" >/dev/null; then
  echo "发现已在运行的服务，先停掉旧的"
  pkill -f "vllm.entrypoints.openai.api_server"
  sleep 5
fi
npu-smi info 2>/dev/null | sed -n '/HBM-Usage/,+3p' | tail -n 1

echo "== 2. 启动服务（加载 475 GiB 约需 2-5 分钟）=="
cd "$REPO" || exit 1
export PYTHONDONTWRITEBYTECODE=1
export VLLM_PLUGINS=tileinfer
export VLLM_DSV41=1
setsid bash -c "exec '$PY' -m vllm.entrypoints.openai.api_server \
  --model '$MODEL' --served-model-name dsv41 --trust-remote-code \
  --enforce-eager --max-model-len 64 --max-num-seqs 1 \
  --gpu-memory-utilization 0.3 \
  --no-enable-prefix-caching --no-enable-chunked-prefill \
  --port $PORT" </dev/null >"$LOG" 2>&1 &

echo "日志: $LOG"
echo "== 3. 等待就绪（最多 12 分钟）=="
for i in $(seq 1 72); do
  sleep 10
  if grep -q "Application startup complete" "$LOG" 2>/dev/null; then
    echo "服务已就绪: http://127.0.0.1:${PORT}"
    grep -aE "Loading weights took|Model loading took" "$LOG" | tail -n 2
    exit 0
  fi
  if grep -qaE "Engine core initialization failed|OutOfMemory|Killed" "$LOG" 2>/dev/null; then
    echo "!! 启动失败，日志尾部："
    tail -n 15 "$LOG" | cut -c1-160
    exit 2
  fi
done
echo "!! 等待超时，日志尾部："
tail -n 15 "$LOG" | cut -c1-160
exit 3
