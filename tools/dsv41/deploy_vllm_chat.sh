#!/bin/bash
# 部署 DeepSeek-V4.1-Flash 以支持 /v1/chat/completions（真实多轮对话）
#
# 与 deploy_vllm.sh 的差别只有三点，都是对话必需：
#   1) --chat-template  ：checkpoint 的 tokenizer_config.json 里没有内嵌 chat_template，
#                         模板是独立文件 chat_template.jinja，必须显式指定，否则 chat 接口报错
#   2) --max-model-len  ：默认 64 放不下模板前缀 + 多轮历史，这里给 2048
#   3) 端口默认 8122     ：可与既有的 completions 服务并存做对比
#
# 用法：  bash tools/dsv41/deploy_vllm_chat.sh [port] [max_model_len]
set -u

PORT="${1:-8122}"
MAXLEN="${2:-2048}"
REPO=/home/userlist/zxs/TileInfer
MODEL=/home/models/DeepSeek-V4.1-Flash
TPL="$MODEL/chat_template.jinja"
PY="$REPO/.venv/bin/python"
LOG=/tmp/dsv41_chat.log
NEED_GB="${NEED_GB:-560}"

echo "== 1. 前置检查 =="
AVAIL=$(free -g | awk '/^Mem:/{print $7}')
echo "主机可用内存: ${AVAIL} GB (需要 >= ${NEED_GB} GB)"
[ "$AVAIL" -lt "$NEED_GB" ] && { echo "!! 内存不足，先释放内存"; exit 1; }
[ -f "$TPL" ] || { echo "!! 找不到 $TPL"; exit 1; }
if pgrep -f "vllm.entrypoints.openai.api_server" >/dev/null; then
  echo "发现已有服务，先停掉（同一张卡只跑一个）"
  pkill -f "vllm.entrypoints.openai.api_server"; sleep 5
fi
npu-smi info 2>/dev/null | sed -n '/HBM-Usage/,+3p' | tail -n 1

echo "== 2. 启动对话服务（max-model-len=${MAXLEN}，带 chat template）=="
cd "$REPO" || exit 1
# DSV41_MOE_GROUPED=auto: prefill 侧 MoE 分组 GEMM，实测 -24.5%（受控中位数，
# docs/notes/dsv41-perf-moe-ab.md §6）；1 行输入时自动退回原路径。
export PYTHONDONTWRITEBYTECODE=1 VLLM_PLUGINS=tileinfer VLLM_DSV41=1 DSV41_MOE_GROUPED=auto
setsid bash -c "exec '$PY' -m vllm.entrypoints.openai.api_server \
  --model '$MODEL' --served-model-name dsv41 --trust-remote-code \
  --chat-template '$TPL' \
  --enforce-eager --max-model-len $MAXLEN --max-num-seqs 1 \
  --gpu-memory-utilization 0.3 \
  --no-enable-prefix-caching --no-enable-chunked-prefill \
  --port $PORT" </dev/null >"$LOG" 2>&1 &

echo "日志: $LOG"
echo "== 3. 等待就绪（最多 12 分钟）=="
for i in $(seq 1 72); do
  sleep 10
  if grep -q "Application startup complete" "$LOG" 2>/dev/null; then
    echo "就绪: http://127.0.0.1:${PORT}"
    grep -aE "Loading weights took|Model loading took" "$LOG" | tail -n 2
    exit 0
  fi
  if grep -qaE "Engine core initialization failed|OutOfMemory|Killed" "$LOG" 2>/dev/null; then
    echo "!! 启动失败："; tail -n 15 "$LOG" | cut -c1-160; exit 2
  fi
done
echo "!! 超时："; tail -n 15 "$LOG" | cut -c1-160; exit 3
