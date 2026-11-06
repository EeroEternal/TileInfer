#!/bin/bash
# 验证已部署的 DeepSeek-V4.1-Flash 服务：健康检查 / 模型列表 / 贪心补全 / 期望值断言
#
# 期望值来自本次会话实测：prompt "The capital of France is" → " Paris."（token 11111，与官方 oracle 一致）
#
# 用法：  bash tools/dsv41/verify_vllm.sh [port]
set -u

PORT="${1:-8121}"
BASE="http://127.0.0.1:${PORT}"
FAIL=0

echo "== 1. 健康检查 GET /health =="
CODE=$(curl -s -m 10 -o /dev/null -w '%{http_code}' "$BASE/health")
echo "HTTP ${CODE}  (期望 200)"
[ "$CODE" = "200" ] || { echo "!! 服务未就绪"; exit 1; }

echo
echo "== 2. 模型列表 GET /v1/models =="
curl -s -m 10 "$BASE/v1/models" | head -c 300
echo

echo
echo "== 3. 贪心补全 POST /v1/completions（核心验收）=="
echo 'prompt: "The capital of France is"  期望文本包含 "Paris"'
T0=$(date +%s.%N)
RESP=$(curl -s -m 300 "$BASE/v1/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"dsv41","prompt":"The capital of France is","max_tokens":4,"temperature":0}')
T1=$(date +%s.%N)
echo "$RESP" | head -c 500
echo
echo "耗时: $(echo "$T1 - $T0" | bc) s"
if echo "$RESP" | grep -q "Paris"; then
  echo "✅ PASS：输出包含 Paris"
else
  echo "❌ FAIL：输出不含 Paris"
  FAIL=1
fi

echo
echo "== 4. 第二组 prompt（交叉确认不是偶然）=="
echo 'prompt: "1 + 1 ="  期望以 2 开头'
curl -s -m 300 "$BASE/v1/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"dsv41","prompt":"1 + 1 =","max_tokens":4,"temperature":0}' | head -c 400
echo

echo
echo "== 5. 采样可复现性（同一请求两次，temperature=0 应一致）=="
A=$(curl -s -m 300 "$BASE/v1/completions" -H 'Content-Type: application/json' \
     -d '{"model":"dsv41","prompt":"The capital of France is","max_tokens":2,"temperature":0}' | grep -o '"text":"[^"]*"')
B=$(curl -s -m 300 "$BASE/v1/completions" -H 'Content-Type: application/json' \
     -d '{"model":"dsv41","prompt":"The capital of France is","max_tokens":2,"temperature":0}' | grep -o '"text":"[^"]*"')
echo "第一次: $A"
echo "第二次: $B"
[ "$A" = "$B" ] && echo "✅ 一致" || { echo "❌ 不一致"; FAIL=1; }

echo
if [ "$FAIL" = "0" ]; then
  echo "==== 验证通过 ===="
  exit 0
fi
echo "==== 验证未通过，见上面失败项；服务日志在 /tmp/dsv41_serve.log ===="
exit 1
