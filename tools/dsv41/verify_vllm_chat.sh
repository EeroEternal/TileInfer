#!/bin/bash
# 验证 /v1/chat/completions 能否真实对话（单轮 + 多轮 + 上下文指代）
#
# 用法：  bash tools/dsv41/verify_vllm_chat.sh [port]
#
# 三个设计要点（都是对话特有的坑）：
#   * thinking_mode="chat"：模板默认 "thinking"，会先吐推理再回答；功能验证要关掉，
#     否则你会看到一堆思考过程而不是干净答案（要思考能力时改回 "thinking"）。
#   * 每轮都由客户端带完整历史：本模型在 vLLM 里是"无状态单序列"，服务端不保存会话，
#     这是 OpenAI chat API 的正常用法，但意味着每轮都会重新 prefill 全部历史。
#   * --no-enable-prefix-caching 生效中：没有前缀复用，第 N 轮的延迟随历史长度线性增长。
set -u

PORT="${1:-8122}"
BASE="http://127.0.0.1:${PORT}"
FAIL=0

say() { echo; echo "== $* =="; }

say "1. 健康检查"
CODE=$(curl -s -m 10 -o /dev/null -w '%{http_code}' "$BASE/health")
echo "HTTP ${CODE}"
[ "$CODE" = "200" ] || { echo "!! 服务未就绪"; exit 1; }

say "2. 模型列表"
curl -s -m 10 "$BASE/v1/models" | head -c 260; echo

say '3. 单轮对话：用户问法国首都（thinking_mode=chat，期望含 Paris）'
R1=$(curl -s -m 300 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d '{
  "model":"dsv41",
  "messages":[{"role":"user","content":"What is the capital of France? Answer with just the city name."}],
  "max_tokens":24,"temperature":0,
  "chat_template_kwargs":{"thinking_mode":"chat"}
}')
echo "$R1" | head -c 600; echo
if echo "$R1" | grep -qi "paris"; then echo "✅ PASS"; else echo "❌ FAIL"; FAIL=1; fi

say "4. 多轮对话：客户端带完整历史（验证服务端接受 messages 数组并利用上下文）"
R2=$(curl -s -m 300 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d '{
  "model":"dsv41",
  "messages":[
    {"role":"user","content":"My name is Zhang and I live in Hangzhou."},
    {"role":"assistant","content":"Nice to meet you, Zhang. Hangzhou is a beautiful city."},
    {"role":"user","content":"What is my name and which city do I live in?"}
  ],
  "max_tokens":40,"temperature":0,
  "chat_template_kwargs":{"thinking_mode":"chat"}
}')
echo "$R2" | head -c 700; echo
OK2=1
echo "$R2" | grep -qi "zhang" || { echo "❌ 未复述名字 Zhang"; OK2=0; }
echo "$R2" | grep -qi "hangzhou" || { echo "❌ 未复述城市 Hangzhou"; OK2=0; }
[ "$OK2" = "1" ] && echo "✅ PASS：上下文指代正确" || { echo "❌ FAIL：多轮上下文未生效"; FAIL=1; }

say "5. 系统提示是否生效（system + user）"
R3=$(curl -s -m 300 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d '{
  "model":"dsv41",
  "messages":[
    {"role":"system","content":"You always answer in exactly one word, in uppercase."},
    {"role":"user","content":"capital of France?"}
  ],
  "max_tokens":16,"temperature":0,
  "chat_template_kwargs":{"thinking_mode":"chat"}
}')
echo "$R3" | head -c 500; echo
echo "$R3" | grep -qi "PARIS" && echo "✅ PASS：遵循了 system 指令" || echo "⚠️ 未严格遵循（可能是模型行为，不一定是接口问题）"

say "6. 接口健壮性：超长请求应被拒绝而不是崩溃"
LONG=$(python3 -c "print('a ' * 3000)")
RC=$(curl -s -m 60 -o /tmp/long_resp.json -w '%{http_code}' "$BASE/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"dsv41\",\"messages\":[{\"role\":\"user\",\"content\":\"$LONG\"}],\"max_tokens\":8,\"chat_template_kwargs\":{\"thinking_mode\":\"chat\"}}")
echo "HTTP ${RC}（期望 400，因为超过 max-model-len）"
head -c 200 /tmp/long_resp.json; echo
curl -s -m 10 -o /dev/null -w '健康检查仍然: %{http_code}\n' "$BASE/health"

say "7. 结果"
[ "$FAIL" = "0" ] && echo "==== 对话验证通过 ====" || echo "==== 有失败项，日志 /tmp/dsv41_chat.log ===="
exit "$FAIL"
