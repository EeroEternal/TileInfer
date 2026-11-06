set +e
# Decode-side leaf breakdown: what a single 82 ms token is made of, per module kind.
# max_tokens=8 so the decode tables print (20 decode layer-calls = 2 engine steps).
T=/home/userlist/zxs/TileInfer
NATIVE=$T/.venv_native213/bin/python
export VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1
export TILELANG_CACHE_DIR=/home/userlist/zxs/tl_cache_213
export TILEINFER_PHASE_TIMING=device
export TILEINFER_PHASE_LEAF_LIMIT=600
unset TILEINFER_PREFILL
pkill -f "vllm.entrypoints.cli.main serve" 2>/dev/null; sleep 12
LOG=/tmp/tpot_leaf.log
: > "$LOG"
cd "$T" || exit 1
nohup "$NATIVE" -m vllm.entrypoints.cli.main serve /home/models/Qwen3.6-35B-A3B \
  --served-model-name qwen36 --attention-backend CUSTOM --enforce-eager \
  --max-model-len 2048 --max-num-batched-tokens 2048 --max-num-seqs 4 \
  --gpu-memory-utilization 0.75 --port 8111 --trust-remote-code >> "$LOG" 2>&1 &
for i in $(seq 1 24); do
  sleep 30
  CODE=$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8111/v1/models)
  [ "$CODE" = "200" ] && { echo "UP after $((i*30))s"; break; }
  grep -q "Engine core initialization failed" "$LOG" && { echo "START-FAILED"; exit 1; }
done

BODY='{"model":"qwen36","prompt":"Explain in one sentence why the sky is blue:","max_tokens":40,"temperature":0}'
curl -s -m 1800 http://127.0.0.1:8111/v1/completions -H 'Content-Type: application/json' \
  -d "$BODY" -o /dev/null -w "warmup t=%{time_total}s\n"
for r in 1 2 3; do
  curl -s -m 600 http://127.0.0.1:8111/v1/completions -H 'Content-Type: application/json' \
    -d "$BODY" -o /dev/null -w "run$r t=%{time_total}s (40 tokens)\n"
done
echo "--- decode phase table (last, uncontaminated) ---"
grep -A 26 "TileInfer phase timing" "$LOG" | grep -B 1 -A 24 "decode)" | tail -26
pkill -f "vllm.entrypoints.cli.main serve" 2>/dev/null
echo "########## done ##########"
