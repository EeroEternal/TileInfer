set +e
# Which operator inside the gdn family is the 36.8 ms/step?  Split op:gdn into op:gdn:<function>
# with TILEINFER_PHASE_OPS=detail, then read the decode table (dev/call x calls).
T=/home/userlist/zxs/TileInfer
NATIVE=$T/.venv_native213/bin/python
export VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1
export TILELANG_CACHE_DIR=/home/userlist/zxs/tl_cache_213
export TILEINFER_PHASE_TIMING=device
export TILEINFER_PHASE_OPS=detail
export TILEINFER_PHASE_LEAF_LIMIT=0
unset TILEINFER_PREFILL

for i in $(seq 1 24); do
  npu-smi info | grep -q "No running processes found in NPU 0" && break
  sleep 5
done
cd "$T" || exit 1
LOG=/tmp/gdn_split.log
: > "$LOG"
nohup "$NATIVE" -m vllm.entrypoints.cli.main serve /home/models/Qwen3.6-35B-A3B \
  --served-model-name qwen36 --attention-backend CUSTOM --enforce-eager \
  --max-model-len 2048 --max-num-batched-tokens 2048 --max-num-seqs 4 \
  --gpu-memory-utilization 0.75 --port 8111 --trust-remote-code >> "$LOG" 2>&1 &
for i in $(seq 1 24); do
  sleep 30
  CODE=$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8111/v1/models)
  [ "$CODE" = "200" ] && { echo "up"; break; }
  grep -q "Engine core initialization failed" "$LOG" && { echo "START-FAILED"; exit 1; }
done
BODY='{"model":"qwen36","prompt":"Explain in one sentence why the sky is blue:","max_tokens":40,"temperature":0}'
curl -s -m 1800 http://127.0.0.1:8111/v1/completions -H 'Content-Type: application/json' \
  -d "$BODY" -o /dev/null -w "warmup t=%{time_total}s\n"
for r in 1 2 3; do
  curl -s -m 600 http://127.0.0.1:8111/v1/completions -H 'Content-Type: application/json' \
    -d "$BODY" -o /dev/null -w "run$r t=%{time_total}s\n"
done
echo "--- gdn/attention split (last decode table) ---"
grep -A 26 "TileInfer phase timing" "$LOG" | grep -E "op:|attention|model_forward|phase " | tail -22
pkill -9 -f "VLLM::EngineCore" 2>/dev/null
pkill -9 -f "vllm[.]entrypoints" 2>/dev/null
echo "########## done ##########"
