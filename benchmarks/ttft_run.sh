set +e
T=/home/userlist/zxs/TileInfer
NATIVE=$T/.venv_native213/bin/python
export VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1
export TILELANG_CACHE_DIR=/home/userlist/zxs/tl_cache_213
unset TILEINFER_PHASE_TIMING

used_mb () { npu-smi info | grep -o "[0-9]\+  */ 131072" | head -1 | awk '{print $1}'; }
pkill -9 -f "vllm[.]entrypoints" 2>/dev/null
pkill -9 -f "VLLM::EngineCore" 2>/dev/null
for w in $(seq 1 40); do
  sleep 5
  u=$(used_mb); n=$(ps -eo cmd | grep -c "[v]llm.entrypoints")
  [ "$n" = "0" ] && [ -n "$u" ] && [ "$u" -lt 20000 ] && { echo "device free (${u} MB)"; break; }
done

cd "$T" || exit 1
LOG=/tmp/ttft_run.log
: > "$LOG"
nohup "$NATIVE" -m vllm.entrypoints.cli.main serve /home/models/Qwen3.6-35B-A3B \
  --served-model-name qwen36 --attention-backend CUSTOM --enforce-eager \
  --max-model-len 2048 --max-num-batched-tokens 2048 --max-num-seqs 4 \
  --gpu-memory-utilization 0.75 --port 8111 --trust-remote-code >> "$LOG" 2>&1 &
for i in $(seq 1 30); do
  sleep 20
  CODE=$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8111/v1/models)
  [ "$CODE" = "200" ] && { echo "up"; break; }
  grep -q "Engine core initialization failed" "$LOG" && { echo "START-FAILED"; exit 1; }
done
"$NATIVE" $T/benchmarks/ttft.py http://127.0.0.1:8111 native 2>&1 | tail -5
pkill -9 -f "vllm[.]entrypoints" 2>/dev/null
pkill -9 -f "VLLM::EngineCore" 2>/dev/null
echo "########## done ##########"
