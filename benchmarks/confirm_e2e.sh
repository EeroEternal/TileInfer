set +e
# End-to-end confirmation of the §54 prediction: TPOT ~111 ms (from 122) and TTFT@512 ~1.5 s (from 1.9).
# One native server, defaults on.  bench_arm gives TTFT/throughput, wall_tpot gives the decode ruler.
T=/home/userlist/zxs/TileInfer
NATIVE=$T/.venv_native213/bin/python
export VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1
export TILELANG_CACHE_DIR=/home/userlist/zxs/tl_cache_213
unset TILEINFER_PHASE_TIMING

used_mb () { npu-smi info | grep -o "[0-9]\+  */ 131072" | head -1 | awk '{print $1}'; }
for w in $(seq 1 40); do
  u=$(used_mb); n=$(ps -eo cmd | grep -c "[v]llm.entrypoints")
  if [ "$n" = "0" ] && [ -n "$u" ] && [ "$u" -lt 20000 ]; then echo "device free (${u} MB)"; break; fi
  sleep 5
done
if [ "$(ps -eo cmd | grep -c '[v]llm.entrypoints')" != "0" ]; then echo "DEVICE BUSY"; exit 2; fi

cd "$T" || exit 1
LOG=/tmp/confirm.log
: > "$LOG"
nohup "$NATIVE" -m vllm.entrypoints.cli.main serve /home/models/Qwen3.6-35B-A3B \
  --served-model-name qwen36 --attention-backend CUSTOM --enforce-eager \
  --max-model-len 2048 --max-num-batched-tokens 2048 --max-num-seqs 4 \
  --gpu-memory-utilization 0.75 --port 8111 --trust-remote-code >> "$LOG" 2>&1 &
for i in $(seq 1 24); do
  sleep 20
  CODE=$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8111/v1/models)
  [ "$CODE" = "200" ] && { echo "up"; break; }
  grep -q "Engine core initialization failed" "$LOG" && { echo "START-FAILED"; exit 1; }
done

curl -s -m 1800 http://127.0.0.1:8111/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen36","prompt":"warm up the engine","max_tokens":8,"temperature":0}' -o /dev/null -w "warmup t=%{time_total}s\n"
echo "===== TTFT / throughput (bench_arm) ====="
"$NATIVE" $T/benchmarks/bench_arm.py http://127.0.0.1:8111 confirm /tmp/confirm.json
echo "===== decode ruler (wall_tpot) ====="
"$NATIVE" $T/benchmarks/wall_tpot.py http://127.0.0.1:8111 confirm | tail -2

pkill -9 -f "vllm[.]entrypoints" 2>/dev/null
pkill -9 -f "VLLM::EngineCore" 2>/dev/null
echo "########## done ##########"
