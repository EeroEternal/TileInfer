set +e
# Paired in-process A/B: one server, switches flipped by files between requests.
T=/home/userlist/zxs/TileInfer
NATIVE=$T/.venv_native213/bin/python
export VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1
export TILELANG_CACHE_DIR=/home/userlist/zxs/tl_cache_213
unset TILEINFER_PHASE_TIMING
# The whole point of this run: let the bench flip switches inside the server process.
export TILEINFER_FLAG_WATCH=1 TILEINFER_FLAG_DIR=/tmp/tileinfer_flag

hbm () { npu-smi info | sed -n 's/.*\([0-9]\+\) \/ 131072.*/\1/p' | head -1; }
procs () { ps -eo cmd | grep -c "[v]llm.entrypoints"; }

pkill -9 -f "vllm[.]entrypoints" 2>/dev/null
for i in $(seq 1 40); do
  sleep 5
  used=$(hbm); n=$(procs)
  [ "$n" = "0" ] && [ -n "$used" ] && [ "$used" -lt 5000 ] && { echo "clean: hbm ${used} MB after $((i*5))s"; break; }
done

cd "$T" || exit 1
nohup "$NATIVE" -m vllm.entrypoints.cli.main serve /home/models/Qwen3.6-35B-A3B \
  --served-model-name qwen36 --attention-backend CUSTOM --enforce-eager \
  --max-model-len 2048 --max-num-batched-tokens 2048 --max-num-seqs 4 \
  --gpu-memory-utilization 0.75 --port 8111 --trust-remote-code >> /tmp/paired.log 2>&1 &
for i in $(seq 1 24); do
  sleep 30
  CODE=$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8111/v1/models)
  [ "$CODE" = "200" ] && { echo "up"; break; }
  grep -q "Engine core initialization failed" /tmp/paired.log && { echo "START-FAILED"; exit 1; }
done
"$NATIVE" $T/benchmarks/paired_tpot.py http://127.0.0.1:8111 qwen36
pkill -9 -f "vllm[.]entrypoints" 2>/dev/null
echo "########## done ##########"
