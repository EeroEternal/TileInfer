set +e
# Wait for the device to actually release HBM instead of sleeping a fixed time.
# Measured failure mode: after killing a server holding ~98 GB (util 0.75 of 131 GB), 45 s is not
# always enough and the next start dies with "No available memory for the cache blocks".
T=/home/userlist/zxs/TileInfer
NATIVE=$T/.venv_native213/bin/python
export VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1
export TILELANG_CACHE_DIR=/home/userlist/zxs/tl_cache_213
unset TILEINFER_PHASE_TIMING

hbm () { npu-smi info | sed -n 's/.*\([0-9]\+\) \/ 131072.*/\1/p' | head -1; }

wait_released () {
  pkill -f "vllm[.]entrypoints" 2>/dev/null
  for i in $(seq 1 40); do
    sleep 5
    local used=$(hbm)
    [ -z "$used" ] && continue
    if [ "$used" -lt 5000 ]; then echo "  hbm released after $((i*5))s (${used} MB)"; return 0; fi
  done
  echo "  hbm STILL HELD: $(hbm) MB"; return 1
}

run_arm () {
  local label=$1 envs=$2
  wait_released || return 1
  cd "$T" || return 1
  env $envs nohup "$NATIVE" -m vllm.entrypoints.cli.main serve /home/models/Qwen3.6-35B-A3B \
    --served-model-name qwen36 --attention-backend CUSTOM --enforce-eager \
    --max-model-len 2048 --max-num-batched-tokens 2048 --max-num-seqs 4 \
    --gpu-memory-utilization 0.75 --port 8111 --trust-remote-code >> /tmp/fix_$label.log 2>&1 &
  for i in $(seq 1 24); do
    sleep 30
    CODE=$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8111/v1/models)
    [ "$CODE" = "200" ] && { echo "--- $label up ---"; break; }
    grep -q "Engine core initialization failed" /tmp/fix_$label.log && { echo "$label START-FAILED"; return 1; }
  done
  "$NATIVE" $T/benchmarks/wall_tpot.py http://127.0.0.1:8111 "$label"
}

echo "########## OFF ##########"
run_arm off "TILEINFER_MOE_SORTED=0 TILEINFER_GDN_CACHE=0"
echo "########## ON ##########"
run_arm on "TILEINFER_MOE_SORTED=1 TILEINFER_GDN_CACHE=1"
pkill -f "vllm[.]entrypoints" 2>/dev/null
echo "########## done ##########"
