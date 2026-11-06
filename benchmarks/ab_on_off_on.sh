set +e
# Does the landed work help END TO END?  ON -> OFF -> ON, one session, with a self-check before each
# arm: no vllm process at all and HBM actually released.  Both defects found in §26 are guarded here.
# The two ON runs bracket the OFF run, so the drift is visible instead of being mistaken for a delta.
T=/home/userlist/zxs/TileInfer
NATIVE=$T/.venv_native213/bin/python
export VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1
export TILELANG_CACHE_DIR=/home/userlist/zxs/tl_cache_213
unset TILEINFER_PHASE_TIMING

hbm () { npu-smi info | sed -n 's/.*\([0-9]\+\) \/ 131072.*/\1/p' | head -1; }
procs () { ps -eo cmd | grep -c "[v]llm.entrypoints"; }

clean () {
  pkill -f "vllm[.]entrypoints" 2>/dev/null; pkill -9 -f "vllm[.]entrypoints" 2>/dev/null
  for i in $(seq 1 40); do
    sleep 5
    local used=$(hbm) n=$(procs)
    if [ "$n" = "0" ] && [ -n "$used" ] && [ "$used" -lt 5000 ]; then
      echo "  clean: 0 procs, hbm ${used} MB after $((i*5))s"; return 0
    fi
  done
  echo "  NOT CLEAN: procs=$(procs) hbm=$(hbm) MB"; return 1
}

run_arm () {
  local label=$1 envs=$2
  clean || return 1
  cd "$T" || return 1
  env $envs nohup "$NATIVE" -m vllm.entrypoints.cli.main serve /home/models/Qwen3.6-35B-A3B \
    --served-model-name qwen36 --attention-backend CUSTOM --enforce-eager \
    --max-model-len 2048 --max-num-batched-tokens 2048 --max-num-seqs 4 \
    --gpu-memory-utilization 0.75 --port 8111 --trust-remote-code >> /tmp/sel_$label.log 2>&1 &
  for i in $(seq 1 24); do
    sleep 30
    CODE=$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8111/v1/models)
    [ "$CODE" = "200" ] && { echo "--- $label up (procs=$(procs) hbm=$(hbm) MB) ---"; break; }
    grep -q "Engine core initialization failed" /tmp/sel_$label.log && { echo "$label START-FAILED"; return 1; }
  done
  "$NATIVE" $T/benchmarks/wall_tpot.py http://127.0.0.1:8111 "$label"
}

echo "########## A: ON (defaults) ##########"
run_arm on1 "TILEINFER_MOE_SORTED=1 TILEINFER_GDN_CACHE=1"
echo "########## B: OFF (both disabled) ##########"
run_arm off "TILEINFER_MOE_SORTED=0 TILEINFER_GDN_CACHE=0"
echo "########## A': ON again ##########"
run_arm on2 "TILEINFER_MOE_SORTED=1 TILEINFER_GDN_CACHE=1"
clean
echo "########## done ##########"
