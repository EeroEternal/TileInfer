set +e
# One window, three owed measurements.  Everything here is paired inside a single server process
# (§28: 45% -> 0.7% repeatability), with the §35 rule that a physically impossible sample is
# rejected rather than averaged in.
#
#   1. gdn_conv_lib      decode,  paired, benchmarks/paired_tpot.py
#   2. moe_strict        decode,  paired, benchmarks/paired_tpot.py
#   3. gdn_host_cache    prefill, paired, benchmarks/prefill_pair.py
#
# Preconditions checked before each stage; a stage that cannot run is reported, not silently skipped.
T=/home/userlist/zxs/TileInfer
NATIVE=$T/.venv_native213/bin/python
export VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1
export TILELANG_CACHE_DIR=/home/userlist/zxs/tl_cache_213
export TILEINFER_FLAG_WATCH=1 TILEINFER_FLAG_DIR=/tmp/tileinfer_flag
unset TILEINFER_PHASE_TIMING

idle () { npu-smi info | grep -q "No running processes found in NPU 0"; }

wait_idle () {
  for i in $(seq 1 60); do
    idle && { echo "  device idle after $((i*5))s"; sleep 15; return 0; }
    sleep 5
  done
  echo "  DEVICE BUSY (someone else is using it):"
  npu-smi info | sed -n '/Process id/,+4p'
  return 1
}

start_server () {
  cd "$T" || return 1
  nohup "$NATIVE" -m vllm.entrypoints.cli.main serve /home/models/Qwen3.6-35B-A3B \
    --served-model-name qwen36 --attention-backend CUSTOM --enforce-eager \
    --max-model-len 2048 --max-num-batched-tokens 2048 --max-num-seqs 4 \
    --gpu-memory-utilization 0.75 --port 8111 --trust-remote-code >> /tmp/owing.log 2>&1 &
  for i in $(seq 1 24); do
    sleep 30
    CODE=$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8111/v1/models)
    [ "$CODE" = "200" ] && { echo "  server up"; return 0; }
    grep -q "Engine core initialization failed" /tmp/owing.log && { echo "  START-FAILED"; return 1; }
  done
  echo "  START-TIMEOUT"; return 1
}

wait_idle || { echo "ABORT: device not available"; exit 2; }
: > /tmp/owing.log
start_server || exit 1

# One server, three paired measurements: the switches are flipped through the flag files, so no
# restart is needed between stages.
echo "########## 1/3 gdn_conv_lib (decode) ##########"
"$NATIVE" $T/benchmarks/paired_tpot.py http://127.0.0.1:8111 qwen36 gdn_conv_lib
echo "########## 2/3 moe_strict (decode) ##########"
"$NATIVE" $T/benchmarks/paired_tpot.py http://127.0.0.1:8111 qwen36 moe_strict
echo "########## 3/3 gdn_host_cache (prefill) ##########"
"$NATIVE" $T/benchmarks/prefill_pair.py http://127.0.0.1:8111 qwen36 gdn_host_cache

# Leave no flag override behind: a stale file silently changes the next run's baseline.
rm -f /tmp/tileinfer_flag/*=1 /tmp/tileinfer_flag/*=0 2>/dev/null
pkill -9 -f "VLLM::EngineCore" 2>/dev/null
pkill -9 -f "vllm[.]entrypoints" 2>/dev/null
echo "########## done ##########"
