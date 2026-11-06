set +e
# C3, for the first time on real hardware: capture per-token logprobs from both arms over the same
# fixed prompts, then (a) prove the two arms generated the *same* tokens and (b) compare the logprobs.
# (a) has to come first: different token sequences make every logprob differ for a trivial reason.
T=/home/userlist/zxs/TileInfer
NATIVE=$T/.venv_native213/bin/python
BASE=/home/dlb/.venv_ascend/bin/python
export VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1
unset TILEINFER_PHASE_TIMING

run_arm () {
  local label=$1 port=$2 py=$3 cache=$4 backend=$5
  local idle=$(npu-smi info | grep -c "No running processes found in NPU 0")
  if [ "$idle" != "1" ]; then echo "$label: DEVICE BUSY, skipping"; return 1; fi
  pkill -9 -f "vllm[.]entrypoints" 2>/dev/null; pkill -9 -f "VLLM::EngineCore" 2>/dev/null
  sleep 15
  export TILELANG_CACHE_DIR=$cache
  cd "$T" || return 1
  local extra=""
  [ -n "$backend" ] && extra="--attention-backend $backend"
  nohup "$py" -m vllm.entrypoints.cli.main serve /home/models/Qwen3.6-35B-A3B \
    --served-model-name qwen36 $extra --enforce-eager \
    --max-model-len 2048 --max-num-batched-tokens 2048 --max-num-seqs 4 \
    --gpu-memory-utilization 0.75 --port "$port" --trust-remote-code > "/tmp/c3_$label.log" 2>&1 &
  for i in $(seq 1 24); do
    sleep 20
    CODE=$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:$port/v1/models)
    [ "$CODE" = "200" ] && { echo "$label up"; break; }
    grep -q "Engine core initialization failed" "/tmp/c3_$label.log" && { echo "$label START-FAILED"; return 1; }
  done
  "$py" $T/benchmarks/capture_logprobs.py "http://127.0.0.1:$port" "/tmp/c3_$label.json" 2>&1 | tail -6
}

echo "########## native (CUSTOM) ##########"
run_arm native 8111 "$NATIVE" /home/userlist/zxs/tl_cache_213 CUSTOM
echo "########## baseline (vllm-ascend) ##########"
run_arm baseline 8112 "$BASE" /home/userlist/zxs/tl_cache_base213 ""

pkill -9 -f "vllm[.]entrypoints" 2>/dev/null; pkill -9 -f "VLLM::EngineCore" 2>/dev/null

if [ -f /tmp/c3_native.json ] && [ -f /tmp/c3_baseline.json ]; then
  echo "########## (a) same tokens? ##########"
  "$NATIVE" $T/benchmarks/capture_logprobs.py --compare-tokens \
    /tmp/c3_native.tokens.json /tmp/c3_baseline.tokens.json
  echo "########## (b) logprobs ##########"
  "$NATIVE" $T/benchmarks/ab_report.py --compare /tmp/c3_native.json /tmp/c3_baseline.json --atol 2e-2
else
  echo "MISSING CAPTURE: one arm did not produce a file"
fi
echo "########## done ##########"
