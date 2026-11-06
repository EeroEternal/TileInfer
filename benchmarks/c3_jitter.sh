set +e
# C3's missing control: measure each arm against **itself** before comparing the arms.
#
# Last round native-vs-baseline showed 0.21 nats on a prompt whose tokens matched.  That number only
# means "our numerics differ" if each arm is deterministic over the same 24 greedy tokens.  So:
#   native x2, baseline x2, then native-vs-baseline.
# Each capture is two requests per prompt against the same server, which answers the question that
# matters first (request-level determinism); a server restart is a bigger variable and comes later
# if the same-server answer is "deterministic".
T=/home/userlist/zxs/TileInfer
NATIVE=$T/.venv_native213/bin/python
BASE=/home/dlb/.venv_ascend/bin/python
export VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1
unset TILEINFER_PHASE_TIMING

used_mb () { npu-smi info | grep -o "[0-9]\+  */ 131072" | head -1 | awk '{print $1}'; }

wait_free () {
  for w in $(seq 1 60); do
    local u=$(used_mb) n=$(ps -eo cmd | grep -c "[v]llm.entrypoints")
    if [ "$n" = "0" ] && [ -n "$u" ] && [ "$u" -lt 20000 ]; then
      echo "  device free: ${u} MB used ($((w*5))s)"; return 0
    fi
    sleep 5
  done
  echo "  device busy: $(used_mb) MB"; return 1
}

run_arm () {
  local label=$1 port=$2 py=$3 cache=$4 backend=$5
  wait_free || return 1
  export TILELANG_CACHE_DIR=$cache
  cd "$T" || return 1
  local log=/tmp/jit_$label.log
  : > "$log"                       # truncate: an appended log made a previous run report a stale failure
  local extra=""
  [ -n "$backend" ] && extra="--attention-backend $backend"
  nohup "$py" -m vllm.entrypoints.cli.main serve /home/models/Qwen3.6-35B-A3B \
    --served-model-name qwen36 $extra --enforce-eager \
    --max-model-len 2048 --max-num-batched-tokens 2048 --max-num-seqs 4 \
    --gpu-memory-utilization 0.75 --port "$port" --trust-remote-code >> "$log" 2>&1 &
  for i in $(seq 1 24); do
    sleep 20
    CODE=$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:$port/v1/models)
    [ "$CODE" = "200" ] && { echo "  $label up"; break; }
    grep -q "Engine core initialization failed" "$log" && { echo "  $label START-FAILED"; return 1; }
  done
  "$py" $T/benchmarks/capture_logprobs.py "http://127.0.0.1:$port" "/tmp/jit_${label}_a.json" 2>&1 | tail -2
  "$py" $T/benchmarks/capture_logprobs.py "http://127.0.0.1:$port" "/tmp/jit_${label}_b.json" 2>&1 | tail -2
  pkill -9 -f "vllm[.]entrypoints" 2>/dev/null
  pkill -9 -f "VLLM::EngineCore" 2>/dev/null
  sleep 20
}

cmp () {
  local a=$1 b=$2 label=$3
  echo "--- $label ---"
  "$NATIVE" $T/benchmarks/capture_logprobs.py --compare-tokens "/tmp/jit_${a}.tokens.json" "/tmp/jit_${b}.tokens.json" | tail -3
  "$NATIVE" $T/benchmarks/ab_report.py --compare "/tmp/jit_${a}.json" "/tmp/jit_${b}.json" --atol 1e-3 2>&1 | grep -E "FAIL|ok|RESULT" | head -6
}

echo "########## native ##########"
run_arm native 8111 "$NATIVE" /home/userlist/zxs/tl_cache_213 CUSTOM
echo "########## baseline ##########"
run_arm baseline 8112 "$BASE" /home/userlist/zxs/tl_cache_base213 ""

if [ -f /tmp/jit_native_a.json ] && [ -f /tmp/jit_native_b.json ]; then
  echo "########## native vs native (self-jitter) ##########"
  cmp native_a native_b "native x2"
fi
if [ -f /tmp/jit_baseline_a.json ] && [ -f /tmp/jit_baseline_b.json ]; then
  echo "########## baseline vs baseline (self-jitter) ##########"
  cmp baseline_a baseline_b "baseline x2"
fi
if [ -f /tmp/jit_native_a.json ] && [ -f /tmp/jit_baseline_a.json ]; then
  echo "########## native vs baseline (the C3 question) ##########"
  cmp native_a baseline_a "native vs baseline"
fi
echo "########## done ##########"
