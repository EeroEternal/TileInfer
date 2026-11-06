set +e
# The last acceptance gap: a same-window two-arm measurement.
#
# The arms cannot overlap (each needs ~92 GiB of a 123 GiB card), so they run sequentially, each with
# its own npu-smi snapshot.  Per arm: start the server, run the single-arm harness
# (benchmarks/ab_qwen36_full.py, 3 runs so `cold` is decidable), capture logprobs for the C3
# comparison, stop.  Then compare the two logprob captures across arms.
#
# NOTE the two arms share one vLLM code tree (this repo is an editable install): the only intended
# difference is the venv and `--attention-backend CUSTOM` (C1).
T=/home/userlist/zxs/TileInfer
NATIVE=$T/.venv_native213/bin/python
BASE=/home/dlb/.venv_ascend/bin/python
export VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1
unset TILEINFER_PHASE_TIMING TILEINFER_PHASE_OPS TILEINFER_PREFILL_TIMING TILEINFER_OP_COUNT
unset TILEINFER_FLAG_WATCH TILEINFER_ACLGRAPH TILEINFER_COMPILE TILEINFER_DYNAMO_OPAQUE
unset TILEINFER_GDN_CONV_KERNEL TILEINFER_MOE_DEVICE_PATH

used_mb () { npu-smi info | grep -o "[0-9]\+  */ 131072" | head -1 | awk '{print $1}'; }

snapshot () {
  echo "--- npu-smi snapshot ($1) ---"
  npu-smi info | grep -E "131072|Process id" -A2 | head -6
}

wait_free () {
  pkill -9 -f "vllm[.]entrypoints" 2>/dev/null
  pkill -9 -f "VLLM::EngineCore" 2>/dev/null
  for w in $(seq 1 60); do
    sleep 5
    u=$(used_mb); n=$(ps -eo cmd | grep -c "[v]llm.entrypoints")
    [ "$n" = "0" ] && [ -n "$u" ] && [ "$u" -lt 29000 ] && { echo "  device free (${u} MB)"; return 0; }
  done
  echo "  DEVICE BUSY ($(used_mb) MB)"; return 1
}

run_arm () {
  local label=$1 port=$2 py=$3 cache=$4 backend=$5
  wait_free || return 1
  snapshot "$label before"
  cd "$T" || return 1
  local log=/tmp/arm_$label.log
  : > "$log"
  local extra=""
  [ -n "$backend" ] && extra="--attention-backend $backend"
  TILELANG_CACHE_DIR=$cache nohup "$py" -m vllm.entrypoints.cli.main serve \
    /home/models/Qwen3.6-35B-A3B --served-model-name Qwen3.6-35B-A3B $extra --enforce-eager \
    --max-model-len 2048 --max-num-batched-tokens 2048 --max-num-seqs 16 \
    --gpu-memory-utilization 0.75 --port "$port" --trust-remote-code >> "$log" 2>&1 &
  for i in $(seq 1 30); do
    sleep 20
    CODE=$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:$port/v1/models)
    [ "$CODE" = "200" ] && { echo "  $label up"; break; }
    grep -q "Engine core initialization failed" "$log" && { echo "  $label START-FAILED"; return 1; }
  done
  TILELANG_CACHE_DIR=$cache "$py" "$T/benchmarks/ab_qwen36_full.py" \
    --port "$port" --label "$label" --outdir "/tmp/ab/$label" --runs 2 > "/tmp/ab_${label}_report.log" 2>&1
  echo "  $label harness exit=$?"
  TILELANG_CACHE_DIR=$cache "$py" "$T/benchmarks/capture_logprobs.py" \
    "http://127.0.0.1:$port" "/tmp/c3_${label}.json" > "/tmp/c3_${label}.log" 2>&1
  echo "  $label logprobs captured"
  pkill -9 -f "vllm[.]entrypoints" 2>/dev/null
  pkill -9 -f "VLLM::EngineCore" 2>/dev/null
  sleep 20
}

echo "########## arm 1: native (CUSTOM) ##########"
run_arm native 8321 "$NATIVE" /home/userlist/zxs/tl_cache_213 CUSTOM
echo "########## arm 2: baseline (vllm-ascend) ##########"
run_arm baseline 8322 "$BASE" /home/userlist/zxs/tl_cache_base213 ""

echo "########## (a) same tokens? ##########"
"$NATIVE" "$T/benchmarks/capture_logprobs.py" --compare-tokens \
  /tmp/c3_native.tokens.json /tmp/c3_baseline.tokens.json 2>&1 | tail -3
echo "########## (b) logprobs (atol 2e-2) ##########"
"$NATIVE" "$T/benchmarks/ab_report.py" --compare /tmp/c3_native.json /tmp/c3_baseline.json \
  --atol 2e-2 2>&1 | grep -E "RESULT|FAIL|ok" | head -8
echo "########## (c) per-arm harness key lines ##########"
grep -hE "ttft_short_1tok|decode_128tok|decode_512tok" /tmp/ab_native_report.log 2>/dev/null | head -4
grep -hE "ttft_short_1tok|decode_128tok|decode_512tok" /tmp/ab_baseline_report.log 2>/dev/null | head -4
pkill -9 -f "vllm[.]entrypoints" 2>/dev/null
echo "########## done ##########"
