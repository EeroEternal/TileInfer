set +e
# Which half of our stack is non-deterministic (§44)?
#
#   A) all-TileInfer attention off : TILEINFER_ATTN_BACKEND=reference + TILEINFER_PREFILL=0
#      -> every attention layer is plain torch, GDN and MoE keep our takeovers
#   B) everything on (the default)
#
# Two captures per configuration against ONE server (a second server cannot fit while the first
# holds ~98 GB -- that mistake cost a start last round).  "tokens identical + delta 0" = deterministic.
#
# Reading the result:
#   A deterministic, B not  -> the cause is in our attention (decode kernel or prefill kernel)
#   A not deterministic     -> the cause is in the GDN chain, MoE, or the sampler
T=/home/userlist/zxs/TileInfer
NATIVE=$T/.venv_native213/bin/python
export VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1
export TILELANG_CACHE_DIR=/home/userlist/zxs/tl_cache_213
unset TILEINFER_PHASE_TIMING

used_mb () { npu-smi info | grep -o "[0-9]\+  */ 131072" | head -1 | awk '{print $1}'; }

wait_free () {
  for w in $(seq 1 60); do
    local u=$(used_mb) n=$(ps -eo cmd | grep -c "[v]llm.entrypoints")
    if [ "$n" = "0" ] && [ -n "$u" ] && [ "$u" -lt 20000 ]; then return 0; fi
    sleep 5
  done
  return 1
}

kill_server () {
  pkill -9 -f "vllm[.]entrypoints" 2>/dev/null
  pkill -9 -f "VLLM::EngineCore" 2>/dev/null
  sleep 25
}

run_cfg () {
  local label=$1 port=$2 envs=$3
  wait_free || { echo "$label: device busy, skipped"; return 1; }
  cd "$T" || return 1
  local log=/tmp/det_$label.log
  : > "$log"
  env $envs nohup "$NATIVE" -m vllm.entrypoints.cli.main serve /home/models/Qwen3.6-35B-A3B \
    --served-model-name qwen36 --attention-backend CUSTOM --enforce-eager \
    --max-model-len 2048 --max-num-batched-tokens 2048 --max-num-seqs 4 \
    --gpu-memory-utilization 0.75 --port "$port" --trust-remote-code >> "$log" 2>&1 &
  for i in $(seq 1 24); do
    sleep 20
    CODE=$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:$port/v1/models)
    [ "$CODE" = "200" ] && { echo "=== $label up ==="; break; }
    grep -q "Engine core initialization failed" "$log" && { echo "$label START-FAILED"; kill_server; return 1; }
  done
  "$NATIVE" $T/benchmarks/capture_logprobs.py "http://127.0.0.1:$port" "/tmp/det_${label}_a.json" 2>&1 | tail -3
  "$NATIVE" $T/benchmarks/capture_logprobs.py "http://127.0.0.1:$port" "/tmp/det_${label}_b.json" 2>&1 | tail -3
  ls -l "/tmp/det_${label}_a.json" >/dev/null 2>&1 || echo "  (capture a produced no file)"
  "$NATIVE" $T/benchmarks/capture_logprobs.py --compare-tokens \
    "/tmp/det_${label}_a.tokens.json" "/tmp/det_${label}_b.tokens.json" | tail -2
  "$NATIVE" $T/benchmarks/ab_report.py --compare "/tmp/det_${label}_a.json" "/tmp/det_${label}_b.json" --atol 1e-3 \
    2>&1 | grep -E "RESULT" | head -2
  kill_server
}

echo "########## A: all-TileInfer attention OFF (torch reference attention) ##########"
run_cfg reference 8111 "TILEINFER_ATTN_BACKEND=reference TILEINFER_PREFILL=0"
echo "########## B: default (kernels on) ##########"
run_cfg kernels 8111 ""
kill_server
echo "########## done ##########"
