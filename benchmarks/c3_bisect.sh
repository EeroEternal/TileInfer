set +e
# Bisect the non-determinism (§44): our arm differs from itself between identical requests, while the
# baseline is bit-exact.  TILEINFER_OPS holds the *enabled* family names, so:
#   dense  -> our GDN and MoE takeovers off (attention impl stays) : is the attention path clean?
#   moe    -> only MoE on
#   gdn    -> only GDN on
# Two captures per config against the same server; "tokens identical + delta 0" means deterministic.
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

run_cfg () {
  local fam=$1
  wait_free || { echo "$fam: device busy"; return 1; }
  cd "$T" || return 1
  local log=/tmp/bis_$fam.log
  : > "$log"
  TILEINFER_OPS=$fam nohup "$NATIVE" -m vllm.entrypoints.cli.main serve /home/models/Qwen3.6-35B-A3B \
    --served-model-name qwen36 --attention-backend CUSTOM --enforce-eager \
    --max-model-len 2048 --max-num-batched-tokens 2048 --max-num-seqs 4 \
    --gpu-memory-utilization 0.75 --port 8111 --trust-remote-code >> "$log" 2>&1 &
  for i in $(seq 1 24); do
    sleep 20
    CODE=$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8111/v1/models)
    [ "$CODE" = "200" ] && break
    grep -q "Engine core initialization failed" "$log" && { echo "$fam: START-FAILED"; return 1; }
  done
  "$NATIVE" $T/benchmarks/capture_logprobs.py http://127.0.0.1:8111 "/tmp/bis_${fam}_a.json" > /dev/null 2>&1
  "$NATIVE" $T/benchmarks/capture_logprobs.py http://127.0.0.1:8111 "/tmp/bis_${fam}_b.json" > /dev/null 2>&1
  echo "===== TILEINFER_OPS=$fam ====="
  "$NATIVE" $T/benchmarks/capture_logprobs.py --compare-tokens "/tmp/bis_${fam}_a.tokens.json" "/tmp/bis_${fam}_b.tokens.json" | tail -2
  "$NATIVE" $T/benchmarks/ab_report.py --compare "/tmp/bis_${fam}_a.json" "/tmp/bis_${fam}_b.json" --atol 1e-3 2>&1 | grep -E "RESULT|0\.|1e" | head -5
  pkill -9 -f "vllm[.]entrypoints" 2>/dev/null; pkill -9 -f "VLLM::EngineCore" 2>/dev/null
  sleep 20
}

run_cfg dense
run_cfg moe
run_cfg gdn
pkill -9 -f "vllm[.]entrypoints" 2>/dev/null; pkill -9 -f "VLLM::EngineCore" 2>/dev/null
echo "########## done ##########"
