set +e
# Settle the 10x question of §60: is a 512-token prefill attention step ~3 ms (isolated benchmark) or
# ~30 ms (the phase table)?  `TILEINFER_PREFILL_TIMING=1` makes PrefillPath time each step itself —
# no phase framework, no device events, just perf_counter around the launch and then around a
# synchronise, so the two numbers separate host issue cost from real device cost.
#
# One request at a time (--max-num-seqs 1, serial sends) so a batch cannot blur the per-call figure.
T=/home/userlist/zxs/TileInfer
NATIVE=$T/.venv_native213/bin/python
export VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1
export TILELANG_CACHE_DIR=/home/userlist/zxs/tl_cache_213
export TILEINFER_PREFILL_TIMING=1
unset TILEINFER_PHASE_TIMING TILEINFER_PHASE_OPS

used_mb () { npu-smi info | grep -o "[0-9]\+  */ 131072" | head -1 | awk '{print $1}'; }
pkill -9 -f "vllm[.]entrypoints" 2>/dev/null
pkill -9 -f "VLLM::EngineCore" 2>/dev/null
for w in $(seq 1 60); do
  sleep 5
  u=$(used_mb); n=$(ps -eo cmd | grep -c "[v]llm.entrypoints")
  [ "$n" = "0" ] && [ -n "$u" ] && [ "$u" -lt 20000 ] && { echo "device free (${u} MB)"; break; }
done
if [ "$(ps -eo cmd | grep -c '[v]llm.entrypoints')" != "0" ]; then echo "DEVICE BUSY"; exit 2; fi

cd "$T" || exit 1
LOG=/tmp/ptime.log
: > "$LOG"
nohup "$NATIVE" -m vllm.entrypoints.cli.main serve /home/models/Qwen3.6-35B-A3B \
  --served-model-name qwen36 --attention-backend CUSTOM --enforce-eager \
  --max-model-len 2048 --max-num-batched-tokens 2048 --max-num-seqs 1 \
  --gpu-memory-utilization 0.75 --port 8111 --trust-remote-code >> "$LOG" 2>&1 &
for i in $(seq 1 30); do
  sleep 20
  CODE=$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8111/v1/models)
  [ "$CODE" = "200" ] && { echo "up"; break; }
  grep -q "Engine core initialization failed" "$LOG" && { echo "START-FAILED"; exit 1; }
done

PROMPT=$(python3 -c "print('Summarize in one word: ' + ('The quick brown fox jumps over the lazy dog while the engineer studies the attention kernel. ' * 8)[:2048])")
BODY=$(python3 -c "import json,sys; print(json.dumps({'model':'qwen36','prompt':sys.argv[1],'max_tokens':4,'temperature':0}))" "$PROMPT")
for i in $(seq 1 10); do
  curl -s -m 600 http://127.0.0.1:8111/v1/completions -H 'Content-Type: application/json' -d "$BODY" -o /dev/null
done
sleep 3
echo "===== PrefillPath self-timing (launch vs device-inclusive) ====="
grep "prefill-timing" "$LOG" | tail -6
pkill -9 -f "vllm[.]entrypoints" 2>/dev/null
pkill -9 -f "VLLM::EngineCore" 2>/dev/null
echo "########## done ##########"
