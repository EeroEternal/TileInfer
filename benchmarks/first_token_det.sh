set +e
# The decisive 4-minute experiment (§49.2), sized to survive a window that may be killed at any time:
# ONE server start, then the same capture at three generation lengths.
#
#   CAPTURE_MAX_TOKENS=1   -> is the *first* token reproducible?
#        differs  : the non-determinism is in the prefill or the sampler
#        identical: it is accumulated decode numerics, and the length sweep localises the first flip
#   CAPTURE_MAX_TOKENS=4 / 24 -> the same trajectory further out
#
# Each length is captured twice; the token sequences are compared before any logprob number is read.
T=/home/userlist/zxs/TileInfer
NATIVE=$T/.venv_native213/bin/python
export VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1
export TILELANG_CACHE_DIR=/home/userlist/zxs/tl_cache_213
unset TILEINFER_PHASE_TIMING

used_mb () { npu-smi info | grep -o "[0-9]\+  */ 131072" | head -1 | awk '{print $1}'; }

for w in $(seq 1 40); do
  u=$(used_mb); n=$(ps -eo cmd | grep -c "[v]llm.entrypoints")
  if [ "$n" = "0" ] && [ -n "$u" ] && [ "$u" -lt 20000 ]; then echo "device free (${u} MB)"; break; fi
  sleep 5
done
n=$(ps -eo cmd | grep -c "[v]llm.entrypoints")
if [ "$n" != "0" ]; then echo "DEVICE BUSY (someone else's server); aborting"; exit 2; fi

cd "$T" || exit 1
: > /tmp/onetok.log
nohup "$NATIVE" -m vllm.entrypoints.cli.main serve /home/models/Qwen3.6-35B-A3B \
  --served-model-name qwen36 --attention-backend CUSTOM --enforce-eager \
  --max-model-len 2048 --max-num-batched-tokens 2048 --max-num-seqs 4 \
  --gpu-memory-utilization 0.75 --port 8111 --trust-remote-code >> /tmp/onetok.log 2>&1 &
for i in $(seq 1 24); do
  sleep 20
  CODE=$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8111/v1/models)
  [ "$CODE" = "200" ] && { echo "up"; break; }
  grep -q "Engine core initialization failed" /tmp/onetok.log && { echo "START-FAILED"; exit 1; }
done

for N in 1 4 24; do
  CAPTURE_MAX_TOKENS=$N "$NATIVE" $T/benchmarks/capture_logprobs.py http://127.0.0.1:8111 "/tmp/nt${N}_a.json" > /dev/null 2>&1
  CAPTURE_MAX_TOKENS=$N "$NATIVE" $T/benchmarks/capture_logprobs.py http://127.0.0.1:8111 "/tmp/nt${N}_b.json" > /dev/null 2>&1
  echo "===== max_tokens=$N ====="
  "$NATIVE" $T/benchmarks/capture_logprobs.py --compare-tokens "/tmp/nt${N}_a.tokens.json" "/tmp/nt${N}_b.tokens.json" | tail -3
  "$NATIVE" $T/benchmarks/ab_report.py --compare "/tmp/nt${N}_a.json" "/tmp/nt${N}_b.json" --atol 1e-6 2>&1 | grep -E "RESULT" 
done

pkill -9 -f "vllm[.]entrypoints" 2>/dev/null
pkill -9 -f "VLLM::EngineCore" 2>/dev/null
echo "########## done ##########"
