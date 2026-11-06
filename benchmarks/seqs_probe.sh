set +e
# §8.9.3: throughput stayed at ~18.5 tok/s for conc 4/16/32, which is the signature of
# `--max-num-seqs 4` (only 4 sequences run at a time; the rest queue, hence the 55s/111s TTFT).
# This measures the same scenarios with a wider sequence budget: if throughput rises, the "≥26 tok/s"
# target is a configuration question, not an optimisation question.
T=/home/userlist/zxs/TileInfer
NATIVE=$T/.venv_native213/bin/python
export VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1
export TILELANG_CACHE_DIR=/home/userlist/zxs/tl_cache_213
unset TILEINFER_PHASE_TIMING TILEINFER_PHASE_OPS TILEINFER_PREFILL_TIMING TILEINFER_OP_COUNT
unset TILEINFER_FLAG_WATCH TILEINFER_ACLGRAPH TILEINFER_COMPILE TILEINFER_DYNAMO_OPAQUE
unset TILEINFER_GDN_CONV_KERNEL TILEINFER_MOE_DEVICE_PATH

SEQS=${1:-16}
used_mb () { npu-smi info | grep -o "[0-9]\+  */ 131072" | head -1 | awk '{print $1}'; }
pkill -9 -f "vllm[.]entrypoints" 2>/dev/null
for w in $(seq 1 40); do
  sleep 5
  u=$(used_mb); n=$(ps -eo cmd | grep -c "[v]llm.entrypoints")
  [ "$n" = "0" ] && [ -n "$u" ] && [ "$u" -lt 29000 ] && { echo "device free (${u} MB)"; break; }
done
if [ "$(ps -eo cmd | grep -c '[v]llm.entrypoints')" != "0" ]; then echo "DEVICE BUSY"; exit 2; fi

cd "$T" || exit 1
LOG=/tmp/seqs_$SEQS.log
: > "$LOG"
nohup "$NATIVE" -m vllm.entrypoints.cli.main serve /home/models/Qwen3.6-35B-A3B \
  --served-model-name Qwen3.6-35B-A3B --attention-backend CUSTOM --enforce-eager \
  --max-model-len 2048 --max-num-batched-tokens 2048 --max-num-seqs "$SEQS" \
  --gpu-memory-utilization 0.75 --port 8331 --trust-remote-code >> "$LOG" 2>&1 &
for i in $(seq 1 30); do
  sleep 20
  CODE=$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8331/v1/models)
  [ "$CODE" = "200" ] && { echo "up (max-num-seqs=$SEQS)"; break; }
  grep -q "Engine core initialization failed" "$LOG" && { echo "START-FAILED"; exit 1; }
done

echo "===== conc 4 / 16 / 32 with max-num-seqs=$SEQS ====="
"$NATIVE" "$T/benchmarks/ab_qwen36_full.py" --port 8331 --label "seqs$SEQS" \
  --outdir "/tmp/ab/seqs$SEQS" --runs 1 --conc 4 16 32 > "/tmp/ab_seqs_${SEQS}.log" 2>&1
echo "harness exit=$?"
grep -E 'scenario.*conc_' "/tmp/ab_seqs_${SEQS}.log" | cut -c1-130
pkill -9 -f "vllm[.]entrypoints" 2>/dev/null
echo "########## done ##########"
