set +e
# Measure anyway, on a **small** slice of HBM.
#
# The card is shared with another user's server (121 GB of 131).  Our outstanding measurements are
# compute/dispatch questions, not memory questions: a request needs a handful of 2048-token pages,
# so a 6% utilisation (≈8 GB) is enough.  What the sharing costs is absolute throughput -- and that
# is exactly what the paired design cancels, because both arms of a pair see the same contention.
#
# Absolute numbers from this run must NOT be quoted as TPOT (§32.4); only the paired ratios travel.
T=/home/userlist/zxs/TileInfer
NATIVE=$T/.venv_native213/bin/python
export VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1
export TILELANG_CACHE_DIR=/home/userlist/zxs/tl_cache_213
export TILEINFER_FLAG_WATCH=1 TILEINFER_FLAG_DIR=/tmp/tileinfer_flag
unset TILEINFER_PHASE_TIMING

npu-smi info | sed -n '/Process id/,+4p' | tail -3
echo "--- starting our own server on a small slice ---"
cd "$T" || exit 1
LOG=/tmp/shared_slice.log
: > "$LOG"
nohup "$NATIVE" -m vllm.entrypoints.cli.main serve /home/models/Qwen3.6-35B-A3B \
  --served-model-name qwen36 --attention-backend CUSTOM --enforce-eager \
  --max-model-len 2048 --max-num-batched-tokens 2048 --max-num-seqs 4 \
  --gpu-memory-utilization 0.06 --port 8112 --trust-remote-code >> "$LOG" 2>&1 &
UP=0
for i in $(seq 1 24); do
  sleep 15
  CODE=$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8112/v1/models)
  [ "$CODE" = "200" ] && { echo "up after $((i*15))s"; UP=1; break; }
  grep -q "Engine core initialization failed" "$LOG" && { echo "START-FAILED:"; grep -E "ValueError|Error" "$LOG" | grep -viE "libcudart|triton|driver" | tail -3 | cut -c1-160; break; }
done
if [ "$UP" = "1" ]; then
  echo "########## 1/2 gdn_conv_lib (decode, paired) ##########"
  "$NATIVE" $T/benchmarks/paired_tpot.py http://127.0.0.1:8112 qwen36 gdn_conv_lib
  echo "########## 2/2 moe_strict (decode, paired) ##########"
  "$NATIVE" $T/benchmarks/paired_tpot.py http://127.0.0.1:8112 qwen36 moe_strict
  rm -f /tmp/tileinfer_flag/*=1 /tmp/tileinfer_flag/*=0 2>/dev/null
fi
pkill -9 -f "gpu-memory-utilization 0.06" 2>/dev/null
pkill -9 -f "port 8112" 2>/dev/null
echo "########## done ##########"
