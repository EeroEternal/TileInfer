set +e
# One-command paired measurement for ANY candidate switch (§27/§83.4).
#
#   bash benchmarks/pair_switch.sh gdn_conv_kernel
#   bash benchmarks/pair_switch.sh moe_device_path
#
# Starts one server with the file-flag channel enabled, runs paired_tpot.py (which flips the switch
# in-process, alternating order, and rejects physically impossible samples), then cleans up every
# flag file -- a stale `=1` silently changes the next run's baseline, which has bitten this project.
T=/home/userlist/zxs/TileInfer
NATIVE=$T/.venv_native213/bin/python
SWITCH=${1:?usage: pair_switch.sh <switch-name>}
PORT=${2:-8111}
SEQS=${3:-4}

export VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1
export TILELANG_CACHE_DIR=/home/userlist/zxs/tl_cache_213
export TILEINFER_FLAG_WATCH=1 TILEINFER_FLAG_DIR=/tmp/tileinfer_flag
unset TILEINFER_PHASE_TIMING TILEINFER_PHASE_OPS TILEINFER_PREFILL_TIMING TILEINFER_OP_COUNT
unset TILEINFER_ACLGRAPH TILEINFER_COMPILE TILEINFER_DYNAMO_OPAQUE TILEINFER_MOE_DEVICE_PATH

mkdir -p /tmp/tileinfer_flag && rm -f /tmp/tileinfer_flag/*

used_mb () { npu-smi info | grep -o "[0-9]\+  */ 131072" | head -1 | awk '{print $1}'; }
pkill -9 -f "vllm[.]entrypoints" 2>/dev/null
pkill -9 -f "VLLM::EngineCore" 2>/dev/null
for w in $(seq 1 60); do
  sleep 5
  u=$(used_mb); n=$(ps -eo cmd | grep -c "[v]llm.entrypoints")
  [ "$n" = "0" ] && [ -n "$u" ] && [ "$u" -lt 29000 ] && { echo "device free enough (${u} MB)"; break; }
done
if [ "$(ps -eo cmd | grep -c '[v]llm.entrypoints')" != "0" ]; then echo "DEVICE BUSY"; exit 2; fi

cd "$T" || exit 1
LOG=/tmp/pair_${SWITCH}.log
: > "$LOG"
nohup "$NATIVE" -m vllm.entrypoints.cli.main serve /home/models/Qwen3.6-35B-A3B \
  --served-model-name qwen36 --attention-backend CUSTOM --enforce-eager \
  --max-model-len 2048 --max-num-batched-tokens 2048 --max-num-seqs "$SEQS" \
  --gpu-memory-utilization 0.75 --port "$PORT" --trust-remote-code >> "$LOG" 2>&1 &
for i in $(seq 1 30); do
  sleep 20
  CODE=$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/v1/models)
  [ "$CODE" = "200" ] && { echo "up"; break; }
  grep -q "Engine core initialization failed" "$LOG" && { echo "START-FAILED"; exit 1; }
done

echo "===== paired A/B: $SWITCH ====="
"$NATIVE" "$T/benchmarks/paired_tpot.py" "http://127.0.0.1:$PORT" qwen36 "$SWITCH" 2>&1 | tail -8

rm -f /tmp/tileinfer_flag/* 2>/dev/null
pkill -9 -f "vllm[.]entrypoints" 2>/dev/null
pkill -9 -f "VLLM::EngineCore" 2>/dev/null
echo "########## done ##########"
