set +e
T=/home/userlist/zxs/TileInfer
NATIVE=$T/.venv_native213/bin/python
export VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1
export TILELANG_CACHE_DIR=/home/userlist/zxs/tl_cache_213
export TILEINFER_FLAG_WATCH=1 TILEINFER_FLAG_DIR=/tmp/tileinfer_flag
unset TILEINFER_PHASE_TIMING

# Gate = npu-smi's process table only.  Parsing the HBM column proved fragile (the value sits on the
# second line of the block and the column widths vary), and with an empty process table there is
# nothing left to parse: "No running processes found in NPU 0" is the complete statement.
clean_gate () {
  pkill -9 -f "VLLM::EngineCore" 2>/dev/null
  pkill -9 -f "vllm[.]entrypoints" 2>/dev/null
  for i in $(seq 1 24); do
    sleep 5
    if npu-smi info | grep -q "No running processes found in NPU 0"; then
      echo "  clean: npu idle ($((i*5))s)"; sleep 15; return 0
    fi
  done
  echo "  NOT CLEAN"; npu-smi info | sed -n '/Process id/,+3p'; return 1
}

clean_gate || exit 1
cd "$T" || exit 1
nohup "$NATIVE" -m vllm.entrypoints.cli.main serve /home/models/Qwen3.6-35B-A3B \
  --served-model-name qwen36 --attention-backend CUSTOM --enforce-eager \
  --max-model-len 2048 --max-num-batched-tokens 2048 --max-num-seqs 4 \
  --gpu-memory-utilization 0.75 --port 8111 --trust-remote-code >> /tmp/r31c.log 2>&1 &
for i in $(seq 1 24); do
  sleep 30
  CODE=$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8111/v1/models)
  [ "$CODE" = "200" ] && { echo "up"; break; }
  grep -q "Engine core initialization failed" /tmp/r31c.log && { echo "START-FAILED"; exit 1; }
done
"$NATIVE" $T/benchmarks/paired_tpot.py http://127.0.0.1:8111 qwen36 moe_swiglu
pkill -9 -f "VLLM::EngineCore" 2>/dev/null
pkill -9 -f "vllm[.]entrypoints" 2>/dev/null
echo "########## done ##########"
