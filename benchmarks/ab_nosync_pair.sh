set +e
# Paired A/B of the vectorised decode conv (TILEINFER_gdn_no_sync): one reduction instead of
# `width` mul/add rounds, a slice instead of column gathers, and a control-flow-free state update
# (the general form syncs device->host on every call, 30 of them per decode step).
T=/home/userlist/zxs/TileInfer
NATIVE=$T/.venv_native213/bin/python
export VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1
export TILELANG_CACHE_DIR=/home/userlist/zxs/tl_cache_213
export TILEINFER_FLAG_WATCH=1 TILEINFER_FLAG_DIR=/tmp/tileinfer_flag
unset TILEINFER_PHASE_TIMING

for i in $(seq 1 30); do
  npu-smi info | grep -q "No running processes found in NPU 0" && { echo "clean"; break; }
  # a foreign workload (other users' processes) shows up here too; only wait a bounded time
  [ "$i" = "30" ] && { echo "NOTE: npu busy with other processes:"; npu-smi info | sed -n '/Process id/,+4p'; }
  sleep 5
done

cd "$T" || exit 1
nohup "$NATIVE" -m vllm.entrypoints.cli.main serve /home/models/Qwen3.6-35B-A3B \
  --served-model-name qwen36 --attention-backend CUSTOM --enforce-eager \
  --max-model-len 2048 --max-num-batched-tokens 2048 --max-num-seqs 4 \
  --gpu-memory-utilization 0.75 --port 8111 --trust-remote-code >> /tmp/nosync_pair.log 2>&1 &
for i in $(seq 1 24); do
  sleep 30
  CODE=$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8111/v1/models)
  [ "$CODE" = "200" ] && { echo "up"; break; }
  grep -q "Engine core initialization failed" /tmp/nosync_pair.log && { echo "START-FAILED"; exit 1; }
done
"$NATIVE" $T/benchmarks/paired_tpot.py http://127.0.0.1:8111 qwen36 gdn_no_sync
pkill -9 -f "VLLM::EngineCore" 2>/dev/null
pkill -9 -f "vllm[.]entrypoints" 2>/dev/null
echo "########## done ##########"
