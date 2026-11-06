set +e
# Is ACLGraph reachable after all?  §18.8/§23.1 concluded "no" from a log line that appears *because*
# `--enforce-eager` was passed on every run: vLLM maps that flag to cudagraph_mode=NONE.  The platform
# only forces eager when `graphs.capture_path_ready()` is False — and on this stack `graphs.install()`
# makes it True (probe on the device: "aliased torch.cuda.CUDAGraph, torch.cuda.graph,
# torch.cuda.mem_get_info, torch.cuda.Stream; ... after install, ready: True").
#
# So the experiment is a flag change: drop --enforce-eager and ask for decode graphs.
T=/home/userlist/zxs/TileInfer
NATIVE=$T/.venv_native213/bin/python
export VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1
export TILELANG_CACHE_DIR=/home/userlist/zxs/tl_cache_213
unset TILEINFER_PHASE_TIMING TILEINFER_PHASE_OPS TILEINFER_PREFILL_TIMING TILEINFER_OP_COUNT

used_mb () { npu-smi info | grep -o "[0-9]\+  */ 131072" | head -1 | awk '{print $1}'; }
pkill -9 -f "vllm[.]entrypoints" 2>/dev/null
pkill -9 -f "VLLM::EngineCore" 2>/dev/null
for w in $(seq 1 60); do
  sleep 5
  u=$(used_mb); n=$(ps -eo cmd | grep -c "[v]llm.entrypoints")
  [ "$n" = "0" ] && [ -n "$u" ] && [ "$u" -lt 29000 ] && { echo "device free enough (${u} MB used)"; break; }
done

cd "$T" || exit 1
LOG=/tmp/graph.log
: > "$LOG"
nohup "$NATIVE" -m vllm.entrypoints.cli.main serve /home/models/Qwen3.6-35B-A3B \
  --served-model-name qwen36 --attention-backend CUSTOM \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --max-model-len 2048 --max-num-batched-tokens 2048 --max-num-seqs 4 \
  --gpu-memory-utilization 0.75 --port 8111 --trust-remote-code >> "$LOG" 2>&1 &
for i in $(seq 1 30); do
  sleep 20
  CODE=$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8111/v1/models)
  [ "$CODE" = "200" ] && { echo "up"; break; }
  grep -q "Engine core initialization failed" "$LOG" && { echo "START-FAILED"; break; }
done

echo "===== graph-related log lines ====="
grep -iE "cudagraph|graph capture|inductor|aclgraph|tileinfer graph" "$LOG" | tail -10 | cut -c1-190
if [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8111/v1/models)" = "200" ]; then
  echo "===== TPOT ====="
  "$NATIVE" $T/benchmarks/wall_tpot.py http://127.0.0.1:8111 graphs 2>&1 | tail -2
  echo "===== TTFT ====="
  "$NATIVE" $T/benchmarks/ttft.py http://127.0.0.1:8111 graphs 2>&1 | tail -4
fi
pkill -9 -f "vllm[.]entrypoints" 2>/dev/null
pkill -9 -f "VLLM::EngineCore" 2>/dev/null
echo "########## done ##########"
