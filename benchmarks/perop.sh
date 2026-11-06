set +e
# Where does a 512-token prefill actually spend its time, by operator?
#
# The §56 arithmetic says the ~0.72 s of marginal prompt cost is GDN (30 layers) plus attention
# (10 layers), with MoE now at 2.55 ms/layer.  Before touching anything, get the per-op table:
# `TILEINFER_PHASE_TIMING=device` + `TILEINFER_PHASE_OPS=detail` labels every installed op
# (`op:<family>:<fn>`) and prints the table periodically.
T=/home/userlist/zxs/TileInfer
NATIVE=$T/.venv_native213/bin/python
export VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1
export TILELANG_CACHE_DIR=/home/userlist/zxs/tl_cache_213
export TILEINFER_PHASE_TIMING=device
export TILEINFER_PHASE_OPS=detail

used_mb () { npu-smi info | grep -o "[0-9]\+  */ 131072" | head -1 | awk '{print $1}'; }
pkill -9 -f "vllm[.]entrypoints" 2>/dev/null
pkill -9 -f "VLLM::EngineCore" 2>/dev/null
for w in $(seq 1 40); do
  sleep 5
  u=$(used_mb); n=$(ps -eo cmd | grep -c "[v]llm.entrypoints")
  [ "$n" = "0" ] && [ -n "$u" ] && [ "$u" -lt 20000 ] && break
done

cd "$T" || exit 1
LOG=/tmp/perop.log
: > "$LOG"
nohup "$NATIVE" -m vllm.entrypoints.cli.main serve /home/models/Qwen3.6-35B-A3B \
  --served-model-name qwen36 --attention-backend CUSTOM --enforce-eager \
  --max-model-len 2048 --max-num-batched-tokens 2048 --max-num-seqs 4 \
  --gpu-memory-utilization 0.75 --port 8111 --trust-remote-code >> "$LOG" 2>&1 &
for i in $(seq 1 30); do
  sleep 20
  CODE=$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8111/v1/models)
  [ "$CODE" = "200" ] && { echo "up"; break; }
  grep -q "Engine core initialization failed" "$LOG" && { echo "START-FAILED"; exit 1; }
done

PROMPT=$(python3 -c "print('Summarize in one word: ' + ('The quick brown fox jumps over the lazy dog while the engineer studies the attention kernel. ' * 8)[:2048])")
BODY=$(python3 -c "import json,sys; print(json.dumps({'model':'qwen36','prompt':sys.argv[1],'max_tokens':4,'temperature':0}))" "$PROMPT")
for i in $(seq 1 8); do
  curl -s -m 600 http://127.0.0.1:8111/v1/completions -H 'Content-Type: application/json' -d "$BODY" -o /dev/null
done
sleep 5
echo "===== last phase table (op:gdn / op:moe / attention) ====="
grep -E "op:gdn|op:moe:|^ *gdn:|attention|model_forward" "$LOG" | tail -40
pkill -9 -f "vllm[.]entrypoints" 2>/dev/null
pkill -9 -f "VLLM::EngineCore" 2>/dev/null
echo "########## done ##########"
