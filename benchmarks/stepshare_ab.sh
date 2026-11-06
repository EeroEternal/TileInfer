set +e
# Same-env A/B of the per-step ragged-triple sharing (TILEINFER_ATTN_STEP_SHARE).
# Streaming TPOT is unreliable for this backend (§16), so wall TPOT is derived from
# (total - ttft) / (chunks - 1) out of bench_arm's own JSON.
T=/home/userlist/zxs/TileInfer
NATIVE=$T/.venv_native213/bin/python
export VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1
export TILELANG_CACHE_DIR=/home/userlist/zxs/tl_cache_213
unset TILEINFER_PHASE_TIMING

run_arm () {
  local label=$1 share=$2
  pkill -f "vllm.entrypoints.cli.main serve" 2>/dev/null; sleep 12
  local log=/tmp/stepshare_$label.log
  : > "$log"
  cd "$T" || return 1
  TILEINFER_ATTN_STEP_SHARE=$share nohup "$NATIVE" -m vllm.entrypoints.cli.main serve \
    /home/models/Qwen3.6-35B-A3B --served-model-name qwen36 --attention-backend CUSTOM \
    --enforce-eager --max-model-len 2048 --max-num-batched-tokens 2048 --max-num-seqs 4 \
    --gpu-memory-utilization 0.75 --port 8111 --trust-remote-code >> "$log" 2>&1 &
  for i in $(seq 1 24); do
    sleep 30
    CODE=$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8111/v1/models)
    [ "$CODE" = "200" ] && { echo "$label UP"; break; }
    grep -q "Engine core initialization failed" "$log" && { echo "$label START-FAILED"; return 1; }
  done
  curl -s -m 1800 http://127.0.0.1:8111/v1/completions -H 'Content-Type: application/json' \
    -d '{"model":"qwen36","prompt":"warm up","max_tokens":8,"temperature":0}' -o /dev/null -w "warmup t=%{time_total}s\n"
  "$NATIVE" /tmp/bench_arm.py http://127.0.0.1:8111 "$label" /tmp/stepshare_$label.json > /dev/null
  python3 - "$label" <<'PY'
import json, sys
label = sys.argv[1]
rows = json.load(open(f"/tmp/stepshare_{label}.json"))
for r in rows:
    n = r["chunks_median"]
    wall = (r["total_median_s"] - r["ttft_median_s"]) / max(1.0, n - 1)
    print(f"[{label}] prompt~{r['prompt_tokens_req']}: TTFT {r['ttft_median_s']:.4f}s  "
          f"wall-TPOT {wall*1000:.1f} ms  ({1/wall:.2f} tok/s)")
PY
  grep -E "shared step buffers" "$log" | tail -1 | cut -c1-140
}

echo "########## share=0 ##########"
run_arm off 0
echo "########## share=1 ##########"
run_arm on 1
pkill -f "vllm.entrypoints.cli.main serve" 2>/dev/null
echo "########## done ##########"
