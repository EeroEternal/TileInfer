#!/bin/bash
# End-to-end matrix for the native vllm-ascend V4.1 server on port 8124.
# Usage: bash tools/dsv41/perf_native_matrix.sh <tag>
set -u
cd /home/userlist/zxs/TileInfer
TAG=${1:-mmap}
OUT=.work/native30_results
mkdir -p "$OUT"
PORT=8124
PY=.venv_ascend30/bin/python

hbm_sample() {
  local f=$1
  : > "$f"
  while [ ! -f "${f}.stop" ]; do
    npu-smi info 2>/dev/null | grep '131072' >> "$f"
    sleep 0.2
  done
}

run() {
  local tokens=$1 conc=$2 reps=$3 label=$4
  local hbm="$OUT/hbm_${TAG}_${label}.txt"
  hbm_sample "$hbm" &
  local sampler=$!
  "$PY" tools/dsv41/perf_matrix.py \
    --port "$PORT" --tokens "$tokens" --max-tokens 32 \
    --conc "$conc" --reps "$reps" --label "$label" \
    --out "$OUT/${TAG}_${label}.json" 2>&1 | tee -a "$OUT/matrix_${TAG}.log"
  touch "${hbm}.stop"
  wait "$sampler" 2>/dev/null
  "$PY" - "$hbm" <<'EOF'
import re, sys
vals = []
for line in open(sys.argv[1], errors="ignore"):
    m = re.search(r"(\d+)\s+/\s+131072", line)
    if m:
        vals.append(int(m.group(1)))
print(f"  HBM peak = {max(vals)/1024:.2f} GiB" if vals else "  HBM peak = NA")
EOF
}

# cold, single request
for t in 127 2000 3900 15500; do
  run "$t" 1 1 "native_c1_t${t}_${TAG}"
done

# cold, 4 concurrent (P3 shape)
for t in 127 2000 3900; do
  run "$t" 4 1 "native_c4_t${t}_${TAG}"
done

# warm repeat (prefix cache) for 4K/16K single
run 3900 1 3 "native_c1_t3900_warm_${TAG}"
run 15500 1 2 "native_c1_t15500_warm_${TAG}"

# 32K if the prompt fits
run 30000 1 1 "native_c1_t30000_${TAG}"
