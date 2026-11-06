#!/usr/bin/env bash
# S0 step (reproducible): single-rank checkpoint via the *official* inference/convert.py.
#
#   MODEL=/home/models/DeepSeek-V4.1-Flash OUT=/home/userlist/zxs/dsv41/mp1 \
#       bash tools/dsv41/run_convert_mp1.sh
#
# Verified 2026-11-03: rc=0, output 511,738,545,912 B, peak VmHWM 399,436,876 kB (~399 GB).
# convert.py accumulates every tensor in memory before writing, so peak RSS ~= whole model;
# the box needs >= ~450 GB RAM (it has 628 GB).
set -euo pipefail

M="${MODEL:-/home/models/DeepSeek-V4.1-Flash}"
OUT="${OUT:-/home/userlist/zxs/dsv41/mp1}"
PY="${PY:-/home/userlist/zxs/TileInfer/.venv/bin/python}"

rm -rf "$OUT"
mkdir -p "$OUT"
cd "$M/inference"

"$PY" convert.py \
  --hf-ckpt-path "$M" \
  --save-path "$OUT" \
  --model-parallel 1 \
  --expert-dtype fp4 \
  --tokenizer-path "$M" &
CPID=$!

# sample peak RSS (VmHWM) of the converter while it runs
PEAK=0
while kill -0 "$CPID" 2>/dev/null; do
  R=$(awk '/VmHWM/{print $2}' "/proc/$CPID/status" 2>/dev/null || true)
  if [ -n "${R:-}" ] && [ "$R" -gt "$PEAK" ]; then PEAK=$R; fi
  sleep 10
done
wait "$CPID"
RC=$?
echo "convert_rc=$RC peak_rss_kb=$PEAK out=$OUT"
exit "$RC"
