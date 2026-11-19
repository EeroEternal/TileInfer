#!/usr/bin/env bash
# Multi-card preflight + smoke (does not occupy a card by itself).
#
#   scripts/tp-smoke.sh /home/models/Muse-Glimmer-30B 2
#   scripts/tp-smoke.sh /home/models/Qwen3.6-35B-A3B 4
#
# What it does:
#   1. builds a plan for the requested TP size (vendor/route-O gating is enforced here);
#   2. runs `doctor` with the same TILEINFER_TP_SIZE, which now also checks the host HCCL
#      rootinfo path (an empty directory there makes TP>1 fail with RootInfoDetect);
#   3. prints the serve command for the device step.  Starting the server is deliberately
#      not automated: the box is shared and deployments need explicit approval.
#
# The caller must already have sourced the box's environment (CANN, triton-ascend, venv) and
# set TILEINFER_PYTHON if the default `python` is not the repo venv.
set -euo pipefail

MODEL=${1:?usage: scripts/tp-smoke.sh <model-dir> <tp-size> [port]}
TP=${2:?usage: scripts/tp-smoke.sh <model-dir> <tp-size> [port]}
PORT=${3:-8140}

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${TILEINFER_PYTHON:-python}
OUT=${TILEINFER_SMOKE_DIR:-$(mktemp -d)}
mkdir -p "$OUT"

export TILEINFER_TP_SIZE="$TP"
export TILEINFER_PLAN="$OUT/plan.json"
# Keep the smoke's doctor from reading a stale server snapshot from a previous run.
export TILEINFER_METRICS_PATH="$OUT/metrics.json"

"$PY" -m tileinfer.cli.main plan build --model "$MODEL" --out "$TILEINFER_PLAN" --tp "$TP"
"$PY" -m tileinfer.cli.main doctor --json > "$OUT/doctor.json"
"$PY" - "$OUT/doctor.json" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
for check in data["checks"]:
    print(f'{check["status"]:<4} {check["check"]:<14} {check["detail"]}')
sys.exit(0 if data["ok"] else 1)
PY

cat <<EOF

plan:   $TILEINFER_PLAN
doctor: $OUT/doctor.json

Device step (requires approval; pick free cards first with npu-smi):
  ASCEND_RT_VISIBLE_DEVICES=<free cards> \\
  TILEINFER_PLAN=$TILEINFER_PLAN TILEINFER_TP_SIZE=$TP \\
  python -m vllm.entrypoints.openai.api_server \\
    --model $MODEL --tensor-parallel-size $TP --port $PORT
EOF
