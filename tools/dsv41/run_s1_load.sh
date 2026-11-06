#!/usr/bin/env bash
# S1 (reproducible): build the DSV4.1 parameter tree and stream the HF checkpoint into it on CPU.
#
#   bash tools/dsv41/run_s1_load.sh
#
# Verified 2026-11-03 on tileinfer-102: rc=0, 96,042 tensors / 511.7 GB copied,
# 0 shape/dtype mismatches, peak RSS 572.3 GiB (steady 477.8 GiB), 1.4 min with a warm page cache.
# Needs ~580 GiB of RAM: the tree *is* the checkpoint (513.1 GB) plus one shard tensor of transient.
set -euo pipefail

cd "$(dirname "$0")/../.."

export PYTHONDONTWRITEBYTECODE=1
PY="${PY:-/home/userlist/zxs/TileInfer/.venv/bin/python}"
OUT="${OUT:-/home/userlist/zxs/dsv41/s1_load.json}"

exec "$PY" tools/dsv41/s1_load_cpu.py --json-out "$OUT"
