#!/usr/bin/env python3
"""Print the per-layer role table of DeepSeek-V4.1-Flash (facts §7.4).

For each of the 40 backbone layers: compress ratio, whether it owns its compressed KV, whether it
owns an indexer, whether the two-level candidate pre-filter is in play, whether it carries an
engram table, and which layer's compressed stream / top-k index list it reuses.

    python tools/dsv41/layer_table.py [MODEL_DIR]
"""
import json
import sys

M = sys.argv[1] if len(sys.argv) > 1 else "/home/models/DeepSeek-V4.1-Flash"
cfg = json.load(open(f"{M}/inference/config.json"))

ratio = cfg["compress_ratios"]
kv = set(cfg["kv_source_layers"])
idx = set(cfg["index_source_layers"])
cand = cfg["candidate_source_layer"]
eng = set(cfg["engram_layer_ids"])

print("| layer | ratio | own_kv | own_idx | cand | engram | kv stream from | topk from |")
print("|---:|---:|:--:|:--:|:--:|:--:|---:|---:|")
for lid in range(cfg["n_layers"]):
    kv_src = max([l for l in kv if l <= lid], default="-")
    idx_src = max([l for l in idx if l <= lid], default="-")
    if lid == cand:
        cand_mark = "src"
    elif cand < lid:
        cand_mark = "use"
    else:
        cand_mark = ""
    print(f"| {lid} | {ratio[lid]} | {'Y' if lid in kv else ''} | {'Y' if lid in idx else ''} | "
          f"{cand_mark} | {'Y' if lid in eng else ''} | {kv_src} | {idx_src} |")
print()
print("MTP layers 40-42 (mtp.0/1/2): ratio =", ratio[cfg["n_layers"]:],
      "-> pure sliding window (DSparkAttention asserts compress_ratio == 0)")
