#!/usr/bin/env python3
"""Dump the header of every safetensors shard into one JSON: {name: {dtype, shape, bytes, file}}.

Reads only the 8-byte header length + JSON header of each shard, so it is fast (no tensor data)
even for a 510 GB checkpoint and needs no GPU.

    python tools/dsv41/dump_inventory.py [MODEL_DIR] [OUT_JSON]
"""
import glob
import json
import os
import sys

M = sys.argv[1] if len(sys.argv) > 1 else "/home/models/DeepSeek-V4.1-Flash"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/dsv41_inventory.json"

inv = {}
total = 0
for fp in sorted(glob.glob(os.path.join(M, "*.safetensors"))):
    with open(fp, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        hdr = json.loads(f.read(n))
        hdr.pop("__metadata__", None)
        for name, info in hdr.items():
            size = info["data_offsets"][1] - info["data_offsets"][0]
            inv[name] = {
                "dtype": info["dtype"],
                "shape": info["shape"],
                "file": os.path.basename(fp),
                "bytes": size,
            }
            total += size

json.dump(inv, open(OUT, "w"), indent=0)
print(f"shards={len(glob.glob(os.path.join(M, '*.safetensors')))} tensors={len(inv)} "
      f"bytes={total/1e9:.1f}GB -> {OUT}")
