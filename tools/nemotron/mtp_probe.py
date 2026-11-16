#!/usr/bin/env python3
"""Inspect a safetensors index for MTP / extra-layer weight keys.

Config declaring ``num_nextn_predict_layers=1`` is not evidence that the checkpoint has MTP
heads.  This probe reads only the small ``model.safetensors.index.json`` (never weights) and
reports whether any key lives outside the backbone's 0..num_layers-1 range or is named
mtp/nextn/spec.  A missing result is reported as "not found", not as "MTP unsupported".
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

LAYER = re.compile(r"layers\.(\d+)\.")
NAMED = ("mtp", "nextn", "spec", "eh_proj", "enorm", "hnorm")


def inspect_index(data: dict, num_layers: int) -> dict:
    weight_map = data.get("weight_map") or {}
    keys = list(weight_map)
    named = [k for k in keys if any(token in k.lower() for token in NAMED)]
    indices = []
    for key in keys:
        match = LAYER.search(key)
        if match:
            indices.append(int(match.group(1)))
    max_index = max(indices) if indices else -1
    extra = [k for k in keys if (m := LAYER.search(k)) and int(m.group(1)) >= num_layers]
    return {
        "weight_keys": len(keys),
        "max_layer_index": max_index,
        "named_mtp_keys": named[:20],
        "named_mtp_count": len(named),
        "extra_layer_keys": extra[:20],
        "extra_layer_count": len(extra),
        "mtp_weights_found": bool(named or extra),
        "note": (
            "found" if (named or extra) else "no MTP/extra-layer key in this index; "
            "do not claim MTP without checking the full index"
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", required=True)
    ap.add_argument("--num-layers", type=int, default=52)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    data = json.loads(Path(args.index).read_text(encoding="utf-8"))
    payload = inspect_index(data, args.num_layers)
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
