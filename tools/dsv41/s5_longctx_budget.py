#!/usr/bin/env python3
"""S5: does a long context fit? Allocate the real per-layer attention state at 1M and measure.

The prefill cost of the offload path is dominated by expert traffic, so a full 1M-token prefill is not
something to run casually.  What *can* be answered cheaply and exactly is the memory question the
objective asks ("长上下文不 OOM"): build the same per-layer buffers the model allocates for every
sequence length -- window KV, compressed KV, index keys, plus the rope tables -- for a long
``max_seq_len`` and measure the HBM delta.

    python tools/dsv41/s5_longctx_budget.py --max-seq-len 1048576
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tileinfer.models.dsv41 import Dsv41Config  # noqa: E402
from tileinfer.models.dsv41.block import ModelState  # noqa: E402

M = "/home/models/DeepSeek-V4.1-Flash"
GIB = 2**30


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=M)
    parser.add_argument("--max-seq-len", type=int, default=1 << 20)
    parser.add_argument("--bsz", type=int, default=1)
    args = parser.parse_args(argv)

    import torch_npu  # noqa: F401
    from dataclasses import replace

    dev = "npu:0"
    torch.set_default_dtype(torch.bfloat16)
    base = Dsv41Config.from_json(os.path.join(args.model_dir, "config.json"))
    cfg = replace(base, num_nextn_predict_layers=0, max_seq_len=args.max_seq_len)

    free0, total = torch.npu.mem_get_info()
    print(f"start: HBM free {free0/GIB:.2f} GiB / {total/GIB:.2f} GiB")

    state = ModelState(cfg=cfg, max_seq_len=args.max_seq_len, bsz=args.bsz, device=dev)
    rows = []
    for layer_id in range(cfg.num_hidden_layers):
        before = torch.npu.mem_get_info()[0]
        rt = state.runtime(layer_id, cfg.num_hidden_layers)
        torch.npu.synchronize()
        after = torch.npu.mem_get_info()[0]
        used = before - after
        if used > 0:
            ratio = cfg.compress_ratio(layer_id)
            rows.append((layer_id, ratio, used))

    free1, _ = torch.npu.mem_get_info()
    by_ratio = {}
    for layer_id, ratio, used in rows:
        by_ratio.setdefault(ratio, []).append((layer_id, used))
    print(f"\nsequence length {args.max_seq_len} ({args.max_seq_len/1024:.0f}K), bsz={args.bsz}")
    total_state = 0
    for ratio in sorted(by_ratio):
        items = by_ratio[ratio]
        s = sum(u for _, u in items)
        total_state += s
        first, one = items[0]
        print(f"  ratio {ratio}: {len(items)} layers, {s/GIB:.3f} GiB total "
              f"({one/2**20:.1f} MiB/layer, e.g. layer {first})")
    print(f"\nattention state total: {total_state/GIB:.3f} GiB")
    print(f"HBM after: free {free1/GIB:.2f} GiB (delta {(free0-free1)/GIB:.3f} GiB)")
    dense = 10.42  # measured in s4_full_stack_device.py
    headroom = free1 / GIB
    print(f"dense weights {dense:.2f} GiB + attention state {total_state/GIB:.3f} GiB "
          f"→ leaves {headroom - dense:.2f} GiB for the expert arena and activations")
    ok = headroom - dense > 4.0
    print("RESULT:", "OK" if ok else "TIGHT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
