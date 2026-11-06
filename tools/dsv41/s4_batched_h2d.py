#!/usr/bin/env python3
"""S4 step 3b: one batched pinned H2D per layer instead of six small copies.

Step 3a showed a 17.9 MiB expert taking 24-244 ms through ``.to(dev)`` per tensor, while the measured
pinned bandwidth (53.7 GB/s) implies ~0.33 ms.  The gap is per-copy overhead, so this script builds the
layer's six experts into one pinned host block, copies it as a single transfer and re-slices it on the
device, then re-checks the maths against the CPU reference.

    python tools/dsv41/s4_batched_h2d.py [--experts 6]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tileinfer.models.dsv41 import Dsv41Config, load_shards, shard_paths  # noqa: E402
from tileinfer.models.dsv41.modules import Dsv41Block  # noqa: E402
from tools.dsv41 import ref_cpu  # noqa: E402

M = "/home/models/DeepSeek-V4.1-Flash"
PARTS = ("w1", "w2", "w3")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=M)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--experts", type=int, default=6)
    parser.add_argument("--seqlen", type=int, default=2)
    args = parser.parse_args(argv)

    import torch_npu  # noqa: F401
    from dataclasses import replace

    dev = "npu:0"
    torch.set_default_dtype(torch.bfloat16)
    base = Dsv41Config.from_json(os.path.join(args.model_dir, "config.json"))
    cfg = replace(
        base,
        num_hidden_layers=args.layer + 1,
        num_nextn_predict_layers=0,
        max_seq_len=64,
        compress_ratios=base.compress_ratios[: args.layer + 1],
    )

    class _Layers(torch.nn.Module):
        def __init__(self, blk, layer_id):
            super().__init__()
            self.layers = torch.nn.ModuleDict({str(int(layer_id)): blk})

    block = Dsv41Block(cfg, args.layer, device="cpu")
    rep = load_shards(
        _Layers(block, args.layer),
        shard_paths(args.model_dir),
        verify=True,
        progress_every=100000,
        name_filter=lambda n: n.startswith(f"layers.{args.layer}."),
    )
    print("load:", rep.summary())
    if rep.missing or rep.extra or rep.shape_mismatch or rep.dtype_mismatch:
        raise SystemExit("layer did not load cleanly")

    experts = block.ffn.experts
    # one expert's flat block = w1|w2|w3 weights + their scales, in a fixed order
    tensors = []
    for name in PARTS:
        tensors.append(getattr(experts[0], name).weight)
        tensors.append(getattr(experts[0], name).scale)
    sizes = [t.numel() * t.element_size() for t in tensors]
    slot_bytes = sum(sizes)

    t0 = time.perf_counter()
    arena = torch.empty(slot_bytes * args.experts, dtype=torch.uint8).pin_memory()
    dev_arena = torch.empty(slot_bytes * args.experts, dtype=torch.uint8, device=dev)
    print(f"pinned+device arena {slot_bytes*args.experts/2**20:.1f} MiB in "
          f"{1000*(time.perf_counter()-t0):.0f} ms")

    t0 = time.perf_counter()
    off = 0
    order = []
    for e in range(args.experts):
        for name in PARTS:
            for t in (getattr(experts[e], name).weight, getattr(experts[e], name).scale):
                n = t.numel() * t.element_size()
                arena[off : off + n].copy_(t.view(torch.uint8).flatten())
                order.append((e, name, off, n, tuple(t.shape), t.dtype))
                off += n
    torch.npu.synchronize()
    print(f"staged {args.experts} experts into pinned arena in "
          f"{1000*(time.perf_counter()-t0):.1f} ms")

    # one transfer, then re-slice on the device
    t0 = time.perf_counter()
    dev_arena.copy_(arena)
    torch.npu.synchronize()
    big_ms = 1000 * (time.perf_counter() - t0)
    gbs = dev_arena.numel() / 1e9 / max(big_ms / 1000, 1e-9)
    print(f"one batched H2D {dev_arena.numel()/2**20:.1f} MiB in {big_ms:.2f} ms = {gbs:.1f} GB/s")

    # per-tensor copies for comparison (what step 3a did)
    t0 = time.perf_counter()
    for e in range(args.experts):
        for name in PARTS:
            getattr(experts[e], name).weight.pin_memory().to(dev)
            getattr(experts[e], name).scale.pin_memory().to(dev)
    torch.npu.synchronize()
    small_ms = 1000 * (time.perf_counter() - t0)
    print(f"{args.experts*6} per-tensor copies in {small_ms:.1f} ms "
          f"({small_ms/(args.experts*6):.2f} ms each)")

    x = torch.randn(1, args.seqlen, cfg.hidden_size, dtype=torch.bfloat16)
    worst = 0.0
    for e in range(args.experts):
        want = ref_cpu.linear(x, experts[e].w1.weight, experts[e].w1.scale).float()
        entry = [o for o in order if o[0] == e and o[1] == "w1"]
        off_w, n_w, shape_w = entry[0][2], entry[0][3], entry[0][4]
        off_s, n_s, shape_s = entry[1][2], entry[1][3], entry[1][4]
        w = dev_arena[off_w : off_w + n_w].view(torch.uint8).view(torch.float4_e2m1fn_x2).reshape(shape_w)
        s = dev_arena[off_s : off_s + n_s].view(torch.uint8).view(torch.float8_e8m0fnu).reshape(shape_s)
        got = ref_cpu.linear(x.to(dev), w, s).float().cpu()
        d = (got - want).abs().max().item()
        worst = max(worst, d)
        print(f"  expert {e}: w1 from arena max|diff|={d:.3e}")
    print(f"\nworst max|diff|: {worst:.3e}")
    print("RESULT:", "OK" if worst <= 1e-2 and gbs > 20 else "CHECK")
    return 0 if worst <= 1e-2 else 1


if __name__ == "__main__":
    sys.exit(main())
