#!/usr/bin/env python3
"""S4 step 3c-1: drive the real checkpoint through ``ExpertArena`` on the device.

`tools/dsv41/s4_batched_h2d.py` proved the *approach* (one flat pinned block, 53.6 GB/s, maths
unchanged).  This script proves the *module* -- :class:`tileinfer.models.dsv41.expert_stream.ExpertArena`
-- does the same against real layer-0 experts: pack six experts into the pinned arena, move the whole
arena in one H2D, then recover ``w1`` from the device slots and compare with the CPU reference.

    python tools/dsv41/s4_arena_device.py --experts 6
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
from tileinfer.models.dsv41.expert_stream import ExpertArena, expert_specs  # noqa: E402
from tileinfer.models.dsv41.modules import Dsv41Block  # noqa: E402
from tools.dsv41 import ref_cpu  # noqa: E402

M = "/home/models/DeepSeek-V4.1-Flash"


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
        raise SystemExit("layer did not load cleanly; refusing to benchmark garbage weights")

    experts = block.ffn.experts
    specs = expert_specs(experts[0])
    n = min(args.experts, len(experts))
    arena = ExpertArena(specs=specs, n_slots=n, pin=True)
    print(f"slot {arena.slot_bytes/2**20:.2f} MiB x {n} slots = "
          f"{arena.slot_bytes*n/2**20:.1f} MiB; pinned={arena.host.is_pinned()}")

    t0 = time.perf_counter()
    staging = 0
    for e in range(n):
        slot, hit = arena.bind(args.layer, e)
        staging += arena.pack(slot, experts[e])
        arena.release(args.layer, e)
    print(f"packed {n} experts on the host in {1000*(time.perf_counter()-t0):.2f} ms "
          f"({staging/2**20:.1f} MiB)")

    arena.to_device(dev)  # warm-up: the first call also pays the device allocation
    torch.npu.synchronize()
    t0 = time.perf_counter()
    arena.to_device(dev)
    torch.npu.synchronize()
    big_ms = 1000 * (time.perf_counter() - t0)
    gbs = arena.slot_bytes * n / 1e9 / max(big_ms / 1000, 1e-9)
    print(f"one H2D of the whole arena in {big_ms:.2f} ms = {gbs:.1f} GB/s")

    x = torch.randn(1, args.seqlen, cfg.hidden_size, dtype=torch.bfloat16)
    worst = 0.0
    for e in range(n):
        slot = arena.cache.slot_of((args.layer, e))
        views = arena.views(slot, on_device=True)
        want = ref_cpu.linear(x, experts[e].w1.weight, experts[e].w1.scale).float()
        got = ref_cpu.linear(x.to(dev), views["w1"], views["w1.scale"]).float().cpu()
        d = (got - want).abs().max().item()
        worst = max(worst, d)
        print(f"  expert {e:3d}: slot={slot} views={tuple(views['w1'].shape)} "
              f"{views['w1'].dtype}  max|diff|={d:.3e}")

    print(f"\nworst max|diff|: {worst:.3e}")
    print("stats:", arena.snapshot())
    ok = worst <= 1e-2 and gbs > 20.0
    print("RESULT:", "OK" if ok else "CHECK")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
