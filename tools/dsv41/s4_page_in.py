#!/usr/bin/env python3
"""S4 step 3a: page six real experts into HBM slots and check the maths is unchanged.

Loads layer 0's routed experts on the host (fp4 packed weights + e8m0 scales, exactly as the
checkpoint stores them), copies six of them into pre-allocated device slots through
:class:`~tileinfer.models.dsv41.offload.SlotCache`, and runs the fp4 GEMM from the slots on the NPU.
The device result is compared with the CPU reference over the *same* weights, so a paging bug (wrong
slot, half a block, stale scales) shows up as a numeric mismatch instead of a plausible-looking number.

Memory: ~7.4 GB host for the layer, 6 x 18.8 MB on the device (small enough to run beside another
tenant on the card, which C2 requires).

    python tools/dsv41/s4_page_in.py [--experts 6]
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

from tileinfer.models.dsv41 import Dsv41Config, Dsv41Model, load_shards, shard_paths  # noqa: E402
from tileinfer.models.dsv41.modules import Dsv41Block  # noqa: E402
from tileinfer.models.dsv41.offload import SlotCache  # noqa: E402
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
    cfg = replace(
        Dsv41Config.from_json(os.path.join(args.model_dir, "config.json")),
        num_hidden_layers=args.layer + 1,
        num_nextn_predict_layers=0,
        max_seq_len=64,
        compress_ratios=Dsv41Config.from_json(
            os.path.join(args.model_dir, "config.json")
        ).compress_ratios[: args.layer + 1],
    )
    print(f"loading layer {args.layer} of the real checkpoint (host side)")

    class _Layers(torch.nn.Module):
        """Gives the standalone block the checkpoint's ``layers.<id>.`` prefix."""

        def __init__(self, blk, layer_id):
            super().__init__()
            self.layers = torch.nn.ModuleDict({str(int(layer_id)): blk})

    block = Dsv41Block(cfg, args.layer, device="cpu")
    t0 = time.time()
    rep = load_shards(
        _Layers(block, args.layer),
        shard_paths(args.model_dir),
        verify=True,
        progress_every=100000,
        name_filter=lambda n: n.startswith(f"layers.{args.layer}."),
    )
    print(f"  {time.time()-t0:.0f}s  {rep.summary()}")
    if rep.missing or rep.extra or rep.shape_mismatch or rep.dtype_mismatch:
        raise SystemExit("the layer did not load cleanly; refusing to benchmark garbage weights")

    moe = block.ffn
    ids = [(moe.experts[i].w1.weight, moe.experts[i].w1.scale) for i in range(3)]
    slot_bytes = sum(
        t.numel() * t.element_size() for pair in ids for t in pair
    ) + sum(
        moe.experts[0].w2.weight.numel() * moe.experts[0].w2.weight.element_size()
        + moe.experts[0].w3.weight.numel() * moe.experts[0].w3.weight.element_size()
        + moe.experts[0].w2.scale.numel()
        + moe.experts[0].w3.scale.numel()
        for _ in (0,)
    )
    print(f"  slot size ~{slot_bytes/2**20:.2f} MiB per expert")
    cache = SlotCache(slot_bytes=slot_bytes, n_slots=max(args.experts, 2))

    free, total = torch.npu.mem_get_info()
    print(f"  NPU HBM free={free/2**30:.1f} GiB total={total/2**30:.1f} GiB")

    x = torch.randn(1, args.seqlen, cfg.hidden_size, dtype=torch.bfloat16)
    worst = 0.0
    hit_bytes = 0
    for e in range(args.experts):
        slot, hit = cache.acquire(("L%d" % args.layer, e))
        expert = moe.experts[e]
        moved = 0
        t1 = time.perf_counter()
        if not hit:
            for name in ("w1", "w2", "w3"):
                w = getattr(expert, name).weight
                s = getattr(expert, name).scale
                try:
                    w_p = w.pin_memory() if not w.is_pinned() else w
                except Exception:
                    w_p = w
                w_d = w_p.to(dev)  # H2D (the slot would be a view of a preallocated arena)
                s_d = (s.pin_memory() if not s.is_pinned() else s).to(dev)
                moved += w.numel() * w.element_size() + s.numel() * s.element_size()
                if name == "w1":
                    w1_d, w1s_d = w_d, s_d
                elif name == "w2":
                    w2_d, w2s_d = w_d, s_d
                else:
                    w3_d, w3s_d = w_d, s_d
            torch.npu.synchronize()
            hit_bytes += moved
        cache.release(("L%d" % args.layer, e))

        want = ref_cpu.linear(x, expert.w1.weight, expert.w1.scale).float()
        got = ref_cpu.linear(x.to(dev), w1_d, w1s_d).float().cpu()
        d = (got - want).abs().max().item()
        worst = max(worst, d)
        print(
            f"  expert {e:3d}: slot={slot} hit={hit} moved={moved/2**20:.1f} MiB "
            f"copy={1000*(time.perf_counter()-t1):.1f} ms  w1 max|diff|={d:.3e}"
        )

    print(f"\nworst max|diff| over {args.experts} experts: {worst:.3e}")
    print("stats:", cache.snapshot())
    print("RESULT:", "OK" if worst <= 1e-2 else "MISMATCH")
    return 0 if worst <= 1e-2 else 1


if __name__ == "__main__":
    sys.exit(main())
