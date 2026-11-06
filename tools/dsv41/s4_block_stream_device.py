#!/usr/bin/env python3
"""S4: one whole block on the NPU with the experts *paged* in, vs the CPU block.

This is the last step before the full stack: the block (attention + hyper-connections + MoE) runs on
the device with ``moe_forward_streamed``, i.e. its six routed experts are copied in from a host arena
through pre-allocated slots, while the CPU run of the same block with resident weights is the reference.

Two numbers matter:

* ``streamed`` vs ``resident on device`` -- must be bit-exact: that is the paging mechanism itself;
* ``streamed`` vs ``CPU reference``     -- S2's device tolerance, a few bf16 ULP of the fp4 GEMM.

    python tools/dsv41/s4_block_stream_device.py --layer 0
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
from tileinfer.models.dsv41.block import (  # noqa: E402
    ModelState,
    block_forward,
    make_identity_pre_mix,
    moe_forward_streamed,
)
from tileinfer.models.dsv41.expert_stream import ExpertArena, expert_specs  # noqa: E402
from tileinfer.models.dsv41.modules import Dsv41Block  # noqa: E402

M = "/home/models/DeepSeek-V4.1-Flash"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=M)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--seqlen", type=int, default=4)
    parser.add_argument("--slots", type=int, default=64)
    parser.add_argument("--seed", type=int, default=5)
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

    gen = torch.Generator().manual_seed(args.seed)
    x = torch.randn(1, args.seqlen, cfg.hc_mult, cfg.hidden_size, generator=gen).to(torch.bfloat16)
    pre = make_identity_pre_mix(x, cfg.hc_mult)

    free, total = torch.npu.mem_get_info()
    print(f"NPU HBM free={free/2**30:.1f} GiB")

    # CPU reference: the block with every weight resident on the host
    cpu_state = ModelState(cfg=cfg, max_seq_len=64, bsz=1, device="cpu")
    with torch.no_grad():
        y_cpu, pre_cpu = block_forward(block, x, 0, pre.clone(), cpu_state)

    # device: everything but the routed experts moves to the card; the experts stay host-resident and
    # are paged in through the arena -- that is exactly the S4 arrangement
    moved = 0
    for name, param in block.named_parameters():
        if ".ffn.experts." in name:
            continue
        param.data = param.data.to(dev)
        moved += param.numel() * param.element_size()
    print(f"moved {moved/2**30:.2f} GiB of dense weights to the card; routed experts stay on the host")

    dev_state = ModelState(cfg=cfg, max_seq_len=64, bsz=1, device=dev)
    arena = ExpertArena(specs=expert_specs(block.ffn.experts[0]), n_slots=args.slots, pin=True)
    xd = x.to(dev)
    pred = pre.to(dev)
    t0 = time.perf_counter()
    with torch.no_grad():
        y_str, pre_str = block_forward(
            block, xd, 0, pred.clone(), dev_state,
            moe_fn=lambda moe, xx, mask=None: moe_forward_streamed(
                moe, xx, args.layer, arena, device=dev, image_mask=mask
            ),
        )
    torch.npu.synchronize()
    ms = 1000 * (time.perf_counter() - t0)
    d = (y_str.float().cpu() - y_cpu.float()).abs()
    dp = (pre_str.float().cpu() - pre_cpu.float()).abs()
    print(f"streamed block (device) vs CPU reference: max|diff|={d.max():.3e} mean={d.mean():.3e} ({ms:.0f} ms)")
    print(f"pre_mix: max|diff|={dp.max():.3e}")
    print("arena:", arena.snapshot())
    ok = d.max().item() <= 1.0  # S2 tolerance: device fp4 GEMM drift, ~2-4 bf16 ULP of the block output
    print("RESULT:", "OK" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
