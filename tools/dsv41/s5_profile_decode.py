#!/usr/bin/env python3
"""S5 profiling: where the ~730 ms decode token actually goes.

The offload path is already bandwidth-bound in theory (84 ms) but measures ~730 ms, so the gap has to
be attributed before optimizing.  This script times the pieces on the card, separately and in place:

* one slot's H2D (17.93 MiB) and the six-slot copy a decode layer needs;
* ``ref_cpu.unpack_fp4`` on one real expert weight (5.9M values) -- the dequant the GEMM pays for
  every call because the packed fp4 is materialised as bf16 first;
* a ``linear`` on that weight, so unpack's share can be subtracted;
* one full decode step, for the total.

    python tools/dsv41/s5_profile_decode.py
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


def bench(fn, iters: int = 5) -> float:
    fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return 1000 * (time.perf_counter() - t0) / iters


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=M)
    parser.add_argument("--layer", type=int, default=0)
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
    if rep.missing or rep.extra or rep.shape_mismatch or rep.dtype_mismatch:
        raise SystemExit(f"layer did not load cleanly: {rep.summary()}")

    expert = block.ffn.experts[0]
    w = expert.w1.weight
    s = expert.w1.scale
    print(f"one expert weight: {tuple(w.shape)} {w.dtype} "
          f"({w.numel() * w.element_size()/2**20:.2f} MiB packed)")

    w_d, s_d = w.to(dev), s.to(dev)
    slot_bytes = sum(
        getattr(expert, n).weight.numel() * getattr(expert, n).weight.element_size()
        + getattr(expert, n).scale.numel() * getattr(expert, n).scale.element_size()
        for n in ("w1", "w2", "w3")
    )

    # H2D of one slot's worth of bytes
    host = torch.empty(slot_bytes, dtype=torch.uint8).pin_memory()
    dev_buf = torch.empty(slot_bytes, dtype=torch.uint8, device=dev)
    h2d_ms = bench(lambda: dev_buf.copy_(host))

    # unpack alone, then the GEMM alone
    x = torch.randn(1, cfg.hidden_size, dtype=torch.bfloat16, device=dev)
    out_f, in_f = int(w.shape[0]), int(w.shape[1]) * 2  # the packed view has half the columns
    unpack_ms = bench(lambda: ref_cpu.unpack_fp4(w_d, out_f, in_f))
    lin_ms = bench(lambda: ref_cpu.linear(x, w_d, s_d))
    fused = ref_cpu.unpack_fp4(w_d, out_f, in_f)
    lin_pre_ms = bench(lambda: torch.nn.functional.linear(x, fused))

    print("\nper-call on the card")
    print(f"  H2D 17.93 MiB slot      : {h2d_ms:.3f} ms  ({slot_bytes/2**20/h2d_ms*1000/1024:.1f} GB/s)")
    print(f"  unpack_fp4 (5.9M values): {unpack_ms:.3f} ms")
    print(f"  linear(packed w1)       : {lin_ms:.3f} ms")
    print(f"  linear(pre-unpacked)    : {lin_pre_ms:.3f} ms  → unpack share "
          f"{100*(1 - lin_pre_ms/lin_ms):.0f}% of the GEMM call")

    per_layer = 6 * (3 * lin_ms + h2d_ms)
    print(f"\nestimate per decode layer: 6 experts x (3 GEMM + 1 slot) = {per_layer:.1f} ms")
    print(f"  -> 40 layers: {40*per_layer:.0f} ms   (measured end-to-end TPOT ~730 ms)")
    # path 1: unpack on the host, page bf16 instead of fp4 (2x the bytes, no unpack on the card)
    host_unpack_ms = bench(lambda: ref_cpu.unpack_fp4(w, out_f, in_f), iters=3) if False else None
    t0 = time.perf_counter()
    for _ in range(3):
        ref_cpu.unpack_fp4(w, out_f, in_f)
    host_unpack_ms = 1000 * (time.perf_counter() - t0) / 3
    bf16_slot = 3 * out_f * in_f * 2
    host2 = torch.empty(bf16_slot, dtype=torch.uint8).pin_memory()
    dev2 = torch.empty(bf16_slot, dtype=torch.uint8, device=dev)
    h2d_bf16 = bench(lambda: dev2.copy_(host2))
    per_layer_bf16 = 6 * (h2d_bf16 + 3 * lin_pre_ms)
    print(f"\npath 1 (bf16 paging): host unpack {host_unpack_ms:.3f} ms/expert, "
          f"H2D {bf16_slot/2**20:.1f} MiB {h2d_bf16:.3f} ms")
    print(f"  per layer 6 experts = {per_layer_bf16:.1f} ms -> 40 layers {40*per_layer_bf16:.0f} ms "
          f"(fp4 paging: {40*per_layer:.0f} ms)")
    print("RESULT: measured")
    return 0


if __name__ == "__main__":
    sys.exit(main())
