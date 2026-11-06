#!/usr/bin/env python3
"""S3 single-layer check at *real* shapes and *real* weights.

The whole-stack run disagreed with the oracle by ~0.5 in the top logits, so the divergence has to be
localized.  Everything was bit-exact on a tiny config, so the suspect is a component that only the
real shapes exercise: 384 routed experts, or the 98 GB engram table.

This loads **one layer** (dense weights + that layer's engram table) into both the official
``Block`` and ours, runs one forward on the same input and compares bit-exactly.

    python tools/dsv41/s3_block_check.py --layer 0     # attention + hc + MoE(384 experts), no engram
    python tools/dsv41/s3_block_check.py --layer 1     # ... plus the engram table

Memory: layer 0 ~8 GB, layer 1 ~106 GB.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tileinfer.models.dsv41 import Dsv41Config, load_shards, shard_paths  # noqa: E402

M = "/home/models/DeepSeek-V4.1-Flash"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=M)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--seqlen", type=int, default=5)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer
    from tools.dsv41.s2_layer_check import copy_weights, official_model, oracle_args

    from tileinfer.models.dsv41.block import ModelState, block_forward, make_identity_pre_mix
    from tileinfer.models.dsv41.modules import Dsv41Block

    torch.set_default_dtype(torch.bfloat16)
    base = Dsv41Config.from_json(os.path.join(args.model_dir, "config.json"))
    cfg = replace(base, num_nextn_predict_layers=0, max_seq_len=64)

    inf = os.path.join(args.model_dir, "inference")
    if inf not in sys.path:
        sys.path.insert(0, inf)
    import engram as official_engram  # noqa: E402

    official = official_model(args.model_dir)
    oargs = oracle_args(official, cfg, 64, 1)
    oargs.n_mtp_layers = 0
    oargs.dspark_block_size = 0
    oargs.vision_n_layers = 1  # only so Gate.bias_vl exists (Block never builds the ViT)
    layout = official_engram.EngramLayout.from_args(oargs)

    print(f"building official Block({args.layer}) + our Dsv41Block")
    ref = official.Block(args.layer, oargs, layout)
    ours = Dsv41Block(cfg, args.layer, device="cpu")

    class _Layers(torch.nn.Module):
        """Gives the standalone Block the checkpoint's `layers.<id>.` name prefix."""

        def __init__(self, block, layer_id):
            super().__init__()
            self.layers = torch.nn.ModuleDict({str(int(layer_id)): block})

    prefix = f"layers.{args.layer}."
    paths = shard_paths(args.model_dir)
    report = load_shards(
        _Layers(ref, args.layer), paths, verify=True, progress_every=5000, log=print,
        name_filter=lambda name: name.startswith(prefix),
    )
    print("  load:", report.summary())
    copy_weights(dict(ref.named_parameters()), dict(ours.named_parameters()))

    state = ModelState(cfg=cfg, max_seq_len=64, bsz=1, device="cpu")
    gen = torch.Generator().manual_seed(args.seed)
    x = torch.randn(1, args.seqlen, cfg.hc_mult, cfg.hidden_size, generator=gen).to(torch.bfloat16)
    pre = make_identity_pre_mix(x, cfg.hc_mult)
    with torch.no_grad():
        want, want_pre = ref.forward(x.clone(), 0, pre.clone(), None)
        got, got_pre = block_forward(ours, x.clone(), 0, pre.clone(), state)
    diff = (got - want).abs()
    print(f"layer {args.layer}: max|diff|={diff.max().item():.3e} mean={diff.mean().item():.3e}")
    print(f"  bit-exact: {torch.equal(got, want)}  pre_mix exact: {torch.equal(got_pre, want_pre)}")
    if not torch.equal(got, want):
        # where does it start? compare the pre-attention collapse (the first hc_pre output)
        flat = x.flatten(2).float()
        from tileinfer.models.dsv41.block import hc_mixes

        pre_a, _, _ = hc_mixes(ref, x, ref.hc_attn_fn, ref.hc_attn_scale, ref.hc_attn_base)
        pre_b, _, _ = hc_mixes(ours, x, ours.hc_attn_fn, ours.hc_attn_scale, ours.hc_attn_base)
        print("  hc attn coefficients exact:", torch.equal(pre_a, pre_b))
        print("  weights exact:",
              all(torch.equal(dict(ref.named_parameters())[n], dict(ours.named_parameters())[n])
                  for n in dict(ref.named_parameters())))
    print("RESULT:", "OK" if torch.equal(got, want) else "MISMATCH")
    return 0 if torch.equal(got, want) else 1


if __name__ == "__main__":
    sys.exit(main())
