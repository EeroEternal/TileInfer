#!/usr/bin/env python3
"""S3 layer/segment check at *real* shapes and *real* weights.

The whole-stack run disagreed with the oracle by ~0.5 in the top logits while every component was
bit-exact on a tiny config, so this narrows it down by running a *chain* of real layers (dense weights
plus, where present, the layer's engram table) through both implementations independently -- each side
feeds its own output forward, exactly like the whole-stack run, and every layer's output is compared.

    python tools/dsv41/s3_chain_check.py --layers 2                 # one kv/index source layer
    python tools/dsv41/s3_chain_check.py --layers 2,3,4,5,6,7       # a whole compressed-stream segment
    python tools/dsv41/s3_chain_check.py --layers 20,21,22,23,24   # candidate source + consumer

Memory: 8 GB per layer, plus ~100 GB for a layer that carries an engram table (1 and 14).
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


class _Layers(torch.nn.Module):
    """Gives standalone blocks the checkpoint's ``layers.<id>.`` name prefix."""

    def __init__(self, blocks):
        super().__init__()
        self.layers = torch.nn.ModuleDict({str(int(i)): b for i, b in blocks})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=M)
    parser.add_argument("--layers", default="2")
    parser.add_argument("--seqlen", type=int, default=5)
    parser.add_argument("--decode", type=int, default=0, help="extra single-token decode steps")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--use-engram", action="store_true",
        help="apply the engram contribution before each block, exactly as Transformer.forward does",
    )
    args = parser.parse_args(argv)

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
    oargs.vision_n_layers = 1  # only so Gate.bias_vl exists; no ViT is built by a Block
    layout = official_engram.EngramLayout.from_args(oargs)

    layer_ids = [int(x) for x in args.layers.split(",") if x.strip()]
    print(f"layers {layer_ids}: building official blocks + our blocks")
    refs, ours = {}, {}
    for lid in layer_ids:
        refs[lid] = official.Block(lid, oargs, layout)
        ours[lid] = Dsv41Block(cfg, lid, device="cpu")

    prefixes = tuple(f"layers.{lid}." for lid in layer_ids)
    paths = shard_paths(args.model_dir)
    report = load_shards(
        _Layers([(lid, refs[lid]) for lid in layer_ids]),
        paths,
        verify=True,
        progress_every=5000,
        log=print,
        name_filter=lambda name: name.startswith(prefixes),
    )
    print("  load:", report.summary())
    for lid in layer_ids:
        copy_weights(dict(refs[lid].named_parameters()), dict(ours[lid].named_parameters()))

    state = ModelState(cfg=cfg, max_seq_len=64, bsz=1, device="cpu")
    hash_state = None
    if args.use_engram:
        from transformers import AutoTokenizer

        from tileinfer.models.dsv41.engram import EngramHash

        tok = AutoTokenizer.from_pretrained(args.model_dir)
        hash_state = EngramHash(cfg, tokenizer=tok, max_seq_len=64, device="cpu")
    gen = torch.Generator().manual_seed(args.seed)
    x_ref = torch.randn(1, args.seqlen, cfg.hc_mult, cfg.hidden_size, generator=gen).to(
        torch.bfloat16
    )
    x_ours = x_ref.clone()
    pre_ref = make_identity_pre_mix(x_ref, cfg.hc_mult)
    pre_ours = pre_ref.clone()

    ok = True
    pos = 0
    steps = [args.seqlen] + [1] * args.decode
    for step, seqlen in enumerate(steps):
        if step:
            x_ref = torch.randn(1, 1, cfg.hc_mult, cfg.hidden_size, generator=gen).to(torch.bfloat16)
            x_ours = x_ref.clone()
            pre_ref = make_identity_pre_mix(x_ref, cfg.hc_mult)
            pre_ours = pre_ref.clone()
        with torch.no_grad():
            for lid in layer_ids:
                if args.use_engram and hash_state is not None and refs[lid].engram is not None:
                    from tileinfer.models.dsv41.block import engram_forward

                    ids = torch.randint(
                        0, cfg.vocab_size, (1, args.seqlen), generator=gen
                    )
                    hashes = hash_state.forward(ids, pos, None)
                    idx = refs[lid].engram.layer_hash_index
                    sel = hashes[:, :, idx, :]
                    x_ref = refs[lid].engram(x_ref, sel, None)
                    x_ours = engram_forward(ours[lid].engram, x_ours, sel, None)
                want, want_pre = refs[lid].forward(x_ref.clone(), pos, pre_ref.clone(), None)
                got, got_pre = block_forward(
                    ours[lid], x_ours.clone(), pos, pre_ours.clone(), state
                )
                diff = (got - want).abs()
                same = torch.equal(got, want) and torch.equal(got_pre, want_pre)
                ok = ok and same
                print(
                    f"  step {step} layer {lid:2d}: max|diff|={diff.max().item():.3e} "
                    f"mean={diff.mean().item():.3e} exact={same}"
                )
                x_ref, pre_ref = want, want_pre
                x_ours, pre_ours = got, got_pre
        pos += seqlen
    print("RESULT:", "OK" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
