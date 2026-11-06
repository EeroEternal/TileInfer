#!/usr/bin/env python3
"""S3 whole-stack check on a *truncated* real model: both implementations side by side, one process.

The full 40-layer runs disagreed on the final hidden state (max|diff| 22) even though every
``Block.forward``, the engram wrapper and the engram hashes are bit-exact in isolation.  So the gap
is in the *stack plumbing* -- embed, the engram application, the final collapse/norm/head -- and this
script isolates it: the same real weights, but only the first ``--layers`` layers, so both models fit
in one process and can be compared directly (no dumps, no second run).

    python tools/dsv41/s3_stack_check.py --layers 2     # embed + 2 layers + engram + tail, ~230 GB
    python tools/dsv41/s3_stack_check.py --layers 3     # adds a compressor/indexer layer
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

from tileinfer.models.dsv41 import Dsv41Config, Dsv41Model, load_shards, shard_paths  # noqa: E402

M = "/home/models/DeepSeek-V4.1-Flash"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=M)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-seq-len", type=int, default=64)
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer
    from tools.dsv41.s2_layer_check import copy_weights, official_model, oracle_args

    from tileinfer.models.dsv41.block import ModelState, transformer_forward
    from tileinfer.models.dsv41.engram import EngramHash

    torch.set_default_dtype(torch.bfloat16)
    base = Dsv41Config.from_json(os.path.join(args.model_dir, "config.json"))
    n = int(args.layers)
    own_kv = tuple(x for x in base.kv_source_layers if x < n)
    own_idx = tuple(x for x in base.index_source_layers if x < n)
    cfg = replace(
        base,
        num_hidden_layers=n,
        num_nextn_predict_layers=0,
        max_seq_len=args.max_seq_len,
        compress_ratios=base.compress_ratios[:n],
        kv_source_layers=own_kv,
        index_source_layers=own_idx,
        candidate_source_layer=base.candidate_source_layer if base.candidate_source_layer < n else -1,
    )
    print(f"truncated config: n_layers={n} kv_src={own_kv} idx_src={own_idx} "
          f"engram={[x for x in cfg.engram_layer_ids if x < n]}")

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    ids = tok.encode(args.prompt)
    print(f"prompt {args.prompt!r} -> {ids}")

    official = official_model(args.model_dir)
    oargs = oracle_args(official, cfg, args.max_seq_len, 1)
    oargs.n_mtp_layers = 0
    oargs.dspark_block_size = 0
    oargs.vision_n_layers = 1  # only for Gate.bias_vl; the ViT is not built for a text stack
    oargs.compress_ratios = cfg.compress_ratios
    oargs.kv_source_layers = own_kv
    oargs.index_source_layers = own_idx
    oargs.candidate_source_layer = cfg.candidate_source_layer

    print("building the official Transformer (truncated) and ours")
    ref = official.Transformer(oargs, tok)
    ours = Dsv41Model(cfg, device="cpu", with_vision=False, with_mtp=False)

    prefixes = tuple(f"layers.{i}." for i in range(n)) + ("embed.", "head.", "norm.")
    paths = shard_paths(args.model_dir)
    rep_ref = load_shards(
        ref, paths, verify=True, progress_every=20000, log=print,
        name_filter=lambda name: name.startswith(prefixes),
    )
    print("  official:", rep_ref.summary())
    # the engram tables are 98 GB each: load them once (into the official model) and let ours share
    # the very same storage, otherwise two 17+ layer stacks do not fit in RAM
    rep_ours = load_shards(
        ours, paths, verify=True, progress_every=20000, log=print,
        name_filter=lambda name: name.startswith(prefixes) and ".engram.embed." not in name,
    )
    for i in range(n):
        if ours.layers[i].engram is None:
            continue
        ours.layers[i].engram.embed.weight = ref.layers[i].engram.embed.weight
        ours.layers[i].engram.embed.scale = ref.layers[i].engram.embed.scale
        print(f"  shared the engram table of layer {i} with the oracle")
    print("  ours    :", rep_ours.summary())
    # both sides loaded the same shards, so no weight copy is needed

    state = ModelState(
        cfg=cfg,
        max_seq_len=args.max_seq_len,
        bsz=1,
        device="cpu",
        engram_hash=EngramHash(cfg, tokenizer=tok, max_seq_len=args.max_seq_len, device="cpu"),
    )
    cap_ref, cap_ours = {}, {}

    class _CaptureNorm(torch.nn.Module):
        def __init__(self, inner, cap):
            super().__init__()
            self.inner = inner
            self.cap = cap

        def forward(self, x):
            self.cap["h"] = x.detach().clone()
            out = self.inner(x)
            self.cap["n"] = out.detach().clone()
            return out

    ref.norm = _CaptureNorm(ref.norm, cap_ref)
    with torch.no_grad():
        want_ids, want_logits, _ = ref.forward(torch.tensor([ids]), 0)
        got_logits, _ = transformer_forward(
            ours, torch.tensor([ids]), state, 0, capture=cap_ours
        )
    want_logits = want_logits.float()
    diff = (got_logits - want_logits).abs()
    href, hours = cap_ref.get("h"), cap_ours.get("h")
    if href is not None and hours is not None:
        dh = (href.float() - hours.float()).abs()
        print(
            f"final h (pre-norm): max|diff|={dh.max().item():.3e} mean={dh.mean().item():.3e} "
            f"exact={torch.equal(href, hours)}"
        )
        print(f"  href {tuple(href.shape)} {href.dtype} | hours {tuple(hours.shape)} {hours.dtype}")
        print(f"  differing elements: {int((href != hours).sum())} of {href.numel()}")
        flat = dh.flatten()
        top = torch.topk(flat, 5)
        for val, idx in zip(top.values.tolist(), top.indices.tolist()):
            print(f"    idx {idx}: ref={float(href.flatten()[idx]):+.6e} "
                  f"ours={float(hours.flatten()[idx]):+.6e} diff={val:.3e}")
        from tileinfer.models.dsv41.layer import rms_norm
        n_ours = rms_norm(ours.norm, hours)
        n_ref = cap_ref.get("n")
        if n_ref is not None:
            dn = (n_ref.float() - n_ours.float()).abs()
            print(f"  norm(h): max|diff|={dn.max().item():.3e} exact={torch.equal(n_ref, n_ours)}")
    print(
        f"\nlogits: max|diff|={diff.max().item():.3e} mean={diff.mean().item():.3e} "
        f"exact={torch.equal(got_logits, want_logits)} "
        f"argmax {int(got_logits.argmax())} vs {int(want_ids.flatten()[0])}"
    )
    for name, vec in (("official", want_logits), ("ours", got_logits)):
        vals, idx = torch.topk(vec.flatten(), 5)
        print(f"  {name:8s} top5: {[(int(i), round(float(v), 3)) for i, v in zip(idx.tolist(), vals.tolist())]}")
    print("RESULT:", "OK" if torch.equal(got_logits, want_logits) else "MISMATCH")
    return 0 if torch.equal(got_logits, want_logits) else 1


if __name__ == "__main__":
    sys.exit(main())
