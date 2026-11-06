#!/usr/bin/env python3
"""S3: the first greedy token of the *real* 476 GB checkpoint, on CPU.

Runs one prompt through the whole 40-layer stack -- no MTP stages, no vision (S3 is text-only) --
and records the greedy token plus the top-5 logits, so a device run can be compared against it.

Two independent implementations are available, from the same shards and the same prompt:

    --impl official   the read-only oracle (``inference/model.py`` + the CPU kernel transcription)
    --impl ours       ``tileinfer.models.dsv41`` (the tree S1 loads and S2/S3 validated)

They are bit-exact against each other on tiny configs, so either can be the reference; running both
is the honest check for the real shapes.  Memory: ~513 GB for one process -- run them **sequentially**,
never concurrently.

    python tools/dsv41/s3_first_token.py --impl official --out /tmp/s3_official.json
    python tools/dsv41/s3_first_token.py --impl ours     --out /tmp/s3_ours.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tileinfer.models.dsv41 import (  # noqa: E402
    Dsv41Config,
    Dsv41Model,
    load_shards,
    shard_paths,
)

M = "/home/models/DeepSeek-V4.1-Flash"
GB = 1e9


def _status_kb(field: str) -> int:
    try:
        with open("/proc/self/status") as handle:
            for line in handle:
                if line.startswith(field):
                    return int(line.split()[1])
    except OSError:
        pass
    return 0


def _topk(values: torch.Tensor, k: int = 5):
    v, i = torch.topk(values.flatten().float(), k)
    return [[int(idx), float(val)] for idx, val in zip(i.tolist(), v.tolist())]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=M)
    parser.add_argument("--impl", choices=["ours", "official"], default="official")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--out", default="")
    parser.add_argument("--progress-every", type=int, default=20000)
    parser.add_argument("--dump", default="", help="save the final hidden state + full logits here")
    args = parser.parse_args(argv)

    torch.set_default_dtype(torch.bfloat16)
    base = Dsv41Config.from_json(os.path.join(args.model_dir, "config.json"))
    cfg = replace(base, num_nextn_predict_layers=0, max_seq_len=args.max_seq_len)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    ids = tok.encode(args.prompt)
    print(f"impl={args.impl} prompt={args.prompt!r} -> {len(ids)} token ids {ids[:12]}")

    paths = shard_paths(args.model_dir)
    t0 = time.time()
    if args.impl == "official":
        from tools.dsv41.s2_layer_check import official_model, oracle_args

        official = official_model(args.model_dir)
        oargs = oracle_args(official, cfg, args.max_seq_len, 1)
        oargs.n_mtp_layers = 0
        oargs.dspark_block_size = 0
        oargs.vision_n_layers = 0  # text-only: the ViT is out of S3's scope
        print("building the official Transformer (this allocates the 476 GB parameter tree)")
        model = official.Transformer(oargs, tok)
        print(f"  built in {time.time()-t0:.0f}s")
        report = load_shards(
            model, paths, verify=True, progress_every=args.progress_every, log=print
        )
        print(f"  loaded in {time.time()-t0:.0f}s: {report.summary()}")
        captured = {}
        layer_caps = {}

        class _CaptureBlock(torch.nn.Module):
            def __init__(self, inner, index, caps):
                super().__init__()
                self.inner = inner
                self.index = index
                self.caps = caps

            @property
            def engram(self):
                return self.inner.engram

            def hc_pre(self, *a, **kw):
                return self.inner.hc_pre(*a, **kw)

            def forward(self, *a, **kw):
                out = self.inner(*a, **kw)
                self.caps[self.index] = out[0].detach().mean(dim=2).clone()
                return out

        for _i, _layer in enumerate(list(model.layers)):
            model.layers[_i] = _CaptureBlock(_layer, _i, layer_caps)

        class _CaptureNorm(torch.nn.Module):
            def __init__(self, inner, cap):
                super().__init__()
                self.inner = inner
                self.cap = cap

            def forward(self, x):
                self.cap["h"] = x.detach().clone()
                return self.inner(x)

        model.norm = _CaptureNorm(model.norm, captured)
        with torch.no_grad():
            out_ids, logits, _ = model.forward(torch.tensor([ids]), 0)
        token = int(out_ids.flatten()[0])
        logits = logits.float()
    else:
        from tileinfer.models.dsv41.block import ModelState, transformer_forward
        from tileinfer.models.dsv41.engram import EngramHash

        print("building the tileinfer parameter tree")
        model = Dsv41Model(cfg, device="cpu", with_vision=False, with_mtp=False)
        report = load_shards(
            model, paths, verify=True, progress_every=args.progress_every, log=print
        )
        print(f"  loaded in {time.time()-t0:.0f}s: {report.summary()}")
        state = ModelState(
            cfg=cfg,
            max_seq_len=args.max_seq_len,
            bsz=1,
            device="cpu",
            engram_hash=EngramHash(
                cfg, tokenizer=tok, max_seq_len=args.max_seq_len, device="cpu"
            ),
        )
        t1 = time.time()
        captured = {}
        layer_caps = {}
        with torch.no_grad():
            logits, _ = transformer_forward(
                model, torch.tensor([ids]), state, 0, capture=captured,
                capture_layers=layer_caps,
            )
        print(f"  forward in {time.time()-t1:.1f}s")
        logits = logits.float()
        token = int(logits.argmax())

    elapsed = time.time() - t0
    payload = {
        "impl": args.impl,
        "prompt": args.prompt,
        "prompt_ids": ids,
        "token": token,
        "token_text": tok.decode([token]),
        "top5": _topk(logits),
        "logits_shape": list(logits.shape),
        "elapsed_s": elapsed,
        "peak_rss_kb": _status_kb("VmHWM"),
        "final_rss_kb": _status_kb("VmRSS"),
        "load": report.as_dict(),
    }
    print(
        f"\nRESULT impl={args.impl} token={token} ({payload['token_text']!r}) "
        f"top5={payload['top5']}"
    )
    print(f"peak RSS {payload['peak_rss_kb']/1024/1024:.1f} GiB, {elapsed/60:.1f} min")
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(payload, handle, indent=2)
        print(f"wrote {args.out}")
    if args.dump:
        torch.save(layer_caps, args.dump + ".layers")
        print(f"wrote {args.dump}.layers ({len(layer_caps)} layers)")
        h = captured.get("h")
        torch.save(
            {
                "h": None if h is None else h.detach().float().cpu(),
                "logits": logits.detach().float().cpu(),
                "impl": args.impl,
            },
            args.dump,
        )
        print(f"wrote dump {args.dump} (h shape {None if h is None else tuple(h.shape)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
