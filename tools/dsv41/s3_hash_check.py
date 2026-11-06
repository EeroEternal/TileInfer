#!/usr/bin/env python3
"""S3 cross-check: our engram hash ids vs the oracle's, on the *real* config and tokenizer.

The tiny parity case can only prove the algorithm on small primes; the real config draws its 48
primes from 16,000,000 and its tables are 384,006,168 / 384,016,682 rows.  This script compares the
two hash-id tensors for a real prompt without touching any weights (a few hundred MB, seconds).

    python tools/dsv41/s3_hash_check.py --prompt "The capital of France is"
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
from tileinfer.models.dsv41.engram import EngramHash  # noqa: E402

M = "/home/models/DeepSeek-V4.1-Flash"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=M)
    parser.add_argument("--prompt", default="The capital of France is")
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer
    from tools.dsv41.s2_layer_check import official_model, oracle_args

    inf = os.path.join(args.model_dir, "inference")
    if inf not in sys.path:
        sys.path.insert(0, inf)
    import engram as official_engram  # noqa: E402

    cfg = Dsv41Config.from_json(os.path.join(args.model_dir, "config.json"))
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    ids = torch.tensor([tok.encode(args.prompt)])

    official = official_model(args.model_dir)
    oargs = oracle_args(official, cfg, 1024, 1)
    layout = official_engram.EngramLayout.from_args(oargs)
    print(
        "official layout:",
        [("layer", lay, "rows", sum(p for per in layout.primes[i] for p in per))
         for i, lay in enumerate(layout.layer_ids)],
    )
    ref = official_engram.NgramHashState(oargs, layout, tok)
    want = ref.forward(ids.clone(), 0, None)

    ours = EngramHash(cfg, tokenizer=tok, max_seq_len=1024, device="cpu")
    got = ours.forward(ids.clone(), 0, None)

    print(f"ids {ids.tolist()}  shape {tuple(got.shape)} (official {tuple(want.shape)})")
    same = torch.equal(got, want)
    if same:
        print("hash ids: IDENTICAL")
        print("first positions:", got[0, :3].tolist())
    else:
        diff = (got != want)
        print(f"hash ids: {int(diff.sum())} of {diff.numel()} differ")
        for layer in range(got.size(2)):
            for pos in range(got.size(1)):
                if not torch.equal(got[0, pos, layer], want[0, pos, layer]):
                    print(f"  layer idx {layer} pos {pos}: ours {got[0,pos,layer].tolist()}"
                          f" official {want[0,pos,layer].tolist()}")
    print("RESULT:", "OK" if same else "MISMATCH")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
