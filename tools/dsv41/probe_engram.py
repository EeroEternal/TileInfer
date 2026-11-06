#!/usr/bin/env python3
"""Probe the engram table layout with the *real* tokenizer (S0 evidence, facts §10).

Recomputes, from the official `inference/engram.py` (import only, never modified):
  * the compressed token map size -> must equal `engram_compressed_vocab_size` (99092)
  * the per-(layer, n-gram size, head) prime bucket sizes -> their sum must equal the checkpoint's
    `engram_num_embeddings[layer]` rows
  * the per-layer hash multipliers

    python tools/dsv41/probe_engram.py [MODEL_DIR]
"""
import collections
import json
import sys

M = sys.argv[1] if len(sys.argv) > 1 else "/home/models/DeepSeek-V4.1-Flash"
sys.path.insert(0, f"{M}/inference")

import engram as E  # noqa: E402  (official, read-only)
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(M)
lookup, v = E.build_compressed_token_map(tok)
print("tokenizer len:", len(tok))
print(f"compressed vocab: {v}  (config engram_compressed_vocab_size = 99092)")
print("pad_id(token 2) ->", lookup[2])

cfg = json.load(open(f"{M}/inference/config.json"))


class Args:
    pass


a = Args()
for k, val in cfg.items():
    setattr(a, k, val)

layout = E.EngramLayout.from_args(a)
print("layer_ids:", layout.layer_ids, "max_ngram:", layout.max_ngram_size, "n_heads:", layout.n_heads)
ok = True
for li, layer in enumerate(layout.primes):
    flat = [p for per in layer for p in per]
    rows = layout.num_embeddings[li]
    match = sum(flat) == rows
    ok &= match
    print(f"layer {layout.layer_ids[li]}: {len(flat)} primes, sum={sum(flat)}, "
          f"rows(config)={rows}, match={match}")
    print("   prime range per n-gram size:", [(per[0], per[-1]) for per in layer])

mult = E.compute_hash_multipliers(layout.layer_ids, layout.max_ngram_size, v)
print("multipliers:", mult.tolist())
c = collections.Counter(lookup)
print("distinct compressed ids:", len(c), "| max id:", max(lookup))
print("RESULT:", "OK" if ok and v == cfg["engram_compressed_vocab_size"] else "MISMATCH")
