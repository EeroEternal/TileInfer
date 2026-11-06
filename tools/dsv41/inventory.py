#!/usr/bin/env python3
"""Summarise a dumped safetensors inventory (see dump_inventory.py) into the fact sheet's tables.

    python tools/dsv41/inventory.py [INV_JSON]

Prints:
  * plain totals (tensor count, bytes, dtype histogram, bytes per dtype)
  * one line per distinct tensor-name pattern (count, total size, the shapes seen)
  * per-layer totals
  * the attention/engram tensor list for the interesting layers (0, 2, 8, 14, 20, 24, 36, 39)
  * the "which group costs how much" table used in docs/notes/dsv41-facts.md §1
"""
import json
import re
import sys
from collections import Counter, defaultdict

INV = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/dsv41_inventory.json"))
out = []


def w(s=""):
    out.append(s)


def kind(name):
    n = re.sub(r"^layers\.\d+\.", "layers.*.", name)
    n = re.sub(r"^mtp\.\d+\.", "mtp.*.", n)
    n = re.sub(r"^vision\.blocks\.\d+\.", "vision.blocks.*.", n)
    return re.sub(r"ffn\.experts\.\d+\.", "ffn.experts.N.", n)


w(f"tensors={len(INV)} bytes={sum(v['bytes'] for v in INV.values())/1e9:.1f}GB "
  f"({sum(v['bytes'] for v in INV.values())/2**30:.1f}GiB)")
w("dtype counts: " + str(dict(Counter(v["dtype"] for v in INV.values()))))
by_dtype = defaultdict(int)
for v in INV.values():
    by_dtype[v["dtype"]] += v["bytes"]
w("bytes per dtype: " + str({k: f"{v/1e9:.1f}GB" for k, v in sorted(by_dtype.items())}))

kinds, shape_of, bytes_of = Counter(), defaultdict(set), defaultdict(int)
for name, v in INV.items():
    k = kind(name)
    kinds[k] += 1
    shape_of[k].add((v["dtype"], tuple(v["shape"])))
    bytes_of[k] += v["bytes"]

w("\n=== distinct name patterns (-> count, shapes, size) ===")
for k in sorted(kinds, key=lambda k: -bytes_of[k]):
    shapes = sorted(shape_of[k])
    s = "; ".join(f"{d}{list(sh)}" for d, sh in shapes[:3])
    if len(shapes) > 3:
        s += f" (+{len(shapes)-3} more)"
    w(f"{k:48s} n={kinds[k]:6d} {bytes_of[k]/1e9:9.3f}GB  {s}")

w("\n=== per-layer total ===")
for lid in range(64):
    pre = f"layers.{lid}."
    tot = sum(v["bytes"] for k, v in INV.items() if k.startswith(pre))
    if tot:
        w(f"layers.{lid}: {tot/1e9:.2f}GB")

w("\n=== attention / engram tensors per interesting layer ===")
for lid in (0, 2, 8, 14, 20, 24, 36, 39):
    w(f"-- layers.{lid} --")
    for k, v in sorted(INV.items()):
        if k.startswith(f"layers.{lid}.attn") or k.startswith(f"layers.{lid}.engram"):
            w(f"   {k[len(f'layers.{lid}.'):]:34s} {v['dtype']:9s} {v['shape']}")

# group table used by docs/notes/dsv41-facts.md §1
def group(name):
    if name.startswith("vision."):
        return "vision tower (ViT, 32 blocks)"
    if name.startswith("aligner."):
        return "aligner (2x Linear)"
    if name in ("embed.weight", "head.weight"):
        return "token embedding / output head"
    if name.startswith("image_"):
        return "image span delimiters"
    if name == "norm.weight":
        return "final norm"
    if name.startswith("mtp."):
        return "DSpark MTP (3 stages, 128 experts each)"
    if ".engram." in name:
        return "engram tables (layers 1, 14)"
    if ".ffn.experts." in name:
        return "routed experts (40 layers x 384)"
    if ".shared_experts." in name:
        return "shared expert (40 layers)"
    if re.match(r"layers\.\d+\.ffn\.gate\.", name):
        return "MoE gate (40 layers)"
    if ".attn.indexer." in name:
        return "indexer (8 index-source layers)"
    if ".attn.compressor." in name:
        return "compressor (4 kv-source layers)"
    if re.match(r"layers\.\d+\.attn\.", name):
        return "attention dense (40 layers)"
    if ".hc_" in name:
        return "hyper-connection mixes (40 layers)"
    if name.endswith("_norm.weight"):
        return "attn/ffn norms (80)"
    return "other"


gb, cnt = defaultdict(float), defaultdict(int)
for name, v in INV.items():
    gb[group(name)] += v["bytes"]
    cnt[group(name)] += 1
total = sum(gb.values())
w("\n=== memory groups ===")
w("| group | tensors | GiB | share |")
w("|---|---:|---:|---:|")
for g in sorted(gb, key=lambda g: -gb[g]):
    w(f"| {g} | {cnt[g]} | {gb[g]/2**30:.1f} | {100*gb[g]/total:.1f}% |")
w(f"| **total** | {sum(cnt.values())} | **{total/2**30:.1f}** | 100% |")

text = "\n".join(out)
open("inventory-report.txt", "w").write(text)
print(text)
