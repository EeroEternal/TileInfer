#!/usr/bin/env python3
"""S0 verification: does the converted MP=1 checkpoint match the official model's tensor set?

Builds the official `inference/model.py` Transformer on the meta device (so the 196 GiB engram
tables cost no memory), then compares every parameter/buffer name, shape and dtype against the
header of the converted single-rank checkpoint.

Expected, checked-in exceptions:
  * `mtp.{0,1,2}.embed.weight` / `.head.weight` are intentionally absent: convert.py drops them
    because each DSpark stage ties them to the backbone's embedding / output head.
  * fp4 tensors: safetensors stores F4_E2M1 logically (two values per byte) while torch packs them
    into `float4_e2m1fn_x2`, so the on-disk last dim is twice the in-memory one.
  * 73 bf16->fp32 upcasts: modules that hold fp32 copies of bf16 checkpoint tensors
    (Compressor wkv/wgate at ratio>1, vision RMSNorms, ParallelHead, DSpark confidence/markov
    heads). `safetensors.load_model` casts on assignment.

    python tools/dsv41/verify_mp1_ckpt.py [CKPT.safetensors] [MODEL_DIR]
"""
import json
import sys

import torch
from safetensors import safe_open

CKPT = sys.argv[1] if len(sys.argv) > 1 else "/home/userlist/zxs/dsv41/mp1/model0-mp1.safetensors"
M = sys.argv[2] if len(sys.argv) > 2 else "/home/models/DeepSeek-V4.1-Flash"

sys.path.insert(0, f"{M}/inference")
from model import ModelArgs, Transformer  # noqa: E402  (official oracle, read-only)
from transformers import AutoTokenizer  # noqa: E402

torch.set_default_dtype(torch.bfloat16)  # generate.py does this before building
cfg = json.load(open(f"{M}/inference/config.json"))
args = ModelArgs(**cfg)
tok = AutoTokenizer.from_pretrained(M)

print(f"building model on meta device: {args.n_layers} layers, {args.n_routed_experts} experts")
with torch.device("meta"):
    model = Transformer(args, tok)

want = dict(model.state_dict().items())
TIED = {f"mtp.{i}.{w}.weight" for i in range(args.n_mtp_layers) for w in ("embed", "head")}

have, dtypes = {}, {}
with safe_open(CKPT, "pt", device="cpu") as f:
    for k in f.keys():
        sl = f.get_slice(k)
        have[k] = tuple(sl.get_shape())
        dtypes[k] = sl.get_dtype()

ST_DTYPE = {
    "bfloat16": "BF16",
    "float16": "F16",
    "float32": "F32",
    "int8": "I8",
    "uint8": "U8",
    "float8_e4m3fn": "F8_E4M3",
    "float8_e8m0fnu": "F8_E8M0",
    "float4_e2m1fn_x2": "F4_E2M1",
}

print(f"model state_dict entries: {len(want)}   checkpoint tensors: {len(have)}")
missing = sorted(set(want) - set(have))
extra = sorted(set(have) - set(want))
unexpected = [n for n in missing if n not in TIED]
bad, dt_bad, upcast = [], [], []
for n in sorted(set(want) & set(have)):
    wshape = tuple(want[n].shape)
    if str(want[n].dtype).startswith("torch.float4"):
        wshape = wshape[:-1] + (wshape[-1] * 2,)  # F4_E2M1 on disk: 2 logical values per byte
    if wshape != have[n]:
        bad.append((n, wshape, have[n]))
    mine = ST_DTYPE.get(str(want[n].dtype).replace("torch.", ""), str(want[n].dtype))
    theirs = dtypes[n].upper()
    same = mine == theirs
    if mine == "F4_E2M1" and theirs.startswith("F4"):
        same = True  # this safetensors build names the 4-bit type "F4"
    if n == "head.weight" and mine == "F32" and theirs == "BF16":
        same = True  # ParallelHead keeps the bf16 checkpoint tensor in fp32 (model.py:1005)
    if not same and theirs == "BF16" and mine == "F32":
        upcast.append(n)
        continue
    if not same:
        dt_bad.append((n, mine, theirs))

print(f"missing: {len(missing)} (expected tied MTP embed/head: {len(missing) - len(unexpected)})")
for n in unexpected[:10]:
    print("   MISSING", n, tuple(want[n].shape))
print(f"extra: {len(extra)}")
for n in extra[:10]:
    print("   EXTRA", n, have[n])
print(f"shape mismatches: {len(bad)}")
for n, a, b in bad[:10]:
    print("   SHAPE", n, "model", a, "ckpt", b)
print(f"bf16->fp32 upcasts (documented, load_model casts): {len(upcast)}")
print(f"dtype mismatches: {len(dt_bad)}")
for t in dt_bad[:10]:
    print("   DTYPE", t)

nbytes = sum(want[n].numel() * want[n].element_size() for n in want if n in have)
print(f"in-memory bytes of matched tensors: {nbytes/1e9:.1f} GB")
ok = not unexpected and not extra and not bad and not dt_bad
print("RESULT:", "OK" if ok else "MISMATCH")
sys.exit(0 if ok else 1)
