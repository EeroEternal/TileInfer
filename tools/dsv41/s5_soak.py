#!/usr/bin/env python3
"""S5 soak: mixed-length requests for a configurable duration, with TTFT/TPOT/HBM/RSS series.

One process, one load: dense weights go to the card, the routed experts and both engram tables stay in
host memory and are paged in per layer (the S4 mechanism).  Each request does a prefill plus a few
greedy decode steps; after every request the script samples HBM free/used and the host RSS, so the
result is the curve S5 asks for rather than a single number.

The prompt set mixes a fixed sentence (whose first token is pinned to the oracle's ``11111``, so a
silent corruption shows up immediately) with longer prompts, because the expert fan-out -- and hence
the paging volume -- grows with the prompt.

    python tools/dsv41/s5_soak.py --minutes 30 --out /tmp/s5_soak.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

M = "/home/models/DeepSeek-V4.1-Flash"
PINNED_PROMPT = "The capital of France is"
PINNED_TOKEN = 11111

# increasing length: each entry is (repeat of the base sentence, decode steps)
MIXES = [
    (1, 8),
    (1, 4),
    (2, 4),
    (3, 4),
]


def host_rss_kb() -> int:
    try:
        with open("/proc/self/status") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=M)
    parser.add_argument("--minutes", type=float, default=30.0)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--slots", type=int, default=96)
    parser.add_argument("--out", default="/tmp/s5_soak.json")
    parser.add_argument("--batched", action="store_true", help="use the batched decode MoE")
    args = parser.parse_args(argv)

    import torch_npu  # noqa: F401
    from dataclasses import replace

    from transformers import AutoTokenizer

    from tileinfer.models.dsv41 import Dsv41Config, Dsv41Model, load_shards, shard_paths
    from tileinfer.models.dsv41.block import (
        ModelState,
        moe_forward_batched,
        moe_forward_streamed,
        transformer_forward,
    )
    from tileinfer.models.dsv41.engram import EngramHash
    from tileinfer.models.dsv41.expert_stream import ExpertArena, expert_specs

    dev = "npu:0"
    torch.set_default_dtype(torch.bfloat16)
    base = Dsv41Config.from_json(os.path.join(args.model_dir, "config.json"))
    cfg = replace(base, num_nextn_predict_layers=0, max_seq_len=args.max_seq_len)

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    base_sentence = "The capital of France is Paris. The capital of Germany is Berlin."

    t0 = time.time()
    model = Dsv41Model(cfg, device="cpu", with_vision=False, with_mtp=False)
    rep = load_shards(model, shard_paths(args.model_dir), verify=True, progress_every=50000, log=print)
    print(f"loaded in {time.time()-t0:.0f}s: {rep.summary()}")
    if rep.missing or rep.extra or rep.shape_mismatch or rep.dtype_mismatch:
        raise SystemExit("weights did not load cleanly")

    moved = kept = 0
    for name, param in model.named_parameters():
        if ".ffn.experts." in name or ".engram.embed." in name:
            kept += param.numel() * param.element_size()
            continue
        param.data = param.data.to(dev)
        moved += param.numel() * param.element_size()
    torch.npu.empty_cache()
    print(f"on card {moved/2**30:.2f} GiB | on host {kept/2**30:.2f} GiB")

    arena = ExpertArena(specs=expert_specs(model.layers[0].ffn.experts[0]), n_slots=args.slots, pin=True)
    for i, block in enumerate(model.layers):
        block.ffn._layer_id_for_stream = i

    def streamed_moe(moe, x, image_mask=None):
        layer_id = getattr(moe, "_layer_id_for_stream", 0)
        if args.batched and x.numel() // x.shape[-1] == 1:  # decode uses the batched fast path
            return moe_forward_batched(moe, x, layer_id, arena, device=dev, image_mask=image_mask)
        return moe_forward_streamed(
            moe, x, layer_id, arena, device=dev, image_mask=image_mask
        )

    state = ModelState(
        cfg=cfg,
        max_seq_len=args.max_seq_len,
        bsz=1,
        device=dev,
        engram_hash=EngramHash(cfg, tokenizer=tok, max_seq_len=args.max_seq_len, device="cpu"),
    )

    series = []
    deadline = time.time() + args.minutes * 60.0
    request = 0
    corrupt = 0
    t_start = time.time()
    while time.time() < deadline:
        repeats, steps = MIXES[request % len(MIXES)]
        prompt = PINNED_PROMPT if repeats == 1 and steps == 8 else " ".join([base_sentence] * repeats)
        ids = tok.encode(prompt)
        pos = 0
        t_req = time.perf_counter()
        with torch.no_grad():
            logits, _ = transformer_forward(model, torch.tensor([ids]), state, 0, moe_fn=streamed_moe)
        torch.npu.synchronize()
        ttft = time.perf_counter() - t_req
        first = int(logits.flatten().argmax())
        if prompt == PINNED_PROMPT and first != PINNED_TOKEN:
            corrupt += 1
        out_ids = [first]
        pos = len(ids)
        tpots = []
        for _ in range(steps - 1):
            t1 = time.perf_counter()
            with torch.no_grad():
                logits, _ = transformer_forward(
                    model, torch.tensor([[out_ids[-1]]]), state, pos, moe_fn=streamed_moe
                )
            torch.npu.synchronize()
            tpots.append(time.perf_counter() - t1)
            out_ids.append(int(logits.flatten().argmax()))
            pos += 1
        free, total = torch.npu.mem_get_info()
        row = {
            "request": request,
            "prompt_tokens": len(ids),
            "steps": steps,
            "ttft_ms": round(1000 * ttft, 1),
            "tpot_ms": round(1000 * sum(tpots) / len(tpots), 1) if tpots else None,
            "hbm_used_gib": round((total - free) / 2**30, 2),
            "hbm_free_gib": round(free / 2**30, 2),
            "rss_gib": round(host_rss_kb() / 1024 / 1024, 2),
            "arena_misses": arena.cache.stats.misses,
            "arena_evictions": arena.cache.stats.evictions,
            "arena_failed": arena.cache.stats.failed_acquires,
            "text": tok.decode(out_ids)[:60],
            "elapsed_s": round(time.time() - t_start, 1),
        }
        series.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        request += 1
        if request % 5 == 0:
            with open(args.out, "w") as handle:
                json.dump({"series": series, "corrupt": corrupt}, handle, indent=1)

    with open(args.out, "w") as handle:
        json.dump({"series": series, "corrupt": corrupt}, handle, indent=1)
    ttfts = [r["ttft_ms"] for r in series]
    tpots = [r["tpot_ms"] for r in series if r["tpot_ms"]]
    hbm = [r["hbm_used_gib"] for r in series]
    rss = [r["rss_gib"] for r in series]
    print(
        f"\nrequests={len(series)} corrupt={corrupt} | TTFT {min(ttfts):.0f}-{max(ttfts):.0f} ms "
        f"(mean {sum(ttfts)/len(ttfts):.0f}) | TPOT {min(tpots):.0f}-{max(tpots):.0f} ms "
        f"| HBM used {min(hbm):.1f}-{max(hbm):.1f} GiB | RSS {min(rss):.1f}-{max(rss):.1f} GiB"
    )
    print(f"arena totals: {arena.snapshot()}")
    print(f"wrote {args.out}")
    ok = corrupt == 0 and all(r["arena_failed"] == 0 for r in series)
    print("RESULT:", "OK" if ok else "PROBLEM")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
