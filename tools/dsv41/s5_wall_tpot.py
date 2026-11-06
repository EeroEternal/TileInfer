#!/usr/bin/env python3
"""S5: TPOT by differencing, as ``docs/notes/perf-comparison-rules.md`` requires.

The rules are explicit: a per-token number must come from
``tpot = (T(N2) - T(N1)) / (N2 - N1)`` with ``N1=8, N2=72``, the same prompt, non-streaming, three
repetitions, median -- because prompt cost, first-token cost and per-request overhead all appear in both
terms and cancel.  Per-step timing (what ``s4_full_stack_device.py --gen`` prints) is the "chunk delta"
style the document rejects, so the numbers it reports are indicative only.

The result file records the environment first (versions + ``npu-smi`` temperature/power), per the same
rules, and refuses non-physical pairs (``T(N2) < T(N1)``).

    python tools/dsv41/s5_wall_tpot.py --n1 8 --n2 72 --reps 3
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

M = "/home/models/DeepSeek-V4.1-Flash"
PROMPT = "The capital of France is"


def environment() -> dict:
    """What the rules require in the header of every result file."""
    info: dict = {}
    try:
        import vllm

        info["vllm"] = vllm.__version__
    except Exception:
        info["vllm"] = None
    info["torch"] = torch.__version__
    try:
        import torch_npu

        info["torch_npu"] = torch_npu.__version__
    except Exception:
        info["torch_npu"] = None
    try:
        out = subprocess.run(
            ["npu-smi", "info"], capture_output=True, text=True, timeout=30
        ).stdout
        info["npu_smi_excerpt"] = [
            line.strip() for line in out.splitlines() if "|" in line and ("C" in line or "W" in line)
        ][:4]
    except Exception:
        info["npu_smi_excerpt"] = []
    import platform

    info["python"] = platform.python_version()
    return info


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=M)
    parser.add_argument("--n1", type=int, default=8)
    parser.add_argument("--n2", type=int, default=72)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--slots", type=int, default=400)
    parser.add_argument("--streaming", action="store_true", help="A/B arm: streaming MoE instead of batched")
    parser.add_argument("--out", default="/tmp/s5_wall_tpot.json")
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
    env = environment()
    print("environment:", json.dumps({k: v for k, v in env.items() if k != "npu_smi_excerpt"}))
    for line in env["npu_smi_excerpt"]:
        print("  ", line)

    base = Dsv41Config.from_json(os.path.join(args.model_dir, "config.json"))
    cfg = replace(base, num_nextn_predict_layers=0, max_seq_len=1024)
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    ids = tok.encode(PROMPT)

    t0 = time.time()
    model = Dsv41Model(cfg, device="cpu", with_vision=False, with_mtp=False)
    rep = load_shards(model, shard_paths(args.model_dir), verify=True, progress_every=50000, log=print)
    print(f"loaded in {time.time()-t0:.0f}s: {rep.summary()}")
    if rep.missing or rep.extra or rep.shape_mismatch or rep.dtype_mismatch:
        raise SystemExit("weights did not load cleanly")

    for name, param in model.named_parameters():
        if ".ffn.experts." in name or ".engram.embed." in name:
            continue
        param.data = param.data.to(dev)
    torch.npu.empty_cache()

    arena = ExpertArena(specs=expert_specs(model.layers[0].ffn.experts[0]), n_slots=args.slots, pin=True)
    for i, block in enumerate(model.layers):
        block.ffn._layer_id_for_stream = i

    def moe_fn(moe, x, image_mask=None):
        layer_id = getattr(moe, "_layer_id_for_stream", 0)
        if not args.streaming and x.numel() // x.shape[-1] == 1:
            return moe_forward_batched(moe, x, layer_id, arena, device=dev, image_mask=image_mask)
        return moe_forward_streamed(moe, x, layer_id, arena, device=dev, image_mask=image_mask)

    state = ModelState(
        cfg=cfg,
        max_seq_len=1024,
        bsz=1,
        device=dev,
        engram_hash=EngramHash(cfg, tokenizer=tok, max_seq_len=1024, device="cpu"),
    )

    def run(n_tokens: int) -> tuple[float, list[int]]:
        """One non-streaming request: prefill + n_tokens greedy decodes, wall clock."""
        t = time.perf_counter()
        with torch.no_grad():
            logits, _ = transformer_forward(
                model, torch.tensor([ids]), state, 0, moe_fn=moe_fn
            )
            out = [int(logits.flatten().argmax())]
            pos = len(ids)
            for _ in range(n_tokens - 1):
                logits, _ = transformer_forward(
                    model, torch.tensor([[out[-1]]]), state, pos, moe_fn=moe_fn
                )
                out.append(int(logits.flatten().argmax()))
                pos += 1
        torch.npu.synchronize()
        return time.perf_counter() - t, out

    samples: dict[int, list[float]] = {args.n1: [], args.n2: []}
    texts: dict[int, list[int]] = {}
    for rep_i in range(args.reps):
        for n in (args.n1, args.n2):  # alternate, as the rules require
            dt, out = run(n)
            samples[n].append(dt)
            texts.setdefault(n, out)
            print(f"  rep {rep_i+1} N={n:3d}: {dt:.3f} s", flush=True)

    t1 = statistics.median(samples[args.n1])
    t2 = statistics.median(samples[args.n2])
    if t2 <= t1:
        print(f"NON-PHYSICAL: T({args.n2})={t2:.3f}s <= T({args.n1})={t1:.3f}s -> sample rejected")
        return 1
    tpot = (t2 - t1) / (args.n2 - args.n1)
    spread = (max(samples[args.n2]) - min(samples[args.n2])) / t2
    print(
        f"\nTPOT by differencing: ({t2:.3f} - {t1:.3f}) / ({args.n2} - {args.n1}) = {1000*tpot:.0f} ms/token "
        f"({1/tpot:.2f} tok/s)"
    )
    print(f"  spread over reps at N={args.n2}: {100*spread:.1f}%")
    payload = {
        "environment": env,
        "n1": args.n1,
        "n2": args.n2,
        "samples_s": {str(k): v for k, v in samples.items()},
        "tpot_ms": round(1000 * tpot, 1),
        "spread_pct": round(100 * spread, 1),
        "tokens": {str(k): v for k, v in texts.items()},
        "prompt_ids": ids,
    }
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=1)
    print(f"wrote {args.out}")
    print("RESULT: OK" if spread < 0.25 else "RESULT: NOISY")
    return 0


if __name__ == "__main__":
    sys.exit(main())
