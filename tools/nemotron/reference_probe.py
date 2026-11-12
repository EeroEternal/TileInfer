#!/usr/bin/env python3
"""CPU reference probe for the Nemotron M5 bring-up.

This is the no-vendor-baseline oracle: load the checkpoint with HF transformers on CPU,
run one greedy prompt, and dump text/token ids/first-token top-5.  It never downloads and
never touches NPU; if the checkpoint is absent it exits with a clear error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="local HF checkpoint directory")
    ap.add_argument("--prompt", default="Say hello in one sentence.")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--out", default="", help="JSON output path (default: stdout)")
    ap.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"))
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model_dir = Path(args.model)
    if not model_dir.is_dir():
        print(f"error: model directory not found: {model_dir}", file=sys.stderr)
        return 2

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - environment-specific
        print(f"error: transformers/torch unavailable: {exc}", file=sys.stderr)
        return 2

    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.eval()

    inputs = tok(args.prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
            do_sample=False,
            temperature=0.0,
            return_dict_in_generate=True,
            output_scores=True,
        )
    text = tok.decode(out.sequences[0], skip_special_tokens=False)
    first_scores = out.scores[0][0] if out.scores else None
    top5 = []
    if first_scores is not None:
        vals, idx = torch.topk(first_scores, k=min(5, first_scores.numel()))
        top5 = [
            {"token_id": int(i), "token": tok.decode([int(i)]), "logprob": float(v)}
            for v, i in zip(vals.tolist(), idx.tolist())
        ]

    payload = {
        "model": str(model_dir),
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "dtype": args.dtype,
        "device": "cpu",
        "text": text,
        "token_ids": out.sequences[0].tolist(),
        "first_token_top5": top5,
        "note": "CPU reference; no NPU and no vllm-ascend baseline",
    }
    text_out = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        Path(args.out).write_text(text_out, encoding="utf-8")
    else:
        print(text_out, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
