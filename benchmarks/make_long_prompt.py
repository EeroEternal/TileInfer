#!/usr/bin/env python3
"""Build a prompt of approximately N tokens from a seed file.

Used by the long-context ladder in ``docs/verification-matrix.md``: the length
has to be exact enough that "8K" means 8K prompt tokens, not 8K characters.
Token counting uses the model's own tokenizer, so the number matches what the
server reports in ``usage.prompt_tokens`` within a token or two.

Usage:
    python benchmarks/make_long_prompt.py --tokens 8192 --out /tmp/prompt-8k.txt
    python benchmarks/make_long_prompt.py --tokens 32768 --out /tmp/prompt-32k.txt \
        --seed-file docs/architecture.md

This builds a *length* probe, not a correctness probe.  For "is the answer at
position 200K still right" use a needle test; a repeated seed is fine for TTFT
and TPOT but can make the model degenerate.
"""

from __future__ import annotations

import argparse
import json
import pathlib

from transformers import AutoTokenizer

DEFAULT_MODEL = "/home/models/Qwen3.6-35B-A3B"
DEFAULT_SEED = "README.md"


def build(tokenizer: object, seed: str, target: int) -> tuple[str, int]:
    ids = tokenizer.encode(seed, add_special_tokens=False)
    if not ids:
        raise SystemExit("seed produced no tokens")
    reps = target // len(ids) + 1
    ids = (ids * reps)[:target]

    # decode/encode is not always the identity (byte-level BPE at the seam);
    # trim or pad until the round trip lands within a token of the target.
    text = tokenizer.decode(ids)
    for _ in range(8):
        actual = len(tokenizer.encode(text, add_special_tokens=False))
        if actual == target:
            break
        if actual > target:
            ids = ids[: max(1, target - (actual - target))]
        else:
            ids = ids + ids[: target - actual]
        text = tokenizer.decode(ids)
    return text, len(tokenizer.encode(text, add_special_tokens=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--tokens", type=int, required=True, help="target prompt tokens")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed-file", default=DEFAULT_SEED)
    ap.add_argument("--seed-text", default=None, help="override the seed file")
    args = ap.parse_args()

    seed = args.seed_text
    if seed is None:
        seed = pathlib.Path(args.seed_file).read_text(encoding="utf-8")
    if not seed.strip():
        raise SystemExit(f"empty seed: {args.seed_file}")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    text, actual = build(tok, seed, args.tokens)
    out = pathlib.Path(args.out)
    out.write_text(text, encoding="utf-8")
    print(
        json.dumps(
            {
                "out": str(out),
                "target_tokens": args.tokens,
                "actual_tokens": actual,
                "chars": len(text),
                "seed": args.seed_file if args.seed_text is None else "<inline>",
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
