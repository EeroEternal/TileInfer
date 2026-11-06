"""Capture per-token logprobs so ``ab_report.py --compare`` can run (C3).

Why this file exists
--------------------
``benchmarks/ab_report.py --compare observed.json baseline.json`` has been in the repo since the
start, and C3 makes it the gate for every change -- but nothing produced those two JSON files, so the
gate has never actually been run.  This is the missing half.

What it writes
--------------
``OUT.json``        ``{prompt: [token_logprob, ...]}`` -- exactly the mapping ``--compare`` reads.
``OUT.tokens.json`` ``{prompt: [token_text, ...]}`` -- for humans, and for the sequence check below.

The second file matters: comparing logprobs across two implementations is only meaningful if both
produced the **same token sequence**.  Without ``temperature=0`` and an ``ignore_eos``-style contract
the sequences can diverge and every logprob would differ for the trivial reason that the model took a
different word -- a failure that looks like a numerical regression.  ``--compare-tokens`` reports
that case explicitly, before anyone reads the logprob deltas.

Usage
-----
    python benchmarks/capture_logprobs.py http://127.0.0.1:8111 native.json
    python benchmarks/capture_logprobs.py http://127.0.0.1:8112 baseline.json
    python benchmarks/ab_report.py --compare native.json baseline.json --atol 2e-2
    python benchmarks/capture_logprobs.py --compare-tokens native.tokens.json baseline.tokens.json
"""

import json
import os
import sys
import urllib.request

MAX_TOKENS = int(os.environ.get("CAPTURE_MAX_TOKENS", "24"))
LOGPROB_TOP = 5
MODEL = "qwen36"

#: Fixed, deterministic prompts: a few shapes so a numerics bug that only shows on long context or on
#: repetition has a chance to appear.  Never make these random -- a baseline is only comparable to a
#: capture over the same prompts.
PROMPTS = [
    "Explain in one sentence why the sky is blue:",
    "Question: what is 17 times 23? Answer with the number only:",
    "List three primary colours, comma separated:",
    (
        "Summarize the following in one word. The quick brown fox jumps over the lazy dog while the "
        "engineer studies the attention kernel and the scheduler decides which request to run next. "
    ),
]


def _post(base: str, body: dict) -> dict:
    req = urllib.request.Request(
        f"{base.rstrip('/')}/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=1800) as resp:
        return json.loads(resp.read())


def capture(base: str) -> tuple[dict, dict]:
    logprobs: dict[str, list[float]] = {}
    tokens: dict[str, list[str]] = {}
    for prompt in PROMPTS:
        payload = _post(
            base,
            {
                "model": MODEL,
                "prompt": prompt,
                "max_tokens": MAX_TOKENS,
                "temperature": 0.0,
                "logprobs": LOGPROB_TOP,
                "stream": False,
            },
        )
        choice = payload["choices"][0]
        detail = choice.get("logprobs") or {}
        seq_lp = detail.get("token_logprobs") or []
        seq_tok = detail.get("tokens") or []
        # The first entry is the prompt's own (already-encoded) token and has no logprob.
        paired = [(tok, lp) for tok, lp in zip(seq_tok, seq_lp) if lp is not None]
        logprobs[prompt] = [float(lp) for _tok, lp in paired]
        tokens[prompt] = [str(tok) for tok, _lp in paired]
        print(f"[capture] {len(paired):3d} tokens  {prompt[:48]!r}", flush=True)
    return logprobs, tokens


def compare_tokens(left_path: str, right_path: str) -> int:
    with open(left_path, encoding="utf-8") as fh:
        left = json.load(fh)
    with open(right_path, encoding="utf-8") as fh:
        right = json.load(fh)
    bad = []
    for prompt in sorted(set(left) | set(right)):
        a, b = left.get(prompt), right.get(prompt)
        if a is None or b is None:
            bad.append(f"prompt {prompt[:40]!r}: present on only one side")
        elif a != b:
            first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            bad.append(
                f"prompt {prompt[:40]!r}: sequences diverge at token {first} "
                f"({a[first : first + 3]!r} vs {b[first : first + 3]!r})"
            )
    if bad:
        print("TOKENS DIFFER -- logprob deltas would be meaningless:")
        for line in bad:
            print("  " + line)
        return 1
    print(f"token sequences identical across {len(left)} prompts")
    return 0


def main(argv: list[str]) -> int:
    if "--compare-tokens" in argv:
        rest = [a for a in argv if a != "--compare-tokens"]
        if len(rest) != 2:
            print("usage: capture_logprobs.py --compare-tokens A.tokens.json B.tokens.json")
            return 2
        return compare_tokens(rest[0], rest[1])
    if len(argv) != 2:
        print("usage: capture_logprobs.py <base_url> <out.json>")
        return 2
    base, out = argv
    logprobs, tokens = capture(base)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(logprobs, fh, indent=2, sort_keys=True)
    with open(out.replace(".json", ".tokens.json"), "w", encoding="utf-8") as fh:
        json.dump(tokens, fh, indent=2, sort_keys=True)
    print(f"[capture] wrote {out} and {out.replace('.json', '.tokens.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
