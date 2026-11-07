"""Task-level accuracy: does the greedy answer contain the expected fact?

This complements the first-token and component-level gates: the 100-prompt token-equality gate is
too strict for a different MoE kernel, so we also check whether the generated answer is semantically
correct on a small set of unambiguous facts.
"""

from __future__ import annotations

import argparse
import json
import urllib.request

PROMPTS = [
    ("The capital of France is", "paris"),
    ("The capital of Japan is", "tokyo"),
    ("The capital of Italy is", "rome"),
    ("The capital of Germany is", "berlin"),
    ("The capital of Spain is", "madrid"),
    ("The capital of China is", "beijing"),
    ("The capital of Brazil is", "brasilia"),
    ("The capital of Canada is", "ottawa"),
    ("The capital of Australia is", "canberra"),
    ("The capital of Egypt is", "cairo"),
    ("Water boils at a temperature of", "100"),
    ("The largest planet in the solar system is", "jupiter"),
    ("The author of Romeo and Juliet is", "shakespeare"),
    ("The chemical symbol for gold is", "au"),
    ("The speed of light is approximately", "299"),
    ("The first president of the United States was", "washington"),
    ("The longest river in the world is", "nile"),
    ("The tallest mountain on Earth is", "everest"),
    ("The square root of 144 is", "12"),
    ("Seven plus eight equals", "15"),
]


def one(port: int, prompt: str, max_tokens: int) -> str:
    body = json.dumps(
        {
            "model": "dsv41",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "ignore_eos": True,
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=900) as resp:
        return json.loads(resp.read())["choices"][0]["text"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--out", default="")
    ap.add_argument("--min-acc", type=float, default=0.9)
    args = ap.parse_args()

    hits, rows = 0, []
    for prompt, expected in PROMPTS:
        text = one(args.port, prompt, args.max_tokens)
        ok = expected in text.lower()
        hits += int(ok)
        rows.append({"prompt": prompt, "expected": expected, "text": text, "ok": ok})
        print(f"{'OK ' if ok else 'MISS'} {prompt!r} -> {text!r}")
    acc = hits / len(PROMPTS)
    print(f"task_accuracy={hits}/{len(PROMPTS)} = {acc:.3f}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"accuracy": acc, "rows": rows}, f, ensure_ascii=False, indent=2)
    return 0 if acc >= args.min_acc else 1


if __name__ == "__main__":
    raise SystemExit(main())
