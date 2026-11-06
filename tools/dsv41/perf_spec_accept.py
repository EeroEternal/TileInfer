#!/usr/bin/env python3
"""Measure n-gram speculative-decoding acceptance from **real greedy outputs**.

A proposed token is accepted iff it equals the model's own next greedy token; for a already
generated greedy sequence, that is exactly "proposal matches the upcoming generated token".
So acceptance can be measured without extra forwards, and the achievable speedup is
``generated_tokens / forward_calls`` where a forward yields ``accepted + 1`` tokens.

Usage:
    python tools/dsv41/perf_spec_accept.py --port 8123 --max-new 32 --k 4 --n 3 --limit 10
"""

from __future__ import annotations

import argparse
import json
import urllib.request

from tileinfer.models.dsv41.spec_decode import propose_ngram

PROMPTS = [
    "The capital of France is",
    "The capital of Japan is",
    "Water boils at a temperature of",
    "The largest planet in the solar system is",
    "Write a Python function that returns the sum of a list:",
    "Machine learning is a field of",
    "A transformer uses attention to",
    "The purpose of a memory page cache is",
    "The difference between TCP and UDP is",
    "Gradient descent minimizes",
]


def ask(port: int, prompt: str, max_new: int):
    body = json.dumps({
        "model": "dsv41", "prompt": prompt, "max_tokens": max_new,
        "temperature": 0, "ignore_eos": True, "return_token_ids": True,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions", data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    return [int(t) for t in d["choices"][0]["token_ids"]]


def acceptance(tokens, k: int, n: int):
    proposed = accepted = forwards = 0
    i = 0
    while i < len(tokens):
        cand = propose_ngram(tokens[:i], k, n) if i > 0 else []
        cand = cand[:k]
        proposed += len(cand)
        hit = 0
        for c, t in zip(cand, tokens[i : i + len(cand)]):
            if c == t:
                hit += 1
            else:
                break
        accepted += hit
        i += hit + 1
        forwards += 1
    return proposed, accepted, forwards


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--limit", type=int, default=10)
    args = ap.parse_args()

    total = {"proposed": 0, "accepted": 0, "forwards": 0, "tokens": 0}
    for p in PROMPTS[: args.limit]:
        toks = ask(args.port, p, args.max_new)
        pr, ac, fw = acceptance(toks, args.k, args.n)
        total["proposed"] += pr
        total["accepted"] += ac
        total["forwards"] += fw
        total["tokens"] += len(toks)
        print(
            f"{p[:36]!r:40} tokens={len(toks)} proposed={pr} accepted={ac} "
            f"fwd={fw} acc={ac / max(pr, 1):.2%}",
            flush=True,
        )
    t = total
    print(
        "TOTAL tokens=%d proposed=%d accepted=%d forwards=%d acceptance=%.2f%% "
        "tokens_per_forward=%.2f speedup=%.2fx"
        % (
            t["tokens"], t["proposed"], t["accepted"], t["forwards"],
            100.0 * t["accepted"] / max(t["proposed"], 1),
            t["tokens"] / max(t["forwards"], 1),
            (t["tokens"] / max(t["forwards"], 1)),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
