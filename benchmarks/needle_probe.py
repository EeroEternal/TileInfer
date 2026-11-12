#!/usr/bin/env python3
"""Needle-in-a-haystack prompt builder, with an optional answer check.

The long-context cells of ``docs/verification-matrix.md`` need two things:
speed (``v3_long_ttft.py``) and correctness (this script).  A prompt of ~N
tokens is built from a seed, a unique secret is inserted at a given depth, and
a question is appended.  With ``--port`` the prompt is sent to a running
server and the answer is checked against the secret.

Usage:
    # build only
    python benchmarks/needle_probe.py --tokens 8192 --depth 0.5 \
        --out-prompt /tmp/needle-8k.txt

    # build + ask a running server (saves the raw response too)
    python benchmarks/needle_probe.py --tokens 8192 --depth 0.5 \
        --out-prompt /tmp/needle-8k.txt --port 8322 \
        --out-response /tmp/v3-8k-needle.json

Exit code is 0 when the answer contains the secret, 1 when it does not (or
when the request fails).  Prompt tokens are reported from ``usage``; the local
tokenizer count is only a sanity check.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import urllib.error
import urllib.request

from transformers import AutoTokenizer

DEFAULT_MODEL = "/home/models/Qwen3.6-35B-A3B"
DEFAULT_SEED = "README.md"
DEFAULT_NEEDLE = "The secret access code is TIGER-7421. Remember it."
DEFAULT_QUESTION = "What is the secret access code? Reply with the code only."
DEFAULT_ANSWER = "TIGER-7421"


def build(tokenizer: object, seed: str, tokens: int, depth: float, needle: str, question: str) -> tuple[str, int]:
    seed_ids = tokenizer.encode(seed, add_special_tokens=False)
    needle_ids = tokenizer.encode(needle, add_special_tokens=False)
    question_ids = tokenizer.encode(question, add_special_tokens=False)
    if not seed_ids:
        raise SystemExit("seed produced no tokens")

    filler_target = max(1, tokens - len(needle_ids) - len(question_ids))
    filler = (seed_ids * (filler_target // len(seed_ids) + 1))[:filler_target]
    at = int(max(0.0, min(1.0, depth)) * len(filler))
    ids = filler[:at] + needle_ids + filler[at:]
    text = tokenizer.decode(ids) + "\n\n" + question
    return text, len(tokenizer.encode(text, add_special_tokens=False))


def ask(port: int, prompt: str, max_tokens: int) -> dict:
    payload = {
        "model": "qwen36",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--tokens", type=int, required=True, help="approximate prompt tokens")
    ap.add_argument("--depth", type=float, default=0.5, help="needle position, 0.0-1.0")
    ap.add_argument("--out-prompt")
    ap.add_argument("--seed-file", default=DEFAULT_SEED)
    ap.add_argument("--needle", default=DEFAULT_NEEDLE)
    ap.add_argument("--question", default=DEFAULT_QUESTION)
    ap.add_argument("--expect", default=DEFAULT_ANSWER)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--out-response")
    ap.add_argument("--max-tokens", type=int, default=32)
    args = ap.parse_args()

    seed = pathlib.Path(args.seed_file).read_text(encoding="utf-8")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    text, local_tokens = build(tok, seed, args.tokens, args.depth, args.needle, args.question)

    if args.out_prompt:
        pathlib.Path(args.out_prompt).write_text(text, encoding="utf-8")

    result: dict = {
        "tokens_target": args.tokens,
        "tokens_local": local_tokens,
        "depth": args.depth,
        "needle": args.needle,
        "question": args.question,
        "expect": args.expect,
        "prompt_path": args.out_prompt,
    }
    if args.port is None:
        print(json.dumps(result, ensure_ascii=False))
        return

    try:
        response = ask(args.port, text, args.max_tokens)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        print(json.dumps(result, ensure_ascii=False))
        raise SystemExit(1)

    if args.out_response:
        pathlib.Path(args.out_response).write_text(
            json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    choice = (response.get("choices") or [{}])[0]
    answer = (choice.get("message") or {}).get("content", "")
    usage = response.get("usage") or {}
    result.update(
        {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "answer": answer,
            "ok": args.expect in answer,
        }
    )
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
