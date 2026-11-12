#!/usr/bin/env python3
"""Small fixed task suite for speculative-decoding quality equivalence.

This is not a benchmark; it is a floor check for the S2 quality gate.  The
speculative path is allowed to differ token-by-token from the non-spec path,
but it must not lose task correctness.  Eight short prompts with programmatic
checks are sent to a running server; the same suite must be run against the
non-spec and spec arms.

Usage:
    python benchmarks/spec_quality_probe.py --port 8322 --label mtp \
        --out /tmp/spec-quality-mtp.json
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from typing import Callable


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


def text_of(response: dict) -> str:
    return (response.get("choices") or [{}])[0].get("message", {}).get("content") or ""


def check_arith(text: str) -> bool:
    return bool(re.search(r"\b1776\b", text))


def check_letters(text: str) -> bool:
    return bool(re.search(r"\b9\b", text))


def check_code(text: str) -> bool:
    return "def add" in text and "return" in text


def check_json(text: str) -> bool:
    for blob in re.findall(r"\{[^{}]*\}", text):
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if obj.get("name") == "tile" and obj.get("ok") is True:
            return True
    return False


def check_ready(text: str) -> bool:
    return "READY" in text


def check_logic(text: str) -> bool:
    return bool(re.search(r"\byes\b", text, re.IGNORECASE))


def check_colors(text: str) -> bool:
    low = text.lower()
    hits = [c for c in ("red", "green", "blue") if c in low]
    return len(hits) == 3 and "," in low


def check_len(text: str) -> bool:
    return bool(re.search(r"\b3\b", text))


PROMPTS: list[tuple[str, str, Callable[[str], bool]]] = [
    ("arith", "What is 37 * 48? Reply with the number only.", check_arith),
    ("letters", "How many letters are in the word 'tileinfer'? Reply with the number only.", check_letters),
    ("code", "Write a Python function named add that takes a and b and returns their sum. Reply with code only.", check_code),
    (
        "json",
        'Return a JSON object with keys "name" and "ok"; name must be "tile", ok must be true. Reply with JSON only.',
        check_json,
    ),
    ("ready", "Reply with exactly: READY", check_ready),
    (
        "logic",
        "All bloops are razzies. All razzies are lazzies. Are all bloops lazzies? Answer yes or no.",
        check_logic,
    ),
    ("colors", "Name the three primary colors, separated by commas, with no other text.", check_colors),
    ("len", "In Python, what is len([1, 2, 3])? Reply with the number only.", check_len),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8322)
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()

    items = []
    for name, prompt, check in PROMPTS:
        try:
            response = ask(args.port, prompt, args.max_tokens)
            text = text_of(response)
            ok = bool(check(text))
            items.append(
                {
                    "name": name,
                    "prompt": prompt,
                    "ok": ok,
                    "text": text,
                    "usage": response.get("usage"),
                }
            )
        except Exception as exc:  # noqa: BLE001 - report, don't abort the suite
            items.append({"name": name, "prompt": prompt, "ok": False, "error": f"{type(exc).__name__}: {exc}"})

    passed = sum(1 for item in items if item["ok"])
    result = {"label": args.label, "passed": passed, "total": len(items), "items": items}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({"label": args.label, "passed": passed, "total": len(items), "out": args.out}, ensure_ascii=False))


if __name__ == "__main__":
    main()
