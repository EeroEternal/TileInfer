#!/usr/bin/env python3
"""Compare two vLLM responses for the C2 check (same prompt, temperature 0).

Ignores volatile fields (`id`, `created`, `system_fingerprint`) and compares
the parts that matter: generated text, token-level logprobs when both sides
asked for them, and usage counts.

Usage:
    python benchmarks/compare_responses.py /tmp/v3-8k-needle.json /tmp/base-8k-needle.json
    python benchmarks/compare_responses.py a.json b.json --atol 1e-4

Exit code 0 when equal (logprobs within `--atol`), 1 otherwise.  The summary is
JSON so it can be pasted into an evidence note.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any


def _content(resp: dict[str, Any]) -> str:
    choice = (resp.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    if isinstance(message.get("content"), str):
        return message["content"]
    if isinstance(choice.get("text"), str):
        return choice["text"]
    return ""


def _logprobs(resp: dict[str, Any]) -> tuple[list[str], list[float]]:
    choice = (resp.get("choices") or [{}])[0]
    lp = choice.get("logprobs") or {}
    content = lp.get("content") or []
    if content:
        tokens = [str(item.get("token", "")) for item in content]
        values = [float(item.get("logprob", 0.0)) for item in content]
        return tokens, values
    tokens = [str(t) for t in (lp.get("tokens") or [])]
    values = [float(v) for v in (lp.get("token_logprobs") or [])]
    return tokens, values


def _usage(resp: dict[str, Any]) -> dict[str, int]:
    usage = resp.get("usage") or {}
    return {
        key: usage.get(key, 0)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }


def compare(a: dict[str, Any], b: dict[str, Any], atol: float) -> dict[str, Any]:
    text_a, text_b = _content(a), _content(b)
    tokens_a, lps_a = _logprobs(a)
    tokens_b, lps_b = _logprobs(b)

    token_diff = None
    if tokens_a or tokens_b:
        if len(tokens_a) != len(tokens_b):
            token_diff = f"token count {len(tokens_a)} != {len(tokens_b)}"
        else:
            for i, (ta, tb) in enumerate(zip(tokens_a, tokens_b)):
                if ta != tb:
                    token_diff = f"token[{i}] {ta!r} != {tb!r}"
                    break
            if token_diff is None:
                for i, (la, lb) in enumerate(zip(lps_a, lps_b)):
                    if abs(la - lb) > atol:
                        token_diff = f"logprob[{i}] {la:.6f} vs {lb:.6f} (atol {atol})"
                        break

    max_lp = max((abs(x - y) for x, y in zip(lps_a, lps_b)), default=0.0)
    usage_a, usage_b = _usage(a), _usage(b)
    same = text_a == text_b and token_diff is None and usage_a == usage_b

    first_diff = None
    if text_a != text_b:
        for i, (ca, cb) in enumerate(zip(text_a, text_b)):
            if ca != cb:
                first_diff = {"char": i, "a": text_a[max(0, i - 20) : i + 40], "b": text_b[max(0, i - 20) : i + 40]}
                break
        if first_diff is None:
            first_diff = {"char": min(len(text_a), len(text_b)), "a": text_a[-40:], "b": text_b[-40:]}

    return {
        "same": same,
        "text_equal": text_a == text_b,
        "text_len": [len(text_a), len(text_b)],
        "first_text_diff": first_diff,
        "token_count": [len(tokens_a), len(tokens_b)],
        "tokens_equal": token_diff is None,
        "token_diff": token_diff,
        "max_abs_logprob_diff": round(max_lp, 8),
        "atol": atol,
        "usage": {"a": usage_a, "b": usage_b},
        "content": {"a": text_a[:240], "b": text_b[:240]},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--atol", type=float, default=1e-4, help="logprob tolerance (4-decimal rule)")
    args = ap.parse_args()

    a = json.loads(pathlib.Path(args.a).read_text(encoding="utf-8"))
    b = json.loads(pathlib.Path(args.b).read_text(encoding="utf-8"))
    result = compare(a, b, args.atol)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    sys.exit(0 if result["same"] else 1)


if __name__ == "__main__":
    main()
