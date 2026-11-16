#!/usr/bin/env python3
"""Compare the HF/CPU reference probe with a vLLM completion response.

The reference is text/token-level (greedy, temperature 0).  The service response must be
requested from ``/v1/completions`` with ``logprobs`` so the first generated token has a
top-k map; a chat response uses a different template and cannot be compared to the raw
prompt reference.

Exit 0 only when the generated text and the first-token top-k logprobs agree within
``--atol``.  The result is JSON, suitable as C2 evidence.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any


def _unwrap(resp: dict[str, Any]) -> dict[str, Any]:
    # Accept a bare completion, an OpenAI response, or an evidence wrapper.
    for key in ("response", "completion", "result"):
        value = resp.get(key)
        if isinstance(value, dict) and "choices" in value:
            return value
    return resp


def _first_choice(resp: dict[str, Any]) -> dict[str, Any]:
    choices = _unwrap(resp).get("choices") or []
    return choices[0] if choices else {}


def _service_text(choice: dict[str, Any]) -> str:
    if isinstance(choice.get("text"), str):
        return choice["text"]
    message = choice.get("message") or {}
    return message.get("content", "") if isinstance(message.get("content"), str) else ""


def _service_top5(choice: dict[str, Any]) -> dict[str, float]:
    lp = choice.get("logprobs") or {}
    content = lp.get("content") or []
    if content and isinstance(content[0], dict):
        # Chat style: each entry has a chosen token and optional top_logprobs.
        first = content[0]
        top = first.get("top_logprobs") or []
        out = {}
        for item in top:
            if isinstance(item, dict) and "token" in item:
                out[str(item["token"])] = float(item.get("logprob", 0.0))
        if not out and "token" in first:
            out[str(first["token"])] = float(first.get("logprob", 0.0))
        return out
    top = lp.get("top_logprobs") or []
    if top and isinstance(top[0], dict):
        return {str(k): float(v) for k, v in top[0].items()}
    tokens = lp.get("tokens") or []
    values = lp.get("token_logprobs") or []
    if tokens and values:
        return {str(tokens[0]): float(values[0])}
    return {}


def _first_diff(a: str, b: str) -> dict[str, Any] | None:
    if a == b:
        return None
    for i, (ca, cb) in enumerate(zip(a, b)):
        if ca != cb:
            return {"char": i, "ref": a[max(0, i - 20) : i + 40], "service": b[max(0, i - 20) : i + 40]}
    i = min(len(a), len(b))
    return {"char": i, "ref": a[-40:], "service": b[-40:]}


def compare(ref: dict[str, Any], resp: dict[str, Any], atol: float) -> dict[str, Any]:
    choice = _first_choice(resp)
    ref_text = ref.get("text", "")
    service_text = _service_text(choice)
    first = ref.get("first_token_top5") or []
    ref_top = {str(item["token"]): float(item["logprob"]) for item in first if "token" in item}
    service_top = _service_top5(choice)

    common = sorted(set(ref_top) & set(service_top))
    max_diff = max((abs(ref_top[t] - service_top[t]) for t in common), default=None)
    missing = sorted(set(ref_top) - set(service_top))
    extra = sorted(set(service_top) - set(ref_top))
    text_equal = ref_text == service_text
    # Top-k boundaries can tie.  Require enough overlap and a bounded logprob gap rather
    # than an exact set match; report the strict comparison separately.
    enough_overlap = len(common) >= max(1, min(len(ref_top), len(service_top)) - 1)
    top_close = bool(ref_top) and bool(service_top) and max_diff is not None and max_diff <= atol and enough_overlap
    top_equal = bool(ref_top) and bool(service_top) and not missing and not extra and max_diff is not None and max_diff <= atol
    same = text_equal and top_close

    return {
        "same": same,
        "text_equal": text_equal,
        "first_text_diff": _first_diff(ref_text, service_text),
        "text_len": [len(ref_text), len(service_text)],
        "ref_first_top5": ref_top,
        "service_first_top5": service_top,
        "top5_common": common,
        "top5_missing_in_service": missing,
        "top5_extra_in_service": extra,
        "top5_strict_equal": top_equal,
        "top5_close_tie_aware": top_close,
        "max_abs_logprob_diff": max_diff,
        "atol": atol,
        "ref_text": ref_text[:240],
        "service_text": service_text[:240],
        "usage": _unwrap(resp).get("usage", {}),
        "reference_id": ref.get("reference_id", ""),
        "transformers_version": ref.get("transformers_version", ""),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("reference", help="JSON from tools/nemotron/reference_probe.py")
    ap.add_argument("response", help="JSON from /v1/completions")
    ap.add_argument("--atol", type=float, default=1e-4, help="first-token logprob tolerance")
    ap.add_argument("--out", default="", help="write result JSON here (default: stdout)")
    args = ap.parse_args()
    ref = json.loads(pathlib.Path(args.reference).read_text(encoding="utf-8"))
    resp = json.loads(pathlib.Path(args.response).read_text(encoding="utf-8"))
    result = compare(ref, resp, args.atol)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        pathlib.Path(args.out).write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    sys.exit(0 if result["same"] else 1)


if __name__ == "__main__":
    main()
