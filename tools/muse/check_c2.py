"""Check a running MuseGlimmer server against the manifest's C2 sample.

    python tools/muse/check_c2.py --base-url http://127.0.0.1:8360

The manifest records a first-token top-5 from a float32 CPU oracle
(``tools/muse/c2_oracle.py``). The served model is bf16, so the comparison uses the declared
tolerance (top-1 logprob and max top-5 logprob) plus exact token-set equality.
"""

from __future__ import annotations

import argparse
import json
import sys

import requests

from tileinfer.manifests import find_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8360")
    parser.add_argument("--model", default="muse")
    parser.add_argument("--manifest", default="muse_glimmer")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    manifest = find_manifest(args.manifest)
    sample = manifest.benchmark
    if not sample:
        print(f"{args.manifest} has no benchmark sample", file=sys.stderr)
        return 1
    tolerance = sample.get("tolerance", {})
    top1_tol = float(tolerance.get("top1_logprob", 0.0))
    top5_tol = float(tolerance.get("top5_max_logprob", 0.0))

    response = requests.post(
        args.base_url.rstrip("/") + "/v1/completions",
        json={
            "model": args.model,
            "prompt": sample["prompt"],
            "max_tokens": int(sample.get("max_tokens", 1)),
            "temperature": float(sample.get("temperature", 0.0)),
            "logprobs": int(sample.get("top_logprobs", 5)),
            "echo": False,
        },
        timeout=300,
    )
    response.raise_for_status()
    payload = response.json()["choices"][0]
    actual = payload["logprobs"]["top_logprobs"][0]
    first = payload["logprobs"]["tokens"][0]

    expected = {token: float(value) for token, value in sample["top5"]}
    actual_tokens = set(actual)
    tokens_match = actual_tokens == set(expected)
    first_match = first == sample["first_token"]
    top1_diff = abs(float(actual.get(sample["first_token"], float("nan"))) - expected[sample["first_token"]])
    common = actual_tokens & set(expected)
    max_diff = max((abs(actual[t] - expected[t]) for t in common), default=float("inf"))

    report = {
        "first_token": first,
        "expected_first_token": sample["first_token"],
        "first_token_match": first_match,
        "token_set": sorted(actual_tokens),
        "expected_token_set": sorted(expected),
        "token_set_match": tokens_match,
        "top1_logprob_diff": top1_diff,
        "top1_tolerance": top1_tol,
        "top5_max_logprob_diff": max_diff,
        "top5_tolerance": top5_tol,
        "pass": bool(first_match and tokens_match and top1_diff <= top1_tol and max_diff <= top5_tol),
    }
    text = json.dumps(report, ensure_ascii=False, indent=1)
    print(text)
    if args.json_out:
        with open(args.json_out, "w") as handle:
            handle.write(text)
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
