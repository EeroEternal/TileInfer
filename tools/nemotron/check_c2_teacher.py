#!/usr/bin/env python3
"""Teacher-forced C2 acceptance check.

Why teacher-forced
------------------
Comparing two sampled greedy trajectories is only meaningful while both sides take the same
token.  After the first divergence every later "logprob delta" mixes numerical error with a
different prefix.  This checker forces the reference prefix (token ids) on the served model and
compares the *next-token distribution* at every step.

Acceptance (the project's existing cross-implementation A/B convention)
-----------------------------------------------------------------------
``benchmarks/two_arm.sh`` compares two arms with ``--atol 2e-2`` after requiring the same token
sequence.  This tool uses the same 2e-2 absolute logprob tolerance, plus the usual greedy
top-1 / top-5 criteria:

* top-1 token id must match the reference (an exact HF top-1 tie is recorded separately);
* top-5 overlap must be at least four;
* max absolute logprob delta on the chosen token must be <= ``--atol`` (default 2e-2).

It is deliberately not a 1e-4 "four decimal places" test: that is a same-stack determinism
check, not a cross-backend reference check.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_ATOL = 2e-2
DEFAULT_TOP5_MIN = 4
DEFAULT_TIE_LOGIT_TOL = 0.125


def check(
    hf_steps: dict[str, Any],
    service: list[dict[str, Any]],
    *,
    atol: float = DEFAULT_ATOL,
    top5_min: int = DEFAULT_TOP5_MIN,
    tie_logit_tol: float = DEFAULT_TIE_LOGIT_TOL,
) -> dict[str, Any]:
    by_prompt = {str(row["prompt"]): row for row in service}
    prompts = []
    for prompt, reference in hf_steps.items():
        row = by_prompt.get(prompt)
        if row is None:
            prompts.append({"prompt": prompt, "ok": False, "errors": ["missing teacher capture"]})
            continue
        errors: list[str] = []
        max_delta = 0.0
        deltas: list[float] = []
        same = tied = mismatch = top5_ok = 0
        failures: list[dict[str, Any]] = []
        service_steps = {int(s["step"]): s for s in row.get("steps", []) if "error" not in s}
        for index, candidates in enumerate(reference.get("steps", [])):
            served = service_steps.get(index)
            if served is None or not candidates:
                errors.append(f"step {index}: missing service scores")
                continue
            top = candidates[0]
            got = str(served["tokens"][0])
            got_top = served["top_logprobs"][0]
            got_lp = float(got_top[got])
            hit = next((c for c in candidates if str(c["tok"]) == got), None)
            if got == str(top["tok"]):
                same += 1
                top1_ok = True
            elif hit is not None and abs(float(hit["logit"]) - float(top["logit"])) <= tie_logit_tol:
                tied += 1
                top1_ok = True
            else:
                mismatch += 1
                top1_ok = False
            overlap = len(set(got_top) & {str(c["tok"]) for c in candidates})
            top5_ok += int(overlap >= top5_min)
            delta = abs(float(hit["logprob"]) - got_lp) if hit is not None else float("inf")
            if hit is not None:
                deltas.append(delta)
                max_delta = max(max_delta, delta)
            step_ok = top1_ok and overlap >= top5_min and delta <= atol
            if not step_ok:
                failures.append(
                    {
                        "step": index,
                        "served": got,
                        "reference": top["tok"],
                        "top1_ok": top1_ok,
                        "top5_overlap": overlap,
                        "logprob_delta": None if hit is None else delta,
                    }
                )
        if failures:
            errors.append(f"{len(failures)} step(s) outside acceptance")
        prompts.append(
            {
                "prompt": prompt,
                "ok": not errors,
                "steps": len(reference.get("steps", [])),
                "same_top1": same,
                "tied_top1": tied,
                "top1_mismatch": mismatch,
                "top5_ge_min": top5_ok,
                "max_logprob_delta": max_delta,
                "mean_logprob_delta": sum(deltas) / len(deltas) if deltas else None,
                "failures": failures,
                "errors": errors,
            }
        )
    return {
        "ok": bool(prompts) and all(p["ok"] for p in prompts),
        "criteria": {"atol": atol, "top5_min": top5_min, "tie_logit_tol": tie_logit_tol},
        "prompts": prompts,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hf-steps", type=Path, required=True)
    ap.add_argument("--service", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    ap.add_argument("--top5-min", type=int, default=DEFAULT_TOP5_MIN)
    ap.add_argument("--tie-logit-tol", type=float, default=DEFAULT_TIE_LOGIT_TOL)
    args = ap.parse_args(argv)
    result = check(
        json.loads(args.hf_steps.read_text()),
        json.loads(args.service.read_text()),
        atol=args.atol,
        top5_min=args.top5_min,
        tie_logit_tol=args.tie_logit_tol,
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
