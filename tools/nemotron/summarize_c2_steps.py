#!/usr/bin/env python3
"""Summarise C2 per-step reference evidence without turning drift into a pass.

The C2 gate in ``check_c2_logprobs.py`` is deliberately a first-token distribution
check.  This helper complements it by comparing *every* generated step against the
HF top-5 table, so a report can show where a trajectory stays top-1 equivalent and
where later bf16 drift appears.  It never changes the gate result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _summarize(
    prompt: str,
    service: dict[str, Any],
    steps: list[list[dict[str, Any]]],
    *,
    top5_min: int,
    logprob_tol: float,
    tie_logit_tol: float,
) -> dict[str, Any]:
    n = min(len(service.get("ids", [])), len(steps))
    same_top1 = 0
    tied_alt = 0
    top1_mismatch = 0
    top5_ok = 0
    delta_ok = 0
    chosen_missing = 0
    all_ok = 0
    first_bad: list[dict[str, Any]] = []
    for i in range(n):
        candidates = steps[i]
        if not candidates:
            continue
        top = candidates[0]
        chosen = str(service["tokens"][i])
        svc_top = service["top_logprobs"][i]
        svc_top1 = max(svc_top, key=svc_top.get)
        hit = next((c for c in candidates if str(c["tok"]) == chosen), None)
        delta = None
        if hit is not None:
            delta = abs(float(hit["logprob"]) - float(svc_top[chosen]))
            delta_ok += int(delta <= logprob_tol)
        else:
            chosen_missing += 1
        overlap = len(set(svc_top) & {str(c["tok"]) for c in candidates})
        top5_ok += int(overlap >= top5_min)
        if chosen == str(top["tok"]):
            same_top1 += 1
            top1_ok = True
        else:
            tied = hit is not None and abs(float(hit["logit"]) - float(top["logit"])) <= tie_logit_tol
            top1_ok = bool(tied)
            if tied:
                tied_alt += 1
            else:
                top1_mismatch += 1
        step_ok = top1_ok and overlap >= top5_min and delta is not None and delta <= logprob_tol
        all_ok += int(step_ok)
        if not step_ok and len(first_bad) < 8:
            first_bad.append(
                {
                    "step": i,
                    "chosen": chosen,
                    "hf_top1": top["tok"],
                    "top1_ok": top1_ok,
                    "top5_overlap": overlap,
                    "logprob_delta": delta,
                }
            )
    return {
        "prompt": prompt,
        "steps": n,
        "same_top1": same_top1,
        "tied_alt": tied_alt,
        "top1_mismatch": top1_mismatch,
        "top5_ge_min": top5_ok,
        "logprob_delta_ok": delta_ok,
        "chosen_missing_from_hf_top5": chosen_missing,
        "all_ok": all_ok,
        "first_bad": first_bad,
    }


def summarize(
    hf_steps: dict[str, Any],
    service: list[dict[str, Any]],
    *,
    top5_min: int = 4,
    logprob_tol: float = 0.5,
    tie_logit_tol: float = 0.125,
) -> dict[str, Any]:
    prompts = []
    for row in service:
        prompt = str(row["prompt"])
        reference = hf_steps.get(prompt)
        if reference is None:
            continue
        prompts.append(
            _summarize(
                prompt,
                row,
                reference.get("steps", []),
                top5_min=top5_min,
                logprob_tol=logprob_tol,
                tie_logit_tol=tie_logit_tol,
            )
        )
    return {
        "criteria": {
            "top5_min": top5_min,
            "logprob_tol": logprob_tol,
            "tie_logit_tol": tie_logit_tol,
        },
        "prompts": prompts,
    }


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "| prompt | steps | same top-1 | tied alt | top-1 mismatch | top-5 ≥ min | delta ok | all ok |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["prompts"]:
        lines.append(
            "| {prompt} | {steps} | {same_top1} | {tied_alt} | {top1_mismatch} | "
            "{top5_ge_min} | {logprob_delta_ok} | {all_ok} |".format(**row)
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hf-steps", type=Path, required=True)
    ap.add_argument("--service", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--top5-min", type=int, default=4)
    ap.add_argument("--logprob-tol", type=float, default=0.5)
    ap.add_argument("--tie-logit-tol", type=float, default=0.125)
    args = ap.parse_args(argv)
    result = summarize(
        json.loads(args.hf_steps.read_text()),
        json.loads(args.service.read_text()),
        top5_min=args.top5_min,
        logprob_tol=args.logprob_tol,
        tie_logit_tol=args.tie_logit_tol,
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.write_text(payload, encoding="utf-8")
    else:
        print(_markdown(result), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
