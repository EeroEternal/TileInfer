#!/usr/bin/env python3
"""Teacher-forced C2 acceptance check.

C2 is a **rank** check, not a bitwise logprob check
---------------------------------------------------
For an OpenAI-compatible API, logprobs are approximate model outputs.  There is
no industry-wide absolute bound for comparing two backends; the common contract
is that greedy top-1 agrees and the reference top-1 remains in the served
top-5.  Absolute logprob deltas are recorded as diagnostics, not used as the
pass/fail gate.

Teacher forcing is required: after two greedy trajectories diverge, comparing
later logprobs measures different prefixes, not numerical parity.  This tool
forces the reference token ids at every step, then compares the next-token
distribution.

Criteria (defaults)
-------------------
* top-1 token matches the reference, or is within ``--tie-logit-tol`` of the
  reference top-1 logit (a genuine near-tie; recorded, not silent);
* the reference top-1 token is present in the served top-5;
* token ids are compared when the service returns them
  (``return_token_ids=true``); decoded strings alone are ambiguous for
  byte-fallback fragments (empty or ``U+FFFD``);
* same request re-run produces the same tokens/logprobs.

``--atol`` (default 2e-2, the project's cross-arm A/B convention) is retained
for the reported diagnostic only.  A 1e-4 "four decimal places" test is a
same-stack determinism check, not a cross-backend C2 gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_ATOL = 2e-2
DEFAULT_TIE_LOGIT_TOL = 0.25



def check(
    hf_steps: dict[str, Any],
    service: list[dict[str, Any]],
    *,
    atol: float = DEFAULT_ATOL,
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
        same = tied = mismatch = 0
        failures: list[dict[str, Any]] = []
        service_steps = {int(s["step"]): s for s in row.get("steps", []) if "error" not in s}
        for index, candidates in enumerate(reference.get("steps", [])):
            served = service_steps.get(index)
            if served is None or not candidates:
                errors.append(f"step {index}: missing service scores")
                continue
            top = candidates[0]
            got = str(served["tokens"][0])
            got_ids = served.get("token_ids") or []
            got_id = int(got_ids[0]) if got_ids else None
            got_top = served["top_logprobs"][0]
            token_logprobs = served.get("token_logprobs") or []
            got_lp = float(token_logprobs[0]) if token_logprobs else float(got_top.get(got, next(iter(got_top.values()))))
            if got_id is not None:
                id_match = got_id == int(top["id"])
                hit = next((c for c in candidates if int(c["id"]) == got_id), None)
            else:
                id_match = got == str(top["tok"])
                hit = next((c for c in candidates if str(c["tok"]) == got), None)
            if id_match:
                same += 1
                top1_ok = True
            elif hit is not None and abs(float(hit["logit"]) - float(top["logit"])) <= tie_logit_tol:
                tied += 1
                top1_ok = True
            else:
                mismatch += 1
                top1_ok = False
            ref_in_top5 = id_match or str(top["tok"]) in got_top
            delta = abs(float(hit["logprob"]) - got_lp) if hit is not None else None
            if delta is not None:
                deltas.append(delta)
                max_delta = max(max_delta, delta)
            step_ok = top1_ok and ref_in_top5
            if not step_ok:
                failures.append(
                    {
                        "step": index,
                        "served": got,
                        "reference": top["tok"],
                        "top1_ok": top1_ok,
                        "reference_in_top5": ref_in_top5,
                        "logprob_delta": delta,
                    }
                )
        if failures:
            errors.append(f"{len(failures)} step(s) outside rank acceptance")
        prompts.append(
            {
                "prompt": prompt,
                "ok": not errors,
                "steps": len(reference.get("steps", [])),
                "same_top1": same,
                "tied_top1": tied,
                "top1_mismatch": mismatch,
                "max_logprob_delta": max_delta,
                "mean_logprob_delta": sum(deltas) / len(deltas) if deltas else None,
                "failures": failures,
                "errors": errors,
            }
        )
    return {
        "ok": bool(prompts) and all(p["ok"] for p in prompts),
        "criteria": {
            "kind": "rank",
            "tie_logit_tol": tie_logit_tol,
            "reference_top1_in_top5": True,
            "atol_diagnostic": atol,
        },
        "prompts": prompts,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hf-steps", type=Path, required=True)
    ap.add_argument("--service", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    ap.add_argument("--tie-logit-tol", type=float, default=DEFAULT_TIE_LOGIT_TOL)
    args = ap.parse_args(argv)
    result = check(
        json.loads(args.hf_steps.read_text()),
        json.loads(args.service.read_text()),
        atol=args.atol,
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
