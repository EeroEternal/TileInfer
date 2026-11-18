#!/usr/bin/env python3
"""Check C2 token/logprob evidence against the CPU HF reference.

C2 is a reference comparison for the no-vendor-baseline case.  The reference is a CPU
Transformers run in bf16; the served model is bf16 on Ascend.  A useful gate therefore has to
distinguish a real token error from a bf16 tie and from low-rank logprob drift:

* the served top-1 must be the HF top-1, or a token the HF run also scores as tied at the top;
* at least four of the five HF top-5 tokens must be present in the served top-5;
* the served top-1 logprob must be within ``--logprob-tol`` of the HF top-1;
* generated token ids are always reported; with ``--require-sequence`` they must match, except
  that the first divergence may be a documented HF tie (pass ``--hf-steps`` with every
  per-step top-5 to enable that branch).  The logprob gate itself does not require the whole
  sampled trajectory to match: C2 is a first-token distribution check, while MTP exactness is
  the token-for-token check.

Strict equality is reported separately.  This tool never turns a failed strict comparison into
a pass by itself; the tie-aware branch needs the explicit per-step evidence file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> Any:
    return json.loads(path.read_text())


def _service_top(entry: dict[str, Any]) -> dict[str, float]:
    top = entry.get("top_logprobs") or []
    if not top:
        raise ValueError(f"service entry for {entry.get('prompt')!r} has no top_logprobs")
    return {str(k): float(v) for k, v in top[0].items()}


def _sequence_ok(
    service_ids: list[int],
    hf: dict[str, Any],
    steps: dict[str, Any] | None,
    tie_logit_tol: float,
) -> tuple[bool, str]:
    hf_ids = list(hf["generated_token_ids"])
    if hf_ids == service_ids:
        return True, "ids identical"
    diff = next((i for i, (a, b) in enumerate(zip(hf_ids, service_ids)) if a != b), None)
    if diff is None:
        return False, "length differs with a common prefix"
    if steps is None:
        return False, f"first token divergence at {diff}, no per-step tie evidence"
    step = steps.get("steps", [])
    if diff >= len(step):
        return False, f"first token divergence at {diff}, no score for that step"
    candidates = step[diff]
    got = service_ids[diff]
    hit = [c for c in candidates if int(c["id"]) == got]
    if not hit:
        return False, f"divergence at {diff}: served {got} is not in HF top-5"
    best_logit = float(candidates[0]["logit"])
    if abs(float(hit[0]["logit"]) - best_logit) > tie_logit_tol:
        return False, (
            f"divergence at {diff}: served {got} logit {hit[0]['logit']} is not tied "
            f"with HF top-1 {candidates[0]['id']} (logit {best_logit})"
        )
    return True, f"tie-aware: divergence at {diff}, {got} tied with HF top-1"


def check(
    hf: dict[str, Any],
    service: list[dict[str, Any]],
    hf_steps: dict[str, Any] | None,
    *,
    logprob_tol: float,
    top5_min: int,
    tie_logit_tol: float,
    require_sequence: bool = False,
) -> dict[str, Any]:
    by_prompt = {str(x["prompt"]): x for x in service}
    rows = []
    for hf_row in hf["results"]:
        prompt = str(hf_row["prompt"])
        svc = by_prompt.get(prompt)
        if svc is None:
            rows.append({"prompt": prompt, "ok": False, "errors": ["missing service result"]})
            continue
        errors: list[str] = []
        hf_top = hf_row["first_token_top5"]
        hf_top1 = hf_top[0]
        svc_top = _service_top(svc)
        svc_top1 = max(svc_top, key=svc_top.get)
        svc_top1_lp = svc_top[svc_top1]
        d_lp = abs(svc_top1_lp - float(hf_top1["logprob"]))
        tied = [
            c
            for c in hf_top
            if c["token"] == svc_top1 and abs(float(c["logit"]) - float(hf_top1["logit"])) <= tie_logit_tol
        ]
        top1_ok = svc_top1 == hf_top1["token"] or bool(tied)
        if not top1_ok:
            errors.append(f"top-1 {svc_top1!r} != HF {hf_top1['token']!r} and is not tied")
        overlap = len(set(svc_top) & {c["token"] for c in hf_top})
        if overlap < top5_min:
            errors.append(f"top-5 overlap {overlap} < {top5_min}")
        if d_lp > logprob_tol:
            errors.append(f"top-1 logprob delta {d_lp:.6f} > {logprob_tol}")
        steps = (hf_steps or {}).get(prompt)
        seq_ok, seq_detail = _sequence_ok(
            [int(x) for x in svc.get("ids", [])], hf_row, steps, tie_logit_tol
        )
        if require_sequence and not seq_ok:
            errors.append(seq_detail)
        rows.append(
            {
                "prompt": prompt,
                "ok": not errors,
                "strict_ids": [int(x) for x in svc.get("ids", [])] == list(hf_row["generated_token_ids"]),
                "top1": svc_top1,
                "hf_top1": hf_top1["token"],
                "top1_tied": bool(tied),
                "top1_logprob_delta": d_lp,
                "top5_overlap": overlap,
                "sequence": seq_detail,
                "sequence_ok": seq_ok,
                "errors": errors,
            }
        )
    return {
        "ok": bool(rows) and all(r["ok"] for r in rows),
        "criteria": {
            "logprob_tol": logprob_tol,
            "top5_min": top5_min,
            "tie_logit_tol": tie_logit_tol,
            "require_sequence": require_sequence,
        },
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hf", type=Path, required=True, help="multi-prompt HF reference JSON")
    ap.add_argument("--service", type=Path, required=True, help="service C2 JSON (completion responses)")
    ap.add_argument("--hf-steps", type=Path, default=None, help="optional per-step HF tie evidence")
    ap.add_argument("--logprob-tol", type=float, default=0.5)
    ap.add_argument("--top5-min", type=int, default=4)
    ap.add_argument("--tie-logit-tol", type=float, default=0.125)
    ap.add_argument(
        "--require-sequence",
        action="store_true",
        help="fail when generated ids diverge (only documented HF ties are allowed)",
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    hf = _load(args.hf)
    service = _load(args.service)
    steps = _load(args.hf_steps) if args.hf_steps else None
    if isinstance(service, dict) and "results" in service:
        service = service["results"]
    result = check(
        hf,
        service,
        steps,
        logprob_tol=args.logprob_tol,
        top5_min=args.top5_min,
        tie_logit_tol=args.tie_logit_tol,
        require_sequence=args.require_sequence,
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
