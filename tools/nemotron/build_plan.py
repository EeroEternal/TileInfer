#!/usr/bin/env python3
"""Build a reviewable TileInfer plan for the Nemotron draft, without a device.

The draft is deliberately not in ``MANIFESTS`` because its benchmark block is still pending.
This tool lets the bring-up validate the plan/config gate before that promotion: it writes the
plan the bridge would consume, and reports whether the plan is servable.  A non-servable plan is
still useful (it records the gap); it must not be described as a supported deployment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tileinfer.manifests.nemotron_3_5 import NEMOTRON_3_5_DRAFT  # noqa: E402
from tileinfer.resolve.decide import decide  # noqa: E402
from tileinfer.resolve.probe import probe  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="/tmp/m5-nemotron-plan.json")
    ap.add_argument("--policy", default="vendor_first", choices=("vendor_first", "tileinfer_first"))
    args = ap.parse_args(argv)

    plan, report = decide(NEMOTRON_3_5_DRAFT, probe(), policy=args.policy)
    view = {
        "out": args.out,
        "plan_id": plan.plan_id,
        "model": plan.model.name,
        "architecture": plan.model.arch,
        "servable": report.servable,
        "gaps": [{"capability": g.capability, "required": g.required, "reason": g.reason} for g in report.gaps],
        "selections": [
            {"capability": s.capability, "provider": s.provider, "impl": s.impl} for s in plan.selections
        ],
        "access_points": plan.access_points,
        "note": "plan for bring-up only; not a support claim; draft is not in MANIFESTS",
    }
    plan.write(args.out)
    print(json.dumps(view, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
