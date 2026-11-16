#!/usr/bin/env python3
"""Emit the target's static gap report for two *declared* environments.

No checkpoint, no tensor read, no device, no service.  The two scenarios answer different
questions:

* ``vendor-absent``: what the declaration needs that neither side supplies.
* ``vendor-present-static``: what vllm-ascend advertises today; for Mamba2 this is still a
  ``state.linear`` gap because the vendor provider covers GDN only.

The environment is constructed from version anchors, not probed from a live accelerator, so
the output is a static prior - not evidence that the model runs.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from tileinfer.manifests.nemotron_3_5 import BENCHMARK_PENDING, NEMOTRON_3_5_DRAFT
from tileinfer.resolve.decide import decide
from tileinfer.resolve.probe import Environment


def _env(*, vendor: bool) -> Environment:
    return Environment(
        vllm="0.23.0",
        vllm_ascend="0.23.0" if vendor else None,
        torch="2.10.0+cpu",
        python="3.12",
        device_type="A5",
        has_vendor=vendor,
        notes={"provenance": "static version anchors; no device probe"},
    )


def build() -> dict:
    scenarios = {}
    for name, vendor in (("vendor-absent", False), ("vendor-present-static", True)):
        env = _env(vendor=vendor)
        plan, report = decide(NEMOTRON_3_5_DRAFT, env, policy="vendor_first")
        scenarios[name] = {
            "environment": env.as_dict(),
            "servable": report.servable,
            "gap_report": report.as_dict(),
            "plan": plan.as_dict(),
        }

    present = scenarios["vendor-present-static"]["gap_report"]
    required = [g["capability"] for g in present["gaps"] if g["required"]]
    assert required == ["state.linear"], required

    return {
        "model": NEMOTRON_3_5_DRAFT.name,
        "manifest_id": NEMOTRON_3_5_DRAFT.manifest_id,
        "benchmark_pending": BENCHMARK_PENDING,
        "hardware_used": False,
        "weights_read": False,
        "note": (
            "static prior only; the vendor-present scenario is a declaration of advertised "
            "coverage, not a live service result"
        ),
        "scenarios": scenarios,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="", help="write JSON here (default: stdout)")
    args = ap.parse_args()
    text = json.dumps(build(), ensure_ascii=False, indent=2) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
