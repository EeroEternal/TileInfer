"""``python -m tileinfer.cli.plan`` - build / show / verify the deployment plan."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional

from ..contracts.plan import Plan
from ..manifests import find_manifest
from ..resolve.decide import POLICIES, decide
from ..resolve.probe import probe


def _read_config(model_path: str) -> Dict[str, Any]:
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, encoding="utf-8") as fh:
        return json.load(fh)


def manifest_key_from_config(config: Dict[str, Any]) -> str:
    """Pick the declaration key from the checkpoint config (arch, then model_type)."""
    architectures = config.get("architectures") or []
    if architectures:
        return str(architectures[0])
    text = config.get("text_config") or {}
    if text.get("model_type"):
        return str(text["model_type"])
    return str(config.get("model_type", ""))


def build(args: argparse.Namespace) -> int:
    config = _read_config(args.model)
    key = args.manifest_id or manifest_key_from_config(config)
    manifest = find_manifest(key)
    env = probe()
    plan, report = decide(manifest, env, policy=args.policy)

    plan.write(args.out)
    gaps_path = args.gaps or (os.path.splitext(args.out)[0] + ".gaps.json")
    report.write(gaps_path)
    print(plan.summary())
    print(report.summary())
    for gap in report.gaps:
        print(f"GAP {gap.capability}: {gap.reason}")
    if not report.servable:
        print("NOT SERVABLE on this machine", file=sys.stderr)
        return 2
    return 0


def show(args: argparse.Namespace) -> int:
    plan = Plan.load(args.plan)
    data = plan.as_dict()
    data["summary"] = plan.summary()
    print(json.dumps(data, ensure_ascii=False, indent=2))
    return 0


def verify(args: argparse.Namespace) -> int:
    plan = Plan.load(args.plan)
    env = probe()
    live = {
        "vllm": env.vllm or "missing",
        "vllm_ascend": env.vllm_ascend or "missing",
        "python": env.python,
    }
    want = {
        "vllm": plan.versions.vllm,
        "vllm_ascend": plan.versions.vllm_ascend,
        "python": plan.versions.python,
    }
    mismatches = {k: (want[k], live[k]) for k in want if want[k] != live[k]}
    if mismatches:
        print("VERSION MISMATCH (plan -> live):", json.dumps(mismatches, ensure_ascii=False))
        return 1
    print("OK", plan.summary())
    return 0


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(prog="tileinfer.cli.plan", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="probe the machine and write a plan file")
    p_build.add_argument("--model", required=True, help="checkpoint directory containing config.json")
    p_build.add_argument("--out", required=True, help="plan JSON to write")
    p_build.add_argument("--gaps", default=None, help="gap report JSON (default: <out>.gaps.json)")
    p_build.add_argument("--manifest-id", default=None, help="override the declaration lookup key")
    p_build.add_argument("--policy", default="vendor_first", choices=POLICIES)
    p_build.set_defaults(func=build)

    p_show = sub.add_parser("show", help="print a plan")
    p_show.add_argument("plan")
    p_show.set_defaults(func=show)

    p_verify = sub.add_parser("verify", help="compare a plan's version triangle with this machine")
    p_verify.add_argument("plan")
    p_verify.set_defaults(func=verify)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
