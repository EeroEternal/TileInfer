#!/usr/bin/env python3
"""Audit the checkpoint's weight names against the declared layer sequence.

Reads only ``model.safetensors.index.json``; never opens a tensor.  This catches the
bring-up failures that happen before a single matmul:

* a layer whose checkpoint keys say MoE but whose declaration says Mamba (or vice versa);
* a missing ``gate`` / ``shared_experts`` / Mamba state parameter;
* an MTP branch whose keys do not match ``mtp_layers_block_type``;
* an index that points at shards that are not on disk.

It does not prove shapes or values - only names and shard presence.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from tileinfer.manifests.nemotron_3_5 import NEMOTRON_3_5_DRAFT

LAYER_RE = re.compile(r"^backbone\.layers\.(\d+)\.")


def _classify(keys: list[str]) -> str:
    if any(k.endswith(".mixer.A_log") or k.endswith(".mixer.in_proj.weight") for k in keys):
        return "mamba"
    if any(".mixer.experts." in k for k in keys):
        return "moe"
    if any(k.endswith(".mixer.q_proj.weight") for k in keys):
        return "attention"
    if any(".mixer.gate_proj." in k or ".mixer.up_proj." in k for k in keys):
        return "mlp"
    return "unknown"


def _expected_kind(declared: str) -> str:
    return {"full_attention": "attention", "mamba": "mamba", "moe": "moe", "mlp": "mlp"}[declared]


def audit(model_dir: Path) -> dict:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        return {"model_dir": str(model_dir), "index_present": False, "mapping_ok": False}
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map") or {}

    by_layer: dict[int, list[str]] = {}
    mtp: dict[int, list[str]] = {}
    other: list[str] = []
    for name in weight_map:
        m = LAYER_RE.match(name)
        if m:
            by_layer.setdefault(int(m.group(1)), []).append(name)
        elif name.startswith("mtp.layers."):
            parts = name.split(".")
            mtp.setdefault(int(parts[2]), []).append(name)
        else:
            other.append(name)

    errors: list[str] = []
    observed: list[str] = []
    declared = list(NEMOTRON_3_5_DRAFT.layer_types)
    for i, want in enumerate(declared):
        keys = by_layer.get(i, [])
        got = _classify(keys)
        observed.append(got)
        expected = _expected_kind(want)
        if got != expected:
            errors.append(f"layer {i}: declared {want} ({expected}), checkpoint keys look like {got}")
            continue
        if got == "mamba":
            required = {
                "backbone.layers.%d.mixer.A_log" % i,
                "backbone.layers.%d.mixer.D" % i,
                "backbone.layers.%d.mixer.dt_bias" % i,
                "backbone.layers.%d.mixer.in_proj.weight" % i,
                "backbone.layers.%d.mixer.out_proj.weight" % i,
                "backbone.layers.%d.mixer.norm.weight" % i,
                "backbone.layers.%d.norm.weight" % i,
            }
            missing = sorted(required - set(keys))
            if missing:
                errors.append(f"layer {i}: mamba keys missing {missing}")
        elif got == "moe":
            required = {
                "backbone.layers.%d.mixer.gate.weight" % i,
                "backbone.layers.%d.mixer.shared_experts.up_proj.weight" % i,
                "backbone.layers.%d.mixer.shared_experts.down_proj.weight" % i,
                "backbone.layers.%d.norm.weight" % i,
            }
            missing = sorted(required - set(keys))
            if missing:
                errors.append(f"layer {i}: MoE keys missing {missing}")
            experts = sorted(
                {
                    int(k.split(".mixer.experts.")[1].split(".")[0])
                    for k in keys
                    if ".mixer.experts." in k
                }
            )
            if experts != list(range(NEMOTRON_3_5_DRAFT.moe.num_experts)):
                errors.append(
                    f"layer {i}: experts {experts[:3]}..{experts[-3:] if experts else []} "
                    f"!= 0..{NEMOTRON_3_5_DRAFT.moe.num_experts - 1}"
                )
            if any("experts." in k and "gate_proj" in k for k in keys):
                errors.append(f"layer {i}: gated expert key present in a non-gated MoE checkpoint")
        elif got == "attention":
            required = {
                "backbone.layers.%d.mixer.q_proj.weight" % i,
                "backbone.layers.%d.mixer.k_proj.weight" % i,
                "backbone.layers.%d.mixer.v_proj.weight" % i,
                "backbone.layers.%d.mixer.o_proj.weight" % i,
                "backbone.layers.%d.norm.weight" % i,
            }
            missing = sorted(required - set(keys))
            if missing:
                errors.append(f"layer {i}: attention keys missing {missing}")

    if set(by_layer) != set(range(len(declared))):
        errors.append(f"checkpoint layer ids {sorted(by_layer)} != declared 0..{len(declared) - 1}")

    mtp_pattern = getattr(NEMOTRON_3_5_DRAFT, "mtp_layer_types", None) or ("full_attention", "moe")
    mtp_observed = []
    for i, want in enumerate(mtp_pattern):
        keys = mtp.get(i, [])
        got = _classify(keys)
        mtp_observed.append(got)
        expected = _expected_kind(want)
        if got != expected:
            errors.append(f"mtp layer {i}: declared {want}, checkpoint keys look like {got}")
        if got == "attention":
            for suffix in (".eh_proj.weight", ".enorm.weight", ".hnorm.weight", ".mixer.q_proj.weight"):
                if f"mtp.layers.{i}{suffix}" not in keys:
                    errors.append(f"mtp layer {i}: missing {suffix}")

    shards = sorted(set(weight_map.values()))
    local = {p.name for p in model_dir.glob("*.safetensors")}
    missing_shards = [s for s in shards if s not in local]
    empty_shards = [s for s in shards if s in local and (model_dir / s).stat().st_size == 0]

    return {
        "model_dir": str(model_dir),
        "index_present": True,
        "weight_keys": len(weight_map),
        "declared_layers": len(declared),
        "observed_kinds": dict(Counter(observed)),
        "declared_kinds": dict(Counter(_expected_kind(x) for x in declared)),
        "mtp_observed": mtp_observed,
        "mtp_keys": sum(len(v) for v in mtp.values()),
        "other_keys": len(other),
        "shards_declared": len(shards),
        "shards_present": len(local),
        "shards_missing": missing_shards,
        "shards_empty": empty_shards,
        "mapping_ok": not errors,
        "errors": errors,
        "weights_read": False,
        "hardware_used": False,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--strict", action="store_true", help="fail if mapping is not clean")
    args = ap.parse_args()
    payload = audit(Path(args.model))
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0 if (payload.get("mapping_ok") or not args.strict) else 1


if __name__ == "__main__":
    raise SystemExit(main())
