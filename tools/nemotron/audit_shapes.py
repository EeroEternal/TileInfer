#!/usr/bin/env python3
"""Check the shapes of the tensors whose headers are already on disk.

Reads only safetensors headers.  The expected shapes are derived from the real config plus
the declared layer pattern; this catches the classic bring-up bug where a config field is
accepted but the checkpoint was written with a different projection width.

Only present shards are checked.  ``complete=false`` is expected while the download is
partial; ``shape_ok`` refers to what is actually on disk.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
from pathlib import Path

LAYER_RE = re.compile(r"^backbone\.layers\.(\d+)\.mixer\.(.+)$")
MTP_RE = re.compile(r"^mtp\.layers\.(\d+)\.mixer\.(.+)$")


def _read_header(path: Path) -> dict:
    with path.open("rb") as fh:
        header_len = struct.unpack("<Q", fh.read(8))[0]
        raw = fh.read(header_len)
    return {k: v for k, v in json.loads(raw).items() if k != "__metadata__"}


def _expect(config: dict, layer_kind: str, suffix: str) -> tuple[int, ...] | None:
    hidden = int(config["hidden_size"])
    vocab = int(config["vocab_size"])
    if suffix == "embed_tokens.weight" or suffix == "lm_head.weight":
        return (vocab, hidden)
    if suffix == "norm.weight" and layer_kind != "mamba":
        return (hidden,)

    mamba_heads = int(config["mamba_num_heads"])
    mamba_head_dim = int(config["mamba_head_dim"])
    state = int(config["ssm_state_size"])
    groups = int(config["n_groups"])
    conv_kernel = int(config["conv_kernel"])
    intermediate = mamba_heads * mamba_head_dim
    conv_channels = intermediate + 2 * groups * state
    in_proj_out = 2 * intermediate + 2 * groups * state + mamba_heads

    if layer_kind == "mamba":
        table = {
            "A_log": (mamba_heads,),
            "D": (mamba_heads,),
            "dt_bias": (mamba_heads,),
            "conv1d.weight": (conv_channels, 1, conv_kernel),
            "conv1d.bias": (conv_channels,),
            "in_proj.weight": (in_proj_out, hidden),
            "out_proj.weight": (hidden, intermediate),
            "norm.weight": (intermediate,),
        }
        return table.get(suffix)

    if layer_kind == "moe":
        experts = int(config["n_routed_experts"])
        moe_inter = int(config["moe_intermediate_size"])
        shared_inter = int(config.get("moe_shared_expert_intermediate_size") or 0)
        table = {
            "gate.weight": (experts, hidden),
            "gate.e_score_correction_bias": (experts,),
            "shared_experts.up_proj.weight": (shared_inter, hidden),
            "shared_experts.down_proj.weight": (hidden, shared_inter),
        }
        m = re.match(r"experts\.(\d+)\.(up_proj|down_proj)\.weight$", suffix)
        if m:
            return (moe_inter, hidden) if m.group(2) == "up_proj" else (hidden, moe_inter)
        return table.get(suffix)

    if layer_kind == "attention":
        q_heads = int(config["num_attention_heads"])
        kv_heads = int(config["num_key_value_heads"])
        head_dim = int(config.get("head_dim") or hidden // q_heads)
        table = {
            "q_proj.weight": (q_heads * head_dim, hidden),
            "k_proj.weight": (kv_heads * head_dim, hidden),
            "v_proj.weight": (kv_heads * head_dim, hidden),
            "o_proj.weight": (hidden, q_heads * head_dim),
        }
        return table.get(suffix)

    if layer_kind == "mlp":
        inter = int(config["intermediate_size"])
        table = {
            "up_proj.weight": (inter, hidden),
            "down_proj.weight": (hidden, inter),
            "gate_proj.weight": (inter, hidden),
        }
        return table.get(suffix)
    return None


def _kind_from_config(config: dict, index: int) -> str:
    block = config.get("layers_block_type") or []
    raw = block[index] if index < len(block) else ""
    return {"mamba": "mamba", "moe": "moe", "attention": "attention", "mlp": "mlp"}.get(raw, "unknown")


def audit(model_dir: Path) -> dict:
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    pattern = list(config.get("layers_block_type") or [])
    mtp_pattern = list(config.get("mtp_layers_block_type") or [])
    errors: list[str] = []
    checked = 0
    skipped = 0
    for path in sorted(model_dir.glob("*.safetensors")):
        header = _read_header(path)
        for name, entry in header.items():
            shape = tuple(entry.get("shape") or ())
            m = LAYER_RE.match(name)
            mtp = MTP_RE.match(name)
            expected = None
            if m:
                layer = int(m.group(1))
                kind = _kind_from_config(config, layer)
                expected = _expect(config, kind, m.group(2))
            elif mtp:
                layer = int(mtp.group(1))
                raw = mtp_pattern[layer] if layer < len(mtp_pattern) else ""
                kind = {"attention": "attention", "moe": "moe"}.get(raw, "unknown")
                expected = _expect(config, kind, mtp.group(2))
            elif name in ("backbone.embeddings.weight", "lm_head.weight"):
                expected = _expect(config, "", "embed_tokens.weight" if "embeddings" in name else "lm_head.weight")
            else:
                skipped += 1
                continue
            if expected is None:
                skipped += 1
                continue
            checked += 1
            if shape != expected:
                errors.append(f"{path.name}: {name} shape {shape} != expected {expected}")
    return {
        "model_dir": str(model_dir),
        "layers_declared": len(pattern),
        "mtp_layers_declared": len(mtp_pattern),
        "tensors_checked": checked,
        "tensors_skipped": skipped,
        "shape_ok": not errors,
        "errors": errors,
        "weights_read": False,
        "hardware_used": False,
        "note": "safetensors headers only; missing shards are not covered",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()
    payload = audit(Path(args.model))
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0 if (payload["shape_ok"] or not args.strict) else 1


if __name__ == "__main__":
    raise SystemExit(main())
