#!/usr/bin/env python3
"""Turn a local NemotronH ``config.json`` into a declaration *draft*.

This is deliberately not a deployable manifest: benchmark top-5, KV layout and the final
capability list still need the reference run and review.  It refuses to invent any of them;
its only job is to remove the mechanical transcription errors when the checkpoint arrives.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PATTERN = {"M": "mamba", "*": "full_attention", "-": "mlp", "E": "moe"}
BLOCK = {"mamba": "M", "attention": "*", "mlp": "-", "moe": "E"}


def normalize_config(data: dict) -> dict:
    """Adapt the checkpoint's newer config keys to vLLM 0.23's NemotronHConfig.

    The checkpoint ships ``layers_block_type`` / ``mtp_layers_block_type``; vLLM 0.23's
    config class has a read-only ``layers_block_type`` property and expects
    ``hybrid_override_pattern``.  Passing the raw config raises AttributeError before any
    model code runs.  This adapter is a deployment shim; it does not edit the checkpoint.
    """
    out = dict(data)
    block = out.pop("layers_block_type", None)
    mtp_block = out.pop("mtp_layers_block_type", None)
    if block is not None:
        try:
            pattern = "".join(BLOCK[item] for item in block)
        except KeyError as exc:
            raise ValueError(f"unknown layers_block_type {exc.args[0]!r}") from None
        existing = out.get("hybrid_override_pattern")
        if existing is not None and existing != pattern:
            raise ValueError("hybrid_override_pattern disagrees with layers_block_type")
        out["hybrid_override_pattern"] = pattern
    if mtp_block is not None:
        try:
            mtp = "".join(BLOCK[item] for item in mtp_block)
        except KeyError as exc:
            raise ValueError(f"unknown mtp_layers_block_type {exc.args[0]!r}") from None
        existing = out.get("mtp_hybrid_override_pattern")
        if existing is not None and existing != mtp:
            raise ValueError("mtp_hybrid_override_pattern disagrees with mtp_layers_block_type")
        out["mtp_hybrid_override_pattern"] = mtp
    for src, dst in (
        ("n_groups", "mamba_n_groups"),
        ("conv_kernel", "mamba_d_conv"),
        ("expand", "mamba_expand"),
        ("chunk_size", "mamba_chunk_size"),
        ("time_step_min", "mamba_dt_min"),
        ("time_step_max", "mamba_dt_max"),
        ("time_step_limit", "mamba_dt_limit"),
        ("time_step_floor", "mamba_dt_init_floor"),
        ("use_conv_bias", "mamba_conv_bias"),
        ("norm_eps", "layer_norm_epsilon"),
    ):
        if src in out:
            value = out.pop(src)
            if dst not in out:
                out[dst] = value
    return out


def _pattern(data: dict) -> str:
    """Accept both vLLM's pattern string and the checkpoint's block-type list."""
    if "hybrid_override_pattern" in data:
        pattern = str(data["hybrid_override_pattern"])
    elif "layers_block_type" in data:
        try:
            pattern = "".join(BLOCK[item] for item in data["layers_block_type"])
        except KeyError as exc:
            raise ValueError(f"unknown layers_block_type {exc.args[0]!r}") from None
    else:
        raise ValueError("config has neither hybrid_override_pattern nor layers_block_type")
    n = int(data["num_hidden_layers"])
    if len(pattern) != n:
        raise ValueError(f"pattern length {len(pattern)} != num_hidden_layers {n}")
    return pattern


def parse_config(data: dict) -> dict:
    data = normalize_config(data)
    n = int(data["num_hidden_layers"])
    pattern = _pattern(data)
    try:
        layer_types = [PATTERN[ch] for ch in pattern]
    except KeyError as exc:
        raise ValueError(f"unknown pattern character {exc.args[0]!r}") from None

    num_heads = int(data["num_attention_heads"])
    num_kv_heads = int(data.get("num_key_value_heads", num_heads))
    if num_kv_heads <= 0 or num_heads % num_kv_heads:
        raise ValueError(f"num_attention_heads {num_heads} not divisible by num_kv_heads {num_kv_heads}")
    mamba_heads = int(data["mamba_num_heads"])
    mamba_groups = int(data["n_groups"] if "n_groups" in data else data["mamba_n_groups"])
    if mamba_groups <= 0 or mamba_heads % mamba_groups:
        raise ValueError(f"mamba_num_heads {mamba_heads} not divisible by mamba_n_groups {mamba_groups}")
    max_len = int(data.get("max_position_embeddings", 0))
    if max_len <= 0:
        raise ValueError("max_position_embeddings must be positive")

    out = {
        "arch": data.get("architectures", [""])[0],
        "model_type": data.get("model_type", ""),
        "num_layers": n,
        "layer_types": layer_types,
        "attention": {
            "kind": "gqa",
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": int(data.get("head_dim", data.get("attention_head_dim", 0))),
        },
        "mamba": {
            "kind": "mamba2",
            "num_heads": mamba_heads,
            "head_dim": int(data["mamba_head_dim"]),
            "state_size": int(data["ssm_state_size"]),
            "n_groups": mamba_groups,
            "conv_kernel": int(data["conv_kernel"] if "conv_kernel" in data else data["mamba_d_conv"]),
            "expand": int(data["expand"] if "expand" in data else data["mamba_expand"]),
        },
        "max_position_embeddings": max_len,
        "num_nextn_predict_layers": int(data.get("num_nextn_predict_layers", 0)),
        "mtp_pattern": _mtp_pattern(data),
        "mtp_weights": "unknown (declared layers do not prove the checkpoint has MTP heads)",
        "unresolved": [
            "benchmark.top5 (needs the HF/transformers reference run)",
            "kv layout (full attention + Mamba state pages)",
            "required_capabilities (review after the op audit)",
        ],
    }
    if "E" in pattern:
        try:
            experts = int(data["n_routed_experts"])
            top_k = int(data["num_experts_per_tok"])
            out["moe"] = {
                "num_experts": experts,
                "top_k": top_k,
                "intermediate_size": int(data["moe_intermediate_size"]),
            }
        except KeyError as exc:
            raise ValueError(f"pattern has E but config lacks MoE field {exc.args[0]!r}") from None
        if not (0 < top_k <= experts):
            raise ValueError(f"invalid MoE top_k {top_k} for {experts} experts")
    return out


def _mtp_pattern(data: dict) -> str:
    if "mtp_hybrid_override_pattern" in data:
        return str(data["mtp_hybrid_override_pattern"])
    if "mtp_layers_block_type" in data:
        try:
            return "".join(BLOCK[item] for item in data["mtp_layers_block_type"])
        except KeyError as exc:
            raise ValueError(f"unknown mtp_layers_block_type {exc.args[0]!r}") from None
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--normalized-out", default="", help="write a vLLM 0.23-compatible config")
    args = ap.parse_args()
    raw = json.loads(Path(args.config).read_text(encoding="utf-8"))
    payload = parse_config(raw)
    if args.normalized_out:
        Path(args.normalized_out).write_text(
            json.dumps(normalize_config(raw), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
