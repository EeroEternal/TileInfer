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


def parse_config(data: dict) -> dict:
    n = int(data["num_hidden_layers"])
    pattern = str(data.get("hybrid_override_pattern", ""))
    if len(pattern) != n:
        raise ValueError(f"pattern length {len(pattern)} != num_hidden_layers {n}")
    try:
        layer_types = [PATTERN[ch] for ch in pattern]
    except KeyError as exc:
        raise ValueError(f"unknown pattern character {exc.args[0]!r}") from None

    num_heads = int(data["num_attention_heads"])
    num_kv_heads = int(data.get("num_key_value_heads", num_heads))
    if num_kv_heads <= 0 or num_heads % num_kv_heads:
        raise ValueError(f"num_attention_heads {num_heads} not divisible by num_kv_heads {num_kv_heads}")
    mamba_heads = int(data["mamba_num_heads"])
    mamba_groups = int(data["mamba_n_groups"])
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
            "conv_kernel": int(data["mamba_d_conv"]),
            "expand": int(data["mamba_expand"]),
        },
        "max_position_embeddings": max_len,
        "num_nextn_predict_layers": int(data.get("num_nextn_predict_layers", 0)),
        "mtp_pattern": data.get("mtp_hybrid_override_pattern", ""),
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    payload = parse_config(json.loads(Path(args.config).read_text(encoding="utf-8")))
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
