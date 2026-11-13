#!/usr/bin/env python3
"""Read-only static audit for the Nemotron M5 bring-up.

No vLLM import, no NPU, no service.  It only reads the vendored source trees so the
offline phase can state what is present and what is still unknown before a card is used.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def _line(path: Path, needle: str) -> str:
    for i, line in enumerate(_read(path).splitlines(), 1):
        if needle in line:
            return f"{path}:{i}: {line.strip()}"
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--prefix",
        default=".venv_v3/lib/python3.12/site-packages",
        help="site-packages containing vllm and vllm_ascend",
    )
    ap.add_argument("--out", default="", help="write JSON here (default: stdout)")
    ap.add_argument("--strict", action="store_true", help="exit 1 if a required check is missing")
    args = ap.parse_args()

    root = Path(args.prefix)
    vllm = root / "vllm"
    asc = root / "vllm_ascend"
    registry = vllm / "model_executor/models/registry.py"
    model = vllm / "model_executor/models/nemotron_h.py"
    mtp = vllm / "model_executor/models/nemotron_h_mtp.py"
    config = vllm / "transformers_utils/configs/nemotron_h.py"
    spec_patch = asc / "patch/platform/patch_speculative_config.py"
    asc_models = asc / "models/__init__.py"
    mamba = asc / "ops/triton/mamba"
    mamba_config = asc / "patch/platform/patch_mamba_config.py"
    triton_patch = asc / "patch/worker/patch_triton.py"
    mamba2_hits = [
        str(path.relative_to(root))
        for path in asc.rglob("*.py")
        if "mamba_mixer2" in _read(path) or "ssd_combined" in _read(path)
    ]
    project = Path(__file__).resolve().parents[2]
    ti_mamba = project / "tileinfer/vllm/ops/mamba2.py"
    ti_moe = project / "tileinfer/vllm/ops/moe_guard.py"

    checks = [
        {
            "id": "vllm_model_registry",
            "ok": bool(re.search(r'"NemotronHForCausalLM"', _read(registry))),
            "evidence": _line(registry, '"NemotronHForCausalLM"'),
            "meaning": "vLLM 0.23 has the NemotronH model class",
        },
        {
            "id": "vllm_mtp_registry",
            "ok": bool(re.search(r'"NemotronHMTPModel"', _read(registry))),
            "evidence": _line(registry, '"NemotronHMTPModel"'),
            "meaning": "vLLM 0.23 has the NemotronH MTP class",
        },
        {
            "id": "vllm_model_file",
            "ok": model.is_file() and "MambaMixer2" in _read(model),
            "evidence": str(model) if model.is_file() else "",
            "meaning": "model forward includes the Mamba2 mixer",
        },
        {
            "id": "vllm_mtp_file",
            "ok": mtp.is_file() and "NemotronHMTP" in _read(mtp),
            "evidence": str(mtp) if mtp.is_file() else "",
            "meaning": "MTP model file is present",
        },
        {
            "id": "vllm_config",
            "ok": config.is_file() and 'model_type = "nemotron_h"' in _read(config),
            "evidence": _line(config, 'model_type = "nemotron_h"'),
            "meaning": "NemotronHConfig is present",
        },
        {
            "id": "ascend_spec_patch",
            "ok": "nemotron_h_mtp" in _read(spec_patch),
            "evidence": _line(spec_patch, "nemotron_h_mtp"),
            "meaning": "Ascend spec patch maps nemotron_h to MTP",
        },
        {
            "id": "ascend_model_registration",
            "ok": "Nemotron" in _read(asc_models),
            "evidence": _line(asc_models, "Nemotron"),
            "meaning": "Ascend registers a Nemotron model class (currently expected: absent)",
        },
        {
            "id": "ascend_mamba_ops",
            "ok": mamba.is_dir() and any(mamba.glob("*.py")),
            "evidence": str(mamba) if mamba.is_dir() else "",
            "meaning": "Ascend has Mamba Triton op files (model-level support still unknown)",
        },
        {
            "id": "ascend_hybrid_mamba_config",
            "ok": mamba_config.is_file() and "HybridAttentionMambaModelConfig" in _read(mamba_config),
            "evidence": _line(mamba_config, "HybridAttentionMambaModelConfig"),
            "meaning": "Ascend patches hybrid attention/Mamba cache config (GDN-oriented)",
        },
        {
            "id": "ascend_causal_conv1d_patch",
            "ok": triton_patch.is_file() and "causal_conv1d_update_npu" in _read(triton_patch),
            "evidence": _line(triton_patch, "causal_conv1d_update_npu"),
            "meaning": "Ascend replaces causal_conv1d update for NPU",
        },
        {
            "id": "ascend_mamba2_mixer_patch",
            "ok": bool(mamba2_hits),
            "evidence": ", ".join(mamba2_hits[:5]),
            "meaning": (
                "Ascend has a Mamba2 mixer/SSD path (expected: absent; absence is a gap "
                "candidate, not proof that the model cannot run)"
            ),
        },
        {
            "id": "tileinfer_mamba2_ops",
            "ok": ti_mamba.is_file()
            and "def mamba_chunk_scan_combined_varlen" in _read(ti_mamba)
            and "def selective_state_update" in _read(ti_mamba),
            "evidence": str(ti_mamba) if ti_mamba.is_file() else "",
            "meaning": (
                "TileInfer torch Mamba2 replacement exists (chunk scan + selective state update); "
                "target-shape numerics and graph capture are still unknown"
            ),
        },
        {
            "id": "tileinfer_non_gated_moe",
            "ok": ti_moe.is_file()
            and "install_ascend_non_gated_experts" in _read(ti_moe)
            and "relu2_no_mul" in _read(project / "tileinfer/vllm/ops/moe.py"),
            "evidence": str(ti_moe) if ti_moe.is_file() else "",
            "meaning": (
                "TileInfer routes non-gated relu2 experts through fused_experts on Ascend; "
                "target-shape and multi-rank behaviour are still unknown"
            ),
        },
    ]

    unknown = [
        "target checkpoint config.json (layers/pattern/MTP/context)",
        "whether the checkpoint actually contains MTP heads/weights",
        "whether Ascend can execute the NemotronH model end to end",
        "graph capture for Mamba state / hybrid attention",
        "hybrid KV/MambaSpec and weight mapping on Ascend",
        "reference implementation runtime on CPU",
        "target-shape Mamba2 numerics vs an independent implementation",
        "target-shape non-gated MoE numerics vs an independent implementation",
    ]
    payload = {
        "model": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
        "static_only": True,
        "hardware_used": False,
        "prefix": str(root),
        "checks": checks,
        "unknown": unknown,
        "static_servable": False,
        "note": (
            "static prior only: absence of Ascend registration does not prove the model "
            "cannot run; presence of a TileInfer replacement is not target-shape evidence"
        ),
    }

    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        print(text, end="")

    if args.strict and not all(c["ok"] for c in checks):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
