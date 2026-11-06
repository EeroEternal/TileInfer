#!/usr/bin/env python3
"""S2 residual analysis: how large *is* the device deviation, in bf16 units?

For the two composed stages (``attn_out`` and the layer output ``x``) this prints, at the worst
element, the reference magnitude, one bf16 ULP at that magnitude, and the deviation in ULP, plus the
RMS-relative error and the fraction of elements that differ at all.  It exists to answer one
question: is a >1e-2 absolute deviation a *bug* or simply one bf16 ULP at a magnitude of 4-8?

    python tools/dsv41/s2_residual.py --layers 2
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tileinfer.models.dsv41 import Dsv41Config  # noqa: E402
from tileinfer.models.dsv41.layer import (  # noqa: E402
    LayerRuntime,
    SharedAttentionState,
    attention_forward,
)
from tileinfer.models.dsv41.modules import Dsv41Attention  # noqa: E402
from tools.dsv41.s2_layer_check import (  # noqa: E402
    copy_weights,
    fill_weights,
    official_model,
    oracle_args,
)

M = "/home/models/DeepSeek-V4.1-Flash"


def _bf16_ulp(value: float) -> float:
    if value == 0.0:
        return 0.0
    return 2.0 ** (math.floor(math.log2(abs(value))) - 8)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=M)
    parser.add_argument("--layers", default="2")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    cfg = Dsv41Config.from_json(os.path.join(args.model_dir, "config.json"))
    official = official_model(args.model_dir)
    torch.set_default_dtype(torch.bfloat16)
    import torch_npu  # noqa: F401

    try:
        torch.npu.matmul.allow_hf32 = False
    except Exception:  # noqa: BLE001
        pass

    for layer_id in [int(x) for x in args.layers.split(",") if x.strip()]:
        oracle_layer = official.Attention(layer_id, oracle_args(official, cfg, 32, 1))
        params = dict(oracle_layer.named_parameters())
        fill_weights(params, seed=1234 + 17 * layer_id)
        ours = Dsv41Attention(cfg, layer_id, device="cpu")
        copy_weights(params, dict(ours.named_parameters()))

        def run(device):
            module = ours.to(device) if device != "cpu" else ours
            rt = LayerRuntime(
                cfg=cfg,
                layer_id=layer_id,
                shared=SharedAttentionState(),
                bsz=1,
                max_seq_len=32,
                device=device,
            ).build()
            gen = torch.Generator().manual_seed(args.seed)
            x = torch.randn(1, 8, cfg.hidden_size, dtype=torch.bfloat16, generator=gen).to(device)
            rec = {}
            attention_forward(module, x, 0, rt, cfg, recorder=rec)
            return {k: v.detach().float().cpu() for k, v in rec.items()}

        cpu = run("cpu")
        npu = run("npu:0")
        print(f"\nlayer {layer_id}: composed stages, CPU (== official oracle) vs NPU")
        for stage in ("attn_out", "x"):
            ref, got = cpu[stage], npu[stage]
            diff = (got - ref).abs()
            worst = float(diff.max())
            idx = int(diff.flatten().argmax())
            ref_at = float(ref.flatten()[idx])
            ulp = _bf16_ulp(ref_at)
            rms_ref = float(ref.pow(2).mean().sqrt())
            rms_d = float(diff.pow(2).mean().sqrt())
            differing = float((diff > 0).float().mean()) * 100.0
            print(
                f"  {stage:9s} max|diff|={worst:.4e} at ref={ref_at:+.4e}"
                f" | 1 bf16 ulp there = {ulp:.3e} -> {worst / ulp if ulp else 0:.2f} ulp"
            )
            print(
                f"            rms|x|={rms_ref:.4f} rms|diff|={rms_d:.3e} "
                f"rel_rms={rms_d / max(rms_ref, 1e-30):.3e} | differing elements {differing:.2f}%"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
