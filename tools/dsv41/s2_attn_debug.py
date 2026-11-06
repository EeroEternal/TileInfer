#!/usr/bin/env python3
"""S2 diagnosis: where exactly does the NPU run diverge from the CPU reference?

``s2_layer_check.py`` answers "how far off is the device run"; this answers "at which stage".
It runs the *same* layer with the *same* weights and the *same* seeded input twice -- once on the
CPU (which is bit-identical to the official oracle, see ``s2_layer_check.py``) and once on the NPU --
and compares the stage tensors our forward records: ``qr``, ``q``, ``window_kv``, ``compress_kv``,
``compress_idxs``, ``idxs``, ``attn_out``, ``x``.

The first stage whose deviation jumps from "0 ulp" / "a few ulp" to "materially different" is the
operator to fix.  Stages before it are bit-clean on the device, so the fix is local.

    python tools/dsv41/s2_attn_debug.py --layers 2
    python tools/dsv41/s2_attn_debug.py --layers 2,20 --steps 8,1,1,1
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
STAGES = ("qr", "q", "window_kv", "compress_kv", "compress_idxs", "idxs", "attn_out", "x")


def _stats(ref: torch.Tensor, got: torch.Tensor):
    ref = ref.float()
    got = got.float()
    if ref.dtype != got.dtype:
        got = got.to(ref.dtype)
    diff = (got - ref).abs()
    _, exponent = torch.frexp(ref.abs())
    ulp = torch.pow(2.0, (exponent - 8).float()).clamp_min(1e-38)
    return float(diff.max()), float(diff.mean()), float((diff / ulp).max())


def run_steps(module, cfg, layer_id, steps, device, seed=7):
    """Run one layer through the step sequence on ``device``, returning per-step stage records."""
    rt = LayerRuntime(
        cfg=cfg,
        layer_id=layer_id,
        shared=SharedAttentionState(),
        bsz=1,
        max_seq_len=32,
        device=device,
    ).build()
    gen = torch.Generator().manual_seed(seed)
    total = sum(steps)
    x_seq = torch.randn(1, total, cfg.hidden_size, dtype=torch.bfloat16, generator=gen)
    out = []
    pos = 0
    for seqlen in steps:
        rec = {}
        x = x_seq[:, pos : pos + seqlen].to(device)
        attention_forward(module, x, pos, rt, cfg, recorder=rec)
        out.append({k: v.detach().float().cpu() for k, v in rec.items()})
        pos += seqlen
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=M)
    parser.add_argument("--layers", default="2")
    parser.add_argument("--steps", default="8,1,1,1")
    parser.add_argument("--device", default="npu")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    cfg = Dsv41Config.from_json(os.path.join(args.model_dir, "config.json"))
    official = official_model(args.model_dir)
    steps = [int(x) for x in args.steps.split(",") if x.strip()]
    torch.set_default_dtype(torch.bfloat16)
    if args.device == "npu":
        import torch_npu  # noqa: F401

        try:
            torch.npu.matmul.allow_hf32 = False
        except Exception:  # noqa: BLE001
            pass

    for layer_id in [int(x) for x in args.layers.split(",") if x.strip()]:
        # one weight set, shared by both devices
        oracle_layer = official.Attention(layer_id, oracle_args(official, cfg, 32, 1))
        params = dict(oracle_layer.named_parameters())
        fill_weights(params, seed=1234 + 17 * layer_id)
        ours = Dsv41Attention(cfg, layer_id, device="cpu")
        copy_weights(params, dict(ours.named_parameters()))

        cpu_records = run_steps(ours, cfg, layer_id, steps, "cpu", seed=args.seed)
        device_module = ours.to(args.device)
        dev_records = run_steps(device_module, cfg, layer_id, steps, args.device, seed=args.seed)

        print(f"\nlayer {layer_id}: stage-by-stage, CPU (== official oracle) vs {args.device}")
        print(f"  {'stage':16s} {'step':>4s} {'shape':>18s} {'max|diff|':>12s} "
              f"{'mean|diff|':>12s} {'max ulp':>10s}")
        for step_index, (cpu_rec, dev_rec) in enumerate(zip(cpu_records, dev_records)):
            for name in STAGES:
                if name not in cpu_rec or name not in dev_rec:
                    continue
                ref, got = cpu_rec[name], dev_rec[name]
                if ref.dtype != got.dtype or ref.shape != got.shape:
                    print(f"  {name:16s} {step_index:4d}  !! dtype/shape {ref.shape}/{ref.dtype} "
                          f"vs {got.shape}/{got.dtype}")
                    continue
                if ref.dtype == torch.float32 and ref.numel() == 0:
                    continue
                mx, mean, ulp = _stats(ref, got)
                exact = " (bit-exact)" if mx == 0.0 else ""
                print(f"  {name:16s} {step_index:4d} {str(tuple(ref.shape)):>18s} "
                      f"{mx:12.3e} {mean:12.3e} {ulp:10.2f}{exact}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
