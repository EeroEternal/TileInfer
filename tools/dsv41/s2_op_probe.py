#!/usr/bin/env python3
"""S2 op-level isolation: which single torch op differs between CPU and NPU?

``s2_attn_debug.py`` narrows it to a stage; this isolates the operator inside that stage by replaying
the *recorded* CPU tensors through the same reference functions on both devices:

* ``sparse_attn`` (fp32 matmuls, exp, bf16 p-cast, sink in the denominator),
* ``act_quant`` + ``fp8_gemm`` (the fp8 path into ``wo_b``),
* ``rotate`` (cos/sin path).

    python tools/dsv41/s2_op_probe.py --layers 2
"""

from __future__ import annotations

import argparse
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
from tools.dsv41 import ref_cpu  # noqa: E402
from tools.dsv41.s2_layer_check import (  # noqa: E402
    copy_weights,
    fill_weights,
    official_model,
    oracle_args,
)

M = "/home/models/DeepSeek-V4.1-Flash"


def _cmp(name, ref, got):
    ref = ref.float()
    got = got.float().to(ref.dtype)
    diff = (got - ref).abs()
    _, exponent = torch.frexp(ref.abs())
    ulp = torch.pow(2.0, (exponent - 8).float()).clamp_min(1e-38)
    print(
        f"  {name:26s} max={float(diff.max()):.3e} mean={float(diff.mean()):.3e} "
        f"max_ulp={float((diff/ulp).max()):.2f} "
        f"{'(bit-exact)' if float(diff.max()) == 0 else ''}"
    )


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
    dev = "npu:0"

    for layer_id in [int(x) for x in args.layers.split(",") if x.strip()]:
        oracle_layer = official.Attention(layer_id, oracle_args(official, cfg, 32, 1))
        params = dict(oracle_layer.named_parameters())
        fill_weights(params, seed=1234 + 17 * layer_id)
        ours = Dsv41Attention(cfg, layer_id, device="cpu")
        copy_weights(params, dict(ours.named_parameters()))

        rt = LayerRuntime(
            cfg=cfg,
            layer_id=layer_id,
            shared=SharedAttentionState(),
            bsz=1,
            max_seq_len=32,
            device="cpu",
        ).build()
        gen = torch.Generator().manual_seed(args.seed)
        x = torch.randn(1, 8, cfg.hidden_size, dtype=torch.bfloat16, generator=gen)
        rec = {}
        attention_forward(ours, x, 0, rt, cfg, recorder=rec)

        q, kv, idxs = rec["q"], torch.cat([rec["window_kv"], rec["compress_kv"]], dim=1), rec["idxs"]
        x_in, wo_a_out = x, rec["attn_out"]
        print(f"\nlayer {layer_id}: same inputs, CPU vs NPU, per operator")

        print("  [sparse_attn]")
        o_cpu = ref_cpu.sparse_attn(q, kv, ours.attn_sink, idxs, cfg.head_dim**-0.5)
        o_npu = ref_cpu.sparse_attn(
            q.to(dev), kv.to(dev), ours.attn_sink.to(dev), idxs.to(dev), cfg.head_dim**-0.5
        ).cpu()
        _cmp("sparse_attn out", o_cpu, o_npu)

        print("  [activation quantisation + fp8 gemm]")
        groups = cfg.o_groups
        o_flat = torch.einsum(
            "bsgd,grd->bsgr",
            wo_a_out.view(1, 8, groups, -1),
            ours.wo_a.weight.view(groups, cfg.o_lora_rank, -1),
        ).flatten(2)
        xq_cpu, s_cpu = ref_cpu.act_quant(
            o_flat.float(), 32, "ue8m0", torch.float8_e8m0fnu, False
        )
        xq_npu, s_npu = ref_cpu.act_quant(
            o_flat.float().to(dev), 32, "ue8m0", torch.float8_e8m0fnu, False
        )
        xq_npu, s_npu = xq_npu.cpu(), s_npu.cpu()
        _cmp("quantised activation", xq_cpu.float(), xq_npu.float())
        flips = int((s_cpu != s_npu).sum())
        print(
            f"  {'e8m0 scale blocks':26s} differing={flips} of {s_cpu.numel()} "
            f"({100.0*flips/max(s_cpu.numel(),1):.3f}%)  <-- a flip changes a whole 32-element block"
        )
        c_cpu = ref_cpu.fp8_gemm(
            xq_cpu, s_cpu, ours.wo_b.weight, ours.wo_b.scale, torch.float8_e8m0fnu, 32
        )
        c_npu = ref_cpu.fp8_gemm(
            xq_cpu.to(dev), s_cpu.to(dev), ours.wo_b.weight.to(dev), ours.wo_b.scale.to(dev),
            torch.float8_e8m0fnu, 32,
        ).cpu()
        _cmp("fp8_gemm(wo_b) same act", c_cpu, c_npu)
        _cmp("wo_a einsum", o_flat, torch.einsum(
            "bsgd,grd->bsgr",
            wo_a_out.view(1, 8, groups, -1).to(dev),
            ours.wo_a.weight.view(groups, cfg.o_lora_rank, -1).to(dev),
        ).cpu().flatten(2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
