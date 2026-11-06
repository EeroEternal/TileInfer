#!/usr/bin/env python3
"""S5: split the batched fp4 GEMM into its three phases, at real expert shapes.

After the batched decode path landed (TPOT 566 ms), the question is what is left inside it.  This
script times the three phases of ``_fp4_linear_batched`` separately at the real shapes
(``k=6`` experts, ``2304x5120`` and ``5120x2304``), so the remaining ~500 ms can be attributed without
loading the 476 GB checkpoint.

    python tools/dsv41/s5_profile_batched_gemm.py
"""

from __future__ import annotations

import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tools.dsv41 import ref_cpu  # noqa: E402


def bench(fn, iters: int = 5) -> float:
    fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return 1000 * (time.perf_counter() - t0) / iters


def main() -> int:
    import torch_npu  # noqa: F401

    dev = "npu:0"
    torch.set_default_dtype(torch.bfloat16)
    k = 6
    dim, hid = 5120, 2304

    gen = torch.Generator().manual_seed(0)
    # a real fp4 weight: quantise once on the host, then stack k copies on the card
    w = torch.randn(hid, dim, generator=gen)
    packed, scale = ref_cpu.quantize_weight_fp4_block32(w)
    # NPU has no repeat() for the fp4 storage dtype, so tile the bytes and view (as the arena does)
    packed_k = packed.to(dev).unsqueeze(0).repeat(k, 1, 1).view(torch.float4_e2m1fn_x2)
    scale = scale.to(dev)
    scale_k = scale.view(torch.uint8).unsqueeze(0).repeat(k, 1, 1).view(torch.float8_e8m0fnu)
    x = torch.randn(k, 1, dim, generator=gen).to(torch.bfloat16).to(dev)

    unpack_ms = bench(lambda: ref_cpu.unpack_fp4_lut(packed_k, k * hid, dim, dtype=torch.float32))
    flat = ref_cpu.unpack_fp4_lut(packed_k, k * hid, dim, dtype=torch.float32)
    scale_f = ref_cpu.scale_to_float(scale_k)

    def phase_scale():
        return flat.reshape(k, hid, dim // 32, 32) * scale_f.unsqueeze(-1)

    scale_ms = bench(phase_scale)
    wf = phase_scale().reshape(k, hid, dim)
    bmm_ms = bench(lambda: torch.bmm(x, wf.transpose(1, 2)))

    xq, xs = ref_cpu.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu, False)
    xf = xq.float() * ref_cpu.scale_to_float(xs).repeat_interleave(32, dim=-1)
    gemm_ms = bench(lambda: torch.bmm(xf, wf.transpose(1, 2)))

    print(f"k={k} experts, w1 shape {hid}x{dim} (packed {packed_k.shape})")
    print(f"  unpack_fp4_lut : {unpack_ms:7.2f} ms  ({flat.numel()/1e6:.1f}M values)")
    print(f"  scale broadcast: {scale_ms:7.2f} ms")
    print(f"  bmm (bf16)     : {bmm_ms:7.2f} ms")
    print(f"  bmm (fp32 x)   : {gemm_ms:7.2f} ms  <- what the path actually runs")
    per_call = unpack_ms + scale_ms + gemm_ms
    print(f"  sum per GEMM   : {per_call:7.2f} ms")
    print(f"\nextrapolated per token: 3 GEMMs x 40 layers x {per_call:.1f} ms = "
          f"{3*40*per_call/1000:.2f} s (measured MoE ~0.40-0.49 s, so the model's own shapes/shortcuts differ)")
    print("RESULT: measured")
    return 0


if __name__ == "__main__":
    sys.exit(main())
