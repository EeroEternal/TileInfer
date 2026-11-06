"""Probe: can ``deep_gemm.fp8_fp4_gemm_nt`` replace our unpack+scale+bmm path on this NPU?

Why this matters (see ``docs/notes/dsv41-perf-plan.md``): our MoE expert GEMM currently spends
0.80 ms unpacking fp4 + 0.42 ms broadcasting the block scale + 0.18 ms on the bmm per layer/GEMM
(~1.40 ms).  DeepSeek's own production framework (DeepInfer, ``layers/linear.py``) does the same
computation in one call:

    deep_gemm.fp8_fp4_gemm_nt(x, weight, out, out_acc, recipe=(1, 1, gran_k))

If that call works here, the ~170 ms/token of unpack+scale disappears.  This probe is deliberately
tiny (M=8, N=256, K=256) so it can run while other tenants use the card, and it checks the result
against a straightforward dequantize-then-matmul reference in float32.

Run:  ASCEND_RT_VISIBLE_DEVICES=0 python tools/dsv41/perf_deepgemm_probe.py
Exit: 0 = kernel usable and numerically close, 1 = not usable (with the reason printed)
"""

from __future__ import annotations

import sys
import time

import torch

torch.manual_seed(0)
DEV = "npu:0"


def _e8m0_to_float(scale: torch.Tensor) -> torch.Tensor:
    """e8m0 stores a power of two in the byte; 127 -> 2^0."""
    return torch.pow(2.0, scale.view(torch.uint8).to(torch.int32).float() - 127.0)


def _dequant_fp4(packed: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Unpack packed fp4 (2 values per byte) the same way our runtime does."""
    from tileinfer.models.dsv41 import ops as dsv_ops

    o = dsv_ops.ops() if callable(getattr(dsv_ops, "ops", None)) else None
    if o is not None and hasattr(o, "unpack_fp4_lut"):
        return o.unpack_fp4_lut(packed, out_features=rows, in_features=cols).float()
    raise RuntimeError("no unpack_fp4_lut available for the reference path")


def main() -> int:
    print("=== 0. import deep_gemm ===")
    try:
        import deep_gemm
    except Exception as exc:  # noqa: BLE001
        print("import failed:", type(exc).__name__, exc)
        return 1
    print("deep_gemm:", getattr(deep_gemm, "__version__", "?"))
    fn = getattr(deep_gemm, "fp8_fp4_gemm_nt", None)
    print("fp8_fp4_gemm_nt:", "present" if fn else "MISSING")
    if fn is None:
        return 1

    M, N, K = 8, 256, 256
    BK = 32

    print("\n=== 1. build a tiny fp8(act) x fp4(weight) case ===")
    a_f = torch.randn(M, K, dtype=torch.float32)
    # per-token, block-32 quantization of the activations (mirrors act_quant)
    a_blk = a_f.view(M, K // BK, BK)
    a_scale_f = a_blk.abs().amax(dim=-1, keepdim=True).clamp_min(1e-9) / 448.0
    a_q = (a_blk / a_scale_f).clamp(-448, 448).to(torch.float8_e4m3fn).view(M, K)
    a_sf = a_scale_f.squeeze(-1).contiguous()  # [M, K/32]

    # weight: fp4 e2m1 in the same packed layout our loader produces
    b_f = torch.randn(N, K, dtype=torch.float32)
    b_blk = b_f.view(N, K // BK, BK)
    b_scale_f = b_blk.abs().amax(dim=-1, keepdim=True).clamp_min(1e-9) / 6.0
    b_q4 = (b_blk / b_scale_f).clamp(-6, 6)
    lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
    idx = (b_q4.unsqueeze(-1) - lut.unsqueeze(0).unsqueeze(0).unsqueeze(0)).abs().argmin(-1)
    b_packed = ((idx[..., 0::2] << 4) | idx[..., 1::2]).to(torch.uint8)   # [N, K/2] uint8
    b_sf = b_scale_f.squeeze(-1).contiguous()           # [N, K/32]

    a_q_d, a_sf_d = a_q.to(DEV), a_sf.to(DEV)
    b_packed_d, b_sf_d = b_packed.view(torch.float4_e2m1fn_x2).to(DEV), b_sf_d0 if False else b_sf.to(DEV)
    out = torch.zeros(M, N, dtype=torch.bfloat16, device=DEV)

    print(f"a {tuple(a_q_d.shape)} {a_q_d.dtype}  sfA {tuple(a_sf_d.shape)} {a_sf_d.dtype}")
    print(f"b {tuple(b_packed_d.shape)} {b_packed_d.dtype}  sfB {tuple(b_sf_d.shape)} {b_sf_d.dtype}")

    print("\n=== 2. reference: dequantize both sides, matmul in fp32 ===")
    a_ref = a_blk.reshape(M, K)
    b_ref = b_blk.reshape(N, K)
    ref = a_ref @ b_ref.t()

    print("\n=== 3. call the fused kernel (try the documented layouts) ===")
    fp4 = torch.float4_e2m1fn_x2
    b2d = b_packed.view(fp4).reshape(N, K // 2).to(DEV)     # [N, K/2] packed fp4
    k32 = K // BK
    # sf layout is ambiguous from the docs: try MK/KM for the activations and NK/KN for the weight
    sfA_opts = {"[M,K32]": a_sf_d, "[K32,M]": a_sf_d.t().contiguous()}
    sfB_opts = {"[N,K32]": b_sf_d, "[K32,N]": b_sf_d.t().contiguous()}
    t0 = time.time()
    ok = None
    for an, a_sf in sfA_opts.items():
        for bn, b_sf in sfB_opts.items():
            for recipe in [(1, 1, BK), (1, 1, 16)]:
                try:
                    out.zero_()
                    fn((a_q_d, a_sf), (b2d, b_sf), out, None, recipe=recipe)
                    torch.npu.synchronize()
                    ok = (an, bn, recipe)
                    print(f"ACCEPTED: sfA{an} sfB{bn} recipe{recipe}")
                    break
                except Exception as exc:  # noqa: BLE001
                    print(f"rejected sfA{an} sfB{bn} recipe{recipe} -> {str(exc).splitlines()[0][:100]}")
            if ok:
                break
        if ok:
            break
    if ok is None:
        return 1
    torch.npu.synchronize()
    dt = time.time() - t0
    got = out.float().cpu()
    rel = (got - ref).abs().max() / ref.abs().max().clamp_min(1e-6)
    print(f"kernel ok in {dt*1000:.1f} ms; max rel err = {float(rel):.4e}")
    print("PASS" if float(rel) < 5e-2 else "NUMERIC MISMATCH")
    return 0 if float(rel) < 5e-2 else 1


if __name__ == "__main__":
    sys.exit(main())
