"""Device probe (needs NPU): isolate WHERE the pre-NZ weight path breaks.

D45 found the pre-NZ route (GM(pre-NZ) -> UB(compact_nz) -> L1) lowers offline without the nd2nz
conversion, but on device it produces garbage.  This probe narrows the cause to a single hop:

    plain_nd    GM(ND) -> L1                          (reference)
    ub_plain    GM(ND) -> UB -> L1  (no annotation)   (UB staging, nd2nz-scatter path)
    ub_nz_pack  GM(NZ packed) -> UB(compact_nz) -> L1 (the pre-NZ route)

Result on the reference machine (2026-10-14):
    plain_nd    diff 0.0113   (exact)
    ub_plain    diff 0.0113   (exact -- UB staging itself is fine)
    ub_nz_pack  diff ~1e36    (WRONG)

so the failure is in the compact-NZ L1 image, not the UB hop and not the pack order alone: the
pre-NZ physical order needs an L0B-side geometry the raw ub2l1 post-copy does not produce.  This
sharpens E36's "pre-NZ direct load is not expressible" into a concrete, reproducible statement.

Usage::

    source .agent_loop/run/env.sh
    python benchmarks/probes/moe_prenz_device.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import tilelang  # noqa: E402
import tilelang.ascend.language as T  # noqa: E402
from tilelang.layout import make_ascend_compact_nz_layout  # noqa: E402
from tilelang.language import annotate_layout  # noqa: E402

from tileinfer.kernels.moe.prenz import pack_nd_to_compact_nz  # noqa: E402
from tileinfer.utils import is_npu_available  # noqa: E402

M, N, KK, ROWS = 32, 64, 512, 16
DT = "bfloat16"


def _kernel(use_ub: bool, use_nz: bool):
    def build():
        @T.prim_func
        def main(W: T.Tensor((N, KK), DT), X: T.Tensor((M, KK), DT), Out: T.Tensor((M, N), DT)):
            with T.Kernel(1) as bx:
                w_l1 = T.alloc_l1((N, KK), DT)
                x_l1 = T.alloc_l1((M, KK), DT)
                a = T.alloc_l0a((M, KK), DT)
                b = T.alloc_l0b((N, KK), DT)
                acc = T.alloc_l0c((M, N), "float32")
                o = T.alloc_shared((ROWS, N), "float32")
                oc = T.alloc_shared((ROWS, N), DT)
                if use_ub:
                    w_ub = T.alloc_shared((N + 1 if use_nz else N, KK), DT)
                    if use_nz:
                        annotate_layout({w_ub: make_ascend_compact_nz_layout(w_ub)})
                    T.copy(W[0:N, 0:KK], w_ub[0:N, 0:KK])
                    T.copy(w_ub[0:N, 0:KK], w_l1)
                else:
                    T.copy(W[0:N, 0:KK], w_l1)
                T.copy(X[0:M, 0:KK], x_l1)
                T.copy(x_l1, a)
                T.copy(w_l1, b)
                T.gemm(a, b, acc, transpose_B=True, clear_accum=True)
                T.dual_copy(acc, o)
                with T.SimtVF(threads=128):
                    for r, c in T.Parallel(ROWS, N):
                        oc[r, c] = T.Cast(DT, o[r, c])
                T.dual_copy(oc[0:ROWS, 0:N], Out[0:M, 0:N])

        return main

    return build


def _run(build, w, x):
    k = tilelang.compile(build(), out_idx=[])
    out = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    k(w.contiguous(), x.contiguous(), out)
    torch.npu.synchronize()
    return out.float()


def main():
    if not is_npu_available():
        print("no NPU available; this probe needs the device")
        return 1
    torch.manual_seed(0)
    w = (torch.randn((N, KK), dtype=torch.float32) * 0.1).to("npu").to(torch.bfloat16)
    x = (torch.randn((M, KK), dtype=torch.float32) * 0.5).to("npu").to(torch.bfloat16)
    ref = x.float() @ w.float().T

    o0 = _run(_kernel(False, False), w, x)
    print(f"plain_nd   diff {(o0 - ref).abs().max().item():.4f}")
    o1 = _run(_kernel(True, False), w, x)
    print(f"ub_plain   diff {(o1 - ref).abs().max().item():.4f}")
    o2 = _run(_kernel(True, True), pack_nd_to_compact_nz(w), x)
    d = (o2 - ref).abs().max().item()
    print(f"ub_nz_pack diff {d:.4f}  (expected: large -- the negative result)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
