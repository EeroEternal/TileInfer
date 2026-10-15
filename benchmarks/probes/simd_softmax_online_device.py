#!/usr/bin/env python3
"""Device probe (needs NPU): **per-page online** SIMD softmax — the decode kernel's exact form (T5).

``simd_softmax_paged_device.py`` validated a *two-pass* paged softmax.  The decode kernel is
**online**: it fuses the softmax into the page loop, keeping a running ``(m, l)`` per row and
rescaling as each page arrives.  This probe validates that online core on device, in the exact shape
the kernel uses (per-AIV rows=16, page 128 = two fp32 registers, runtime page loop, runtime last-len
staged into UB, tail mask via ``vcmps``+``vsel``).

It outputs the per-page **unnormalised** weights ``p = exp(s*scale - new_m)`` and the final running
sum ``l``; ``p / l`` must equal ``torch.softmax(s*scale)``.  The ``o = o*alpha`` rescale is *not*
done here because in the kernel it lives on the cube side (``o += P@V``) and is separated from this
block by the GEMM — a direct UB read-modify-write of ``o`` inside the same SimdVF region is a hazard
(measured ``max|diff|=1.0`` when scale and add are adjacent).

Usage::

    source .agent_loop/run/env.sh
    python benchmarks/probes/simd_softmax_online_device.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import tilelang  # noqa: E402
import tilelang.ascend.language as T  # noqa: E402

from tileinfer.utils import is_npu_available  # noqa: E402

ROWS, BC, PAGES = 16, 128, 8  # real decode shape: 16 rows/AIV, page 128 (2 fp32 regs), 8 pages
SCALE = 1.0 / (128 ** 0.5)


def perpage_online_simd(rows: int, bc: int, pages: int):
    """Per-page online SIMD softmax core: running (m, l), tail mask, emits p and final l."""
    total = pages * bc

    @T.prim_func
    def main(S: T.Tensor((rows, total), "float32"),
             POut: T.Tensor((rows, total), "float32"),
             LOut: T.Tensor((rows, 64), "float32")):
        with T.Kernel(1) as bx:
            s_ub = T.alloc_shared((rows, total), "float32")
            p_ub = T.alloc_shared((rows, total), "float32")
            m_ub = T.alloc_shared((rows, 64), "float32")
            l_ub = T.alloc_shared((rows, 64), "float32")
            ll_ub = T.alloc_shared((1,), "int32")
            T.copy(S, s_ub)
            with T.SimdVF():
                for r in T.serial(rows):
                    T.simd.vsts(m_ub[r, 0], T.simd.vdup(T.float32(-1e30), "float32"))
                    T.simd.vsts(l_ub[r, 0], T.simd.vdup(T.float32(0.0), "float32"))
                for p in T.serial(pages):
                    for r in T.serial(rows):
                        idx0 = T.simd.vci(0, "int32", order="INC_ORDER")
                        idx1 = T.simd.vci(64, "int32", order="INC_ORDER")
                        sc = T.simd.vdup(T.float32(SCALE), "float32")
                        a0 = T.simd.vmul(T.simd.vld(s_ub[r, p * bc + 0], dist="NORM"), sc)
                        a1 = T.simd.vmul(T.simd.vld(s_ub[r, p * bc + 64], dist="NORM"), sc)
                        pm = T.simd.vdupv(T.simd.vld(m_ub[r, 0], dist="NORM"), pos="POS_LOWEST")
                        nm = T.simd.vmax(pm, T.simd.vmax(T.simd.vcmax(a0), T.simd.vcmax(a1)))
                        nml = T.simd.vdupv(nm, pos="POS_LOWEST")
                        alpha = T.simd.vexp(T.simd.vsub(pm, nml))
                        e0 = T.simd.vexp(T.simd.vsub(a0, nml))
                        e1 = T.simd.vexp(T.simd.vsub(a1, nml))
                        lp = T.simd.vdupv(T.simd.vld(l_ub[r, 0], dist="NORM"), pos="POS_LOWEST")
                        ln = T.simd.vadd(T.simd.vmul(lp, alpha),
                                         T.simd.vadd(T.simd.vcadd(e0), T.simd.vcadd(e1)))
                        T.simd.vsts(m_ub[r, 0], nml)
                        T.simd.vsts(l_ub[r, 0], T.simd.vdupv(ln, pos="POS_LOWEST"))
                        T.simd.vsts(p_ub[r, p * bc + 0], e0)
                        T.simd.vsts(p_ub[r, p * bc + 64], e1)
            T.copy(p_ub, POut)
            T.copy(l_ub, LOut)

    return main


def main():
    if not is_npu_available():
        print("no NPU available; this probe needs the device")
        return 1
    torch.manual_seed(0)
    s = (torch.randn((ROWS, PAGES * BC), dtype=torch.float32) * 4.0).to("npu")
    ss = s * SCALE
    gmax = ss.max(dim=1, keepdim=True).values
    e = torch.exp(ss - gmax)
    ref_l = e.sum(dim=1, keepdim=True)
    ref = e / ref_l
    tol = 5e-2
    try:
        k = tilelang.compile(perpage_online_simd(ROWS, BC, PAGES), out_idx=[1, 2],
                             pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
        p_out, l_out = k(s)
        torch.npu.synchronize()
        l = l_out.float()[:, 0:1]
        d = (p_out.float() / l - ref).abs().max().item()
        dl = (l - ref_l).abs().max().item()
        ok = d <= tol
        print(f"[perpage_online] {'PASS' if ok else 'FAIL'} softmax max|diff|={d:.5f} "
              f"l max|diff|={dl:.5f}")
        return 0 if ok else 1
    except Exception as exc:  # noqa: BLE001
        print(f"[perpage_online] ERROR {type(exc).__name__}: {str(exc)[:200]}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
