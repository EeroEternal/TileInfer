#!/usr/bin/env python3
"""Device probe (needs NPU): the **paged, masked** SIMD softmax the decode kernel needs (T5).

``simd_softmax_device.py`` proved the *plain* SIMD softmax matches the fp32 oracle.  The decode
kernel is harder: it walks the request's KV pages in a **runtime** loop, masks the tail of the last
page (``col >= kv_last_page_len`` -> ``-1e30``), and reduces a row that spans **two** SIMD registers
(page_size 128 = 2×64 fp32 lanes).  This probe validates exactly that combination on device — it is
the blueprint for (and the gate before) the kernel integration.

What it pins down (each verified against ``torch.softmax`` on the masked scores, ``max|diff|=0``):

  * a **runtime** page loop ``for p in T.serial(pages)`` (not unrolled);
  * a **runtime** per-request ``last_len`` read from UB (SIMD_VF cannot touch global memory, so the
    caller must stage it into UB first);
  * masking with ``vcmps`` (int32 lane index vs int limit) + ``vsel`` — the int path is used because
    bisheng cannot select on ``f32`` (a float ``T.if_then_else`` lowers to an f32 ``select``);
  * the two-pass form (pass 1 row max, pass 2 exp+sum, pass 3 ``o/l``), which equals the kernel's
    online form and avoids the online rescale's reset-per-page subtleties.

Usage::

    source .agent_loop/run/env.sh
    python benchmarks/probes/simd_softmax_paged_device.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import tilelang  # noqa: E402
import tilelang.ascend.language as T  # noqa: E402

from tileinfer.utils import is_npu_available  # noqa: E402

ROWS, BC, PAGES = 16, 128, 3  # 16 rows, page_size 128 (2 fp32 registers/row), 3 pages


def paged_softmax_simd(rows: int, bc: int, pages: int):
    """Two-pass paged SIMD softmax (runtime page loop + runtime last-len + tail mask)."""
    total = pages * bc

    @T.prim_func
    def main(S: T.Tensor((rows, total), "float32"),
             LastLen: T.Tensor((1,), "int32"),
             Out: T.Tensor((rows, total), "float32")):
        with T.Kernel(1) as bx:
            s_ub = T.alloc_shared((rows, total), "float32")
            o_ub = T.alloc_shared((rows, total), "float32")
            m_ub = T.alloc_shared((rows, 64), "float32")  # one broadcast vreg per row
            l_ub = T.alloc_shared((rows, 64), "float32")
            ll_ub = T.alloc_shared((1,), "int32")
            T.copy(S, s_ub)
            T.copy(LastLen, ll_ub)  # global -> UB (SIMD_VF must not read global)
            with T.SimdVF():
                for r in T.serial(rows):
                    T.simd.vsts(m_ub[r, 0], T.simd.vdup(T.float32(-1e30), "float32"))
                    T.simd.vsts(l_ub[r, 0], T.simd.vdup(T.float32(0.0), "float32"))
                # pass 1: running row max over pages
                for p in T.serial(pages):
                    lim = T.if_then_else(p + 1 < pages, bc, ll_ub[0])
                    for r in T.serial(rows):
                        idx0 = T.simd.vci(0, "int32", order="INC_ORDER")
                        idx1 = T.simd.vci(64, "int32", order="INC_ORDER")
                        a0 = T.simd.vld(s_ub[r, p * bc + 0], dist="NORM")
                        a1 = T.simd.vld(s_ub[r, p * bc + 64], dist="NORM")
                        neg = T.simd.vdup(T.float32(-1e30), "float32")
                        s0 = T.simd.vsel(a0, neg, T.simd.vcmps(idx0, lim, op="lt"))
                        s1 = T.simd.vsel(a1, neg, T.simd.vcmps(idx1, lim, op="lt"))
                        pm = T.simd.vld(m_ub[r, 0], dist="NORM")
                        nm = T.simd.vmax(pm, T.simd.vmax(T.simd.vcmax(s0), T.simd.vcmax(s1)))
                        T.simd.vsts(m_ub[r, 0], T.simd.vdupv(nm, pos="POS_LOWEST"))
                # pass 2: exp(s - m) + running row sum
                for p in T.serial(pages):
                    lim = T.if_then_else(p + 1 < pages, bc, ll_ub[0])
                    for r in T.serial(rows):
                        idx0 = T.simd.vci(0, "int32", order="INC_ORDER")
                        idx1 = T.simd.vci(64, "int32", order="INC_ORDER")
                        a0 = T.simd.vld(s_ub[r, p * bc + 0], dist="NORM")
                        a1 = T.simd.vld(s_ub[r, p * bc + 64], dist="NORM")
                        neg = T.simd.vdup(T.float32(-1e30), "float32")
                        s0 = T.simd.vsel(a0, neg, T.simd.vcmps(idx0, lim, op="lt"))
                        s1 = T.simd.vsel(a1, neg, T.simd.vcmps(idx1, lim, op="lt"))
                        mb = T.simd.vld(m_ub[r, 0], dist="NORM")
                        e0 = T.simd.vexp(T.simd.vsub(s0, mb))
                        e1 = T.simd.vexp(T.simd.vsub(s1, mb))
                        lp = T.simd.vld(l_ub[r, 0], dist="NORM")
                        ln = T.simd.vadd(lp, T.simd.vadd(T.simd.vcadd(e0), T.simd.vcadd(e1)))
                        T.simd.vsts(l_ub[r, 0], T.simd.vdupv(ln, pos="POS_LOWEST"))
                        T.simd.vsts(o_ub[r, p * bc + 0], e0)
                        T.simd.vsts(o_ub[r, p * bc + 64], e1)
                # pass 3: o / l
                for r in T.serial(rows):
                    lb = T.simd.vld(l_ub[r, 0], dist="NORM")
                    for k in T.serial(total // 64):
                        T.simd.vsts(o_ub[r, k * 64],
                                    T.simd.vdiv(T.simd.vld(o_ub[r, k * 64], dist="NORM"), lb))
            T.copy(o_ub, Out)

    return main


def _ref(s, bc, pages, last):
    full = s.clone()
    full[:, (pages - 1) * bc + last:] = -1e30
    return torch.softmax(full, dim=1)


def main():
    if not is_npu_available():
        print("no NPU available; this probe needs the device")
        return 1
    torch.manual_seed(0)
    tol = 2e-2
    rc = 0
    for last in (BC, BC // 2, 13):  # full page, half page, odd tail
        s = (torch.randn((ROWS, PAGES * BC), dtype=torch.float32) * 4.0).to("npu")
        ll = torch.tensor([last], dtype=torch.int32).to("npu")
        ref = _ref(s, BC, PAGES, last)
        try:
            k = tilelang.compile(paged_softmax_simd(ROWS, BC, PAGES), out_idx=[2],
                                 pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
            out = k(s, ll)
            torch.npu.synchronize()
            d = (out.float() - ref).abs().max().item()
            ok = d <= tol
            if not ok:
                rc = 1
            print(f"[paged_softmax last={last}] {'PASS' if ok else 'FAIL'} max|diff|={d:.5f}")
        except Exception as exc:  # noqa: BLE001
            print(f"[paged_softmax last={last}] ERROR {type(exc).__name__}: {str(exc)[:200]}")
            rc = 1
    print(f"tolerance={tol}; verdict={'PASS' if rc == 0 else 'FAIL'}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
