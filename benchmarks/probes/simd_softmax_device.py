#!/usr/bin/env python3
"""Device probe (needs NPU): is a **SIMD** softmax numerically equivalent to the SIMT one (T5)?

The offline probe ``benchmarks/probes/simd_softmax_lower.py`` proved the SIMD softmax is
*expressible* (it lowers to ``vlds_norm``/``vcmax``/``vsub``/``vexp``/``vsts_norm``).  What it did
**not** prove is numerical equivalence on device, which is the gate before T5 can be ported into the
attention kernels.  This probe closes that: it runs both a SIMT softmax and a SIMD softmax on the
device and compares each against the fp32 torch reference, at **two widths**:

  * ``BC=64``  — one fp32 vector register (the minimal chain);
  * ``BC=128`` — the real decode ``page_size``; a row spans **two** registers, so the reduction must
    be merged across registers (``vmax(vcmax(r0), vcmax(r1))`` / ``vadd(vcadd(e0), vcadd(e1))``).
    This is the shape the kernel integration actually needs.

Semantics wired in here (read off ``tilelang/src/tl_templates/ascend/simd_inst.h``):
  * ``vld``      load one vector register (fp32 → 64 lanes);
  * ``vcmax``    full-vector max reduction (``asc_reduce_max``), result in lane 0;
  * ``vadd``/    merge the two registers' lane-0 results (cross-register reduction);
    ``vmax``
  * ``vdupv``    broadcast lane 0 to all lanes (turns the reduction into the row constant);
  * ``vexp``     vector exp;
  * ``vcadd``    full-vector sum reduction (``asc_reduce_sum``);
  * ``vdiv``     vector / vector.

Usage::

    source .agent_loop/run/env.sh
    python benchmarks/probes/simd_softmax_device.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import tilelang  # noqa: E402
import tilelang.ascend.language as T  # noqa: E402

from tileinfer.utils import is_npu_available  # noqa: E402

ROWS = 32
WIDTHS = (64, 128)  # 64 = one fp32 register; 128 = real decode page_size (cross-register)


def _simt_softmax(rows: int, bc: int):
    """Baseline: the current kernel's SIMT softmax (``T.SimtVF`` + reducer)."""

    @T.prim_func
    def main(S: T.Tensor((rows, bc), "float32"), Out: T.Tensor((rows, bc), "float32")):
        with T.Kernel(1) as bx:
            s_ub = T.alloc_shared((rows, bc), "float32")
            o_ub = T.alloc_shared((rows, bc), "float32")
            T.copy(S, s_ub)
            with T.SimtVF(threads=128):
                s_frag = T.alloc_fragment((rows, bc), "float32")
                for r, c in T.Parallel(rows, bc):
                    s_frag[r, c] = s_ub[r, c]
                row_max = T.alloc_reducer((rows,), "float32", op="max")
                T.reducer_init(row_max)
                for r, c in T.Parallel(rows, bc):
                    T.reducer_update(row_max[r], s_frag[r, c])
                m = T.alloc_fragment((rows,), "float32")
                T.finalize_reducer(row_max, m)
                for r, c in T.Parallel(rows, bc):
                    s_frag[r, c] = T.exp(s_frag[r, c] - m[r])
                row_sum = T.alloc_reducer((rows,), "float32", op="sum")
                T.reducer_init(row_sum)
                for r, c in T.Parallel(rows, bc):
                    T.reducer_update(row_sum[r], s_frag[r, c])
                z = T.alloc_fragment((rows,), "float32")
                T.finalize_reducer(row_sum, z)
                for r, c in T.Parallel(rows, bc):
                    o_ub[r, c] = s_frag[r, c] / z[r]
            T.copy(o_ub, Out)

    return main


def _simd_softmax_64(rows: int):
    """SIMD softmax, one fp32 register per row (BC=64)."""

    @T.prim_func
    def main(S: T.Tensor((rows, 64), "float32"), Out: T.Tensor((rows, 64), "float32")):
        with T.Kernel(1) as bx:
            s_ub = T.alloc_shared((rows, 64), "float32")
            o_ub = T.alloc_shared((rows, 64), "float32")
            T.copy(S, s_ub)
            with T.SimdVF():
                for r in T.serial(rows):
                    reg = T.simd.vld(s_ub[r, 0], dist="NORM")
                    m = T.simd.vdupv(T.simd.vcmax(reg), pos="POS_LOWEST")
                    e = T.simd.vexp(T.simd.vsub(reg, m))
                    z = T.simd.vdupv(T.simd.vcadd(e), pos="POS_LOWEST")
                    T.simd.vsts(o_ub[r, 0], T.simd.vdiv(e, z))
            T.copy(o_ub, Out)

    return main


def _simd_softmax_128(rows: int):
    """SIMD softmax, **two** fp32 registers per row (BC=128 = real decode page_size).

    The reduction is merged across the two registers: ``vmax(vcmax(r0), vcmax(r1))`` and
    ``vadd(vcadd(e0), vcadd(e1))`` — this is what the kernel integration needs (a page is 128
    tokens, so a row spans two SIMD registers).
    """

    @T.prim_func
    def main(S: T.Tensor((rows, 128), "float32"), Out: T.Tensor((rows, 128), "float32")):
        with T.Kernel(1) as bx:
            s_ub = T.alloc_shared((rows, 128), "float32")
            o_ub = T.alloc_shared((rows, 128), "float32")
            T.copy(S, s_ub)
            with T.SimdVF():
                for r in T.serial(rows):
                    r0 = T.simd.vld(s_ub[r, 0], dist="NORM")
                    r1 = T.simd.vld(s_ub[r, 64], dist="NORM")
                    m = T.simd.vmax(T.simd.vcmax(r0), T.simd.vcmax(r1))
                    mb = T.simd.vdupv(m, pos="POS_LOWEST")
                    e0 = T.simd.vexp(T.simd.vsub(r0, mb))
                    e1 = T.simd.vexp(T.simd.vsub(r1, mb))
                    z = T.simd.vadd(T.simd.vcadd(e0), T.simd.vcadd(e1))
                    zb = T.simd.vdupv(z, pos="POS_LOWEST")
                    T.simd.vsts(o_ub[r, 0], T.simd.vdiv(e0, zb))
                    T.simd.vsts(o_ub[r, 64], T.simd.vdiv(e1, zb))
            T.copy(o_ub, Out)

    return main


def _simd_softmax(rows: int, bc: int):
    return _simd_softmax_64(rows) if bc == 64 else _simd_softmax_128(rows)


def _run(build, rows: int, bc: int):
    k = tilelang.compile(build(rows, bc), out_idx=[1], pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    })
    return k


def main():
    if not is_npu_available():
        print("no NPU available; this probe needs the device")
        return 1
    torch.manual_seed(0)
    tol = 2e-2
    rc = 0
    for bc in WIDTHS:
        # Realistic score magnitudes: attention logits are ~O(1..10) scaled.
        s = (torch.randn((ROWS, bc), dtype=torch.float32) * 4.0).to("npu")
        ref = torch.softmax(s, dim=1)
        print(f"== width BC={bc} ({bc // 64} fp32 register(s)/row) ==")
        for label, build in (("simd", _simd_softmax), ("simt", _simt_softmax)):
            try:
                k = _run(build, ROWS, bc)
                out = k(s)
                torch.npu.synchronize()
                diff = (out.float() - ref).abs().max().item()
                # row-sum sanity: a correct softmax row sums to 1
                rs = out.float().sum(dim=1)
                rs_err = (rs - 1.0).abs().max().item()
                ok = diff <= tol
                # The verdict rests on the SIMD path (the candidate).  The SIMT path is only
                # informational here: this synthetic harness copies a single dense tile and does
                # not reproduce the real kernel's `dual_copy`/GEMM-output layout, so its reducer can
                # be wrong at BC>64 for reasons unrelated to the comparison.  The device kernel
                # suite (tests/test_tilelang_ascend950_decode.py) is the SIMT ground truth.
                if label == "simd" and not ok:
                    rc = 1
                note = "" if label == "simd" else "  (informational)"
                print(f"  [{label}] {'PASS' if ok else 'FAIL'}  max|diff|={diff:.5f}  "
                      f"max|rowsum-1|={rs_err:.5f}{note}")
            except Exception as exc:  # noqa: BLE001
                print(f"  [{label}] ERROR {type(exc).__name__}: {str(exc)[:200]}")
                if label == "simd":
                    rc = 1
    print(f"tolerance={tol}; SIMD verdict={'PASS' if rc == 0 else 'FAIL'}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
