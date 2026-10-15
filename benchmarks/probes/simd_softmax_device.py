#!/usr/bin/env python3
"""Device probe (needs NPU): is a **SIMD** softmax numerically equivalent to the SIMT one (T5)?

The offline probe ``benchmarks/probes/simd_softmax_lower.py`` proved the SIMD softmax is
*expressible* (it lowers to ``vlds_norm``/``vcmax``/``vsub``/``vexp``/``vsts_norm``).  What it did
**not** prove is numerical equivalence on device, which is the gate before T5 can be ported into the
attention kernels.  This probe closes that: it runs both a SIMT softmax and a SIMD softmax on the
device and compares each against the fp32 torch reference.

Semantics wired in here (read off ``tilelang/src/tl_templates/ascend/simd_inst.h``):
  * ``vld``      load one vector register (fp32 → 64 lanes, so one *row* here is 64 columns);
  * ``vcmax``    full-vector max reduction (``asc_reduce_max``), result in lane 0;
  * ``vdupv``    broadcast lane 0 to all lanes (turns the reduction into the row constant);
  * ``vexp``     vector exp;
  * ``vcadd``    full-vector sum reduction (``asc_reduce_sum``);
  * ``vdiv``     vector / vector.

BC=64 is one fp32 vector register here — the minimal shape that exercises the whole chain without
the cross-register reduction merge a 128-wide row would need (that merge is the follow-up once this
passes).  Each row is reduced over its 64 columns independently.

Result is written to stderr-free stdout as two `diff` numbers; the caller (or a human) reads the
verdict off them: both `max|diff|` must be within tolerance of the fp32 oracle.

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

ROWS, BC = 32, 64  # BC = one fp32 SIMD register (64 lanes)


def _simt_softmax():
    """Baseline: the current kernel's SIMT softmax (``T.SimtVF`` + reducer)."""

    @T.prim_func
    def main(S: T.Tensor((ROWS, BC), "float32"), Out: T.Tensor((ROWS, BC), "float32")):
        with T.Kernel(1) as bx:
            s_ub = T.alloc_shared((ROWS, BC), "float32")
            o_ub = T.alloc_shared((ROWS, BC), "float32")
            T.copy(S, s_ub)
            with T.SimtVF(threads=128):
                s_frag = T.alloc_fragment((ROWS, BC), "float32")
                for r, c in T.Parallel(ROWS, BC):
                    s_frag[r, c] = s_ub[r, c]
                row_max = T.alloc_reducer((ROWS,), "float32", op="max")
                T.reducer_init(row_max)
                for r, c in T.Parallel(ROWS, BC):
                    T.reducer_update(row_max[r], s_frag[r, c])
                m = T.alloc_fragment((ROWS,), "float32")
                T.finalize_reducer(row_max, m)
                for r, c in T.Parallel(ROWS, BC):
                    s_frag[r, c] = T.exp(s_frag[r, c] - m[r])
                row_sum = T.alloc_reducer((ROWS,), "float32", op="sum")
                T.reducer_init(row_sum)
                for r, c in T.Parallel(ROWS, BC):
                    T.reducer_update(row_sum[r], s_frag[r, c])
                z = T.alloc_fragment((ROWS,), "float32")
                T.finalize_reducer(row_sum, z)
                for r, c in T.Parallel(ROWS, BC):
                    o_ub[r, c] = s_frag[r, c] / z[r]
            T.copy(o_ub, Out)

    return main


def _simd_softmax():
    """Candidate: ``T.SimdVF`` + ``T.simd.*`` (one row = one vector register)."""

    @T.prim_func
    def main(S: T.Tensor((ROWS, BC), "float32"), Out: T.Tensor((ROWS, BC), "float32")):
        with T.Kernel(1) as bx:
            s_ub = T.alloc_shared((ROWS, BC), "float32")
            o_ub = T.alloc_shared((ROWS, BC), "float32")
            T.copy(S, s_ub)
            with T.SimdVF():
                for r in T.serial(ROWS):
                    row = T.simd.vld(s_ub[r, 0], dist="NORM")
                    m = T.simd.vdupv(T.simd.vcmax(row), pos="POS_LOWEST")  # row max, broadcast
                    e = T.simd.vexp(T.simd.vsub(row, m))
                    z = T.simd.vdupv(T.simd.vcadd(e), pos="POS_LOWEST")   # row sum, broadcast
                    o = T.simd.vdiv(e, z)
                    T.simd.vsts(o_ub[r, 0], o)
            T.copy(o_ub, Out)

    return main


def _run(build):
    k = tilelang.compile(build(), out_idx=[1], pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    })
    return k


def main():
    if not is_npu_available():
        print("no NPU available; this probe needs the device")
        return 1
    torch.manual_seed(0)
    # Realistic score magnitudes: attention logits are ~O(1..10) scaled.
    s = (torch.randn((ROWS, BC), dtype=torch.float32) * 4.0).to("npu")
    ref = torch.softmax(s, dim=1)

    tol = 2e-2
    rc = 0
    for label, build in (("simt", _simt_softmax), ("simd", _simd_softmax)):
        try:
            k = _run(build)
            out = k(s)
            torch.npu.synchronize()
            diff = (out.float() - ref).abs().max().item()
            # row-sum sanity: a correct softmax row sums to 1
            rs = out.float().sum(dim=1)
            rs_err = (rs - 1.0).abs().max().item()
            status = "PASS" if diff <= tol else "FAIL"
            if status == "FAIL":
                rc = 1
            print(f"[{label}] {status}  max|diff|={diff:.5f}  max|rowsum-1|={rs_err:.5f}")
        except Exception as exc:  # noqa: BLE001
            print(f"[{label}] ERROR {type(exc).__name__}: {str(exc)[:200]}")
            rc = 1
    print(f"tolerance={tol}; verdict={'PASS' if rc == 0 else 'FAIL'}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
