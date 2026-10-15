#!/usr/bin/env python3
"""Offline (no-NPU) probe: is a **SIMD** softmax expressible, replacing the SIMT one (T5)?

The Ascend950 attention kernels write their online softmax with ``T.SimtVF`` + ``T.alloc_reducer``
("a few lines instead of a hundred", see the module docstring of
``kernels/attention/paged_decode_ascend950.py``).  That keeps vector time on the critical path:
the SIMT reducer lowers to per-thread scalar loops, whereas the upstream FlashAttention path uses
the packed ``T.simd.*`` micro-API (``vexp``/``vmax``/``vcgmax`` ...), which lowers to CCE vector
intrinsics and is the documented ~2x item still on the table (next-tasks.md T5).

This probe lowers the *current* SIMT softmax pattern and a candidate **SIMD** softmax pattern
offline and prints the generated CCE instruction mix.  It needs no device: the verdict ("the
softmax can be written with T.simd.* in a T.SimdVF() region on this backend") is read straight off
the generated source.

What it establishes:
  * baseline (SIMT): the per-row reduce/exp lowers to scalar thread loops (``for``/reducer calls);
  * candidate (SIMD): the same math lowers to ``vlds_norm`` / ``vcmax`` / ``vsub`` / ``vexp`` /
    ``vcgmax`` / ``vmul`` / ``vsts_norm`` vector intrinsics.

Caveat: this proves the *mechanism* is expressible; it does not prove a speedup (the SIMD row
width vs the SIMT thread count, and the UB liveness, are the new costs) and it does not prove
numerical equivalence (the SIMD exp precision must be checked against the fp32 oracle on device).
Both need device iteration.  The real integration point is
``kernels/attention/paged_decode_ascend950.py`` (decode softmax + the two prefill softmaxes); this
probe only fixes the *shape* of the rewrite.

Usage::

    source .agent_loop/run/env.sh
    python benchmarks/probes/simd_softmax_lower.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import tilelang.ascend.language as T  # noqa: E402
from tilelang.tools.compile_only import compile_kernel_source  # noqa: E402

ROWS, BC = 32, 128  # one AIV's half of a block_q=128 tile at page_size=128


def simt_softmax_kernel():
    """Current pattern: ``T.SimtVF`` + reducer max/sum, scalar per-thread."""

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


def simd_softmax_kernel():
    """Candidate: ``T.SimdVF`` + ``T.simd.*`` packed vector intrinsics, one row per op."""

    @T.prim_func
    def main(S: T.Tensor((ROWS, BC), "float32"), Out: T.Tensor((ROWS, BC), "float32")):
        with T.Kernel(1) as bx:
            s_ub = T.alloc_shared((ROWS, BC), "float32")
            o_ub = T.alloc_shared((ROWS, BC), "float32")
            T.copy(S, s_ub)
            with T.SimdVF():
                for r in T.serial(ROWS):
                    row = T.simd.vld(s_ub[r, 0], dist="NORM")
                    m = T.simd.vcmax(row)                 # row max (reduction)
                    d = T.simd.vsub(row, m)               # x - max
                    e = T.simd.vexp(d)                    # exp
                    z = T.simd.vcgmax(e)                  # row sum (reduction)
                    o = T.simd.vmul(e, z)                 # / sum
                    T.simd.vsts(o_ub[r, 0], o)
            T.copy(o_ub, Out)

    return main


def _report(label, fn):
    try:
        src = compile_kernel_source(fn(), target="ascend")
    except Exception as exc:  # noqa: BLE001
        print(f"[{label}] LOWER FAILED: {type(exc).__name__}: {str(exc)[:400]}")
        return
    simd_ops = ("vlds_norm", "vld", "vsts_norm", "vsts", "vexp", "vcmax", "vcgmax",
                "vsub", "vmul", "vadd", "vmax")
    counts = {op: len(re.findall(rf"\b{op}\b", src)) for op in simd_ops}
    simd_total = sum(counts[op] for op in simd_ops)
    scalar_loop = len(re.findall(r"\bfor\s*\(", src))
    print(f"[{label}] bytes={len(src)} simd_ops={simd_total} scalar_for={scalar_loop} "
          + " ".join(f"{op}={counts[op]}" for op in simd_ops if counts[op]))
    for line in src.splitlines():
        if any(k in line for k in simd_ops) or "reducer" in line.lower():
            print("      ", line.strip()[:170])


def main():
    print(f"=== baseline: SIMT softmax ({ROWS}x{BC}) ===")
    _report("simt", simt_softmax_kernel)
    print(f"=== candidate: SIMD softmax ({ROWS}x{BC}) ===")
    _report("simd", simd_softmax_kernel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
