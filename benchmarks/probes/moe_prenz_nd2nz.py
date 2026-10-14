"""Offline (no-NPU) probe: does a *pre-packed NZ* weight load avoid the nd2nz conversion?

E36/P4 concluded the GM->L1 nd2nz conversion is the structural MoE bottleneck (~50 GB/s/core)
and that "pre-NZ direct load is inexpressible in the kernel layer".  Reading the tilelang
compiler source shows that claim is too absolute: ``src/ascend/transform/insert_nd2nz.cc`` has a
``RewriteUBToL1`` branch that, when the *UB source* carries a ``make_ascend_compact_nz_layout``
annotation, emits a raw DMA with fractal-stride correction instead of the nd2nz conversion.  GM
has no such rewrite, so the route is three-stage:

    GM(pre-NZ flat) --plain DMA--> UB(compact_nz) --raw DMA--> L1

This probe lowers the current (GM->L1) and the candidate (via UB) weight-load patterns offline
and prints the generated source's nd2nz/ub2l1 instruction mix.  It needs no device: the verdict
("the weight path can avoid gm2l1_nd2nz") is read straight off the generated CCE.

What it establishes:
  * baseline: the weight load lowers to ``asc_copy_gm2l1_nd2nz`` (cube-side format conversion);
  * candidate: the weight load lowers to ``asc_copy_gm2ub_align`` + ``asc_copy_ub2l1`` (plain
    DMA on the vector side, no nd2nz on the weight path).

Caveat: this proves the *mechanism* is expressible; it does not prove a speedup (the GM->UB hop
and UB capacity are the new costs) and it does not prove numerical correctness (the pre-NZ
weight must be physically packed in the compact-NZ order the layout describes -- that packer
belongs to the vLLM weight-loading path).  Both need device iteration.

Usage::

    source .agent_loop/run/env.sh
    python benchmarks/probes/moe_prenz_nd2nz.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import tilelang.ascend.language as T  # noqa: E402
from tilelang.layout import make_ascend_compact_nz_layout  # noqa: E402
from tilelang.language import annotate_layout  # noqa: E402
from tilelang.tools.compile_only import compile_kernel_source  # noqa: E402

K, N, M = 128, 128, 64


def baseline_kernel():
    """Current pattern: GM(ND) -> L1 => ``asc_copy_gm2l1_nd2nz``."""

    @T.prim_func
    def main(W: T.Tensor((N, K), "bfloat16"), Out: T.Tensor((M, N), "bfloat16")):
        with T.Kernel(1) as bx:
            w_l1 = T.alloc_l1((N, K), "bfloat16")
            a_l1 = T.alloc_l1((M, K), "bfloat16")
            a_l0 = T.alloc_l0a((M, K), "bfloat16")
            b_l0 = T.alloc_l0b((N, K), "bfloat16")
            acc = T.alloc_l0c((M, N), "float32")
            T.copy(W[0:N, 0:K], w_l1)
            T.copy(W[0:M, 0:K], a_l1)
            T.copy(a_l1, a_l0)
            T.copy(w_l1, b_l0)
            T.gemm(a_l0, b_l0, acc, transpose_B=True, clear_accum=True)
            T.copy(acc, Out)

    return main


def prepacked_kernel():
    """Candidate: GM(pre-NZ) -> UB(compact_nz) -> L1, no nd2nz on the weight path."""

    @T.prim_func
    def main(Wnz: T.Tensor((N, K), "bfloat16"), Out: T.Tensor((M, N), "bfloat16")):
        with T.Kernel(1) as bx:
            w_ub = T.alloc_shared((N + 1, K), "bfloat16")  # +1 pad row per the layout contract
            w_l1 = T.alloc_l1((N, K), "bfloat16")
            a_l1 = T.alloc_l1((M, K), "bfloat16")
            a_l0 = T.alloc_l0a((M, K), "bfloat16")
            b_l0 = T.alloc_l0b((N, K), "bfloat16")
            acc = T.alloc_l0c((M, N), "float32")
            annotate_layout({w_ub: make_ascend_compact_nz_layout(w_ub)})
            T.copy(Wnz[0:N, 0:K], w_ub[0:N, 0:K])   # GM -> UB : plain DMA
            T.copy(w_ub[0:N, 0:K], w_l1)            # UB(compact_nz) -> L1 : raw DMA
            T.copy(Wnz[0:M, 0:K], a_l1)
            T.copy(a_l1, a_l0)
            T.copy(w_l1, b_l0)
            T.gemm(a_l0, b_l0, acc, transpose_B=True, clear_accum=True)
            T.copy(acc, Out)

    return main


def _report(label, fn):
    try:
        src = compile_kernel_source(fn(), target="ascend")
    except Exception as exc:  # noqa: BLE001
        print(f"[{label}] LOWER FAILED: {type(exc).__name__}: {str(exc)[:300]}")
        return
    print(f"[{label}] bytes={len(src)} "
          f"gm2l1_nd2nz={len(re.findall('gm2l1_nd2nz', src))} "
          f"gm2ub={len(re.findall('gm2ub', src))} ub2l1={len(re.findall('ub2l1', src))}")
    for line in src.splitlines():
        if "nd2nz" in line or "gm2ub" in line or "ub2l1" in line:
            print("      ", line.strip()[:170])


def main():
    print("=== baseline: GM(ND) -> L1 ===")
    _report("baseline", baseline_kernel)
    print("=== candidate: pre-NZ -> UB(compact_nz) -> L1 ===")
    _report("prepacked", prepacked_kernel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
