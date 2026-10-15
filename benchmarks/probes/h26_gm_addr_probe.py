"""H26 离线探针:access_ptr 的 GM 侧寻址发射(静态偏移 + 动态 strip/K-slice)。

H26 生产内核需要把 align 直载的 GM 实参指到 ``W13nz[eid*2*NT + h*NT + nt, 0, k0*BN]``
(strip 动态索引 × K-slice 静态/动态偏移)。本探针(CPU-only)离线 lower 三个变体,
检查 emitted GM 指针表达式是否把偏移折进去:
1. 静态偏移:``access_ptr(Wnz[0, 128])`` → 期望 ``&(Wnz[128])`` 或等价 +256B;
2. 动态 strip + 静态 K 偏移:``access_ptr(W[e_idx, 0, 128*BN])``;
3. 动态 strip + 动态 K 偏移:``access_ptr(W[e_idx, 0, k0*BN])``,k0 循环变量。

用法::
    source .agent_loop/run/env.sh
    python benchmarks/probes/h26_gm_addr_probe.py
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import tilelang.ascend.language as T
from tilelang.language.tir.op import call_extern
from tilelang.tools.compile_only import compile_kernel_source

BN, BK, K, M = 256, 128, 4096, 32
BURSTS = BN * BK * 2 // 4096   # 平坦 4096B burst
BURST_BYTES = 4096


def _emit(idx_expr, k_expr):
    @T.prim_func
    def main(
        Wnz: T.Tensor((4, BN, K), "bfloat16"),
        X: T.Tensor((M, K), "bfloat16"),
        Out: T.Tensor((M, BN), "bfloat16"),
    ):
        with T.Kernel(1) as bx:
            w_l1 = T.alloc_l1((BN, BK), "bfloat16")
            a_l1 = T.alloc_l1((M, BK), "bfloat16")
            a_l0 = T.alloc_l0a((M, BK), "bfloat16")
            b_l0 = T.alloc_l0b((BN, BK), "bfloat16")
            acc = T.alloc_l0c((M, BN), "float32")
            T.evaluate(call_extern("void", "asc_set_gm2l1_loop_size", 1, 1))
            T.evaluate(call_extern("void", "asc_set_gm2l1_loop1_stride", 0, 0))
            T.evaluate(call_extern("void", "asc_set_gm2l1_loop2_stride", 0, 0))
            T.evaluate(
                call_extern(
                    "void", "asc_copy_gm2l1_align",
                    T.access_ptr(w_l1, "r"),
                    T.access_ptr(Wnz[idx_expr, 0, k_expr], "r"),
                    BURSTS, BURST_BYTES, 0, 0, 0, 0, BURST_BYTES, BURST_BYTES,
                )
            )
            T.copy(X[0:M, 0:BK], a_l1)
            T.copy(a_l1, a_l0)
            T.copy(w_l1, b_l0)
            T.gemm(a_l0, b_l0, acc, transpose_B=True, clear_accum=True)
            T.copy(acc, Out)

    return main


def _static():
    return _emit(0, BN)


def _dyn_strip():
    @T.prim_func
    def main(
        Wnz: T.Tensor((4, BN, K), "bfloat16"),
        Idx: T.Tensor((4,), "int32"),
        X: T.Tensor((M, K), "bfloat16"),
        Out: T.Tensor((M, BN), "bfloat16"),
    ):
        with T.Kernel(1) as bx:
            w_l1 = T.alloc_l1((BN, BK), "bfloat16")
            a_l1 = T.alloc_l1((M, BK), "bfloat16")
            a_l0 = T.alloc_l0a((M, BK), "bfloat16")
            b_l0 = T.alloc_l0b((BN, BK), "bfloat16")
            acc = T.alloc_l0c((M, BN), "float32")
            e_idx = Idx[0]
            T.evaluate(call_extern("void", "asc_set_gm2l1_loop_size", 1, 1))
            T.evaluate(call_extern("void", "asc_set_gm2l1_loop1_stride", 0, 0))
            T.evaluate(call_extern("void", "asc_set_gm2l1_loop2_stride", 0, 0))
            T.evaluate(
                call_extern(
                    "void", "asc_copy_gm2l1_align",
                    T.access_ptr(w_l1, "r"),
                    T.access_ptr(Wnz[e_idx, 0, BN], "r"),
                    BURSTS, BURST_BYTES, 0, 0, 0, 0, BURST_BYTES, BURST_BYTES,
                )
            )
            T.copy(X[0:M, 0:BK], a_l1)
            T.copy(a_l1, a_l0)
            T.copy(w_l1, b_l0)
            T.gemm(a_l0, b_l0, acc, transpose_B=True, clear_accum=True)
            T.copy(acc, Out)

    return main


def _dyn_kloop():
    @T.prim_func
    def main(
        Wnz: T.Tensor((4, BN, K), "bfloat16"),
        Idx: T.Tensor((4,), "int32"),
        X: T.Tensor((M, K), "bfloat16"),
        Out: T.Tensor((M, BN), "bfloat16"),
    ):
        with T.Kernel(1) as bx:
            w_l1 = T.alloc_l1((BN, BK), "bfloat16")
            a_l1 = T.alloc_l1((M, BK), "bfloat16")
            a_l0 = T.alloc_l0a((M, BK), "bfloat16")
            b_l0 = T.alloc_l0b((BN, BK), "bfloat16")
            acc = T.alloc_l0c((M, BN), "float32")
            e_idx = Idx[0]
            for kk in T.serial(K // BK):
                k0 = kk * BK
                T.evaluate(call_extern("void", "asc_set_gm2l1_loop_size", 1, 1))
                T.evaluate(call_extern("void", "asc_set_gm2l1_loop1_stride", 0, 0))
                T.evaluate(call_extern("void", "asc_set_gm2l1_loop2_stride", 0, 0))
                T.evaluate(
                    call_extern(
                        "void", "asc_copy_gm2l1_align",
                        T.access_ptr(w_l1, "r"),
                        T.access_ptr(Wnz[e_idx, 0, k0], "r"),
                        BURSTS, BURST_BYTES, 0, 0, 0, 0, BURST_BYTES, BURST_BYTES,
                    )
                )
                T.copy(X[0:M, k0:k0 + BK], a_l1)
                T.copy(a_l1, a_l0)
                T.copy(w_l1, b_l0)
                T.gemm(a_l0, b_l0, acc, transpose_B=True, clear_accum=True)
            T.copy(acc, Out)

    return main


def _align_lines(src):
    return [l.strip()[:260] for l in src.splitlines() if "asc_copy_gm2l1_align" in l]


def main():
    cases = [
        ("静态偏移 Wnz[0, BN]", _static, "128"),
        ("动态 strip Wnz[e, 0, BN]", _dyn_strip, None),
        ("动态 K 循环 Wnz[e, 0, k0]", _dyn_kloop, None),
    ]
    fails = 0
    for label, fn, want_off in cases:
        try:
            src = compile_kernel_source(fn(), target="ascend")
        except Exception as exc:  # noqa: BLE001
            print(f"[{label}] LOWER FAILED: {type(exc).__name__}: {str(exc)[:300]}")
            fails += 1
            continue
        lines = _align_lines(src)
        print(f"[{label}] align 调用 {len(lines)} 处:")
        for l in lines[:2]:
            print("   ", l)
        if want_off is not None:
            ok = any(want_off in l for l in lines)
            print(f"    期望静态偏移 {want_off}(元素): {'OK' if ok else 'MISSING'}")
            fails += 0 if ok else 1
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
