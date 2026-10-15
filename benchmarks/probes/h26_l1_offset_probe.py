"""H26 离线复核(E70 风险注记):非 0 偏移 L1 buffer 的 access_ptr 发射。

风险:生产 MoE 内核中权重 L1 buffer 不是第一个 L1 分配(生产序:a_l1 → wg/wu_l1),
若 access_ptr 对非 0 arena 偏移的 buffer 发射裸 arena 指针(丢 +offset),align 直载会
写错 L1 地址 → 数值破坏。本探针(CPU-only)用双 L1 buffer + 三种分配序离线 lower,
检查 emitted 指针表达式是否含偏移,并与 baseline `T.copy(GM→L1 nd2nz)` 的 dst 发射对照。

用法::
    source .agent_loop/run/env.sh
    python benchmarks/probes/h26_l1_offset_probe.py
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import tilelang.ascend.language as T
from tilelang.language.tir.op import call_extern
from tilelang.tools.compile_only import compile_kernel_source

N, K, M = 128, 128, 32
BURSTS = K // 16
GROUP_BYTES = N * 32


def extern_three_buffer():
    """三 L1 buffer 强制 w_l1 落非 0 arena 偏移,extern 直载目标。"""

    @T.prim_func
    def main(
        Wnz: T.Tensor((N, K), "bfloat16"),
        X: T.Tensor((M, K), "bfloat16"),
        Out: T.Tensor((M, N), "bfloat16"),
    ):
        with T.Kernel(1) as bx:
            c_l1 = T.alloc_l1((M, K), "bfloat16")
            a_l1 = T.alloc_l1((M, K), "bfloat16")
            w_l1 = T.alloc_l1((N, K), "bfloat16")
            a_l0 = T.alloc_l0a((M, K), "bfloat16")
            b_l0 = T.alloc_l0b((N, K), "bfloat16")
            acc = T.alloc_l0c((M, N), "float32")
            T.copy(X[0:M, 0:K], a_l1)
            T.copy(X[0:M, 0:K], c_l1)
            T.evaluate(call_extern("void", "asc_set_gm2l1_loop_size", 1, 1))
            T.evaluate(call_extern("void", "asc_set_gm2l1_loop1_stride", 0, 0))
            T.evaluate(call_extern("void", "asc_set_gm2l1_loop2_stride", 0, 0))
            T.evaluate(
                call_extern(
                    "void", "asc_copy_gm2l1_align",
                    T.access_ptr(w_l1, "r"), T.access_ptr(Wnz, "r"),
                    BURSTS, GROUP_BYTES,
                    0, 0, 0, 0,
                    GROUP_BYTES, GROUP_BYTES,
                )
            )
            T.copy(c_l1, a_l0)
            T.copy(w_l1, b_l0)
            T.gemm(a_l0, b_l0, acc, transpose_B=True, clear_accum=True)
            T.copy(acc, Out)

    return main


def baseline_three_buffer():
    """同分配序 baseline:三 buffer 全走 T.copy(GM→L1 nd2nz),对照各 buffer 发射。"""

    @T.prim_func
    def main(
        Wnz: T.Tensor((N, K), "bfloat16"),
        X: T.Tensor((M, K), "bfloat16"),
        Out: T.Tensor((M, N), "bfloat16"),
    ):
        with T.Kernel(1) as bx:
            c_l1 = T.alloc_l1((M, K), "bfloat16")
            a_l1 = T.alloc_l1((M, K), "bfloat16")
            w_l1 = T.alloc_l1((N, K), "bfloat16")
            a_l0 = T.alloc_l0a((M, K), "bfloat16")
            b_l0 = T.alloc_l0b((N, K), "bfloat16")
            acc = T.alloc_l0c((M, N), "float32")
            T.copy(X[0:M, 0:K], a_l1)
            T.copy(X[0:M, 0:K], c_l1)
            T.copy(Wnz[0:N, 0:K], w_l1)
            T.copy(c_l1, a_l0)
            T.copy(w_l1, b_l0)
            T.gemm(a_l0, b_l0, acc, transpose_B=True, clear_accum=True)
            T.copy(acc, Out)

    return main


def extern_big_first():
    """w_l1 缩小 + 大 buffer 先分配,强制 w_l1 落非 0 偏移;L1 经 UB 回 GM 消费(防 DCE)。"""

    @T.prim_func
    def main(
        Wnz: T.Tensor((N, K), "bfloat16"),
        X: T.Tensor((M, K), "bfloat16"),
        Z1: T.Tensor((M, K), "bfloat16"),
        Z2: T.Tensor((M, K), "bfloat16"),
        Z3: T.Tensor((4 * N, K), "bfloat16"),
    ):
        with T.Kernel(1) as bx:
            big_l1 = T.alloc_l1((4 * N, K), "bfloat16")
            a_l1 = T.alloc_l1((M, K), "bfloat16")
            w_l1 = T.alloc_l1((M, K), "bfloat16")
            u_a = T.alloc_shared((M, K), "bfloat16")
            u_w = T.alloc_shared((M, K), "bfloat16")
            u_b = T.alloc_shared((4 * N, K), "bfloat16")
            T.copy(X[0:M, 0:K], a_l1)
            T.copy(X[0:4 * N, 0:K], big_l1)
            T.evaluate(call_extern("void", "asc_set_gm2l1_loop_size", 1, 1))
            T.evaluate(call_extern("void", "asc_set_gm2l1_loop1_stride", 0, 0))
            T.evaluate(call_extern("void", "asc_set_gm2l1_loop2_stride", 0, 0))
            T.evaluate(
                call_extern(
                    "void", "asc_copy_gm2l1_align",
                    T.access_ptr(w_l1, "r"), T.access_ptr(Wnz, "r"),
                    M // 16, M * 32, 0, 0, 0, 0, M * 32, M * 32,
                )
            )
            T.copy(a_l1, u_a)
            T.copy(w_l1, u_w)
            T.copy(big_l1, u_b)
            T.copy(u_a, Z1[0:M, 0:K])
            T.copy(u_w, Z2[0:M, 0:K])
            T.copy(u_b, Z3[0:4 * N, 0:K])

    return main


def baseline_big_first():
    """同形 baseline:big/a/w 全 T.copy(nd2nz),取 Wnz 的真实 arena 偏移。"""

    @T.prim_func
    def main(
        Wnz: T.Tensor((N, K), "bfloat16"),
        X: T.Tensor((M, K), "bfloat16"),
        Z1: T.Tensor((M, K), "bfloat16"),
        Z2: T.Tensor((M, K), "bfloat16"),
        Z3: T.Tensor((4 * N, K), "bfloat16"),
    ):
        with T.Kernel(1) as bx:
            big_l1 = T.alloc_l1((4 * N, K), "bfloat16")
            a_l1 = T.alloc_l1((M, K), "bfloat16")
            w_l1 = T.alloc_l1((M, K), "bfloat16")
            u_a = T.alloc_shared((M, K), "bfloat16")
            u_w = T.alloc_shared((M, K), "bfloat16")
            u_b = T.alloc_shared((4 * N, K), "bfloat16")
            T.copy(X[0:M, 0:K], a_l1)
            T.copy(X[0:4 * N, 0:K], big_l1)
            T.copy(Wnz[0:M, 0:K], w_l1)
            T.copy(a_l1, u_a)
            T.copy(w_l1, u_w)
            T.copy(big_l1, u_b)
            T.copy(u_a, Z1[0:M, 0:K])
            T.copy(u_w, Z2[0:M, 0:K])
            T.copy(u_b, Z3[0:4 * N, 0:K])

    return main


def _pointers(src, label):
    print(f"--- {label} ---")
    for line in src.splitlines():
        s = line.strip()
        if "asc_copy_gm2l1" in s:
            print("  ", s[:240])


def main():
    src_ext = compile_kernel_source(extern_three_buffer(), target="ascend")
    src_base = compile_kernel_source(baseline_three_buffer(), target="ascend")
    _pointers(src_base, "baseline 三 buffer:全 T.copy(nd2nz)")
    _pointers(src_ext, "h26 三 buffer:c/a T.copy + w extern align")

    # 逐 buffer 对照:baseline 中各 GM 源对应的 dst 偏移 vs extern 中同 buffer 的偏移
    def dst_offsets(src, op):
        out = {}
        for line in src.splitlines():
            if op in line:
                m = re.search(r"buf_dyn_l1(?:\s*\+\s*(\d+))?", line)
                s = re.search(r"&\((\w+)\[0\]\)", line)
                if m and s:
                    out[s.group(1)] = int(m.group(1) or 0)
        return out

    base_off = dst_offsets(src_base, "asc_copy_gm2l1_nd2nz")
    ext_off = dst_offsets(src_ext, "asc_copy_gm2l1_align")
    print(f"baseline 各 buffer dst 偏移: {base_off}")
    print(f"extern  w_l1 dst 偏移:      {ext_off}")
    ok = base_off.get("Wnz") is not None and ext_off.get("Wnz") == base_off["Wnz"]
    print("VERDICT:", "OFFSET-OK(access_ptr 与 T.copy 对同一 buffer 发射一致)"
          if ok else "OFFSET-MISMATCH")

    # 区分实验:强制 w_l1 落非 0 偏移(旁边放更大 buffer)
    print("=== 区分实验:big_l1(2N,K)先分配,强制 w_l1 非 0 ===")
    src_ext2 = compile_kernel_source(extern_big_first(), target="ascend")
    src_base2 = compile_kernel_source(baseline_big_first(), target="ascend")
    _pointers(src_base2, "baseline big-first:全 T.copy")
    _pointers(src_ext2, "h26 big-first:extern align 到 w_l1")
    base_off2 = dst_offsets(src_base2, "asc_copy_gm2l1_nd2nz")
    ext_off2 = dst_offsets(src_ext2, "asc_copy_gm2l1_align")
    print(f"baseline 各 buffer dst 偏移: {base_off2}")
    print(f"extern  w_l1 dst 偏移:      {ext_off2}")
    nonz = base_off2.get("Wnz", 0) != 0
    ok2 = nonz and ext_off2.get("Wnz") == base_off2["Wnz"]
    print(f"w_l1 实际非 0 偏移: {nonz}")
    print("VERDICT2:", "OFFSET-OK(非 0 偏移也被正确发射)"
          if ok2 else "OFFSET-MISMATCH")
    return 0 if ok and ok2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
