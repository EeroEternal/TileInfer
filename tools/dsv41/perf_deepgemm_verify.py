"""融合 fp8×fp4 GEMM（deep_gemm / DeepGEMM-Ascend）的数值与速度验证。

契约（源码 + 实测确定，见 docs/notes/dsv41-perf-deepgemm-jit-note.md）：
    a = (fp8_e4m3fn [M, K],  sf fp32 [M, K/gran_k])
    b = (int8        [N, K/2], sf fp32 [N, K/gran_k])   # int8 是 packed-FP4 的规范存储
    d = bf16 或 fp32 [M, N]
    recipe = (1, 1, gran_k)                             # 本模型的 act 是 block-32

环境要求（缺一不可）：
    export LD_PRELOAD=/home/userlist/zxs/TileInfer/.work/gcc13/lib64/libstdc++.so.6
    bash tools/dsv41/patch_deepgemm_headers.sh          # CANN 9.3 没有 ascendc_assert

对比对象是我们运行时当前的三步走（unpack_fp4 + 广播 block scale + bmm，实测 1.40 ms/层/GEMM）。
本脚本只做一件事：同一组随机数据，比较
    (1) 反量化后 fp32 matmul（参考值）
    (2) deep_gemm.fp8_fp4_gemm_nt（被测）
并给出两者的耗时对比与显存占用。

用法： python tools/dsv41/perf_deepgemm_verify.py [--shape M,N,K]
"""

from __future__ import annotations

import argparse
import sys
import time

import torch

# fp4 e2m1 的 16 个码点（bit3 为符号位）
FP4_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
)


def quant_fp8_block32(x: torch.Tensor, bk: int = 32):
    """per-token、block-32 的 fp8 量化（复刻运行时 act_quant 的语义）。"""
    m, k = x.shape
    blk = x.view(m, k // bk, bk)
    sf = blk.abs().amax(dim=-1, keepdim=True).clamp_min(1e-9) / 448.0
    q = (blk / sf).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q.reshape(m, k), sf.squeeze(-1).contiguous(), sf


def quant_fp4_block32(w: torch.Tensor, bk: int = 32):
    """把 bf16 权重压成 packed fp4（2 值/字节）+ block-32 e8m0 风格的 fp32 scale。"""
    n, k = w.shape
    blk = w.view(n, k // bk, bk)
    sf = blk.abs().amax(dim=-1, keepdim=True).clamp_min(1e-9) / 6.0
    q = (blk / sf).clamp(-6, 6)
    idx = (q.unsqueeze(-1) - FP4_LUT).abs().argmin(-1).to(torch.int32)   # [N, K/32, 32]
    idx = idx.reshape(n, k)
    packed = ((idx[:, 0::2] << 4) | idx[:, 1::2]).to(torch.uint8)        # [N, K/2]
    return packed, sf.squeeze(-1).contiguous(), sf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="8,256,256", help="M,N,K")
    ap.add_argument("--reps", type=int, default=20)
    args = ap.parse_args()
    M, N, K = (int(v) for v in args.shape.split(","))
    BK = 32
    dev = "npu:0"
    torch.manual_seed(0)

    import deep_gemm  # 延迟导入：没装/没补丁时也能给出清晰错误
    print(f"deep_gemm {getattr(deep_gemm, '__version__', '?')}  shape M={M} N={N} K={K} gran_k={BK}")

    x = torch.randn(M, K)
    w = torch.randn(N, K)

    a_q, a_sf, a_sf3 = quant_fp8_block32(x, BK)
    b_packed, b_sf, b_sf3 = quant_fp4_block32(w, BK)

    a_q_d = a_q.to(dev)
    a_sf_d = a_sf.to(dev)
    b_i8_d = b_packed.view(torch.int8).to(dev)     # int8 是 packed-FP4 的规范存储
    b_sf_d = b_sf.to(dev)
    d = torch.zeros(M, N, dtype=torch.bfloat16, device=dev)

    print(f"a {tuple(a_q_d.shape)} {a_q_d.dtype}  sfA {tuple(a_sf_d.shape)}")
    print(f"b {tuple(b_i8_d.shape)} {b_i8_d.dtype}  sfB {tuple(b_sf_d.shape)}")

    # ---- 参考值：反量化两侧后 fp32 matmul ----
    a_ref = (a_q.float().view(M, K // BK, BK) * a_sf3).reshape(M, K)
    b_idx = b_packed.to(torch.int32)
    b_nib = torch.empty(N, K, dtype=torch.long)
    b_nib[:, 0::2] = b_idx >> 4
    b_nib[:, 1::2] = b_idx & 0xF
    b_ref = (FP4_LUT[b_nib].view(N, K // BK, BK) * b_sf3).reshape(N, K)
    ref = a_ref @ b_ref.t()

    # ---- 被测 ----
    def run():
        deep_gemm.fp8_fp4_gemm_nt((a_q_d, a_sf_d), (b_i8_d, b_sf_d), d, None, recipe=(1, 1, BK))

    run()
    torch.npu.synchronize()
    got = d.float().cpu()
    max_abs = float((got - ref).abs().max())
    rel = max_abs / max(float(ref.abs().max()), 1e-6)
    print(f"numeric: max|diff|={max_abs:.4e}  rel={rel:.4e}")

    # ---- 计时（差分法：N2 次 vs N1 次，取中位数）----
    import statistics

    def timeit(n):
        ts = []
        for _ in range(5):
            torch.npu.synchronize()
            t0 = time.perf_counter()
            for _ in range(n):
                run()
            torch.npu.synchronize()
            ts.append((time.perf_counter() - t0) / n * 1000.0)   # ms/iter
        return statistics.median(ts)

    n1, n2 = 20, 220
    t1, t2 = timeit(n1), timeit(n2)
    per = (t2 * n2 - t1 * n1) / (n2 - n1)
    print(f"timing: T({n1})={t1:.4f} ms  T({n2})={t2:.4f} ms  ->  per-iter {per:.4f} ms")

    ok = rel < 5e-2
    print("RESULT:", "PASS" if ok else "NUMERIC MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
