"""Persistent TileLang dequant: 28 cores walk all (row-block, k-block) tiles."""

import time

import torch
import torch_npu
import tilelang
import tilelang.ascend.language as T

from tools.dsv41 import ref_cpu

NUM_CORES = 28


def dequant_kernel(N: int, K: int, BN: int = 16, BK: int = 1024, threads: int = 256):
    NB = K // 32
    nkb = K // BK
    ntiles = ((N + BN - 1) // BN) * nkb
    rounds = (ntiles + NUM_CORES - 1) // NUM_CORES

    @T.prim_func
    def main(
        P: T.Tensor((N, K // 2), "uint8"),
        S: T.Tensor((N, NB), "uint8"),
        Y: T.Tensor((N, K), "bfloat16"),
    ):
        with T.Kernel(NUM_CORES) as core:
            p_ub = T.alloc_shared((BN, BK // 2), "uint8")
            s_ub = T.alloc_shared((BN, BK // 32), "uint8")
            y_ub = T.alloc_shared((BN, BK), "bfloat16")
            for it in T.serial(rounds):
                tile = core + it * NUM_CORES
                if tile < ntiles:
                    bx = tile // nkb
                    bk = tile % nkb
                    T.copy(
                        P[bx * BN : (bx + 1) * BN, bk * (BK // 2) : (bk + 1) * (BK // 2)],
                        p_ub,
                    )
                    T.copy(
                        S[bx * BN : (bx + 1) * BN, bk * (BK // 32) : (bk + 1) * (BK // 32)],
                        s_ub,
                    )
                    with T.SimtVF(threads=threads):
                        for r, c in T.Parallel(BN, BK):
                            byte = T.Cast("int32", p_ub[r, c // 2])
                            code = (byte >> (4 * (c % 2))) & 15
                            mag = code & 7
                            val = T.if_then_else(
                                mag == 0, T.float32(0.0),
                                T.if_then_else(
                                    mag == 1, T.float32(0.5),
                                    T.if_then_else(
                                        mag == 2, T.float32(1.0),
                                        T.if_then_else(
                                            mag == 3, T.float32(1.5),
                                            T.if_then_else(
                                                mag == 4, T.float32(2.0),
                                                T.if_then_else(
                                                    mag == 5, T.float32(3.0),
                                                    T.if_then_else(mag == 6, T.float32(4.0),
                                                                   T.float32(6.0)),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            )
                            val = T.if_then_else((code & 8) != 0, -val, val)
                            scale = T.exp2(
                                T.Cast("float32", T.Cast("int32", s_ub[r, c // 32])) - 127.0
                            )
                            y_ub[r, c] = T.Cast("bfloat16", val * scale)
                    T.copy(y_ub, Y[bx * BN : (bx + 1) * BN, bk * BK : (bk + 1) * BK])

    return main


def main():
    dev = "npu:0"
    N0, K0 = 64, 128
    torch.manual_seed(0)
    w = torch.randn(N0, K0)
    packed0, scale0 = ref_cpu.quantize_weight_fp4_block32(w)
    packed0, scale0 = packed0.to(dev), scale0.to(dev)
    out0 = torch.empty(N0, K0, dtype=torch.bfloat16, device=dev)
    k0 = tilelang.compile(dequant_kernel(N0, K0, 16, 128, 256), out_idx=[])
    k0(packed0, scale0.view(torch.uint8), out0)
    ref = ref_cpu.unpack_fp4_lut(packed0, N0, K0, dtype=torch.bfloat16)
    ref = (ref.reshape(N0, K0 // 32, 32) * ref_cpu.scale_to_float(scale0).unsqueeze(-1)).reshape(N0, K0)
    print("maxdiff", (out0.float() - ref.float()).abs().max().item())

    N, K = 6 * 2304, 5120
    packed = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=dev)
    sbits = torch.randint(120, 135, (N, K // 32), dtype=torch.uint8, device=dev)
    out = torch.empty(N, K, dtype=torch.bfloat16, device=dev)
    for BN, BK, th in ((8, 512, 128), (16, 512, 128), (16, 1024, 256), (8, 1024, 256)):
        try:
            kk = tilelang.compile(dequant_kernel(N, K, BN, BK, th), out_idx=[])
            for _ in range(3):
                kk(packed, sbits, out)
            torch.npu.synchronize()
            t0 = time.perf_counter()
            for _ in range(20):
                kk(packed, sbits, out)
            torch.npu.synchronize()
            print("persist BN=%d BK=%d th=%d  %.3f ms" % (BN, BK, th, (time.perf_counter() - t0) / 20 * 1e3))
        except Exception as e:  # noqa: BLE001
            print("persist BN=%d BK=%d th=%d ERR %s" % (BN, BK, th, str(e)[:100]))

    t0 = time.perf_counter()
    for _ in range(20):
        u = ref_cpu.unpack_fp4_lut(packed, N, K, dtype=torch.bfloat16)
        u = (u.reshape(N, K // 32, 32) * ref_cpu.scale_to_float(sbits.view(torch.float8_e8m0fnu)).unsqueeze(-1)).reshape(N, K)
    torch.npu.synchronize()
    print("torch %.3f ms" % ((time.perf_counter() - t0) / 20 * 1e3))


if __name__ == "__main__":
    main()
