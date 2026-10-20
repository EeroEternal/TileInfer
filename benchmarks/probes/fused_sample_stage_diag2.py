"""fused_sample 分阶段诊断 v2:精确复刻主内核的每个阶段,逐段加长。

崩溃 507035(type 3)是越界/非法地址。本探针用与主内核**完全相同的分块参数**
(blk=8192, n_chunks=2)和相同的 SIMD 写法,逐段打开,定位第一个崩的阶段。
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import tileinfer  # noqa: F401
import tilelang
import tilelang.ascend.language as T

from tileinfer.kernels.sampling.fused_sample import fused_sample_plan

DEV = torch.device("npu", 0)
torch.npu.set_device(DEV)

V = 16384
B = 2
VL = 64
PLAN = fused_sample_plan(V, B)
BLK = PLAN["blk"]
NCHUNK = PLAN["n_chunks"]
NVEC = PLAN["n_vec"]
print(f"[diag] V={V} blk={BLK} n_chunks={NCHUNK} n_vec={NVEC}", flush=True)


def build(stage: int):
    @T.prim_func
    def main(
        logits: T.Tensor((B, V), T.float32),
        out: T.Tensor((B,), T.int32),
    ):
        with T.Kernel(B) as row:
            vals_ub = T.alloc_shared((BLK,), T.float32)
            st_ub = T.alloc_shared((8,), T.float32)
            quart_ub = T.alloc_shared((4,), T.float32)
            terms = T.alloc_shared((4,), T.float32)

            with T.SimtVF(threads=1):
                st_ub[0] = T.float32(1.0)
                st_ub[1] = T.float32(-1e38)

            # stage>=1: max
            for c in T.serial(NCHUNK):
                T.copy(logits[row, c * BLK : (c + 1) * BLK], vals_ub)
                with T.SimdVF():
                    full = T.simd.pset(32)
                    one = T.simd.pset(32, "PAT_VL1")
                    inv = T.simd.vld(st_ub[0], "BRC_B32")
                    acc = T.simd.alloc_local((1,), T.float32)
                    acc[0] = T.simd.vld(st_ub[1], "BRC_B32")
                    for g in T.serial(NVEC):
                        acc[0] = T.simd.vmax(
                            acc[0], T.simd.vcmax(T.simd.vmul(T.simd.vld(vals_ub[g * VL]), inv), full), full
                        )
                    T.simd.vsts(st_ub[1], acc[0], one, "ONEPT_B32")

            # stage>=2: sum
            if stage >= 2:
                with T.SimtVF(threads=1):
                    st_ub[2] = T.float32(0.0)
                for c in T.serial(NCHUNK):
                    T.copy(logits[row, c * BLK : (c + 1) * BLK], vals_ub)
                    with T.SimdVF():
                        full = T.simd.pset(32)
                        one = T.simd.pset(32, "PAT_VL1")
                        inv = T.simd.vld(st_ub[0], "BRC_B32")
                        mx = T.simd.vld(st_ub[1], "BRC_B32")
                        acc = T.simd.alloc_local((1,), T.float32)
                        acc[0] = T.simd.vld(st_ub[2], "BRC_B32")
                        for g in T.serial(NVEC):
                            acc[0] = T.simd.vadd(
                                acc[0],
                                T.simd.vexp(
                                    T.simd.vsub(T.simd.vmul(T.simd.vld(vals_ub[g * VL]), inv), mx)
                                ),
                                full,
                            )
                        T.simd.vsts(st_ub[2], acc[0], one, "ONEPT_B32")

            # stage>=3: reinterpret 位插值取分位点
            if stage >= 3:
                with T.SimdVF():
                    full = T.simd.pset(32)
                    one = T.simd.pset(32, "PAT_VL1")
                    lo_i = T.reinterpret(T.simd.vld(st_ub[1], "BRC_B32"), T.int32x64)
                    hi_i = T.reinterpret(T.simd.vld(st_ub[2], "BRC_B32"), T.int32x64)
                    width = T.simd.vsub(hi_i, lo_i, full)
                    q0 = T.simd.vadd(lo_i, T.simd.vshrs(width, 2, full), full)
                    T.simd.vsts(quart_ub[0], T.reinterpret(q0, T.float32x64), one, "ONEPT_B32")

            # stage>=4: 带掩码的三档归约
            if stage >= 4:
                with T.SimtVF(threads=1):
                    terms[0] = T.float32(0.0)
                for c in T.serial(NCHUNK):
                    T.copy(logits[row, c * BLK : (c + 1) * BLK], vals_ub)
                    with T.SimdVF():
                        full = T.simd.pset(32)
                        one = T.simd.pset(32, "PAT_VL1")
                        inv = T.simd.vld(st_ub[0], "BRC_B32")
                        mx = T.simd.vld(st_ub[1], "BRC_B32")
                        q0v = T.simd.vld(quart_ub[0], "BRC_B32")
                        a0 = T.simd.alloc_local((1,), T.float32)
                        a0[0] = T.simd.vld(terms[0], "BRC_B32")
                        for g in T.serial(NVEC):
                            e = T.simd.vexp(
                                T.simd.vsub(T.simd.vmul(T.simd.vld(vals_ub[g * VL]), inv), mx)
                            )
                            a0[0] = T.simd.vadd(a0[0], e, T.simd.vcmp(e, q0v, full, "ge"), "MODE_MERGING")
                        T.simd.vsts(terms[0], a0[0], one, "ONEPT_B32")

            # stage>=5: 回写 UB + vcmax 归约(greedy 选择的核心动作)
            if stage >= 5:
                with T.SimtVF(threads=1):
                    st_ub[5] = T.float32(-1e38)
                for c in T.serial(NCHUNK):
                    T.copy(logits[row, c * BLK : (c + 1) * BLK], vals_ub)
                    with T.SimdVF():
                        full = T.simd.pset(32)
                        one = T.simd.pset(32, "PAT_VL1")
                        inv = T.simd.vld(st_ub[0], "BRC_B32")
                        mx = T.simd.vld(st_ub[1], "BRC_B32")
                        q0v = T.simd.vld(quart_ub[0], "BRC_B32")
                        best = T.simd.alloc_local((1,), T.float32)
                        best[0] = T.simd.vld(st_ub[5], "BRC_B32")
                        for g in T.serial(NVEC):
                            e = T.simd.vexp(
                                T.simd.vsub(T.simd.vmul(T.simd.vld(vals_ub[g * VL]), inv), mx)
                            )
                            hit = T.simd.vcmp(e, q0v, full, "ge")
                            em = T.simd.vsel(e, T.simd.vdup(-1e38, T.float32, full), hit)
                            T.simd.vsts(vals_ub[g * VL], em)
                            best[0] = T.simd.vmax(best[0], em, full)
                        T.simd.vsts(st_ub[5], best[0], one, "ONEPT_B32")

            # stage>=6: 第二遍标量尾扫(SimtVF 内读 UB)
            if stage >= 6:
                with T.SimtVF(threads=1):
                    st_ub[6] = T.float32(0.0)
                for c in T.serial(NCHUNK):
                    T.copy(logits[row, c * BLK : (c + 1) * BLK], vals_ub)
                    with T.SimdVF():
                        full = T.simd.pset(32)
                        inv = T.simd.vld(st_ub[0], "BRC_B32")
                        mx = T.simd.vld(st_ub[1], "BRC_B32")
                        q0v = T.simd.vld(quart_ub[0], "BRC_B32")
                        bestv = T.simd.vld(st_ub[5], "BRC_B32")
                        for g in T.serial(NVEC):
                            e = T.simd.vexp(
                                T.simd.vsub(T.simd.vmul(T.simd.vld(vals_ub[g * VL]), inv), mx)
                            )
                            hit = T.simd.vcmp(e, q0v, full, "ge")
                            em = T.simd.vsel(e, T.simd.vdup(-1e38, T.float32, full), hit)
                            T.simd.vsts(vals_ub[g * VL], em)
                    for g in T.serial(NVEC):
                        with T.SimtVF(threads=1):
                            for lane in T.serial(VL):
                                idx = c * BLK + g * VL + lane
                                match = T.min(st_ub[6] == 0, vals_ub[g * VL + lane] == st_ub[5])
                                st_ub[7] = T.Select(match, T.cast(idx, T.float32), st_ub[7])
                                st_ub[6] = T.Select(match, T.float32(1.0), st_ub[6])

            with T.SimtVF(threads=1):
                out[row] = T.cast(st_ub[1], T.int32)

    return main


logits = torch.randn(B, V, dtype=torch.float32, device=DEV)
for stage in range(1, 7):
    try:
        k = tilelang.compile(build(stage), out_idx=None)
        out = torch.zeros(B, dtype=torch.int32, device=DEV)
        k(logits, out)
        torch.npu.synchronize()
        print(f"[diag] stage={stage}: OK", flush=True)
        del k
        torch.npu.empty_cache()
    except Exception as e:
        print(f"[diag] stage={stage}: FAIL {type(e).__name__}: {str(e)[:160]}", flush=True)
        break
