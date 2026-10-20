"""诊断采样分支:输出 keep_mass / u / 阈值,判断逆 CDF 为何不随 seed 变化。"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import tileinfer  # noqa: F401
import tilelang
import tilelang.ascend.language as T

DEV = torch.device("npu", 0)
torch.npu.set_device(DEV)

V, B = 256, 2
BLK, NCH, NV = 256, 1, 4


@T.prim_func
def main(
    logits: T.Tensor((B, V), T.float32),
    seed: T.Tensor((B,), T.int64),
    dbg: T.Tensor((B, 4), T.float32),
    out: T.Tensor((B,), T.int32),
):
    with T.Kernel(B) as row:
        buf = T.alloc_shared((BLK,), T.float32)
        st = T.alloc_shared((8,), T.float32)
        T.copy(logits[row, :], buf)
        # max
        with T.SimtVF(threads=1):
            st[1] = T.float32(-1e38)
        with T.SimdVF():
            full = T.simd.pset(32)
            one = T.simd.pset(32, "PAT_VL1")
            acc = T.simd.alloc_local((1,), T.float32)
            acc[0] = T.simd.vdup(-1e38, T.float32, full)
            for g in T.serial(NV):
                acc[0] = T.simd.vmax(acc[0], T.simd.vcmax(T.simd.vld(buf[g * 64]), full), full)
            T.simd.vsts(st[1], acc[0], one, "ONEPT_B32")
        # total(keep_mass,阈值=0 即全保留)
        with T.SimtVF(threads=1):
            st[3] = T.float32(0.0)
        with T.SimdVF():
            full = T.simd.pset(32)
            one = T.simd.pset(32, "PAT_VL1")
            mx = T.simd.vld(st[1], "BRC_B32")
            acc2 = T.simd.alloc_local((1,), T.float32)
            acc2[0] = T.simd.vdup(0.0, T.float32, full)
            for g in T.serial(NV):
                acc2[0] = T.simd.vadd(
                    acc2[0], T.simd.vexp(T.simd.vsub(T.simd.vld(buf[g * 64]), mx)), full
                )
            T.simd.vsts(st[3], acc2[0], one, "ONEPT_B32")
        # u 与目标
        with T.SimtVF(threads=1):
            T.rng_init(seed[row], 0)
            st[6] = T.cast(T.rng_rand(), T.float32) / T.float32(4294967296.0) * st[3]
            dbg[row, 0] = st[1]
            dbg[row, 1] = st[3]
            dbg[row, 2] = st[6]
            dbg[row, 3] = T.cast(T.rng_rand(), T.float32)
            # 逆 CDF:标量遍历(最小实现,验证语义)
            st[4] = T.float32(0.0)
            out[row] = 0
        for i in T.serial(V):
            with T.SimtVF(threads=1):
                p = T.vexp(T.vsub(buf[i], st[1])) if hasattr(T, "vexp") else buf[i]
                st[4] = st[4] + p
                hit = st[4] >= st[6]
                out[row] = T.Select(T.min(hit, out[row] == 0), T.cast(i, T.int32), out[row])


k = tilelang.compile(main, out_idx=None)
logits = torch.randn(B, V, dtype=torch.float32, device=DEV)
seeds = torch.tensor([13, 7929], dtype=torch.int64, device=DEV)
dbg = torch.zeros(B, 4, dtype=torch.float32, device=DEV)
out = torch.zeros(B, dtype=torch.int32, device=DEV)
k(logits, seeds, dbg, out)
torch.npu.synchronize()
print(f"[dbg] out={out.tolist()}", flush=True)
print(f"[dbg] max={dbg[:,0].tolist()}", flush=True)
print(f"[dbg] total={dbg[:,1].tolist()}", flush=True)
print(f"[dbg] target={dbg[:,2].tolist()}", flush=True)
print(f"[dbg] second_rand={dbg[:,3].tolist()}", flush=True)
