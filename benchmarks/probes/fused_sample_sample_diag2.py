"""诊断采样分支内部状态:keep_mass / u / 阈值 / 是否更新 pick。

用一个最小内核复刻采样阶段 4,输出中间量。
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

V, B = 256, 2
P = fused_sample_plan(V, B)
BLK, NCH, NV = P["blk"], P["n_chunks"], P["n_vec"]


@T.prim_func
def main(
    logits: T.Tensor((B, V), T.float32),
    seed: T.Tensor((B,), T.int64),
    dbg: T.Tensor((B, 6), T.float32),
    out: T.Tensor((B,), T.int32),
):
    with T.Kernel(B) as row:
        vals_ub = T.alloc_shared((BLK,), T.float32)
        st = T.alloc_shared((8,), T.float32)
        red = T.alloc_shared((4,), T.float32)
        pick = T.alloc_shared((4,), T.int32)
        done = T.alloc_shared((4,), T.int32)
        bounds = T.alloc_shared((4,), T.float32)

        # max
        with T.SimtVF(threads=1):
            st[1] = T.float32(-1e38)
            bounds[0] = T.float32(0.0)
        for c in T.serial(NCH):
            T.copy(logits[row, c * BLK : (c + 1) * BLK], vals_ub)
            with T.SimdVF():
                full = T.simd.pset(32)
                one = T.simd.pset(32, "PAT_VL1")
                acc = T.simd.alloc_local((1,), T.float32)
                acc[0] = T.simd.vdup(-1e38, T.float32, full)
                for g in T.serial(NV):
                    acc[0] = T.simd.vmax(acc[0], T.simd.vcmax(T.simd.vld(vals_ub[g * 64]), full), full)
                T.simd.vsts(red[0], acc[0], one, "ONEPT_B32")
            with T.SimtVF(threads=1):
                st[1] = T.max(st[1], red[0])

        # keep_mass(阈值=0 → 全保留)
        with T.SimtVF(threads=1):
            st[3] = T.float32(0.0)
        for c in T.serial(NCH):
            T.copy(logits[row, c * BLK : (c + 1) * BLK], vals_ub)
            with T.SimdVF():
                full = T.simd.pset(32)
                one = T.simd.pset(32, "PAT_VL1")
                mx = T.simd.vld(st[1], "BRC_B32")
                thr = T.simd.vld(bounds[0], "BRC_B32")
                acc = T.simd.alloc_local((1,), T.float32)
                acc[0] = T.simd.vdup(0.0, T.float32, full)
                for g in T.serial(NV):
                    e = T.simd.vexp(T.simd.vsub(T.simd.vld(vals_ub[g * 64]), mx))
                    em = T.simd.vsel(e, T.simd.vdup(0.0, T.float32, full), T.simd.vcmp(e, thr, full, "ge"))
                    acc[0] = T.simd.vadd(acc[0], em, full)
                T.simd.vsts(red[0], acc[0], one, "ONEPT_B32")
            with T.SimtVF(threads=1):
                st[3] = st[3] + red[0]

        # u 与逆 CDF
        with T.SimtVF(threads=1):
            T.rng_init(seed[row], 0)
            st[6] = T.cast(T.rng_rand(), T.float32) / T.float32(4294967296.0) * st[3]
            st[4] = T.float32(0.0)
            pick[0] = 0
            done[0] = 0
            dbg[row, 0] = st[1]
            dbg[row, 1] = st[3]
            dbg[row, 2] = st[6]

        for c in T.serial(NCH):
            T.copy(logits[row, c * BLK : (c + 1) * BLK], vals_ub)
            for g in T.serial(NV):
                with T.SimdVF():
                    full = T.simd.pset(32)
                    one = T.simd.pset(32, "PAT_VL1")
                    mx = T.simd.vld(st[1], "BRC_B32")
                    thr = T.simd.vld(bounds[0], "BRC_B32")
                    e = T.simd.vexp(T.simd.vsub(T.simd.vld(vals_ub[g * 64]), mx))
                    hit = T.simd.vcmp(e, thr, full, "ge")
                    em = T.simd.vsel(e, T.simd.vdup(0.0, T.float32, full), hit)
                    T.simd.vsts(vals_ub[g * 64], em)
                    T.simd.vsts(red[0], T.simd.vdupv(T.simd.vcadd(em, full), full, "POS_LOWEST"), one, "ONEPT_B32")
                with T.SimtVF(threads=1):
                    spans = st[4] + red[0] >= st[6]
                    inside = T.min(done[0] == 0, spans)
                    for lane in T.serial(64):
                        idx = c * BLK + g * 64 + lane
                        proceed = T.min(inside, done[0] == 0)
                        nxt = st[4] + vals_ub[g * 64 + lane]
                        crossed = T.min(proceed, nxt >= st[6])
                        pick[0] = T.Select(crossed, idx, pick[0])
                        done[0] = T.Select(crossed, 1, done[0])
                        st[4] = T.Select(proceed, nxt, st[4])
                    st[4] = T.Select(T.min(done[0] == 0, inside == 0), st[4] + red[0], st[4])

        with T.SimtVF(threads=1):
            dbg[row, 3] = T.cast(pick[0], T.float32)
            dbg[row, 4] = T.cast(done[0], T.float32)
            dbg[row, 5] = st[4]
            out[row] = pick[0]


k = tilelang.compile(main, out_idx=None)
g = torch.Generator(device="cpu").manual_seed(0)
logits = (torch.randn(B, V, generator=g) * 3.0).to(torch.float32).to(DEV)
seeds = torch.tensor([13, 7929], dtype=torch.int64, device=DEV)
dbg = torch.zeros(B, 6, dtype=torch.float32, device=DEV)
out = torch.zeros(B, dtype=torch.int32, device=DEV)
k(logits, seeds, dbg, out)
torch.npu.synchronize()
for r in range(B):
    print(f"[sdiag] row{r}: max={dbg[r,0]:.3f} keep={dbg[r,1]:.3f} "
          f"target={dbg[r,2]:.3f} pick={int(dbg[r,3])} done={int(dbg[r,4])} "
          f"run={dbg[r,5]:.3f}", flush=True)
print(f"[sdiag] ref argmax={torch.argmax(logits, -1).tolist()}", flush=True)
