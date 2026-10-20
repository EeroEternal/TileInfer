"""在 V=16384 下复刻主内核结构,但用极简实现,确认规模是否是因素。"""

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
B = 1
VL = 64
P = fused_sample_plan(V, B)
BLK, NCH, NV = P["blk"], P["n_chunks"], P["n_vec"]
print(f"[big] blk={BLK} n_chunks={NCH} n_vec={NV}", flush=True)


@T.prim_func
def main(
    logits: T.Tensor((B, V), T.float32),
    out_tok: T.Tensor((B,), T.int32),
):
    with T.Kernel(B) as row:
        buf = T.alloc_shared((BLK,), T.float32)
        sid = T.alloc_shared((8,), T.float32)
        red = T.alloc_shared((4,), T.float32)

        # max:多 chunk,每 chunk 只写 red,标量域合并
        with T.SimtVF(threads=1):
            sid[0] = T.float32(-1e38)
        for c in T.serial(NCH):
            T.copy(logits[row, c * BLK : (c + 1) * BLK], buf)
            with T.SimdVF():
                full = T.simd.pset(32)
                one = T.simd.pset(32, "PAT_VL1")
                acc = T.simd.alloc_local((1,), T.float32)
                acc[0] = T.simd.vdup(-1e38, T.float32, full)
                for g in T.serial(NV):
                    acc[0] = T.simd.vmax(acc[0], T.simd.vcmax(T.simd.vld(buf[g * VL]), full), full)
                T.simd.vsts(red[0], acc[0], one, "ONEPT_B32")
            with T.SimtVF(threads=1):
                sid[0] = T.max(sid[0], red[0])

        # sum
        with T.SimtVF(threads=1):
            sid[1] = T.float32(0.0)
        for c in T.serial(NCH):
            T.copy(logits[row, c * BLK : (c + 1) * BLK], buf)
            with T.SimdVF():
                full = T.simd.pset(32)
                one = T.simd.pset(32, "PAT_VL1")
                mx = T.simd.vld(sid[0], "BRC_B32")
                acc = T.simd.alloc_local((1,), T.float32)
                acc[0] = T.simd.vdup(0.0, T.float32, full)
                for g in T.serial(NV):
                    acc[0] = T.simd.vadd(
                        acc[0], T.simd.vexp(T.simd.vsub(T.simd.vld(buf[g * VL]), mx)), full
                    )
                T.simd.vsts(red[0], acc[0], one, "ONEPT_B32")
            with T.SimtVF(threads=1):
                sid[1] = sid[1] + red[0]

        # 选择:标量 argmax(只用已加载的 buf,最后 chunk)
        with T.SimtVF(threads=1):
            sid[2] = T.float32(-1e38)
            out_tok[row] = 0
        for c in T.serial(NCH):
            T.copy(logits[row, c * BLK : (c + 1) * BLK], buf)
            for i in T.serial(BLK):
                hi = buf[i] > sid[2]
                sid[2] = T.Select(hi, buf[i], sid[2])
                out_tok[row] = T.Select(hi, c * BLK + i, out_tok[row])


k = tilelang.compile(main, out_idx=None)
print("[big] compiled", flush=True)

torch.manual_seed(0)
logits = torch.randn(B, V, dtype=torch.float32, device=DEV)
out = torch.zeros(B, dtype=torch.int32, device=DEV)
k(logits, out)
torch.npu.synchronize()
ref = torch.argmax(logits, dim=-1).to(torch.int32)
print(f"[big] out={out.tolist()} ref={ref.tolist()} "
      f"{'PASS' if torch.equal(out, ref) else 'FAIL'}", flush=True)
