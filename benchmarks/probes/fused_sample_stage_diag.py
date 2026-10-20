"""fused_sample 分阶段诊断:逐个阶段跑,定位崩在哪一步。

崩溃码 507035(type 3)通常是越界/非法地址。本探针把内核拆成只跑到第 N 阶段,
逐步加长,定位第一个崩的阶段。
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import tileinfer  # noqa: F401
import tilelang
import tilelang.ascend.language as T

DEV = torch.device("npu", 0)
torch.npu.set_device(DEV)

V = 16384
B = 2
VL = 64
BLK = 16384
NCHUNK = 1
NVEC = BLK // VL


def build_stage(stage: int):
    """stage=1 只做 max;2 加 sum;3 加搜索;4 加保留质量;5 加选择。"""

    @T.prim_func
    def main(
        logits: T.Tensor((B, V), T.float32),
        out: T.Tensor((B,), T.int32),
    ):
        with T.Kernel(B) as row:
            vals_ub = T.alloc_shared((BLK,), T.float32)
            st_ub = T.alloc_shared((8,), T.float32)

            # stage 1: max
            with T.SimtVF(threads=1):
                st_ub[1] = T.float32(-1e38)
            for c in T.serial(NCHUNK):
                T.copy(logits[row, c * BLK : (c + 1) * BLK], vals_ub)
                with T.SimdVF():
                    full = T.simd.pset(32)
                    one = T.simd.pset(32, "PAT_VL1")
                    acc = T.simd.alloc_local((1,), T.float32)
                    acc[0] = T.simd.vld(st_ub[1], "BRC_B32")
                    for g in T.serial(NVEC):
                        acc[0] = T.simd.vmax(acc[0], T.simd.vcmax(T.simd.vld(vals_ub[g * VL]), full), full)
                    T.simd.vsts(st_ub[1], acc[0], one, "ONEPT_B32")

            if stage >= 5:
                with T.SimtVF(threads=1):
                    out[row] = T.cast(st_ub[1], T.int32)

    return main


for stage in (1, 5):
    try:
        k = tilelang.compile(build_stage(stage), out_idx=None)
        logits = torch.randn(B, V, dtype=torch.float32, device=DEV)
        out = torch.zeros(B, dtype=torch.int32, device=DEV)
        k(logits, out)
        torch.npu.synchronize()
        if stage >= 5:
            ref = logits.argmax(dim=-1)
            print(f"stage={stage}: OK, max-cast vs argmax (informational) out={out.tolist()}", flush=True)
        else:
            print(f"stage={stage}: OK", flush=True)
    except Exception as e:
        print(f"stage={stage}: FAIL {type(e).__name__}: {str(e)[:200]}", flush=True)
