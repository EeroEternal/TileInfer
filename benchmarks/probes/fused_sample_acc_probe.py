"""验证 alloc_local 在 T.serial 内的跨迭代累加是否正确。"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import tileinfer  # noqa: F401
import tilelang
import tilelang.ascend.language as T

DEV = torch.device("npu", 0)
torch.npu.set_device(DEV)

N = 256
VL = 64
NV = N // VL
B = 1


@T.prim_func
def main(x: T.Tensor((B, N), T.float32), out: T.Tensor((B,), T.float32)):
    with T.Kernel(B) as row:
        buf = T.alloc_shared((N,), T.float32)
        T.copy(x[row, :], buf)
        with T.SimdVF():
            full = T.simd.pset(32)
            one = T.simd.pset(32, "PAT_VL1")
            acc = T.simd.alloc_local((1,), T.float32)
            acc[0] = T.simd.vdup(0.0, T.float32, full)
            for g in T.serial(NV):
                acc[0] = T.simd.vadd(acc[0], T.simd.vld(buf[g * VL]), full)
            T.simd.vsts(out[row], T.simd.vdupv(T.simd.vcadd(acc[0], full), full, "POS_LOWEST"), one, "ONEPT_B32")


k = tilelang.compile(main, out_idx=None)
x = torch.ones(B, N, dtype=torch.float32, device=DEV)
out = torch.zeros(B, dtype=torch.float32, device=DEV)
k(x, out)
torch.npu.synchronize()
print(f"[acc] sum={out[0].item():.1f} expect={float(N):.1f} "
      f"{'PASS' if abs(out[0].item() - N) < 1e-3 else 'FAIL'}", flush=True)
