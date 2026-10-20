"""最小复现:单独验证 vexp / vdiv / vsel / vcmp 在 SimdVF 内是否可用。"""

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


def build(op: str):
    @T.prim_func
    def main(x: T.Tensor((N,), T.float32), out: T.Tensor((N,), T.float32)):
        with T.Kernel(1) as _:
            buf = T.alloc_shared((N,), T.float32)
            T.copy(x[:], buf)
            with T.SimdVF():
                full = T.simd.pset(32)
                for g in T.serial(NV):
                    v = T.simd.vld(buf[g * VL])
                    if op == "exp":
                        r = T.simd.vexp(v)
                    elif op == "mul":
                        r = T.simd.vmul(v, T.simd.vdup(2.0, T.float32))
                    elif op == "cmp_sel":
                        m = T.simd.vcmp(v, T.simd.vdup(0.0, T.float32), full, "ge")
                        r = T.simd.vsel(v, T.simd.vdup(-1.0, T.float32, full), m)
                    elif op == "cadd":
                        s = T.simd.vcadd(v, full)
                        r = T.simd.vdupv(s, full, "POS_LOWEST")
                    else:
                        r = v
                    T.simd.vsts(buf[g * VL], r)
            # 回写必须在 SimdVF 块**之后**单独做(SIMD 块内的 vsts 是否对
            # 后续 T.copy 可见取决于编译器插入的 mem_bar;显式分块更可靠)。
            T.copy(buf, out[:])
            # 诊断:把中间值也写出来,便于判断是编译崩还是执行崩

    return main


x = torch.randn(N, dtype=torch.float32, device=DEV)
for op in ("pass", "mul", "exp", "cmp_sel", "cadd"):
    try:
        k = tilelang.compile(build(op), out_idx=None)
        out = torch.zeros(N, dtype=torch.float32, device=DEV)
        k(x, out)
        torch.npu.synchronize()
        ref = {
            "pass": x,
            "mul": x * 2,
            "exp": torch.exp(x),
            "cmp_sel": torch.where(x >= 0, x, torch.full_like(x, -1.0)),
            "cadd": None,
        }[op]
        if ref is not None:
            err = (out - ref).abs().max().item()
            print(f"[op] {op:8s}: OK maxerr={err:.6g}", flush=True)
        else:
            print(f"[op] {op:8s}: OK (reduction, out[0]={out[0].item():.6g})", flush=True)
        del k
        torch.npu.empty_cache()
    except Exception as e:
        print(f"[op] {op:8s}: FAIL {type(e).__name__}: {str(e)[:150]}", flush=True)
