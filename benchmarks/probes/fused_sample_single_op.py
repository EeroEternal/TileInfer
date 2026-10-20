"""单算子隔离探针:每次只测一个算子,独立进程,区分编译崩/执行崩。

用法: python fused_sample_single_op.py <op>
      op ∈ {pass, mul, exp, cmp, sel, cadd, max, addmask, reinterpret}
退出码: 0=OK, 2=编译崩, 3=执行崩, 4=结果错
"""

import sys
import traceback

import torch

import tileinfer  # noqa: F401
import tilelang
import tilelang.ascend.language as T

DEV = torch.device("npu", 0)
torch.npu.set_device(DEV)

N = 256
VL = 64
NV = N // VL
OP = sys.argv[1] if len(sys.argv) > 1 else "pass"


def build():
    @T.prim_func
    def main(x: T.Tensor((N,), T.float32), out: T.Tensor((N,), T.float32)):
        with T.Kernel(1) as _:
            buf = T.alloc_shared((N,), T.float32)
            sid = T.alloc_shared((4,), T.float32)
            T.copy(x[:], buf)
            if OP in ("max", "addmask"):
                with T.SimtVF(threads=1):
                    sid[0] = T.float32(-1e38)
                    sid[1] = T.float32(0.0)
            with T.SimdVF():
                full = T.simd.pset(32)
                one = T.simd.pset(32, "PAT_VL1")
                if OP == "max":
                    acc = T.simd.alloc_local((1,), T.float32)
                    acc[0] = T.simd.vld(sid[0], "BRC_B32")
                if OP == "addmask":
                    acc2 = T.simd.alloc_local((1,), T.float32)
                    acc2[0] = T.simd.vld(sid[1], "BRC_B32")
                for g in T.serial(NV):
                    v = T.simd.vld(buf[g * VL])
                    if OP == "pass":
                        r = v
                    elif OP == "mul":
                        r = T.simd.vmul(v, T.simd.vdup(2.0, T.float32))
                    elif OP == "exp":
                        r = T.simd.vexp(v)
                    elif OP == "cmp":
                        m = T.simd.vcmp(v, T.simd.vdup(0.0, T.float32), full, "ge")
                        r = T.simd.vsel(v, T.simd.vdup(-1.0, T.float32, full), m)
                    elif OP == "cadd":
                        s = T.simd.vcadd(v, full)
                        r = T.simd.vdupv(s, full, "POS_LOWEST")
                    elif OP == "max":
                        acc[0] = T.simd.vmax(acc[0], v, full)
                        r = v
                    elif OP == "addmask":
                        acc2[0] = T.simd.vadd(acc2[0], v, T.simd.vcmp(v, T.simd.vdup(0.0, T.float32), full, "ge"), "MODE_MERGING")
                        r = v
                    elif OP == "reinterpret":
                        iv = T.reinterpret(v, T.int32x64)
                        r = T.reinterpret(T.simd.vadd(iv, T.simd.vdup(1, T.int32, full), full), T.float32x64)
                    else:
                        r = v
                    T.simd.vsts(buf[g * VL], r)
                if OP == "max":
                    T.simd.vsts(sid[0], acc[0], one, "ONEPT_B32")
                if OP == "addmask":
                    T.simd.vsts(sid[1], acc2[0], one, "ONEPT_B32")
            T.copy(buf, out[:])

    return main


try:
    k = tilelang.compile(build(), out_idx=None)
except Exception:
    print(f"[one] {OP}: COMPILE_FAIL", flush=True)
    traceback.print_exc()
    sys.exit(2)

print(f"[one] {OP}: COMPILED", flush=True)
x = torch.randn(N, dtype=torch.float32, device=DEV)
out = torch.zeros(N, dtype=torch.float32, device=DEV)
try:
    k(x, out)
    torch.npu.synchronize()
except Exception:
    print(f"[one] {OP}: RUN_FAIL", flush=True)
    traceback.print_exc()
    sys.exit(3)

ref = {
    "pass": x,
    "mul": x * 2,
    "exp": torch.exp(x),
    "cmp": torch.where(x >= 0, x, torch.full_like(x, -1.0)),
    "reinterpret": None,
    "cadd": None,
    "max": None,
    "addmask": None,
}.get(OP)
if ref is not None:
    err = (out - ref).abs().max().item()
    print(f"[one] {OP}: OK maxerr={err:.6g}", flush=True)
    sys.exit(0 if err < 1e-5 else 4)
print(f"[one] {OP}: OK (out[0]={out[0].item():.6g})", flush=True)
sys.exit(0)
