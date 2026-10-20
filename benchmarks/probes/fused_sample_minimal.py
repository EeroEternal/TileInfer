"""最小可运行内核:只做 argmax(V=256),逐步加料定位崩溃点。

每个变体独立进程;用 subprocess 隔离,防止一次崩掉全丢。
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

VARIANTS = {
    "v0_copy": """
        with T.Kernel(B) as row:
            buf = T.alloc_shared((V,), T.float32)
            T.copy(logits[row, :], buf)
            T.copy(buf, out_f[:])
""",
    "v1_max_simt": """
        with T.Kernel(B) as row:
            buf = T.alloc_shared((V,), T.float32)
            sid = T.alloc_shared((4,), T.float32)
            T.copy(logits[row, :], buf)
            with T.SimtVF(threads=1):
                sid[0] = T.float32(-1e38)
                for i in T.serial(V):
                    sid[0] = T.max(sid[0], buf[i])
                out_tok[row] = T.cast(sid[0], T.int32)
""",
    "v2_max_simd": """
        with T.Kernel(B) as row:
            buf = T.alloc_shared((V,), T.float32)
            sid = T.alloc_shared((4,), T.float32)
            T.copy(logits[row, :], buf)
            with T.SimtVF(threads=1):
                sid[0] = T.float32(-1e38)
            with T.SimdVF():
                full = T.simd.pset(32)
                one = T.simd.pset(32, "PAT_VL1")
                acc = T.simd.alloc_local((1,), T.float32)
                acc[0] = T.simd.vdup(-1e38, T.float32, full)
                for g in T.serial(V // 64):
                    acc[0] = T.simd.vmax(acc[0], T.simd.vcmax(T.simd.vld(buf[g * 64]), full), full)
                T.simd.vsts(sid[0], acc[0], one, "ONEPT_B32")
            with T.SimtVF(threads=1):
                out_tok[row] = T.cast(sid[0], T.int32)
""",
    "v3_sum_simd": """
        with T.Kernel(B) as row:
            buf = T.alloc_shared((V,), T.float32)
            sid = T.alloc_shared((4,), T.float32)
            red = T.alloc_shared((4,), T.float32)
            T.copy(logits[row, :], buf)
            with T.SimtVF(threads=1):
                sid[0] = T.float32(-1e38)
                sid[1] = T.float32(0.0)
            with T.SimdVF():
                full = T.simd.pset(32)
                one = T.simd.pset(32, "PAT_VL1")
                acc = T.simd.alloc_local((1,), T.float32)
                acc[0] = T.simd.vdup(-1e38, T.float32, full)
                for g in T.serial(V // 64):
                    acc[0] = T.simd.vmax(acc[0], T.simd.vcmax(T.simd.vld(buf[g * 64]), full), full)
                T.simd.vsts(sid[0], acc[0], one, "ONEPT_B32")
            with T.SimdVF():
                full = T.simd.pset(32)
                one = T.simd.pset(32, "PAT_VL1")
                mx = T.simd.vld(sid[0], "BRC_B32")
                acc2 = T.simd.alloc_local((1,), T.float32)
                acc2[0] = T.simd.vdup(0.0, T.float32, full)
                for g in T.serial(V // 64):
                    acc2[0] = T.simd.vadd(acc2[0], T.simd.vexp(T.simd.vsub(T.simd.vld(buf[g * 64]), mx)), full)
                T.simd.vsts(sid[1], acc2[0], one, "ONEPT_B32")
            with T.SimtVF(threads=1):
                out_tok[row] = T.cast(sid[1], T.int32)
""",
    "v4_search_reinterpret": """
        with T.Kernel(B) as row:
            buf = T.alloc_shared((V,), T.float32)
            sid = T.alloc_shared((8,), T.float32)
            T.copy(logits[row, :], buf)
            with T.SimtVF(threads=1):
                sid[0] = T.float32(1e-9)
                sid[1] = T.float32(1.0)
            with T.SimdVF():
                full = T.simd.pset(32)
                one = T.simd.pset(32, "PAT_VL1")
                lo_i = T.reinterpret(T.simd.vld(sid[0], "BRC_B32"), T.int32x64)
                hi_i = T.reinterpret(T.simd.vld(sid[1], "BRC_B32"), T.int32x64)
                width = T.simd.vsub(hi_i, lo_i, full)
                q0 = T.simd.vadd(lo_i, T.simd.vshrs(width, 2, full), full)
                T.simd.vsts(sid[2], T.reinterpret(q0, T.float32x64), one, "ONEPT_B32")
            with T.SimtVF(threads=1):
                out_tok[row] = T.cast(sid[2], T.int32)
""",
    "v5_select_loop": """
        with T.Kernel(B) as row:
            buf = T.alloc_shared((V,), T.float32)
            sid = T.alloc_shared((8,), T.float32)
            red = T.alloc_shared((4,), T.float32)
            T.copy(logits[row, :], buf)
            with T.SimtVF(threads=1):
                sid[0] = T.float32(-1e38)
                sid[1] = T.float32(0.5)
                sid[2] = T.float32(0.0)
            with T.SimdVF():
                full = T.simd.pset(32)
                one = T.simd.pset(32, "PAT_VL1")
                acc = T.simd.alloc_local((1,), T.float32)
                acc[0] = T.simd.vdup(-1e38, T.float32, full)
                for g in T.serial(V // 64):
                    acc[0] = T.simd.vmax(acc[0], T.simd.vcmax(T.simd.vld(buf[g * 64]), full), full)
                T.simd.vsts(sid[0], acc[0], one, "ONEPT_B32")
            with T.SimtVF(threads=1):
                for i in T.serial(V):
                    if buf[i] == sid[0]:
                        sid[2] = T.float32(1.0)
                out_tok[row] = T.cast(sid[0], T.int32)
""",
}

CHILD = r'''
import sys
sys.path.insert(0, %(root)r)
import torch, tileinfer, tilelang
import tilelang.ascend.language as T
DEV = torch.device("npu", 0)
torch.npu.set_device(DEV)
V, B = 256, 1

def build():
    @T.prim_func
    def main(logits: T.Tensor((B, V), T.float32),
             out_tok: T.Tensor((B,), T.int32),
             out_f: T.Tensor((V,), T.float32)):
%(body)s
    return main

k = tilelang.compile(build(), out_idx=None)
logits = torch.randn(B, V, dtype=torch.float32, device=DEV)
tok = torch.zeros(B, dtype=torch.int32, device=DEV)
of = torch.zeros(V, dtype=torch.float32, device=DEV)
k(logits, tok, of)
torch.npu.synchronize()
print("[var] OK", flush=True)
'''

for name, body in VARIANTS.items():
    body_ind = "\n".join("        " + ln if ln.strip() else ln for ln in body.strip().splitlines())
    src = CHILD % {"root": str(ROOT), "body": body_ind}
    # tilelang 需要 inspect.getsource,`-c` 传代码拿不到 → 写临时文件再跑
    tmp = Path(f"/tmp/fs_var_{name}.py")
    tmp.write_text(src)
    try:
        p = subprocess.run([sys.executable, "-u", str(tmp)],
                           capture_output=True, text=True, timeout=600)
        if p.returncode == 0:
            print(f"[var] {name}: OK")
        else:
            msg = (p.stdout + p.stderr).strip().splitlines()
            err = [ln for ln in msg if "Error" in ln or "error" in ln][-1:] or msg[-1:]
            print(f"[var] {name}: FAIL rc={p.returncode} | {err}")
    except subprocess.TimeoutExpired:
        print(f"[var] {name}: TIMEOUT")
