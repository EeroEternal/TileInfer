"""阶段 3 结构探针:三层嵌套(serial → copy → SimdVF → SimtVF)是否可用。

逐个增加嵌套层数,定位崩在哪个组合。
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
tmpdir = Path("/tmp/fs_s3")
tmpdir.mkdir(exist_ok=True)

V = 256
B = 1
BLK = 256
NCH = 1
NV = 4

BASE = r'''
import sys
sys.path.insert(0, %(root)r)
import torch, tileinfer, tilelang
import tilelang.ascend.language as T
DEV = torch.device("npu", 0)
torch.npu.set_device(DEV)
V, B, BLK, NCH, NV = %(V)d, %(B)d, %(BLK)d, %(NCH)d, %(NV)d

@T.prim_func
def main(logits: T.Tensor((B, V), T.float32), out_tok: T.Tensor((B,), T.int32)):
    with T.Kernel(B) as row:
        buf = T.alloc_shared((BLK,), T.float32)
        sid = T.alloc_shared((8,), T.float32)
        red = T.alloc_shared((8,), T.float32)
%(body)s

k = tilelang.compile(main, out_idx=None)
print("[s3] compiled", flush=True)
logits = torch.randn(B, V, dtype=torch.float32, device=DEV)
out = torch.zeros(B, dtype=torch.int32, device=DEV)
k(logits, out)
torch.npu.synchronize()
print(f"[s3] OK out={out.tolist()}", flush=True)
'''

BODIES = {
    # A: 外层 serial + copy + SimdVF + SimtVF(单轮)
    "A_one_round": """
        with T.SimtVF(threads=1):
            sid[0] = T.float32(1e-9)
            sid[1] = T.float32(1.0)
            sid[2] = T.float32(0.0)
            sid[3] = T.float32(0.0)
        for r in T.serial(1):
            for c in T.serial(NCH):
                T.copy(logits[row, c * BLK : (c + 1) * BLK], buf)
                with T.SimdVF():
                    full = T.simd.pset(32)
                    one = T.simd.pset(32, "PAT_VL1")
                    lo_i = T.reinterpret(T.simd.vld(sid[0], "BRC_B32"), T.int32x64)
                    hi_i = T.reinterpret(T.simd.vld(sid[1], "BRC_B32"), T.int32x64)
                    w = T.simd.vsub(hi_i, lo_i, full)
                    q = T.simd.vadd(lo_i, T.simd.vshrs(w, 1, full), full)
                    T.simd.vsts(sid[2], T.reinterpret(q, T.float32x64), one, "ONEPT_B32")
                with T.SimtVF(threads=1):
                    sid[3] = sid[3] + sid[2]
        with T.SimtVF(threads=1):
            out_tok[row] = T.cast(sid[3], T.int32)
""",
    # B: 多轮(16)+ 同上
    "B_16_rounds": """
        with T.SimtVF(threads=1):
            sid[0] = T.float32(1e-9)
            sid[1] = T.float32(1.0)
            sid[2] = T.float32(0.0)
            sid[3] = T.float32(0.0)
        for r in T.serial(16):
            for c in T.serial(NCH):
                T.copy(logits[row, c * BLK : (c + 1) * BLK], buf)
                with T.SimdVF():
                    full = T.simd.pset(32)
                    one = T.simd.pset(32, "PAT_VL1")
                    lo_i = T.reinterpret(T.simd.vld(sid[0], "BRC_B32"), T.int32x64)
                    hi_i = T.reinterpret(T.simd.vld(sid[1], "BRC_B32"), T.int32x64)
                    w = T.simd.vsub(hi_i, lo_i, full)
                    q = T.simd.vadd(lo_i, T.simd.vshrs(w, 1, full), full)
                    T.simd.vsts(sid[2], T.reinterpret(q, T.float32x64), one, "ONEPT_B32")
                with T.SimtVF(threads=1):
                    sid[3] = sid[3] + sid[2]
        with T.SimtVF(threads=1):
            out_tok[row] = T.cast(sid[3], T.int32)
""",
    # C: 多轮 + 每轮内层 SimdVF 里再套 for g 扫向量
    "C_16_rounds_vecscan": """
        with T.SimtVF(threads=1):
            sid[0] = T.float32(1e-9)
            sid[1] = T.float32(1.0)
            sid[2] = T.float32(0.0)
            sid[3] = T.float32(0.0)
        for r in T.serial(16):
            for c in T.serial(NCH):
                T.copy(logits[row, c * BLK : (c + 1) * BLK], buf)
                with T.SimdVF():
                    full = T.simd.pset(32)
                    one = T.simd.pset(32, "PAT_VL1")
                    lo_i = T.reinterpret(T.simd.vld(sid[0], "BRC_B32"), T.int32x64)
                    hi_i = T.reinterpret(T.simd.vld(sid[1], "BRC_B32"), T.int32x64)
                    w = T.simd.vsub(hi_i, lo_i, full)
                    q = T.simd.vadd(lo_i, T.simd.vshrs(w, 1, full), full)
                    qf = T.reinterpret(q, T.float32x64)
                    acc = T.simd.alloc_local((1,), T.float32)
                    acc[0] = T.simd.vdup(0.0, T.float32, full)
                    for g in T.serial(NV):
                        acc[0] = T.simd.vadd(acc[0], T.simd.vld(buf[g * 64]), T.simd.vcmp(T.simd.vld(buf[g * 64]), qf, full, "ge"), "MODE_MERGING")
                    T.simd.vsts(sid[2], acc[0], one, "ONEPT_B32")
                with T.SimtVF(threads=1):
                    sid[3] = sid[3] + sid[2]
        with T.SimtVF(threads=1):
            out_tok[row] = T.cast(sid[3], T.int32)
""",
}

for name, body in BODIES.items():
    src = BASE % {"root": str(ROOT), "V": V, "B": B, "BLK": BLK, "NCH": NCH, "NV": NV,
                  "body": body}
    tmp = tmpdir / f"{name}.py"
    tmp.write_text(src)
    try:
        p = subprocess.run([sys.executable, "-u", str(tmp)],
                           capture_output=True, text=True, timeout=900, env={**os.environ})
        Path(f"/tmp/fs_s3/{name}.log").write_text(p.stdout + p.stderr)
        if p.returncode == 0:
            print(f"[s3] {name}: OK")
        else:
            allout = (p.stdout + p.stderr).splitlines()
            err = [ln for ln in allout if "Error" in ln][-1:] or allout[-1:]
            print(f"[s3] {name}: FAIL rc={p.returncode} | {err}")
    except subprocess.TimeoutExpired:
        print(f"[s3] {name}: TIMEOUT")
