"""阶段 3 精确复刻探针:把主内核阶段 3 的代码原样搬进来,逐项删减定位。

变体:
  full      = 完整阶段 3(三档质量 + 三档数量 + 收窄)
  no_cnt    = 去掉三档数量归约
  no_narrow = 去掉收窄(只统计不更新 bounds)
  one_round = MAX_SEARCH_ROUNDS=1
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
tmpdir = Path("/tmp/fs_s3b")
tmpdir.mkdir(exist_ok=True)

TMPL = r'''
import sys
sys.path.insert(0, %(root)r)
import torch, tileinfer, tilelang
import tilelang.ascend.language as T

DEV = torch.device("npu", 0)
torch.npu.set_device(DEV)
V, B = 256, 1
BLK, NCH, NV = V, 1, V // 64
ROUNDS = %(rounds)d

@T.prim_func
def main(logits: T.Tensor((B, V), T.float32), out_tok: T.Tensor((B,), T.int32)):
    with T.Kernel(B) as row:
        vals_ub = T.alloc_shared((BLK,), T.float32)
        st_ub = T.alloc_shared((8,), T.float32)
        quart_ub = T.alloc_shared((4,), T.float32)
        bounds_ub = T.alloc_shared((4,), T.float32)
        red_ub = T.alloc_shared((4,), T.float32)
        %(decls)s

        with T.SimtVF(threads=1):
            st_ub[0] = T.float32(1.0)
            st_ub[1] = T.float32(0.0)
            st_ub[2] = T.float32(1.0)

        for _r in T.serial(ROUNDS):
            with T.SimdVF():
                full = T.simd.pset(32)
                one = T.simd.pset(32, "PAT_VL1")
                lo_i = T.reinterpret(T.simd.vld(bounds_ub[0], "BRC_B32"), T.int32x64)
                hi_i = T.reinterpret(T.simd.vld(bounds_ub[1], "BRC_B32"), T.int32x64)
                width = T.simd.vsub(hi_i, lo_i, full)
                q0 = T.simd.vadd(lo_i, T.simd.vshrs(width, 2, full), full)
                q1 = T.simd.vadd(lo_i, T.simd.vshrs(width, 1, full), full)
                q2 = T.simd.vsub(hi_i, T.simd.vshrs(T.simd.vadds(width, 3, full), 2, full), full)
                T.simd.vsts(quart_ub[0], T.reinterpret(q0, T.float32x64), one, "ONEPT_B32")
                T.simd.vsts(quart_ub[1], T.reinterpret(q1, T.float32x64), one, "ONEPT_B32")
                T.simd.vsts(quart_ub[2], T.reinterpret(q2, T.float32x64), one, "ONEPT_B32")

            %(zero)s

            for c in T.serial(NCH):
                T.copy(logits[row, c * BLK : (c + 1) * BLK], vals_ub)
                with T.SimdVF():
                    full = T.simd.pset(32)
                    one = T.simd.pset(32, "PAT_VL1")
                    inv = T.simd.vld(st_ub[0], "BRC_B32")
                    mx = T.simd.vld(st_ub[1], "BRC_B32")
                    q0v = T.simd.vld(quart_ub[0], "BRC_B32")
                    q1v = T.simd.vld(quart_ub[1], "BRC_B32")
                    q2v = T.simd.vld(quart_ub[2], "BRC_B32")
                    a0 = T.simd.alloc_local((1,), T.float32)
                    a1 = T.simd.alloc_local((1,), T.float32)
                    a2 = T.simd.alloc_local((1,), T.float32)
                    a0[0] = T.simd.vdup(0.0, T.float32, full)
                    a1[0] = T.simd.vdup(0.0, T.float32, full)
                    a2[0] = T.simd.vdup(0.0, T.float32, full)
                    %(cntdecl)s
                    for g in T.serial(NV):
                        e = T.simd.vexp(T.simd.vsub(T.simd.vmul(T.simd.vld(vals_ub[g * 64]), inv), mx))
                        a0[0] = T.simd.vadd(a0[0], e, T.simd.vcmp(e, q0v, full, "ge"), "MODE_MERGING")
                        a1[0] = T.simd.vadd(a1[0], e, T.simd.vcmp(e, q1v, full, "ge"), "MODE_MERGING")
                        a2[0] = T.simd.vadd(a2[0], e, T.simd.vcmp(e, q2v, full, "ge"), "MODE_MERGING")
                        %(cntacc)s
                    T.simd.vsts(red_ub[0], a0[0], one, "ONEPT_B32")
                    T.simd.vsts(red_ub[1], a1[0], one, "ONEPT_B32")
                    T.simd.vsts(red_ub[2], a2[0], one, "ONEPT_B32")
                    %(cntstore)s
                with T.SimtVF(threads=1):
                    sums_ub[0] = sums_ub[0] + red_ub[0]
                    sums_ub[1] = sums_ub[1] + red_ub[1]
                    sums_ub[2] = sums_ub[2] + red_ub[2]
                    %(cntadd)s

            %(narrow)s

        with T.SimtVF(threads=1):
            out_tok[row] = T.cast(sums_ub[0], T.int32)

k = tilelang.compile(main, out_idx=None)
print("[s3b] compiled", flush=True)
logits = torch.randn(B, V, dtype=torch.float32, device=DEV)
out = torch.zeros(B, dtype=torch.int32, device=DEV)
k(logits, out)
torch.npu.synchronize()
print(f"[s3b] OK out={out.tolist()}", flush=True)
'''

NO_CNT = {
    "decls": "sums_ub = T.alloc_shared((4,), T.float32)",
    "zero": "with T.SimtVF(threads=1):\n                sums_ub[0] = T.float32(0.0)\n                sums_ub[1] = T.float32(0.0)\n                sums_ub[2] = T.float32(0.0)",
    "cntdecl": "",
    "cntacc": "",
    "cntstore": "",
    "cntadd": "",
    "narrow": """with T.SimtVF(threads=1):
                if sums_ub[2] >= sums_ub[0]:
                    bounds_ub[0] = quart_ub[2]
                else:
                    bounds_ub[1] = quart_ub[2]""",
}

WITH_CNT = {
    "decls": "sums_ub = T.alloc_shared((4,), T.float32)\n        cnt_ub = T.alloc_shared((8,), T.int32)",
    "zero": "with T.SimtVF(threads=1):\n                sums_ub[0] = T.float32(0.0)\n                sums_ub[1] = T.float32(0.0)\n                sums_ub[2] = T.float32(0.0)\n                cnt_ub[0] = 0\n                cnt_ub[1] = 0\n                cnt_ub[2] = 0",
    "cntdecl": """n0 = T.simd.alloc_local((1,), T.int32)
                    n1 = T.simd.alloc_local((1,), T.int32)
                    n2 = T.simd.alloc_local((1,), T.int32)
                    n0[0] = T.simd.vdup(0, T.int32, full)
                    n1[0] = T.simd.vdup(0, T.int32, full)
                    n2[0] = T.simd.vdup(0, T.int32, full)
                    onei = T.simd.vdup(1, T.int32, full)""",
    "cntacc": """n0[0] = T.simd.vadd(n0[0], onei, T.simd.vcmp(e, q0v, full, "ge"), "MODE_MERGING")
                        n1[0] = T.simd.vadd(n1[0], onei, T.simd.vcmp(e, q1v, full, "ge"), "MODE_MERGING")
                        n2[0] = T.simd.vadd(n2[0], onei, T.simd.vcmp(e, q2v, full, "ge"), "MODE_MERGING")""",
    "cntstore": """T.simd.vsts(cnt_ub[3], T.simd.vcadd(n0[0], full))
                    T.simd.vsts(cnt_ub[4], T.simd.vcadd(n1[0], full))
                    T.simd.vsts(cnt_ub[5], T.simd.vcadd(n2[0], full))""",
    "cntadd": """cnt_ub[0] = cnt_ub[0] + cnt_ub[3]
                    cnt_ub[1] = cnt_ub[1] + cnt_ub[4]
                    cnt_ub[2] = cnt_ub[2] + cnt_ub[5]""",
    "narrow": """with T.SimtVF(threads=1):
                if sums_ub[2] >= sums_ub[0]:
                    bounds_ub[0] = quart_ub[2]
                elif sums_ub[1] >= sums_ub[0]:
                    bounds_ub[0] = quart_ub[1]
                else:
                    bounds_ub[0] = quart_ub[0]""",
}

CASES = [
    ("no_cnt_r16", NO_CNT, 16),
    ("with_cnt_r16", WITH_CNT, 16),
    ("with_cnt_r1", WITH_CNT, 1),
    ("no_cnt_r1", NO_CNT, 1),
]

for name, sub, rounds in CASES:
    src = TMPL % {"root": str(ROOT), "rounds": rounds, **sub}
    tmp = tmpdir / f"{name}.py"
    tmp.write_text(src)
    try:
        p = subprocess.run([sys.executable, "-u", str(tmp)],
                           capture_output=True, text=True, timeout=900, env={**os.environ})
        Path(f"/tmp/fs_s3b/{name}.log").write_text(p.stdout + p.stderr)
        if p.returncode == 0:
            print(f"[s3b] {name}: OK")
        else:
            allout = (p.stdout + p.stderr).splitlines()
            err = [ln for ln in allout if "Error" in ln][-1:] or allout[-1:]
            print(f"[s3b] {name}: FAIL rc={p.returncode} | {err}")
    except subprocess.TimeoutExpired:
        print(f"[s3b] {name}: TIMEOUT")
