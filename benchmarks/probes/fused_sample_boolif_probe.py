"""验证 T.prim_func 内 `if <python bool>:` 是否按编译期常量求值。

预期:flag=True 时应只编译 A 分支(输出 111),False 时只编译 B 分支(输出 222)。
若两者都输出同一个值,说明 DSL 把 Python bool 当成了 TIR 条件(错误)。
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
B = 1


def build(flag: bool):
    @T.prim_func
    def main(out: T.Tensor((B,), T.int32)):
        with T.Kernel(B) as row:
            with T.SimtVF(threads=1):
                if flag:
                    out[row] = T.int32(111)
                else:
                    out[row] = T.int32(222)

    return main


for flag in (True, False):
    k = tilelang.compile(build(flag), out_idx=None)
    out = torch.zeros(B, dtype=torch.int32, device=DEV)
    k(out)
    torch.npu.synchronize()
    expect = 111 if flag else 222
    got = out[0].item()
    print(f"[boolif] flag={flag}: out={got} expect={expect} "
          f"{'PASS' if got == expect else 'FAIL'}", flush=True)
    del k
    torch.npu.empty_cache()
