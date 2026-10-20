"""验证 T.rng_init / T.rng_rand 在 SimtVF 内的行为:是否随 seed 变化。"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import tileinfer  # noqa: F401
import tilelang
import tilelang.ascend.language as T

DEV = torch.device("npu", 0)
torch.npu.set_device(DEV)

B = 8


@T.prim_func
def main(seed: T.Tensor((B,), T.int64), out: T.Tensor((B,), T.float32)):
    with T.Kernel(B) as row:
        with T.SimtVF(threads=1):
            T.rng_init(seed[row], 0)
            out[row] = T.cast(T.rng_rand(), T.float32) / T.float32(4294967296.0)


k = tilelang.compile(main, out_idx=None)
seeds = torch.arange(B, dtype=torch.int64, device=DEV) * 7919 + 13
out = torch.zeros(B, dtype=torch.float32, device=DEV)
k(seeds, out)
torch.npu.synchronize()
vals = out.tolist()
print("[rng] values:", [round(v, 6) for v in vals], flush=True)
uniq = len(set(round(v, 6) for v in vals))
print(f"[rng] distinct={uniq}/{B} -> {'PASS' if uniq >= B - 1 else 'FAIL(seeds 未生效)'}",
      flush=True)
