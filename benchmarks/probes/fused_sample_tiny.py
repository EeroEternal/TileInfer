"""最小规模 greedy 验证:V=256(4 个向量),B=1,确认内核基本正确性。

先证明"小规模能跑对",再逐步放大。若这里就崩,说明是结构性错误。
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import tileinfer  # noqa: F401
import tilelang
from tileinfer.kernels.sampling.fused_sample import fused_sample_tl

DEV = torch.device("npu", 0)
torch.npu.set_device(DEV)

V = 256
B = 1

k = tilelang.compile(fused_sample_tl(V, B, do_sample=False), out_idx=None)
print("[tiny] compiled", flush=True)

torch.manual_seed(0)
logits = torch.randn(B, V, dtype=torch.float32, device=DEV)
temps = torch.ones(B, dtype=torch.float32, device=DEV)
tp = torch.ones(B, dtype=torch.float32, device=DEV)
tk = torch.zeros(B, dtype=torch.int32, device=DEV)
sd = torch.arange(B, dtype=torch.int64, device=DEV)
out = torch.zeros(B, dtype=torch.int32, device=DEV)

k(logits, temps, tp, tk, sd, out)
torch.npu.synchronize()
ref = torch.argmax(logits, dim=-1).to(torch.int32)
print(f"[tiny] out={out.tolist()} ref={ref.tolist()} "
      f"{'PASS' if torch.equal(out, ref) else 'FAIL'}", flush=True)
