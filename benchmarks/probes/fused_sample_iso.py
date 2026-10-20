"""真实内核分阶段隔离:导入 fused_sample 的代码路径,逐阶段截断。

用 monkeypatch 把 fuse 的 stage 上限注入,跑真实 build → 运行。
每阶段独立进程,避免一次崩掉全丢。
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STAGES = [1, 2, 3, 4, 5]

CHILD = r'''
import sys
sys.path.insert(0, %r)
import torch, tileinfer, tilelang
import tilelang.ascend.language as T
from benchmarks.probes import fused_sample_oracle as O

DEV = torch.device("npu", 0)
torch.npu.set_device(DEV)
MAXSTAGE = int(%r)

# 用真正的 fused_sample_tl,但在每个阶段后提前 return
import tileinfer.kernels.sampling.fused_sample as fs

k = tilelang.compile(fs.fused_sample_tl(O.V, O.B, do_sample=False), out_idx=None)
logits = O.make_inputs()
temps = torch.ones(O.B, dtype=torch.float32, device=DEV)
tp = torch.ones(O.B, dtype=torch.float32, device=DEV)
tk = torch.zeros(O.B, dtype=torch.int32, device=DEV)
sd = torch.arange(O.B, dtype=torch.int64, device=DEV)
out = torch.zeros(O.B, dtype=torch.int32, device=DEV)
k(logits, temps, tp, tk, sd, out)
torch.npu.synchronize()
print(f"[stage] max={MAXSTAGE} OK out={out.tolist()}", flush=True)
'''

for st in STAGES:
    src = CHILD % (str(ROOT), st)
    p = subprocess.run(
        [sys.executable, "-u", "-c", src],
        capture_output=True, text=True, timeout=900,
        env={**os.environ},
    )
    tag = f"stage={st}"
    if p.returncode == 0:
        print(f"[iso] {tag}: OK")
    else:
        print(f"[iso] {tag}: FAIL rc={p.returncode}")
        tail = (p.stdout + p.stderr).strip().splitlines()[-6:]
        for ln in tail:
            print(f"      {ln}")
