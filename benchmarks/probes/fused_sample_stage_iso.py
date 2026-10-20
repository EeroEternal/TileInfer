"""按 _stage_limit 逐段打开,定位崩溃发生在哪个阶段。每段独立进程。"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CHILD = r'''
import sys
sys.path.insert(0, %(root)r)
import torch, tileinfer, tilelang
from tileinfer.kernels.sampling.fused_sample import fused_sample_tl

DEV = torch.device("npu", 0)
torch.npu.set_device(DEV)
V, B = 256, 1
LIMIT = %(limit)d

k = tilelang.compile(fused_sample_tl(V, B, do_sample=False, _stage_limit=LIMIT), out_idx=None)
print("[st] compiled", flush=True)
g = torch.Generator(device="cpu").manual_seed(0)
logits = (torch.randn(B, V, generator=g) * 3.0).to(torch.float32).to(DEV)
temps = torch.ones(B, dtype=torch.float32, device=DEV)
tp = torch.ones(B, dtype=torch.float32, device=DEV)
tk = torch.zeros(B, dtype=torch.int32, device=DEV)
sd = torch.arange(B, dtype=torch.int64, device=DEV)
out = torch.zeros(B, dtype=torch.int32, device=DEV)
k(logits, temps, tp, tk, sd, out)
torch.npu.synchronize()
ref = torch.argmax(logits, dim=-1).to(torch.int32)
print(f"[st] LIMIT={LIMIT} out={out.tolist()} ref={ref.tolist()} "
      f"{'MATCH' if torch.equal(out, ref) else 'DIFF'}", flush=True)
'''

tmpdir = Path("/tmp/fs_stage")
tmpdir.mkdir(exist_ok=True)

for limit in (5, 4, 3, 2):
    tmp = tmpdir / f"st_{limit}.py"
    tmp.write_text(CHILD % {"root": str(ROOT), "limit": limit})
    try:
        p = subprocess.run([sys.executable, "-u", str(tmp)],
                           capture_output=True, text=True, timeout=900, env={**os.environ})
        if p.returncode == 0:
            line = [ln for ln in p.stdout.splitlines() if ln.startswith("[st] LIMIT")][-1:]
            print(f"[iso] limit={limit}: OK {line}")
        else:
            Path(f"/tmp/fs_stage/out_{limit}.log").write_text(p.stdout + p.stderr)
            allout = (p.stdout + p.stderr).splitlines()
            err = [ln for ln in allout if "507035" in ln or "Error" in ln][-1:]
            print(f"[iso] limit={limit}: FAIL rc={p.returncode} | {err}")
    except subprocess.TimeoutExpired:
        print(f"[iso] limit={limit}: TIMEOUT")
