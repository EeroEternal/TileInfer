"""Device-time comparison (op_summary) for all bench scenarios.

The wall-clock numbers mix in the TileLang FFI host launch cost (~6 ms/call,
E13), which vanishes under graph capture/replay — so the kernel-level verdict
must be device time.  This script runs each scenario under the NPU profiler
and reports per-step device kernel time for both chains.
"""

import csv
import glob
import json
import os
import subprocess
import sys

SCRIPT = """
import json, sys, torch, torch_npu
from torch_npu.profiler import profile, ProfilerActivity
sys.path.insert(0, "/home/userlist/zxs/TileInfer/benchmarks")
import bench_moe_grouped as B
from bench_moe_grouped import build_case

g = torch.Generator(device="cpu").manual_seed(42)
B.w13_tl = (torch.randn((B.E, 2*B.I, B.H), dtype=torch.float32, generator=g)*0.02).to("npu").to(torch.bfloat16)
B.w2_tl = (torch.randn((B.E, B.H, B.I), dtype=torch.float32, generator=g)*0.02).to("npu").to(torch.bfloat16)
B.w13_ac = B.w13_tl.transpose(1,2).contiguous()
B.w2_ac = B.w2_tl.transpose(1,2).contiguous()

name, tokens, skewed, which = sys.argv[1], int(sys.argv[2]), sys.argv[3]=="1", sys.argv[4]
x, topk_ids = build_case(tokens, seed=7, skewed=skewed, dev="npu")
base = B.prepare_baseline(x, topk_ids, "npu")
ours = B.prepare_ours(x, topk_ids, "npu")
fn = base if which=="base" else ours
for _ in range(10):
    fn()
torch.npu.synchronize()
import tempfile, os as _os
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.NPU]) as prof:
    for _ in range(20):
        fn()
    torch.npu.synchronize()
tp = _os.path.join(tempfile.gettempdir(), f"moedt_{sys.argv[1]}_{sys.argv[4]}.json")
prof.export_chrome_trace(tp)
print("PROF_DONE", tp)
"""

results = {}
scenarios = [
    ("decode_b32", 32, "0"), ("prefill_400", 400, "0"),
]
DECODER = json.JSONDecoder()

def load_trace(path):
    """Tolerate trailing garbage after the first JSON document."""
    with open(path) as f:
        return DECODER.raw_decode(f.read())[0]

for name, tokens, skewed in scenarios:
    for which in ("base", "ti"):
        code = SCRIPT
        out_dir = f"/tmp/opencode/moeprof_{name}_{which}"
        os.makedirs(out_dir, exist_ok=True)
        for stale in glob.glob(f"/tmp/moedt_{name}_{which}.json*"):
            os.remove(stale)
        r = subprocess.run(
            ["/home/userlist/zxs/TileInfer/.venv/bin/python", "-c", code, name, str(tokens), skewed, which],
            capture_output=True, text=True, cwd=out_dir,
        )
        # parse the chrome trace: sum device-side kernel durations per name
        tp = f"/tmp/moedt_{name}_{which}.json"
        per_step = {}
        if os.path.exists(tp):
            tr = load_trace(tp)
            events = tr if isinstance(tr, list) else tr.get("traceEvents", [])
            agg = {}
            for e in events:
                if e.get("ph") == "X":
                    n = e.get("name", "")
                    if any(k in n for k in ("main_kernel", "GroupedMatmul", "SwiGlu", "Swiglu")):
                        a = agg.setdefault(n[:44], [0, 0.0])
                        a[0] += 1
                        a[1] += e.get("dur", 0)
            for n, (c, t) in agg.items():
                per_step[n] = round(t / 20, 1)  # us per step
        results[f"{name}:{which}"] = per_step
        print(name, which, per_step, flush=True)

with open(".agent_loop/run/e19_device_time.json", "w") as f:
    json.dump(results, f, indent=2)
print("saved .agent_loop/run/e19_device_time.json")
