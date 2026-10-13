"""GDN 融合内核(gdn_chunk_wu)微基准:H10 门槛取证。

口径(E20c/E22 教训):npu Event 事件计时、预热后取中位、与在役完全同参;
对照 = 在役 triton solve_tril + recompute_w_u_fwd(两核合计)。

H10 门槛:融合核设备时间 ≤ 被替换两核合计 × 0.6(T=128 形状)。

用法: source .agent_loop/run/env.sh && setsid .venv/bin/python benchmarks/bench_gdn_wu.py
输出: .agent_loop/run/e23_gdn_wu_bench.json
"""

import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
import torch_npu  # noqa: E402

torch.npu.set_device(0)
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29634")
from vllm.config import VllmConfig, set_current_vllm_config  # noqa: E402
from vllm.distributed import init_distributed_environment, initialize_model_parallel  # noqa: E402

init_distributed_environment(world_size=1, rank=0, local_rank=0, distributed_init_method="env://")
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton  # noqa: E402

init_device_properties_triton()

results = {}

with set_current_vllm_config(VllmConfig()):
    initialize_model_parallel(tensor_model_parallel_size=1)
    from tileinfer.testing.gdn_reference import kkt_reference
    from vllm_ascend.ops.triton.fla.solve_tril import solve_tril
    from vllm_ascend.ops.triton.fla.wy_fast import recompute_w_u_fwd
    from vllm_ascend.ops.triton.fla.utils import prepare_chunk_indices
    from vllm_ascend.ops.triton.fla.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
    from tileinfer.kernels.gdn.chunk_wu_ascend950 import gdn_chunk_wu

    HK, HV, D = 16, 32, 128

    def bench_event(fn, n=30, warmup=15):
        for _ in range(warmup):
            fn()
        torch.npu.synchronize()
        ts = []
        s = torch.npu.Event(enable_timing=True)
        e = torch.npu.Event(enable_timing=True)
        for _ in range(n):
            s.record()
            fn()
            e.record()
            torch.npu.synchronize()
            ts.append(s.elapsed_time(e))
        return round(statistics.median(ts), 4)

    for t_total in (128, 8192):
        g = torch.Generator(device="cpu").manual_seed(0)
        k = torch.randn(1, t_total, HK, D, generator=g).to(torch.bfloat16)
        k = torch.nn.functional.normalize(k.float(), dim=-1).to(torch.bfloat16).npu()
        v = (torch.randn(1, t_total, HV, D, generator=g) * 0.2).to(torch.bfloat16).npu()
        beta = torch.rand(1, t_total, HV, generator=g).to(torch.bfloat16).npu()
        g_cum = (-torch.rand(1, t_total, HV, generator=g).cumsum(1) * 0.05).float().npu()
        cu = torch.tensor([0, t_total], dtype=torch.int64).npu()
        chunk_idx = prepare_chunk_indices(cu.cpu(), 64).npu()
        a_kkt = chunk_scaled_dot_kkt_fwd(k=k, beta=beta, g_cumsum=g_cum, cu_seqlens=cu, chunk_indices=None, output_dtype=torch.float32)
        a_tri = solve_tril(A=a_kkt, cu_seqlens=cu, chunk_indices_large_block=None, chunk_indices_bt=None, output_dtype=k.dtype)
        torch.npu.synchronize()

        def baseline():
            a_t = solve_tril(A=a_kkt, cu_seqlens=cu, chunk_indices_large_block=None, chunk_indices_bt=None, output_dtype=k.dtype)
            return recompute_w_u_fwd(k=k, v=v, beta=beta, A=a_t, g_cumsum=g_cum, cu_seqlens=cu, chunk_indices=None)

        def fused():
            return gdn_chunk_wu(a_kkt, k, v, beta, g_cum, cu, chunk_idx)

        base_ms = bench_event(baseline)
        fused_ms = bench_event(fused)
        w_b, u_b = baseline()
        w_f, u_f = fused()
        dw = (w_f.float() - w_b.float()).abs().max().item()
        du = (u_f.float() - u_b.float()).abs().max().item()
        results[f"T{t_total}"] = {
            "baseline_solve_tril_plus_wu_ms_p50": base_ms,
            "fused_ms_p50": fused_ms,
            "ratio_fused_over_baseline": round(fused_ms / base_ms, 4),
            "gate_0_6_pass": fused_ms <= 0.6 * base_ms,
            "max_abs_diff_vs_baseline_w": round(dw, 6),
            "max_abs_diff_vs_baseline_u": round(du, 6),
            "per_layer_30_ms": round(fused_ms * 30, 3),
        }
        torch.npu.synchronize()
        print(json.dumps({f"T{t_total}": results[f"T{t_total}"]}, indent=1), flush=True)

with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".agent_loop/run/e23_gdn_wu_bench.json"), "w") as f:
    json.dump(results, f, indent=1)
print("BENCH DONE")
