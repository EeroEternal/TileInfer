"""Microbenchmark: fused TileLang MoE kernels vs the vllm-ascend aclnn chain.

Baseline (what vllm-ascend actually runs for bf16 unquantised MoE, see
``vllm_ascend/ops/fused_moe/moe_mlp.py::unquant_apply_mlp``)::

    npu_grouped_matmul(x, w13.T, split_item=2) -> npu_swiglu -> npu_grouped_matmul(., w2.T)

Ours::

    TileLang gate+up+SwiGLU fused kernel -> TileLang down kernel

Both sides start from their natural input layout (baseline: compact rows grouped
by expert + per-expert counts; ours: padded slot layout), i.e. routing/align
preparation is excluded — the "fixed grouping layout" contract.  Weights are
pre-transposed once for the baseline (a layout choice, not per-step work).

Real Qwen3.6-35B-A3B shapes: E=256, hidden=2048, inter=512, top_k=8, bf16.

Usage::

    .venv/bin/python benchmarks/bench_moe_grouped.py --out .agent_loop/run/e19_moe_bench.json
"""

from __future__ import annotations

import argparse
import json
import time

import torch
import torch_npu

from tileinfer.kernels.moe import build_down_kernel, build_gate_up_kernel
from tileinfer.testing.moe_reference import build_slot_layout, gather_slots

E = 256
H = 2048
I = 512
TOPK = 8
BM = 32


def build_case(tokens: int, seed: int, skewed: bool, dev: str):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn((tokens, H), dtype=torch.float32, generator=g).to(dev).to(torch.bfloat16)
    if skewed:
        flat = torch.zeros((tokens * TOPK,), dtype=torch.int64)
        flat[: tokens * TOPK // 2] = 0  # half of all slots on expert 0
        flat[tokens * TOPK // 2 :] = torch.randint(1, E, (tokens * TOPK // 2,), generator=g)
        topk_ids = flat.view(tokens, TOPK).to(dev)
    else:
        topk_ids = torch.randint(0, E, (tokens, TOPK), generator=g).to(dev)
    return x, topk_ids


def prepare_ours(x, topk_ids, dev):
    topk = topk_ids.shape[1]
    expanded = x.unsqueeze(1).expand(-1, topk, -1).reshape(-1, H)
    lay = build_slot_layout(topk_ids, BM, num_experts=E)
    rows_slots = lay["num_tiles"] * BM
    x_slots = gather_slots(expanded, lay["sorted_ids"], lay["num_expanded"]).contiguous()
    # P4 final-round knobs (E36): larger W nd2nz ops + fully contiguous down
    # reads; best-of-sweep (1.065x vs BN/BK=128 defaults, still 0.81x vs aclnn).
    k1 = build_gate_up_kernel(
        num_tiles_max=lay["num_tiles"], rows_slots=rows_slots, num_experts=E,
        inter=I, hidden=H, block_m=BM, block_n=256,
    )
    k2 = build_down_kernel(
        num_tiles_max=lay["num_tiles"], rows_slots=rows_slots, num_experts=E,
        inter=I, hidden=H, block_m=BM, block_n=64, block_k=512,
    )
    y = torch.empty((rows_slots, I), dtype=torch.bfloat16, device=dev)
    out = torch.empty((rows_slots, H), dtype=torch.bfloat16, device=dev)
    ids = lay["expert_ids"].contiguous()
    sids = lay["sorted_ids"].contiguous()

    def run():
        k1(x_slots, w13_tl, ids, sids, y)
        k2(y, w2_tl, ids, sids, out)

    return run


def prepare_baseline(x, topk_ids, dev):
    """Compact layout + counts, then the exact unquant_apply_mlp chain."""
    flat = topk_ids.reshape(-1).to(torch.int64)
    counts = torch.bincount(flat, minlength=E)
    order = torch.argsort(flat, stable=True)
    expanded = x.unsqueeze(1).expand(-1, TOPK, -1).reshape(-1, H)
    x_compact = expanded.index_select(0, order).contiguous()
    counts_i64 = counts.to(torch.int64)

    def run():
        g = torch_npu.npu_grouped_matmul(
            x=[x_compact], weight=[w13_ac], split_item=2, group_list_type=1,
            group_type=0, group_list=counts_i64,
        )[0]
        a = torch_npu.npu_swiglu(g)
        return torch_npu.npu_grouped_matmul(
            x=[a], weight=[w2_ac], split_item=2, group_list_type=1,
            group_type=0, group_list=counts_i64,
        )[0]

    return run


def bench(fn, warmup=20, iters=100):
    """Pipelined device time: one event pair around `iters` back-to-back calls.

    Per-call sync (record/stop around a single launch) measures CPU enqueue
    latency on this stack (the queue drains before the next launch arrives),
    which swamps the device time by an order of magnitude.  Back-to-back
    launches measure what serving actually cares about: saturated-device time
    per step (or, if the CPU can't keep up, the honest CPU ceiling — for both
    sides equally).
    """
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    stop = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    stop.record()
    torch.npu.synchronize()
    return start.elapsed_time(stop) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".agent_loop/run/e19_moe_bench.json")
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()

    dev = "npu"
    global w13_tl, w2_tl, w13_ac, w2_ac
    g = torch.Generator(device="cpu").manual_seed(42)
    w13_tl = (torch.randn((E, 2 * I, H), dtype=torch.float32, generator=g) * 0.02).to(dev).to(torch.bfloat16)
    w2_tl = (torch.randn((E, H, I), dtype=torch.float32, generator=g) * 0.02).to(dev).to(torch.bfloat16)
    w13_ac = w13_tl.transpose(1, 2).contiguous()  # [E, H, 2I] aclnn layout
    w2_ac = w2_tl.transpose(1, 2).contiguous()    # [E, I, H]

    scenarios = [
        ("decode_b1", 1, False),
        ("decode_b4", 4, False),
        ("decode_b16", 16, False),
        ("decode_b32", 32, False),
        ("prefill_128", 128, False),
        ("prefill_400", 400, False),
        ("decode_b32_skew", 32, True),
    ]
    results = {}
    for name, tokens, skewed in scenarios:
        x, topk_ids = build_case(tokens, seed=hash(name) % (2**31), skewed=skewed, dev=dev)
        base = prepare_baseline(x, topk_ids, dev)
        ours = prepare_ours(x, topk_ids, dev)
        t0 = time.time()
        tb = bench(base, iters=args.iters)
        to = bench(ours, iters=args.iters)
        results[name] = {
            "tokens": tokens,
            "baseline_ms": round(tb, 4),
            "tilelang_ms": round(to, 4),
            "speedup": round(tb / to, 3),
            "expand_rows": int(tokens * TOPK),
        }
        print(f"{name:16s} tokens={tokens:4d} baseline={tb:8.4f}ms tilelang={to:8.4f}ms "
              f"speedup={tb / to:6.3f}x  (prep {time.time() - t0:.0f}s)")

    geo = 1.0
    for v in results.values():
        geo *= v["speedup"]
    geo **= 1.0 / len(results)
    summary = {"results": results, "geomean_speedup": round(geo, 4)}
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nGEOMEAN speedup = {geo:.3f}x  -> {args.out}")


if __name__ == "__main__":
    main()
