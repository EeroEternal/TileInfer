"""H28 微基准:折叠 grouped(E+1)vs 分解路径(grouped E + shared 串行链)。

门槛(H28 线):decode 形状(b1/b8/b32)折叠 ≥1.0x 分解路径(墙钟流水口径,同
bench_moe_grouped)。分解臂=在役 unquant 链原语(npu_grouped_matmul×2+swiglu)
+ shared gate_up GEMM + swiglu + down GEMM + gate GEMV + sigmoid 标乘;
折叠臂=同链 E+1(权重含 shared 行),topk+1 列。

用法::
    .venv/bin/python benchmarks/probes/h28_fold_bench.py \
        --out .agent_loop/run/h28_fold_bench.json
"""

from __future__ import annotations

import argparse
import json
import time

import torch
import torch_npu

E = 256
H = 2048
I = 512
TOPK = 8


def _chain(x_compact, counts, w13_ac, w2_ac):
    g = torch_npu.npu_grouped_matmul(
        x=[x_compact], weight=[w13_ac], split_item=2, group_list_type=1,
        group_type=0, group_list=counts,
    )[0]
    a = torch_npu.npu_swiglu(g)
    return torch_npu.npu_grouped_matmul(
        x=[a], weight=[w2_ac], split_item=2, group_list_type=1,
        group_type=0, group_list=counts,
    )[0]


def bench(fn, warmup=20, iters=200):
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
    ap.add_argument("--out", default=".agent_loop/run/h28_fold_bench.json")
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()
    dev = "npu"

    g = torch.Generator(device="cpu").manual_seed(42)
    w13 = (torch.randn((E, 2 * I, H), generator=g) * 0.02).to(dev).to(torch.bfloat16)
    w2 = (torch.randn((E, H, I), generator=g) * 0.02).to(dev).to(torch.bfloat16)
    sh_w13 = (torch.randn((2 * I, H), generator=g) * 0.02).to(dev).to(torch.bfloat16)
    sh_w2 = (torch.randn((H, I), generator=g) * 0.02).to(dev).to(torch.bfloat16)
    gate_w = (torch.randn((H,), generator=g) * 0.05).to(dev).to(torch.float32)
    w13_ac = w13.transpose(1, 2).contiguous()
    w2_ac = w2.transpose(1, 2).contiguous()
    w13_f = torch.cat([w13, sh_w13.unsqueeze(0)], dim=0).transpose(1, 2).contiguous()
    w2_f = torch.cat([w2, sh_w2.unsqueeze(0)], dim=0).transpose(1, 2).contiguous()

    results = {}
    speedups = []
    for tokens in (1, 8, 32, 128):
        x = torch.randn((tokens, H), generator=g).to(dev).to(torch.bfloat16)
        topk_ids = torch.randint(0, E, (tokens, TOPK), generator=g).to(dev)
        topk_w = torch.rand((tokens, TOPK), generator=g).to(dev).to(torch.float32)

        # 分解臂 prep(固定分组布局契约,两侧同构,prep 不计时)
        flat = topk_ids.reshape(-1).to(torch.int64)
        counts = torch.bincount(flat, minlength=E).to(torch.int64)
        order = torch.argsort(flat, stable=True)
        x_routed = (
            x.unsqueeze(1).expand(-1, TOPK, -1).reshape(-1, H)
            .index_select(0, order).contiguous()
        )

        w_sorted = topk_w.reshape(-1).index_select(0, order)
        agg_idx = order // TOPK

        def decomp():
            out = _chain(x_routed, counts, w13_ac, w2_ac)
            agg = torch.zeros((tokens, H), device=dev, dtype=torch.float32)
            agg.index_add_(0, agg_idx, out.to(torch.float32).mul_(w_sorted.unsqueeze(1)))
            gl = torch.nn.functional.linear(x.float(), gate_w.unsqueeze(0))
            shared = torch.nn.functional.linear(
                torch_npu.npu_swiglu(torch.nn.functional.linear(x, sh_w13)), sh_w2
            )
            return agg + shared.float() * torch.sigmoid(gl)

        # 折叠臂 prep
        gate_logit = torch.nn.functional.linear(x.float(), gate_w.unsqueeze(0))
        ids_f, w_f = torch.cat(
            [topk_ids, torch.full((tokens, 1), E, dtype=topk_ids.dtype, device=dev)], dim=1
        ), torch.cat([topk_w, torch.sigmoid(gate_logit)], dim=1).contiguous()
        flat_f = ids_f.reshape(-1).to(torch.int64)
        counts_f = torch.bincount(flat_f, minlength=E + 1).to(torch.int64)
        order_f = torch.argsort(flat_f, stable=True)
        x_f = (
            x.unsqueeze(1).expand(-1, TOPK + 1, -1).reshape(-1, H)
            .index_select(0, order_f).contiguous()
        )

        wf_sorted = w_f.reshape(-1).index_select(0, order_f)
        aggf_idx = order_f // (TOPK + 1)

        def folded():
            out = _chain(x_f, counts_f, w13_f, w2_f)
            agg = torch.zeros((tokens, H), device=dev, dtype=torch.float32)
            agg.index_add_(0, aggf_idx, out.to(torch.float32).mul_(wf_sorted.unsqueeze(1)))
            return agg

        td = bench(decomp, iters=args.iters)
        tf = bench(folded, iters=args.iters)
        sp = td / tf
        speedups.append(sp)
        results[f"b{tokens}"] = {
            "tokens": tokens, "decomposed_ms": round(td, 4),
            "folded_ms": round(tf, 4), "speedup": round(sp, 3),
        }
        print(f"b{tokens:<4d} decomposed={td:8.4f}ms folded={tf:8.4f}ms speedup={sp:6.3f}x")

    geo = 1.0
    for s in speedups[:3]:
        geo *= s
    geo **= 1.0 / 3
    summary = {"results": results, "decode_geomean_b1_8_32": round(geo, 4)}
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDECODE GEOMEAN(b1/b8/b32)= {geo:.3f}x -> {args.out}")


if __name__ == "__main__":
    main()
