"""H23: DeepGEMM-Ascend m_grouped_bf16_gemm_nt_contiguous vs aclnn MoE chain.

对比口径与 benchmarks/bench_moe_grouped.py 一致(固定分组布局契约,路由/对齐准备
计入准备阶段不计入计时;基准侧 = vllm-ascend unquant_apply_mlp 原链)。

deep_gemm 侧布局 = psum 连续布局(csrc/apis/gemm.hpp):
  组 g 真实行占 [align(psum[g-1], 256), psum[g]),grouped_layout = 前缀和 int32[G],
  BLOCK_M 恒 256(jit.hpp kMkAlignment)。

运行环境(须先 source .agent_loop/run/env.sh):
  PATH=/home/userlist/zxs/TileInfer/.work/gcc13/bin:$PATH
  LD_LIBRARY_PATH=/home/userlist/zxs/TileInfer/.work/gcc13/lib64:$LD_LIBRARY_PATH
  DG_JIT_CACHE_DIR=/home/userlist/zxs/TileInfer/.work/dg_jit_cache

用法:
  .venv/bin/python benchmarks/probes/h23_deepgemm_grouped_bench.py \
      --out .agent_loop/run/e69_h23_deepgemm_bench.json
输出末行: SUM h23 verdict=... (decode_b32 与 prefill_400 合计比值 ≥1.0 为达标)
"""

from __future__ import annotations

import argparse
import json
import time

import torch
import torch_npu

import deep_gemm

import sys
sys.path.insert(0, "/home/userlist/zxs/TileInfer/benchmarks")
import bench_moe_grouped as B

E = B.E          # 256
H = B.H          # 2048
I = B.I          # 512
TOPK = B.TOPK    # 8
ALIGN = 256      # DeepGEMM-Ascend psum 布局硬对齐(jit.hpp kMkAlignment)


def aligned(x: int) -> int:
    return (x + ALIGN - 1) // ALIGN * ALIGN


def prepare_deepgemm(x, topk_ids, dev):
    """psum 连续布局准备(固定分组契约:与基准侧 compact+argsort 同为一次性准备)。"""
    flat = topk_ids.reshape(-1).to(torch.int64)
    counts = torch.bincount(flat, minlength=E)
    order = torch.argsort(flat, stable=True)
    expanded = x.unsqueeze(1).expand(-1, TOPK, -1).reshape(-1, H)
    x_sorted = expanded.index_select(0, order).contiguous()  # [tokens*TOPK, H]

    counts_list = counts.tolist()
    starts, psum_list, ptr = [], [], 0
    for c in counts_list:
        starts.append(ptr)
        ptr += c
        psum_list.append(ptr)
        ptr = aligned(ptr)
    m_pad = ptr
    psum = torch.tensor(psum_list, dtype=torch.int32, device=dev)

    a13 = torch.zeros((m_pad, H), dtype=torch.bfloat16, device=dev)
    cs = torch.cumsum(counts, 0).tolist()  # x_sorted 内组 e 的紧凑切片
    src_rows, dst_rows = [], []
    for e in range(E):
        if counts_list[e]:
            src_rows.append(torch.arange(cs[e] - counts_list[e], cs[e], device=dev))
            dst_rows.append(torch.arange(starts[e], psum_list[e], device=dev))
    src = torch.cat(src_rows) if src_rows else torch.empty(0, dtype=torch.int64, device=dev)
    valid = torch.cat(dst_rows) if dst_rows else torch.empty(0, dtype=torch.int64, device=dev)
    a13[valid] = x_sorted[src]

    d13 = torch.empty((m_pad, 2 * I), dtype=torch.bfloat16, device=dev)
    d2 = torch.empty((m_pad, H), dtype=torch.bfloat16, device=dev)

    def run():
        deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a13, B.w13_tl, d13, psum)
        g = torch_npu.npu_swiglu(d13)
        deep_gemm.m_grouped_bf16_gemm_nt_contiguous(g, B.w2_tl, d2, psum)
        return d2

    return run, (a13, psum, d13, d2, valid, x_sorted)


def reference_out(x, topk_ids, dev):
    """fp32 参照(逐 slot),返回 [tokens*TOPK, H] 按展开序。

    按专家分组循环计算,避免物化 w13[flat](prefill_400 时 ~27GB)。
    """
    flat = topk_ids.reshape(-1).to(torch.int64)
    n = flat.numel()
    expanded = x.unsqueeze(1).expand(-1, TOPK, -1).reshape(-1, H).float()
    w13 = B.w13_tl.float()
    w2 = B.w2_tl.float()
    y1 = torch.empty((n, 2 * I), dtype=torch.float32, device=dev)
    out = torch.empty((n, H), dtype=torch.float32, device=dev)
    for e in range(E):
        idx = (flat == e).nonzero(as_tuple=True)[0]
        if idx.numel():
            y1[idx] = expanded[idx] @ w13[e].T
    act = torch.nn.functional.silu(y1[:, :I]) * y1[:, I:]
    for e in range(E):
        idx = (flat == e).nonzero(as_tuple=True)[0]
        if idx.numel():
            out[idx] = act[idx] @ w2[e].T
    return out


def numeric_check(x, topk_ids, prepared, dev):
    run, (a13, psum, d13, d2, valid, x_sorted) = prepared
    out = run()
    torch.npu.synchronize()
    ref = reference_out(x, topk_ids, dev)
    flat = topk_ids.reshape(-1).to(torch.int64)
    order = torch.argsort(flat, stable=True)
    got = out.index_select(0, valid).float()  # valid 升序 = (专家,组内) 序 = x_sorted 真实行序
    r = ref.index_select(0, order).float()
    diff = (got - r).abs().max().item()
    scale = r.abs().max().item()
    return diff, scale


def bench(fn, warmup=20, iters=100):
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
    ap.add_argument("--out", default=".agent_loop/run/e69_h23_deepgemm_bench.json")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--skip-check", action="store_true")
    args = ap.parse_args()

    dev = "npu"
    g = torch.Generator(device="cpu").manual_seed(42)
    B.w13_tl = (torch.randn((E, 2 * I, H), dtype=torch.float32, generator=g) * 0.02).to(dev).to(torch.bfloat16)
    B.w2_tl = (torch.randn((E, H, I), dtype=torch.float32, generator=g) * 0.02).to(dev).to(torch.bfloat16)
    B.w13_ac = B.w13_tl.transpose(1, 2).contiguous()
    B.w2_ac = B.w2_tl.transpose(1, 2).contiguous()

    scenarios = [
        ("decode_b1", 1, False), ("decode_b4", 4, False), ("decode_b16", 16, False),
        ("decode_b32", 32, False), ("prefill_128", 128, False),
        ("prefill_400", 400, False), ("decode_b32_skew", 32, True),
    ]
    results = {}
    check_first = None
    for name, tokens, skewed in scenarios:
        x, topk_ids = B.build_case(tokens, seed=hash(name) % (2**31), skewed=skewed, dev=dev)
        base = B.prepare_baseline(x, topk_ids, dev)
        prepared = prepare_deepgemm(x, topk_ids, dev)
        dg = prepared[0]
        if check_first is None and not args.skip_check:
            diff, scale = numeric_check(x, topk_ids, prepared, dev)
            check_first = (name, diff, scale)
            print(f"[numeric] {name}: maxdiff={diff:.4g} (ref absmax={scale:.4g})")
            if diff > 3e-2 * max(scale, 1e-6):
                print(f"[numeric] FAIL — 数值超 TOL,跳过性能计")
                results[name] = {"tokens": tokens, "error": "numeric"}
                continue
        tb = bench(base, iters=args.iters)
        td = bench(dg, iters=args.iters)
        results[name] = {
            "tokens": tokens,
            "aclnn_ms": round(tb, 4),
            "deepgemm_ms": round(td, 4),
            "ratio_aclnn_over_dg": round(tb / td, 3),
        }
        print(f"{name:16s} tokens={tokens:4d} aclnn={tb:8.4f}ms deepgemm={td:8.4f}ms "
              f"ratio={tb / td:6.3f}x")

    gate_names = ("decode_b32", "prefill_400")
    ratios = [results[n]["ratio_aclnn_over_dg"] for n in gate_names if n in results and "ratio_aclnn_over_dg" in results[n]]
    verdict = "PASS" if len(ratios) == 2 and min(ratios) >= 1.0 else "FAIL"
    summary = {"results": results, "gate": dict(zip(gate_names, ratios)), "verdict": verdict}
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSUM h23 verdict={verdict} gate={summary['gate']} -> {args.out}")


if __name__ == "__main__":
    main()
