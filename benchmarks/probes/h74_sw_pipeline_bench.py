"""H74 微基准:编译器调度页流水(`T.Pipelined(num_stages=N)`)vs 串行页循环。

背景
----
deepseek/DeepInfer 的 Ascend 算子(`tilelang_ops/asc/`,26 个文件)**全部**把任务循环写成
`T.Persistent`/`T.Pipelined(..., num_stages=N)`,由编译器把 GM→L1 的拷贝排到上一轮计算
之下;我们的 `paged_decode_ascend950` 是 `for p in T.serial(num_pages)`,零流水。

H42-③ 曾用手写双缓冲(`paged_decode_pipeline`)尝试同一件事,2026-10-20 设备微基准判
disproven(8K: 0.598 → 0.652 ms)。本探针检验的是**另一种实现**:把编排交给编译器。

判据(预登记)
------------
A) 数值:各 stage 配置与串行版**逐位等价**(err ≤ 1e-2,同 H32 设备 TOL 口径);
   不达标即否决,不看性能。
B) 性能:b8 × 8K/16K,stage ∈ {2,3,4} 应至少有一档 ≥ 串行版;否则归档为负面结果。
C) 短形状零税(E61 纪律):b8 × 1 tok,|Δ| ≤ 5%。
D) 与 H42-③ 手写版对照:若编译器版也不赚,说明"页循环流水"这条路整体不成立,
   而不是实现方式问题。

运行
----
    source .agent_loop/run/env.sh
    python benchmarks/probes/h74_sw_pipeline_bench.py

输出行格式:
    [H74] <label> stages=<n> <ms> ms err=<e> bw=<gbps> GB/s speedup=<x>
最后一行 `[H74] BENCH DONE`(门槛判定由分析方做)。
"""

import os
import sys
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import tileinfer  # noqa: F401  (注册设备;只导入子模块会被 tilelang 当 CPU target)
from tileinfer.kernels.attention.paged_decode_ascend950 import (
    build_decode_kernel,
    declared_pool_cap,
)

torch.npu.set_device(torch.device("npu", 0))

DEV = torch.device("npu", 0)
DT = torch.bfloat16
KVH, GROUP, PAGE, DIM = 8, 4, 128, 256
WIDTH_PAGES = 256
ITERS = 30
STAGES = (2, 3, 4)


def make_meta(lens, pool_pages, seed=0):
    """构造 ragged 页表。

    注意:`kv_indices` 的 ABI 长度是**声明容量** `declared_pool_cap(...)`,
    不是真实池大小;内核只按 `kv_indptr` 派生的范围访问,超出真实池的槽位永不解引用。
    """
    indptr = [0]
    lpl = []
    for L in lens:
        indptr.append(indptr[-1] + -(-L // PAGE))
        lpl.append(PAGE if L % PAGE == 0 else L % PAGE)
    n = indptr[-1]
    cap = declared_pool_cap(pool_pages)
    g = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(cap, generator=g).to(DEV)[: max(n, 1)].to(torch.int32)
    idx = perm.repeat(-(-n // max(perm.numel(), 1)))[:n].contiguous()
    if idx.numel() < cap:
        pad = torch.zeros(cap - idx.numel(), dtype=torch.int32, device=DEV)
        idx = torch.cat([idx, pad])
    return (
        torch.tensor(indptr, dtype=torch.int32, device=DEV),
        idx,
        torch.tensor(lpl, dtype=torch.int32, device=DEV),
    )


def run_case(label, lens, stages_list=STAGES):
    batch = len(lens)
    pool_pages = declared_pool_cap(batch * WIDTH_PAGES + 1)
    g = torch.Generator(device="cpu").manual_seed(7)
    k_cache = torch.randn(pool_pages, PAGE, KVH, DIM, dtype=torch.float32, device=DEV).to(DT)
    v_cache = torch.randn(pool_pages, PAGE, KVH, DIM, dtype=torch.float32, device=DEV).to(DT)
    indptr, indices, lpl = make_meta(lens, pool_pages)

    BR = max(32, ((GROUP + 31) // 32) * 32)
    q = (torch.randn(batch, KVH * GROUP, DIM, generator=g) * 0.2).to(torch.float32)
    q_pad = torch.zeros(batch, KVH * BR, DIM, dtype=torch.float32, device=DEV)
    q_pad.view(batch, KVH, BR, DIM)[:, :, :GROUP] = q.view(batch, KVH, GROUP, DIM).to(DEV)
    q_pad = q_pad.to(DT)
    out_pad = torch.zeros(batch, KVH * BR, DIM, dtype=DT, device=DEV)

    def kernel(spec_extra):
        return build_decode_kernel(
            batch=batch, kv_heads=KVH, group=GROUP, dim=DIM, page_size=PAGE,
            num_pages_cap=pool_pages, dtype="bfloat16", out_manual=True, **spec_extra,
        )

    def call(kern):
        kern(q_pad, k_cache, v_cache, indptr, indices, lpl, out_pad)
        return out_pad

    base = kernel({})
    ref = call(base).clone()
    torch.npu.synchronize()

    def bench(kern):
        for _ in range(3):
            call(kern)
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(ITERS):
            call(kern)
        torch.npu.synchronize()
        return (time.perf_counter() - t0) / ITERS * 1e3

    ms0 = bench(base)
    ms0b = bench(base)  # 二次测量:剔除首次调用的预热偏差
    ms0 = min(ms0, ms0b)
    pages = [-(-L // PAGE) for L in lens]
    gb = sum(pages) * PAGE * KVH * DIM * 2 * 2 / 1e9
    print(f"[H74] {label:28s} stages=0 {ms0:8.3f} ms err=0.0000 "
          f"bw={gb / ms0 * 1e3:7.1f} GB/s speedup=1.00x", flush=True)

    for st in stages_list:
        try:
            kern = kernel({"sw_pipeline": st})
            got = call(kern)
            torch.npu.synchronize()
            err = (got.float() - ref.float()).abs().max().item()
            ms = min(bench(kern), bench(kern))
            print(f"[H74] {label:28s} stages={st} {ms:8.3f} ms err={err:.4f} "
                  f"bw={gb / ms * 1e3:7.1f} GB/s speedup={ms0 / ms:.2f}x", flush=True)
        except Exception as exc:  # 编译/执行失败要显式记录,不能静默跳过
            print(f"[H74] {label:28s} stages={st} FAILED: {type(exc).__name__}: "
                  f"{str(exc)[:160]}", flush=True)


if __name__ == "__main__":
    print(f"[H74] start (simd={os.environ.get('TILEINFER_SIMD_SOFTMAX', '0')})", flush=True)
    print("[H74] == 长上下文:b8 × 8K / 16K ==", flush=True)
    for pages in (64, 128):
        run_case(f"b8 x {pages * PAGE}", [pages * PAGE] * 8)
    print("[H74] == 短形状零税复核(E61)==", flush=True)
    run_case("b8 x 1", [1] * 8)
    run_case("b8 x 128", [128] * 8)
    print("[H74] == c1 单请求(H67 靶面)==", flush=True)
    run_case("b1 x 8192", [8192], stages_list=STAGES)
    run_case("b1 x 2048", [2048], stages_list=STAGES)
    run_case("b1 x 128", [128], stages_list=STAGES)
    print("[H74] == c8 短上下文 ==", flush=True)
    run_case("b8 x 512", [512] * 8, stages_list=STAGES)
    print("[H74] BENCH DONE", flush=True)
