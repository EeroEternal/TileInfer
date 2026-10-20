"""H42-③ 微基准:paged decode 双缓冲页流水内核(pipeline)vs 现役 plain 内核。

背景(E116/E118):b8 长上下文 decode 每请求 1 split,页循环 K/V GM→L1 拷贝与
Cube/Vector 计算全串行(~30µs/页,带宽下界 2.5µs/页,12x 余量)。双缓冲改造把 p+1 页
K/V 拷贝提前到 p 页计算前发起(写入备用 L1 对),数学逐位等价,仅时序重排。

本探针在**单进程内**直接 compile 两个内核(build_decode_kernel ± pipeline spec,
缓存键天然区分),同一输入对照:
  A) 数值 oracle:plain vs pipeline 输出最大误差(判据 ≤1e-2,同 H32 设备 TOL 口径);
  B) 长形状计时:8K/16K ctx b8,判据 pipeline ≥1.5x plain(W1-D57-② 门槛形态);
  C) 短形状零税:128/512 ctx b8,判据 |Δ| ≤5%(E61 短 ctx 税纪律,内核级);
  D) 有效带宽对照:E118 950 paged KV 2006 GB/s 公开上界。

运行(设备窗内,source .agent_loop/run/env.sh 后):
  python benchmarks/probes/h42_pipeline_bench.py
输出行格式:[H42PP] <label> plain=<ms> pipe=<ms> speedup=<x> err=<e> bw=<gbps> GB/s
最后一行:[H42PP] BENCH DONE (门槛判定由分析方做,本脚本只产出数字)
"""

import math
import sys
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# 必须导入包根:它会经由 `tileinfer.utils.resolve_device` 完成 torch_npu 后端注册与
# 当前设备绑定。只导入 `tileinfer.kernels...` 子模块时,tilelang 编译出的内核会被
# 当成 CPU target,调用即报 "allocator anchor has the wrong device type; expected: 1, got: 12"。
import tileinfer  # noqa: F401
from tileinfer.kernels.attention.paged_decode_ascend950 import (
    build_decode_kernel,
    declared_pool_cap,
)

DEV = torch.device("npu", 0)
DT = torch.bfloat16
KVH, GROUP, PAGE, DIM = 8, 4, 128, 256
WIDTH_PAGES = 256
ITERS = 30

# tilelang 的 tvm_ffi 适配器会读 `torch.npu.current_device()` 当 allocator anchor 的
# device type;不显式设置时它会退回 CPU(device type 12),于是内核调用直接报
#   "allocator anchor has the wrong device type; expected: 1, got: 12"
torch.npu.set_device(DEV)


def make_meta(lens, pool_pages, seed=0):
    indptr = [0]
    lpl = []
    for L in lens:
        indptr.append(indptr[-1] + -(-L // PAGE))
        lpl.append(PAGE if L % PAGE == 0 else L % PAGE)
    n = indptr[-1]
    g = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(pool_pages, generator=g).to(DEV)[: max(n, 1)].to(torch.int32)
    idx = perm.repeat(-(-n // max(perm.numel(), 1)))[:n].contiguous()
    return (
        torch.tensor(indptr, dtype=torch.int32, device=DEV),
        idx,
        torch.tensor(lpl, dtype=torch.int32, device=DEV),
    )


def run_case(label, lens):
    batch = len(lens)
    ctx = lens[0]
    pool_pages = declared_pool_cap(batch * WIDTH_PAGES + 1)
    g = torch.Generator(device="cpu").manual_seed(7)
    k_cache = torch.randn(pool_pages, PAGE, KVH, DIM, dtype=torch.float32, device=DEV).to(DT)
    v_cache = torch.randn(pool_pages, PAGE, KVH, DIM, dtype=torch.float32, device=DEV).to(DT)
    indptr, indices, lpl = make_meta(lens, pool_pages)
    # padded Q ABI: [batch, kv_heads * BR, dim], first `group` rows of each KV head valid (ki1 模板)
    BR = max(32, ((GROUP + 31) // 32) * 32)
    q = (torch.randn(batch, KVH * GROUP, DIM, generator=g) * 0.2).to(torch.float32)
    # 注意:必须在 NPU 上直接分配。曾写成 `torch.zeros(..., dtype=float32)`(CPU)再 `.to(DEV)`,
    # tilelang 的 allocator anchor 会拿 CPU device type(12)去比 NPU(1)而拒绝执行:
    #   RuntimeError: main: allocator anchor has the wrong device type; expected: 1, got: 12
    q_pad = torch.zeros(batch, KVH * BR, DIM, dtype=torch.float32, device=DEV)
    q_pad.view(batch, KVH, BR, DIM)[:, :, :GROUP] = q.view(batch, KVH, GROUP, DIM).to(DEV)
    q_pad = q_pad.to(DT)

    plain = build_decode_kernel(
        batch=batch, kv_heads=KVH, group=GROUP, dim=DIM, page_size=PAGE,
        num_pages_cap=pool_pages, dtype="bfloat16", out_manual=False,
    )
    pipe = build_decode_kernel(
        batch=batch, kv_heads=KVH, group=GROUP, dim=DIM, page_size=PAGE,
        num_pages_cap=pool_pages, dtype="bfloat16", out_manual=False, pipeline=True,
    )

    out_p = plain(q, k_cache, v_cache, indptr, indices, lpl)
    out_d = pipe(q, k_cache, v_cache, indptr, indices, lpl)
    err = (out_p.float() - out_d.float()).abs().max().item()

    def bench(kern, out):
        for _ in range(3):
            kern(q, k_cache, v_cache, indptr, indices, lpl)
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(ITERS):
            kern(q, k_cache, v_cache, indptr, indices, lpl)
        torch.npu.synchronize()
        return (time.perf_counter() - t0) / ITERS * 1e3

    ms_p = bench(plain, out_p)
    ms_d = bench(pipe, out_d)
    # 有效 KV 带宽(双缓冲侧):batch × ctx × 2(K+V) × KVH × DIM × 2B
    bw = batch * ctx * 2 * KVH * DIM * 2 / (ms_d / 1e3) / 1e9
    sp = ms_p / ms_d if ms_d > 0 else float("nan")
    print(f"[H42PP] {label} plain={ms_p:.3f}ms pipe={ms_d:.3f}ms speedup={sp:.3f}x "
          f"err={err:.4f} bw={bw:.0f}GB/s", flush=True)
    return sp, err


def main():
    cases = [
        ("ctx128_b8", [128] * 8),
        ("ctx512_b8", [512] * 8),
        ("ctx8k_b8", [8192] * 8),
        ("ctx16k_b8", [16384] * 8),
    ]
    sp_8k = err_8k = None
    for label, lens in cases:
        sp, err = run_case(label, lens)
        if label == "ctx8k_b8":
            sp_8k, err_8k = sp, err
    print(f"[H42PP] gate: 8k speedup={sp_8k:.3f}x (>=1.5) err={err_8k:.4f} (<=1e-2)", flush=True)
    print("[H42PP] BENCH DONE", flush=True)


if __name__ == "__main__":
    main()
