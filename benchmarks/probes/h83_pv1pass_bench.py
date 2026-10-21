"""H83-② PV 通路隔离微基准:pv_single_pass 变体 vs 现役 D2 双 pass。

判据(H83 预登记):PV 通路隔离口径 ≥1.05x(plen=64 页形状,即 8K prefill 的
页迭代深度);H80 联合门槛(≥1.5x 总账)在本基准上**不判定**——那是重写框架
全量落地后的权威微基准口径。

用法(设备窗):.venv/bin/python benchmarks/probes/h83_pv1pass_bench.py
输出:每形状 warmup+计时,打印 speedup 与 VERDICT 行。
"""

from __future__ import annotations

import time

import torch

from tileinfer.kernels.attention.paged_decode_ascend950 import forward_prefill
from tileinfer.metadata import RaggedMetadata
from tileinfer.plan import plan_query_tiles

DTYPE = torch.bfloat16


def build_case(kv_pages: int, qo_len: int, kv_heads: int = 2, group: int = 8, dim: int = 256,
               page_size: int = 128, seed: int = 0):
    device = torch.device("npu", 0)
    torch.manual_seed(seed)
    batch = 1
    kv_len = kv_pages * page_size
    total = kv_pages
    k_cache = torch.randn(total, page_size, kv_heads, dim, dtype=DTYPE, device=device)
    v_cache = torch.randn(total, page_size, kv_heads, dim, dtype=DTYPE, device=device)
    indices = torch.arange(total, dtype=torch.int32, device=device)
    q = 0.2 * torch.randn(batch, kv_heads * group, qo_len, dim, dtype=DTYPE, device=device)
    qo = torch.tensor([qo_len], dtype=torch.int32, device=device)
    qo_indptr = torch.tensor([0, qo_len], dtype=torch.int32, device=device)
    meta = RaggedMetadata(
        kv_indptr=torch.tensor([0, total], dtype=torch.int32, device=device),
        kv_indices=indices,
        kv_last_page_len=torch.tensor([page_size], dtype=torch.int32, device=device),
        page_size=page_size,
        qo_indptr=qo_indptr,
    )
    page_counts = (meta.kv_indptr[1:] - meta.kv_indptr[:-1]).to(torch.int64)
    sched = plan_query_tiles(
        qo.to(torch.int64) * group, page_counts, block_q=128, load_balance=False
    )
    return q, k_cache, v_cache, meta, sched, qo


def bench(case, pv_single_pass: bool, warmup: int = 5, iters: int = 20) -> float:
    q, k_cache, v_cache, meta, sched, qo = case

    def run():
        return forward_prefill(
            q, k_cache, v_cache, meta, sched, qo, group=8, pv_single_pass=pv_single_pass
        )

    for _ in range(warmup):
        run()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        run()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    # 形状:8K prefill 全量重扫(plen=64 页,BR=128 → 单 tile 串行 64 页迭代);
    # 16K prefill(plen=128 页);短 chunked(append 1K over 8K,plen 深但行少)。
    shapes = [
        ("8K prefill (plen=64)", 64, 8192),
        ("16K prefill (plen=128)", 128, 16384),
        ("append 1K over 8K", 64, 1024),
    ]
    print(f"{'shape':<24} {'D2 (ms)':>10} {'1pass (ms)':>11} {'speedup':>8}")
    verdict_ok = True
    for label, pages, qo in shapes:
        case = build_case(pages, qo)
        t_ref = bench(case, pv_single_pass=False)
        t_new = bench(case, pv_single_pass=True)
        sp = t_ref / t_new
        print(f"{label:<24} {t_ref * 1e3:>10.3f} {t_new * 1e3:>11.3f} {sp:>8.3f}x")
        if sp < 1.0:
            verdict_ok = False
    print("VERDICT:", "PV-PASS-FASTER" if verdict_ok else "PV-PASS-NOT-FASTER")
    print("(≥1.05x 隔离门槛判定见 H83;本基准不判 H80 总门槛)")


if __name__ == "__main__":
    main()
