"""P8 kernel-level microbench: full split plans vs the plain kernel, serving config.

Qwen3.6-35B-A3B full-attention layers: kv_heads=8, group=4 (32 qo heads), head_dim=256.
Measures (all through the real forward_split + merge chain, plan-owned buffers):

* long-context decode (b4 x 64/128/256 pages) plain vs split at kv_tile_pages 16/32;
* empty-tile + merge overhead of a *full* plan serving a short batch (b8 x 1 page) and a
  mixed batch — the cost short-context traffic pays when split-KV is globally enabled.

Run: source env.sh && python benchmarks/probes/p8_split_fullplan_bench.py
"""

import sys, time
from pathlib import Path

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tileinfer import BatchAttention
from tileinfer.metadata import RaggedMetadata

dev = torch.device("npu", 0)
DT = torch.bfloat16
KVH, GROUP, PAGE, DIM = 8, 4, 128, 256
TOL = 3e-2


def meta_for(lens, pool_pages, seed=0):
    page = PAGE
    indptr = [0]
    lpl = []
    for L in lens:
        indptr.append(indptr[-1] + -(-L // page))
        lpl.append(page if L % page == 0 else L % page)
    n = indptr[-1]
    g = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(pool_pages, generator=g).to(dev)[: max(n, 1)].to(torch.int32)
    idx = perm.repeat(-(-n // max(perm.numel(), 1)))[:n].contiguous()
    return RaggedMetadata(
        kv_indptr=torch.tensor(indptr, dtype=torch.int32, device=dev),
        kv_indices=idx,
        kv_last_page_len=torch.tensor(lpl, dtype=torch.int32, device=dev),
        page_size=page,
    )


def run_case(label, batch, lens, width_pages, ktp, iters=30):
    torch.manual_seed(0)
    pool_pages = batch * width_pages + 1
    k_cache = torch.randn(pool_pages, PAGE, KVH, DIM, dtype=DT, device=dev)
    v_cache = torch.randn_like(k_cache)
    real = meta_for(lens, pool_pages)
    full = meta_for([width_pages * PAGE] * batch, pool_pages, seed=1)
    q = torch.randn(batch, KVH * GROUP, 1, DIM, dtype=DT, device=dev) * 0.2

    attn = BatchAttention(backend="tilelang-ascend950", device=dev, dtype=DT)
    plan = attn.plan(
        kv_indptr=full.kv_indptr,
        kv_indices=full.kv_indices,
        kv_last_page_len=full.kv_last_page_len,
        page_size=PAGE,
        num_qo_heads=KVH * GROUP,
        num_kv_heads=KVH,
        head_dim=DIM,
        causal=False,
        kv_tile_pages=ktp,
        k_cache=k_cache,
        v_cache=v_cache,
    )
    for _ in range(3):
        attn.run(q, k_cache, v_cache, plan=plan, kv_indptr=real.kv_indptr,
                 kv_indices=real.kv_indices, kv_last_page_len=real.kv_last_page_len)
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        attn.run(q, k_cache, v_cache, plan=plan, kv_indptr=real.kv_indptr,
                 kv_indices=real.kv_indices, kv_last_page_len=real.kv_last_page_len)
    torch.npu.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1e3
    # correctness vs the plain plan on the same real lens
    if ktp:
        plain = attn.plan(
            kv_indptr=real.kv_indptr, kv_indices=real.kv_indices,
            kv_last_page_len=real.kv_last_page_len, page_size=PAGE,
            num_qo_heads=KVH * GROUP, num_kv_heads=KVH, head_dim=DIM,
            causal=False, kv_tile_pages=0, k_cache=k_cache, v_cache=v_cache,
        )
        ref = attn.run(q, k_cache, v_cache, plan=plain)
        torch.npu.synchronize()
        got = attn.run(q, k_cache, v_cache, plan=plan, kv_indptr=real.kv_indptr,
                       kv_indices=real.kv_indices, kv_last_page_len=real.kv_last_page_len)
        torch.npu.synchronize()
        err = (got.float() - ref.float()).abs().max().item()
    else:
        err = 0.0
    pages = [-(-L // PAGE) for L in lens]
    gb = sum(pages) * PAGE * KVH * DIM * 2 * 2 / 1e9
    print(f"{label:44s} ktp={ktp:3d} {ms:8.3f} ms  err={err:.4f}  bw={gb/ms*1e3:7.1f} GB/s",
          flush=True)
    return ms


if __name__ == "__main__":
    print("== long-context decode (b4, serving config kvh8 g4 d256) ==")
    for pages in (64, 128, 256):
        lens = [pages * PAGE] * 4
        run_case(f"b4 x {pages*PAGE} tok plain", 4, lens, pages, 0)
        run_case(f"b4 x {pages*PAGE} tok split16", 4, lens, pages, 16)
        run_case(f"b4 x {pages*PAGE} tok split32", 4, lens, pages, 32)
    print("== short-context cost of a *full* plan (empty tiles + merge) ==")
    run_case("b8 x 1 tok plain", 8, [1] * 8, 256, 0)
    run_case("b8 x 1 tok full-plan split16", 8, [1] * 8, 256, 16)
    run_case("b8 x 1 tok full-plan split32", 8, [1] * 8, 256, 32)
    run_case("b4 x 1 tok plain", 4, [1] * 4, 256, 0)
    run_case("b4 x 1 tok full-plan split16", 4, [1] * 4, 256, 16)
    print("== mixed batch through a full plan ==")
    run_case("b4 mixed 32k/1 tok plain", 4, [256 * PAGE, 256 * PAGE, 1, 1], 256, 0)
    run_case("b4 mixed 32k/1 tok split16", 4, [256 * PAGE, 256 * PAGE, 1, 1], 256, 16)
    print("BENCH DONE")
