"""W2 微基准:decode split-KV 并发税机理取证(E52 结论 5 的取证方向)。

背景:P8/E40 兑现了 split-KV 长上下文 decode 收益(1.41x/2.04x),但并发档有 11-19% 税
(conc_16 0.893x / conc_32 0.814x);当前集成层 ktp 是全局单值(TILEINFER_KV_TILE_PAGES,
所有 bucket 同值)。E52 候选机理:
  ① 满额 plan 空 tile 的 merge 零权重吸收(partial/LSE 写带仍在);
  ② 块开销 ~15µs/波 × 波数(随 ktp 增大);
  ③ merge 内核 launch 固定成本。
对照:FlashInfer decode.py(2026-10 主线)split 决策按 units=batch×kv_heads 分带——
≈1.7 波以下才 split;中等 batch 用 split-2/4;提供 fixed_split_size/disable_split_kv。

矩阵(serving 真实配置 kvh=8 g=4 d256 page=128;plan width=256 页镜像引擎桶 32):
  A 税现场(短上下文×大并发):b8/b16/b32 × 1tok,plain vs split16/32;
  B 长上下文×并发(split 在大 batch 是否仍赢):b32 × 8K/16K,plain vs split16/32;
  C width 敏感性(税是否随 plan 宽度线性→机理①):b32 1tok,width 16/64/256 @split16。
输出含 SUM 汇总行(vs_plain 比值),供 watcher/轮脚本自动解析。

运行:source .agent_loop/run/env.sh && python benchmarks/probes/w2_split_tax_bench.py
"""

import os, sys, time
from pathlib import Path

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("TILEINFER_ALLOW_SPLIT_KV", "1")  # 后端核心的 split opt-in 门禁(KI-1)

from tileinfer import BatchAttention
from tileinfer.metadata import RaggedMetadata

dev = torch.device("npu", 0)
DT = torch.bfloat16
KVH, GROUP, PAGE, DIM = 8, 4, 128, 256  # serving 真实配置(Qwen3.6 full-attn 层)
TOL = 3e-2


def meta_for(lens, pool_pages, seed=0):
    indptr = [0]
    lpl = []
    for L in lens:
        indptr.append(indptr[-1] + -(-L // PAGE))
        lpl.append(PAGE if L % PAGE == 0 else L % PAGE)
    n = indptr[-1]
    g = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(pool_pages, generator=g).to(dev)[: max(n, 1)].to(torch.int32)
    idx = perm.repeat(-(-n // max(perm.numel(), 1)))[:n].contiguous()
    return RaggedMetadata(
        kv_indptr=torch.tensor(indptr, dtype=torch.int32, device=dev),
        kv_indices=idx,
        kv_last_page_len=torch.tensor(lpl, dtype=torch.int32, device=dev),
        page_size=PAGE,
    )


def run_case(label, batch, real_tok, width_pages, ktp, iters=50, summary=None):
    """full plan(width_pages 满额)跑 real_tok 短真实上下文——镜像引擎满额 plan 语义。"""
    torch.manual_seed(0)
    pool_pages = batch * width_pages + 1
    k_cache = torch.randn(pool_pages, PAGE, KVH, DIM, dtype=DT, device=dev)
    v_cache = torch.randn_like(k_cache)
    real = meta_for([real_tok] * batch, pool_pages)
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
    # 数值 vs 同 real lens 的 plain plan
    err = 0.0
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
    tiles = batch * max(1, -(-width_pages // max(ktp, 1))) if ktp else batch
    ok = "OK" if err <= TOL else "FAIL"
    print(f"{label:40s} ktp={ktp:3d} {ms:8.3f} ms  err={err:.4f} {ok}  tiles={tiles}",
          flush=True)
    if summary is not None:
        summary[(width_pages, batch, ktp)] = ms
    return ms


def main():
    print("== A 并发税现场:短上下文 × 大 batch(满额 plan width=256) ==")
    sumA = {}
    for b in (8, 16, 32):
        for ktp in (0, 16, 32):
            iters = 100 if b < 32 else 50
            ms = run_case(f"b{b} x 1tok w256", b, 1, 256, ktp, iters=iters, summary=sumA)
    print("  -- A 汇总(税=split/plain) --")
    for b in (8, 16, 32):
        base = sumA.get((256, b, 0))
        for ktp in (16, 32):
            v = sumA.get((256, b, ktp))
            if base and v:
                print(f"  SUM A b{b} ktp{ktp} {v:7.3f} ms tax {v/base:.3f}x_of_plain",
                      flush=True)
    print("== B 长上下文 × b32(split 大 batch 是否仍赢,width=真实) ==")
    for pages in (64, 128):
        for ktp in (0, 16, 32):
            run_case(f"b32 x {pages*PAGE}tok w{pages}", 32, pages * PAGE, pages, ktp,
                     iters=20)
    print("== C width 敏感性:b32 1tok @split16(plan 宽度→空 tile/merge 写带) ==")
    for w in (16, 64, 256):
        run_case(f"b32 x 1tok w{w} split16", 32, 1, w, 16, iters=50)
    print("W2 BENCH DONE")


if __name__ == "__main__":
    main()
