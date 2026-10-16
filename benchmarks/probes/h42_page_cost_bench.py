"""H42 微基准:长上下文 decode attention 页循环成本分解(ktp 扫描 × SIMD softmax)。

背景(E116/E117):TI paged decode b8 分带(ktp128)下 8K(64 页)/16K(128 页)每请求
仅 1 个实 split,页循环串行 ~30µs/页,带宽下界 ~2.5µs/页(12x 余量)。本探针量化:
  A) ktp 扫描(并行度轴):b8 × 8K/16K,plain vs split{16,32,128} —— 验证 E40 结论在
     width=256 全宽 plan 引擎形态下的复现(E40 历史为 width=引擎桶 32 形态);
  B) SIMD softmax 轴(内核注释自述 decode shape ~2.6x):同矩阵 × TILEINFER_SIMD_SOFTMAX,
     判据=长 ctx 收益幅度与短形状(1 tok)是否零税(E61 纪律);
  C) 有效带宽对照:DeepGEMM-Ascend paged MQA decode 在 950 达 2006 GB/s(E118),我方
     GB/s 差距即页循环串行化的量化余量。

SIMD 开关在 T trace 期生效且不入内核缓存键,故 SIMD on/off 必须分进程跑(本脚本按
env 单模式运行,由调用方各跑一次)。运行:
  source .agent_loop/run/env.sh && \
  TILEINFER_SIMD_SOFTMAX=0 python benchmarks/probes/h42_page_cost_bench.py   # off 轮
  TILEINFER_SIMD_SOFTMAX=1 python benchmarks/probes/h42_page_cost_bench.py   # on 轮
输出行格式: [H42] simd=<0|1> <label> ktp=<k> <ms> ms err=<e> bw=<gbps> GB/s
"""

import os
import sys
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tileinfer import BatchAttention
from tileinfer.metadata import RaggedMetadata

DEV = torch.device("npu", 0)
DT = torch.bfloat16
KVH, GROUP, PAGE, DIM = 8, 4, 128, 256
WIDTH_PAGES = 256  # max_model_len=32K 引擎页表宽度(内核页),与在役 plan 形态一致
SIMD = os.environ.get("TILEINFER_SIMD_SOFTMAX", "0") not in ("", "0", "false", "False")


def meta_for(lens, pool_pages, seed=0):
    indptr = [0]
    lpl = []
    for L in lens:
        indptr.append(indptr[-1] + -(-L // PAGE))
        lpl.append(PAGE if L % PAGE == 0 else L % PAGE)
    n = indptr[-1]
    g = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(pool_pages, generator=g).to(DEV)[: max(n, 1)].to(torch.int32)
    idx = perm.repeat(-(-n // max(perm.numel(), 1)))[:n].contiguous()
    return RaggedMetadata(
        kv_indptr=torch.tensor(indptr, dtype=torch.int32, device=DEV),
        kv_indices=idx,
        kv_last_page_len=torch.tensor(lpl, dtype=torch.int32, device=DEV),
        page_size=PAGE,
    )


def run_case(label, batch, lens, ktp, iters=30):
    pool_pages = batch * WIDTH_PAGES + 1
    k_cache = torch.randn(pool_pages, PAGE, KVH, DIM, dtype=DT, device=DEV)
    v_cache = torch.randn_like(k_cache)
    real = meta_for(lens, pool_pages)
    full = meta_for([WIDTH_PAGES * PAGE] * batch, pool_pages, seed=1)
    q = torch.randn(batch, KVH * GROUP, 1, DIM, dtype=DT, device=DEV) * 0.2

    attn = BatchAttention(backend="tilelang-ascend950", device=DEV, dtype=DT)
    plan = attn.plan(
        kv_indptr=full.kv_indptr, kv_indices=full.kv_indices,
        kv_last_page_len=full.kv_last_page_len, page_size=PAGE,
        num_qo_heads=KVH * GROUP, num_kv_heads=KVH, head_dim=DIM, causal=False,
        kv_tile_pages=ktp, k_cache=k_cache, v_cache=v_cache,
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

    if ktp:
        plain = attn.plan(
            kv_indptr=real.kv_indptr, kv_indices=real.kv_indices,
            kv_last_page_len=real.kv_last_page_len, page_size=PAGE,
            num_qo_heads=KVH * GROUP, num_kv_heads=KVH, head_dim=DIM, causal=False,
            kv_tile_pages=0, k_cache=k_cache, v_cache=v_cache,
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
    print(f"[H42] simd={int(SIMD)} {label:34s} ktp={ktp:3d} {ms:8.3f} ms "
          f"err={err:.4f} bw={gb / ms * 1e3:7.1f} GB/s", flush=True)
    return ms


if __name__ == "__main__":
    print(f"[H42] simd={int(SIMD)} start", flush=True)
    print(f"[H42] == A/B 轴:b8 长上下文(width={WIDTH_PAGES}p 全宽 plan,引擎形态) ==", flush=True)
    for pages in (64, 128):  # 8K / 16K
        lens = [pages * PAGE] * 8
        run_case(f"b8 x {pages * PAGE} tok plain", 8, lens, 0)
        run_case(f"b8 x {pages * PAGE} tok split128", 8, lens, 128)
        run_case(f"b8 x {pages * PAGE} tok split32", 8, lens, 32)
        run_case(f"b8 x {pages * PAGE} tok split16", 8, lens, 16)
    print(f"[H42] == C 轴:短形状零税复核(E61 纪律) ==", flush=True)
    run_case("b8 x 1 tok plain", 8, [1] * 8, 0)
    run_case("b8 x 1 tok full-plan split128", 8, [1] * 8, 128)
    run_case("b8 x 1 tok full-plan split16", 8, [1] * 8, 16)
    print("[H42] BENCH DONE", flush=True)
