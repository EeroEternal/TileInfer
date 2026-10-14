"""H18 步骤②:内核级微基准——隔离「多行短 chunk」形状因子(E46 取证的裁决实验)。

背景:引擎 profiler(E46)显示 TI prefill 内核在 C8 分块负载下占计算时间 88%、单调用
p50~110ms,而单流长 chunk(E40/0a99edd)同一内核 TI≈FIA。本基准在**同一内核、同一
KV 前缀(16384 tok/行)、同一总 q 量(8192 tok)**下扫描 (batch × qo/行) 形状:

    A (1, 8192) 256 tiles | B (2, 4096) | C (4, 2048) | D (8, 1024) | E (8, 512)
    F (8, 256) | G (16, 512) 256 tiles | S 阶梯(8 行,qo=1024,kv 按 (b+1)*2048 递增,
    模拟真实分块早期)

健康标度下全部 case 耗时应近似相等(总 page 访问量相同);若多行短 chunk 病态存在,
D/E/F 相对 A 的膨胀倍数即该因子的定量。
"""
import os, sys, time
from pathlib import Path

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("TILEINFER_POOL_CAP", "16384")

from tileinfer.metadata import RaggedMetadata
from tileinfer.plan import plan_query_tiles_from_host
from tileinfer.kernels.attention.paged_decode_ascend950 import (
    build_prefill_kernel,
    forward_prefill,
    declared_pool_cap,
)
from tileinfer.integrations.vllm_ascend import _pad_tile_schedule, _next_pow2

dev = torch.device("npu", 0)
DT = torch.bfloat16
KVH, GROUP, PAGE, DIM, HEADS = 8, 4, 128, 256, 32
KV_FULL = 16384
POOL = 16384  # 声明容量(容量视图,实际分配见下——内核只解引用被用到的 span)

k_cache = torch.randn(POOL // 8, PAGE, KVH, DIM, dtype=DT, device=dev)  # 2048 页实际池
v_cache = torch.randn_like(k_cache)
print(f"actual pool pages={k_cache.shape[0]} declared_cap={declared_pool_cap(k_cache.shape[0])}", flush=True)


def meta_for(lens, seed=0):
    indptr, lpl = [0], []
    for L in lens:
        indptr.append(indptr[-1] + -(-L // PAGE))
        lpl.append(PAGE if L % PAGE == 0 else L % PAGE)
    n = indptr[-1]
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.randperm(k_cache.shape[0], generator=g).to(dev)[:n].to(torch.int32).contiguous()
    return RaggedMetadata(
        kv_indptr=torch.tensor(indptr, dtype=torch.int32, device=dev),
        kv_indices=idx,
        kv_last_page_len=torch.tensor(lpl, dtype=torch.int32, device=dev),
        page_size=PAGE,
    )


def run_case(label, batch, qo, lens, iters=20, pool_cap=POOL):
    max_qo = max(qo)
    max_tiles = sum((q * GROUP + 127) // 128 for q in qo)
    kernel_batch = 1 << (batch - 1).bit_length() if batch > 1 else 1
    kernel_batch += 1  # trash 行
    kernel_tiles = _next_pow2(max_tiles)
    spec = (kernel_batch, KVH, GROUP, DIM, PAGE, pool_cap, kernel_tiles, 128, "bfloat16")
    t0 = time.time()
    build_prefill_kernel(
        batch=kernel_batch, kv_heads=KVH, group=GROUP, dim=DIM, page_size=PAGE,
        num_pages_cap=pool_cap, num_tiles=kernel_tiles, block_q=128, dtype="bfloat16",
    )
    print(f"  [{label}] spec={spec} compile {time.time()-t0:.1f}s", flush=True)

    meta = meta_for(lens, seed=hash(label) % 999)
    pages_host = [(-(-L // PAGE)) for L in lens]
    schedule = plan_query_tiles_from_host(
        [q * GROUP for q in qo], pages_host, device=dev, block_q=128, load_balance=False
    )
    if kernel_tiles != schedule.num_tiles:
        schedule = _pad_tile_schedule(schedule, kernel_tiles, kernel_batch - 1)
    qo_dev = torch.tensor(qo, dtype=torch.int32, device=dev)
    q_pad = (torch.randn(batch, HEADS, max_qo, DIM, dtype=torch.float32, device=dev) * 0.2).to(DT)

    for _ in range(3):
        forward_prefill(q_pad, k_cache, v_cache, meta, schedule, qo_dev,
                        group=GROUP, block_q=128, kernel_batch=kernel_batch)
    torch.npu.synchronize()
    t0 = time.time()
    for _ in range(iters):
        forward_prefill(q_pad, k_cache, v_cache, meta, schedule, qo_dev,
                        group=GROUP, block_q=128, kernel_batch=kernel_batch)
    torch.npu.synchronize()
    ms = (time.time() - t0) / iters * 1e3
    print(f"  [{label}] batch={batch} qo/row={max_qo} kv/row={max(lens)} tiles={kernel_tiles} "
          f"calls={len(qo)} -> {ms:.2f} ms/call", flush=True)
    return ms


CASES = [
    ("A_b1_q8192", 1, [8192], [KV_FULL]),
    ("B_b2_q4096", 2, [4096, 4096], [KV_FULL, KV_FULL]),
    ("C_b4_q2048", 4, [2048] * 4, [KV_FULL] * 4),
    ("D_b8_q1024", 8, [1024] * 8, [KV_FULL] * 8),
    ("E_b8_q512_t128", 8, [512] * 8, [KV_FULL] * 8),
    ("F_b8_q256_t64", 8, [256] * 8, [KV_FULL] * 8),
    ("G_b16_q512", 16, [512] * 16, [KV_FULL] * 16),
    ("S_stair_b8_q1024", 8, [1024] * 8, [(b + 1) * 2048 for b in range(8)]),
]

if __name__ == "__main__":
    only = sys.argv[1:] or None
    results = {}
    for label, batch, qo, lens in CASES:
        if only and label not in only:
            continue
        results[label] = run_case(label, batch, qo, lens)
    print("\n== 汇总(healthy 标度 ⇒ 全 case 应近似相等)", flush=True)
    base = results.get("A_b1_q8192")
    for k, v in results.items():
        print(f"  {k:22s} {v:9.2f} ms   vs A: {v/base if base else float('nan'):.2f}x", flush=True)
