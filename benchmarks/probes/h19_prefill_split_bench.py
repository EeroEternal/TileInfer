"""H19 微基准:split-KV prefill 在 serving 真实形状下的内核级收益(E46/E47 的裁决实验)。

背景:E46/E47 实证 W1-C8 病态 = TI prefill 内核每 tile 串行 KV 页遍历(向量链 96% 忙,
prefill 相位 tiles 占用 <10%);H19 把每个 q-block tile 沿 KV 轴切分为并行 partial + LSE 合并。
本基准在 serving 真实配置(Qwen3.6 full-attn 层:kvh=2、group=8、dim=256、page=128)下,
对 16K 长上下文 chunk 场景(8 请求 × 每步 1024 tok,KV 前缀 16384/行)扫描 kv_tile_pages:

    U  unsplit(对照,= 0a99edd 现状)         ktp=64 → 2 splits   ktp=32 → 4
    ktp=16 → 8 splits(在役 16K C8 预期档)    S  阶梯(8 行 kv 递增,模拟 chunk 早期)

每档输出 ms/call(含 index_select 打包、LSE 合并、scatter 解包的 eager 主机成本)与
对 U 的加速比;数值以与 U 输出的 max|diff| 校验(TOL 3e-2)。
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
    forward_prefill_split,
    plan_prefill_split_ctx,
    declared_pool_cap,
)
from tileinfer.integrations.vllm_ascend import _pad_tile_schedule, _next_pow2

dev = torch.device("npu", 0)
DT = torch.bfloat16
KVH, GROUP, PAGE, DIM, HEADS = 2, 8, 128, 256, 16  # serving 真实配置
KV_FULL = 16384
POOL = 16384

k_cache = torch.randn(POOL // 8, PAGE, KVH, DIM, dtype=DT, device=dev)
v_cache = torch.randn_like(k_cache)
CAP = declared_pool_cap(k_cache.shape[0])
print(f"actual pool pages={k_cache.shape[0]} declared_cap={CAP}", flush=True)


def meta_for(lens, seed=0):
    indptr, lpl = [0], []
    for L in lens:
        indptr.append(indptr[-1] + -(-L // PAGE))
        lpl.append(PAGE if L % PAGE == 0 else L % PAGE)
    n = indptr[-1]
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.randperm(k_cache.shape[0], generator=g).to(dev)[:n].to(torch.int32).contiguous()
    if idx.numel() < CAP:
        idx = torch.cat([idx, idx.new_zeros(CAP - idx.numel())])
    return idx, RaggedMetadata(
        kv_indptr=torch.tensor(indptr, dtype=torch.int32, device=dev),
        kv_indices=idx,
        kv_last_page_len=torch.tensor(lpl, dtype=torch.int32, device=dev),
        page_size=PAGE,
    )


def make_case(batch, qo, lens, seed):
    idx, meta = meta_for(lens, seed)
    pages_host = [(-(-L // PAGE)) for L in lens]
    lpl_host = [PAGE if L % PAGE == 0 else L % PAGE for L in lens]
    max_qo = max(qo)
    q = (torch.randn(batch, HEADS, max_qo, DIM, dtype=torch.float32, device=dev) * 0.2).to(DT)
    return idx, meta, pages_host, lpl_host, q, max_qo


def timeit(fn, iters=20, warmup=3):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.time() - t0) / iters * 1e3


def run_unsplit(label, batch, qo, lens, ref_out):
    kernel_batch = _next_pow2(batch) + 1
    max_tiles = sum((q * GROUP + 127) // 128 for q in qo)
    kernel_tiles = _next_pow2(max_tiles)
    t0 = time.time()
    build_prefill_kernel(
        batch=kernel_batch, kv_heads=KVH, group=GROUP, dim=DIM, page_size=PAGE,
        num_pages_cap=CAP, num_tiles=kernel_tiles, block_q=128, dtype="bfloat16",
    )
    print(f"  [{label}] compile unsplit tiles={kernel_tiles} {time.time()-t0:.1f}s", flush=True)
    idx, meta, pages_host, lpl_host, q, max_qo = make_case(batch, qo, lens, hash(label) % 999)
    sched = plan_query_tiles_from_host(
        [x * GROUP for x in qo], pages_host, device=dev, block_q=128, kv_tile_pages=0,
        load_balance=False,
    )
    if kernel_tiles != sched.num_tiles:
        sched = _pad_tile_schedule(sched, kernel_tiles, kernel_batch - 1)
    qo_dev = torch.tensor(qo, dtype=torch.int32, device=dev)
    ms = timeit(lambda: forward_prefill(
        q, k_cache, v_cache, meta, sched, qo_dev, group=GROUP, block_q=128,
        kernel_batch=kernel_batch))
    out = forward_prefill(q, k_cache, v_cache, meta, sched, qo_dev, group=GROUP, block_q=128,
                          kernel_batch=kernel_batch)
    if ref_out is not None:
        print(f"  [{label}] {ms:.2f} ms/call (ref)", flush=True)
    else:
        print(f"  [{label}] {ms:.2f} ms/call", flush=True)
    return ms, out


def run_split(label, batch, qo, lens, ktp, ref_out):
    kernel_batch = _next_pow2(batch) + 1
    pages_host0 = [(-(-L // PAGE)) for L in lens]
    # H20 语义:每 (req,qblock) 分段数 = head 段 + 1 frontier,最坏
    # ceil((pages-1)/ktp)+1 —— 估算按上界(+1)取,ctx 对 kernel_tiles 不足是静默截断。
    splits_per_req = [max(1, -(-p // ktp)) + 1 for p in pages_host0]
    split_tiles = sum(
        ((x * GROUP + 127) // 128) * s for x, s in zip(qo, splits_per_req)
    )
    kernel_tiles = _next_pow2(split_tiles)
    ms_compile = _next_pow2(max(splits_per_req))
    idx, meta, pages_host, lpl_host, q, max_qo = make_case(batch, qo, lens, hash(label) % 999)
    t0 = time.time()
    ctx = plan_prefill_split_ctx(
        list(qo), pages_host, lpl_host, group=GROUP, kv_tile_pages=ktp,
        kernel_batch=kernel_batch, kernel_tiles=kernel_tiles, kv_heads=KVH, block_q=128,
        max_qo_tokens=max_qo, page_size=PAGE, indices=idx, device=dev,
    )
    print(f"  [{label}] ctx {time.time()-t0:.1f}s tiles={kernel_tiles} ms_compile={ms_compile}",
          flush=True)
    ms = timeit(lambda: forward_prefill_split(q, k_cache, v_cache, ctx, group=GROUP))
    out = forward_prefill_split(q, k_cache, v_cache, ctx, group=GROUP)
    diff = (out.float() - ref_out.float()).abs().max().item()
    print(f"  [{label}] ktp={ktp} splits/req={splits_per_req[0]} tiles={kernel_tiles} "
          f"-> {ms:.2f} ms/call  diff_vs_unsplit={diff:.4f}", flush=True)
    return ms


if __name__ == "__main__":
    only = sys.argv[1:] or None
    results = {}
    CASE = ("16K_b8_q1024", 8, [1024] * 8, [KV_FULL] * 8)
    label, batch, qo, lens = CASE
    if not only or "U" in only or "ALL" in only:
        results["U_unsplit"], ref = run_unsplit(label, batch, qo, lens, None)
        if not only or "ALL" in only:
            for ktp in (64, 32, 16):
                results[f"ktp{ktp}"] = run_split(label, batch, qo, lens, ktp, ref)
        elif any(x.startswith("ktp") for x in only):
            pass
    else:
        # 分档补跑模式:先取 unsplit 参照输出(免编译走缓存)
        _, ref = run_unsplit(label, batch, qo, lens, None)
        for ktp in (64, 32, 16):
            tag = f"ktp{ktp}"
            if tag in only:
                results[tag] = run_split(label, batch, qo, lens, ktp, ref)
    if "S" in (only or []) or "ALL" in (only or []):
        Sl = ("stair_b8_q1024", 8, [1024] * 8, [(b + 1) * 2048 for b in range(8)])
        _, sref = run_unsplit(Sl[0], *Sl[1:], None)
        results["S_ktp16"] = run_split(Sl[0], *Sl[1:], 16, sref)
    print("\n== 汇总(vs unsplit)", flush=True)
    base = results.get("U_unsplit")
    for k, v in results.items():
        print(f"  {k:12s} {v:9.2f} ms   speedup {base/v if base and v else float('nan'):.2f}x",
              flush=True)
