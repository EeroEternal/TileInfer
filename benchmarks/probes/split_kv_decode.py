"""Split-KV decode: measures the bandwidth it buys and checks the merge contract.

Run on the machine with the NPU and the working stack (see docs/architecture.md):

    source <CANN 9.3.x>/set_env.sh
    python benchmarks/probes/split_kv_decode.py

The problem it solves: the decode kernel launches `batch x kv_heads` blocks, so a single long
request starves the cores (measured 51 GB/s at 32k tokens, `docs/performance.md`).  Splitting the
KV axis turns one 8-block launch into 128 blocks and measures ~3x the bandwidth.

The merge here is done on the host in torch, on purpose: it validates the existing merge contract
out of `tileinfer.plan`/`testing.reference` before the device merge kernel exists.  The host merge
is a Python loop per request and must NOT be used in a serving path.
"""

import sys, time, statistics
from pathlib import Path

import torch

import torch_npu  # noqa: F401
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tileinfer.kernels.attention.paged_decode_ascend950 import build_split_kernel, padded_rows
from tileinfer.plan import plan_decode_tiles
from tileinfer.testing.reference import reference_attention
from tileinfer.metadata import RaggedMetadata

dev = torch.device("npu", 0); DT = torch.bfloat16
kvh, g, page, dim = 8, 8, 128, 128
BR = padded_rows(g)
tol = 3e-2

def build(lens, kv_tile_pages, seed=0):
    torch.manual_seed(seed)
    batch = len(lens)
    page_counts = torch.tensor([-(-L // page) for L in lens], dtype=torch.int32, device=dev)
    indptr = torch.zeros(batch + 1, dtype=torch.int32, device=dev)
    indptr[1:] = torch.cumsum(page_counts, dim=0).to(torch.int32)
    total = int(indptr[-1].item())
    k_cache = torch.randn(total, page, kvh, dim, dtype=DT, device=dev)
    v_cache = torch.randn(total, page, kvh, dim, dtype=DT, device=dev)
    idx = torch.arange(total, dtype=torch.int32, device=dev)
    meta = RaggedMetadata(kv_indptr=indptr, kv_indices=idx,
                          kv_last_page_len=torch.tensor(
                              [page if L % page == 0 else L % page for L in lens],
                              dtype=torch.int32, device=dev), page_size=page)
    q = torch.randn(batch, kvh * g, 1, dim, dtype=DT, device=dev) * 0.2
    q_pad = torch.zeros(batch, kvh * BR, dim, dtype=DT, device=dev)
    q_pad.view(batch, kvh, BR, dim)[:, :, :g] = q.view(batch, kvh, g, dim)
    sched = plan_decode_tiles(page_counts, block_q=1, kv_tile_pages=kv_tile_pages,
                              load_balance=False)
    return q, q_pad, k_cache, v_cache, meta, sched

def reference(q, k_cache, v_cache, meta, lens):
    m = RaggedMetadata(kv_indptr=meta.kv_indptr.cpu(), kv_indices=meta.kv_indices.cpu(),
                       kv_last_page_len=meta.kv_last_page_len.cpu(), page_size=page)
    m.qo_indptr = torch.arange(len(lens) + 1, dtype=torch.int32)
    return reference_attention(q.cpu().float(), k_cache.cpu().float(), v_cache.cpu().float(), m).float()

def host_merge(part_out, part_lse, sched, batch, lens):
    """Merge exactly like the reference's contract: weighted average with weights exp(lse-lse_all)."""
    out = torch.zeros(batch, kvh, BR, dim, dtype=torch.float32, device=dev)
    for b in range(batch):
        tiles = [t for t in range(sched.num_tiles) if int(sched.seq_ids[t]) == b]
        lse = torch.stack([part_lse[t * kvh + bh, :g] for t in tiles for bh in [0]])  # placeholder
    # do it per kv head to keep it readable
    res = torch.zeros(batch, kvh, g, dim, dtype=torch.float32, device=dev)
    for b in range(batch):
        tiles = [t for t in range(sched.num_tiles) if int(sched.seq_ids[t]) == b]
        for bh in range(kvh):
            lse_stack = torch.stack([part_lse[t * kvh + bh, :g].float() for t in tiles], 0)  # [S, g]
            lse_all = torch.logsumexp(lse_stack, 0)                                          # [g]
            acc = torch.zeros(g, dim, dtype=torch.float32, device=dev)
            for t in tiles:
                w = torch.exp(part_lse[t * kvh + bh, :g].float() - lse_all)                  # [g]
                acc += w[:, None] * part_out[t * kvh + bh, :g, :].float()
            res[b, bh] = acc
    return res

def run(lens, kv_tile_pages, iters=10, check=True):
    q, q_pad, k_cache, v_cache, meta, sched = build(lens, kv_tile_pages)
    num_tiles = sched.num_tiles
    seq_ids = sched.seq_ids.to(torch.int32)
    starts = sched.kv_page_starts.to(torch.int32)
    plens = sched.kv_page_lens.to(torch.int32)
    part_out = torch.empty(num_tiles * kvh, BR, dim, dtype=DT, device=dev)
    part_lse_flat = torch.empty(num_tiles * kvh * BR, dtype=torch.float32, device=dev)
    t0 = time.time()
    kern = build_split_kernel(batch=len(lens), kv_heads=kvh, group=g, dim=dim, page_size=page,
                              num_pages_cap=int(k_cache.shape[0]), num_tiles=num_tiles)
    compile_s = time.time() - t0
    kern(q_pad, k_cache, v_cache, meta.kv_indptr, meta.kv_indices, meta.kv_last_page_len,
         seq_ids, starts, plens, part_out, part_lse_flat)
    torch.npu.synchronize()
    part_lse = part_lse_flat.view(num_tiles * kvh, BR)

    msg = ""
    if check:
        exp = reference(q, k_cache, v_cache, meta, lens)
        got = host_merge(part_out, part_lse, sched, len(lens), lens)
        d = (got.cpu().view(len(lens), kvh * g, 1, dim) - exp).abs().max().item()
        msg = f" max|diff|={d:.4f} {'OK' if d < tol else 'WRONG'}"
    for _ in range(3):
        kern(q_pad, k_cache, v_cache, meta.kv_indptr, meta.kv_indices, meta.kv_last_page_len,
             seq_ids, starts, plens, part_out, part_lse_flat)
    torch.npu.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.time(); kern(q_pad, k_cache, v_cache, meta.kv_indptr, meta.kv_indices,
                               meta.kv_last_page_len, seq_ids, starts, plens, part_out, part_lse_flat)
        torch.npu.synchronize(); ts.append((time.time() - t0) * 1e3)
    ms = statistics.median(ts)
    nbytes = 2 * sum(lens) * kvh * dim * 2
    print(f"  tiles={num_tiles:3d} blocks={num_tiles*kvh:4d} kv_tile_pages={kv_tile_pages:4d} "
          f"{ms:7.3f} ms  {nbytes/(ms*1e-3)/1e9:6.1f} GB/s (compile {compile_s:.0f}s){msg}", flush=True)

print("correctness with real splits (host merge vs dense oracle):")
run([1], 2048, check=True, iters=3)
run([4096], 8, check=True, iters=3)      # 32 pages -> 4 tiles -> merge across 4 partials
print("starved shape (1 request, 32768 tokens = 256 pages):")
run([32768], 0, check=False)
run([32768], 16, check=False)
run([32768], 8, check=False)
print("starved shape (1 request, 65536 tokens = 512 pages):")
run([65536], 16, check=False)
print("wide control (64 requests, 512 tokens each):")
run([512] * 64, 0, check=False)
