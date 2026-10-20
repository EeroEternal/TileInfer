"""H74 判别实验:DeepInfer 的两类「多核/流水」手法在我们的内核形状下是否适用。

DeepInfer 的 asc 算子骨架:
    cor es = get_num_vec_cores()
    with T.Kernel(cores) as pid:
        for task in T.Persistent([...], cores, pid, num_stages=N): ...

要点是**每个核吃多个 task**(Persistent = 任务摊平到固定核数),不是页内流水。
我们的 attention kernel 用 `T.Kernel(num_blocks)` + `for p in T.serial(num_pages)`,
即"核数=任务数,每核内部再串行页循环"。

本探针只测「页内流水」这一维是否有效(H74-Pipelined 已在 h74_sw_pipeline_bench 测过),
这里补一条对照:**在同一页循环里把 num_stages 拉高到 8/16**,看是否存在拐点,
以及 b1 场景(核数远少于 40 个 vector core)是否因 launch 开销主导。

判据:
  b1 × {128, 2048, 8192} 的 stages ∈ {0,2,4,8,16}
  若没有任何一档 > 1.05x → 「页内流水」方向整体证否,应转向「提高核数」。
"""

import sys
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import tileinfer  # noqa: F401
from tileinfer.kernels.attention.paged_decode_ascend950 import (
    build_decode_kernel,
    declared_pool_cap,
)

torch.npu.set_device(torch.device("npu", 0))

DEV = torch.device("npu", 0)
DT = torch.bfloat16
KVH, GROUP, PAGE, DIM = 8, 4, 128, 256
WIDTH_PAGES = 256
ITERS = 50
STAGES = (0, 2, 4, 8, 16)


def make_meta(lens, pool_pages, seed=0):
    indptr, lpl = [0], []
    for L in lens:
        indptr.append(indptr[-1] + -(-L // PAGE))
        lpl.append(PAGE if L % PAGE == 0 else L % PAGE)
    n = indptr[-1]
    cap = declared_pool_cap(pool_pages)
    g = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(cap, generator=g).to(DEV)[: max(n, 1)].to(torch.int32)
    idx = perm.repeat(-(-n // max(perm.numel(), 1)))[:n].contiguous()
    if idx.numel() < cap:
        idx = torch.cat([idx, torch.zeros(cap - idx.numel(), dtype=torch.int32, device=DEV)])
    return (
        torch.tensor(indptr, dtype=torch.int32, device=DEV),
        idx,
        torch.tensor(lpl, dtype=torch.int32, device=DEV),
    )


def run_case(label, lens):
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

    def kernel(st):
        return build_decode_kernel(
            batch=batch, kv_heads=KVH, group=GROUP, dim=DIM, page_size=PAGE,
            num_pages_cap=pool_pages, dtype="bfloat16", out_manual=True, sw_pipeline=st,
        )

    def call(kern):
        kern(q_pad, k_cache, v_cache, indptr, indices, lpl, out_pad)
        return out_pad

    def bench(kern):
        for _ in range(5):
            call(kern)
        torch.npu.synchronize()
        best = 1e9
        for _ in range(2):
            t0 = time.perf_counter()
            for _ in range(ITERS):
                call(kern)
            torch.npu.synchronize()
            best = min(best, (time.perf_counter() - t0) / ITERS * 1e3)
        return best

    results = {}
    ref = None
    for st in STAGES:
        try:
            kern = kernel(st)
            got = call(kern)
            torch.npu.synchronize()
            if ref is None:
                ref = got.clone()
                err = 0.0
            else:
                err = (got.float() - ref.float()).abs().max().item()
            results[st] = (bench(kern), err)
        except Exception as exc:
            results[st] = (None, str(exc)[:80])

    base = results[0][0]
    for st in STAGES:
        ms, err = results[st]
        if ms is None:
            print(f"[H74D] {label:14s} stages={st:2d} FAILED: {err}", flush=True)
        else:
            print(f"[H74D] {label:14s} stages={st:2d} {ms:8.4f} ms err={err:.4f} "
                  f"speedup={base / ms:.3f}x", flush=True)


if __name__ == "__main__":
    print("[H74D] 页内流水深扫(num_stages 拐点探测)", flush=True)
    for L in (128, 2048, 8192):
        run_case(f"b1 x {L}", [L])
    for L in (128, 8192):
        run_case(f"b8 x {L}", [L] * 8)
    print("[H74D] DONE", flush=True)
