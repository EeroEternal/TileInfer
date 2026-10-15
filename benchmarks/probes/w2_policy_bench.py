"""W2 矩阵 D:H21 per-bucket ktp 策略的悬崖扫描与长上下文收益复核。

机理(E61 波次模型):split-KV schedule 发射 ``batch × ceil(width/ktp)`` 个 tile,
时间 ≈ ceil(tiles/28 AIV) × 波时;空 tile(短上下文,页循环被钳 0)与实 tile 同价。
预测悬崖:batch×splits ≤ 28(1 波)⇒ 零税;跨过 28 ⇒ +1 波(~0.1-0.2ms/层)。
H21 策略候选(待本矩阵确认):{1:16, 2:32, 4:64, 8:128, ≥16:0(plain)}。

矩阵:
  D1 悬崖:b∈{2,4,8} × ktp∈{0,16,32,64,128} × 1tok(短上下文税;0=plain 参照)
  D2 长上下文收益:同 batch × 8K/16K × ktp∈{0,16,32,64,128}
  D3 策略整定:每个 batch 取 D2 中「不越悬崖的最大收益 ktp」,复测短上下文确认零税。

注意:ktp∈{64,128} 的 decode split/merge 内核无缓存,首次触发真实编译(~2-3min/规格);
窗口紧张时用 KTPS 环境变量收窄扫描集(如 KTPS=0,16,32)。数值契约:每个 split 点
vs 同 real lens 的 plain plan,maxdiff ≤ 3e-2。

运行:source .agent_loop/run/env.sh && python benchmarks/probes/w2_policy_bench.py
输出 SUM 行供 watcher 自动解析:`SUM b=<b> ktp=<k> ctx=<ctx> ms=<ms> vs_plain=<r> err=<e>`
"""

import os
import sys
import time
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
WIDTH = 256  # 引擎表宽(kernel 页):32K 上下文
# 池页数对齐引擎声明容量(pow2)→ 内核 spec(batch,kvh,group,dim,page,pool_cap)与引擎
# 一致,矩阵 D 编译的 split/merge 内核可被 A/B serve 的 prewarm 磁盘缓存命中(免二次编译)。
POOL = int(os.environ.get("POOL", "16384"))
TOL = 3e-2
KTPS = [int(x) for x in os.environ.get("KTPS", "0,16,32,64,128").split(",")]
BATCHES = [int(x) for x in os.environ.get("BATCHES", "2,4,8").split(",")]
ITERS = int(os.environ.get("ITERS", "50"))


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


def run_case(batch, real_tok, ktp, iters=ITERS):
    """满额 plan(width=256)跑 real_tok 真实上下文——镜像引擎满额 plan 语义。"""
    torch.manual_seed(0)
    pool_pages = max(batch * WIDTH + 1, POOL)
    k_cache = torch.randn(pool_pages, PAGE, KVH, DIM, dtype=DT, device=dev)
    v_cache = torch.randn_like(k_cache)
    real = meta_for([real_tok] * batch, pool_pages)
    full = meta_for([WIDTH * PAGE] * batch, pool_pages, seed=1)
    q = torch.randn(batch, KVH * GROUP, 1, DIM, dtype=DT, device=dev) * 0.2

    attn = BatchAttention(backend="tilelang-ascend950", device=dev, dtype=DT)

    def make_plan(meta, k):
        return attn.plan(
            kv_indptr=meta.kv_indptr,
            kv_indices=meta.kv_indices,
            kv_last_page_len=meta.kv_last_page_len,
            page_size=PAGE,
            num_qo_heads=KVH * GROUP,
            num_kv_heads=KVH,
            head_dim=DIM,
            causal=False,
            kv_tile_pages=k,
            k_cache=k_cache,
            v_cache=v_cache,
        )

    plain_plan = make_plan(real, 0)
    ref = attn.run(q, k_cache, v_cache, plan=plain_plan)
    torch.npu.synchronize()

    if ktp == 0:
        got, err = ref, 0.0
        plan = plain_plan
    else:
        plan = make_plan(full, ktp)
        got = attn.run(q, k_cache, v_cache, plan=plan, kv_indptr=real.kv_indptr,
                       kv_indices=real.kv_indices, kv_last_page_len=real.kv_last_page_len)
        torch.npu.synchronize()
        err = (got.float() - ref.float()).abs().max().item()

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
    return ms, err


def main():
    print(f"== W2 矩阵 D(H21 悬崖扫描): batches={BATCHES} ktps={KTPS} width={WIDTH} ==",
          flush=True)
    baseline = {}  # (batch, real_tok) -> plain ms
    fails = 0
    for batch in BATCHES:
        for real_tok in (1, 8192, 16384):
            for ktp in KTPS:
                tiles = batch * max(1, -(-WIDTH // ktp)) if ktp else batch
                tag = f"b={batch} ktp={ktp:3d} ctx={real_tok:5d} tiles={tiles:4d}"
                try:
                    ms, err = run_case(batch, real_tok, ktp)
                except Exception as exc:  # noqa: BLE001 - 单点失败不断链
                    print(f"{tag}  EXC {type(exc).__name__}: {exc}", flush=True)
                    fails += 1
                    continue
                if ktp == 0:
                    baseline[(batch, real_tok)] = ms
                    ratio = 1.0
                else:
                    ratio = ms / baseline[(batch, real_tok)]
                ok = "OK" if err <= TOL else "FAIL"
                if err > TOL:
                    fails += 1
                print(
                    f"{tag}  {ms:8.3f} ms  vs_plain={ratio:5.2f}x  err={err:.4f} {ok}",
                    flush=True,
                )
                print(
                    f"SUM b={batch} ktp={ktp} ctx={real_tok} ms={ms:.3f} "
                    f"vs_plain={ratio:.3f} err={err:.4f}",
                    flush=True,
                )
    print(f"WAITS: scans complete, fails={fails}", flush=True)


if __name__ == "__main__":
    main()
