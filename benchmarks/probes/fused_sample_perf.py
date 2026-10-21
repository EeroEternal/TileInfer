"""fused_sample 性能对比:融合内核 vs 原生「sort + multinomial」路径。

测什么
------
1. **内核端到端延迟**:真实词表 V=248320、batch=8/32,多次迭代取中位数。
2. **原生等价路径**:torch 上做 softmax → sort → cumsum → 取阈值 → 采样,
   在同一设备上计时(只测排序+阈值部分,这是被替代的对象)。
3. **判定**:融合内核延迟应**显著低于**排序路径,否则「不值得」——
   这是这个移植存在的唯一理由。

注意
----
torch-npu 的 sort 实现与 vLLM 原生采样不完全相同,这里是**量级参考**,
不是逐位等价对比。真实收益以端到端 vLLM 基准为准(见 docs/performance.md)。
"""

import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import tileinfer  # noqa: F401
import tilelang
from tileinfer.kernels.sampling.fused_sample import fused_sample_plan, fused_sample_tl

DEV = torch.device("npu", 0)
torch.npu.set_device(DEV)

V = 248320  # 真实词表
WARMUP = 3
ITERS = 20


def bench(fn, iters=ITERS):
    for _ in range(WARMUP):
        fn()
    torch.npu.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.npu.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts), min(ts)


def main() -> None:
    print(f"[perf] V={V} 分块规划: {fused_sample_plan(V, 8)}", flush=True)

    for B in (8, 32):
        print(f"\n[perf] === batch={B} ===", flush=True)
        g = torch.Generator(device="cpu").manual_seed(0)
        logits = (torch.randn(B, V, generator=g) * 3.0).to(torch.float32).to(DEV)
        temps = torch.ones(B, dtype=torch.float32, device=DEV)
        tp = torch.full((B,), 0.9, dtype=torch.float32, device=DEV)
        tk = torch.zeros(B, dtype=torch.int32, device=DEV)
        seeds = torch.arange(B, dtype=torch.int64, device=DEV)

        # ---- 融合内核 ----
        try:
            kern = tilelang.compile(
                fused_sample_tl(V, batch=B, do_sample=True), out_idx=None
            )
            out = torch.zeros(B, dtype=torch.int32, device=DEV)

            def run_fused():
                kern(logits, temps, tp, tk, seeds, out)

            med, lo = bench(run_fused)
            print(f"[perf] fused kernel     : 中位 {med:7.3f} ms  最小 {lo:7.3f} ms",
                  flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[perf] fused kernel FAILED: {type(exc).__name__}: {exc}", flush=True)
            med = None

        # ---- 原生等价路径(排序) ----
        probs = torch.softmax(logits, dim=-1)

        def run_sort():
            sp, si = torch.sort(probs, dim=-1, descending=True)
            cum = torch.cumsum(sp, dim=-1)
            keep = cum - sp < 0.9
            masked = probs * keep.scatter(-1, si, keep)
            masked = masked / masked.sum(-1, keepdim=True)
            # 逆 CDF 采样
            u = torch.rand(B, 1, device=DEV)
            cdf = torch.cumsum(masked, dim=-1)
            return torch.searchsorted(cdf, u)

        med_s, lo_s = bench(run_sort)
        print(f"[perf] sort+cdf (torch) : 中位 {med_s:7.3f} ms  最小 {lo_s:7.3f} ms",
              flush=True)

        if med is not None:
            ratio = med_s / med
            verdict = "PASS" if ratio > 1.2 else "NO-GAIN"
            print(f"[perf] 排序/融合 = {ratio:.2f}x  {verdict}", flush=True)

        del logits, probs
        torch.npu.empty_cache()


if __name__ == "__main__":
    main()
