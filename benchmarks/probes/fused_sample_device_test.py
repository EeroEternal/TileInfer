"""fused_sample 设备数值验证(与 torch oracle 对照)。

判据(预登记,避免事后调参)
--------------------------
1. **greedy 必须逐位等价**:`do_sample=False` 的 argmax 与 torch 完全一致。
2. **top-k 语义**:内核选中的 token 必须落在「logits 前 k 大」集合内。
3. **保留集一致性**:top-p=0.9 / top-k=8 时,内核采样 200 次**零越界**
   (选中的 token 必在理论保留集内)。
4. **分布拟合**:卡方/df 应接近 1(接受域 [0.5, 1.5]),且最大频率偏差 < 0.02。
5. **边界**:top_p=1 且 k=0(都不启用)不崩。

不做的验证:不比对与原生 `multinomial` 的逐个随机数(随机数流实现不同)。

运行
----
    source .agent_loop/run/env.sh
    python benchmarks/probes/fused_sample_device_test.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import tileinfer  # noqa: F401
import tilelang
from tileinfer.kernels.sampling.fused_sample import fused_sample_tl

DEV = torch.device("npu", 0)
torch.npu.set_device(DEV)

V = 16384
B = 2
N_DRAW = 2000
FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[dev] {name}: {'PASS' if ok else 'FAIL'} {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def build(do_sample: bool):
    return tilelang.compile(
        fused_sample_tl(V, batch=B, do_sample=do_sample), out_idx=None
    )


def run(kern, logits, temps, tp, tk, seed):
    out = torch.zeros(B, dtype=torch.int32, device=DEV)
    kern(logits, temps, tp, tk, seed, out)
    torch.npu.synchronize()
    return out


def main() -> int:
    g = torch.Generator(device="cpu").manual_seed(0)
    logits = (torch.randn(B, V, generator=g) * 3.0).to(torch.float32).to(DEV)
    temps = torch.ones(B, dtype=torch.float32, device=DEV)
    ones = torch.ones(B, dtype=torch.float32, device=DEV)
    zeros = torch.zeros(B, dtype=torch.int32, device=DEV)
    seeds = torch.arange(B, dtype=torch.int64, device=DEV)

    # ---- 判据 1:greedy 逐位 ----
    k_g = build(False)
    got = run(k_g, logits, temps, ones, zeros, seeds)
    ref = torch.argmax(logits, dim=-1).to(torch.int32)
    check("1 greedy 逐位", torch.equal(got, ref), f"got={got.tolist()} ref={ref.tolist()}")
    del k_g
    torch.npu.empty_cache()

    # ---- 判据 3:保留集(理论) ----
    lg = logits.double()
    mx = lg.max(dim=-1, keepdim=True).values
    e = torch.exp(lg - mx)
    total = e.sum(dim=-1, keepdim=True)
    sp, si = torch.sort(e, descending=True, dim=-1)
    cum = torch.cumsum(sp, dim=-1)
    n_keep = (cum < 0.9 * total).sum(dim=-1) + 1
    keep_top_p = [set(si[r, : n_keep[r]].tolist()) for r in range(B)]
    keep_top_k = [set(si[r, :8].tolist()) for r in range(B)]

    k_s = build(True)
    tp09 = torch.full((B,), 0.9, dtype=torch.float32, device=DEV)
    k8 = torch.full((B,), 8, dtype=torch.int32, device=DEV)

    for label, tp, tk, keep in (
        ("top_p=0.9", tp09, zeros, keep_top_p),
        ("top_k=8", ones, k8, keep_top_k),
    ):
        outside = 0
        for i in range(200):
            sd = torch.full((B,), i * 7919 + 13, dtype=torch.int64, device=DEV)
            out = run(k_s, logits, temps, tp, tk, sd)
            for r in range(B):
                if out[r].item() not in keep[r]:
                    outside += 1
        check(f"3 保留集 {label}", outside == 0, f"越界={outside}/400")

    # ---- 判据 2:top-k 命中前 k 大 ----
    for kk in (1, 4, 16):
        tk = torch.full((B,), kk, dtype=torch.int32, device=DEV)
        out = run(k_s, logits, temps, ones, tk, seeds)
        topk_idx = torch.topk(logits, kk, dim=-1).indices
        ok = all(out[r].item() in topk_idx[r].tolist() for r in range(B))
        check(f"2 top_k={kk}", ok, f"out={out.tolist()}")

    # ---- 判据 4:分布拟合 ----
    for label, tp, tk, use_k in (
        ("top_p=0.9", tp09, zeros, 0),
        ("top_k=8", ones, k8, 8),
    ):
        probs = torch.softmax(logits.float(), dim=-1)
        if use_k:
            idx = torch.topk(probs, use_k, dim=-1).indices
            mask = torch.zeros_like(probs, dtype=torch.bool)
            mask.scatter_(-1, idx, True)
            probs = probs * mask
        else:
            sp2, si2 = torch.sort(probs, descending=True, dim=-1)
            cum2 = torch.cumsum(sp2, dim=-1)
            keep = cum2 - sp2 < 0.9
            mask = torch.zeros_like(probs, dtype=torch.bool)
            mask.scatter_(-1, si2, keep)
            probs = probs * mask
        probs = probs / probs.sum(dim=-1, keepdim=True)

        counts = torch.zeros(B, V, dtype=torch.int64, device=DEV)
        for i in range(N_DRAW):
            sd = torch.full((B,), i * 7919 + 13, dtype=torch.int64, device=DEV)
            out = run(k_s, logits, temps, tp, tk, sd)
            counts.scatter_add_(
                1, out.unsqueeze(1), torch.ones(B, 1, dtype=torch.int64, device=DEV)
            )

        chi2_max, max_dev = 0.0, 0.0
        for r in range(B):
            sel = probs[r] * N_DRAW >= 5.0
            obs = counts[r][sel].double()
            exp = probs[r][sel].double() * N_DRAW
            if exp.numel() > 1:
                c2 = (((obs - exp) ** 2) / exp).sum().item()
                chi2_max = max(chi2_max, c2 / (exp.numel() - 1))
            freq = counts[r].double() / N_DRAW
            max_dev = max(max_dev, (freq - probs[r].double()).abs().max().item())
        check(
            f"4 分布拟合 {label}",
            0.5 <= chi2_max <= 1.5 and max_dev < 0.02,
            f"chi2/df={chi2_max:.3f} max_dev={max_dev:.4f}",
        )

    # ---- 判据 5:边界 ----
    try:
        out = run(k_s, logits, temps, ones, zeros, seeds)
        check("5 边界(top_p=1,k=0)", True, f"out={out.tolist()}")
    except Exception as exc:  # noqa: BLE001
        check("5 边界(top_p=1,k=0)", False, f"{type(exc).__name__}: {exc}")

    print(f"[dev] {'ALL PASS' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}", flush=True)
    return 0 if not FAILS else 1


if __name__ == "__main__":
    raise SystemExit(main())
