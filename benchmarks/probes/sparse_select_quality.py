"""L-B 页级 top-k 稀疏选择质量 oracle(H81 判据先行,iter-239)。

预登记判据(立项门槛,见 HYPOTHESES H81 / EVIDENCE E335):
  稠密 trace 中,centroid-score top-50% 页覆盖 >=99% softmax 质量 -> L-B 立项实施;
  <99% -> L-B disproven 关闭(注意力页分布均匀则稀疏不可行)。

选择器模型(与未来内核同构):
  - 摘要 = 每页 K 均值 centroid(每页 2*head_dim*4B ≈ 2KB,KV head 级,O(页数) 读取);
  - 打分 = q · centroid(逐 Q head),GQA 下同 KV head 的 Q heads 各自 top-k 取并集
    (页集按 KV head 共享,与「只读选中页 K/V」的内核现实一致);
  - 覆盖率按 Q head 独立计量,判据口径 = min(Q heads)(最严:每个 head 都要达标)。

运行模式:
  默认 = 合成分布(correlated/uniform,GQA 16Q/2KV/head_dim 256,S in {8192,16384}),
  仅验证脚本逻辑与合成信号,不构成立项判据;
  --load <pt> = 真实推理 dump(Q,K,V 张量字典)替换输入,判据以真实 trace 为准。
"""

import argparse
import json
import math
import sys

import torch

HQ, HKV, D = 16, 2, 256
PAGE = 128
TOP_FRAC = 0.5
COVER_GATE = 0.99


def _dense_pages(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """q (HQ, D), k (HKV, S, D) -> 页质量 (HQ, n_pages),fp32 稳定 softmax(GQA 展开)。"""
    k_full = k.repeat_interleave(HQ // HKV, dim=0)  # (HQ, S, D)
    logits = torch.einsum("hd,hsd->hs", q, k_full) / math.sqrt(D)
    logits = logits - logits.amax(dim=-1, keepdim=True)
    p = torch.softmax(logits, dim=-1)  # (HQ, S)
    s = p.shape[-1]
    return p.view(HQ, s // PAGE, PAGE).sum(dim=-1)


def _select_pages(q: torch.Tensor, k: torch.Tensor, budget: int) -> torch.Tensor:
    """centroid 选择器:返回每个 KV head 的页集掩码 (HKV, n_pages) bool。

    score[h_q, pg] = q_h · centroid_pg;同 KV head 的 Q heads 各自 top-budget 取并集。
    """
    n_pages = k.shape[1] // PAGE
    centroid = k.view(HKV, n_pages, PAGE, D).mean(dim=2)  # (HKV, n_pages, D)
    masks = torch.zeros(HKV, n_pages, dtype=torch.bool)
    for hk in range(HKV):
        qs = q[hk * (HQ // HKV) : (hk + 1) * (HQ // HKV)]  # 该 KV head 的 Q heads
        scores = torch.einsum("gd,pd->gp", qs, centroid[hk])  # (G, n_pages)
        # 联合打分(页集按 KV head 硬共享,budget 不随 G 并集膨胀)
        joint = scores.max(dim=0).values
        top = joint.topk(budget).indices
        masks[hk][top] = True
    return masks


def run_case(mode: str, s: int, seed: int = 0, load: dict | None = None) -> dict:
    if load is not None:
        q, k, v = load["q"].float(), load["k"].float(), load["v"].float()
    elif mode == "correlated":
        g = torch.Generator().manual_seed(seed)
        k = torch.randn(HKV, s, D, generator=g)
        v = torch.randn(HKV, s, D, generator=g)
        q2 = torch.randn(HKV, D, generator=g) * 0.1
        idx = torch.randint(0, s, (HKV,), generator=g)
        for h in range(HKV):
            k[h, idx[h]] += q2[h] * 8.0
            q2[h] = q2[h] + k[h, idx[h]] * 0.5
        q = q2.repeat(HQ // HKV, 1) + torch.randn(HQ, D, generator=g) * 0.05
    else:
        g = torch.Generator().manual_seed(seed)
        k = torch.randn(HKV, s, D, generator=g)
        v = torch.randn(HKV, s, D, generator=g)
        q = torch.randn(HQ, D, generator=g)

    n_pages = s // PAGE
    budget = max(1, int(n_pages * TOP_FRAC))
    page_q = _dense_pages(q, k)  # (HQ, n_pages)
    masks = _select_pages(q, k, budget)  # (HKV, n_pages)
    covers = []
    for hq in range(HQ):
        hk = hq // (HQ // HKV)
        sel = page_q[hq][masks[hk]]
        covers.append((sel.sum() / page_q[hq].sum()).item())
    return {
        "mode": mode,
        "s": s,
        "n_pages": n_pages,
        "budget": budget,
        "cover_min": min(covers),
        "cover_mean": sum(covers) / len(covers),
        "cover_per_head": [round(c, 4) for c in covers],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", type=str, default=None, help="真实推理 dump (.pt: q/k/v)")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    results = []
    if args.load:
        dump = torch.load(args.load, map_location="cpu", weights_only=True)
        s = dump["k"].shape[1]
        results.append(run_case("real", s, load=dump))
    else:
        for mode in ("correlated", "uniform"):
            for s in (8192, 16384):
                results.append(run_case(mode, s))

    print("[select-quality] H81 判据口径: top-50%% 页 min-cover >= %.0f%% → 立项" % (COVER_GATE * 100))
    verdict = None
    for r in results:
        print(
            f"[select-quality] {r['mode']:10s} S={r['s']:6d} pages={r['n_pages']} "
            f"budget={r['budget']} cover_min={r['cover_min']:.4f} cover_mean={r['cover_mean']:.4f} "
            f"per_head={r['cover_per_head']}"
        )
        if r["mode"] == "real":
            verdict = "PASS_LB_GATE" if r["cover_min"] >= COVER_GATE else "LB_DISPROVEN"
    if verdict:
        print(f"[select-quality] VERDICT: {verdict}")
    else:
        print("[select-quality] 合成分布仅逻辑验证;立项判据以 --load 真实 trace 为准")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
