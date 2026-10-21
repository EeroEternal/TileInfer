"""L-B 页级 top-k 稀疏选择质量 oracle(H81 判据先行,iter-239;H85 跨层共享 iter-247)。

预登记判据(立项门槛,见 HYPOTHESES H81 / EVIDENCE E335):
  稠密 trace 中,centroid-score top-50% 页覆盖 >=99% softmax 质量 -> L-B 立项实施;
  <99% -> L-B disproven 关闭(注意力页分布均匀则稀疏不可行)。

H85 跨层共享判据(E362,iter-247):
  共享选择器(跨层平均质心,层×Q-head 联合 max 打分)的 top-B% 页集对**每一层**均
  >=99% 覆盖所需的最小预算 f*:f* <= 0.5 -> 共享可行(选择开销 ×10→×1 摊薄);
  f* > 0.8 -> 共享 disproven(退回逐层独立选择,H81 主体不受影响);中间 -> 记阈值。

选择器模型(与未来内核同构):
  - 摘要 = 每页 K 均值 centroid(每页 2*head_dim*4B ≈ 2KB,KV head 级,O(页数) 读取);
  - 打分 = q · centroid(逐 Q head),GQA 下同 KV head 的 Q heads 各自 top-k 取并集
    (页集按 KV head 共享,与「只读选中页 K/V」的内核现实一致);
  - 覆盖率按 Q head 独立计量,判据口径 = min(Q heads)(最严:每个 head 都要达标)。

运行模式:
  默认 = 合成分布(correlated/uniform,GQA 16Q/2KV/head_dim 256,S in {8192,16384}),
  仅验证脚本逻辑与合成信号,不构成立项判据;
  --load <pt> = 真实推理 dump 替换输入,判据以真实 trace 为准。dump 格式:
    单层 {"q": (HQ,D), "k": (HKV,S,D), "v": ...}(向后兼容);
    多层 {"layers": [{"q":..., "k":...}, ...]}(H85 口径,逐层独立+共享双判据)。
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
SHARE_BUDGETS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
SHARE_FAIL = 0.8


def _pad_to_pages(t: torch.Tensor) -> torch.Tensor:
    """(HKV, S, D) 尾部 pad 0 到页倍数(真实 decode 步 S=ctx+已生成,非页对齐)。"""
    s = t.shape[1]
    pad = (-s) % PAGE
    if pad:
        t = torch.nn.functional.pad(t, (0, 0, 0, pad))
    return t


def _dense_pages(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """q (HQ, D), k (HKV, S, D) -> 页质量 (HQ, n_pages),fp32 稳定 softmax(GQA 展开)。"""
    k_full = k.repeat_interleave(HQ // HKV, dim=0)  # (HQ, S, D)
    logits = torch.einsum("hd,hsd->hs", q, k_full) / math.sqrt(D)
    logits = logits - logits.amax(dim=-1, keepdim=True)
    p = torch.softmax(logits, dim=-1)  # (HQ, S)
    s = p.shape[-1]
    pad = (-s) % PAGE
    if pad:
        p = torch.nn.functional.pad(p, (0, pad))  # padding token 零质量
    return p.view(HQ, s // PAGE + (1 if pad else 0), PAGE).sum(dim=-1)


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


def _cover(q: torch.Tensor, k: torch.Tensor, masks: torch.Tensor) -> list[float]:
    """逐 Q head 覆盖率(masks 为 (HKV, n_pages) 页集)。"""
    page_q = _dense_pages(q, k)
    covers = []
    for hq in range(HQ):
        hk = hq // (HQ // HKV)
        sel = page_q[hq][masks[hk]]
        covers.append((sel.sum() / page_q[hq].sum()).item())
    return covers


def run_case(mode: str, s: int, seed: int = 0, load: dict | None = None) -> dict:
    if load is not None:
        q = load["q"].float()
        k = _pad_to_pages(load["k"]).float()
        v = _pad_to_pages(load["v"]).float()
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


def _shared_select(centroids: torch.Tensor, qs: torch.Tensor, budget: int) -> torch.Tensor:
    """H85 共享选择器:跨层平均质心 + 层×Q-head 联合 max 打分。

    centroids: (L, HKV, n_pages, D);qs: (L, HQ, D)。返回 (HKV, n_pages) bool。
    """
    n_pages = centroids.shape[2]
    centroid_mean = centroids.mean(dim=0)  # (HKV, n_pages, D)
    masks = torch.zeros(HKV, n_pages, dtype=torch.bool)
    for hk in range(HKV):
        joint = None
        for li in range(centroids.shape[0]):
            g0 = hk * (HQ // HKV)
            scores = torch.einsum("gd,pd->gp", qs[li][g0 : g0 + HQ // HKV], centroid_mean[hk])
            m = scores.max(dim=0).values  # (n_pages,)
            joint = m if joint is None else torch.maximum(joint, m)
        top = joint.topk(budget).indices
        masks[hk][top] = True
    return masks


def run_share_case(layers: list[dict]) -> dict | None:
    """H85 判据:共享选择器最小可行预算 f*。多层 dump 才有意义。"""
    if len(layers) < 2:
        return None
    ls = []
    for ld in layers:
        q, k = ld["q"].float(), _pad_to_pages(ld["k"]).float()
        s = k.shape[1]
        n_pages = s // PAGE
        centroid = k.view(HKV, n_pages, PAGE, D).mean(dim=2)
        ls.append({"q": q, "k": k, "n_pages": n_pages, "centroid": centroid})
    centroids = torch.stack([l["centroid"] for l in ls])  # (L, HKV, n_pages, D)
    qs = torch.stack([l["q"] for l in ls])  # (L, HQ, D)

    curve = []
    f_star = None
    for frac in SHARE_BUDGETS:
        budget = max(1, int(ls[0]["n_pages"] * frac))
        masks = _shared_select(centroids, qs, budget)
        all_covers = []
        for li, l in enumerate(ls):
            all_covers.extend(_cover(l["q"], l["k"], masks))
        mn = min(all_covers)
        curve.append({"budget_frac": frac, "cover_min": round(mn, 4)})
        if f_star is None and mn >= COVER_GATE:
            f_star = frac
    verdict = None
    if f_star is not None and f_star <= TOP_FRAC:
        verdict = "SHARE_PASS"
    elif f_star is None or f_star > SHARE_FAIL:
        verdict = "SHARE_DISPROVEN"
    else:
        verdict = "SHARE_THRESHOLD"
    return {"n_layers": len(ls), "f_star": f_star, "curve": curve, "verdict": verdict}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", type=str, default=None, help="真实推理 dump (.pt: q/k/v)")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    results = []
    share_result = None
    if args.load:
        dump = torch.load(args.load, map_location="cpu", weights_only=True)
        if "layers" in dump:
            layers = dump["layers"]
            s = layers[0]["k"].shape[1]
            for li, ld in enumerate(layers):
                results.append(run_case(f"real_L{li}", s, load=ld))
            share_result = run_share_case(layers)
        else:
            s = dump["k"].shape[1]
            results.append(run_case("real", s, load=dump))
    else:
        for mode in ("correlated", "uniform"):
            for s in (8192, 16384):
                results.append(run_case(mode, s))

    print("[select-quality] H81 判据口径: top-50%% 页 min-cover >= %.0f%% → 立项" % (COVER_GATE * 100))
    verdict = None
    real_mins = [r["cover_min"] for r in results if r["mode"].startswith("real")]
    for r in results:
        print(
            f"[select-quality] {r['mode']:10s} S={r['s']:6d} pages={r['n_pages']} "
            f"budget={r['budget']} cover_min={r['cover_min']:.4f} cover_mean={r['cover_mean']:.4f} "
            f"per_head={r['cover_per_head']}"
        )
    if real_mins:
        # H81 判据口径 = 所有 real 层 × 所有 Q head 的最小覆盖(最严口径)
        gmin = min(real_mins)
        verdict = "PASS_LB_GATE" if gmin >= COVER_GATE else "LB_DISPROVEN"
        print(f"[select-quality] H81 全层最小覆盖: {gmin:.4f} (layers={len(real_mins)})")
    if share_result is not None:
        print(f"[select-quality] H85 共享判据: layers={share_result['n_layers']} "
              f"f*={share_result['f_star']} curve={share_result['curve']}")
        print(f"[select-quality] H85 VERDICT: {share_result['verdict']} "
              f"(口径: f*<=0.5 共享可行 / >{SHARE_FAIL} 共享判死)")
    if verdict:
        print(f"[select-quality] VERDICT: {verdict}")
    else:
        print("[select-quality] 合成分布仅逻辑验证;立项判据以 --load 真实 trace 为准")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"cases": results, "share": share_result}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
