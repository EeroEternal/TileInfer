"""L-B2 分层 KV(池化余页)CPU oracle(H89,iter-253)。

判据(HYPOTHESES H89,预登记):
  真实 decode trace 上,分层注意力输出(选中页精确 token + 未选页页均值 landmark
  单虚拟 token 混合 softmax)与 exact 输出的逐 (层, Q-head) 相对 L2 误差,
  max ≤ TOL(3e-2)@ f=0.5(oracle-mass 选择器上界)⇒ 分层变体立项;
  超差 ⇒ L-B2 分层变体判死。

机理:未选页 128 个 token 压缩为 (K̄,ṽ) 后,读取量从 (1-f)·S token 降到
  (1-f)·S/128;误差来源=页内注意力权重方差 + softmax 归一化偏移(landmark 分数
  q·K̄ 对页质量 Σexp(q·k) 的替代偏差)。

选择器双口径:
  mass     = 按真实页质量(组内 Q-head max)top-f —— 上界口径,判据以它为准;
  centroid = q·页K均值(组内 Q-head max 联合打分)——E373 已证伪选择器,对照用。

运行:
  python lb2_hier_kv_oracle.py --selftest          # CPU 合成契约(小形状数学正确性)
  python lb2_hier_kv_oracle.py --load dump.pt      # 真实 trace 判据(E364/E373 格式)
"""

import argparse
import json
import math

import torch

HQ, HKV, D = 16, 2, 256
PAGE = 128
TOL = 3e-2
FRACS = (0.0, 0.125, 0.25, 0.5)
GATE_FRAC = 0.5


def pad_to_pages(t: torch.Tensor) -> torch.Tensor:
    """(HKV, S, D) 尾部 pad 0 到页倍数。"""
    s = t.shape[1]
    pad = (-s) % PAGE
    if pad:
        t = torch.nn.functional.pad(t, (0, 0, 0, pad))
    return t


def exact_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """q (HQ, D), k/v (HKV, S, D) -> exact 输出 (HQ, D),fp32 稳定 softmax(GQA 展开)。"""
    k_full = k.repeat_interleave(HQ // HKV, dim=0)
    v_full = v.repeat_interleave(HQ // HKV, dim=0)
    logits = torch.einsum("hd,hsd->hs", q, k_full) / math.sqrt(D)
    logits = logits - logits.amax(dim=-1, keepdim=True)
    p = torch.softmax(logits, dim=-1)
    return torch.einsum("hs,hsd->hd", p, v_full)


def page_masses(q: torch.Tensor, k: torch.Tensor, valid_lens: torch.Tensor) -> torch.Tensor:
    """页质量 (HQ, n_pages):逐页 softmax 概率和(padding token 零质量)。"""
    k_full = k.repeat_interleave(HQ // HKV, dim=0)
    logits = torch.einsum("hd,hsd->hs", q, k_full) / math.sqrt(D)
    logits = logits - logits.amax(dim=-1, keepdim=True)
    p = torch.softmax(logits, dim=-1)
    s = p.shape[-1]
    pad = (-s) % PAGE
    if pad:
        p = torch.nn.functional.pad(p, (0, pad))
    pm = p.view(HQ, -1, PAGE).sum(dim=-1)
    return pm


def select_mass(q: torch.Tensor, k: torch.Tensor, budget: int,
                valid_lens: torch.Tensor) -> torch.Tensor:
    """oracle-mass 选择器:组内 Q-head 页质量 max 联合 top-budget → (HKV, n_pages) bool。"""
    pm = page_masses(q, k, valid_lens)  # (HQ, n_pages)
    joint = torch.zeros(HKV, pm.shape[-1])
    for hk in range(HKV):
        joint[hk] = pm[hk * (HQ // HKV):(hk + 1) * (HQ // HKV)].max(dim=0).values
    masks = torch.zeros(HKV, pm.shape[-1], dtype=torch.bool)
    for hk in range(HKV):
        masks[hk][joint[hk].topk(budget).indices] = True
    return masks


def select_centroid(q: torch.Tensor, k: torch.Tensor, budget: int,
                    valid_lens: torch.Tensor) -> torch.Tensor:
    """centroid 选择器(E373 同构):q·页K均值,组内 Q-head max 联合 top-budget。"""
    n_pages = k.shape[1] // PAGE
    denom = valid_lens.float().clamp(min=1)[:, None]  # (n_pages, 1) 有效 token 数
    ck = k.view(HKV, n_pages, PAGE, D).sum(dim=2) / denom  # 有效均值
    masks = torch.zeros(HKV, n_pages, dtype=torch.bool)
    for hk in range(HKV):
        qs = q[hk * (HQ // HKV):(hk + 1) * (HQ // HKV)]
        joint = torch.einsum("gd,pd->gp", qs, ck[hk]).max(dim=0).values
        masks[hk][joint.topk(budget).indices] = True
    return masks


def hier_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   masks: torch.Tensor, valid_lens: torch.Tensor) -> torch.Tensor:
    """分层输出:选中页精确 token + 未选页 landmark(K̄,ṽ)单 token,混合 softmax。

    landmark logit = q·K̄/√d + log(n_valid_pg) —— 质量保持项:页质量 Σ_i exp(q·k_i)
    ≈ n·exp(q·K̄)(一阶 Taylor),缺 log 计数项会使混合 softmax 中未选页被系统性
    低估 n 倍(f=0 全池化时统一抵消、混合 f 下错误重分配,合成契约捕获)。

    返回 (HQ, D)。k/v 为已 pad 的 (HKV, S_pad, D);valid_lens (n_pages,) 每页有效 token 数。
    """
    n_pages = k.shape[1] // PAGE
    k_pg = k.view(HKV, n_pages, PAGE, D)
    v_pg = v.view(HKV, n_pages, PAGE, D)
    denom = valid_lens.float().clamp(min=1)[:, None]
    k_bar = k_pg.sum(dim=2) / denom  # (HKV, n_pages, D) 有效均值
    v_bar = v_pg.sum(dim=2) / denom
    lm_logcount = valid_lens.float().clamp(min=1).log()
    out = torch.empty(HQ, D)
    g = HQ // HKV
    for hk in range(HKV):
        sel = masks[hk]
        # 精确段:选中页 token 展平
        k_exact = k_pg[hk][sel].reshape(-1, D)
        v_exact = v_pg[hk][sel].reshape(-1, D)
        # landmark 段:未选页(含部分有效页的均值)
        k_lm = k_bar[hk][~sel]
        v_lm = v_bar[hk][~sel]
        lc = lm_logcount[~sel]
        for qi in range(g):
            qh = q[hk * g + qi]
            le = (k_exact @ qh) / math.sqrt(D) if k_exact.numel() else k_exact.new_zeros(0)
            ll = (k_lm @ qh) / math.sqrt(D) + lc
            logits = torch.cat([le, ll])
            logits = logits - logits.amax()
            p = torch.softmax(logits, dim=-1)
            vals = torch.cat([v_exact, v_lm])
            out[hk * g + qi] = p @ vals
    return out


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).norm() / b.norm().clamp(min=1e-12)).item()


def load_dump(path: str):
    """E364/E373 dump 格式:单层 {"q","k","v"} 或多层 {"layers":[...]}。"""
    d = torch.load(path, map_location="cpu", weights_only=False)
    return d["layers"] if "layers" in d else [d]


def run_dump(path: str, fracs=FRACS) -> dict:
    layers = load_dump(path)
    results = []
    for li, ld in enumerate(layers):
        q = ld["q"].float()
        k = pad_to_pages(ld["k"]).float()
        v = pad_to_pages(ld["v"]).float()
        s = k.shape[1]
        n_pages = s // PAGE
        valid_lens = torch.full((n_pages,), PAGE, dtype=torch.long)
        valid_lens[-1] = PAGE - ((-ld["k"].shape[1]) % PAGE) or PAGE
        out_e = exact_attention(q, k, v)
        for frac in fracs:
            budget = int(round(n_pages * frac))
            for sel_name, sel_fn in (("mass", select_mass), ("centroid", select_centroid)):
                if budget == 0:
                    masks = torch.zeros(HKV, n_pages, dtype=torch.bool)
                else:
                    masks = sel_fn(q, k, budget, valid_lens)
                errs = [rel_err(hier_attention(q, k, v, masks, valid_lens)[h],
                                out_e[h]) for h in range(HQ)]
                results.append({
                    "layer": li, "frac": frac, "selector": sel_name,
                    "rel_max": round(max(errs), 5),
                    "rel_mean": round(sum(errs) / len(errs), 5),
                })
    summary = {}
    for frac in fracs:
        for sel_name in ("mass", "centroid"):
            rows = [r for r in results if r["frac"] == frac and r["selector"] == sel_name]
            summary[f"f{frac}_{sel_name}"] = {
                "rel_max_all_layers": max(r["rel_max"] for r in rows),
                "read_ratio": round(frac + (1 - frac) / PAGE, 4),
            }
    gate = summary[f"f{GATE_FRAC}_mass"]["rel_max_all_layers"]
    return {
        "path": path, "n_layers": len(layers), "n_pages": layers[0]["k"].shape[1] // PAGE,
        "summary": summary, "per_layer": results,
        "gate": {"frac": GATE_FRAC, "selector": "mass", "tol": TOL,
                 "rel_max": gate, "pass": bool(gate <= TOL)},
    }


def selftest() -> int:
    """合成契约:小形状数学正确性 + 门控行为(H89 实验链 CPU 契约)。"""
    ok = True
    torch.manual_seed(7)
    # 1) 全页精确选择(mass 选择器 budget=n_pages)⇒ 分层输出 ≡ exact
    s = 3 * PAGE
    k = torch.randn(HKV, s, D)
    v = torch.randn(HKV, s, D)
    q = torch.randn(HQ, D)
    valid = torch.full((s // PAGE,), PAGE, dtype=torch.long)
    masks = torch.ones(HKV, s // PAGE, dtype=torch.bool)
    e = exact_attention(q, k, v)
    h = hier_attention(q, k, v, masks, valid)
    r = rel_err(h, e)
    print(f"[1] 全页精确 rel={r:.2e} (期望 ~0)")
    ok &= r < 1e-5

    # 2) f=0 landmark 数学:单页场景 S=PAGE,输出 = softmax(q·K̄)ṽ vs exact
    k1 = torch.randn(HKV, PAGE, D)
    v1 = torch.randn(HKV, PAGE, D)
    q1 = torch.randn(HQ, D)
    valid1 = torch.full((1,), PAGE, dtype=torch.long)
    m0 = torch.zeros(HKV, 1, dtype=torch.bool)
    e1 = exact_attention(q1, k1, v1)
    h1 = hier_attention(q1, k1, v1, m0, valid1)
    # 手工参照
    ref = torch.empty(HQ, D)
    gsz = HQ // HKV
    for hk in range(HKV):
        kb = k1[hk].mean(dim=0)
        vb = v1[hk].mean(dim=0)
        for qi in range(gsz):
            # 单 landmark:softmax 退化为 1,输出=ṽ(权重=1)
            ref[hk * gsz + qi] = vb
    r2 = rel_err(h1, ref)
    print(f"[2] f=0 landmark vs 手工参照 rel={r2:.2e} (期望 ~0)")
    ok &= r2 < 1e-5

    # 3) 尖锐分布 + mass 选择器:top-1 页含绝大部分质量时,大 f 误差应小于小 f
    s3 = 4 * PAGE
    k3 = torch.randn(HKV, s3, D) * 0.1
    v3 = torch.randn(HKV, s3, D)
    q3 = torch.randn(HQ, D) * 0.5
    # 构造强信号页:page 0 的 k 与 q 对齐且幅度大
    for hk in range(HKV):
        k3[hk, :PAGE] = q3[hk * gsz] * 6.0 + torch.randn(PAGE, D) * 0.05
    valid3 = torch.full((4,), PAGE, dtype=torch.long)
    errs = {}
    for frac in (0.0, 0.25, 0.5):
        b = max(1, int(4 * frac)) if frac > 0 else 0
        m = select_mass(q3, k3, b, valid3) if b else torch.zeros(HKV, 4, dtype=torch.bool)
        errs[frac] = rel_err(hier_attention(q3, k3, v3, m, valid3),
                             exact_attention(q3, k3, v3))
    print(f"[3] 尖锐分布误差随 f 单调下降: f0={errs[0.0]:.3f} f.25={errs[0.25]:.3f} "
          f"f.5={errs[0.5]:.3f}")
    ok &= errs[0.5] < errs[0.0] and errs[0.25] < errs[0.0]

    # 4) 非页对齐尾页:pad 页 landmark 不产生 NaN/Inf 且全页精确 rel≈0
    s4 = 2 * PAGE + 37
    k4 = torch.randn(HKV, s4, D)
    v4 = torch.randn(HKV, s4, D)
    q4 = torch.randn(HQ, D)
    kp = pad_to_pages(k4)
    vp = pad_to_pages(v4)
    valid4 = torch.full((3,), PAGE, dtype=torch.long)
    valid4[-1] = 37
    m4 = torch.ones(HKV, 3, dtype=torch.bool)
    e4 = exact_attention(q4, kp, vp)
    h4 = hier_attention(q4, kp, vp, m4, valid4)
    r4 = rel_err(h4, e4)
    print(f"[4] 非页对齐(尾页 37)全页精确 rel={r4:.2e} (期望 ~0)")
    ok &= r4 < 1e-4 and torch.isfinite(h4).all()

    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", help="真实 trace dump(.pt,E364/E373 格式)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", help="结果 JSON 输出路径")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not args.load:
        ap.error("需要 --load <dump.pt> 或 --selftest")
    res = run_dump(args.load)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
    print(json.dumps(res["summary"], indent=2, ensure_ascii=False))
    print("GATE f=0.5 mass:", res["gate"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
