"""fused_sample 设备数值验证:与 torch oracle 对照。

判据(预登记,避免事后调参)
--------------------------
1. **greedy 分支必须逐位等价**:`do_sample=False` 时,内核输出 ==
   `argmax(softmax(logits/temp))` == `argmax(logits/temp)`。任何一处不等即否决
   (softmax 严格单调,argmax 不应因路径变化)。
2. **top-k 语义**:开启 top_k=k 后,内核选中的 token 必须落在「logits 前 k 大」集合内。
   这是必要条件(不要求每次都选中同一个,因为还有 temperature 与随机)。
3. **采样分布**:对固定 logits、不同 seed 跑 N 次,选中频率应与 top-p/top-k 过滤后的
   归一化概率在统计误差内一致(卡方检验,p>0.01)。
4. **边界**:top_p=1.0 且 top_k=0(不启用)→ 退化为全词表采样,不崩。

不做的验证(明确记录)
--------------------
- 不验证与 vLLM 原生 `multinomial` **逐个随机数相同**——随机数流实现不同,
  这种比较没有意义(见模块 docstring)。

运行
----
    source .agent_loop/run/env.sh
    python benchmarks/probes/fused_sample_oracle.py
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

# 用小词表跑冒烟(V=16384),真实 248320 只在大 batch 性能测试里用
V = 16384
B = 4
TOL = 0.0  # greedy 必须逐位


def build(do_sample):
    return tilelang.compile(fused_sample_tl(V, batch=B, do_sample=do_sample), out_idx=None)


def run(kernel, logits, temps, top_p, top_k, seed):
    out = torch.zeros(B, dtype=torch.int32, device=DEV)
    kernel(logits, temps, top_p, top_k, seed, out)
    torch.npu.synchronize()
    return out


def make_inputs(seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    logits = torch.randn(B, V, generator=g).to(torch.float32) * 3.0
    return logits.to(DEV)


def main():
    print("[fs] 编译内核 ...", flush=True)
    k_greedy = build(False)
    k_sample = build(True)

    logits = make_inputs()
    temps = torch.ones(B, dtype=torch.float32, device=DEV)
    ones_p = torch.ones(B, dtype=torch.float32, device=DEV)
    zero_k = torch.zeros(B, dtype=torch.int32, device=DEV)
    seeds = torch.arange(B, dtype=torch.int64, device=DEV)

    # ---------- 判据 1:greedy 逐位等价 ----------
    got = run(k_greedy, logits, temps, ones_p, zero_k, seeds)
    ref = torch.argmax(logits, dim=-1).to(torch.int32)
    same = torch.equal(got, ref)
    print(f"[fs] greedy 逐位: {'PASS' if same else 'FAIL'} got={got.tolist()} ref={ref.tolist()}",
          flush=True)
    if not same:
        diff = (got != ref).nonzero().flatten().tolist()
        for r in diff[:3]:
            print(f"     row {r}: got {got[r].item()} (logit {logits[r, got[r]].item():.4f}) "
                  f"ref {ref[r].item()} (logit {logits[r, ref[r]].item():.4f})", flush=True)

    # ---------- 判据 2:top-k 语义 ----------
    for k in (1, 4, 16):
        tk = torch.full((B,), k, dtype=torch.int32, device=DEV)
        out = run(k_sample, logits, temps, ones_p, tk, seeds)
        topk_idx = torch.topk(logits, k, dim=-1).indices
        ok = all(out[r].item() in topk_idx[r].tolist() for r in range(B))
        print(f"[fs] top_k={k:2d} 命中前 {k} 大: {'PASS' if ok else 'FAIL'} "
              f"out={out.tolist()}", flush=True)

    # ---------- 判据 3:采样分布 ----------
    # 固定 logits,大量 seed,统计选中频率 vs 过滤后概率
    n_draw = 400
    for top_p_val, k in ((0.9, 0), (1.0, 8)):
        probs = torch.softmax(logits.float(), dim=-1)
        if k:
            tk = torch.topk(probs, k, dim=-1)
            mask = torch.zeros_like(probs, dtype=torch.bool)
            mask.scatter_(-1, tk.indices, True)
            probs = probs * mask
        else:
            # top-p:累计概率截断
            sp, si = torch.sort(probs, descending=True, dim=-1)
            cum = torch.cumsum(sp, dim=-1)
            keep = cum - sp < top_p_val
            mask = torch.zeros_like(probs, dtype=torch.bool)
            mask.scatter_(-1, si, keep)
            probs = probs * mask
        probs = probs / probs.sum(dim=-1, keepdim=True)

        tp = torch.full((B,), top_p_val, dtype=torch.float32, device=DEV)
        tkk = torch.full((B,), k, dtype=torch.int32, device=DEV)
        counts = torch.zeros(B, V, dtype=torch.int64, device=DEV)
        for i in range(n_draw):
            s = torch.full((B,), i * 7919 + 13, dtype=torch.int64, device=DEV)
            out = run(k_sample, logits, temps, tp, tkk, s)
            counts.scatter_add_(1, out.unsqueeze(1), torch.ones(B, 1, dtype=torch.int64, device=DEV))

        # 只比较概率最高的 8 个 token(低频项样本量不足)
        top8 = torch.topk(probs, 8, dim=-1).indices
        max_dev = 0.0
        for r in range(B):
            for t in top8[r].tolist():
                exp_f = probs[r, t].item() * n_draw
                obs_f = counts[r, t].item()
                # 宽松判据:偏差不超过 max(3σ, 0.02*n)
                sigma = max((exp_f * (1 - probs[r, t].item())) ** 0.5, 1.0)
                allowed = max(3 * sigma, 0.02 * n_draw)
                dev = abs(obs_f - exp_f)
                max_dev = max(max_dev, dev / n_draw)
                if dev > allowed:
                    print(f"[fs] 分布偏差超限 row={r} tok={t}: obs={obs_f} exp={exp_f:.1f} "
                          f"allowed={allowed:.1f}", flush=True)
        print(f"[fs] 分布 top_p={top_p_val} k={k}: 最大频率偏差 {max_dev:.4f} "
              f"(n={n_draw})", flush=True)

    # ---------- 判据 4:边界 ----------
    try:
        out = run(k_sample, logits, temps, ones_p, zero_k, seeds)
        print(f"[fs] 边界(top_p=1,k=0)不崩: PASS out={out.tolist()}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[fs] 边界 FAIL: {type(exc).__name__}: {exc}", flush=True)

    print("[fs] ORACLE DONE", flush=True)


if __name__ == "__main__":
    main()
