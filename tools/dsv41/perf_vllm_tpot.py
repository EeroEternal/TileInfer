"""vLLM 路径的 TPOT 测量：严格按 perf-comparison-rules 的差分法。

口径（docs/notes/perf-comparison-rules.md）：
  * 同一 prompt、非流式、`temperature=0`；
  * 先测 `max_tokens=N1` 的总时长 T1，再测 `max_tokens=N2` 的 T2；
  * **TPOT = (T2 − T1) / (N2 − N1)**（差分法，避免把 prefill/连接开销算进 per-token）；
  * ≥3 次重复取中位数，报极差；禁止用"单步计时"充当 TPOT。

用法：
  python tools/dsv41/perf_vllm_tpot.py --port 8122 --n1 8 --n2 72 --reps 3
  python tools/dsv41/perf_vllm_tpot.py --port 8122 --out /tmp/tpot_default.json --label default
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request

PROMPT = "The capital of France is"


def one_call(port: int, max_tokens: int, timeout: float = 900.0) -> tuple[float, int]:
    """返回 (总时长秒, 实际 completion tokens)。"""
    body = json.dumps({
        "model": "dsv41",
        "prompt": PROMPT,
        "max_tokens": max_tokens,
        "temperature": 0,
        # 必须忽略 EOS，否则模型会在 "Paris." 后早停，T(N1) 与 T(N2) 几乎相同，
        # 差分法会算出 1.5 ms/token 这种非物理结果（本会话已踩过）。
        "ignore_eos": True,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read())
    dt = time.perf_counter() - t0
    return dt, int(payload.get("usage", {}).get("completion_tokens", 0))


def median_of(fn, reps: int) -> tuple[float, float]:
    vals = [fn() for _ in range(reps)]
    med = statistics.median(vals)
    spread = (max(vals) - min(vals)) / med * 100 if med else float("inf")
    return med, spread


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8122)
    ap.add_argument("--n1", type=int, default=8)
    ap.add_argument("--n2", type=int, default=72)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--label", default="default")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.n2 <= args.n1:
        print("n2 必须大于 n1")
        return 2

    print(f"[{args.label}] endpoint=127.0.0.1:{args.port}  N1={args.n1} N2={args.n2} reps={args.reps}")
    try:
        samples1 = [one_call(args.port, args.n1) for _ in range(args.reps)]
        samples2 = [one_call(args.port, args.n2) for _ in range(args.reps)]
        t1 = statistics.median(s[0] for s in samples1)
        t2 = statistics.median(s[0] for s in samples2)
        s1 = (max(s[0] for s in samples1) - min(s[0] for s in samples1)) / t1 * 100
        s2 = (max(s[0] for s in samples2) - min(s[0] for s in samples2)) / t2 * 100
        # 用**实际**生成 token 数做分母：请求的 max_tokens 未必真的生成满
        ct1 = int(statistics.median(s[1] for s in samples1))
        ct2 = int(statistics.median(s[1] for s in samples2))
        print(f"[{args.label}] 实际生成 tokens: N1={ct1} N2={ct2}（请求 {args.n1}/{args.n2}）")
        if ct2 <= ct1:
            print(f"[{args.label}] 生成数没有增加，差分法不可用；请检查是否早停")
            return 1
    except Exception as exc:  # noqa: BLE001
        print(f"请求失败: {type(exc).__name__}: {exc}")
        return 1

    tpot = (t2 - t1) / (ct2 - ct1) * 1000.0
    if t2 <= t1:
        print(f"非物理结果（T2<=T1: {t2:.3f} <= {t1:.3f}），样本作废")
        return 1
    print(f"[{args.label}] T({args.n1})={t1:.3f}s (极差 {s1:.1f}%)  "
          f"T({args.n2})={t2:.3f}s (极差 {s2:.1f}%)")
    print(f"[{args.label}] TPOT = ({t2:.3f} − {t1:.3f}) / ({ct2} − {ct1}) = {tpot:.1f} ms/token")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({
                "label": args.label, "port": args.port, "n1": args.n1, "n2": args.n2,
                "reps": args.reps, "t1_s": t1, "t2_s": t2,
                "spread_t1_pct": s1, "spread_t2_pct": s2, "tpot_ms": tpot,
            }, fh, indent=2)
        print(f"[{args.label}] 写入 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
