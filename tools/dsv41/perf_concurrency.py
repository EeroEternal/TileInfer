"""并发（--max-num-seqs >= 2）验证：正确性 + 每请求 TPOT（差分法）。

背景（docs/notes/dsv41-perf-plan.md §13-§16）：本栈此前把整批当成单序列，`--max-num-seqs >= 2`
必然错乱。P3 接线后用 `VLLM_DSV41_MULTI_REQ=1` 启用按请求切分。

口径：
  * 同时发 K 个请求（线程），每个请求 `max_tokens=N`；
  * 先测 N1 的墙钟 T1，再测 N2 的 T2；并发下 **每步耗时 = (T2 − T1) / (N2 − N1)**，
    因为 K 个请求是并行推进的，墙钟只随步数增长；
  * ≥3 次重复取中位数，报极差；
  * 正确性：每个请求的输出都必须包含 `Paris`（与单请求的 `Paris` 门一致）。

用法：
  python tools/dsv41/perf_concurrency.py --port 8122 --conc 4 --n1 8 --n2 40 --reps 3
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
import urllib.request

PROMPT = "The capital of France is"


def one(port: int, max_tokens: int, out: list, timeout: float = 900.0):
    body = json.dumps({
        "model": "dsv41",
        "prompt": PROMPT,
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read())
        out.append(payload)
    except Exception as exc:  # noqa: BLE001
        out.append({"error": f"{type(exc).__name__}: {exc}"})


def round_trip(port: int, conc: int, max_tokens: int) -> tuple[float, list]:
    outs: list = []
    threads = [threading.Thread(target=one, args=(port, max_tokens, outs)) for _ in range(conc)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return time.perf_counter() - t0, outs


def wait_ready(port: int, timeout_s: int = 900) -> bool:
    """轮询 ``/health`` 直到 200（vLLM 启动期间返回 503）。

    没有这一步，脚本会在服务还在加载时发请求，拿到 503/连接错误后误判为"测量失败"——
    这是 2026-11-05 实测踩到的坑。
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:  # noqa: BLE001 - 未就绪的各种表现都按"继续等"处理
            pass
        time.sleep(10)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8122)
    ap.add_argument("--conc", type=int, default=4)
    ap.add_argument("--n1", type=int, default=8)
    ap.add_argument("--n2", type=int, default=40)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--wait-ready", type=int, default=0,
                    help="最多等这么多秒直到 /health 返回 200（0 = 不等）")
    args = ap.parse_args()
    if args.n2 <= args.n1:
        print("n2 必须大于 n1")
        return 2
    if args.wait_ready:
        print(f"等待服务就绪（最多 {args.wait_ready}s）…")
        if not wait_ready(args.port, args.wait_ready):
            print("服务未在超时内就绪")
            return 4

    print(f"并发 {args.conc}，N1={args.n1} N2={args.n2} reps={args.reps}")
    t1s, t2s, texts = [], [], []
    for _ in range(args.reps):
        t1, out1 = round_trip(args.port, args.conc, args.n1)
        t2, out2 = round_trip(args.port, args.conc, args.n2)
        t1s.append(t1)
        t2s.append(t2)
        texts.extend(o.get("choices", [{}])[0].get("text", "") for o in out2 if "error" not in o)
        errs = [o["error"] for o in (out1 + out2) if "error" in o]
        if errs:
            print(f"请求失败: {errs[0][:120]}")
            return 1

    m1, m2 = statistics.median(t1s), statistics.median(t2s)
    spread1 = (max(t1s) - min(t1s)) / m1 * 100
    spread2 = (max(t2s) - min(t2s)) / m2 * 100
    per_step = (m2 - m1) / (args.n2 - args.n1) * 1000.0
    n_ok = sum(1 for t in texts if "Paris" in t)
    print(f"T({args.n1})={m1:.3f}s (极差 {spread1:.1f}%)  T({args.n2})={m2:.3f}s (极差 {spread2:.1f}%)")
    print(f"并发每步耗时 = ({m2:.3f} − {m1:.3f}) / ({args.n2} − {args.n1}) = {per_step:.1f} ms")
    print(f"正确性: {n_ok}/{len(texts)} 个请求输出含 Paris")
    if n_ok != len(texts):
        print("样例:", [t[:30] for t in texts[:4]])
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
