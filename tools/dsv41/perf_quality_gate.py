#!/usr/bin/env python3
"""质量门：100 prompt 贪心一致率（批量 decode vs 已验证的单请求串行路径）。

官方 ``inference/`` 在 CPU 上约 140 s/token（facts §5），100 prompt × 多 token 不可行；
S3/S4 已把单请求串行路径逐位对拍过官方（首 token 11111、S2 整块 ≤1e-2），所以这里用它作
**代理 oracle**：同一服务、同一权重、同一启动参数，只切 ``VLLM_DSV41_BATCHED_FWD``。

用法：
    # 批量臂（服务端 BATCHED_FWD=1，并发 4）
    python tools/dsv41/perf_quality_gate.py --port 8123 --conc 4 --out /tmp/q_batched.json
    # 串行臂（服务端 BATCHED_FWD=0，并发 1）
    python tools/dsv41/perf_quality_gate.py --port 8123 --conc 1 --out /tmp/q_serial.json
    python tools/dsv41/perf_quality_gate.py --compare /tmp/q_batched.json /tmp/q_serial.json

判据：完整 token 序列一致率 ≥95%，且逐位置一致率 ≥95%。
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

PROMPTS = [
    "The capital of France is",
    "The capital of Japan is",
    "The capital of Italy is",
    "The capital of Germany is",
    "The capital of Spain is",
    "The capital of China is",
    "The capital of Brazil is",
    "The capital of Canada is",
    "The capital of Australia is",
    "The capital of Egypt is",
    "Water boils at a temperature of",
    "The largest planet in the solar system is",
    "The author of Romeo and Juliet is",
    "The chemical symbol for gold is",
    "The speed of light is approximately",
    "The first president of the United States was",
    "The longest river in the world is",
    "The tallest mountain on Earth is",
    "The human heart has",
    "Photosynthesis converts sunlight into",
    "A triangle has",
    "The square root of 144 is",
    "Ten minus three equals",
    "Seven plus eight equals",
    "Twelve divided by four equals",
    "What is 17 multiplied by 23?",
    "What is 99 plus 1?",
    "What is 15 percent of 200?",
    "The next number in 2, 4, 8, 16 is",
    "The next number in 1, 1, 2, 3, 5 is",
    "If x + 5 = 12, then x is",
    "The derivative of x squared is",
    "The integral of 1/x is",
    "A prime number greater than 10 is",
    "The factorial of five is",
    "In Python, the keyword to define a function is",
    "In Python, a list is written with",
    "In Python, the result of len([1, 2, 3]) is",
    "In JavaScript, the keyword for a constant is",
    "In C, the entry point of a program is",
    "A SQL query that returns all rows uses",
    "The HTML tag for a paragraph is",
    "The Git command to create a branch is",
    "The Linux command to list files is",
    "An HTTP status code for not found is",
    "JSON stands for",
    "A binary search has time complexity",
    "A hash table lookup is typically",
    "The output of print(2 ** 10) is",
    "The output of print('ab' * 2) is",
    "The output of print(sorted([3, 1, 2])) is",
    "The output of print(len('hello')) is",
    "The output of print(bool(0)) is",
    "The output of print(1 == 1.0) is",
    "A variable in Python can be declared with",
    "The pandas function to read a CSV is",
    "The numpy function to create a zero array is",
    "The torch function to multiply matrices is",
    "The git command to see the diff is",
    "The docker command to list containers is",
    "The shell command to change directory is",
    "The regex symbol for any character is",
    "The difference between TCP and UDP is",
    "The purpose of DNS is",
    "The purpose of TLS is",
    "Machine learning is a field of",
    "A neural network is composed of",
    "Gradient descent minimizes",
    "Overfitting means the model",
    "A transformer uses attention to",
    "Reinforcement learning trains an agent by",
    "The bias-variance tradeoff describes",
    "Data normalization means",
    "The main advantage of batch processing is",
    "The main cost of large language model inference is",
    "Quantization reduces the model's",
    "KV cache stores",
    "A GPU differs from a CPU in",
    "The purpose of a memory page cache is",
    "The difference between latency and throughput is",
    "The purpose of a lock in concurrent programming is",
    "A deadlock occurs when",
    "The difference between a process and a thread is",
    "The purpose of a compiler is",
    "The purpose of an operating system is",
    "The capital of South Korea is",
    "The capital of India is",
    "The capital of Russia is",
    "The capital of Mexico is",
    "The capital of Turkey is",
    "The capital of Thailand is",
    "The capital of Vietnam is",
    "The capital of Argentina is",
    "The capital of Kenya is",
    "中国的首都是",
    "日本的首都是",
    "法国的首都是",
    "德国的首都是",
    "太阳系中最大的行星是",
    "勾股定理指出直角三角形的",
]


def one(port: int, prompt: str, max_tokens: int) -> dict:
    body = json.dumps(
        {
            "model": "dsv41",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "ignore_eos": True,
            "return_token_ids": True,
            "logprobs": 5,
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=900) as resp:
            payload = json.loads(resp.read())
        choice = payload["choices"][0]
        tokens = choice.get("token_ids")
        if tokens is None:
            return {"prompt": prompt, "error": "token_ids missing", "text": choice.get("text")}
        return {
            "prompt": prompt,
            "token_ids": [int(t) for t in tokens],
            "text": choice.get("text", ""),
            "logprobs": choice.get("logprobs"),
            "elapsed": time.perf_counter() - t0,
        }
    except Exception as exc:  # noqa: BLE001
        return {"prompt": prompt, "error": f"{type(exc).__name__}: {exc}"}


def wait_ready(port: int, timeout_s: int) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(5)
    return False


def run(args) -> int:
    if not wait_ready(args.port, args.wait_ready):
        print("service not ready")
        return 2
    prompts = PROMPTS[: args.limit] if args.limit else PROMPTS
    print(f"prompts={len(prompts)} conc={args.conc} max_tokens={args.max_tokens}")
    done = 0
    lock = threading.Lock()
    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.conc)) as pool:
        futures = [pool.submit(one, args.port, p, args.max_tokens) for p in prompts]
        for fut in as_completed(futures):
            row = fut.result()
            results.append(row)
            with lock:
                done += 1
                if done % 10 == 0 or done == len(prompts):
                    print(f"  {done}/{len(prompts)}", flush=True)
    order = {p: i for i, p in enumerate(prompts)}
    results.sort(key=lambda r: order[r["prompt"]])
    payload = {
        "port": args.port,
        "conc": args.conc,
        "max_tokens": args.max_tokens,
        "results": results,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    errors = [r for r in results if "error" in r]
    print(f"wrote {args.out}; errors={len(errors)}")
    if errors:
        print(errors[0])
        return 1
    return 0


def compare(left_path: str, right_path: str, threshold: float, delta: float = 1.0) -> int:
    left = json.load(open(left_path, encoding="utf-8"))
    right = json.load(open(right_path, encoding="utf-8"))
    lrows = {r["prompt"]: r for r in left["results"]}
    rrows = {r["prompt"]: r for r in right["results"]}
    prompts = [p for p in lrows if p in rrows]
    if not prompts:
        print("no overlapping prompts")
        return 2
    full_ok = 0
    pos_ok = pos_total = 0
    bounded_ok = 0
    gaps = []
    bad = []
    for p in prompts:
        a = lrows[p].get("token_ids") or []
        b = rrows[p].get("token_ids") or []
        n = min(len(a), len(b))
        same_pos = sum(1 for i in range(n) if a[i] == b[i])
        pos_ok += same_pos
        pos_total += max(len(a), len(b), 1)
        for i in range(n):
            if a[i] == b[i]:
                bounded_ok += 1
                continue
            # logprob 有界判据：右臂 token 在左臂 top-5 内，且与左臂 chosen 的 logprob 差 ≤ δ
            lr = lrows[p].get("logprobs") or {}
            rr = rrows[p].get("logprobs") or {}
            try:
                st = rr["tokens"][i]
                bt = lr["top_logprobs"][i]
                chosen = lr["token_logprobs"][i]
                gap = abs(bt[st] - chosen) if st in bt else None
            except Exception:  # noqa: BLE001
                gap = None
            if gap is not None:
                gaps.append(gap)
                if gap <= delta:
                    bounded_ok += 1
        if a == b and a:
            full_ok += 1
        else:
            bad.append((p, a, b))
    rate = full_ok / len(prompts)
    pos_rate = pos_ok / max(pos_total, 1)
    bounded_rate = bounded_ok / max(pos_total, 1)
    print(f"prompts={len(prompts)} full_sequence_agreement={rate:.3f} ({full_ok}/{len(prompts)})")
    print(f"position_agreement={pos_rate:.3f} ({pos_ok}/{pos_total})")
    print(
        f"logprob_bound delta={delta:.2f}: bounded_agreement={bounded_rate:.3f} "
        f"({bounded_ok}/{pos_total}); divergent positions in top-5={len(gaps)}"
    )
    if gaps:
        gaps_sorted = sorted(gaps)
        print(
            f"  gap median={gaps_sorted[len(gaps_sorted)//2]:.3f} "
            f"p90={gaps_sorted[int(0.9*(len(gaps_sorted)-1))]:.3f} max={gaps_sorted[-1]:.3f}"
        )
    for p, a, b in bad[:10]:
        print(f"  MISMATCH {p!r}\n    batched={a}\n    serial ={b}")
    ok = rate >= threshold and pos_rate >= threshold
    print(f"RESULT {'PASS' if ok else 'FAIL'} (threshold {threshold:.0%})")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--conc", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（调试）")
    ap.add_argument("--wait-ready", type=int, default=120)
    ap.add_argument("--out", default="/tmp/q.json")
    ap.add_argument("--compare", nargs=2, metavar=("LEFT", "RIGHT"))
    ap.add_argument("--threshold", type=float, default=0.95)
    ap.add_argument("--delta", type=float, default=1.0,
                    help="logprob 有界判据的 δ（nats）；默认 1.0，仓库 §52 的端到端上限为 0.21")
    args = ap.parse_args()
    if args.compare:
        return compare(args.compare[0], args.compare[1], args.threshold, args.delta)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
