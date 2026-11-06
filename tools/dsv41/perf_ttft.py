"""TTFT（首 token 延迟）与 decode 速度测量：走 vLLM 的流式接口。

为什么需要它（性能目标 P2 的验收项）：现有基线只有 TPOT（743 ms/token 的整段均值），
无法区分"prefill 慢"还是"decode 慢"。TTFT 用 `stream: true` 测得，是拆解二者的唯一手段。

口径（与 docs/notes/perf-comparison-rules.md 一致）：
  * 同一 venv、同一服务启动参数；
  * ≥3 次重复，报 P50/P90 与极差；
  * TTFT = 从发请求到收到**第一个 SSE chunk** 的墙钟时间；
  * decode 速率 = (总时长 − TTFT) / (生成 token 数 − 1)，即差分法思想（扣掉首 token）。

用法：
  python tools/dsv41/perf_ttft.py --port 8122 --reps 5
  python tools/dsv41/perf_ttft.py --prompt-file /tmp/long_prompt.txt --max-tokens 16
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request

DEFAULT_PROMPTS = {
    "short": "The capital of France is",
    "medium": "请用三句话介绍杭州。",
}


def build_prompt(text: str, target_tokens: int | None) -> str:
    """可选：把 prompt 重复到大约 target_tokens（粗估 3 字符 ≈ 1 token）。"""
    if not target_tokens:
        return text
    need = max(1, target_tokens * 3 // max(1, len(text)))
    return (text + " ") * need


def stream_once(port: int, prompt: str, max_tokens: int, timeout: float = 600.0,
                endpoint: str = "chat"):
    """发一次流式请求，返回 (ttft_s, total_s, chunks, text)。

    endpoint="chat" 走 /v1/chat/completions（含 chat 模板渲染）；
    endpoint="completions" 走 /v1/completions（纯 prompt，无模板），
    两者 TTFT 之差即模板渲染成本。
    """
    if endpoint == "completions":
        payload = {
            "model": "dsv41",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
        }
        url = f"http://127.0.0.1:{port}/v1/completions"
    else:
        payload = {
            "model": "dsv41",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
            "chat_template_kwargs": {"thinking_mode": "chat"},
        }
        url = f"http://127.0.0.1:{port}/v1/chat/completions"
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ttft = None
    chunks = 0
    parts: list[str] = []
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            choice = obj.get("choices", [{}])[0]
            if endpoint == "completions":
                piece = choice.get("text")
            else:
                piece = choice.get("delta", {}).get("content")
            if piece:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                chunks += 1
                parts.append(piece)
    total = time.perf_counter() - t0
    return (ttft if ttft is not None else float("nan")), total, chunks, "".join(parts)


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    k = max(0, min(len(xs) - 1, int(round((len(xs) - 1) * p))))
    return sorted(xs)[k]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8122)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--target-tokens", type=int, default=0,
                    help="把 prompt 撑到约这么多 token（测 prefill 随长度的变化）")
    ap.add_argument("--prompt-file", default="")
    ap.add_argument("--endpoint", default="chat", choices=["chat", "completions"])
    args = ap.parse_args()

    if args.prompt_file:
        prompts = {"file": open(args.prompt_file, encoding="utf-8").read()}
    else:
        prompts = DEFAULT_PROMPTS
    if args.target_tokens:
        prompts = {k: build_prompt(v, args.target_tokens) for k, v in prompts.items()}
        print(f"prompt 撑到约 {args.target_tokens} token（粗略）")

    print(f"endpoint=127.0.0.1:{args.port}/{args.endpoint}  reps={args.reps}  max_tokens={args.max_tokens}")
    rc = 0
    for name, prompt in prompts.items():
        ttfts, totals, rates = [], [], []
        for _ in range(args.reps):
            try:
                ttft, total, chunks, text = stream_once(args.port, prompt, args.max_tokens,
                                                 endpoint=args.endpoint)
            except Exception as exc:  # noqa: BLE001
                print(f"[{name}] 请求失败: {type(exc).__name__}: {exc}")
                rc = 1
                break
            if chunks < 1:
                continue                      # 一个块都没收到，样本无效
            # TTFT 只需要首块；decode 速率仅在 >1 块时可算（1 块时整段就含全部输出）
            ttfts.append(ttft)
            totals.append(total)
            if chunks > 1:
                rates.append((total - ttft) / (chunks - 1) * 1000.0)
        if not ttfts:
            print(f"[{name}] 无有效样本")
            continue
        spread = (max(ttfts) - min(ttfts)) / statistics.median(ttfts) * 100
        print(f"[{name}] TTFT  P50={pct(ttfts,0.5)*1000:.0f} ms  P90={pct(ttfts,0.9)*1000:.0f} ms  "
              f"极差={spread:.0f}%")
        if rates:
            print(f"[{name}] decode {statistics.median(rates):.0f} ms/chunk  "
                  f"（整段 {statistics.median(totals):.1f} s，chunks={chunks}）")
        else:
            print(f"[{name}] decode 不可算（chunks=1，整段 {statistics.median(totals):.1f} s）")
        print(f"[{name}] 首块文本: {text[:40]!r}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
