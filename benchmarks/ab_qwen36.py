#!/usr/bin/env python
"""Qwen3.6-35B-A3B(vLLM on Ascend950PR)TileInfer vs FIA 的 A/B 基准。

方法(skills/vllm-AB.md):
- 一次只切换 attention backend,服务端配置逐项相同;同一请求集;
- 指标:首 token 延迟(TTFT)、每 token 延迟(TPOT,P50/P99)、128-token 生成吞吐、并发吞吐;
- 每组 3 次取中位数;另取首 token top-5 logprobs 做数值一致性对照;
- 原始数字落盘 JSON,供 PERFORMANCE_REPORT.md 引用。

用法: python ab_qwen36.py --port 8321 --label fia --outdir .agent_loop/run/ab_fia
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

PROMPT = (
    "用三句话介绍分页注意力机制在推理服务中的作用,要求提到 KV cache、"
    "页表和显存碎片。"
)
LONG_PROMPT = (
    "以下是三段关于_tile_level 计算的背景材料。"
) * 60 + "\n请概括材料中反复出现的核心思想,一句话即可。"


def post(url: str, payload: dict, timeout: float = 600.0) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def gen_once(base: str, model: str, max_tokens: int, stream: bool, max_time: float = 600.0):
    """一次生成;stream=True 时返回 (ttft, tpot 列表, 文本, completion_tokens)。"""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": stream,
    }
    if stream:
        payload["stream_options"] = {"include_usage": true_usage()}
    t0 = time.perf_counter()
    if not stream:
        d = post(base + "/v1/chat/completions", payload, timeout=max_time)
        wall = time.perf_counter() - t0
        n = d["usage"]["completion_tokens"]
        return {"ttft": wall, "tpot": wall / max(1, n - 1), "tokens": n, "text_len": len(d["choices"][0]["message"]["content"])}
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    ttft = None
    toks = []
    with urllib.request.urlopen(req, timeout=max_time) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            body = line[6:]
            if body == "[DONE]":
                break
            chunk = json.loads(body)
            if chunk.get("usage"):
                total = chunk["usage"]["completion_tokens"]
                continue
            delta = chunk["choices"][0].get("delta", {}).get("content")
            now = time.perf_counter()
            if delta:
                if ttft is None:
                    ttft = now - t0
                toks.append(now)
    last = toks[-1] if toks else time.perf_counter()
    tpots = [
        toks[i] - toks[i - 1] for i in range(1, len(toks)) if toks[i] - toks[i - 1] > 0
    ]
    return {
        "ttft": ttft if ttft is not None else last - t0,
        "tpot": statistics.median(tpots) if tpots else (last - t0),
        "tpots": tpots,
        "tokens": len(toks),
        "gen_wall": last - t0,
    }


def true_usage():
    return True


def top_logprobs(base: str, model: str) -> list:
    d = post(
        base + "/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": True,
            "top_logprobs": 5,
        },
    )
    items = (d["choices"][0].get("logprobs") or {}).get("content") or []
    if not items:
        return []
    return [(t["token"], round(t["logprob"], 4)) for t in (items[0].get("top_logprobs") or [])]


def bench_concurrency(base: str, model: str, conc: int, max_tokens: int = 128) -> dict:
    def one(_):
        return gen_once(base, model, max_tokens, stream=True)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=conc) as ex:
        results = list(ex.map(one, range(conc)))
    wall = time.perf_counter() - t0
    total = sum(r["tokens"] for r in results)
    return {
        "conc": conc,
        "wall": round(wall, 3),
        "tokens": total,
        "throughput": round(total / wall, 2),
        "ttfts": [round(r["ttft"] * 1000, 1) for r in results],
        "tpot_ms_median": round(statistics.median([r["tpot"] for r in results]) * 1000, 2),
    }


def percentile(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1)))))
    return xs[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8321)
    ap.add_argument("--model", default="Qwen3.6-35B-A3B")
    ap.add_argument("--label", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--conc", type=int, nargs="+", default=[1, 4])
    args = ap.parse_args()

    import os

    os.makedirs(args.outdir, exist_ok=True)
    base = f"http://127.0.0.1:{args.port}"

    # 0) 就绪探测
    for _ in range(240):
        try:
            post(base + "/v1/chat/completions", {"model": args.model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}, timeout=30)
            break
        except Exception:
            time.sleep(5)
    else:
        raise SystemExit("server not ready")

    report = {"label": args.label, "model": args.model, "port": args.port, "runs": args.runs}

    # 1) 数值对照:首 token top-5 logprobs(每次运行都记,取第一次做 A/B 对照)
    report["top5_logprobs"] = top_logprobs(base, args.model)

    # 2) 单流:TTFT(max_tokens=1)与 128-token 生成(TPOT 取实测 decode 墙钟/token),3 次取中位数
    ttfts, tpot_runs, tputs = [], [], []
    for r in range(args.runs):
        m = gen_once(base, args.model, 1, stream=False)
        ttfts.append(m["ttft"] * 1000)
        g = gen_once(base, args.model, 128, stream=True)
        decode_wall = g["gen_wall"] - g["ttft"]
        tpot_runs.append(decode_wall / max(1, g["tokens"] - 1) * 1000)
        tputs.append((g["tokens"] - 1) / decode_wall if decode_wall > 0 else 0.0)
        time.sleep(1)
    report["single_stream"] = {
        "ttft_ms_median": round(statistics.median(ttfts), 1),
        "ttft_ms_all": [round(x, 1) for x in ttfts],
        "tpot_ms_median": round(statistics.median(tpot_runs), 2),
        "tpot_ms_all": [round(x, 2) for x in tpot_runs],
        "tokps_median": round(statistics.median(tputs), 2),
    }

    # 3) 并发吞吐
    report["concurrency"] = [
        bench_concurrency(base, args.model, c) for c in args.conc
    ]

    out = os.path.join(args.outdir, f"ab_{args.label}.json")
    with open(out, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
