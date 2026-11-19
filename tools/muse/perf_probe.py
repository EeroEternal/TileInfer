#!/usr/bin/env python3
"""MuseGlimmer C4 probe: streaming TTFT / TPOT / throughput, single and concurrent.

    python tools/muse/perf_probe.py --base-url http://127.0.0.1:8360 \
        --prompt-tokens 956 --max-tokens 64 --runs 2 --json-out /tmp/muse-c4.json

    python tools/muse/perf_probe.py --base-url http://127.0.0.1:8360 \
        --prompt-tokens 956 --max-tokens 64 --concurrency 8

The prompt is repeated text; the reported ``prompt_tokens`` is the server's tokenizer count
from ``usage``, not the character length.  ``ttft`` is measured to the first non-empty
content/reasoning delta; ``tpot`` divides the remaining wall time by the generated tokens.
No latency is inferred from the completion length.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE_TEXT = (
    "The following is background material about paged attention and KV cache management. "
    "It repeats on purpose so the prompt can reach a target token count without changing "
    "the task. "
)


def prompt_of(char_target: int) -> str:
    reps = max(1, char_target // len(BASE_TEXT))
    return BASE_TEXT * reps + "\nSummarise the repeated topic in one sentence."


def stream_once(base: str, model: str, prompt: str, max_tokens: int, timeout: float = 900.0) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ttft = None
    first_t = None
    last_t = None
    usage = None
    chunks = 0
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            body = line[6:]
            if body == "[DONE]":
                break
            chunk = json.loads(body)
            if chunk.get("usage"):
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            text = (
                delta.get("content")
                or delta.get("reasoning_content")
                or delta.get("reasoning")
                or ""
            )
            if text:
                now = time.perf_counter()
                if ttft is None:
                    ttft = now - t0
                    first_t = now
                last_t = now
                chunks += 1
    end = time.perf_counter()
    completion = (usage or {}).get("completion_tokens") or chunks
    prompt_tokens = (usage or {}).get("prompt_tokens")
    tpot = None
    if ttft is not None and first_t is not None and last_t is not None and completion > 1:
        tpot = (last_t - first_t) / (completion - 1)
    return {
        "ttft_s": ttft,
        "tpot_s": tpot,
        "wall_s": end - t0,
        "completion_tokens": completion,
        "prompt_tokens": prompt_tokens,
        "chunks": chunks,
    }


def _median(values):
    values = [v for v in values if v is not None]
    return round(statistics.median(values), 4) if values else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8360")
    parser.add_argument("--model", default="muse")
    parser.add_argument("--prompt-tokens", type=int, default=956)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    prompt = prompt_of(args.prompt_tokens * 4)
    report = {
        "base_url": args.base_url,
        "model": args.model,
        "target_prompt_tokens": args.prompt_tokens,
        "max_tokens": args.max_tokens,
        "concurrency": args.concurrency,
    }

    if args.concurrency <= 1:
        runs = [
            stream_once(args.base_url, args.model, prompt, args.max_tokens)
            for _ in range(max(1, args.runs))
        ]
        report["runs"] = runs
        report["prompt_tokens"] = runs[-1]["prompt_tokens"]
        report["ttft_s_median"] = _median([r["ttft_s"] for r in runs])
        report["tpot_s_median"] = _median([r["tpot_s"] for r in runs])
        report["tok_s_median"] = _median(
            [r["completion_tokens"] / r["wall_s"] for r in runs if r["wall_s"]]
        )
    else:
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [
                pool.submit(stream_once, args.base_url, args.model, prompt, args.max_tokens)
                for _ in range(args.concurrency)
            ]
            runs = [f.result() for f in futures]
        wall = time.perf_counter() - t0
        total = sum(r["completion_tokens"] for r in runs)
        report["runs"] = runs
        report["prompt_tokens"] = runs[-1]["prompt_tokens"]
        report["wall_s"] = round(wall, 4)
        report["total_completion_tokens"] = total
        report["tok_s"] = round(total / wall, 2) if wall else None
        report["max_ttft_s"] = round(max(r["ttft_s"] for r in runs if r["ttft_s"] is not None), 4)
        report["completed"] = sum(1 for r in runs if r["completion_tokens"] >= args.max_tokens)

    text = json.dumps(report, ensure_ascii=False, indent=1)
    print(text)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
