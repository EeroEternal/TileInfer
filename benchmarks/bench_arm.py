"""Streaming TTFT/TPOT bench for one OpenAI-compatible arm.  stdlib only."""
from __future__ import annotations

import json
import statistics
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8111"
LABEL = sys.argv[2] if len(sys.argv) > 2 else "arm"
OUT = sys.argv[3] if len(sys.argv) > 3 else "/tmp/bench.json"

FILLER = ("The quick brown fox jumps over the lazy dog while the engineer studies the "
          "attention kernel and the scheduler decides which request to run next. ")


def prompt_for(tokens: int) -> str:
    # ~4 chars/token for English; overshoot is fine, the server tokenizes.
    chars = max(8, tokens * 4)
    text = (FILLER * (chars // len(FILLER) + 1))[:chars]
    return "Summarize in one word: " + text


def wait_ready(timeout=900):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{BASE}/v1/models", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(5)
    return False


def run_one(prompt: str, max_tokens: int):
    body = json.dumps({
        "model": "qwen36",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }).encode()
    req = urllib.request.Request(
        f"{BASE}/v1/completions", data=body,
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    ttft = None
    chunks = 0
    last = None
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                event = json.loads(payload)
            except Exception:
                continue
            choices = event.get("choices") or []
            if not choices:
                continue
            text = choices[0].get("text") or ""
            if text or ttft is None:
                now = time.perf_counter()
                if ttft is None and text:
                    ttft = now - start
                if text:
                    chunks += 1
                last = now
    total = time.perf_counter() - start
    if ttft is None:
        ttft = total
    tpot = (total - ttft) / max(1, chunks - 1) if chunks > 1 else float("nan")
    return {"ttft_s": ttft, "total_s": total, "chunks": chunks, "tpot_s": tpot}


def main() -> int:
    if not wait_ready():
        print(f"[{LABEL}] server never became ready at {BASE}")
        return 1
    results = []
    for prompt_tokens, max_tokens in ((16, 32), (128, 32), (512, 32)):
        runs = []
        for _ in range(3):
            try:
                runs.append(run_one(prompt_for(prompt_tokens), max_tokens))
            except Exception as exc:
                print(f"[{LABEL}] run failed for {prompt_tokens}: {exc}")
        if not runs:
            continue
        row = {
            "label": LABEL,
            "prompt_tokens_req": prompt_tokens,
            "max_tokens": max_tokens,
            "ttft_median_s": round(statistics.median(r["ttft_s"] for r in runs), 4),
            "tpot_median_s": round(statistics.median(r["tpot_s"] for r in runs), 4),
            "total_median_s": round(statistics.median(r["total_s"] for r in runs), 4),
            "chunks_median": statistics.median(r["chunks"] for r in runs),
            "runs": runs,
        }
        results.append(row)
        print(f"[{LABEL}] prompt~{prompt_tokens} max_new={max_tokens}: "
              f"TTFT {row['ttft_median_s']}s  TPOT {row['tpot_median_s']}s  "
              f"total {row['total_median_s']}s  "
              f"({1.0 / row['tpot_median_s']:.2f} tok/s)" if row["tpot_median_s"] else "")
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"[{LABEL}] wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
