"""Direct TTFT measurement: streaming time to the first non-empty chunk, three lengths, median.

``bench_arm.py`` is the historical harness but it runs several sizes internally and its output is
easy to lose in a pipeline (§55: two runs produced nothing).  This does one thing: for each prompt
length, N repetitions, report the median first-chunk time.  Streaming is the *right* instrument for
TTFT on this backend (§16: only the per-token deltas from streaming were ever wrong).
"""

import json
import statistics
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8111"
LABEL = sys.argv[2] if len(sys.argv) > 2 else "arm"
MODEL = sys.argv[3] if len(sys.argv) > 3 else "qwen36"
LENGTHS = (16, 128, 512)
REPS = 3

FILLER = (
    "The quick brown fox jumps over the lazy dog while the engineer studies the attention kernel "
    "and the scheduler decides which request to run next. "
)


def prompt_for(tokens: int) -> str:
    chars = max(8, tokens * 4)
    return "Summarize in one word: " + (FILLER * (chars // len(FILLER) + 1))[:chars]


def ttft(prompt: str, max_tokens: int = 8) -> float:
    body = json.dumps(
        {"model": MODEL, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0, "stream": True}
    ).encode()
    req = urllib.request.Request(
        f"{BASE}/v1/completions", data=body, headers={"Content-Type": "application/json"}
    )
    start = time.perf_counter()
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
            if choices and choices[0].get("text"):
                return time.perf_counter() - start
    raise ValueError("no non-empty chunk")


def wait_ready(timeout: float = 900.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{BASE}/v1/models", timeout=5) as resp:
                if resp.status == 200:
                    return
        except Exception:
            pass
        time.sleep(5)
    raise SystemExit(f"server never became ready at {BASE}")


def main() -> int:
    wait_ready()
    ttft(prompt_for(16))  # warm
    rows = []
    for tokens in LENGTHS:
        text = prompt_for(tokens)
        samples = [ttft(text) for _ in range(REPS)]
        med = statistics.median(samples)
        rows.append({"prompt_tokens_req": tokens, "ttft_median_s": round(med, 4)})
        print(
            f"[{LABEL}] prompt~{tokens:3d}: TTFT {med:.4f}s  "
            f"(samples {', '.join(f'{s:.3f}' for s in samples)})",
            flush=True,
        )
    with open(f"/tmp/ttft_{LABEL}.json", "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
