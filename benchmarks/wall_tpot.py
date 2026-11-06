"""Wall-clock per-token cost with ~1-2% resolution, by differencing two generation lengths.

Why not the obvious measurement
-------------------------------
Streaming chunk deltas disagreed with wall clock for this backend (0.138-0.166 s/token from chunks
vs 0.082 s/token end to end, §16), and ``(total - ttft) / n`` mixes in the scheduling and first-token
cost.  Both were used as TPOT at some point in this project, and neither survives being checked
against a second method.

The robust form cancels everything that is not prefill-free decode:

    tpot = (T(N2) - T(N1)) / (N2 - N1)

with the *same* prompt for both requests, so the prompt cost, the first-token cost, the per-request
overhead and the client's own latency all appear in both terms and drop out.  What is left is the
marginal cost of one more generated token — which is exactly TPOT.

Repeat ``REPS`` times and report the median plus the spread: the spread is the resolution, and
without it a "3% improvement" claim cannot be told apart from noise.
"""

import json
import statistics
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8111"
LABEL = sys.argv[2] if len(sys.argv) > 2 else "arm"
MODEL = sys.argv[3] if len(sys.argv) > 3 else "qwen36"

N1 = 8
N2 = 72
REPS = 3
PROMPT = "Explain in one sentence why the sky is blue:"


def generate(max_tokens: int) -> float:
    """One non-streaming completion; returns wall-clock seconds."""
    body = json.dumps(
        {
            "model": MODEL,
            "prompt": PROMPT,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": False,
            "ignore_eos": True,
        }
    ).encode()
    req = urllib.request.Request(
        f"{BASE}/v1/completions", data=body, headers={"Content-Type": "application/json"}
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as resp:
        payload = json.loads(resp.read())
    elapsed = time.perf_counter() - start
    usage = payload.get("usage") or {}
    produced = usage.get("completion_tokens")
    if produced != max_tokens:
        raise SystemExit(
            f"[{LABEL}] asked for {max_tokens} tokens but the server returned {produced}; "
            "the slope would be wrong"
        )
    return elapsed


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
    raise SystemExit(f"[{LABEL}] server never became ready at {BASE}")


def main() -> int:
    wait_ready()
    generate(N1)  # warm: plan lookup, allocator, kernels
    generate(N2)
    samples = []
    for rep in range(REPS):
        t1 = generate(N1)
        t2 = generate(N2)
        samples.append((t2 - t1) / (N2 - N1))
        print(
            f"[{LABEL}] rep{rep}: t({N1})={t1:.4f}s t({N2})={t2:.4f}s "
            f"-> {samples[-1] * 1e3:.2f} ms/token",
            flush=True,
        )
    med = statistics.median(samples)
    spread = (max(samples) - min(samples)) / med
    print(
        f"[{LABEL}] TPOT median {med * 1e3:.2f} ms/token  ({1 / med:.2f} tok/s)  "
        f"spread {spread * 100:.1f}% (resolution ≈ {spread * 100 / 2:.1f}%)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
