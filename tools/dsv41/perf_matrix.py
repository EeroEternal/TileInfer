"""End-to-end TTFT / TPOT / ITL matrix for the vLLM OpenAI endpoint.

Streaming request, one token per SSE chunk (vLLM's default).  For each request:

  TTFT = time to the first non-empty chunk
  ITL  = deltas between consecutive token chunks (P50/P90 reported)
  TPOT = (last_token_time - first_token_time) / (n_tokens - 1)

Concurrent runs start all requests from a barrier; each request gets its own TTFT/ITL.
"""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

SENTENCE = "The capital of France is Paris. "


def _percentile(vals, p):
    if not vals:
        return None
    s = sorted(vals)
    k = min(len(s) - 1, max(0, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]


def prompt_of(tokens: int) -> str:
    # ~7 tokens per sentence; the exact length is measured by /tokenize
    return SENTENCE * max(1, tokens // 7)


def tokenize(port: int, text: str) -> int:
    body = json.dumps({"model": "dsv41", "prompt": text}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/tokenize",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return len(json.loads(r.read())["tokens"])


def one(port: int, prompt: str, max_tokens: int, barrier=None) -> dict:
    body = json.dumps(
        {
            "model": "dsv41",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "ignore_eos": True,
            "stream": True,
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    if barrier is not None:
        barrier.wait()
    t0 = time.perf_counter()
    first = None
    times = []
    usage_tokens = None
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for raw in resp:
            if not raw.startswith(b"data: "):
                continue
            data = raw[6:].strip()
            if data == b"[DONE]":
                break
            obj = json.loads(data)
            if obj.get("usage"):
                usage_tokens = int(obj["usage"].get("completion_tokens") or 0)
            ch = obj.get("choices") or []
            if not ch:
                continue
            text = ch[0].get("text", "")
            if text:
                now = time.perf_counter()
                if first is None:
                    first = now
                times.append(now)
    if first is None:
        return {"error": "no content"}
    itls = [(times[i] - times[i - 1]) * 1e3 for i in range(1, len(times))]
    n = usage_tokens or len(times)
    tpot = (times[-1] - first) / (n - 1) * 1e3 if n > 1 else None
    return {
        "ttft_ms": (first - t0) * 1e3,
        "tpot_ms": tpot,
        "itl_p50_ms": statistics.median(itls) if itls else None,
        "itl_p90_ms": _percentile(itls, 90),
        "itl_max_ms": max(itls) if itls else None,
        "n_tokens": n,
        "chunks": len(times),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--conc", type=int, default=1)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    prompt = prompt_of(args.tokens)
    plen = tokenize(args.port, prompt)
    print(f"[{args.label}] prompt_tokens={plen} max_tokens={args.max_tokens} conc={args.conc} reps={args.reps}")

    all_rows = []
    for rep in range(args.reps):
        barrier = threading.Barrier(args.conc)
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.conc) as ex:
            rows = list(ex.map(lambda _: one(args.port, prompt, args.max_tokens, barrier), range(args.conc)))
        wall = (time.perf_counter() - t0) * 1e3
        for i, r in enumerate(rows):
            r["request"] = i
            r["rep"] = rep
        all_rows.extend(rows)
        ok = [r for r in rows if "error" not in r]
        if ok:
            def _m(key):
                v = statistics.median(r[key] for r in ok if r.get(key) is not None)
                return f"{v:.1f}" if v is not None else "NA"

            print(
                f"  rep{rep}: wall={wall:.0f}ms ttft_med={_m('ttft_ms')} "
                f"tpot_med={_m('tpot_ms')} itl_p50={_m('itl_p50_ms')} itl_p90={_m('itl_p90_ms')}"
            )
    ok = [r for r in all_rows if "error" not in r]
    summary = {
        "label": args.label,
        "port": args.port,
        "prompt_tokens": plen,
        "max_tokens": args.max_tokens,
        "conc": args.conc,
        "reps": args.reps,
        "ttft_ms_median": statistics.median(r["ttft_ms"] for r in ok) if ok else None,
        "tpot_ms_median": statistics.median(r["tpot_ms"] for r in ok) if ok else None,
        "itl_p50_ms_median": statistics.median(r["itl_p50_ms"] for r in ok) if ok else None,
        "itl_p90_ms_median": statistics.median(r["itl_p90_ms"] for r in ok) if ok else None,
        "itl_max_ms_median": statistics.median(r["itl_max_ms"] for r in ok) if ok else None,
        "wall_ms_median": statistics.median([r.get("wall", 0) for r in all_rows]) if all_rows else None,
        "rows": all_rows,
    }
    print(
        f"[{args.label}] TTFT={summary['ttft_ms_median']:.1f}ms "
        f"TPOT={summary['tpot_ms_median'] if summary['tpot_ms_median'] is not None else float('nan'):.1f}ms "
        f"ITL_P50={summary['itl_p50_ms_median'] if summary['itl_p50_ms_median'] is not None else float('nan'):.1f}ms "
        f"ITL_P90={summary['itl_p90_ms_median'] if summary['itl_p90_ms_median'] is not None else float('nan'):.1f}ms"
    )
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
