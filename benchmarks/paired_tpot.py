"""Paired A/B **inside one server process**: flip the switch between requests, not between runs.

§27: restarting the server per arm has a 45% noise floor (same config measured 188.60 and 130.16
ms/token), because "which arm came first" dominates.  Pairing removes it: both modes see the same
device state, the same clock state and the same warm-up; only the code path differs.

Each measurement is the differential ruler from ``wall_tpot.py`` (two generation lengths on one
prompt, so prompt/first-token/per-request costs cancel).  The statistic that matters is the
**paired difference** per pair, and its spread — not the spread of the raw medians.

Requires the server to run with ``TILEINFER_FLAG_WATCH=1`` and ``TILEINFER_FLAG_DIR`` on a path this
script can write (it runs on the same host).
"""

import json
import statistics
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8111"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "qwen36"

N1, N2 = 8, 72
PAIRS = 4
PROMPT = "Explain in one sentence why the sky is blue:"
SWITCHES = tuple(
    name for name in (s.strip() for s in
                      (sys.argv[3] if len(sys.argv) > 3 else "moe_sorted,gdn_cache").split(","))
    if name
)


def set_mode(on: bool) -> None:
    from tileinfer.serving.flags import set_switch

    for name in SWITCHES:
        set_switch(name, on)


def generate(max_tokens: int) -> float:
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
    produced = (payload.get("usage") or {}).get("completion_tokens")
    if produced != max_tokens:
        raise SystemExit(f"asked for {max_tokens} tokens, server returned {produced}")
    return elapsed


def measure() -> float:
    """One differential measurement; refuses to return a physically impossible number.

    ``t(N2) < t(N1)`` cannot happen if both requests ran the same path, so it means the sample was
    corrupted (another workload on the device, a retried request, a stalled chunk).  §34 shipped a
    pair with ``off = -1378 ms/token`` and it dragged the reported ratio from ~1.06 to 1.028; a bad
    number must be rejected here rather than filtered by eye afterwards.
    """
    t1 = generate(N1)
    t2 = generate(N2)
    tpot = (t2 - t1) / (N2 - N1)
    if tpot <= 0:
        raise ValueError(
            f"impossible sample: t({N2})={t2:.3f}s <= t({N1})={t1:.3f}s -> {tpot * 1e3:.1f} ms/token"
        )
    return tpot


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
    set_mode(True)
    generate(N1)
    generate(N2)  # warm both modes' code paths
    set_mode(False)
    generate(N1)
    generate(N2)

    samples = {"on": [], "off": []}
    diffs = []
    rejected = 0
    pair = 0
    while len(diffs) < PAIRS and pair < PAIRS * 3:
        order = ("on", "off") if pair % 2 == 0 else ("off", "on")
        pair += 1
        got = {}
        try:
            for mode in order:
                set_mode(mode == "on")
                got[mode] = measure()
        except ValueError as exc:
            rejected += 1
            print(f"[pair{pair - 1}] REJECTED: {exc}", flush=True)
            continue
        samples["on"].append(got["on"])
        samples["off"].append(got["off"])
        diffs.append(got["off"] - got["on"])
        print(
            f"[pair{pair - 1}] order={order[0]}->{order[1]}  on={got['on'] * 1e3:.1f}  "
            f"off={got['off'] * 1e3:.1f}  (off-on)={diffs[-1] * 1e3:+.1f} ms/token",
            flush=True,
        )

    if not diffs:
        print("### no valid pair survived; nothing to report")
        return 1

    on_med = statistics.median(samples["on"])
    off_med = statistics.median(samples["off"])
    d_med = statistics.median(diffs)
    d_spread = max(diffs) - min(diffs)
    print(
        f"### valid pairs {len(diffs)}, rejected {rejected}\n"
        f"### on median {on_med * 1e3:.1f} ms/token   off median {off_med * 1e3:.1f} ms/token\n"
        f"### paired difference median {d_med * 1e3:+.1f} ms/token "
        f"(= {d_med / on_med * 100:+.1f}% of the on-arm), spread {d_spread * 1e3:.1f} ms "
        f"(= {d_spread / on_med * 100:.1f}%)\n"
        f"### measured_ratio on/off = {off_med / on_med:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
