"""Paired prefill measurement: marginal cost **per prompt token**, with the switch flipped in place.

Why a new ruler
---------------
``wall_tpot.py`` measures decode by differencing two *generation* lengths.  The GDN host-cache change
(§38) acts on **prefill** instead, so it needs the same trick on the prompt axis:

    prefill_ms_per_token = (T(P2) - T(P1)) / (P2 - P1)

with ``P1=128`` and ``P2=512``: the per-request overhead, the tokenizer, the scheduler hand-off and
the first-token cost are all present in both terms and cancel, leaving the marginal cost of one more
prompt token.  ``T`` is the streaming time to the first non-empty chunk — streaming is reliable for
*time-to-first-token* on this backend (§16 says so explicitly, and only its per-token deltas were
wrong).

Pairing: the switch is flipped between measurements **inside one server process** (§28: that raised
repeatability from 45% to 0.7%), and the order alternates so a monotone drift cannot look like an
effect.  A sample with ``T(P2) <= T(P1)`` is physically impossible and rejected (§35).
"""

import json
import statistics
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8111"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "qwen36"
SWITCHES = tuple(
    name.strip()
    for name in (sys.argv[3] if len(sys.argv) > 3 else "gdn_host_cache").split(",")
    if name.strip()
)

P1, P2 = 128, 512
PAIRS = 4
FILLER = (
    "The quick brown fox jumps over the lazy dog while the engineer studies the attention kernel "
    "and the scheduler decides which request to run next. "
)


def prompt_for(tokens: int) -> str:
    chars = max(8, tokens * 4)
    return "Summarize in one word: " + (FILLER * (chars // len(FILLER) + 1))[:chars]


def set_mode(on: bool) -> None:
    from tileinfer.serving.flags import set_switch

    for name in SWITCHES:
        set_switch(name, on)


def time_to_first_token(prompt: str) -> float:
    """Seconds until the first non-empty stream chunk (i.e. the prefill is done)."""
    body = json.dumps(
        {"model": MODEL, "prompt": prompt, "max_tokens": 4, "temperature": 0.0, "stream": True}
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
            if not choices:
                continue
            if choices[0].get("text"):
                return time.perf_counter() - start
    raise ValueError("stream produced no non-empty chunk")


def measure() -> float:
    t1 = time_to_first_token(prompt_for(P1))
    t2 = time_to_first_token(prompt_for(P2))
    per_token_ms = (t2 - t1) * 1000.0 / (P2 - P1)
    if per_token_ms <= 0:
        raise ValueError(
            f"impossible sample: T({P2})={t2:.3f}s <= T({P1})={t1:.3f}s -> {per_token_ms:.3f} ms/token"
        )
    return per_token_ms


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
    time_to_first_token(prompt_for(P1))
    time_to_first_token(prompt_for(P2))
    set_mode(False)
    time_to_first_token(prompt_for(P1))
    time_to_first_token(prompt_for(P2))

    on, off, diffs = [], [], []
    rejected = 0
    attempt = 0
    while len(diffs) < PAIRS and attempt < PAIRS * 3:
        order = ("on", "off") if attempt % 2 == 0 else ("off", "on")
        attempt += 1
        got = {}
        try:
            for mode in order:
                set_mode(mode == "on")
                got[mode] = measure()
        except ValueError as exc:
            rejected += 1
            print(f"[pair{attempt - 1}] REJECTED: {exc}", flush=True)
            continue
        on.append(got["on"])
        off.append(got["off"])
        diffs.append(got["off"] - got["on"])
        print(
            f"[pair{attempt - 1}] order={order[0]}->{order[1]}  on={got['on']:.3f}  "
            f"off={got['off']:.3f} ms/prompt-token  (off-on)={diffs[-1]:+.3f}",
            flush=True,
        )

    if not diffs:
        print("### no valid pair survived")
        return 1
    on_med, off_med = statistics.median(on), statistics.median(off)
    d_med = statistics.median(diffs)
    spread = max(diffs) - min(diffs)
    print(
        f"### switches={','.join(SWITCHES)}\n"
        f"### valid pairs {len(diffs)}, rejected {rejected}\n"
        f"### on median {on_med:.3f} ms/prompt-token   off median {off_med:.3f}\n"
        f"### paired difference median {d_med:+.3f} ms/prompt-token "
        f"(= {d_med / on_med * 100:+.1f}%), spread {spread:.3f} "
        f"(= {spread / on_med * 100:.1f}%)\n"
        f"### measured_ratio on/off = {off_med / on_med:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
