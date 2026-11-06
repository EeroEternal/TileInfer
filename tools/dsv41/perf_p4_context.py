#!/usr/bin/env python3
"""P4：长上下文（4K/16K/32K）单请求 TTFT + HBM/RSS 峰值。

不依赖 vLLM 内部接口：TTFT 走 OpenAI 流式 completions；峰值用后台采样
``npu-smi info``（HBM used）与 ``/proc/<VLLM::EngineCore>/status``（VmRSS）。

用法：
    python tools/dsv41/perf_p4_context.py --port 8123 --prompt-file /tmp/p4_32k.txt \
        --max-tokens 2 --out /tmp/p4_32k.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import threading
import time
import urllib.request


def _pids() -> tuple[int, int]:
    out = subprocess.run(["ps", "-eo", "pid,cmd"], capture_output=True, text=True).stdout
    api = engine = 0
    for line in out.splitlines():
        if "vllm.entrypoints.openai.api_server" in line and "--port" in line:
            api = int(line.split()[0])
        if "VLLM::EngineCore" in line or "VLLMEngineCor" in line:
            engine = int(line.split()[0])
    return api, engine


def _hbm_used_mb() -> int:
    out = subprocess.run(["npu-smi", "info"], capture_output=True, text=True).stdout
    m = re.search(r"(\d+)\s*/\s*131072", out)
    return int(m.group(1)) if m else 0


def _rss_mb(pid: int) -> int:
    try:
        with open(f"/proc/{pid}/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
    except Exception:  # noqa: BLE001
        pass
    return 0


class Sampler(threading.Thread):
    def __init__(self, engine_pid: int):
        super().__init__(daemon=True)
        self.engine_pid = engine_pid
        self.stop = threading.Event()
        self.hbm = 0
        self.rss = 0

    def run(self) -> None:
        while not self.stop.is_set():
            self.hbm = max(self.hbm, _hbm_used_mb())
            self.rss = max(self.rss, _rss_mb(self.engine_pid))
            time.sleep(1.0)


def ttft(port: int, prompt: str, max_tokens: int) -> tuple[float, float, int]:
    body = json.dumps({
        "model": "dsv41", "prompt": prompt, "max_tokens": max_tokens,
        "temperature": 0, "stream": True, "ignore_eos": True,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions", data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    first = None
    chunks = 0
    with urllib.request.urlopen(req, timeout=3600) as resp:
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
            if obj.get("choices", [{}])[0].get("text"):
                if first is None:
                    first = time.perf_counter() - t0
                chunks += 1
    return (first if first is not None else float("nan")), time.perf_counter() - t0, chunks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--max-tokens", type=int, default=2)
    ap.add_argument("--out", default="/tmp/p4.json")
    args = ap.parse_args()
    prompt = open(args.prompt_file, encoding="utf-8").read()
    api, engine = _pids()
    print(f"api={api} engine={engine} prompt_chars={len(prompt)}")
    s = Sampler(engine)
    s.start()
    t, total, chunks = ttft(args.port, prompt, args.max_tokens)
    s.stop.set()
    s.join(timeout=5)
    payload = {
        "prompt_file": args.prompt_file,
        "prompt_chars": len(prompt),
        "ttft_s": t,
        "total_s": total,
        "chunks": chunks,
        "hbm_peak_mb": s.hbm,
        "rss_peak_mb": s.rss,
    }
    print(json.dumps(payload, ensure_ascii=False))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    return 0 if t == t else 1


if __name__ == "__main__":
    raise SystemExit(main())
