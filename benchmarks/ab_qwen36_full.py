#!/usr/bin/env python
"""Qwen3.6-35B-A3B(vLLM on Ascend950PR)TileInfer vs FIA 扩展 A/B 基准。

覆盖:单 token / 多 token(128、512)/ 并发 4-16-32 / 长上下文(8K、16K 输入)/ 16K 纯 prefill。
指标(对齐 vLLM 官方 benchmark_serving 口径):
  - TTFT P50/P99          首 token 延迟
  - TPOT P50/P99          每输出 token 延迟,官方定义 (request_latency - ttft)/(output_tokens-1)
  - ITL P50/P99           token 间间隔分布
  - 输出吞吐 tok/s        单流与并发聚合
  - 请求吞吐 req/s        并发场景
每组 3 次取中位数;另取首 token top-5 logprobs 做数值一致性对照。
配套官方工具:`vllm bench serve --dataset-name random --percentile-metrics ttft,tpot,itl`
双源交叉验证。

用法: python ab_qwen36_full.py --port 8321 --label fia_eager --outdir .agent_loop/run/ab_full_fia_eager
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

SHORT_PROMPT = (
    "用三句话介绍分页注意力机制在推理服务中的作用,要求提到 KV cache、页表和显存碎片。"
)
_LONG_BASE = (
    "背景材料:分页注意力机制把 KV cache 划分为固定大小的页,通过页表把逻辑位置映射到"
    "物理页,从而消除预分配连续显存带来的碎片问题;在解码阶段,内核按页表索引读取当前"
    "序列涉及的每一个物理页,并对末页做长度掩码;调度器在页不足时触发扩容或逐出。"
)


def make_long_prompt(target_tokens: int) -> str:
    """重复材料构造长 prompt;Qwen 系中文约 1 token/字,按字数近似,实际值以 usage 为准。"""
    chars = int(target_tokens * 1.15)
    reps = max(1, chars // len(_LONG_BASE))
    return _LONG_BASE * reps + "\n请概括材料中反复出现的核心思想,一句话即可。"


def post(url: str, payload: dict, timeout: float = 900.0) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def gen_stream(base: str, model: str, prompt: str, max_tokens: int):
    """流式生成一次;返回官方口径指标原料。

    ttft        首 token 延迟(秒)
    tpot_off    官方 TPOT = (总墙钟 - ttft) / (tokens-1)
    itls        token 间间隔列表(秒),即 ITL 分布样本
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        base + "/v1/chat/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ttft, toks, prompt_tokens = None, [], None
    with urllib.request.urlopen(req, timeout=900.0) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            body = line[6:]
            if body == "[DONE]":
                break
            chunk = json.loads(body)
            if chunk.get("usage"):
                prompt_tokens = (chunk["usage"] or {}).get("prompt_tokens")
                continue
            delta = (chunk.get("choices") or [{}])[0].get("delta", {}).get("content")
            now = time.perf_counter()
            if delta:
                if ttft is None:
                    ttft = now - t0
                toks.append(now)
    last = toks[-1] if toks else time.perf_counter()
    wall = last - t0
    itls = [toks[i] - toks[i - 1] for i in range(1, len(toks)) if toks[i] - toks[i - 1] > 0]
    tpot_off = (wall - (ttft or 0.0)) / max(1, len(toks) - 1) if len(toks) > 1 else wall
    return {
        "ttft": ttft if ttft is not None else wall,
        "tpot_off": tpot_off,
        "itls": itls,
        "tokens": len(toks),
        "wall": wall,
        "prompt_tokens": prompt_tokens,
    }


def pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1)))))
    return round(xs[k] * 1000, 2)  # -> ms


def med_ms(xs):
    return round(statistics.median(xs) * 1000, 2) if xs else None


def scenario_single(base, model, label, prompt, max_tokens, runs):
    """单流场景:跨次数聚合 TTFT/TPOT/ITL 的 P50/P99 + 输出吞吐。"""
    ttfts, tpots, itls, tputs, ptoks = [], [], [], [], None
    for _ in range(runs):
        r = gen_stream(base, model, prompt, max_tokens)
        ttfts.append(r["ttft"])
        tpots.append(r["tpot_off"])
        itls += r["itls"]
        if r["wall"] - (r["ttft"] or 0) > 0 and r["tokens"] > 1:
            tputs.append((r["tokens"] - 1) / (r["wall"] - r["ttft"]))
        ptoks = r["prompt_tokens"] or ptoks
        time.sleep(0.5)
    out = {
        "scenario": label, "prompt_tokens": ptoks, "max_tokens": max_tokens, "runs": runs,
        "ttft_ms_p50": pct(ttfts, 50), "ttft_ms_p99": pct(ttfts, 99),
        "tpot_ms_p50": pct(tpots, 50), "tpot_ms_p99": pct(tpots, 99),
        "itl_ms_p50": pct(itls, 50), "itl_ms_p99": pct(itls, 99),
        "out_toks_median": med_ms(tputs),
    }
    print(json.dumps(out, ensure_ascii=False), flush=True)
    return out


def scenario_conc(base, model, conc, max_tokens, runs):
    """并发场景:吞吐(输出 tok/s + 请求 req/s)+ 跨请求 TTFT/TPOT/ITL P50/P99。"""
    agg = {"scenario": f"conc_{conc}", "max_tokens": max_tokens, "runs": runs, "conc": conc}
    tputs, rputs, ttfts, tpots, itls = [], [], [], [], []
    for _ in range(runs):
        def one(_i):
            return gen_stream(base, model, SHORT_PROMPT, max_tokens)
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=conc) as ex:
            rs = list(ex.map(one, range(conc)))
        wall = time.perf_counter() - t0
        total = sum(r["tokens"] for r in rs)
        tputs.append(total / wall if wall > 0 else 0.0)
        rputs.append(conc / wall if wall > 0 else 0.0)
        for r in rs:
            ttfts.append(r["ttft"])
            if r["tokens"] > 1:
                tpots.append(r["tpot_off"])
            itls += r["itls"]
        time.sleep(1)
    agg.update({
        "out_toks_p50": med_ms(tputs), "req_toks_p50": med_ms(rputs),
        "ttft_ms_p50": pct(ttfts, 50), "ttft_ms_p99": pct(ttfts, 99),
        "tpot_ms_p50": pct(tpots, 50), "tpot_ms_p99": pct(tpots, 99),
        "itl_ms_p50": pct(itls, 50), "itl_ms_p99": pct(itls, 99),
    })
    print(json.dumps(agg, ensure_ascii=False), flush=True)
    return agg


def top_logprobs(base, model):
    d = post(base + "/v1/chat/completions", {
        "model": model, "messages": [{"role": "user", "content": SHORT_PROMPT}],
        "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 5,
    })
    items = (d["choices"][0].get("logprobs") or {}).get("content") or []
    return [(t["token"], round(t["logprob"], 4)) for t in (items[0].get("top_logprobs") or [])] if items else []


def wait_ready(base: str, model: str) -> None:
    for _ in range(240):
        try:
            post(base + "/v1/chat/completions",
                 {"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1},
                 timeout=30)
            return
        except Exception:
            time.sleep(5)
    raise SystemExit("server not ready")


def probe_assertions(probe_path: str, phase: str) -> dict:
    """在役探针断言(B5c/任务书第 1c 项):A/B 数字有效性前提。

    before: ready_plans>=1 且 capture_calls>0(图模式已进图);快照 fallbacks/markers。
    after:  测量窗口 fallbacks 增量必须为 0,markers 必须增长(内核真实服役)。
    """
    import os

    if not probe_path or not os.path.exists(probe_path):
        return {}
    with open(probe_path) as f:
        m = json.load(f)
    if phase == "before":
        assert m.get("ready_plans", 0) >= 1, f"probe: 没有 READY plan: {m}"
        assert m.get("capture_calls", 0) > 0, f"probe: capture_calls=0,TileInfer 未进图: {m}"
        assert not m.get("exceptions", 0), f"probe: 异常熔断已触发: {m}"
        print(f"[probe-before] {m}", flush=True)
    else:
        print(f"[probe-after] {m}", flush=True)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8321)
    ap.add_argument("--model", default="Qwen3.6-35B-A3B")
    ap.add_argument("--label", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--conc", type=int, nargs="+", default=[4, 16, 32])
    ap.add_argument("--warmup", type=int, default=1,
                    help="每场景先做 N 次不计入统计的预热(双侧同参,公平)")
    ap.add_argument("--probe", default="",
                    help="TileInfer 在役探针 JSON 路径(TILEINFER_METRICS_PATH);空=FIA 侧跳过")
    args = ap.parse_args()

    import os
    os.makedirs(args.outdir, exist_ok=True)
    base = f"http://127.0.0.1:{args.port}"

    wait_ready(base, args.model)

    p8k, p16k = make_long_prompt(8000), make_long_prompt(16000)
    report = {"label": args.label, "model": args.model, "port": args.port,
              "metrics_note": "TTFT/TPOT/ITL 单位 ms;TPOT=官方定义 (latency-ttft)/(tokens-1);"
                              "out_toks=输出 token 吞吐 tok/s;req_toks=请求吞吐 req/s",
              "scenarios": []}

    report["top5_logprobs"] = top_logprobs(base, args.model)

    probe_before = probe_assertions(args.probe, "before")

    scenarios = [
        ("ttft_short_1tok", SHORT_PROMPT, 1),
        ("decode_128tok", SHORT_PROMPT, 128),
        ("decode_512tok", SHORT_PROMPT, 512),
        ("long8k_128tok", p8k, 128),
        ("long16k_128tok", p16k, 128),
        ("ttft_long16k_1tok", p16k, 1),
    ]
    if args.warmup:
        print(f"[warmup] 每场景预热 {args.warmup} 次(不计入统计)", flush=True)
        for _, prompt, mt in scenarios:
            for _ in range(args.warmup):
                gen_stream(base, args.model, prompt, mt)
        for c in args.conc:
            scenario_conc(base, args.model, c, 128, 1)  # 并发场景预热走 runs=1 输出被丢弃
        report["warmup_runs"] = args.warmup

    for label, prompt, mt in scenarios:
        report["scenarios"].append(scenario_single(base, args.model, label, prompt, mt, args.runs))
    for c in args.conc:
        report["scenarios"].append(scenario_conc(base, args.model, c, 128, args.runs))

    if probe_before:
        probe_after = probe_assertions(args.probe, "after")
        assert probe_after.get("fallbacks", -1) == probe_before.get("fallbacks", -2), \
            "probe: 测量窗口出现 FIA 回退,A/B 数字无效"
        assert probe_after.get("kernel_markers", 0) > probe_before.get("kernel_markers", 0), \
            "probe: 测量窗口 marker 未增长,TileInfer 未服役,数字无效"
        report["probe_before"] = probe_before
        report["probe_after"] = probe_after

    out = os.path.join(args.outdir, f"ab_full_{args.label}.json")
    with open(out, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("SAVED:", out, flush=True)


if __name__ == "__main__":
    main()
