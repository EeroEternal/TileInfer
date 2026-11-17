# MuseGlimmer-30B on V3 — device validation (2026-11-17)

> Model: `/home/models/Muse-Glimmer-30B` (real weights, BF16).
> Host: 8 × Ascend950PR (86 GB HBM each); all runs on one card, chosen via
> `ASCEND_RT_VISIBLE_DEVICES` (card 0 was held by another tenant during part of the session).
> This note records **what actually ran**, not a support claim. "Supported" requires C1–C4;
> C2 is still open (see §6).

## 1. Environment

| Item | Value |
|---|---|
| CANN | 9.3.0 (weekly 20260930), `source .../cann-9.3.0/set_env.sh` |
| Driver | 25.7.0 |
| torch / torch_npu | 2.10.0+cpu / 2.10.0.post4 |
| vLLM / vllm-ascend | 0.23.0 / 0.23.0 (950 wheel) |
| transformers / hub | 5.5.4 / 1.33.0 (checkpoint tokenizer_class is v5 `TokenizersBackend`) |
| triton / triton-ascend | 3.2.0 / 3.2.2 (vLLM vendors the Ascend triton build) |
| NNAL / ATB | 9.2.0-beta.1 (`libatb.so` is required by `torch_npu._npu_rotary_embedding`) |
| glibcxx | bundled `libstdc++.so.6.0.30` (CANN opmaster needs GLIBCXX_3.4.29/30) |

The bridge is installed editable from this repository; the entry point is the only one
registered (`vllm.general_plugins -> tileinfer.bridge.install:install`).

## 2. Reproduction

```bash
source <CANN>/set_env.sh
source <NNAL>/nnal/atb/set_env.sh
export LD_LIBRARY_PATH=<repo>/.work/glibcxx:$LD_LIBRARY_PATH
export PYTHONPATH=<CANN>/python/site-packages:<CANN>/opp/built-in/op_impl/ai_core/tbe
export TILELANG_CACHE_DIR=$HOME/.cache/tilelang-ascend950
export TILEINFER_MANIFEST=muse_glimmer
export ASCEND_RT_VISIBLE_DEVICES=<free card>
source <repo>/.venv/bin/activate

python -m vllm.entrypoints.openai.api_server \
  --model /home/models/Muse-Glimmer-30B --served-model-name muse \
  --trust-remote-code --max-model-len 32768 --max-num-seqs 8 \
  --gpu-memory-utilization 0.85 --reasoning-parser atem --port 8360
```

`python -m tileinfer.cli.main plan build --model ... --out plan.json` gives
`servable=True gaps=0`, selections = 5 × `vllm_ascend.*`, access points =
`plugin_entry, attn_select, method_wrap, graph_hook, model_table`.

## 3. C3 — graph capture (measured)

With `max-model-len 4096`, one request of 100 output tokens, `TILEINFER_PLAN` carrying
`instrumentation.dump_every_decode_steps=1`:

```
captured_graphs=2 (PIECEWISE 1 + FULL 1, FULL_AND_PIECEWISE)
decode_calls=99  decode_graph_hits=99  decode_fallbacks=0  errors=0
```

With `max-model-len 32768` the same launch captures 8 graphs (PIECEWISE 4 + FULL 4).
So graph capture is real, and steady-state decode stays on graphs with zero fallbacks in
this run. This **corrects the earlier "graph gives no gain" note**, which only compared
TPOT: graph does not change TPOT, but it lowers TTFT (see §4).

## 4. C4 — absolute numbers (no vendor A/B)

Streaming client, temperature 0, 64 output tokens, prefix cache warm, one card.
`prompt_tokens` are tokenizer counts (the repeated-text prompts are below their nominal
character length).

| prompt tokens | mode | TTFT | TPOT | tok/s |
|---:|---|---:|---:|---:|
| 956 | eager | 423.6 ms | 40.4 ms | 25.1 |
| 956 | graph | **328.0 ms** | 40.6 ms | 25.0 |
| 6,356 | eager | 1,998.1 ms | 40.8 ms | 24.9 |
| 6,356 | graph | **1,259.9 ms** | 41.0 ms | 24.8 |
| 25,256 | eager | 3,817.3 ms | 40.9 ms | 24.8 |
| 25,256 | graph | 3,752.0 ms | 41.1 ms | 24.7 |

Concurrency (8 requests, 956-token prompts, 64 output each, `max-num-seqs 8`):

| mode | wall | output tok/s | max TTFT |
|---|---:|---:|---:|
| eager | 2.92 s | **175.4** | 280.0 ms |
| graph | 2.93 s | 174.6 | 286.5 ms |

Cold start (fresh `VLLM_CACHE_ROOT`, eager, `max-model-len 8192`): **38 s** from launch to
`Application startup complete`; weights load 2.08 s, engine init 1.93 s.

Absolute single-stream decode is ~41 ms/token (24.5 tok/s), stable across 1K/8K/32K prompts
and across cards (0/1/4 measured: 41.29–41.32 ms). This machine is ~11% slower than the
**unverified** 37 ms/tok line in `muse-glimmer-static.md`; that line has no command, prompt
or date, so by `performance.md` §1 it is `[待复核]`, not a baseline. Most plausible source
of the gap is the NNAL build (9.2.0-beta.1 against CANN 9.3), not the card.

## 5. Weight integrity (incident + check)

The first copy of `model-00002-of-00002.safetensors` had a **2,719,744,000-byte zero prefix**
(`4096 × 664000`), i.e. a preallocated hole materialised by a copy; the safetensors header
length read as 0 and vLLM failed with `invalid JSON in header`. After re-upload the header is
11,720 bytes, 89 tensors, `data_start + max_end == file_size`. Shard 1 (50 GB, 1347 tensors)
was always valid.

Every delivery must run a header/index check before serving; `scripts/verify-safetensors.py`
does that and exits non-zero on a truncated/zeroed header.

## 6. Gates

| Gate | State |
|---|---|
| C1 能跑 | ✅ (HTTP 200, chat completions, reasoning + content channels) |
| C2 跑对 | ❌ **open** — `manifests/muse_glimmer.py` still has `benchmark={}`; first-token top-5 / logprob parity is not captured |
| C3 能上图 | ✅ (8 graphs @32K; 99/99 hits, 0 fallbacks @4K) |
| C4 跑得快 | ⚠️ absolute only (41 ms/tok, 175 tok/s conc8); no vendor A/B and no 128K/combo grid |

Do **not** write "MuseGlimmer supported" until C2 is closed. What can be said today:
the V3 bridge hosts this vendor-absent architecture end to end on a single 950PR, with
real weights, graph capture and an ATEM parser, at ~41 ms/token eager/graph.
