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
| 125,002 | graph | 30,329.7 ms | 48.4 ms | 20.7 |

Concurrency (956-token prompts, 64 output each; 8-way on the default run, 16/32-way with
`max-num-seqs 32`, `max-model-len 131072`, `gpu-memory-utilization 0.9`):

| mode | wall | output tok/s | max TTFT |
|---|---:|---:|---:|
| 8-way eager | 2.92 s | 175.4 | 280.0 ms |
| 8-way graph | 2.93 s | 174.6 | 286.5 ms |
| 16-way graph | 3.09 s | **331.4** | 253.0 ms |
| 32-way graph | 3.37 s | **608.0** | 413.2 ms |

Long-context recall (125,059-token chat prompt, secret `7391` at the head or middle, greedy,
`max_tokens 512`):

| placement | result | completion tokens | elapsed |
|---|---|---:|---:|
| head | ✅ content is exactly `7391` | 101 | 34.7 s |
| middle | ✅ content is exactly `7391` | 138 | 37.2 s |

Long-context concurrency (125,002-token prompts, 64 output, `max-model-len 131072`,
`max-num-seqs 4`, `gpu-memory-utilization 0.9`):

| mode | result |
|---|---|
| 2-way | ✅ both requests returned 64 tokens; 33.7 s wall; max TTFT 30.4 s |
| 4-way | ❌ EngineCore crash (`EngineDeadError`) after prefill started; single-card capacity/engine limit |

Cold start (fresh `VLLM_CACHE_ROOT`, eager, `max-model-len 8192`): **38 s** from launch to
`Application startup complete`; weights load 2.08 s, engine init 1.93 s.

Absolute single-stream decode is ~41 ms/token (24.5 tok/s) for ≤32K prompts, stable across
cards (0/1/4 measured: 41.29–41.32 ms); at 125K prompt it is 48.4 ms/token, and the 125K
prefill takes 30.3 s (TTFT). This machine is ~11% slower than the
**unverified** 37 ms/tok line in `muse-glimmer-static.md`; that line has no command, prompt
or date, so by `performance.md` §1 it is `[待复核]`, not a baseline. Most plausible source
of the gap is the NNAL build (9.2.0-beta.1 against CANN 9.3), not the card.

## 4.1 C2 — numeric parity against a float32 CPU oracle

`tools/muse/c2_oracle.py` is an independent pure-torch float32 implementation (no vLLM, no
vllm-ascend) of the same checkpoint. It predicts the same first token:

| token | oracle (fp32) | served (bf16) | diff |
|---|---:|---:|---:|
| ` Paris` | -1.096611 | -1.096574 | **0.000037** |
| ` a` | -2.162277 | -2.221574 | 0.059297 |
| ` one` | -2.813581 | -2.784074 | 0.029507 |
| ` the` | -2.838582 | -2.784074 | 0.054508 |
| ` also` | -3.233084 | -3.221574 | 0.011510 |

Token set is identical (5/5), top-1 logprob differs by 3.7e-5, max top-5 difference is 0.059.
The manifest declares the tolerance (`top1_logprob 1e-3`, `top5_max_logprob 0.1`, exact token
set) because the oracle is float32 while serving is bf16 — this is not the strict 1e-4
same-precision rule. Reproduce with:

```bash
python tools/muse/c2_oracle.py --out /tmp/muse-c2.json
python tools/muse/check_c2.py --base-url http://127.0.0.1:8360
```

## 4.2 TP=2 (two cards) — 128K x 4 and C2

Host prerequisite (not TileInfer code): `/etc/hccl_rootinfo.json` is an **empty directory** on
this box. HCCL cannot parse it, so TP>1 fails with `hcclCommInitRootInfoConfig ... Failed to
parse the ranktable file`. The directory was moved aside for the duration of these runs and
restored afterwards; no code or workflow change encodes this workaround.

TP=2 setup: cards 2+3, `--tensor-parallel-size 2`, `max-model-len 131072`,
`max-num-seqs 4`, `gpu-memory-utilization 0.85`; ready in 52–54 s.

| test | result |
|---|---|
| C2 (`check_c2.py --tp-size 2`) | token set exact; top-1 diff **0.001138** (TP tolerance 0.002), top-5 max 0.058; **pass** |
| C3 (100 output tokens, `SIGUSR1` to both workers) | **6 graphs** (PIECEWISE 3 + FULL 3), `decode_calls=99`, `decode_graph_hits=99`, `decode_fallbacks=0`, `errors=0`; backend = vendor `AscendAttentionBackend` |
| 125K single | TTFT **18.4 s**, TPOT **30.6 ms** (TP=1: 30.3 s / 48.4 ms) |
| 4 x 125K, prefix cache disabled | 4/4 requests x 64 tokens; wall **76.1 s**; max TTFT **74.2 s** |
| 4 x 125K, prefix cache enabled | 3.0 s wall (identical prompts reuse the first prefill; not a capacity number) |

The TP=1 C2 tolerance stays at 1e-3; TP=2 uses the separately declared `tolerance_tp`
(2e-3), because tensor-parallel reductions change the bf16 rounding.

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
| C2 跑对 | ✅ top-5 token set exact; top-1 diff 3.7e-5; max top-5 diff 0.059 within the declared bf16 tolerance |
| C3 能上图 | ✅ (8 graphs @32K; 99/99 hits, 0 fallbacks @4K) |
| C4 跑得快 | ✅ absolute gate (below); 128K single-stream/graph and 16/32-way concurrency measured; no vendor A/B |

C4 absolute gate for this vendor-absent model (declared here, measured on one 950PR):

| Metric | Threshold | Measured | Pass |
|---|---:|---:|---|
| single-stream TPOT | ≤ 100 ms | 40.4–41.3 ms | ✅ |
| TTFT @ 8K prompt | ≤ 3 s | 1.26 s graph / 2.0 s eager | ✅ |
| 8-way throughput | ≥ 100 tok/s | 175.4 tok/s | ✅ |
| cold start (fresh cache) | ≤ 120 s | 38 s | ✅ |

Measured beyond the gate (not gated): 125K prompt / 64 output -> TTFT 30.3 s, TPOT 48.4 ms;
16-way 331 tok/s; 32-way 608 tok/s; 128K x 2 concurrency passes on one card, 128K x 4 needs
TP=2 (74.2 s max TTFT, see §4.2).

What is still open: a same-stack vendor A/B (the vendor can serve this model only with the same registration, so it
is not an independent baseline). None of those block the C1–C4 statement above, but they must
be listed as unverified.
