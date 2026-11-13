# MuseGlimmer-30B on TileInfer/Ascend 950PR — P0 status (2026-11-12)

Target: `/home/models/Muse-Glimmer-30B` (52 dense layers, hidden 6656, 32Q/2KV GQA @ head_dim
128, iRoPE: 39 sliding-window + 13 full-attention layers, untied lm_head).

## What passed (P0 mechanical rehearsal — DONE)

Dummy-weight serving run completed **end to end** on the TileInfer v2 native stack
(`.venv_native213`: torch_npu 2.13.0rc1 + triton_ascend 3.2.2 + vllm 0.23.0, no vllm-ascend):

- config: `MuseGlimmerHFConfig` shim + `AutoConfig.register("muse_glimmer")` — ModelConfig OK
- model build: all 52 layers built on NPU, `support_torch_compile` dynamic-dim inference OK
- KV cache: 873,600 tokens @ `--max-model-len 8192`, max concurrency 106.64x
- serving: `POST /v1/completions` returned HTTP 200 with 4 generated tokens (garbage text is
  expected — dummy/random weights)

Launch recipe (the one that works; note **explicit env**, the heredoc `source /dev/stdin`
variant silently skipped the vLLM general-plugin load and failed at ModelConfig):

```bash
source /home/userlist/zxs/TileInfer/scripts/env-ascend102.sh
env PYTHONPATH=/home/userlist/.ernest/TileInfer TILEINFER_OPS=none setsid nohup \
  /home/userlist/zxs/TileInfer/.venv_native213/bin/python -m vllm.entrypoints.openai.api_server \
  --model /home/models/Muse-Glimmer-30B --load-format dummy --max-model-len 8192 \
  --max-num-seqs 1 --enforce-eager --gpu-memory-utilization 0.82 --block-size 128 --port 8331
```

Code added (all in `/home/userlist/.ernest/TileInfer`, per instruction — NOT in zxs):

- `tileinfer/vllm/models/muse_glimmer.py` — backport of vLLM main's muse_glimmer (text-only,
  no SupportsMultiModal) with a `MuseGlimmerHFConfig` shim that promotes **only** text_config
  (vision_config's `window_attention` layer_type is rejected by transformers 5.5.4's strict
  validator). Per-layer sliding windows are passed through (`per_layer_sliding_window`) — the
  backend computes them (see gap #1). 0.23 API fixes baked in: `islice` PP iteration, no
  `from __future__ annotations` (breaks `support_torch_compile` dim inference), classic
  `stacked_params_mapping` in `load_weights` (0.23 has no `WeightsMapper.orig_to_new_stacked`),
  `is_text_generation_model` + `embed_input_ids` delegation.
- `tileinfer/vllm/plugin.py` `_install_models` — one actor line registering the model.
- `tileinfer/vllm/profiles/muse_glimmer.py` — model profile (arch match, head_size (128,),
  group_min 16, block-first fused kv_shape, hybrid=None, op_family="dense").
- `tileinfer/vllm/ops/families/dense.py` — empty op family (no takeovers; dense model runs on
  vLLM's stock layer impls on NPU).
- `tileinfer/vllm/attn/impl.py` + `attn/metadata.py` — sliding-window compute (gap #1, below).

Env note: the editable dist-info's `entry_points.txt` was hand-restored to the v2 content
(console_scripts / vllm.general_plugins / vllm.platform_plugins) after a restore-reinstall had
rewritten the finder mapping; RECORD hash no longer matches (pip warns, functionality fine).

## Gaps (ordered)

- **#0 architecture backport — CLOSED** (this work). Root cause: muse_glimmer exists only in
  vLLM main (@ ef5f913, 2026-11-09); vLLM 0.23.0's registry has no such architecture, and
  transformers 5.5.4 (< 5.15) doesn't know the model_type.
- **#1 sliding-window attention — CLOSED for compute (2026-11-12, same day).** The window is a
  *compute-side* property in vLLM 0.23's unified-spec mode: with the hybrid KV cache manager off
  (attention-only profile), `unify_hybrid_kv_cache_specs` converts SlidingWindowSpec →
  FullAttentionSpec while keeping `sliding_window`, and the engine logs "The compute of layers
  like sliding window is still saved". Implemented in the backend:
  - `TileInferAttentionImpl` accepts `sliding_window` (stores `window_size`; the public
    `sliding_window` attribute stays `None` — `get_per_layer_parameters` indexes it as a tuple).
  - Windowed layers never reach the full-causal paged kernels: decode bumps
    `decode_window_reference`, prefill bumps `prefill_window_reference`, both run the reference
    path whose mask (`window_left = w - 1`, absolute positions) is exact for decode, chunked
    prefill extends, and degenerate windows (pinned by `tests/test_windowed_attention.py`, 18
    cases against a slice-based oracle).
  - `TileInferMetadataBuilder.get_cudagraph_support` downgrades windowed specs to
    `AttentionCGSupport.NEVER` (the reference path walks the page table on the host) — note
    0.23's enum member is `NEVER`, not `NONE`.
  - Device validation (dummy weights, window=2048, 39 sliding + 13 full layers): a 4,085-token
    prompt chunked-prefilled and 8 decode tokens served, HTTP 200.
- **#1b out-of-window KV reclamation — CLOSED (2026-11-13).** `ModelProfile.mixed_attention_specs`
  (Muse sets it) makes the platform answer `support_hybrid_kv_cache() == True` for an
  attention-only mixed model: vLLM builds a FullAttentionSpec group (13 layers) and a
  SlidingWindowSpec group (39 layers), each with its own pool, block table and slot mapping,
  and the sliding-window allocator keeps only the in-window blocks real — out-of-window table
  entries point at the reserved NULL block (id 0, always readable), which the windowed
  compute masks out. Validated on device with dummy weights (two groups, 3,706-token prompt,
  HTTP 200) and with real weights (3,460-token recall exact).
- **#1 windowed decode on the kernel path — CLOSED (2026-11-13).** `attn/window.py`
  (`WindowedDecodeStager`) stages the last `window` tokens of each request into a per-layer
  scratch pool on the device (fixed shapes, no host sync) and runs the *unchanged* full-causal
  paged decode kernel on the scratch pool with a synthetic `arange` table and clamped
  `seq_lens` — causal over exactly the window. Every kernel-path fallback (plan_not_ready /
  plan_failed → reference on the scratch) stays correct because the scratch *is* the window.
  Prefill keeps the masked reference path. `TILEINFER_WINDOW_DECODE=0` is the A/B control.
  Pinned by CPU tests (cross-page-boundary window numerics vs a pool-read oracle, two-step
  stability, dispatch counters) and on device: decode TPOT **102–108 → 75–80 ms/tok (~1.35x)**
  with real weights; long-context recall unchanged.
  - Worker pre-warm drives windowed layers through `prepare_windowed_buckets` (scratch states
    + plans) instead of the real-pool path — the real-pool prep would build a state (and
    compile a plan) the staged step never looks up.
- **#2 hybrid KV manager for attention-only profiles — folded into #1b** (same mechanism).
- **#3 vendor (vllm-ascend) route — BLOCKED.** No vllm-ascend installed anywhere; the 950 wheel
  (`/home/userlist/ww/vllm_ascend-0.23.0-950-...whl`) needs torch==2.10.0 + torch-npu 2.10.0.post4
  (wheels exist at `/home/dlb/_wheels/`) — a separate venv — and its `vllm_ascend_C.so` fails to
  load on torch 2.13 (undefined symbol). PyPI proxy (socks5://127.0.0.1:1234) was down.
- **#4 ATEM reasoning parser — CLOSED (2026-11-13).** The checkpoint's chat template ends the
  generation prompt at a bare `<|start|>assistant`; the model emits
  ` to=self` reasoning `<|eom|>` ` to=user` answer `<|eot|>` (special tokens stripped, so the
  channel markers survive as plain text). `tileinfer/vllm/reasoning.py` registers an
  `atem` reasoning parser (OOT, from the plugin install: `--reasoning-parser atem`): non-
  streaming `extract_reasoning` + a stateless-from-text streaming splitter that holds back
  marker-prefix tails so `" to=user"` never splits across chunks (17 CPU cases pin the
  reassembly property, including boundary-split markers). Device-verified: chat `content` is
  clean (`中国首都北京。`, `1+1等于 **2**。`, `7391`), reasoning carries the to=self channel.
  Degenerate model behaviour (answering inside the self channel) is reported entirely as
  reasoning — honest, not user-facing content.
- **Hybrid-mode memory note (new, device-observed).** `serving/buckets.capacity_view` re-binds
  pool storage at the pow2 declared capacity via `Tensor.set_`; on torch_npu the extension
  past the real pool **materialises a storage resize** (real allocation, once per pool). The
  unified pool fit in the utilisation headroom; the hybrid per-group pools are larger and the
  first prefill OOMed at 0.82 utilisation. Workaround for now: lower
  `--gpu-memory-utilization` (0.72 validated); the proper fix is a resize-free declared
  capacity (or charging the headroom in vLLM's profile pass).
- **Launch recipe (current, everything on):**
  `--gpu-memory-utilization 0.72 --block-size 128 --enforce-eager --reasoning-parser atem`
  with `PYTHONPATH=/home/userlist/.ernest/TileInfer TILEINFER_OPS=none`.
- **#4 real weights — RESOLVED 2026-11-13.** Both shards on disk (49.95GB + 9.6GB, ~56GiB).
  Real-weight serving validated on the device (v2 native stack, port as committed):
  - weights load in ~5 s (page cache hot); no loader errors — the classic
    `stacked_params_mapping` path + vision-key skip works against the real index;
  - coherent generation: "中国的首都是" → "北京，北京的面积是16410平方千米…" (factually right);
  - long-context recall across the window boundary: secret at token 0 of a 3,461-token prompt
    (> 2048 sliding window) recalled exactly — the 13 full-attention layers carry global info as
    iRoPE intends;
  - decode throughput ~102–108 ms/tok (~9.5 tok/s, single seq, eager): the 39 windowed layers
    decode on the reference path, which now dominates — kernel-side windowed decode is the next
    performance item (see #1b);
  - chat outputs carry a literal " to=self" prefix: that is the checkpoint's own ATEM
    tool-calling template (chat_template.jinja:181 renders
    `<|start|>assistant to=self<|message|>` for the reasoning channel), not a backend bug — a
    serving-config item (reasoning parser / template handling) for whoever productionises this.

## Pre-existing .ernest test failures (not from this work, verified via git stash)

- `test_attn_contract.py::test_backend_identity_points_at_our_impl_and_builder` — asserts
  `get_name() == "TILEINFER"` but `.ernest`'s backend returns `"CUSTOM"` (fails on clean HEAD;
  likely fixed among zxs's 22 ahead-commits).
- `test_model_profiles.py::test_a_dummy_profile_drives_platform_and_backend` — asserts exactly
  one MLA reject reason; under a real vLLM the ABC base adds its own ("MLA not supported").
- `test_moe_ops.py::test_fused_experts_validation` — order-dependent: fails after another test
  has touched the NPU device in the same pytest process.

## Do-not-touch

`/home/userlist/zxs/TileInfer` repo content (user's working tree, 22 commits ahead of
.ernest/main); its `.venv_native213` is the runtime venv. Off-limits: `/home/userlist/ww/*`,
`/home/dlb/_wheels/*` (read-only use only).
