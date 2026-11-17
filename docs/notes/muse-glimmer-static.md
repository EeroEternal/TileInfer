# MuseGlimmer-30B on TileInfer/Ascend 950PR — status (2026-11-16, V3-only)

Target: `/home/models/Muse-Glimmer-30B` (52 dense layers, hidden 6656, 32Q/2KV GQA @ head_dim
128, iRoPE: 39 sliding-window + 13 full-attention layers, untied lm_head).

**Runtime form: V3 only.** The declaration lives in `tileinfer/manifests/muse_glimmer.py`; the
upstream-facing implementation lives in `tileinfer/bridge/models/` (`muse_glimmer.py` for the
architecture + AutoConfig shim, `atem.py` for the chat parser) and is injected by the
`model_table` seam. Attention, KV management, worker and graph capture are the vendor's
(vllm-ascend). There is no V2 copy of this model any more: the old
`tileinfer/vllm/{models/muse_glimmer,profiles/muse_glimmer,reasoning}.py` paths and the
Muse-only `dense` op family were deleted before the whole v2 tree was archived under
`legacy/v2/`.

**Vendor route update (2026-11-13, device-verified).** vllm-ascend 0.23.0 *can* serve this
model — the earlier "sliding window unsupported" conclusion was wrong. The failure was
`ACLNN_ERR_INNER_TILING_ERROR (561002)` whose plog reason is
``In non-PA scenarios, value tensors must be contiguous``: the fused-qkv ``split`` leaves ``v``
a strided view for multi-token prefill, and vllm-ascend's sliding-window branch calls CANN's FIA
without making the inputs contiguous (the full-attention branch goes through
``DeviceOperator``, which does). Upstream fixed the vendor side in ``cfb01b1c8`` (#17925, first
in v0.30.0); the model-side equivalent (``v = v.contiguous()`` with a comment) is now in the
port. With it, the model registered through the V3 `model_table` seam returns HTTP 200 on the
vendor stack, produces clean chat content, recalls the 3,460-token secret exactly, and decodes at
**37 ms/tok** (eager, single sequence). The earlier 75–78 ms/tok figure belongs to the retired
V2 backend below and is kept only as history.

## Status ledger

| # | Item | State |
|---|---|---|
| 0 | Architecture backport | CLOSED |
| 1 | Sliding-window compute (decode kernel path + prefill reference) | CLOSED |
| 1b | Out-of-window KV reclamation (per-group pools) | CLOSED |
| 2 | Hybrid KV manager for attention-only profiles | CLOSED (folded into 1b) |
| 3 | Vendor (vllm-ascend) route | RESOLVED (root cause found + model-side fix; 37 ms/tok) |
| 4 | ATEM reasoning parser | CLOSED |
| 5 | Real weights end-to-end | DONE |
| 6 | Concurrency (8 seqs, mixed lengths) | VALIDATED |
| 7 | Capacity-grid OOM at 0.82 utilisation | FIXED |

Items 1, 1b and 7 document the retired V2 backend that produced the first correct tokens. They
are kept as root-cause history; the V3 runtime does not execute that code. On V3 the same
properties come from vllm-ascend (windowed FIA, hybrid KV groups, worker/graph capture).

## #0 architecture backport (CLOSED)

Root cause: `muse_glimmer` exists only in vLLM main (@ ef5f913, 2026-11-09); vLLM 0.23.0's
registry has no such architecture, and the venv's transformers 5.5.4 does not know the
`model_type` (config.json cannot be parsed before the architecture lookup even happens).

- `tileinfer/bridge/models/muse_glimmer.py` — text-only port (no SupportsMultiModal) with a
  `MuseGlimmerHFConfig` shim that promotes **only** text_config (vision_config's
  `window_attention` layer_type is rejected by transformers 5.5.4's strict validator).
  0.23 API fixes: `islice` PP iteration (PPMissingLayer padding), no
  `from __future__ import annotations` (string annotations break `support_torch_compile`'s
  dynamic-dim inference), classic `stacked_params_mapping` in `load_weights` (0.23 has no
  `WeightsMapper.orig_to_new_stacked`), `is_text_generation_model` + `embed_input_ids`
  delegation (VllmModelForTextGeneration protocol). Per-layer sliding windows are passed through
  (`per_layer_sliding_window`); the vendor attention backend computes them (#1).
- `tileinfer/manifests/muse_glimmer.py` — declaration (52 layers, window 2048, iRoPE, output
  gate, `model_impl`, `reasoning_parser*`).  This is the only place the model's facts live.
- `tileinfer/bridge/seams.py::install_model_table` — runs `register_config` /
  `register_models` and registers the parser; no V2 plugin or model profile is involved.

## #1 sliding-window compute — V2-era history (CLOSED, not on the V3 path)

Two mechanisms, dispatched by `TileInferAttentionImpl.forward`:

- **Decode — staged kernel path** (archived v2 `legacy/v2/tileinfer/vllm/attn/window.py`; the
  v3 path does not install it). The paged decode kernel is
  full-causal only; `WindowedDecodeStager` gathers the last `window` tokens of each request into a
  compact per-layer scratch pool (fixed shapes, device-side indices, no host sync) and runs the
  *unchanged* kernel on it with a synthetic `arange` block table and clamped `seq_lens` — causal
  over exactly the gathered range **is** the window. Every kernel-path fallback
  (plan_not_ready / plan_failed -> reference on the scratch) stays correct because the scratch
  *is* the window. `TILEINFER_WINDOW_DECODE=0` restores the masked-reference decode (A/B control).
  Same-geometry layers share one stager (`shared_stager`), and `stage()` skips the gather when
  the metadata object is unchanged (the engine hands every layer of a group the same instance) —
  one gather per step instead of 39.
- **Prefill — masked reference path.** `window_left = w - 1` over absolute positions
  (`q_pos = kv_len - qo_len + i`, which keeps chunked-prefill extends exact).

Supporting details:

- `TileInferAttentionImpl` accepts `sliding_window` and stores it as `window_size`; the public
  `sliding_window` attribute stays `None` because `get_per_layer_parameters` indexes it as a
  tuple.
- `TileInferMetadataBuilder.get_cudagraph_support` downgrades windowed specs to
  `AttentionCGSupport.NEVER` (0.23's enum member is `NEVER`, not `NONE`) — the reference prefill
  walks the page table on the host, and the staged decode path is not verified under capture.
- Worker pre-warm drives windowed layers through `prepare_windowed_buckets` (scratch states +
  plans) instead of the real-pool path.
- Tests: `tests/test_windowed_attention.py` (22 cases) — window math vs a slice-based oracle
  (decode / chunked extend / degenerate windows), staged numerics across an engine-page boundary
  vs a pool-read oracle, two-step stability, shared-stager skip, dispatch counters, builder
  downgrade.

## #1b/#2 out-of-window KV reclamation (CLOSED)

`ModelProfile.mixed_attention_specs` (Muse sets it) makes the platform answer
`support_hybrid_kv_cache() == True` for an attention-only mixed model. vLLM then builds a
FullAttentionSpec group (13 layers) and a SlidingWindowSpec group (39 layers), each with its own
pool, block table and slot mapping; the sliding-window allocator keeps only in-window blocks real
— out-of-window table entries point at the reserved NULL block (id 0, always readable), which the
windowed compute masks out. Device-validated: two groups, 3,706-token prompt HTTP 200 (dummy),
3,460-token recall exact (real weights).

## #3 vendor (vllm-ascend) route (RESOLVED)

- **Environment**: `/home/userlist/ww/py_venv` (torch 2.10.0 + torch_npu 2.10.0.post4 +
  triton-ascend 3.2.2 + vllm 0.23.0 + vllm-ascend 0.23.0-950), used **read-only** during the
  probe: the model was registered by a ``vllm.general_plugins`` entry point shipped from a `/tmp`
  directory on ``PYTHONPATH``. That temporary plugin is now retired; the same registration is the
  V3 bridge's ``model_table`` seam (`tileinfer/bridge/models/`). Verified live: the registration
  reaches both the API server and the EngineCore process.
- **Root cause** (from ``/root/ascend/log/debug/plog``):
  ``paged_attention_checker.cpp:1001 ... Reason: In non-PA scenarios, value tensors must be
  contiguous`` → ``NnopbaseExecutorDoTiling`` → errno 561002. Non-PA = prefill with
  ``block_table is None``. ``v`` is a strided view because of the fused-qkv split; ``k`` becomes
  contiguous through ``qk_norm``/``reshape``; single-token decode is contiguous (size-1 stride
  is irrelevant), which is why only prefill failed.
- **Code paths**: full-attention prefill uses ``DeviceOperator.npu_fused_infer_attention_score``
  (``device_op.py:1150`` — calls ``.contiguous()``) → fine; the sliding-window branch calls raw
  ``torch_npu.npu_fused_infer_attention_score`` (``attention_v1.py:1315-1331``) without it →
  fails. That asymmetry explains the "sliding window off = 200" control exactly.
- **Fix**: vendor side ``cfb01b1c8`` (2026-10-06, PR #17925) adds
  ``if block_table is None: key/value = .contiguous()``; first release v0.30.0. Model side (this
  repo): ``v = v.contiguous()`` in ``muse_glimmer.py`` before ``self.attn``.
- **Validation on the vendor stack** (real weights, eager): clean content (``中国首都北京。``),
  3,460-token secret recalled (``7391``), TPOT 37 ms/tok.

## #4 ATEM reasoning parser (CLOSED)

The checkpoint's chat template ends the generation prompt at a bare `<|start|>assistant`; the
model emits ` to=self` reasoning `<|eom|>` ` to=user` answer `<|eot|>` (special tokens are
stripped, so the channel markers survive as plain text). `tileinfer/bridge/models/atem.py`
registers an `atem` reasoning parser (from the V3 `model_table` seam; `--reasoning-parser atem`):
non-streaming `extract_reasoning` plus a stateless-from-text streaming splitter that holds back
marker-prefix tails so `" to=user"` never splits across chunks. Degenerate outputs (the model
answering inside the self channel) are reported entirely as reasoning. Tests:
`tests/test_atem_reasoning.py` (18 cases; char-by-char streaming must reassemble to the
non-streaming split). Device-verified: chat `content` is clean (`中国首都北京。`,
`1+1等于 **2**。`, `7391`), reasoning carries the to=self channel.

## #5 real weights (DONE)

Both shards on disk (49.95GB + 9.6GB, ~56GiB); load ~5 s (page cache hot), no loader errors.
Coherent generation ("中国的首都是" -> "北京，北京的面积是16410平方千米…"), long-context recall
across the window boundary (secret at token 0 of a 3,460-token prompt recalled exactly).

## #6 concurrency (VALIDATED)

8 concurrent requests, mixed prompt lengths (two above the 2048 window), all HTTP 200; first
round 109 s incl. plan compiles, warm round 3.5 s; zero tracebacks. Exercises bucket>1 staging
states, per-bucket plans and group metadata.

## #7 capacity-grid OOM at 0.82 utilisation — V2-era history (FIXED)

`serving/buckets.capacity_view` re-binds pool storage at the declared capacity via `Tensor.set_`;
on torch_npu growing the storage **re-allocates the whole pool** (not just the delta), which OOMed
the hybrid two-group deployment at 0.82 utilisation.

Two code-level facts shaped the fix:

- Skipping the extension is impossible: the tilelang runtime enforces the packed ABI shape
  (`KCache shape[0] == num_pages_cap`).
- Rounding inside `get_kv_cache_shape` is impossible: the runner sizes the raw buffer from the
  scheduler's `num_blocks` and applies the backend shape as a **view** (shape must fit).

Final fix (two halves, both in code):

- `declared_pool_cap` quantizes onto an **eighth-of-a-pow2 grid** (`base*(8+j)/8`): the extension
  bounded at pool/8, key-stable under page-count drift (9425/9899 -> 10240), exact pools get
  `cap == pages`. The kernel module's second, divergent copy now delegates to
  `serving/buckets` (one source of truth).
- `TileInferModelRunner._allocate_kv_cache_tensors` pads each attention KV buffer to
  `grid_round(pages) * page_size_bytes` at birth. The scheduler's `num_blocks` is untouched (the
  extra pages are never scheduled), so at reshape time `num_blocks` derives from the padded numel
  = the grid, `declared_pool_cap` == actual pages, and every `capacity_view` hits its
  `cap == pages` fast path — no `set_`, no resize, no allocation at serving time.
  `grid_round` is idempotent, which is what keeps the worker's synthetic pre-warm pools on the
  same plan keys as the padded real pools.

Device-validated at **0.82** (previously OOMed at first prefill): coherent generation, recall
exact, TPOT unchanged.

## Performance — retired V2 backend (real weights, single seq, eager)

- decode TPOT: **75–78 ms/tok** (~13 tok/s) after the staged kernel path (was 102–108 ms/tok with
  the masked-reference decode); shared staging does not change it (the gather was never the
  bottleneck — it saves ~90 MiB of duplicate scratch pools and 38 redundant gathers per step).
- windowed **prefill** still runs the masked reference path; long-prompt TTFT is the remaining
  performance item.
- **Vendor stack (vllm-ascend 0.23.0-950, read-only registration, same model): 37 ms/tok** —
  roughly 2x the v2 stack. Its attention path is the vendor's optimized FIA; the v2 stack's
  windowed staging + reference prefill is what costs the difference.
  **`[待复核]`**: this line has no command, prompt or date, so it is a lead, not a baseline.
  The 2026-11-17 re-validation on this repository's venv measured **41.3 ms/tok** (three runs,
  cards 0/1/4); see
  [`muse-glimmer-v3-validation-20261117.md`](muse-glimmer-v3-validation-20261117.md).

## Launch recipe (V3, vllm-ascend environment)

```bash
# /home/userlist/ww/py_venv: torch 2.10 + torch_npu + vllm 0.23.0 + vllm-ascend 0.23.0-950
env PYTHONPATH=/home/userlist/.ernest/TileInfer TILEINFER_MANIFEST=muse_glimmer \
  /home/userlist/ww/py_venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model /home/models/Muse-Glimmer-30B --max-model-len 8192 --max-num-seqs 1 \
  --enforce-eager --gpu-memory-utilization 0.82 \
  --reasoning-parser atem --port 8345
```

`TILEINFER_MANIFEST` makes the bridge build the plan from the declaration at boot. A frozen
deployment can instead write a plan and point `TILEINFER_PLAN` at it:

```bash
tileinfer plan build --model /home/models/Muse-Glimmer-30B --out /tmp/muse.plan.json
env PYTHONPATH=/home/userlist/.ernest/TileInfer TILEINFER_PLAN=/tmp/muse.plan.json ...
```

The old V2 recipe (`.venv_native213`, `TILEINFER_OPS=none`, no manifest) is retired: V2 no
longer contains any Muse code.

## V3 wiring (landed 2026-11-16)

The bridge hosts vendor-absent models end to end, with no V2 involvement:

- Declaration: `manifests/muse_glimmer.py` (52 layers, 39 sliding + 13 NoPE global,
  window 2048, output gate, `model_impl`, `reasoning_parser`).  Schema gained
  `attention.sliding`, `sliding_window`, `rope_disabled_layers`, `model_impl`,
  `reasoning_parser(_impl)`; the C2 benchmark sample became optional (a brand-new model is
  declared before its sample exists).
- Decision: with a vendor environment, every capability selects the vendor (including
  `attention.sliding` -> the same Ascend attention backend, sparse mode 4); the plan adds
  the `model_table` access point and carries the model/parser injection fields.
- Seam: `install_model_table` imports `plan.model_impl` and runs its `register_config` /
  `register_models` actors, then registers the parser under the plan's name.  The model and
  parser implementations live in `tileinfer/bridge/models/`, the only layer allowed to import
  the engine.
- Removed V2 surface: `tileinfer/vllm/models/muse_glimmer.py`,
  `tileinfer/vllm/profiles/muse_glimmer.py`, `tileinfer/vllm/reasoning.py`, and the Muse-only
  `tileinfer/vllm/ops/families/dense.py`; the V2 plugin and profile registry no longer mention
  the model. The remaining v2 tree is archived under `legacy/v2/` and is not importable.
- Usage: install the bridge in the vllm-ascend environment and set
  `TILEINFER_MANIFEST=muse_glimmer` (or point `TILEINFER_PLAN` at a built plan).
- **Device re-validation under the V3 layout: DONE on 2026-11-17** (8 × Ascend950PR, real
  weights, single card): C1 pass; C3 pass (8 graphs @32K, 99/99 decode hits, 0 fallbacks);
  C4 absolute 41 ms/tok, 175 tok/s at 8-way concurrency; cold start 38 s with a fresh cache.
  Environment deltas vs the reference probe: transformers 5.5.4 + hub 1.33 (the checkpoint
  tokenizer is `TokenizersBackend`), NNAL/ATB 9.2.0-beta.1, triton-ascend 3.2.2, CANN 9.3.0.
  Commands and numbers: [`muse-glimmer-v3-validation-20261117.md`](muse-glimmer-v3-validation-20261117.md).

C2 is now wired: `manifests/muse_glimmer.py` carries a first-token top-5 frozen from the
float32 CPU oracle (`tools/muse/c2_oracle.py`), with a declared bf16 tolerance and a
`tools/muse/check_c2.py` checker. The served model matched the oracle's token set exactly
(top-1 diff 3.7e-5; max top-5 diff 0.059).

## Pre-existing .ernest test failures (not from this work, verified via git stash)

- `test_attn_contract.py::test_backend_identity_points_at_our_impl_and_builder` — asserts
  `get_name() == "TILEINFER"` but `.ernest`'s backend returns `"CUSTOM"` (fails on clean HEAD).
- `test_model_profiles.py::test_a_dummy_profile_drives_platform_and_backend` — asserts exactly one
  MLA reject reason; under a real vLLM the ABC base adds its own ("MLA not supported").
- `test_moe_ops.py::test_fused_experts_validation` — order-dependent: fails after another test has
  touched the NPU device in the same pytest process.

## Do-not-touch

`/home/userlist/zxs/TileInfer` repo content (user's working tree, 51 commits ahead of the old
base, 23 of them unpublished); its `.venv_native213` is the runtime venv. Off-limits:
`/home/userlist/ww/*`, `/home/dlb/_wheels/*` (read-only use only).
