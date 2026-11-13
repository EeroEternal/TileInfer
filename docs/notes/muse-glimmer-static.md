# MuseGlimmer-30B on TileInfer/Ascend 950PR — status (2026-11-13, code-verified)

Target: `/home/models/Muse-Glimmer-30B` (52 dense layers, hidden 6656, 32Q/2KV GQA @ head_dim
128, iRoPE: 39 sliding-window + 13 full-attention layers, untied lm_head).

**Runtime form: this work lives in the v2 tree (`tileinfer/vllm/`), which is the only runnable
path on this box.** The v3 bridge (`tileinfer/bridge/`) is merged and functional, but it is
vendor-first by construction: `install_model_table` is an empty seam ("no model replacement
requested"), the only manifest is `qwen3_6`, `attn_select` chooses among vllm-ascend's registered
backends, and `install.py` wraps `vllm_ascend.platform.NPUPlatform`. vllm-ascend is not installed
(950 wheel needs torch 2.10, this venv is torch 2.13) and has no MuseGlimmer support at all
(GitHub code search: 0 hits). v3 has no landing spot for a model the vendor does not serve, so
the Muse work is executed on the v2 runtime; its v3 home (if/when that environment exists) is the
`model_table` seam plus a manifest — see "v3 mapping" at the bottom.

## Status ledger

| # | Item | State |
|---|---|---|
| 0 | Architecture backport | CLOSED |
| 1 | Sliding-window compute (decode kernel path + prefill reference) | CLOSED |
| 1b | Out-of-window KV reclamation (per-group pools) | CLOSED |
| 2 | Hybrid KV manager for attention-only profiles | CLOSED (folded into 1b) |
| 3 | Vendor (vllm-ascend) route | BLOCKED (no adaptation + torch 2.10 ABI) |
| 4 | ATEM reasoning parser | CLOSED |
| 5 | Real weights end-to-end | DONE |
| 6 | Concurrency (8 seqs, mixed lengths) | VALIDATED |
| 7 | Capacity-grid OOM at 0.82 utilisation | FIXED |

## #0 architecture backport (CLOSED)

Root cause: `muse_glimmer` exists only in vLLM main (@ ef5f913, 2026-11-09); vLLM 0.23.0's
registry has no such architecture, and the venv's transformers 5.5.4 does not know the
`model_type` (config.json cannot be parsed before the architecture lookup even happens).

- `tileinfer/vllm/models/muse_glimmer.py` — text-only port (no SupportsMultiModal) with a
  `MuseGlimmerHFConfig` shim that promotes **only** text_config (vision_config's
  `window_attention` layer_type is rejected by transformers 5.5.4's strict validator).
  0.23 API fixes: `islice` PP iteration (PPMissingLayer padding), no
  `from __future__ import annotations` (string annotations break `support_torch_compile`'s
  dynamic-dim inference), classic `stacked_params_mapping` in `load_weights` (0.23 has no
  `WeightsMapper.orig_to_new_stacked`), `is_text_generation_model` + `embed_input_ids`
  delegation (VllmModelForTextGeneration protocol). Per-layer sliding windows are passed through
  (`per_layer_sliding_window`); the backend computes them (#1).
- `tileinfer/vllm/plugin.py` `_install_models` — registers the model (architecture + AutoConfig).
- `tileinfer/vllm/profiles/muse_glimmer.py` — model profile (arch match, head_size (128,),
  group_min 16, block-first fused kv_shape, hybrid=None, mixed_attention_specs=True,
  op_family="dense").
- `tileinfer/vllm/ops/families/dense.py` — empty op family (no takeovers; the dense path runs on
  vLLM's stock layer implementations on NPU).

## #1 sliding-window compute (CLOSED)

Two mechanisms, dispatched by `TileInferAttentionImpl.forward`:

- **Decode — staged kernel path** (`tileinfer/vllm/attn/window.py`). The paged decode kernel is
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

## #4 ATEM reasoning parser (CLOSED)

The checkpoint's chat template ends the generation prompt at a bare `<|start|>assistant`; the
model emits ` to=self` reasoning `<|eom|>` ` to=user` answer `<|eot|>` (special tokens are
stripped, so the channel markers survive as plain text). `tileinfer/vllm/reasoning.py` registers
an `atem` reasoning parser (OOT, from the v2 plugin install; `--reasoning-parser atem`):
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

## #7 capacity-grid OOM at 0.82 utilisation (FIXED)

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

## Performance (real weights, single seq, eager)

- decode TPOT: **75–78 ms/tok** (~13 tok/s) after the staged kernel path (was 102–108 ms/tok with
  the masked-reference decode); shared staging does not change it (the gather was never the
  bottleneck — it saves ~90 MiB of duplicate scratch pools and 38 redundant gathers per step).
- windowed **prefill** still runs the masked reference path; long-prompt TTFT is the remaining
  performance item.

## Launch recipe (current, everything on)

```bash
source /home/userlist/zxs/TileInfer/scripts/env-ascend102.sh
env PYTHONPATH=/home/userlist/.ernest/TileInfer TILEINFER_OPS=none setsid nohup \
  /home/userlist/zxs/TileInfer/.venv_native213/bin/python -m vllm.entrypoints.openai.api_server \
  --model /home/models/Muse-Glimmer-30B --max-model-len 8192 --max-num-seqs 1 \
  --enforce-eager --gpu-memory-utilization 0.82 --block-size 128 \
  --reasoning-parser atem --port 8345
```

Note: use explicit `env` (not a heredoc `source /dev/stdin`) — the heredoc variant silently skips
the vLLM general-plugin load and fails at ModelConfig. The venv's editable dist-info
`entry_points.txt` was hand-restored to the v2 content (console_scripts / vllm.general_plugins /
vllm.platform_plugins); RECORD hash no longer matches (pip warns, functionality fine).

## v3 mapping (what the code must gain before this runs under the bridge)

The v3 tree has no landing spot for a model the vendor does not serve. Concretely:

1. `contracts/manifest.py` has no sliding-window concept (`AttentionSpec.kind` is "full";
   `layer_types` only allows `full_attention` / `linear_attention`) — needs a sliding layer type
   and a window field.
2. `manifests/` contains only `qwen3_6` — needs `muse_glimmer.py`.
3. `bridge/seams.py::install_model_table` is empty (`"no model replacement requested"`) — needs the
   real OOT registration (AutoConfig + ModelRegistry), which is exactly what the v2 plugin's
   `_install_models` + config shim do today.
4. `install_attention_selector` selects among vllm-ascend's registered backends; a vendor-absent
   model needs the seam to fall back to the TileInfer backend (registered in vLLM's
   `AttentionBackendEnum.CUSTOM`).

None of this can be device-validated on this box until vllm-ascend is installable (torch 2.10
venv) and adapted to MuseGlimmer; the v2 path above is the currently verified deliverable.

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
