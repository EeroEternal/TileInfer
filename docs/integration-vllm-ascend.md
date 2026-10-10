# Integrating TileInfer into vLLM-Ascend

Target: `--attention-backend TILEINFER` selects TileInfer instead of the CANN FIA path.

The runnable reference implementation of everything below is
[`examples/vllm_ascend_tileinfer_backend.py`](../examples/vllm_ascend_tileinfer_backend.py); this
document explains the design decisions behind it.

## Status (2026-10-10)

**Wired and selected, not yet usable.**  What is verified end to end on the reference box
(CANN 9.3.0 + vLLM 0.23 + vllm_ascend 0.23 (950) + tilelang 0.1.15, model served via the OpenAI API):

* the plugin registers and vLLM resolves the class (`AttentionBackendEnum.CUSTOM.get_class()` is
  `tileinfer.integrations.vllm_ascend.TileInferBackend`, checked at startup so a silent fallback
  cannot happen);
* `forward_impl` is entered for decode-only batches, and the plugin **declines** configurations its
  kernel does not support, with a logged reason (`head_size=64` for Qwen2.5-0.5B on the reference
  box) and falls back to the Ascend FIA path, which is the designed behaviour;
* the server serves coherent completions on the stock path (37 prompt + 24 generated tokens in 2.8 s
  on Qwen2.5-0.5B).

**Registered as an official vLLM plugin, and the decode path is reached.**  The correct mechanism is
an entry point, not an import hook:

```toml
[project.entry-points."vllm.general_plugins"]
tileinfer = "tileinfer.integrations.vllm_ascend:install"
```

vLLM calls it in *every* process that loads plugins (API server, EngineCore, workers), after its own
imports are done - which is what a `sitecustomize.py` cannot do: that runs before vLLM is importable,
`import vllm_ascend.ops` fails, and the swallowed exception leaves the backend unregistered with no
trace.  Verified: `importlib.metadata.entry_points(group="vllm.general_plugins")` lists
`tileinfer`, and with `TILEINFER_VLLM=1` the EngineCore log shows **our TileLang kernel being
compiled inside the engine** (the generated CCE with the CANN headers) - i.e. `forward_impl` reached
`_plan()`.

**Compilation moved off the request path.**  The plugin compiles in a **background thread** and serves
through FIA until the kernel is ready (`TileInferDecodeAttention.plan_if_ready`), because a serving
step must never block on a compile and a client will not wait minutes.  Verified in the engine: when
a background compile fails, the server keeps answering requests (the throwaway request that triggers
it returns normally, and the log says "staying on the FIA path") - no crash, no hang, no stall.

**Milestone reached: the kernel runs inside vLLM.**  With the three fixes below, the EngineCore builds
the plan and serves the decode step through TileInfer:

```
(EngineCore) TileInfer plan READY: bucket=1 max_pages=16 block_size=128
             -> <AttentionPlan decode backend=tilelang-ascend950 ...>
```

and the first decode step of Qwen2.5-1.5B (`head_dim=128`, group 6) produces the **same top-5
logprobs as the stock FIA operator, to four decimals** (`Tiles -0.0085, T -5.5085, Tile -6.6335,
A -6.7585, Sure -7.3835`).  Three things had to be fixed to get there, in order of discovery:

1. **registration mechanism** - the official `vllm.general_plugins` entry point (a `sitecustomize`
   hook runs before vLLM is importable and fails silently);
2. **the backend registry** - `_ensure_builtin_backends()` returned early once `reference` was
   registered, so a failure on a later backend left the registry permanently incomplete; the
   per-import diagnostics that found this are still in the code (KI-2);
3. **a bug in the TileLang wheel** - `tl_templates/ascend/numeric_limits.h` uses `std::bit_cast`
   without including `<bit>`, so *every* kernel compile fails on this CANN release with
   `no member named 'bit_cast' in namespace 'std'`.  `scripts/patch-tilelang-ascend-bitcast.py`
   rewrites those call sites to `__builtin_bit_cast` - which is what the colleague's working
   environment had already done by hand, and how this was spotted (the two `numeric_limits.h` files
   differed in exactly those nine lines).

**Performance is not measured yet, and the one number we have is not trustworthy.**  The A/B run that
produced the logprobs above still had 27 compile attempts failing in the background (the fixes landed
mid-run) and the plan only became READY at the very end, so its 91.7 s for 117 tokens is dominated by
compilation starving the engine, not by the kernel.  A clean measurement needs all buckets warmed first
and zero fallbacks during the measured window.

**The background compile is not free: it blocks the engine.**  A TileLang compile holds the Python
interpreter, so while the background thread compiles, the EngineCore stops answering - one A/B run
wedged that way.  The supported recipe is therefore to pre-warm the buckets **before** starting the
server (`scripts/prewarm-tileinfer-kernels.py`, see `architecture.md`), so the engine's own compile
call is a cache hit; the background thread stays as a safety net, not as the normal path.

**What is still open: the stock-FIA baseline crashed in that same A/B** (`EngineDeadError` after its
first step, on the same model and settings), which is unexplained and has to be looked at before the
two backends can be compared over a full generation.

**What is not solved: the backend is not visible in the EngineCore's registry.**  In the plugin's own
process `list_backends()` is `['reference', 'tilelang', 'tilelang-ascend950']`, but inside the
EngineCore the same call gives `['reference']` and `get_backend("tilelang-ascend950")` raises
`unknown backend ... known: ['reference']` **with no import error recorded** - i.e. the two later
backends neither registered nor failed to import, which the current code cannot produce by reading it
(the imports are independent, and a failure would be reported in the message).  The plugin therefore
falls back to FIA, exactly as designed, and the model serves.

Tested and eliminated: a second `tileinfer` on `PYTHONPATH` alongside the editable install (two module
objects would fill one registry and consult the other) - removing the `PYTHONPATH` entry changes
nothing.  The next step is one diagnostic run: log `list_backends()` and `_IMPORT_ERRORS` both at
plugin-load time and at plan time inside the EngineCore, which will say whether the modules are
imported but unregistered, or never imported.  `benchmarks/probes/` is the place for that probe.

**What is also not solved: the first-use compile inside the engine.**  Standalone that compile takes ~2
minutes; inside the EngineCore it was still running after 20+ minutes (the compiler's diagnostics
flood the EngineCore's logger, and the process is CPU-starved next to vLLM's workers).  A serving
process cannot compile on the first request anyway, so the fix is a **warm-up**: compile the buckets a
deployment will use at start-up (`build_decode_kernel(...)` for each bucket, with `TILELANG_CACHE_DIR`
warm), and keep the compile out of the request path.  Until that is done the integration stays
**opt-in** (`TILEINFER_VLLM=1`).

Two earlier claims are worth correcting explicitly:

* an intermittent device fault seen in the EngineCore was **not** ours - the same server, model and
  request with TileInfer switched off (`TILEINFER_DISABLE=1`) crashes identically, so that model
  (a hand-made tiny Qwen2) is at fault; the retraction is recorded in `known-issues.md`;
* the numerical A/B run *before* the plugin mechanism was fixed compared FIA against FIA (identical
  logprobs to 4 decimals), so it proved nothing about our kernel - it is not quoted as a result.

Four things about the platform cost real time and are worth knowing before touching this code:

1. `vllm_ascend.platform.NPUPlatform.get_attn_backend_cls` **ignores `--attention-backend`** for
   everything except FLASH_ATTN ("Ascend NPU will use its registered plugin backend instead"), so a
   registry entry alone changes nothing; the plugin wraps that method.
2. vLLM runs the engine in a **separate process**: a patch applied by the launcher reaches the API
   server only, so the registration and the shim live in a `sitecustomize.py` that every interpreter
   in the venv loads.
3. `import vllm_ascend.attention.attention_v1` first raises a circular-import error
   (`DeviceOperator`); `import vllm_ascend.ops` has to come first.
4. vLLM's usage-reporting thread dies on this box (`cpuinfo` JSONDecodeError) and takes EngineCore
   with it — `VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1` are required.

## Where it plugs in

```
vllm_ascend/
├── attention/
│   ├── attention_v1.py      # FIA path (AscendAttentionBackend)
│   ├── fa3_v1.py            # external flash_attn_npu_v3 ("CUSTOM"/"FLASH_ATTN")
│   ├── mla_v1.py, dsa_v1.py, sfa_v1.py
│   └── tileinfer_v1.py      # <- new: TileInfer backend
└── platform.py              # backend selection / registration
```

`fa3_v1.py` is the template: it subclasses `vllm.v1.attention.backend.AttentionBackend`, returns
`get_name()`, `get_impl_cls()`, `get_builder_cls()`, `get_kv_cache_shape()`,
`get_supported_kernel_block_sizes()`, and reuses vLLM's existing metadata builder type.  TileInfer
follows the same shape, with one difference: it owns its own metadata conversion instead of
reusing the FIA builder, because the whole point is to consume the *engine's* metadata directly.

## Three integration decisions

**1. KV layout: keep NHD.**  `get_kv_cache_shape()` returns
`(2, num_blocks, block_size, num_kv_heads, head_size)`, identical to the FIA path.  A model can
switch backends without re-allocating its cache, and TileInfer's kernels — which are written for
exactly this layout — need no repack at the boundary.  A fused `[2, ...]` cache is sliced into
`k_cache, v_cache` in `forward`, which is free (views, not copies).

**2. Metadata conversion happens once per batch shape, on the host.**  vLLM's
`block_tables` is `[batch, max_pages]` with padding; `plan_from_page_table` compacts it to
`kv_indptr` / `kv_indices` / `kv_last_page_len`.  Doing this in the plan (rather than per kernel
block) is what removes the padding walk from the kernel and gives the scheduler real information
to balance on.

**3. The plan is built before capture.**  vLLM's attention metadata builder runs on every step, so
`plan_from_page_table` is called there; it is a cache hit in steady state (same batch shape → same
plan) and the plan's device-side metadata buffers are refreshed by `run`.  Consequently, the
captured ACLGraph contains a `run` call with static shapes, no allocation and no compilation.

## Deviations to be aware of

* vLLM updates `block_tables` in place per step.  TileInfer's plan holds references to those
  buffers, so this works; if a future vLLM version starts allocating fresh metadata tensors per
  step, pass them to `run(kv_indptr=..., kv_indices=..., kv_last_page_len=...)` instead (supported).
* Decode-only today: for prefill chunks either keep the FIA path (dispatch on
  `AttentionMode`) or use `backend="reference"` as a correctness fallback.  Phase 2 wires the
  model-level fallback.
* Batch-size changes recompile.  Until batch bucketing lands, warm up the shapes a deployment
  actually uses (there is a warmup hook in the backend interface: `finalize_plan`).
