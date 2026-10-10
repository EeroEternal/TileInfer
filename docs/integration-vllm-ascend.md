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

**Blocked on KI-1**: the first decode *through the TileLang kernel* inside the EngineCore faults the
device (`vector core exception`, 507035) even though the identical shape passes standalone, so the
integration is **opt-in** (`TILEINFER_VLLM=1`) and must stay off until that is fixed.  See
[`known-issues.md`](known-issues.md#ki-1--split-kv-wedges-the-device-when-several-shapes-share-one-process-open).

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
