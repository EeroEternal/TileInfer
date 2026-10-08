# Integrating TileInfer into vLLM-Ascend

Target: `--attention-backend TILEINFER` selects TileInfer instead of the CANN FIA path.

The runnable reference implementation of everything below is
[`examples/vllm_ascend_tileinfer_backend.py`](../examples/vllm_ascend_tileinfer_backend.py); this
document explains the design decisions behind it.

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
