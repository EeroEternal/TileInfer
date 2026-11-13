# LFM3-1.7B adaptation evidence (route O)

Status: **C1 pass; C2 within declared tolerance; C3 pass; C4 pass (conc=1 reference; concurrency absolute).**

## Why upstream cannot serve it

- vLLM 0.23 has no `Lfm3ForCausalLM` in `_VLLM_MODELS`.
- The transformers-backend fallback loads the checkpoint but fails in `profile_run` at
  `modeling_lfm3.py:271` (`attn_output * gate`), because vLLM passes a flattened token
  layout while the HF attention assumes a batch dimension.
- The vendor state provider is GDN (`vllm_ascend.ops.gdn`); LFM3's recurrent layers are
  gated depthwise convolutions, which vllm-ascend does not implement.
- LFM3 also has two RoPE dimensions (64 on full-attention layers, 128 on sliding layers).
  The Ascend GQA rope path keeps one process-wide cos/sin buffer, which cannot represent
  both, and the vendor's buffer update asserts on the mismatch.

## What was added

| Layer | Change |
|---|---|
| L1 declaration | `tileinfer/manifests/lfm3_1_7.py` + registry: 24 layers (13 conv / 5 sliding / 6 full), 16 Q / 4 KV at head_dim 128, short-conv kernel 7, sliding window 256, per-layer RoPE, benchmark sample |
| L3 contracts | `AttentionSpec` gained `sliding_window` and `rope_by_layer`; new `StateSpec`; `ModelManifest` gained `state` and `model_impl`; layer types allow `conv`/`sliding_attention` |
| L2 decision | `decide()` adds a `model.arch` row for `model_impl`; `state.linear` stays vendor GDN for Qwen3.6 and selects `tileinfer.short_conv` only when the declaration says `kind="short_conv"` |
| L4 provider | `providers/state/short_conv.py` (pure torch causal depthwise conv + rolling state); `providers/catalog.py` declares it with evidence and delete condition |
| L5 bridge | `install_model_table()` registers the route-O class and applies the LFM3 config/rope compatibility patches; `install_impl_table()` recognises the short-conv provider as consumed by the model; `model_table` is critical when a model row exists |
| Model | `tileinfer/models/lfm3/model.py`: gated short conv with vLLM short-conv state metadata, sliding/full GQA with per-head output gate and QK RMSNorm, zero-tail partial RoPE, sandwich norms, SwiGLU, stacked qkv/w13 weight loading, short-conv state shape padded to the Ascend attention page size |
| Compat | `tileinfer/models/lfm3/compat.py`: transformers 5.16/5.17 shims for `conv` layer-type remapping and the nested rope dict; the vllm-ascend global rope buffer is disabled from the bridge |

## Reproduction

The run needs transformers 5.17 (the baseline 4.57 cannot parse the LFM3 config). Install
TileInfer into an isolated target so the real `vllm.general_plugins` entry point loads:

```bash
cd /home/userlist/ww/TileInfer_xjx1
/home/userlist/ww/py_venv/bin/pip install --no-deps --no-build-isolation \
    --target /tmp/tileinfer_pkg .
/home/userlist/ww/py_venv/bin/python -m tileinfer.cli.plan build \
    --model /home/models/LFM3-1.7B --out /tmp/lfm3-plan.json
```

The plan is `0279ee548e634470` (7 selections, 5 access points, `state.linear` from
`tileinfer.short_conv`, `model.arch` from `tileinfer.models.lfm3.model`). The plugin run:

```bash
PYTHONPATH=/tmp/tileinfer_pkg \
TILEINFER_PLAN=/tmp/lfm3-plan.json \
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
/home/userlist/ww/py_venv/bin/python -c "from vllm import LLM; ..."
```

Observed log (no manual `install()` call):

```text
[tileinfer] TileInfer v3 bridge: plan=0279ee548e634470 model=LFM3-1.7B selections=7 ...
seams=config_gate,graph_hook,attn_select,impl_table,model_table,method_wrap | skipped=none
engine ready 17.2
OUT: ' Paris.\n\nThe capital of Japan is Tokyo'
```

`tileinfer doctor` with the same plan: python / vllm / vllm_ascend / device_type / plan /
layer_guard / bridge / graph_metrics all PASS (captured=4, fallbacks=0).

## C2: reference comparison

Reference is HF transformers 5.17 on the same NPU (no vendor baseline exists):

```text
HF top5:  [(' Paris', -0.1842), (' the', -2.6842), (':\n\n', -4.3717), ('Paris', -4.4967), (':', -5.4342)]
vLLM top5: [(' Paris', -0.1810), (' the', -2.6810), ('Paris', -4.4310), (':\n\n', -4.4310), (':', -5.4310)]
```

The top-5 set matches and every logprob is inside the 0.1 tolerance declared in the
manifest; ranks 3/4 swap. Greedy generation matches the first two tokens then diverges
(`What is ...` vs `The capital of ...`) because those later logits are closer than the
remaining FIA-vs-SDPA numerical difference. This is **not** a bit-exact C2.

Layer-by-layer comparison (HF decoder-layer hooks vs vLLM forward hooks, same prompt) shows
the divergence is gradual and comes from attention/MLP accumulation, not from a wrong
weight or a broken convolution:

```text
layer 00 conv    max=0.00098 mean=0.000029
layer 10 full    max=0.01562 mean=0.000583
layer 18 full    max=0.06250 mean=0.002304
layer 22 full    max=0.05029 mean=0.005642
```

The final pre-norm layer 23 output has the same RMS (0.239) in both stacks; the 1.81 RMS
value is the post-`embedding_norm` output (HF's `last_hidden_state`), so an exact comparison
needs to apply the final norm to the vLLM output. The remaining difference is FIA vs SDPA
accumulation.

## C3: graph

The short-conv path was vectorised over `state_indices_*`/`query_start_loc_*` (no
per-request Python loop and no device sync), so graph capture now succeeds:

```text
capture_calls=1 captured_graphs=4 (PIECEWISE 2 + FULL 2)
decode_calls=7 hits=7 fallbacks=0 errors=0
```

Batch-2 generation is coherent in both eager and graph modes. A ~1K-token prompt with
`max_num_batched_tokens=512` (chunked prefill) and two requests also runs under graphs with
4 captured graphs and 0 fallbacks, which exercises the `has_initial_states_p` state path.

`tileinfer doctor` with the graph metrics snapshot:

```text
PASS python / vllm / vllm_ascend / device_type / plan / layer_guard / bridge / graph_metrics
OK
```

## C4: performance

After replacing the grouped `conv1d` decode and the non-contiguous state gather/scatter with
seven fused multiply-adds over a contiguous state slice, TPOT dropped from ~138 ms to
5-6 ms per token (a regression in the first implementation, not a model property).

vLLM/TileInfer, 512-token engine, 64 output tokens, 1/4/16 concurrent requests:

| prompt | conc | TTFT p50 / p90 | TPOT | throughput |
|---|---|---|---|---|
| 1024 | 1 | 195.8 / 195.8 ms | 6.29 ms | 107.9 tok/s |
| 1024 | 4 | 89.6 / 89.7 ms | 7.37 ms | 461.4 tok/s |
| 1024 | 16 | 285.9 / 363.4 ms | 7.90 ms | 1293.9 tok/s |
| 4096 | 1 | 73.7 / 73.7 ms | 5.27 ms | 157.7 tok/s |
| 4096 | 4 | 246.2 / 320.6 ms | 7.51 ms | 352.3 tok/s |
| 4096 | 16 | 766.0 / 1226.3 ms | 14.28 ms | 610.7 tok/s |
| 7680 | 1 | 138.7 / 138.7 ms | 5.30 ms | 135.3 tok/s |
| 7680 | 4 | 396.9 / 474.0 ms | 9.16 ms | 264.9 tok/s |
| 7680 | 16 | 1296.9 / 2126.2 ms | 23.29 ms | 364.3 tok/s |

HF transformers reference, conc=1: TTFT 349.7 / 78.9 / 148.9 ms and TPOT 13.77 / 11.69 /
11.73 ms for 1K / 4K / 8K. The conc=1 gate (TTFT and TPOT <= 1.10x HF) passes; concurrency
has no HF reference and is recorded as absolute throughput. Full graph capture: 8 graphs,
567/567 decode hits, 0 fallbacks, 0 errors.

## Limitations

- C2 is tolerance-based, not bit-exact; C4 has not been measured.
- Single card only: `Lfm3ShortConv` raises for TP > 1.
- Requires transformers 5.16+ (tested 5.17) because of the checkpoint's remote config code.
- The short-conv state is padded (2048 x 32 bf16) to satisfy vllm-ascend's page-size
  alignment; the math uses the first six slots.
