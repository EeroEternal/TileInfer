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
| L3 contracts | `AttentionSpec` gained `sliding_window` and `rope_by_layer`; new `StateSpec`; `ModelManifest` gained `state` and a module-style `model_impl`; layer types allow `conv`/`sliding_attention` |
| L2 decision | `decide()` keeps `model_impl` on the `Plan` and adds the `model_table` access point; `state.linear` stays vendor GDN for Qwen3.6 and selects `tileinfer.short_conv` only when the declaration says `kind="short_conv"` |
| L4 provider | `providers/state/short_conv.py` (pure torch causal depthwise conv + rolling state); `providers/catalog.py` declares it as a route-O-scoped provider with evidence and delete condition |
| L5 bridge | `install_model_table()` imports `tileinfer.bridge.models.lfm3` and calls its `register_config()` / `register_models()` actors; `install_impl_table()` recognises the short-conv provider as consumed by the model; `model_table` is critical when the plan requests it |
| Model | `tileinfer/bridge/models/lfm3_model.py`: gated short conv with vLLM short-conv state metadata, sliding/full GQA with per-head output gate and QK RMSNorm, zero-tail partial RoPE, sandwich norms, SwiGLU, stacked qkv/w13 weight loading, short-conv state shape padded to the Ascend attention page size |
| Registration | `tileinfer/bridge/models/lfm3.py`: `register_config()` applies the transformers/rope shims and disables the vllm-ascend global rope buffer; `register_models()` registers `Lfm3ForCausalLM` by class path |

## Reproduction

The run needs transformers 5.17 (the baseline 4.57 cannot parse the LFM3 config) and the
Ascend Triton package, which `vllm-ascend` imports instead of upstream Triton:

```bash
cd /home/xjx/TileInfer
source scripts/env-ascend102.sh
source .venv_lfm3/bin/activate
export PYTHONPATH=/home/zxs/triton-ascend-3.2.2:$PYTHONPATH
python -m tileinfer.cli.plan build \
    --model /home/models/LFM3-1.7B --out /tmp/lfm3-plan-real.json
```

The plan rebuilt after the v3 `model_impl` merge is `d0ded082891fbb99` (7 selections,
6 access points, `state.linear` from `tileinfer.short_conv`,
`model_impl=tileinfer.bridge.models.lfm3`).  It has no `model.arch` row: the model is
registered through the module's `register_config()` / `register_models()` actors.

**P0/P1 update (2026-11-17):** the bridge now carries the shared `WeightMapSpec` and
`StateLayout`; LFM3's `load_weights()` uses `LFM3_WEIGHT_MAP` and its `_padded_state_len()`
uses `short_conv_state_layout()`. The vllm-ascend global-RoPE patch is registered as
`lfm3_rope_global_buffer` in `tileinfer/bridge/compat_hooks.py`. C1 was re-run after the
migration (same plan, same output); C2/C4 were not repeated as a full p50/p90 gate.

C1/C2/C3 were rerun with this plan on the real checkpoint:

```bash
TILEINFER_PLAN=/tmp/lfm3-plan-real.json \
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
python -c "from vllm import LLM; ..."
```

Observed log (no manual `install()` call):

```text
[tileinfer] TileInfer v3 bridge: plan=d0ded082891fbb99 model=LFM3-1.7B selections=7
(...) access_points=6 fallback=forbid | seams=config_gate,graph_hook,attn_select,
impl_table,model_table | skipped=none | metrics=/tmp/tileinfer-v3-metrics.json
OUT: ' Paris.\n\nThe capital of Japan is Tokyo.\n\nThe capital of Australia is Canberra'
```

`tileinfer doctor` with the same plan: python / vllm / vllm_ascend / device_type / plan /
layer_guard / bridge / graph_metrics all PASS.

## C2: reference comparison

Reference is HF transformers 5.17 on the same NPU (no vendor baseline exists):

```text
HF top5:  [(' Paris', -0.1842), (' the', -2.6842), (':\n\n', -4.3717), ('Paris', -4.4967), (':', -5.4342)]
vLLM top5: [(' Paris', -0.1854), (' the', -2.6854), (':\n\n', -4.3729), ('Paris', -4.4354), (':', -5.3729)]
```

The top-5 set matches and every logprob is inside the 0.1 tolerance declared in the
manifest. Greedy generation matches the first two tokens then diverges
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
capture_calls=1 captured_graphs=8 (PIECEWISE 4 + FULL 4)
decode_calls=16 hits=16 fallbacks=0 errors=0
```

(The earlier class-path run captured 4 graphs; the rebuilt plan under the module-style
`model_impl` captures 8.)

Batch-2 generation is coherent in both eager and graph modes. A ~1K-token prompt with
`max_num_batched_tokens=512` (chunked prefill) and two requests also runs under graphs with
4 captured graphs and 0 fallbacks, which exercises the `has_initial_states_p` state path.

`tileinfer doctor` with the graph metrics snapshot:

```text
PASS python / vllm / vllm_ascend / device_type / plan / layer_guard / bridge / graph_metrics
OK
```

## C4: performance

Re-measured on 2026-11-19 with the unified adapter/action pipeline, using the LFM3 OpenAI
server on one Ascend950PR, 64 output tokens per request.  Prompt sizes are the server's
actual token counts (the probe repeats text; 1024/4096/7680 nominal produce 781/3058/5698
tokens).  `perf_probe.py` streams and records TTFT/TPOT; concurrency has no HF reference and
is reported as absolute throughput.

| prompt tokens | conc | TTFT p50 / p90 | TPOT | throughput |
|---|---|---|---|---|
| 781 | 1 | 26.0 / 26.3 ms | 5.87 ms | 161.8 tok/s |
| 781 | 4 | 91.0 / 92.0 ms | 9.78 ms | 361.1 tok/s |
| 781 | 16 | 223.7 / 278.8 ms | 10.81 ms | 1119.8 tok/s |
| 3058 | 1 | 72.8 / 84.5 ms | 5.91 ms | 143.9 tok/s |
| 3058 | 4 | 248.0 / 270.2 ms | 10.55 ms | 288.4 tok/s |
| 3058 | 16 | 603.1 / 965.8 ms | 15.43 ms | 622.7 tok/s |
| 5698 | 1 | 125.5 / 135.4 ms | 5.96 ms | 128.0 tok/s |
| 5698 | 4 | 408.0 / 494.9 ms | 11.87 ms | 226.7 tok/s |
| 5698 | 16 | 1095.1 / 1757.3 ms | 20.46 ms | 405.2 tok/s |

Conc=1 TTFT/TPOT remain within the declared 1.10x HF reference gate; concurrency is
recorded as absolute throughput.  The older class-path numbers are superseded by this run.
The short-conv decode path is still the seven fused multiply-adds over a contiguous state
slice (grouped `conv1d` was ~138 ms TPOT).

## Limitations

- C2 is tolerance-based, not bit-exact (top-5 set matches, every logprob inside 0.1).
- C4 is measured on one card; concurrency has no HF reference and is absolute only.
- Single card only: `Lfm3ShortConv` raises for TP > 1.
- Requires transformers 5.16+ (tested 5.17) because of the checkpoint's remote config code.
- The short-conv state is padded (2048 x 32 bf16) to satisfy vllm-ascend's page-size
  alignment; the math uses the first six slots.
