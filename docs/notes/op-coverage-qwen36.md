# Operator coverage for the Qwen3.6 forward path (vLLM 0.23.0, out-of-tree platform)

**Question this answers.** For every operator the Qwen3.6 model actually calls, is it

* **bound** - we replace it (`tileinfer/vllm/ops/families/gdn_moe.py::required_names()` or the patch
  registry), or the implementation behind it is ours (the attention backend);
* **fallback** - vLLM's own Python/torch path runs it on an out-of-tree (OOT) platform, because the
  operator is a `CustomOp` whose OOT dispatch lands on a real `forward_native` (or the call site is
  plain `torch`); or it is a `torch.ops._C`/Triton call that Qwen3.6 does not take (the path is
  named in the row);
* **MISSING** - `torch.ops._C`/`_moe_C`/Triton with no native path, or a Python path that reaches
  `None`/`assert` because no OOT implementation exists, and nothing of ours binds it. It raises on
  the first real step.

**Verdict: no operator is MISSING.** The routed-expert MoE kernel - the last one - is now **bound**:
`unquantized_fused_moe` registers an out-of-tree `UnquantizedFusedMoEMethod` subclass through
`CustomOp.register_oot`, which is the extension point 0.23 documents for exactly this class, and
routes the expert compute to the existing torch port (`tileinfer/vllm/ops/moe.py:fused_experts`).
Everything a dense-architecture Qwen3.6 and every *non-MoE* layer of the MoE architecture does is
bound or falls back to real torch, and the MoE architecture now completes a forward step on an OOT
platform without a CUDA/Triton kernel. See [The operator that was MISSING](#the-operator-that-was-missing)
for the failure it removes, the exact source lines the replacement depends on, and what would delete
it.

## Sources read (not memory)

Every line number below refers to the tag-`v0.23.0` file of that name in `reference/vllm-0.23/`
(`__`-separated filename convention). Files fetched for this note are listed in the
[source map](#source-map); the ones that were already present are unchanged.

Method: start at `vllm/model_executor/models/qwen3_5.py`, follow every `__init__`/`forward` edge,
and for each leaf decide the class from the **dispatch rule** below plus the callee's body - never
from the operator's name.

## The two rules that decide everything

### 1. `CustomOp` dispatch on an OOT platform

```
custom_op.py:185   enabled = self._enforce_enable or self.enabled()
custom_op.py:191-194   if not enabled:  return self.maybe_compile(self.forward_native, ...)
custom_op.py:196-207   rocm->forward_hip | cpu->forward_cpu | tpu | xpu | OOT->forward_oot | else forward_cuda
custom_op.py:169-172   def forward_oot(self, ...): return self.forward_native(*args, **kwargs)
custom_op.py:138-144   def forward_native(self, ...): raise NotImplementedError
```

So for `TileInferPlatform.is_out_of_tree() == True` (pinned by
`tests/test_platform_contract.py:815`) the ladder ends at **`forward_native`**, and:

* a subclass with a real `forward_native` → **fallback**, and it is safe;
* a subclass with only `forward_cuda` (i.e. the inherited `forward_native` that raises) →
  **MISSING**, unless we bind it.

`maybe_compile` is a no-op here: our platform sets `simple_compile_backend = "eager"`
(`tileinfer/vllm/platform.py`), and `custom_op.py:229-231` returns the function unchanged for the
eager backend.

### 2. 0.23.0 added an explicit OOT branch to MoE, and it is a *stub*

```
oracle/unquantized.py:168-169   if current_platform.is_out_of_tree():
                                    return UnquantizedMoeBackend.OOT, None      # <- experts_cls is None
unquantized_fused_moe_method.py:55   self.unquantized_backend, self.experts_cls = select_unquantized_moe_backend(...)
unquantized_fused_moe_method.py:217-222   if self.unquantized_backend in [TPU, OOT]:  # OOT handles internally
                                              return                                   # <- no kernel is ever built
fused_moe_method_base.py:156-161   if self.moe_kernel is None:
                                       if hasattr(self, "experts_cls"): return self.experts_cls.is_monolithic()
```

"OOT handles internally" means *the platform is expected to register an OOT subclass* - that is the
documented extension point, and vLLM's own docstring names this exact class
(`custom_op.py:326-329`: `@UnquantizedFusedMoEMethod.register_oot class HPUUnquantizedFusedMoEMethod`).
Stock 0.23 registers none (`grep register_oot tileinfer/` → only this ledger's consumer now), so the
OOT branch lands on `None`. We now register one - `unquantized_fused_moe`, see
[The operator that was MISSING](#the-operator-that-was-missing) - and binding it is a **class-level**
act: `install.py` grew a third patch kind, `kind="custom_op_oot"`, because neither of the other two
can express it (see the [why](#the-operator-that-was-missing)).

## A. Model shell (embedding / final norm / logits)

| operator | call site (file:line) | class | evidence | consequence if wrong |
|---|---|---|---|---|
| `F.embedding` (VocabParallelEmbedding.forward) | `vllm__model_executor__models__qwen3_next.py:502-503` → `...vocab_parallel_embedding.py:477-497` → `:77-78` | fallback | `UnquantizedEmbeddingMethod.embedding` returns `F.embedding(input_, layer.weight)` (`vocab_parallel_embedding.py:77-78`); `VocabParallelEmbedding` is a `PluggableLayer`, plain `nn.Module` forward, no `torch.ops` | would mean the embedding table is a kernel; it is not |
| `F.linear` (lm_head projection) | `qwen3_5.py:515-519` → `logits_processor.py:96` → `vocab_parallel_embedding.py:67-75` | fallback | `dispatch_unquantized_gemm()` returns `default_unquantized_gemm` for anything not ROCm/CPU (`layers/utils.py:321-327`), whose body is `F.linear(x, weight, bias)` (`:92-98`) | ditto |
| `GemmaRMSNorm` → `ir.ops.rms_norm` | `qwen3_5.py:173-178,244`; `qwen3_next.py:283-284,379-384,498,541` → `layernorm.py:137-172` | fallback | `GemmaRMSNorm.forward_cuda` calls `forward_native` (`layernorm.py:167-172`), and `forward_native` calls `ir.ops.rms_norm` (`:156-165`); that IrOp's `native` implementation is `x.pow(2).mean(...)` in pure torch (`vllm__ir__ops__layernorm.py:9-21`) and dispatch falls back to it when no priority list is set (`vllm__ir__op.py:334-342`) | a norm that silently rescales every activation |
| `LogitsProcessor.forward` | `qwen3_5.py:515-519` → `logits_processor.py:54-104` | fallback | body is `quant_method.apply` + `tensor_model_parallel_all_gather` + a slice; `soft_cap`/`scale` are `torch.tanh`/`*=` | wrong logits, not a crash |

## B. Full-attention layer (`Qwen3NextAttention`, the non-GDN layers)

| operator | call site (file:line) | class | evidence | consequence if wrong |
|---|---|---|---|---|
| `QKVParallelLinear` (qkv_proj) | `qwen3_next.py:242-250,292` | fallback | `UnquantizedLinearMethod` → `dispatch_unquantized_gemm()` → `F.linear` (`layers/utils.py:321-327`) | - |
| `GemmaRMSNorm` (q_norm, k_norm) | `qwen3_next.py:283-284,306-311` | fallback | same as row A3 | - |
| `MRotaryEmbedding` / `RotaryEmbedding` | `qwen3_next.py:260-265,313` → `rotary_embedding/base.py:203-219`, `rotary_embedding/mrope.py:263-322` | fallback | `RotaryEmbeddingBase` is `@CustomOp.register("rotary_embedding")` (`base.py:14-15`); `forward_native` is `forward_static`, pure torch (`base.py:161-219`); `MRotaryEmbedding.forward_native` is pure torch and calls `apply_rotary_emb.forward_native` **explicitly** (`mrope.py:305,316`), i.e. it bypasses the cuda/Triton `triton_mrope` branch (`mrope.py:343`) | cos/sin applied with the wrong layout |
| `ApplyRotaryEmb` | `mrope.py:305,316` → `rotary_embedding/common.py:123-225` | fallback | `@CustomOp.register("apply_rotary_emb")`, `forward_native` = `forward_static` (`common.py:216-225`), pure torch | - |
| `unified_attention_with_output` → **our** `TileInferAttentionImpl.forward` | `attention/attention.py:437-529,751`; registration `:777-782` | bound | the op is Python-registered by `direct_register_custom_op` (`:777-782`), and its body calls `self.impl.forward(...)` (`:751`); the impl comes from our backend (`TileInferAttentionBackend`, `forward_includes_kv_cache_update = True`, `tileinfer/vllm/attn/backend.py:107`) | no attention at all |
| `maybe_calc_kv_scales` | `attention/attention.py:456-459` (guarded), `:613-628` | fallback (not taken) | the branch requires `self.calculate_kv_scales`, which is `cache_config.calculate_kv_scales` (`:226`) - False for a bf16 cache; the op is Python-registered anyway (`:641-644`) | - |
| `unified_kv_cache_update` | `attention/attention.py:490-508` (guarded) | fallback (not taken) | guarded by `not self.attn_backend.forward_includes_kv_cache_update` (`:493`), which is **False** for our backend (`attn/backend.py:107`), so our impl does the write | would be a second, redundant KV write |
| `RowParallelLinear` (o_proj) | `qwen3_next.py:252-258,321` | fallback | `F.linear` + all-reduce | - |

## C. GDN / linear-attention layer (`QwenGatedDeltaNetAttention`)

The layer is a `PluggableLayer`, and on our platform `__init__` pins `self._forward_method =
self.forward_cuda` (`qwen_gdn_linear_attn.py:451-463`) - the *name* says cuda, the body is plain
torch plus one Python-registered op (`:908-967`). The GDN math itself is the FLA chain ours.

| operator | call site (file:line) | class | evidence | consequence if wrong |
|---|---|---|---|---|
| `torch.ops.vllm.qwen_gdn_attention_core` | `qwen_gdn_linear_attn.py:956` → body `:1699-1736`, registration `:1751-1756` | fallback | registered by `direct_register_custom_op` (`:1751-1756`); it resolves the layer from `forward_context.no_compile_layers` and calls `self._forward_core` (`:1720-1736`) | - |
| `causal_conv1d_fn` | `qwen_gdn_linear_attn.py:1365-1375` | **bound** | upstream body launches `_causal_conv1d_fwd_kernel` (`mamba/ops/causal_conv1d.py:16`, wrapper `:468`); no `torch.ops._C`, no native branch. Family target `...mamba.ops.causal_conv1d.causal_conv1d_fn` + the consumer alias `_GDN.causal_conv1d_fn` (`gdn_moe.py:249-264`) | prefill conv raises (Triton on a device it was not built for) |
| `causal_conv1d_update` | `qwen_gdn_linear_attn.py:1344-1357,1378-1388,1674-1682` | **bound** | `_causal_conv1d_update_kernel` (`causal_conv1d.py:749`, wrapper `:1069`) - Triton only | decode raises on the first step |
| `fused_post_conv_prep` | `qwen_gdn_linear_attn.py:1428-1439` (also warmup `:1113-1124`) | **bound** | body is `_fused_post_conv_kernel` (`fused_gdn_prefill_post_conv.py:20`) + launcher `:152`; ten-argument signature our port reproduces; the layer passes `apply_l2norm=True` and the chunk call passes `use_qk_l2norm_in_kernel=False` (`:1529`), so q/k normalisation happens **only** here | prefill cannot even build its inputs |
| `chunk_gated_delta_rule` (class `ChunkGatedDeltaRule`) | `qwen_gdn_linear_attn.py:1518-1530` (call), `:554` (build) | **bound** | `@CustomOp.register("chunk_gated_delta_rule")` (`:290`); `__init__` pins `self._forward_method = self.forward_native` when the backend is not flashinfer/cutedsl (`:306-311`), and that `forward_native` is upstream's **Triton** chain (`:345-373` → `fla__ops__chunk.py:138`). Our takeover injects `forward_native` **and** `forward_oot` (`ops/install.py:219-303`) | prefill raises (the "native" name is misleading - it is Triton) |
| `fused_sigmoid_gating_delta_rule_update` | `qwen_gdn_linear_attn.py:1457,1484,1542` | **bound** | body is one Triton kernel (`fused_sigmoid_gating.py:24`, wrapper `:181`) | every decode step raises |
| `fused_recurrent_gated_delta_rule_packed_decode` | `qwen_gdn_linear_attn.py:1684-1695` (only when `VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE`, `:557-559`) | **bound** | Triton kernel (`fused_recurrent.py:256`, wrapper `:339`) | decode raises when the env flag is on |
| `l2norm_fwd` | imported by the GDN module (`:32`) and by `chunk.py:17` | **bound** | Triton kernels (`l2norm.py:28,59,79`, wrapper `:95`). On the Qwen3.6 path it is reached only *inside* the two replaced ops (both are called with `use_qk_l2norm_in_kernel=True`, `:1473,1498,1557`) and inside `fused_post_conv_prep`, so the binding is a belt-and-braces ABI guard rather than a direct call | only if the GDN family is switched off |
| `chunk_local_cumsum` | `fla__ops__chunk.py:16` (only importer) | fallback (unreachable) | our takeover replaces `chunk_gated_delta_rule` wholesale, and `qwen_gdn_linear_attn.py` never calls `chunk_local_cumsum` directly (it is not in the `:24-33` import block). The family reports it unbound on purpose - ABI mismatch, `gdn_moe.py:313-332` | none today; becomes live if the chunk takeover is ever removed |
| `RMSNormGated` | `qwen_gdn_linear_attn.py:536-543,866` | fallback | the class used is `layers.layernorm.RMSNormGated` (`:34`), a `CustomOp` whose `forward_native` is the pure-torch `forward_static` (`layernorm.py:226-289`); only `forward_cuda` imports the Triton `rmsnorm_fn` (`:291-305`). The `fla/ops/layernorm_guard.py:374` class is **not** used by this model | GDN output unscaled / raises |
| GDN projections: `in_proj_qkvz`, `in_proj_ba`, `out_proj`, `conv1d` | `qwen_gdn_linear_attn.py:467-472,480-552,869` | fallback | `MergedColumnParallelLinear` / `ColumnParallelLinear` / `RowParallelLinear` → `F.linear` | - |
| `fused_gdn_gating` (module-level Triton) | defined `qwen_gdn_linear_attn.py:1795-1827` | fallback (dead code) | grep over the whole 0.23 tree: only its own definition and its own kernel launch; Qwen3.6's decode uses `fused_sigmoid_gating_delta_rule_update` instead (`:1457`) | none |

## D. MoE layer (`Qwen3NextSparseMoeBlock` → `FusedMoE` → `MoERunner`)

| operator | call site (file:line) | class | evidence | consequence if wrong |
|---|---|---|---|---|
| `gate` (ReplicatedLinear, F.linear) | `qwen3_next.py:125-139,186-190` (the runner owns it: `moe_runner.py:307-308` reports `is_internal_router` true) → `moe_runner.py:773-778` | fallback | `F.linear`; the block then passes `router_logits=hidden_states` and the runner computes the logits itself (`moe_runner.py:776-778`) | - |
| `torch.ops.vllm.moe_forward` | `fused_moe__runner__moe_runner.py:270-281` (selection), `:90-105` (body), registration `:169-184` | fallback | Python-registered, body dispatches to `layer.runner._forward_impl` | - |
| **`unquantized_fused_moe`** (`UnquantizedFusedMoEMethod`, OOT subclass) | `unquantized_fused_moe_method.py:55` (select), `:311` (forward), `:217-222` (no kernel built); consumed at `layer.py:457` (load time) and `moe_runner.py:516` / `:532` (per step) | **bound** | `tileinfer/vllm/ops/fused_moe_oot.py` registers an OOT subclass through `UnquantizedFusedMoEMethod.register_oot` (`custom_op.py:331-353`; the documented example is this class, `:326-329`), installed by `install.py` kind `custom_op_oot` (`families/gdn_moe.py`, op `unquantized_fused_moe`). The subclass replaces the `None` the oracle returns (`oracle/unquantized.py:168-169`) with `TorchExperts`, whose `is_monolithic()` is `False` (`fused_moe_method_base.py:157-158`), and fills the kernel slot the OOT branch never builds (`:217-222`) with `TorchMoEKernel`, whose `is_monolithic` is a **False attribute** (`:161`) and whose `apply` calls `tileinfer/vllm/ops/moe.py:fused_experts` (`:311-324`). `supports_internal_mk` is then True, so `layer.py:457` returns before `maybe_init_modular_kernel` can reach the raising `maybe_make_prepare_finalize` (`:70-78`); the step goes `router.select_experts` → `apply` → `forward_oot` → `forward_native` → `moe_kernel.apply` (`moe_runner.py:524-539`). Quantised/bias/LoRA/non-gated/DP-EP-PCP configs are refused at construction. | without it: `AttributeError: 'NoneType' object has no attribute 'is_monolithic'` at load time; with it: the torch port |
| `topk_softmax` | `fused_moe__router__fused_topk_router.py:98` → `vllm___custom_ops.py:2494-2509` | **bound** | the wrapper is an unconditional `torch.ops._moe_C.topk_softmax(...)` (`:2502`) with **no** torch branch; our target is the module attribute (`gdn_moe.py:361-381`) and the router calls it as `ops.topk_softmax` (`fused_topk_router.py:24`), i.e. through the module | routing raises `AttributeError`/`RuntimeError` on `_moe_C` |
| `moe_align_block_size` | `fused_moe__moe_align_block_size.py:90` → `vllm___custom_ops.py:2346-2363` | **bound** | same shape: unconditional `torch.ops._moe_C.moe_align_block_size` (`:2355`). The 0.23 call site goes through `ops.moe_align_block_size` (module attribute) from the *layer helper* module, so our module-attribute patch intercepts it as well; the family's older note about `.../fused_moe/fused_moe.py:1465` is the same helper | raises once a Triton/MK expert path asks for the slot layout |
| `fused_experts` | defined `fused_moe__fused_moe.py:1474-1515` (→ `torch.ops.vllm.fused_experts` `:1490`) | **bound (reached through `moe_kernel.apply`)** | grep of the whole local tree: the Python wrapper is imported for re-export (`fused_moe____init__.py:118-121,156`), and `moe_runner.py:447` only reads a *kernel attribute* of the same name. Nothing calls the wrapper directly: 0.23's modular `MoERunner` goes `router.select_experts` → `quant_method.apply` → `moe_kernel.apply` (`moe_runner.py:524-539`). Since this ledger's ratchet was last run, the *consumer* exists and is ours: `fused_moe_oot.TorchMoEKernel.apply` is the `moe_kernel` of the `unquantized_fused_moe` record below, and its body is this same function object (`tileinfer/vllm/ops/moe.py:fused_experts`, reused by identity - `tests/test_fused_moe_oot.py` pins `is`) | on its own the wrapper binding changes nothing on 0.23 - it is the *body* the OOT method calls through the kernel slot; do not mistake it for coverage of the wrapper name |
| `SiluAndMul` (`silu_and_mul`, shared-expert MLP) | `qwen2_moe.py:111,116` via `qwen3_next.py:147-156` and the dense branch `qwen3_5.py:162-169` | fallback | `@CustomOp.register("silu_and_mul")` (`activation.py:117`); `forward_native` is a `@staticmethod` with `F.silu(x[..., :d]) * x[..., d:]` (`:137-141`); `self.op = torch.ops._C.silu_and_mul` is set **only** `if current_platform.is_cuda_alike() or is_xpu()` (`:130-135`), so on OOT there is no `_C` op to call and the ladder takes `forward_oot` → `forward_native` | the family's `[unverified]` note is now superseded - see [Corrections](#corrections-to-existing-records) |
| shared-expert gate (`F.linear` + `F.sigmoid`) | `qwen2_moe.py:119-120` | fallback | plain torch | - |

## E. Dense MLP (`Qwen3.5`/`3.6` non-MoE architecture)

| operator | call site (file:line) | class | evidence | consequence if wrong |
|---|---|---|---|---|
| `MergedColumnParallelLinear` / `RowParallelLinear` | `qwen2_moe.py:90-106,115-117` | fallback | `F.linear` | - |
| `SiluAndMul` (`silu_and_mul`) | `qwen2_moe.py:111,116` | fallback | same as row D7 | - |

## F. Sampler path (`vllm/v1/sample/`)

| operator | call site (file:line) | class | evidence | consequence if wrong |
|---|---|---|---|---|
| `apply_top_k_top_p` → Triton or torch sort | `sampler.py:286-291` → `ops__topk_topp_sampler.py:128-129,334-349` | fallback (with a Triton caveat) | `TopKTopPSampler.__init__` falls through every platform test and ends at `self.forward = self.forward_native` (`:128-129`); `forward_native` calls `apply_top_k_top_p` (`:143`), which takes `apply_top_k_top_p_triton` when `HAS_TRITON and logits.shape[0] >= 8` (`:345-346`) and the pure-torch sort otherwise (`:352-393`). `HAS_TRITON` is true whenever `triton` is importable **and exactly one driver is active** (`vllm__triton_utils__importing.py:14-51`) - with `triton-ascend` that can be true, and then batch>=8 sampling runs a Triton kernel | a garbled/incorrect token distribution at batch>=8; set the torch path if the Triton build is not trusted (nothing to bind - it is not a `torch.ops` symbol) |
| `random_sample` / `apply_temperature` / `compute_logprobs` | `sampler.py:227-237,304-306`; `topk_topp_sampler.py:419-504` | fallback | pure torch (`softmax`, `exponential_`, `div_`, `argmax`) | - |
| `apply_repetition_penalties` | `layers__utils.py:80-83` (via `sample__ops__penalties.py:11-32`) | fallback | **the wrapper has a torch branch**: `if logits.is_cuda and logits.is_contiguous(): cuda else: apply_repetition_penalties_torch` (`vllm___custom_ops.py:390-397`), and the torch body is the exact same math (`:360-373`). Also gated: `apply_penalties` returns early when `no_penalties` (`sampler.py:428-429`) | none - it works, only `repetition_penalty != 1` reaches it at all |
| `apply_bad_words` | `sampler.py:400-401` → `v1/sample/ops/bad_words.py:30-36` | fallback | plain Python + slice assignment (`:27`) | - |
| `batched_count_greater_than` | `sampler.py:219,347` → `v1/sample/ops/logprobs.py:10-27` | fallback | `@torch.compile(backend=current_platform.simple_compile_backend)` - i.e. `eager` here - over `(x >= values).sum(-1)` | - |
| builtin logits processors (min_p / logit_bias / min_tokens) | `sampler.py:282-284,404-405` → `logits_processor__builtin.py:101,161,234` | fallback | three `apply()` bodies, all pure torch, no `torch.ops` anywhere in the file | - |

## Totals

Totals: 12 bound / 28 fallback / 0 MISSING, over 40 rows.

By group: GDN 7 bound + 5 fallback; MoE 4 bound + 4 fallback; full attention 1 bound +
7 fallback; model shell 4 fallback; dense MLP 2 fallback; sampler 6 fallback.

The mismatch worth remembering - `fused_experts` (D7) is bound but has no direct caller in 0.23.0 - is
now explained rather than open: the consumer is the OOT method's kernel slot (`moe_kernel.apply`),
which is the operator this note used to call MISSING. The two direct MoE bindings remain
`topk_softmax` (D5, the router) and `moe_align_block_size` (D6, the slot layout). That is precisely
how the MISSING row appeared: our 0.2x takeover point for the expert MLP moved out from under us when
0.23 replaced the `fused_experts` call with the modular `MoERunner` plus a per-platform kernel - and
it is why the fix is a *registered kernel class*, not a rebinding.

## The operator that was MISSING

<!-- BEGIN MISSING -->

<!-- END MISSING -->

The block above is empty **by construction**, and `tests/test_op_coverage_note.py` keeps it that way:
it fails if a `MISSING` row and this block disagree in either direction. The operator that used to be
listed here is now bound as `unquantized_fused_moe`
(`tileinfer/vllm/ops/fused_moe_oot.py`, registered by `families/gdn_moe.py` through the third patch
kind `kind="custom_op_oot"`).

**What it was.** The one MoE expert MLP, and the reason the engine could not start. Qwen3.6-MoE has a
MoE block in every layer that is not in `mlp_only_layers` (`qwen3_next.py:359-369`), and the failure
was a Python-level `None`/`assert`, not a numeric one - so it fired as soon as anything asked the
quant method whether it is monolithic, which happens at **model-load time**:
`gpu_model_runner.py:5193` calls `prepare_communication_buffer_for_model(self.model)` right after
`load_model`, and per `layer.py:451` that drives `FusedMoE.maybe_init_modular_kernel` (`:454-457`),
whose first statement reads `self.quant_method.is_monolithic`. Failing that, the profiling run
(`gpu_model_runner.py:6168` → `:6228 self._dummy_run(...)`) hit the same read through
`moe_runner.py:516`. Two independent failure points, both from one root (`experts_cls is None`, no
kernel built):

1. `is_monolithic` did `self.experts_cls.is_monolithic()` with `experts_cls is None`
   (`fused_moe_method_base.py:156-158`), because `select_unquantized_moe_backend` returns
   `(OOT, None)` (`oracle/unquantized.py:168-169`) → **`AttributeError`**;
2. if (1) were patched alone, `unquantized_fused_moe_method.py:311` `assert self.moe_kernel is not
   None` → **`AssertionError`**, because `process_weights_after_loading` returns early for OOT
   (`:217-222`) before `_setup_kernel` (`:263-273`).

**The fix, as built.** Register the OOT subclass vLLM documents - not loose methods, and not a
`@property` rebind.

*Mechanism (`kind="custom_op_oot"` in `install.py`).* `CustomOp.__new__` (`custom_op.py:109-128`)
substitutes `op_registry_oot[cls.__name__]` for **every** instantiation, and
`CustomOp.register_oot` (`:331-353`) is the documented way to fill that registry - its own example is
`@UnquantizedFusedMoEMethod.register_oot` (`:326-329`). `layer.py:359` constructs
`UnquantizedFusedMoEMethod(self.moe_config)`, so registration changes what the loader builds without
touching a call site. The binding's `impl` is therefore a *factory* (``impl(base_cls) -> subclass``,
`fused_moe_oot.make_oot_method`), and the patch body calls `base_cls.register_oot(subclass)`. It is a
one-shot, process-wide act (vLLM has no `unregister_oot`), so the family switch is read once in
`apply()` instead of per call; the in-tree class is **not** modified (only `_tileinfer_oot_*` markers
are added), which is why the stock class still behaves as stock when the family is off.

*What the base class actually requires of `experts_cls`.* Exactly one member, and it is a **method**,
not an attribute: `fused_moe_method_base.py:157-158` does `hasattr(self, "experts_cls")` and then
`self.experts_cls.is_monolithic()`. So `fused_moe_oot.TorchExperts.is_monolithic()` returns `False`
(raises nothing, no arguments) - `True` would send the step to `apply_monolithic` with raw
`router_logits` (`moe_runner.py:516-522`), a different function. The other uses of an `experts_cls`
in the tree pass it to `make_unquantized_moe_kernel` (`oracle/unquantized.py:329-360`), which the OOT
branch never reaches, so nothing else is implemented. Note the trap the old sketch fell into: a plain
truthy **attribute** here is not "close enough", because the base class *calls* it -
`tests/test_fused_moe_oot.py` proves the plain-truthy variant raises `TypeError`.

*What else the source needs (the part the first sketch missed).* Making `is_monolithic` answer
`False` is **not sufficient**, and not even the right first step: with `moe_kernel` still `None`,
`layer.py:457` falls through to `ensure_moe_quant_config_init()` and then
`base_quant_method.maybe_make_prepare_finalize(...)`, which
`unquantized_fused_moe_method.py:70-78` *raises* from ("this class should not be called") - a second
crash in the same method, one line later. So the subclass fills the kernel slot the OOT branch leaves
empty: `moe_kernel = TorchMoEKernel()`. Then `supports_internal_mk` is True
(`fused_moe_method_base.py:38-41`), `layer.py:457` returns immediately, and the step takes the
modular branch `router.select_experts` → `apply` → `forward_oot` → `forward_native` → `moe_kernel.apply`
(`moe_runner.py:524-539`, `custom_op.py:204-205`). `forward_native` is **not** overridden: vLLM's own
body (`unquantized_fused_moe_method.py:311-324`) does the argument marshalling, so there is no copy of
it in our code to drift. The members the base and the runner read from that stand-in, each pinned by
a line reference and by a test:

| member | read at | ours |
|---|---|---|
| `is_monolithic` (**attribute**) | `fused_moe_method_base.py:161` | `False` |
| `can_overlap_shared_experts` | `fused_moe_method_base.py:43-49` → `shared_experts.py:92-112` | `False` (so the runner applies the shared expert itself, `moe_runner.py:512-514` / `:673-676`) |
| `output_is_reduced()` | `moe_runner.py:369-374` | `False` (unreduced output: the runner's TP all-reduce still runs) |
| `prepare_finalize.topk_indices_dtype()` | `fused_moe_method_base.py:137-138` | `None` (do not recast `topk_ids`) |
| `apply(**kwargs)` | `unquantized_fused_moe_method.py:311-324` | `tileinfer.vllm.ops.moe.fused_experts` |
| `apply_monolithic(...)` | `unquantized_fused_moe_method.py:373-374` | raises (never monolithic) |

Three further overrides exist for reasons that are visible in the source: `supports_eplb → False`
(the inherited `True` at `:66-68` advertises EPLB the stand-in cannot do; `layer.py:374` then refuses
loudly), `maybe_make_prepare_finalize → None` (belt and braces; `None` is what the base returns with
no all2all backend, `fused_moe_method_base.py:105-115`), and a `process_weights_after_loading` guard
that fails loudly if a future vLLM stops taking the OOT early return and builds a real kernel over
converted weights.

*The compute body is the operator we already had.* `TorchMoEKernel.apply` calls
`tileinfer.vllm.ops.moe.fused_experts` - the **same object** the `fused_experts` binding (D7) wraps,
reused by identity, not reimplemented. That is the "repoint" of D7: on 0.23 the wrapper name is still
dead, but the implementation is now reached through `moe_kernel.apply`, which is exactly the slot
0.23 left empty.

*Refusals.* `check_unquantized_config` refuses at construction (i.e. at load, before a weight is
read): a quantised marker (`quant_dtype`/`weight_quant_dtype`/`moe._QUANT_*`), an `in_dtype` outside
bf16/fp16/fp32, `has_bias`, `is_lora_enabled`, `is_act_and_mul=False`, and DP/EP/PCP > 1 (these last
because `supports_internal_mk` is True, so `moe_runner.py:682-706` skips the naive dispatch/combine and
our kernel does not implement it). `TorchMoEKernel.apply` repeats the dtype check on the weights, so a
quantised checkpoint cannot be cast into the fp32 port. TP > 1 is fine: the runner all-reduces the
combined output itself because `output_is_reduced()` is `False`.

**What remains unverifiable without hardware.** Everything above is a contract read off the local
0.23 sources and pinned by CPU tests; the following are *not* proven here:

* that the engine's loader really instantiates the quant method at the point `layer.py:359` shows
  (the code is local, but `gpu_model_runner.py` is not in `reference/vllm-0.23/`, so the call chain
  `prepare_communication_buffer_for_model` → `layer.py:451-457` is taken from the ledger's earlier
  reading, not re-read);
* that nothing *else* in the unvendored engine touches `quant_method.moe_kernel` expecting a real
  `mk.FusedMoEKernel` (graph capture / memory profiling paths). Our stand-in exposes only the members
  the local source reads;
* real numerics: the port is fp32-in/fp32-accumulate and casts the result, so bf16 expert output will
  differ from the Triton kernel by rounding - the CPU tests pin equality with our own reference, not
  with vLLM's kernel;
* that a full 30-layer Qwen3.6-MoE step fits/does not fit the NPU memory budget after
  `prepare_communication_buffer_for_model` asks a non-modular method for its buffer sizes.

**What deletes it.** A future upstream that builds an OOT `moe_kernel` itself - i.e.
`select_unquantized_moe_backend` returning a kernel class for `is_out_of_tree()`, or
`unquantized_fused_moe_method.py:217-222` stopping its early return. That is the binding's
`remove_when`, verbatim: **"upstream builds an OOT moe_kernel"**. Until then, `TILEINFER_OPS=moe`
(or the default set) is what makes the MoE checkpoints runnable on Ascend, and the module is deleted,
not adapted, when upstream catches up.

## Corrections to existing records

1. **`silu_and_mul` is verified, and the `dense` family is not needed for correctness.**
   `gdn_moe.py:433-449` carries `[unverified] - no activation module in reference/vllm-0.23`. The
   module is there now: `SiluAndMul` is called at `qwen2_moe.py:111/116` (shared expert and the
   dense MLP branch), and `activation.py:130-141` shows it has a pure-torch `forward_native` plus a
   `_C` op that is **only** installed on cuda-alike/xpu. So the operator is a *fallback* on Ascend,
   and binding it is a performance choice, not a crash fix. If it is ever bound, the target should be
   `vllm.model_executor.layers.activation.silu_and_mul` (the CustomOp name) as well as
   `vllm._custom_ops.silu_and_mul`, and `remove_when` becomes "when the dense family is enabled".
2. **`chunk_local_cumsum` is not merely "not a drop-in" - it is unreachable.** Its only importer is
   `fla__ops__chunk.py:16`, inside the chain our `chunk_gated_delta_rule` takeover replaces; the GDN
   layer never names it. The declined-ABI entry can stay, but it should stop being counted as a
   live gap (it is the single `unbound` entry in `report.unbound`, so `report.ok is False` for a
   reason that is now known to be harmless).
3. **`fused_experts` has no *direct* caller** in 0.23.0 (see D7). The wrapper name is still dead -
   `fused_moe.py:1474-1515` is a definition site, and 0.23 replaced its consumer with the modular
   `MoERunner` plus a per-platform kernel. The replacement consumer now exists and is ours:
   `fused_moe_oot.TorchMoEKernel.apply` is the `moe_kernel` of the `unquantized_fused_moe` record
   and calls the *same function object* (`tileinfer.vllm.ops.moe.fused_experts`) through the
   `moe_kernel.apply` slot (`unquantized_fused_moe_method.py:311-324`). So "bound" and "reachable"
   now agree for the implementation, while the wrapper *name* remains a no-op rebinding on 0.23 - the
   distinction the earlier note drew is still the right one, it just no longer hides a gap.

## Not covered by this ledger (deliberately)

* **The vision tower** (`Qwen3_VisionTransformer`, `qwen3_vl.py:578-583,792-797`): built for the
  multimodal architectures but never entered by a text-only request. Its op surface is much larger
  and is a separate ledger before image input is attempted.
* **Speculative decoding** (`fused_recurrent_gated_delta_rule`, `spec_*` GDN branches): off in v0;
  the GDN spec branch reuses `fused_sigmoid_gating_delta_rule_update`, which is bound.
* **Quantized paths** (`awq_marlin`, `gptq`, `fp8`, `mxfp4`): `supported_quantization = []` on our
  platform and `verify_quantization` rejects them, so their `_C`/`_moe_C` kernels are unreachable.
* **`torch.compile`**: `simple_compile_backend = "eager"` makes `CustomOp.maybe_compile` a no-op
  (`custom_op.py:229-231`), so no Inductor codegen is required for any row above.

## Non-operator dependencies on the same path

These are not operators and are not classified, but they sit between the rows above and would fail
the first step just as loudly:

* `tensor_model_parallel_all_gather` / `tensor_model_parallel_all_reduce`
  (`logits_processor.py:75-104`, `vocab_parallel_embedding.py:496`, `qwen2_moe.py` projections) - a
  TP group of size 1 must still exist and be initialised.
* `get_forward_context().attn_metadata` being a dict keyed by layer prefix for GDN
  (`qwen_gdn_linear_attn.py:1275-1284`) and `no_compile_layers[layer_name]` for the attention op
  (`attention.py:749`).
* `is_pin_memory_available()` (`sampler.py:68`) for the sampler's staging tensors.
* `HAS_TRITON` (`vllm__triton_utils__importing.py:14-51`): when it is false, the FLA/MoE modules
  still *import* (the placeholders supply `jit`/`cdiv`), but `vllm/model_executor/layers/fused_moe/
  __init__.py:150-157` replaces `fused_experts`/`fused_topk` with raising stubs and never imports
  `...fused_moe.fused_moe`, so the family's `fused_experts` target stops resolving (reported as
  `unbound`, a warning - not a crash).

## Source map

Fetched for this note into `reference/vllm-0.23/` (0.23.0 tag, `__` convention):

```
vllm/model_executor/models/qwen3_next.py        vllm/model_executor/models/qwen2_moe.py
vllm/model_executor/models/qwen3_vl.py          vllm/model_executor/layers/layernorm.py
vllm/model_executor/layers/activation.py        vllm/model_executor/layers/logits_processor.py
vllm/model_executor/layers/vocab_parallel_embedding.py
vllm/model_executor/layers/linear.py            vllm/model_executor/layers/utils.py
vllm/model_executor/layers/rotary_embedding/{__init__,base,common,mrope,dual_chunk_rope}.py
vllm/model_executor/layers/attention/{__init__,attention}.py
vllm/model_executor/layers/attention_layer_base.py
vllm/model_executor/layers/fla/ops/{layernorm_guard,fused_recurrent,fused_sigmoid_gating,cumsum,
    l2norm,utils,chunk,chunk_delta_h,chunk_o,chunk_scaled_dot_kkt,solve_tril,wy_fast}.py
vllm/model_executor/layers/mamba/{mamba_utils}.py
vllm/model_executor/layers/fused_moe/{__init__,config,activation,unquantized_fused_moe_method,
    fused_moe_method_base,moe_align_block_size,layer}.py
vllm/model_executor/layers/fused_moe/oracle/unquantized.py
vllm/model_executor/layers/fused_moe/experts/fallback.py
vllm/model_executor/layers/fused_moe/runner/{moe_runner,moe_runner_interface,shared_experts}.py
vllm/model_executor/layers/fused_moe/router/{__init__,base_router,fused_topk_router,router_factory}.py
vllm/ir/{__init__,op}.py  vllm/ir/ops/{__init__,layernorm}.py
vllm/triton_utils/{__init__,importing}.py
vllm/v1/sample/{sampler}.py  vllm/v1/sample/ops/{topk_topp_sampler,penalties,bad_words,logprobs}.py
vllm/v1/sample/logits_processor/builtin.py
```

`rotary_embedding` and `attention` are packages in 0.23.0 (not modules), and
`.../fused_moe/router/router.py` does not exist (`router_factory.py` is the factory).
