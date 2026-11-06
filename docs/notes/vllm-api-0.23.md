# vLLM 0.23.0 API reference (verified from source, not memory)

Source: `raw.githubusercontent.com/vllm-project/vllm/v0.23.0/...`, stored in `runtime/vllm-api/`
(gitignored, one file per path with `__` as separator). Re-fetch:

```powershell
python .work/fetch_vllm_api.py
```

This file records only what was read out of the source. Anything not listed here is still unknown.

## Platform (`vllm/platforms/interface.py`, 36 KB)

**The `Platform` ABC is much larger than the port plan assumed: ~70 methods/attributes.**  The ones
a device backend must care about, grouped:

| Group | Methods seen in the ABC |
|---|---|
| identity | `device_name`, `device_type`, `dispatch_key`, `_enum`, `is_out_of_tree`, `is_*` predicates |
| device | `get_device_name`, `get_device_uuid`, `get_device_total_memory`, `set_device`, `device_id_to_physical_device_id`, `get_device_capability`… |
| memory | `get_current_memory_usage`, `is_pin_memory_available`, `inference_mode` |
| config | `check_and_update_config`, `apply_config_platform_defaults`, `pre_register_and_update`, **`update_block_size_for_backend`**, **`_align_hybrid_block_size`**, `register_custom_kv_cache_specs`, `verify_model_arch`, `verify_quantization` |
| backends | **`get_attn_backend_cls`**, `get_supported_vit_attn_backends`, `get_vit_attn_backend` |
| compile | `get_pass_manager_cls`, `get_compile_backend`, `get_compile_backend`, `import_kernels`, `import_ir_kernels` |
| distributed | **`get_device_communicator_cls`**, `use_all_gather`, `use_custom_allreduce` |
| misc | `get_infinity_values`, `can_update_inplace`, `opaque_attention_op`, `validate_request`, `supports_fp8`, `supports_mx`, `fp8_dtype` |

Two findings that change the plan:

1. **`update_block_size_for_backend` / `_align_hybrid_block_size` exist upstream** - the "engine
   block 2048 aligned with the mamba page" rule (which the old adapter reached around with sub-pages)
   is a first-class platform hook. So the platform should own the block-size decision instead of the
   attention backend compensating for it.
2. The surface is big enough that "implement `Platform` from scratch" is wrong; derive from the
   ABC, implement what raises, and let `check_and_update_config` assert the rest loudly.

## GDN / mamba path (`vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`, 68 KB)

Imports (verbatim):

```python
from vllm.model_executor.layers.fla.ops import (...)          # flash-linear-attention ops
from vllm.model_executor.layers.fla.ops.chunk import l2norm_fwd
from vllm.model_executor.layers.fla.ops.utils import FLA_CHUNK_SIZE
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (...)
from vllm.model_executor.layers.mamba.mamba_utils import (...)
from vllm.model_executor.layers.fla.ops import ... are_gdn_triton_kernels_available
```

**Correction to the earlier risk assessment**: this path is *not* `torch.ops._C`-only. vLLM 0.23
ships its own `vllm/model_executor/layers/fla/ops/` package plus a
`are_gdn_triton_kernels_available` gate, which strongly suggests a Triton path **with a fallback**.
That matters a lot: if a pure-PyTorch fallback exists, GDN correctness can be reached without
writing six kernels, and the kernels become a performance step rather than a hard gate.

**Next action (first thing next round)**: read `fla/ops/__init__.py`, `fla/ops/chunk.py`,
`mamba/ops/causal_conv1d.py` and grep for `are_gdn_triton_kernels_available` to determine exactly
which branch runs when Triton is absent. The op list for `tileinfer/vllm/ops/gdn.py` follows from
that, not from `torch.ops._C`.

### UPDATE (same round) - the "fallback" hypothesis is REFUTED

The gate is ROCm-only:

```python
GDN_AITER_TRITON_AVAILABLE = rocm_aiter_ops.are_gdn_triton_kernels_available()
```

and `fla/ops/__init__.py` re-exports a **Triton** FLA port:

```python
from .chunk import chunk_gated_delta_rule           # chunk_delta_h, chunk_o, chunk_scaled_dot_kkt,
from .fused_gdn_prefill_post_conv import fused_post_conv_prep   # cumsum, l2norm, solve_tril, wy_fast
from .fused_recurrent import (fused_recurrent_gated_delta_rule,
                              fused_recurrent_gated_delta_rule_packed_decode)
from .fused_sigmoid_gating import fused_sigmoid_gating_delta_rule_update
```

`chunk.py` imports `chunk_delta_h`, `chunk_o`, `chunk_scaled_dot_kkt`, `cumsum`, `l2norm`,
`solve_tril`, `wy_fast` - all Triton, no torch branch on the NPU path. So GDN really has **no native
fallback**: the work list is those seven FLA sub-ops plus `causal_conv1d` (prefill + update).
`tileinfer/testing/gdn_reference.py` and `tileinfer/kernels/gdn/` already cover part of it.

**Next action**: transcribe the seven signatures from `fla/ops/*.py`, implement torch versions in
`tileinfer/vllm/ops/gdn.py` (correctness first), then swap in the TileInfer kernels where measured
faster.

### Transcribed signatures (four of the seven fetched; files in `runtime/vllm-api/fla_*.py`)

```python
# l2norm.py (4.1 KB)
def l2norm_fwd(x: Tensor, eps: float = 1e-6, output_dtype: dtype | None = None)

# cumsum.py (8.1 KB)
def chunk_local_cumsum(g: Tensor, chunk_size: int, reverse: bool = False,
                       cu_seqlens: Tensor | None = None,
                       chunk_indices: Tensor | None = None, head_first: ...)
    # dispatches to chunk_local_cumsum_scalar / _vector

# solve_tril.py (19.5 KB)
def solve_tril(A: Tensor, cu_seqlens: Tensor | None = None,
               chunk_indices: Tensor | None = None, output_dtype: dtype = float)
    # block-inverse of (I + A), where A is STRICTLY lower triangular (A.triu() == 0):
    #   out = (I + A)^-1, NOT (I - A)^-1   <- verified against the kernel (b_A = -A; b_A += I)
    # 16x16 -> 32 -> 64 block-triangular merge: the whole BT x BT block is inverted, not just
    # its diagonal 16x16 sub-blocks
    # kernels: solve_tril_16x16_kernel, merge_16x16_to_32x32_inverse_kernel,
    #          merge_16x16_to_64x64_inverse_kernel

# fused_recurrent.py (21.3 KB)
def fused_recurrent_gated_delta_rule(q, k, v, g, beta=None, scale=None, initial_state=None)
def fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv, a, b, A_log, dt_bias, scale, initial_state, out)
    # the decode path: packed q|k|v in one buffer + per-token gate inputs
```

Still to fetch: `chunk_delta_h.py`, `chunk_o.py`, `chunk_scaled_dot_kkt.py`, `wy_fast.py`,
`fused_gdn_prefill_post_conv.py`, `fused_sigmoid_gating.py`, and for `causal_conv1d` the exported
functions of `mamba/ops/causal_conv1d.py` (49.5 KB, already downloaded).

**Not to guess**: the `eps` convention in `l2norm_fwd` and the `beta`/`g` layout in
`fused_recurrent_*` must be read out of the kernel bodies before implementing - a plausible-looking
normalisation with the wrong clamp silently changes the model's numerics.

## Attention (`vllm/v1/attention/backend.py`, 36 KB; `backends/registry.py`, 10 KB)

Both fetched; the exact `AttentionBackend` ABC, `AttentionImpl.forward` signature and
`CommonAttentionMetadata` fields still need transcribing (next round), plus
`vllm/v1/attention/backends/utils.py` for `PAD_SLOT_ID`.

## Not yet read

`vllm/v1/worker/gpu_worker.py` (50 KB), `vllm/compilation/cuda_graph.py` (14 KB),
`vllm/model_executor/models/qwen3_5.py` (30 KB - the model definition that tells us the real layer
schedule and config fields). All three are already downloaded into `runtime/vllm-api/`.
