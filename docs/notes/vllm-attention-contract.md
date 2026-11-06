# vLLM 0.23 attention contract, transcribed from the local source

Source of truth: `runtime/vllm-api/` (one file per upstream path, `__` as the separator), fetched
from `vllm-project/vllm@v0.23.0`; see `docs/notes/vllm-api-0.23.md` for provenance. **Every line
number below is a line in one of those local files.** Anything marked `[inferred] ` was *not*
established by a local file and must be re-checked on the device before it is relied on.

Files consulted, with sizes as fetched:

| file | upstream path | used for |
|---|---|---|
| `vllm__v1__attention__backend.py` (1044 lines) | `vllm/v1/attention/backend.py` | `AttentionBackend`, `AttentionImpl`, `CommonAttentionMetadata`, `AttentionMetadataBuilder`, `AttentionCGSupport` |
| `vllm__v1__attention__backends__registry.py` (266) | `vllm/v1/attention/backends/registry.py` | `AttentionBackendEnum`, `register_backend` |
| `vllm__v1__attention__backends__utils.py` (911) | `vllm/v1/attention/backends/utils.py` | `PAD_SLOT_ID`, decode/prefill splitting helpers, `get_kv_cache_layout` |
| `vllm__model_executor__custom_op.py` (353) | `vllm/model_executor/custom_op.py` | `CustomOp` dispatch rule |
| `vllm__platforms__interface.py` (1057) | `vllm/platforms/interface.py` | `get_attn_backend_cls` returns a *class path* |
| `vllm__model_executor__models__qwen3_5.py` (809) | `vllm/model_executor/models/qwen3_5.py` | the target model: layer schedule, `Qwen3NextAttention` |
| `vllm__model_executor__layers__mamba__gdn__qwen_gdn_linear_attn.py` (1828) | `.../mamba/gdn/qwen_gdn_linear_attn.py` | GDN layer (not this backend's business, but it is what the full-attention selector must *not* take over) |

---

## 0. One correction to `docs/plan-vllm-native-backend.md`

The port plan's W4 table (`docs/plan-vllm-native-backend.md:132`) lists `accept_output_buffer` as
part of `TileInferAttentionBackend`. **It does not exist in the local 0.23 source.**

```text
$ grep -n accept_output_buffer runtime/vllm-api/*   ->  no matches
```

The complete `AttentionBackend` surface is §1 below; the word appears nowhere in it. A small
forward-compatible hook returning `True` is kept in `tileinfer/vllm/attn/backend.py`
(`accept_output_buffer`, documented as not part of 0.23) because the impl genuinely does write
into the caller's `output` buffer (`:805`), so if a later version starts querying the hook the
answer is already right. It must not be described as a 0.23 API.

---

## 1. `AttentionBackend` — the abstract method list and the hooks

`vllm__v1__attention__backend.py:55` `class AttentionBackend(ABC)`.

**Exactly four abstract methods** (a subclass that omits any of them cannot be instantiated):

| line | signature |
|---|---|
| `:68-70` | *not abstract* — `@staticmethod def get_supported_kernel_block_sizes() -> list[int \| MultipleOf]: return [MultipleOf(1)]` |
| `:72-75` | `@staticmethod @abstractmethod def get_name() -> str` |
| `:77-80` | `@staticmethod @abstractmethod def get_impl_cls() -> type["AttentionImplBase"]` |
| `:82-85` | `@staticmethod @abstractmethod def get_builder_cls()` |
| `:87-96` | `@staticmethod @abstractmethod def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str="auto") -> tuple[int, ...]` |

The class attributes that participate in vLLM's own checks:

| line | member | value |
|---|---|---|
| `:58` | `supported_dtypes: ClassVar[list[torch.dtype]]` | `[torch.float16, torch.bfloat16]` |
| `:59-63` | `supported_kv_cache_dtypes: ClassVar[list["CacheDType"]]` | `["auto", "float16", "bfloat16"]` |
| `:65-66` | `forward_includes_kv_cache_update: bool` | `True` — "does attention's forward() include kv cache update?" |

Non-abstract hooks, with their defaults (these are what the TileInfer backend overrides):

| line | member | default |
|---|---|---|
| `:98-116` | `get_kv_cache_block_dim` | probes `get_kv_cache_shape` with `_S = 1234567` and returns `shape.index(_S)` |
| `:118-147` | `get_kv_cache_stride_order` | `raise NotImplementedError`; the docstring states the physical layout then **equals the logical shape** |
| `:153-155` | `get_supported_head_sizes` | `[]` (= no restriction) |
| `:157-160` | `supports_head_size` | `not supported or head_size in supported` |
| `:162-164` | `supports_dtype` | `dtype in supported_dtypes` |
| `:166-172` | `supports_kv_cache_dtype` | `None` → True; else membership |
| `:174-191` | `supports_block_size` | `None` → True; else `block_size % supported_size == 0` (a `MultipleOf` contributes its `.base`) |
| `:193-202` | `get_preferred_block_size` | default block size if accepted, else `min(...)` of the supported sizes |
| `:204-206` | `is_mla` | `False` |
| `:208-210` | `supports_sink` | `False` |
| `:212-214` | `supports_alibi_sqrt` | `False` |
| `:216-218` | `supports_mm_prefix` | `False` |
| `:220-222` | `is_sparse` | `False` |
| `:224-226` | `supports_per_head_quant_scales` | `False` |
| `:228-237` | `supports_non_causal` | `False` |
| `:239-241` | `supports_batch_invariance` | `False` |
| `:243-245` | `supports_kv_connector` | **`True`** (the one default that is wrong for an impl that writes the pool itself) |
| `:247-254` | `supports_attn_type` | `attn_type == AttentionType.DECODER` |
| `:256-258` | `supports_compute_capability` | `True` |
| `:260-272` | `supports_combination(...) -> str \| None` | `None` |
| `:274-341` | `validate_configuration(...) -> list[str]` | see below |
| `:343-345` | `get_required_kv_cache_layout` | `None` |
| `:347-349` | `is_ssm` | `False` |
| `:149-151` | `full_cls_name` | `(cls.__module__, cls.__qualname__)` |

`validate_configuration` (`:274-341`) — transcribed signature (`:275-291`):

```python
    @classmethod
    def validate_configuration(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: "CacheDType | None",
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        use_per_head_quant_scales: bool,
        device_capability: "DeviceCapability",
        attn_type: str,
        use_non_causal: bool = False,
        use_batch_invariant: bool = False,
        use_kv_connector: bool = False,
    ) -> list[str]:
```

It returns a **list of human-readable reasons**, empty meaning "supported" (`:292`, `:341`). The
generic checks are: head size (`:293-294`), dtype (`:295-296`), KV-cache dtype (`:297-298`), block
size (`:299-300`), mm prefix (`:301-304`), MLA (`:305-309`, note it is *bidirectional*: it also
rejects `use_mla=False` if the backend claims MLA), sinks (`:310-311`), sparse (`:312-316`),
per-head quant scales (`:317-318`), compute capability (`:319-320`), attention type (`:321-322`),
non-causal (`:323-324`), batch invariance (`:325-326`), KV connector (`:327-328`), and finally
`supports_combination` (`:329-340`).

**There is no `num_heads` / `num_kv_heads` parameter**, so the GQA group cannot be validated here.
That is why `TileInferAttentionBackend.check_head_layout` exists and is called from the impl's
`__init__` (model construction, still before the first step). `[inferred]` The claim "impl
construction is start-up" rests on vLLM building an impl per layer while loading the model; the
call site (`ModelRunner.load_model`) is **not** in the fetched subset.

---

## 2. `AttentionImpl` — constructor and forward, verbatim

`vllm__v1__attention__backend.py:770-771`:

```python
class AttentionImpl(AttentionImplBase[T], Generic[T]):
    """Standard attention implementation with forward method."""
```

Constructor (`:780-794`) — this is the argument list `TileInferAttentionImpl.__init__` copies:

```python
    @abstractmethod
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
    ) -> None:
        raise NotImplementedError
```

Forward (`:796-809`) — this is the signature `TileInferAttentionImpl.forward` copies **exactly**:

```python
    @abstractmethod
    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: T,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError
```

Notes that matter for an implementation:

* `output` is **required** and typed `torch.Tensor`, not `Optional` (`:805`): a correct backend
  writes into the engine's buffer. (Older vLLM versions had optional output buffers; 0.23 does
  not, which is the other reason `accept_output_buffer` is absent.)
* `output_scale` / `output_block_scale` (`:806-807`) are the fused-output-quantisation hooks;
  `fused_output_quant_supported` (`:811-823`) is how a backend advertises them. A bf16-only
  backend must reject them explicitly instead of ignoring them.
* `AttentionImpl` declares `kv_cache_dtype: str` (`:773`) and the property `kv_quant_mode`
  (`:775-778`), so an implementation must set `self.kv_cache_dtype`.
* `AttentionImplBase` (`:692-767`) supplies `num_heads` / `head_size` / `scale` as *declared*
  attributes (`:700-703`), `can_return_lse_for_decode` (`:705-707`), `supports_pcp` (`:709-710`),
  `need_to_return_lse_for_decode` (`:715-717`), `supports_quant_query_input` (`:719-727`) and the
  DCP/PCP rank attributes (`:729-736`).
* `AttentionImplBase.__new__` (`:738-764`) runs for **every** subclass and sets
  `dcp_world_size` / `dcp_rank` / `pcp_*`; it catches only `AssertionError` from
  `get_dcp_group()` / `get_pcp_group()` ("DCP might not be initialized in testing", `:746-749`).
  It also derives `need_to_return_lse_for_decode` (`:761-763`). It is **not** an `__init__`:
  `AttentionImplBase` has none, so an implementation must not call `super().__init__()` — that
  would land on `AttentionImpl.__init__`, whose body is `raise NotImplementedError` (`:794`).

`AttentionLayer` is a `Protocol` (`:673-689`) with `_q_scale` / `_k_scale` / `_v_scale`,
`_prob_scale` and their float forms, plus the five-argument `forward`. A backend that does not
implement quantisation may ignore the `layer` argument — but it must still accept it.

---

## 3. `CommonAttentionMetadata` — fields

`vllm__v1__attention__backend.py:359-422` (`@dataclass`), docstring at `:361-366`
("Per-batch attention metadata, shared across layers and backends ... For many of the tensors we
keep both GPU and CPU versions"):

| line | field | meaning (docstring quoted where present) |
|---|---|---|
| `:368` | `query_start_loc: torch.Tensor` | `(batch_size + 1,)`, start of each request in the query tensor (`:370`) |
| `:369` | `query_start_loc_cpu: torch.Tensor` | the host mirror of the same |
| `:372` | `seq_lens: torch.Tensor` | `(batch_size,)`, "the number of computed tokens for each request" (`:373`) |
| `:375` | `num_reqs: int` | "Number of requests" (`:376`) |
| `:377-378` | `num_actual_tokens: int` | "Total number of tokens in batch"; the `TODO(lucas)` at `:377` warns it may be padded |
| `:380` | `max_query_len: int` | "Longest query in batch" (`:381`) |
| `:382` | `max_seq_len: int` | "Longest context length (may be an upper bound)" (`:383`) |
| `:385` | `block_table_tensor: torch.Tensor` | dense page table |
| `:386` | `slot_mapping: torch.Tensor` | per-token KV write slots |
| `:388` | `causal: bool = True` | |
| `:390-391` | `logits_indices_padded`, `num_logits_indices` | for `FastPrefillAttentionBuilder` (`:390`) |
| `:394-396` | `encoder_seq_lens`, `encoder_seq_lens_cpu` | cross-attention (`:394`) |
| `:398-400` | `dcp_local_seq_lens`, `dcp_local_seq_lens_cpu` | decode-context-parallel local lengths (`:400`) |
| `:402-405` | `positions: torch.Tensor \| None = None` | optional `(num_actual_tokens,)` positions (`:403-405`) |
| `:407-410` | `is_prefilling: torch.Tensor \| None = None` | `(batch_size,)` bool, "Used by some backends to distinguish actual decodes from short extends" (`:410`) |
| `:412-416` | `seq_lens_cpu_upper_bound: torch.Tensor \| None = None` | CPU upper bound; "Not safe for kernels that need exact per-row context lengths on decode rows" (`:413-416`) |
| `:418-422` | `_seq_lens_cpu`, `_num_computed_tokens_cpu`, `_num_computed_tokens_cache` | deprecated / internal caches (`:418`: "Will be removed ... v0.15.0") |

Useful methods:

| line | method | note |
|---|---|---|
| `:424-425` | `batch_size()` | `seq_lens.shape[0]` |
| `:427-429` | `naive_query_lens()` | `query_start_loc[1:] - query_start_loc[:-1]` |
| `:431-432` | `replace(**kwargs)` | `dataclasses.replace` |
| `:434-445` | `seq_lens_cpu` property | **deprecated**, forces a H→D copy; prefer device `seq_lens` |
| `:447-462` | `num_computed_tokens_cpu` property | deprecated |
| `:464-469` | `compute_num_computed_tokens()` | `seq_lens - query_lens` — i.e. **`seq_lens` already includes this step's tokens** |
| `:471-500` | `unpadded(num_actual_tokens, num_actual_reqs)` | slices every tensor to the real batch |

`seq_lens` semantics are worth pinning: `:464-469` computes the *cached prefix* as
`seq_lens - query_lens`, so `seq_lens` is the KV length **after** the step's K/V is written. The
TileInfer step rebuild (`tileinfer/serving/step.py:148-231`) needs exactly this value.

Decode/prefill splitting helpers (all in `vllm__v1__attention__backends__utils.py`):

| line | function | rule |
|---|---|---|
| `:436-504` | `split_decodes_prefills_and_extends(cam, decode_threshold=1)` | `if max_query_len <= decode_threshold: return num_reqs, 0, 0, ...` (`:467-468`) |
| `:507-576` | `split_decodes_and_prefills(cam, decode_threshold=1, require_uniform=False, treat_short_extends_as_decodes=True)` | same `max_query_len` early-out (`:542-547`), then per-row `query_lens > decode_threshold` (`:562`) |
| `:367-433` | `make_kv_sharing_fast_prefill_common_attn_metadata` | also keys on `max_query_len == 1` (`:370-373`) |
| `:44` | `PAD_SLOT_ID = -1` | |
| `:45` | `NULL_BLOCK_ID = 0` | |
| `:41` | `KVCacheLayoutType = Literal["NHD", "HND"]` | |
| `:52-79` | `get_kv_cache_layout()` | env / KV-connector default; `HND` is possible |
| `:82-85` | `set_kv_cache_layout()` | override + cache clear |

Conclusion used by the backend: **decode vs prefill is `max_query_len > 1`**, as upstream itself
does it. `is_prefilling` is a per-request flag and cannot describe a batch whose query lengths
disagree.

---

## 4. `AttentionMetadataBuilder` and `AttentionCGSupport`

`vllm__v1__attention__backend.py:523` `class AttentionMetadataBuilder(ABC, Generic[M])`.

| line | member |
|---|---|
| `:526` | `_cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER` — "Do not access directly. Call `get_cudagraph_support()` instead." (`:525`) |
| `:529-530` | `reorder_batch_threshold: int \| None = None` — "Does this backend/builder reorder the batch? If not, set this to None." (`:526-530`) |
| `:532-533` | `supports_update_block_table: bool = False` |
| `:535-546` | `@abstractmethod def __init__(self, kv_cache_spec: "AttentionSpec", layer_names: list[str], vllm_config: "VllmConfig", device: torch.device)` — the body stores the four attributes (`:543-546`) |
| `:548-555` | `@classmethod get_cudagraph_support(cls, vllm_config, kv_cache_spec) -> AttentionCGSupport: return cls._cudagraph_support` |
| `:557-587` | `_init_reorder_batch_threshold(...)` — spec-decode threshold maths (`:564-581`), DCP forces 1 (`:583-587`) |
| `:589-607` | `@abstractmethod def build(self, common_prefix_len: int, common_attn_metadata: CommonAttentionMetadata, fast_build: bool = False) -> M` |
| `:609-622` | `update_block_table(self, metadata, blk_table, slot_mapping) -> M` — "Only needs to be implemented if `supports_update_block_table` is True" (`:620`) |
| `:624-634` | `build_for_cudagraph_capture(...)` — defaults to `build(common_prefix_len=0, ...)` |
| `:636-656` | `build_for_drafting(..., draft_index)` — defaults to `build(..., fast_build=True)` |
| `:658-670` | `use_cascade_attention(...) -> bool` — default `False` |

`AttentionCGSupport` (`:506-520`) — an `Enum` of ints, "constants for the cudagraph support of the
attention backend; here we do not consider the cascade attention":

| line | member | value | meaning (quoted) |
|---|---|---|---|
| `:511-512` | `ALWAYS` | `3` | "Cudagraph always supported; supports mixed-prefill-decode" |
| `:513-516` | `UNIFORM_BATCH` | `2` | batches whose query lengths are all the same ("decodes" are `1 + num_speculative_tokens`) |
| `:517-518` | `UNIFORM_SINGLE_TOKEN_DECODE` | `1` | "Cudagraph supported for batches the only contain query_len==1 decodes" |
| `:519-520` | `NEVER` | `0` | "NO cudagraph support" |

`AttentionMetadata` itself is an empty class (`:352-353`) — metadata types are structurally typed;
subclassing it is optional, and TileInfer does subclass it for readability.

---

## 5. The registry API

`vllm__v1__attention__backends__registry.py`.

```python
# :34  class AttentionBackendEnum(Enum, metaclass=_AttentionBackendEnumMeta):
# :44-99   members: FLASH_ATTN, TRITON_ATTN, ..., CPU_ATTN, TURBOQUANT
# :99-101  CUSTOM = None   # "Placeholder for third-party/custom backends - must be registered
#                          #  before use; set to None to avoid alias with other backend"
```

| line | member | behaviour |
|---|---|---|
| `:18-31` | `_AttentionBackendEnumMeta.__getitem__` | an unknown name raises `ValueError("Unknown attention backend: ... Valid options are: ...")` (`:28-31`) |
| `:103-120` | `get_path(include_classname=True)` | `path = _ATTN_OVERRIDES.get(self, self.value)`; falsy path → `ValueError(f"Backend {self.name} must be registered before use. Use register_backend(Backend.{self.name}, 'your.module.YourClass')")` (`:112-117`) |
| `:122-132` | `get_class()` | `resolve_obj_by_qualname(self.get_path())` (`:132`; imported at `:10`) |
| `:134-140` | `is_overridden()` | |
| `:142-144` | `clear_override()` | `_ATTN_OVERRIDES.pop(self, None)` |
| `:147-207` | `MambaAttentionBackendEnum` | same shape, separate override dict (`:211`) |
| `:210-211` | `_ATTN_OVERRIDES: dict[AttentionBackendEnum, str]`, `_MAMBA_ATTN_OVERRIDES` | the process-wide overrides |
| `:214-266` | `register_backend(backend, class_path=None, is_mamba=False)` | see below |

`register_backend` behaviour, transcribed:

* with `class_path=None` it returns a **decorator** that records
  `f"{cls.__module__}.{cls.__qualname__}"` (`:252-257`, `:266`);
* with a `class_path` it records that string and returns a **no-op** `lambda x: x` (`:259-264`);
* both write into `_ATTN_OVERRIDES` (or `_MAMBA_ATTN_OVERRIDES` when `is_mamba=True`).

How a backend is *selected*. `vllm__platforms__interface.py:259-267`:

```python
    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: "AttentionBackendEnum",
        attn_selector_config: "AttentionSelectorConfig",
        num_heads: int | None = None,
    ) -> str:
        """Get the attention backend class of a device."""
        return ""
```

The platform returns a **class path string**, not a class. So a third-party platform does not have
to mutate `_ATTN_OVERRIDES` at all; it can return
`"tileinfer.vllm.attn.backend.TileInferAttentionBackend"` directly. `[inferred]` That this string
is then resolved with `resolve_obj_by_qualname` is implied by `get_class()` (`registry.py:122-132`)
and by `register_backend` storing module paths, but the resolving call site
(`gpu_model_runner.py` / `vllm/v1/attention/selector.py`) is not in the fetched subset.

Consequence for TileInfer: `tileinfer/vllm/attn/backend.py` exposes
`register_with_vllm(enum=AttentionBackendEnum.CUSTOM) -> str` and does **not** register on import
(importing a module must not change vLLM's process-wide selection; the platform hook, W5, is the
intended wiring).

---

## 6. The `CustomOp` dispatch rule

`vllm__model_executor__custom_op.py`. The rule, in evaluation order:

1. **Construction may be replaced out-of-tree first.** `CustomOp.__new__` (`:109-128`) looks the
   class name up in `op_registry_oot` and instantiates that class instead if present (`:119-128`).
2. `__init__` (`:130-133`) stores `enforce_enable` / `compile_native` and computes
   `self._forward_method = self.dispatch_forward(compile_native=compile_native)`. All calls go
   through `forward` → `self._forward_method(...)` (`:135-136`).
3. `dispatch_forward` (`:174-207`):
   * `enabled = self._enforce_enable or self.enabled()` (`:185`), then
     `compilation_config.enabled_custom_ops.update([cls.name])` **or**
     `disabled_custom_ops.update(...)` (`:186-189`) — so the decision is recorded globally;
   * `if not enabled: return self.maybe_compile(self.forward_native, enable=compile_native)`
     (`:191-194`) — i.e. a disabled op runs the native implementation, optionally `torch.compile`d,
     never a device kernel;
   * otherwise a **fixed platform ladder**, first match wins: `is_rocm()` → `forward_hip`
     (`:196-197`), `is_cpu()` → `forward_cpu` (`:198-199`), `is_tpu()` → `forward_tpu`
     (`:200-201`), `is_xpu()` → `forward_xpu` (`:202-203`), `is_out_of_tree()` → `forward_oot`
     (`:204-205`), else `forward_cuda` (`:206-207`).
     "NOTE(woosuk): Here we assume that vLLM was built for only one specific backend. Currently, we
     do not support dynamic dispatching." (`:175-177`)
4. Default bodies: `forward_native` raises `NotImplementedError` (`:138-144`); `forward_hip` falls
   back to `forward_cuda` (`:149-151`); `forward_xpu`, `forward_cpu`, `forward_tpu`, `forward_oot`
   fall back to `forward_native` (`:153-172`).
5. `enabled()` (`:271-289`): `enabled = f"+{cls.name}" in custom_ops`, `disabled =
   f"-{cls.name}" in custom_ops`, `assert not (enabled and disabled)` (`:285-287`), then
   `(default_on() or enabled) and not disabled` (`:289`). `default_on()` is
   "on by default if `all`, off by default if `none`" (`:291-304`).
6. Registration: `@CustomOp.register(name, dynamic_arg_dims=None)` (`:306-320`) asserts a unique
   name, sets `op_cls.name` and `_dynamic_arg_dims`; `@CustomOp.register_oot(...)` (`:331-353`)
   fills `op_registry_oot`. Registries live at `:21-22`.

**Implication for TileInfer's NPU platform:** an out-of-tree platform reports
`is_out_of_tree() == True`, so the dispatch lands on `forward_oot` (`:204-205`) — an OOT op that
does not override `forward_oot` silently runs `forward_native` (`:169-172`). Every TileInfer
`CustomOp` must therefore implement `forward_oot` (and, for a non-OOT fallback, `forward_native`)
or it will be quietly replaced by the pure-torch path.

---

## 7. What TileInfer declares, and the line that justifies it

| decision | line evidence | why |
|---|---|---|
| `get_kv_cache_shape -> (2, num_blocks, block_size, num_kv_heads, head_size)` | `backend.py:87-96` (the contract), `:98-116` (block dim probe), `:118-147` (unimplemented stride order ⇒ physical == logical) | fused K/V + NHD is exactly what `kv_cache.write_kv_cache_fused` scatters and what `pool_sub_view` re-views zero-copy |
| `get_supported_kernel_block_sizes -> [MultipleOf(128)]` | `:68-70` (signature), `:174-191` (`MultipleOf` handled as `block_size % base`), `:48-52` (`MultipleOf`) | the kernels tile 128 tokens; 2048 (the mamba page) must stay legal |
| `get_supported_head_sizes -> [128, 256]` | `:153-160` | kernel widths; the target model is 16 × 256 |
| `supported_dtypes = [bf16, fp16]`, `supported_kv_cache_dtypes = ["auto","bfloat16","float16"]` | `:58`, `:59-63`, `:162-172` | unquantised only |
| exact-`False` overrides of `is_mla`, `is_sparse`, `supports_sink`, `supports_mm_prefix`, `supports_per_head_quant_scales`, `supports_non_causal`, `supports_batch_invariance`, `supports_kv_connector` | `:204-245` | the defaults are mostly already `False`; the explicit statements make the reject policy auditable, and `supports_kv_connector` **must** be overridden (default `True`, `:243-245`) |
| `forward_includes_kv_cache_update = True` | `:65-66` | the impl writes K/V in `forward` |
| `_cudagraph_support = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE` | `:524-526`, `:548-555`, `:517-518` | the paged kernel is capturable for pure single-token decode only; prefill/mixed goes to the reference path |
| `reorder_batch_threshold = None` | `:527-530` | we do not reorder |
| `supports_update_block_table = False` | `:532-533`, `:609-622` | call site not in the fetched subset; `build` is cheap, so take the verified path |
| `validate_configuration` rejects + calls `super()` | `:274-341` | inherited policy plus explicit statements |
| GQA `group < 2` rejected in `__init__` | `:780-794` (no head counts) vs `:275-291` | not expressible in `validate_configuration` |
| `if not enabled` / `use_cascade_attention() == False` ⇒ `common_prefix_len` must be 0 | `:658-670`, `:589-607` | cascade attention is unimplemented, and a non-zero prefix would silently change the result |

---

## 8. What could **not** be established (and is therefore explicit in the code)

1. **`query` / `key` / `value` / `output` layout.** The signature types them as bare
   `torch.Tensor` (`:800-803`, `:805`); nothing in the fetched subset says whether the engine
   passes `[tokens, heads*dim]` or `[tokens, heads, dim]`. `TileInferAttentionImpl` accepts both
   (and counts a reshape that had to copy).
2. **`slot_mapping` dtype.** Not stated in any fetched vLLM file. `write_kv_cache` requires
   `int64` (`tileinfer/vllm/attn/kv_cache.py:76-82`) and refuses to cast (a cast inside the
   captured step would allocate). `[inferred]` vLLM v1 uses `torch.int64` slot mappings.
3. **`block_table_tensor` dtype.** Not stated upstream. `refresh_step` requires `int32`
   (`tileinfer/serving/step.py:191-197`), again refusing an in-graph cast. `[inferred]` vLLM v1
   allocates the block table as `int32`.
4. **`AttentionSpec.block_size`** is assumed to exist on the spec handed to the builder. Only the
   attribute is read (`getattr`), and the builder degrades to "unknown" if it is missing.
5. **Where `validate_configuration` / the builder's `__init__` / `build_for_cudagraph_capture` are
   called.** None of those call sites are in the fetched files (`gpu_model_runner.py` is not
   fetched). The reject-at-startup claim therefore rests on the fact that vLLM's own
   `validate_configuration` exists and returns reasons; `[inferred]` that it is invoked during
   config resolution.
6. **Whether `AttentionMetadataBuilder.is_prefilling` fast paths matter for us.** No: the backend
   uses `max_query_len` only.
7. **`accept_output_buffer`** — absent (see §0). Kept only as a forward-compatible hook.

---

## 9. Test evidence

`tests/test_attn_contract.py` (57 CPU tests, no vLLM, no device) injects a documented fake
`vllm` package and asserts: the abstract-method surface used, `get_kv_cache_shape` NHD, the
`MultipleOf(128)` block-size policy, accept/reject for every configuration listed in §7, the
builder's three integers and bucket from a fake `CommonAttentionMetadata`, tensor **identity**
(no copies), decode-vs-prefill from `max_query_len` (with `is_prefilling` deliberately contradictory),
the declared `UNIFORM_SINGLE_TOKEN_DECODE` graph level, both transcribed signatures parameter by
parameter, the impl's start-up rejections, and that every `FALLBACK_REASONS` entry has a counter.

Two metadata invariants the backend additionally enforces at build time, both because the paged
kernel would otherwise produce plausible garbage rather than fail:

* a **decode** batch (`max_query_len <= 1`) must have exactly one scheduled query token per
  request (`num_actual_tokens == num_reqs`, or `0` for an empty step), because the staging buffer
  maps request `b` to query row `b`;
* `common_prefix_len` must be `0` (cascade attention is not implemented, `:658-670`).
