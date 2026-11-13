"""Contract tests for the vLLM attention backend, **without vLLM** (CPU only).

Why a fake `vllm` package
-------------------------
The three modules under test (``tileinfer.vllm.attn.backend`` / ``metadata`` / ``impl``) are the
only place in this repository allowed to speak vLLM's API, and they must speak *exactly* the
0.23 dialect transcribed in ``docs/notes/vllm-attention-contract.md``.  CI has no vLLM (and no
device), so the names they import are injected into ``sys.modules`` before the modules are
imported.  That fake lives in ``tests/fake_vllm.py`` and is **shared** with
``tests/test_worker_contract.py``: one fake, one shape, so neither the order these two files run
in nor the first one to import decides what ``vllm`` looks like.  (Two private fakes made this
file fail at *collection time* when the worker test ran first — its fake had no
``vllm.v1.attention``.)  The fake is deliberately minimal and **documented as a fake**: it provides
the *names* and the *two behaviours the tests depend on* (the builder's ``get_cudagraph_support``
ClassVar lookup, and the registry's ``register_backend(enum, path)`` recording).  It does **not**
reproduce the ABC's generic ``validate_configuration`` logic — that is why
``TileInferAttentionBackend.validate_configuration`` states its own rejects explicitly, and why
these tests are meaningful against a stub base.

What is asserted here is the part that can go wrong silently:

* the KV-cache shape is the block-first NHD fused one (``block_dim == 0``) whose declared stride
  order keeps the K/V halves contiguous - the two facts the write path, the sub-page view and
  vLLM's hybrid layout rewrite all depend on;
* the supported kernel block size is ``MultipleOf(128)``;
* every unsupported configuration is rejected by ``validate_configuration``;
* the builder computes ``num_reqs`` / ``num_actual_tokens`` / ``max_query_len`` from the *common*
  metadata and keeps the engine's tensors **by reference** (identity, not equality);
* decode vs prefill is decided by ``max_query_len``, not by ``is_prefilling``;
* the impl's ``__init__`` and ``forward`` parameter names are the transcribed ones (a renamed
  keyword is an immediate ``TypeError`` inside the engine, at the worst possible time);
* every fallback path the impl can take has a counter (no silent pass).

If a real vLLM is installed, the fake is skipped and the real modules are imported — the
assertions below are about TileInfer's contract, not about the stub.
"""

from __future__ import annotations

import dataclasses
import inspect
import sys

import pytest
import torch

import fake_vllm

# ---------------------------------------------------------------------------------------
# The transcribed contract (from reference/vllm-0.23/, line numbers in docs/notes/...)
# ---------------------------------------------------------------------------------------

#: ``AttentionImpl.__init__`` — `vllm/v1/attention/backend.py:780-794`.
TRANSCRIBED_IMPL_INIT = [
    "self",
    "num_heads",
    "head_size",
    "scale",
    "num_kv_heads",
    "alibi_slopes",
    "sliding_window",
    "kv_cache_dtype",
    "logits_soft_cap",
    "attn_type",
    "kv_sharing_target_layer_name",
]

#: ``AttentionImpl.forward`` — `vllm/v1/attention/backend.py:796-809`.
TRANSCRIBED_IMPL_FORWARD = [
    "self",
    "layer",
    "query",
    "key",
    "value",
    "kv_cache",
    "attn_metadata",
    "output",
    "output_scale",
    "output_block_scale",
]

#: ``AttentionBackend.validate_configuration`` — `vllm/v1/attention/backend.py:274-291`.
TRANSCRIBED_VALIDATE_CONFIGURATION = [
    "head_size",
    "dtype",
    "kv_cache_dtype",
    "block_size",
    "use_mla",
    "has_sink",
    "use_sparse",
    "use_mm_prefix",
    "use_per_head_quant_scales",
    "device_capability",
    "attn_type",
    "use_non_causal",
    "use_batch_invariant",
    "use_kv_connector",
]

#: ``CommonAttentionMetadata`` fields — `vllm/v1/attention/backend.py:368-422`.
TRANSCRIBED_COMMON_FIELDS = [
    "query_start_loc",
    "query_start_loc_cpu",
    "seq_lens",
    "num_reqs",
    "num_actual_tokens",
    "max_query_len",
    "max_seq_len",
    "block_table_tensor",
    "slot_mapping",
]


# ---------------------------------------------------------------------------------------
# The shared fake `vllm` (owned by tests/fake_vllm.py, shared with test_worker_contract.py)
# ---------------------------------------------------------------------------------------
#
# One fake, one shape: the attention API below and the worker API the sibling contract test needs
# live in `tests/fake_vllm.py`, which augments an existing fake instead of replacing it (a module
# object another already-imported TileInfer module points into must never be swapped out).  It is a
# helper module rather than a fixture because these names have to exist *now*, at import time of
# the `from tileinfer.vllm.attn...` block below.  Keeping a private copy here is what made the
# suite order-dependent: `vllm` reached by the worker test first lacked `vllm.v1.attention`, and
# the import below then failed during *collection* (interrupting the whole session, not one test).
def _install_fake_vllm() -> str:
    """Install the shared fake (``tests/fake_vllm.py``); return ``fake``/``augmented``/``real``."""
    return fake_vllm.install()


VLLM_MODE = _install_fake_vllm()

# Imported *after* the fake is in sys.modules; module-level imports inside these modules are
# guarded and would otherwise raise the "not importable without vLLM" error by design.
from tileinfer.vllm.attn.backend import (  # noqa: E402
    SUPPORTED_HEAD_SIZES,
    TileInferAttentionBackend,
    register_with_vllm,
)
from tileinfer.vllm.attn.impl import (  # noqa: E402
    FALLBACK_REASONS,
    IMPL_COUNTER_INITIAL,
    IMPL_COUNTERS,
    TileInferAttentionImpl,
    reset_counters,
)
from tileinfer.vllm.attn.metadata import (  # noqa: E402
    METADATA_COUNTERS,
    TileInferMetadata,
    TileInferMetadataBuilder,
)
from tileinfer.serving.buckets import (  # noqa: E402
    BATCH_BUCKETS,
    KERNEL_PAGE_SIZE,
    bucket_for,
    fused_halves,
    pool_sub_view,
)

vllm_backend = sys.modules["vllm.v1.attention.backend"]
vllm_registry = sys.modules["vllm.v1.attention.backends.registry"]
AttentionType = vllm_backend.AttentionType
AttentionCGSupport = vllm_backend.AttentionCGSupport
MultipleOf = vllm_backend.MultipleOf
CommonAttentionMetadata = vllm_backend.CommonAttentionMetadata

#: How many registry mutations happened while the three modules were being imported.  Must be 0:
#: importing a backend module must never change vLLM's process-wide selection.
_REGISTRY_CALLS_AT_IMPORT = (
    len(vllm_registry.registry_calls) if hasattr(vllm_registry, "registry_calls") else None
)


# ---------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------


class _Spec:
    """Stand-in for ``AttentionSpec`` (only ``block_size`` is read)."""

    def __init__(self, block_size: int):
        self.block_size = block_size


class _SchedulerConfig:
    def __init__(self, max_num_seqs: int):
        self.max_num_seqs = max_num_seqs


class _VllmConfig:
    def __init__(self, max_num_seqs: int):
        self.scheduler_config = _SchedulerConfig(max_num_seqs)


def _common(
    query_lens,
    seq_lens,
    *,
    slot_tokens: int | None = None,
    max_query_len: int | None = None,
    table_width: int = 4,
    is_prefilling=None,
):
    """A ``CommonAttentionMetadata`` with consistent start locations.

    Only the fields the backend reads are populated; the fake dataclass defaults the rest
    (``causal``, ``is_prefilling``, the CPU upper bound, ``positions``).
    """
    total = sum(query_lens)
    qsl = [0]
    for length in query_lens:
        qsl.append(qsl[-1] + length)
    qsl_t = torch.tensor(qsl, dtype=torch.int32)
    if max_query_len is None:
        max_query_len = max(query_lens) if query_lens else 0
    n_tokens = total if slot_tokens is None else slot_tokens
    return CommonAttentionMetadata(
        query_start_loc=qsl_t,
        query_start_loc_cpu=qsl_t,
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        num_reqs=len(query_lens),
        num_actual_tokens=n_tokens,
        max_query_len=max_query_len,
        max_seq_len=max(seq_lens) if seq_lens else 0,
        block_table_tensor=torch.zeros((len(query_lens), table_width), dtype=torch.int32),
        slot_mapping=torch.arange(n_tokens, dtype=torch.int64),
        is_prefilling=is_prefilling,
    )


def _builder(block_size: int = 2048, max_num_seqs: int = 8) -> TileInferMetadataBuilder:
    return TileInferMetadataBuilder(
        kv_cache_spec=_Spec(block_size),
        layer_names=["model.layers.3.self_attn"],
        vllm_config=_VllmConfig(max_num_seqs),
        device=torch.device("cpu"),
    )


def _impl(**overrides) -> TileInferAttentionImpl:
    kwargs = dict(
        num_heads=16,
        head_size=256,
        scale=256**-0.5,
        num_kv_heads=2,
        kv_cache_dtype="auto",
    )
    kwargs.update(overrides)
    return TileInferAttentionImpl(**kwargs)


@pytest.fixture(autouse=True)
def _reset_counters():
    reset_counters()
    yield
    reset_counters()


# ---------------------------------------------------------------------------------------
# 1. backend identity, KV shape, block sizes
# ---------------------------------------------------------------------------------------


def test_mode_is_documented():
    # "fake": this file created the fake; "augmented": a fake was already in sys.modules (the
    # sibling contract test ran first) and was completed in place; "real": a real vLLM won.
    # The old private fake returned a fourth mode, "already-imported", and installed *nothing* in
    # that case - which is precisely the order-dependent failure tests/fake_vllm.py removes.
    assert set(fake_vllm.MODES) == {"fake", "augmented", "real"}
    assert VLLM_MODE in fake_vllm.MODES


def test_backend_identity_points_at_our_impl_and_builder():
    from tileinfer.vllm.attn.metadata import TileInferMetadataBuilder as builder_cls

    assert TileInferAttentionBackend.get_name() == "TILEINFER"
    assert TileInferAttentionBackend.get_impl_cls() is TileInferAttentionImpl
    assert TileInferAttentionBackend.get_builder_cls() is builder_cls
    # The impl writes the KV cache inside forward (v1/attention/backend.py:65-66).
    assert TileInferAttentionBackend.forward_includes_kv_cache_update is True


def test_get_kv_cache_shape_is_block_first_so_vllm_skips_the_hybrid_rewrite():
    """Block-first, not the vLLM-native ``(2, num_blocks, ...)``: the layout is load-bearing.

    ``_update_hybrid_attention_mamba_layout`` (``gpu_model_runner.py:7167-7199``) rewrites a
    ``block_dim == 1`` pool into a physically ``[blocks][2][page][heads][dim]`` layout by keeping
    the *shape* and replacing dims 0/1's strides (``:7196-7199``).  That interleaves K and V per
    block, which no contiguous ``[pages, 128, heads, dim]`` view can express.  ``block_dim == 0``
    makes the rewrite ``continue`` (``:7190-7191``), so the block must be dim 0.
    """
    shape = TileInferAttentionBackend.get_kv_cache_shape(
        num_blocks=7, block_size=2048, num_kv_heads=2, head_size=256
    )
    # The page count is NOT rounded: the runner sizes the raw buffer from the scheduler's
    # num_blocks and applies this shape as a view, so a rounded shape would not fit.  The
    # declared-capacity gap is closed in capacity_view instead (no storage extension).
    assert shape == (7, 2, 2048, 2, 256)
    # The block dimension must be discoverable by the ABC's own probe (shape.index(1234567)) - and
    # must be dim 0.
    assert TileInferAttentionBackend.get_kv_cache_shape(
        num_blocks=1234567, block_size=128, num_kv_heads=2, head_size=128
    ).index(1234567) == 0
    assert (
        TileInferAttentionBackend.get_kv_cache_block_dim(2048, 2, 256, cache_dtype_str="auto")
        == 0
    )


def test_the_declared_stride_order_keeps_the_k_and_v_halves_contiguous():
    """The physical order is ``[2][blocks][page][heads][dim]``, and that is what makes the halves flat.

    This replays ``_reshape_kv_cache_tensors`` (``gpu_model_runner.py:7093-7133``) on the backend's
    own declarations and then checks the two properties the write/kernel path depends on.  The
    *default* physical order is checked too: it is the layout that cannot be viewed, which is why
    the override exists.
    """
    blocks, block_size, heads, dim, sub = 8, 512, 2, 8, 4  # 8: already on the capacity grid
    shape = TileInferAttentionBackend.get_kv_cache_shape(blocks, block_size, heads, dim)
    order = TileInferAttentionBackend.get_kv_cache_stride_order()
    assert order == (1, 0, 2, 3, 4)
    assert len(order) == len(shape)
    assert shape[0] == blocks  # a grid-aligned count is handed back unchanged

    # vLLM: allocate the physical order, then permute the view back to the logical shape.
    physical = tuple(shape[i] for i in order)
    inv_order = [order.index(i) for i in range(len(order))]
    raw = torch.arange(
        shape[0] * 2 * block_size * heads * dim, dtype=torch.float32
    ).reshape(physical)
    pool = raw.permute(*inv_order)
    assert tuple(pool.shape) == shape

    k_half, v_half = pool[:, 0], pool[:, 1]
    assert k_half.is_contiguous() and v_half.is_contiguous()
    # The halves are the same allocation, one K (or V) pool apart.
    assert v_half.storage_offset() - k_half.storage_offset() == shape[0] * block_size * heads * dim

    page = pool_sub_view(k_half, sub)
    assert page.data_ptr() == k_half.data_ptr(), "the sub-page view must stay zero-copy"
    assert tuple(page.shape) == (shape[0] * sub, 128, heads, dim)
    for p in (0, 1, sub, shape[0] * sub - 1):
        engine_page, offset = divmod(p, sub)
        assert torch.equal(page[p, 0], k_half[engine_page, offset * 128])

    # Without the override the physical order is the logical one, i.e. [blocks][2][...]: the halves
    # are then strided and the zero-copy sub-page view is impossible.  This is the layout the
    # hybrid rewrite produces for a block_dim==1 backend, and it is *not* faked.
    default = torch.arange(
        shape[0] * 2 * block_size * heads * dim, dtype=torch.float32
    ).reshape(shape)
    assert not default[:, 0].is_contiguous()
    with pytest.raises(ValueError, match="contiguous"):
        pool_sub_view(default[:, 0], sub)



def test_the_hybrid_layout_rewrite_is_a_no_op_for_this_shape():
    """Replay ``_update_hybrid_attention_mamba_layout`` (``gpu_model_runner.py:7189-7199``).

    The helper does **not** reorder the buffer: it keeps ``size=kv_cache.shape`` and replaces dims
    0/1's strides with ``(hidden, 2*hidden)``, i.e. it reinterprets a ``block_dim == 1`` pool as
    physically ``[blocks][2][page][heads][dim]``.  For the shape declared here the probe returns 0
    and the loop ``continue``s, so the tensor is untouched - which is what keeps the K/V halves
    contiguous and the page ids logical.
    """
    blocks, block_size, heads, dim = 5, 512, 2, 8
    shape = TileInferAttentionBackend.get_kv_cache_shape(blocks, block_size, heads, dim)
    block_dim = TileInferAttentionBackend.get_kv_cache_block_dim(block_size, heads, dim)
    assert block_dim == 0

    order = TileInferAttentionBackend.get_kv_cache_stride_order()
    physical = tuple(shape[i] for i in order)
    raw = torch.zeros(physical)
    pool = raw.permute(*[order.index(i) for i in range(len(order))])
    strides_before = pool.stride()

    # The runner's loop body, verbatim in effect:
    for _group in (None,):  # one attention group
        if block_dim == 0:
            continue
        hidden = pool.shape[2:].numel()
        pool.as_strided_(
            size=pool.shape, stride=(hidden, 2 * hidden, *pool.stride()[2:])
        )
    assert pool.stride() == strides_before, "the hybrid rewrite must not touch a block_dim==0 pool"
    assert pool[:, 0].is_contiguous() and pool[:, 1].is_contiguous()

    # And what it *would* have done to the legacy shape: same size, physically interleaved.
    legacy = torch.zeros(2, blocks, block_size, heads, dim)
    hidden = legacy.shape[2:].numel()
    legacy.as_strided_(size=legacy.shape, stride=(hidden, 2 * hidden, *legacy.stride()[2:]))
    assert legacy.stride() == (hidden, 2 * hidden, heads * dim, dim, 1)
    assert not legacy[0].is_contiguous()


def test_supported_kernel_block_sizes_are_multiples_of_128():
    sizes = TileInferAttentionBackend.get_supported_kernel_block_sizes()
    assert len(sizes) == 1
    assert isinstance(sizes[0], MultipleOf)
    assert sizes[0].base == KERNEL_PAGE_SIZE == 128
    assert 2048 % sizes[0].base == 0  # the hybrid mamba page is accepted
    assert 192 % sizes[0].base != 0  # a 1.5-page layout is not


def test_supported_head_sizes_are_the_kernel_widths():
    assert TileInferAttentionBackend.get_supported_head_sizes() == list(SUPPORTED_HEAD_SIZES)
    assert set(SUPPORTED_HEAD_SIZES) == {128, 256}


# ---------------------------------------------------------------------------------------
# 2. validate_configuration
# ---------------------------------------------------------------------------------------


def _validate(**overrides) -> list[str]:
    kwargs = dict(
        head_size=256,
        dtype=torch.bfloat16,
        kv_cache_dtype="auto",
        block_size=2048,
        use_mla=False,
        has_sink=False,
        use_sparse=False,
        use_mm_prefix=False,
        use_per_head_quant_scales=False,
        device_capability=None,
        attn_type=AttentionType.DECODER,
    )
    kwargs.update(overrides)
    return TileInferAttentionBackend.validate_configuration(**kwargs)


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"head_size": 128, "block_size": 128},
        {"block_size": 4096, "dtype": torch.float16, "kv_cache_dtype": "float16"},
        {"block_size": None},  # "let the platform decide" is allowed
        {"kv_cache_dtype": "bfloat16"},
    ],
)
def test_validate_configuration_accepts_the_target_model(overrides):
    assert _validate(**overrides) == []


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"head_size": 64}, "head_size"),
        ({"head_size": 192}, "head_size"),
        ({"dtype": torch.float32}, "dtype"),
        ({"kv_cache_dtype": "fp8"}, "kv_cache_dtype"),
        ({"block_size": 192}, "block_size"),
        ({"block_size": 100}, "block_size"),
        ({"use_mla": True}, "MLA"),
        ({"use_sparse": True}, "sparse"),
        ({"has_sink": True}, "sink"),
        ({"use_mm_prefix": True}, "multimodal"),
        ({"use_per_head_quant_scales": True}, "per-head"),
        ({"use_non_causal": True}, "non-causal"),
        ({"use_batch_invariant": True}, "batch-invariant"),
        ({"use_kv_connector": True}, "connector"),
        ({"attn_type": AttentionType.ENCODER}, "attention type"),
    ],
)
def test_validate_configuration_rejects_unsupported_shapes(overrides, match):
    reasons = _validate(**overrides)
    assert reasons, f"{overrides} was accepted but must be rejected"
    assert any(match.lower() in reason.lower() for reason in reasons), reasons


def test_validate_configuration_signature_matches_the_transcribed_one():
    names = list(inspect.signature(TileInferAttentionBackend.validate_configuration).parameters)
    assert names == TRANSCRIBED_VALIDATE_CONFIGURATION


# ---------------------------------------------------------------------------------------
# 3. metadata builder
# ---------------------------------------------------------------------------------------


def test_builder_keeps_common_tensors_by_reference():
    common = _common(query_lens=[1, 1, 1], seq_lens=[10, 20, 30])
    meta = _builder().build(0, common)

    # Identity, not equality: the engine rewrites block_table_tensor / slot_mapping in place
    # every step, so a copy would be a stale page table (silently wrong output).
    assert meta.query_start_loc is common.query_start_loc
    assert meta.query_start_loc_cpu is common.query_start_loc_cpu
    assert meta.seq_lens is common.seq_lens
    assert meta.block_table_tensor is common.block_table_tensor
    assert meta.slot_mapping is common.slot_mapping


def test_builder_computes_the_three_integers_and_the_bucket():
    common = _common(query_lens=[1, 1, 3], seq_lens=[10, 20, 30])
    meta = _builder(max_num_seqs=8).build(0, common)

    assert isinstance(meta, TileInferMetadata)
    assert meta.num_reqs == 3
    assert meta.num_actual_tokens == 5
    assert meta.max_query_len == 3
    assert meta.bucket == bucket_for(3, BATCH_BUCKETS) == 4
    assert meta.padding_rows == 1
    assert meta.block_size == 2048


def test_decode_vs_prefill_is_decided_by_max_query_len_only():
    decode = _builder().build(0, _common([1, 1, 1], [5, 6, 7]))
    assert decode.max_query_len == 1
    assert decode.is_decode is True and decode.is_prefill is False

    # `is_prefilling` says "no request is still prefilling", yet a 3-token query makes this a
    # prefill batch: per-request flags cannot describe a batch whose query lengths disagree.
    flags = torch.zeros(3, dtype=torch.bool)
    prefill = _builder().build(
        0, _common([3, 1, 1], [5, 6, 7], is_prefilling=flags)
    )
    assert prefill.max_query_len == 3
    assert prefill.is_prefill is True and prefill.is_decode is False


def test_builder_repairs_a_too_small_declared_max_query_len():
    common = _common([4, 1], [9, 4], max_query_len=1)  # a caller bug: too small
    before = METADATA_COUNTERS.get("max_query_len_repaired")
    meta = _builder().build(0, common)
    assert meta.max_query_len == 4  # derived from query_start_loc_cpu, no device sync
    assert METADATA_COUNTERS.get("max_query_len_repaired") == before + 1


def test_builder_rejects_a_decode_batch_with_a_zero_token_request():
    # One row per request only holds when every decode request schedules exactly one token; a
    # zero-token request would shift every later row (plausible garbage, not a crash).
    with pytest.raises(ValueError, match="one query token per request"):
        _builder().build(0, _common([0, 1], [4, 5]))


def test_builder_rejects_cascade_prefix_and_oversized_batches():
    builder = _builder(max_num_seqs=8)
    with pytest.raises(ValueError, match="common_prefix_len"):
        builder.build(64, _common([1], [4]))
    with pytest.raises(ValueError, match="max_num_seqs"):
        builder.build(0, _common([1] * 9, [4] * 9))


def test_builder_rejects_a_block_size_that_is_not_a_whole_number_of_kernel_pages():
    with pytest.raises(ValueError, match="multiple of 128"):
        _builder(block_size=192)
    with pytest.raises(ValueError, match="multiple of"):
        _builder(block_size=100)


def test_builder_declares_single_token_decode_graph_support():
    builder = _builder()
    assert (
        TileInferMetadataBuilder._cudagraph_support
        is AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    )
    # The ABC reads it through this classmethod (v1/attention/backend.py:548-555).
    assert (
        TileInferMetadataBuilder.get_cudagraph_support(None, None)
        is AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    )
    # We never reorder the batch: None is the documented spelling (v1/attention/backend.py:527-530).
    assert TileInferMetadataBuilder.reorder_batch_threshold is None


def test_metadata_dataclass_covers_the_common_fields_it_wraps():
    fields = {f.name for f in dataclasses.fields(TileInferMetadata)}
    # Every common field the backend consumes is present on our metadata (the rest are computed).
    assert set(TRANSCRIBED_COMMON_FIELDS).issubset(fields)


# ---------------------------------------------------------------------------------------
# 4. impl contract
# ---------------------------------------------------------------------------------------


def test_impl_init_signature_matches_the_transcribed_one():
    names = list(inspect.signature(TileInferAttentionImpl.__init__).parameters)
    assert names == TRANSCRIBED_IMPL_INIT


def test_impl_forward_signature_matches_the_transcribed_one():
    signature = inspect.signature(TileInferAttentionImpl.forward)
    assert list(signature.parameters) == TRANSCRIBED_IMPL_FORWARD
    assert signature.parameters["output_scale"].default is None
    assert signature.parameters["output_block_scale"].default is None
    # `output` is required by the ABC (a buffer, not an optional): keep it required.
    assert signature.parameters["output"].default is inspect.Parameter.empty


def test_impl_accepts_the_target_model_shape():
    impl = _impl()
    assert (impl.num_heads, impl.num_kv_heads, impl.head_size) == (16, 2, 256)
    assert impl.gqa_group == 8
    assert impl.sliding_window is None  # get_per_layer_parameters indexes it as a tuple
    assert impl.logits_soft_cap is None
    assert impl.kv_cache_dtype == "auto"


@pytest.mark.parametrize(
    "overrides,error,match",
    [
        ({"num_kv_heads": 16}, ValueError, "group"),
        ({"num_heads": 4, "num_kv_heads": 8}, ValueError, "num_kv_heads"),
        ({"num_heads": 16, "num_kv_heads": 6}, ValueError, "multiple of num_kv_heads"),
        ({"head_size": 64}, ValueError, "head_size"),
        ({"head_size": 192}, ValueError, "head_size"),
        # sliding_window is *accepted* now (windowed layers compute on the reference path —
        # tests/test_windowed_attention.py owns that behaviour); a non-positive window is the
        # remaining configuration error.
        ({"sliding_window": 0}, ValueError, "positive"),
        ({"alibi_slopes": [0.1] * 16}, NotImplementedError, "alibi"),
        ({"logits_soft_cap": 30.0}, NotImplementedError, "soft capping"),
        ({"kv_sharing_target_layer_name": "model.layers.0.self_attn"}, NotImplementedError, "KV-sharing"),
        ({"kv_cache_dtype": "fp8"}, NotImplementedError, "kv_cache_dtype"),
        ({"attn_type": AttentionType.ENCODER}, NotImplementedError, "attention type"),
    ],
)
def test_impl_rejects_unsupported_layer_configurations(overrides, error, match):
    with pytest.raises(error, match=match):
        _impl(**overrides)


# ---------------------------------------------------------------------------------------
# 5. forward: the guards that need no device
# ---------------------------------------------------------------------------------------


def _meta(query_lens=(1, 1), seq_lens=(4, 7), slot_tokens=None) -> TileInferMetadata:
    common = _common(list(query_lens), list(seq_lens), slot_tokens=slot_tokens)
    return _builder().build(0, common)


def _pool(blocks=3, block_size=2048, kv_heads=2, head_size=256, dtype=torch.bfloat16):
    # 3 blocks, not 2: with exactly two blocks the block-first and the legacy K/V-first shapes are
    # indistinguishable from metadata, and `fused_halves` refuses that ambiguity on purpose.
    return torch.zeros((blocks, 2, block_size, kv_heads, head_size), dtype=dtype)


def test_forward_refuses_foreign_metadata_before_touching_the_device():
    impl = _impl()
    with pytest.raises(TypeError, match="TileInferMetadata"):
        impl.forward(
            None,
            torch.zeros(2, 16, 256),
            torch.zeros(2, 2, 256),
            torch.zeros(2, 2, 256),
            _pool(),
            object(),
            torch.zeros(2, 16 * 256),
        )
    assert IMPL_COUNTERS.get("metadata_mismatch") == 1


def test_forward_refuses_fused_output_quantisation():
    impl = _impl()
    with pytest.raises(NotImplementedError, match="output quantisation"):
        impl.forward(
            None,
            torch.zeros(2, 16, 256),
            torch.zeros(2, 2, 256),
            torch.zeros(2, 2, 256),
            _pool(),
            _meta(),
            torch.zeros(2, 16 * 256),
            output_scale=torch.ones(1),
        )
    assert IMPL_COUNTERS.get("output_quant_unsupported") == 1


def test_forward_refuses_an_empty_kv_pool():
    impl = _impl()
    with pytest.raises(RuntimeError, match="empty kv_cache"):
        impl.forward(
            None,
            torch.zeros(2, 16, 256),
            torch.zeros(2, 2, 256),
            torch.zeros(2, 2, 256),
            torch.zeros(0),
            _meta(),
            torch.zeros(2, 16 * 256),
        )
    assert IMPL_COUNTERS.get("empty_kv_cache") == 1


def test_forward_refuses_missing_kv():
    impl = _impl()
    with pytest.raises(RuntimeError, match="key/value = None"):
        impl.forward(
            None,
            torch.zeros(2, 16, 256),
            None,
            None,
            _pool(),
            _meta(),
            torch.zeros(2, 16 * 256),
        )
    assert IMPL_COUNTERS.get("missing_kv") == 1


def test_forward_returns_the_output_unchanged_for_an_empty_step():
    impl = _impl()
    output = torch.zeros(0, 16 * 256, dtype=torch.bfloat16)
    result = impl.forward(
        None,
        torch.zeros(0, 16, 256, dtype=torch.bfloat16),
        torch.zeros(0, 2, 256, dtype=torch.bfloat16),
        torch.zeros(0, 2, 256, dtype=torch.bfloat16),
        _pool(dtype=torch.bfloat16),
        _meta(query_lens=(1, 1), slot_tokens=0),
        output,
    )
    assert result is output
    assert IMPL_COUNTERS.get("empty_step") == 1
    assert IMPL_COUNTERS.get("kv_writes") == 0  # nothing was written for zero tokens


# ---------------------------------------------------------------------------------------
# 6. fallbacks are declared *and* counted
# ---------------------------------------------------------------------------------------


def test_every_fallback_reason_has_a_counter():
    assert FALLBACK_REASONS  # not empty
    for reason in FALLBACK_REASONS:
        assert reason in IMPL_COUNTERS, f"fallback {reason!r} has no counter"
        assert reason in IMPL_COUNTER_INITIAL
    # The one-time discovery counter for a missing kernel backend is not per-step.
    assert "kernel_backend_unavailable" in IMPL_COUNTERS


def test_registry_helper_registers_the_class_explicitly():
    calls = getattr(vllm_registry, "registry_calls", None)
    if calls is None:  # a real vLLM: only check that the call is accepted and returns the path
        path = register_with_vllm()
        assert path.endswith(".TileInferAttentionBackend")
        return
    before = len(calls)
    path = register_with_vllm()
    assert path.endswith(".TileInferAttentionBackend")
    assert len(calls) == before + 1
    enum, recorded = calls[-1]
    assert recorded == path
    assert enum is vllm_registry.AttentionBackendEnum.CUSTOM


def test_importing_the_backend_does_not_mutate_vllms_registry():
    """Importing the module must not register anything: only ``register_with_vllm()`` does."""
    if _REGISTRY_CALLS_AT_IMPORT is None:
        pytest.skip("real vLLM: the fake registry recorder is not available")
    assert _REGISTRY_CALLS_AT_IMPORT == 0
