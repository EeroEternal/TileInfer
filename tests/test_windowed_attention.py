"""Sliding-window (iRoPE) attention: the window semantics and the impl's dispatch, CPU only.

Two layers of the feature are pinned here, because both can go wrong silently:

* **the window arithmetic** — the reference backend masks ``k_pos < q_pos - window_left`` over
  *absolute* positions, so a window of ``w`` tokens (counting the query itself) is
  ``window_left = w - 1`` and a chunked-prefill extend (``kv_len > qo_len``) must offset
  ``q_pos`` by ``kv_len - qo_len``.  The oracle below recomputes the expected output by
  *slicing* the visible range — no masking code at all — so a shared mask bug cannot pass.
* **the dispatch** — a layer carrying ``sliding_window`` must never reach the paged kernels
  (they are full-causal only): decode bumps ``decode_window_reference``, prefill bumps
  ``prefill_window_reference``, and the builder downgrades graph capture for a windowed spec
  (the reference path walks the page table on the host).  A full-attention layer is checked on
  the same shapes to prove the window is what changed the route.

The pool plumbing (write → gather) is exercised with real tensors; the windowed layers'
numeric output is not asserted against the kernels anywhere — the kernels cannot express the
window, which is the whole point of the dispatch.
"""

from __future__ import annotations

import pytest
import torch

import fake_vllm

VLLM_MODE = fake_vllm.install()

from tileinfer.attention.api import BatchAttention  # noqa: E402
from tileinfer.testing.cache import build_paged_cache, random_q  # noqa: E402
from tileinfer.testing.reference import expand_kv_heads, gather_kv  # noqa: E402

# Imported *after* the fake is in sys.modules (the fake augments an existing fake instead of
# replacing it — see tests/fake_vllm.py); this is the same import shape as test_attn_contract.
from vllm.v1.attention.backend import (  # noqa: E402
    AttentionCGSupport,
    CommonAttentionMetadata,
)

from tileinfer.vllm.attn.impl import (  # noqa: E402
    IMPL_COUNTERS,
    TileInferAttentionImpl,
    reset_counters,
)
from tileinfer.vllm.attn.metadata import (  # noqa: E402
    TileInferMetadata,
    TileInferMetadataBuilder,
)

RTOL, ATOL = 1e-2, 2e-3


def assert_close(a, b, msg=""):
    a = a.float()
    b = b.float()
    diff = (a - b).abs().max().item()
    assert torch.allclose(a, b, rtol=RTOL, atol=ATOL), f"{msg} max abs diff {diff:.3e}"


# ---------------------------------------------------------------------------------------
# the independent oracle: slice the visible window, softmax, no masking code
# ---------------------------------------------------------------------------------------


def _windowed_oracle(
    q: torch.Tensor,  # [batch, h_q, qo_len, d]
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    meta,
    *,
    window: int,
    sm_scale: float,
) -> torch.Tensor:
    """Softmax over exactly the ``window`` tokens ending at each query's absolute position.

    ``q_pos = kv_len_b - qo_len_b + i`` (the causal offset a chunked-prefill extend produces);
    the visible keys are ``kv_pos in [q_pos - window + 1, q_pos]``.
    """
    batch, h_q, qo_len, d = q.shape
    h_kv = k_cache.shape[2]
    group = h_q // h_kv
    out = torch.zeros((batch, h_q, qo_len, d), dtype=torch.float32)
    for b in range(batch):
        kb, vb = gather_kv(k_cache, v_cache, meta, b, "NHD")
        kb = expand_kv_heads(kb.float(), group)
        vb = expand_kv_heads(vb.float(), group)
        kv_len = kb.shape[0]
        for i in range(qo_len):
            q_pos = kv_len - qo_len + i
            lo = max(0, q_pos - window + 1)
            ks = kb[lo : q_pos + 1]
            vs = vb[lo : q_pos + 1]
            qb = q[b, :, i, :].float()  # [h_q, d]
            scores = torch.einsum("hd,khd->hk", qb, ks) * sm_scale
            probs = torch.softmax(scores, dim=-1)
            out[b, :, i, :] = torch.einsum("hk,khd->hd", probs, vs)
    return out


def _paged_case(batch, h_q, h_kv, qo_len, kv_len, page_size):
    k = torch.randn(batch, h_kv, kv_len, 64, dtype=torch.float16)
    v = torch.randn(batch, h_kv, kv_len, 64, dtype=torch.float16)
    k_cache, v_cache, meta = build_paged_cache(
        k, v, page_size=page_size, shuffle_pages=True, seed=11
    )
    q = random_q(batch, h_q, qo_len, 64, dtype=torch.float16)
    return q, k_cache, v_cache, meta


def _plan_windowed(q, k_cache, v_cache, meta, qo_lens, window, *, page_size, h_q, h_kv):
    attn = BatchAttention(backend="reference", device="cpu")
    scale = 64**-0.5
    qo_indptr = torch.zeros(len(qo_lens) + 1, dtype=torch.int32)
    qo_indptr[1:] = torch.cumsum(torch.tensor(qo_lens, dtype=torch.int32), dim=0).to(torch.int32)
    plan = attn.plan(
        kv_indptr=meta.kv_indptr,
        kv_indices=meta.kv_indices,
        kv_last_page_len=meta.kv_last_page_len,
        qo_indptr=qo_indptr,
        page_size=page_size,
        num_qo_heads=h_q,
        num_kv_heads=h_kv,
        head_dim=64,
        sm_scale=scale,
        causal=True,
        window_left=window - 1,
        finalize=True,
        k_cache=k_cache,
        v_cache=v_cache,
    )
    return attn.run(q, k_cache, v_cache, plan=plan)


# ---------------------------------------------------------------------------------------
# 1. window arithmetic (decode / extend / degenerate windows)
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("window", [1, 7, 100, 4096])
def test_windowed_decode_matches_the_slice_oracle(window):
    q, k_cache, v_cache, meta = _paged_case(2, 4, 2, 1, 300, 64)
    out = _plan_windowed(q, k_cache, v_cache, meta, [1, 1], window, page_size=64, h_q=4, h_kv=2)
    expected = _windowed_oracle(
        q, k_cache, v_cache, meta, window=window, sm_scale=64**-0.5
    )
    assert out.shape == q.shape
    assert_close(out, expected, msg=f"decode window={window}")


@pytest.mark.parametrize("window", [1, 3, 50])
def test_windowed_extend_matches_the_slice_oracle(window):
    """Chunked prefill: 5 query tokens appended onto a 200-token prefix (kv_len=205).

    ``q_pos`` starts at ``kv_len - qo_len`` — the offset that made chunked prefill the classic
    bug source — and the window rides on the absolute positions.
    """
    q, k_cache, v_cache, meta = _paged_case(1, 4, 2, 5, 205, 64)
    meta.qo_indptr = torch.tensor([0, 5], dtype=torch.int32)
    out = _plan_windowed(q, k_cache, v_cache, meta, [5], window, page_size=64, h_q=4, h_kv=2)
    expected = _windowed_oracle(
        q, k_cache, v_cache, meta, window=window, sm_scale=64**-0.5
    )
    assert_close(out, expected, msg=f"extend window={window}")


def test_window_wider_than_the_context_is_plain_causal():
    """``window_left >= kv_len - 1`` masks nothing; the output must equal causal-full attention."""
    q, k_cache, v_cache, meta = _paged_case(2, 4, 2, 1, 300, 64)
    windowed = _plan_windowed(
        q, k_cache, v_cache, meta, [1, 1], 4096, page_size=64, h_q=4, h_kv=2
    )
    causal = _plan_windowed(
        q, k_cache, v_cache, meta, [1, 1], 10**9, page_size=64, h_q=4, h_kv=2
    )
    assert_close(windowed, causal, msg="wide window vs no window")


def test_different_windows_produce_different_outputs_when_the_window_bites():
    """A 1-token window (only self) cannot equal a 40-token window on random inputs."""
    q, k_cache, v_cache, meta = _paged_case(1, 4, 2, 1, 300, 64)
    one = _plan_windowed(q, k_cache, v_cache, meta, [1], 1, page_size=64, h_q=4, h_kv=2)
    forty = _plan_windowed(q, k_cache, v_cache, meta, [1], 40, page_size=64, h_q=4, h_kv=2)
    assert (one.float() - forty.float()).abs().max().item() > 1e-3


# ---------------------------------------------------------------------------------------
# 2. impl dispatch: a windowed layer never reaches the paged kernels
# ---------------------------------------------------------------------------------------


class _Spec:
    """Stand-in for ``AttentionSpec`` (``block_size`` plus the optional window)."""

    def __init__(self, block_size: int, sliding_window: int | None = None):
        self.block_size = block_size
        self.sliding_window = sliding_window


class _SchedulerConfig:
    def __init__(self, max_num_seqs: int):
        self.max_num_seqs = max_num_seqs


class _VllmConfig:
    def __init__(self, max_num_seqs: int):
        self.scheduler_config = _SchedulerConfig(max_num_seqs)


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


def _builder(sliding_window: int | None = None) -> TileInferMetadataBuilder:
    return TileInferMetadataBuilder(
        kv_cache_spec=_Spec(2048, sliding_window),
        layer_names=["model.layers.3.self_attn"],
        vllm_config=_VllmConfig(8),
        device=torch.device("cpu"),
    )


def _meta(query_lens=(1, 1), seq_lens=(4, 7), sliding_window=None) -> TileInferMetadata:
    query_start = [0]
    for length in query_lens:
        query_start.append(query_start[-1] + length)
    qsl = torch.tensor(query_start, dtype=torch.int32)
    builder = _builder(sliding_window)
    common = CommonAttentionMetadata(
        query_start_loc=qsl,
        query_start_loc_cpu=qsl,
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        num_reqs=len(query_lens),
        num_actual_tokens=sum(query_lens),
        max_query_len=max(query_lens),
        max_seq_len=max(seq_lens),
        block_table_tensor=torch.zeros((len(query_lens), 4), dtype=torch.int32),
        slot_mapping=torch.arange(sum(query_lens), dtype=torch.int64),
    )
    return builder.build(0, common)


def _pool(blocks=3, block_size=2048, dtype=torch.bfloat16):
    # The *physical* stride order the engine allocates: [2][blocks][page][heads][dim] (the
    # backend's declared stride order), permuted into the logical block-first shape.  The K/V
    # halves of this pool are contiguous — which is what pool_sub_view demands when the
    # full-attention decode path builds its zero-copy kernel views.  3 blocks, not 2: with
    # exactly two blocks the block-first and the legacy K/V-first shapes are indistinguishable,
    # and `fused_halves` refuses that ambiguity on purpose.
    raw = torch.zeros((2, blocks, block_size, 2, 256), dtype=dtype)
    return raw.permute(1, 0, 2, 3, 4)


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    reset_counters()
    # The impl's reference singleton binds the default device, which is npu:0 wherever
    # torch_npu is importable — a test-only mismatch with the CPU tensors below.  The engine
    # process keeps both on the device, so this patches the seam, not the behaviour.
    import tileinfer.vllm.attn.impl as impl_mod

    monkeypatch.setattr(
        impl_mod,
        "_reference_attn",
        lambda: BatchAttention(backend="reference", device="cpu"),
    )
    yield
    reset_counters()


def test_windowed_layer_keeps_the_vllm_parameter_contract():
    """``impl.sliding_window`` stays ``None`` (get_per_layer_parameters indexes it as a tuple)."""
    impl = _impl(sliding_window=64)
    assert impl.window_size == 64
    assert impl.sliding_window is None
    assert _impl().window_size is None


@pytest.mark.parametrize("bad", [0, -5])
def test_window_must_be_positive(bad):
    with pytest.raises(ValueError, match="positive"):
        _impl(sliding_window=bad)


def test_windowed_decode_takes_the_staged_kernel_path():
    impl = _impl(sliding_window=64)
    n = 2
    out = impl.forward(
        None,
        torch.zeros(n, 16, 256, dtype=torch.bfloat16),
        torch.zeros(n, 2, 256, dtype=torch.bfloat16),
        torch.zeros(n, 2, 256, dtype=torch.bfloat16),
        _pool(),
        _meta(query_lens=(1, 1), seq_lens=(4, 7)),
        torch.zeros(n, 16 * 256),
    )
    assert out.shape == (n, 16 * 256)
    assert IMPL_COUNTERS.get("decode_window_staged") == 1
    assert IMPL_COUNTERS.get("decode_window_reference") == 0
    assert IMPL_COUNTERS.get("decode_steps") == 0


def test_windowed_decode_reference_escape_hatch(monkeypatch):
    """TILEINFER_WINDOW_DECODE=0 is the A/B control: the masked reference decode."""
    import tileinfer.vllm.attn.impl as impl_mod

    monkeypatch.setenv("TILEINFER_WINDOW_DECODE", "0")
    impl = _impl(sliding_window=64)
    n = 2
    impl.forward(
        None,
        torch.zeros(n, 16, 256, dtype=torch.bfloat16),
        torch.zeros(n, 2, 256, dtype=torch.bfloat16),
        torch.zeros(n, 2, 256, dtype=torch.bfloat16),
        _pool(),
        _meta(query_lens=(1, 1), seq_lens=(4, 7)),
        torch.zeros(n, 16 * 256),
    )
    assert IMPL_COUNTERS.get("decode_window_reference") == 1
    assert IMPL_COUNTERS.get("decode_window_staged") == 0


def test_windowed_decode_staged_numerics_match_the_masked_oracle():
    """The staged kernel path must equal the masked reference — window crossing a page boundary.

    2,500 tokens of context (pages 0 and 1 of the engine pool), window 768: the visible range
    [1,732, 2,499] straddles the page boundary at 2,048, so the gather must splice two engine
    pages into consecutive scratch pages.  The kernel path here is the reference-backed decode
    (CPU), so this pins the *staging* — the same indices feed the tilelang kernel on device.
    """
    impl = _impl(sliding_window=768)
    total, window = 2500, 768
    dtype = torch.bfloat16
    pool = _pool(blocks=9)  # [2, 9, 2048, 2, 256] logical, engine stride order
    _, _, block_size = 2, 9, 2048

    g = torch.Generator().manual_seed(3)
    k_all = torch.randn(total, 2, 256, generator=g).to(dtype)
    v_all = torch.randn(total, 2, 256, generator=g).to(dtype)

    # Tokens 0..total-2 pre-written (the engine pool holds the past); forward writes the last.
    half = pool[:, 0]  # K half: [blocks, block_size, kv_heads, head_size]
    vhalf = pool[:, 1]
    for pos in range(total - 1):
        half[pos // block_size, pos % block_size] = k_all[pos]
        vhalf[pos // block_size, pos % block_size] = v_all[pos]

    q = torch.randn(1, 16, 256, generator=g).to(dtype)
    meta = _meta(query_lens=(1,), seq_lens=(total,), sliding_window=768)
    # The engine table for this request: pages 0 and 1 are live.
    meta.block_table_tensor[0, 0] = 0
    meta.block_table_tensor[0, 1] = 1

    out = torch.zeros(1, 16 * 256, dtype=dtype)
    impl.forward(None, q, k_all[total - 1:], v_all[total - 1:], pool, meta, out)

    # Oracle: the query at absolute position 2,499 attends [2,499-768+1, 2,499] of the pool.
    # Read back from the *pool* (bf16 storage), not from the pre-write tensors: the kernel and
    # the reference both consume the rounded values, so the oracle must too.
    lo = total - window  # 1,732
    ks = half[lo // block_size : (total - 1) // block_size + 1].reshape(-1, 2, 256)[
        lo:total
    ].float()
    vs = vhalf[lo // block_size : (total - 1) // block_size + 1].reshape(-1, 2, 256)[
        lo:total
    ].float()
    kf = ks.repeat_interleave(8, dim=1)  # GQA group 16/2
    vf = vs.repeat_interleave(8, dim=1)
    scores = torch.einsum("hd,khd->hk", q[0].float(), kf) * (256**-0.5)
    probs = torch.softmax(scores, dim=-1)
    expected = torch.einsum("hk,khd->hd", probs, vf)
    got = out.reshape(1, 16, 256)[0].float()
    diff = (got - expected).abs().max().item()
    assert diff < 5e-2, f"staged window decode mismatch: max abs diff {diff:.4f}"


def test_windowed_decode_staged_twice_steps_are_stable():
    """A second decode step (plan ready) must stay correct — the scratch is reused, not stale."""
    impl = _impl(sliding_window=256)
    dtype = torch.bfloat16
    pool = _pool(blocks=9)
    g = torch.Generator().manual_seed(5)
    k_all = torch.randn(600, 2, 256, generator=g).to(dtype)
    v_all = torch.randn(600, 2, 256, generator=g).to(dtype)
    half, vhalf = pool[:, 0], pool[:, 1]
    for pos in range(599):
        half[pos // 2048, pos % 2048] = k_all[pos]
        vhalf[pos // 2048, pos % 2048] = v_all[pos]

    outs = []
    for step in range(2):
        total = 600  # the context length after this step's write
        q = torch.randn(1, 16, 256, generator=g).to(dtype)
        meta = _meta(query_lens=(1,), seq_lens=(total,), sliding_window=256)
        meta.block_table_tensor[0, 0] = 0
        out = torch.zeros(1, 16 * 256, dtype=dtype)
        impl.forward(None, q, k_all[599:], v_all[599:], pool, meta, out)
        outs.append((q, out))

    for q, out in outs:
        lo = total - 256
        ks = half[lo // 2048 : (total - 1) // 2048 + 1].reshape(-1, 2, 256)[lo:total].float()
        vs = vhalf[lo // 2048 : (total - 1) // 2048 + 1].reshape(-1, 2, 256)[lo:total].float()
        ks = ks.repeat_interleave(8, dim=1)  # GQA group 16/2
        vs = vs.repeat_interleave(8, dim=1)
        scores = torch.einsum("hd,khd->hk", q[0].float(), ks) * (256**-0.5)
        probs = torch.softmax(scores, dim=-1)
        expected = torch.einsum("hk,khd->hd", probs, vs)
        diff = (out.reshape(16, 256).float() - expected).abs().max().item()
        assert diff < 5e-2, f"staged decode mismatch: {diff:.4f}"


def test_windowed_prefill_skips_the_paged_prefill_kernel():
    impl = _impl(sliding_window=64)
    n = 5
    impl.forward(
        None,
        torch.zeros(n, 16, 256, dtype=torch.bfloat16),
        torch.zeros(n, 2, 256, dtype=torch.bfloat16),
        torch.zeros(n, 2, 256, dtype=torch.bfloat16),
        _pool(),
        _meta(query_lens=(3, 2), seq_lens=(4, 7)),
        torch.zeros(n, 16 * 256),
    )
    assert IMPL_COUNTERS.get("prefill_window_reference") == 1
    assert IMPL_COUNTERS.get("prefill_kernel") == 0
    assert IMPL_COUNTERS.get("prefill_reference") == 0


def test_full_attention_decode_is_unchanged_by_the_window_code():
    impl = _impl()
    n = 2
    # blocks=9: the kernel-plan path needs a staging table of bucket x width + 1 slots, and the
    # 3-block pool fails that start-up-shaped check before the step computes anything.
    impl.forward(
        None,
        torch.zeros(n, 16, 256, dtype=torch.bfloat16),
        torch.zeros(n, 2, 256, dtype=torch.bfloat16),
        torch.zeros(n, 2, 256, dtype=torch.bfloat16),
        _pool(blocks=9),
        _meta(query_lens=(1, 1), seq_lens=(4, 7)),
        torch.zeros(n, 16 * 256),
    )
    assert IMPL_COUNTERS.get("decode_steps") == 1
    assert IMPL_COUNTERS.get("decode_window_reference") == 0


def test_full_attention_prefill_is_unchanged_by_the_window_code():
    """The reference fallback still counts as before for a layer with no window."""
    impl = _impl()
    n = 5
    impl.forward(
        None,
        torch.zeros(n, 16, 256, dtype=torch.bfloat16),
        torch.zeros(n, 2, 256, dtype=torch.bfloat16),
        torch.zeros(n, 2, 256, dtype=torch.bfloat16),
        _pool(),
        _meta(query_lens=(3, 2), seq_lens=(4, 7)),
        torch.zeros(n, 16 * 256),
    )
    assert IMPL_COUNTERS.get("prefill_steps") == 1
    assert IMPL_COUNTERS.get("prefill_window_reference") == 0


# ---------------------------------------------------------------------------------------
# 3. builder: a windowed spec cannot be graph-captured
# ---------------------------------------------------------------------------------------


def test_windowed_spec_downgrades_cudagraph_support():
    windowed = _builder(sliding_window=4096)
    assert windowed.get_cudagraph_support(None, _Spec(2048, 4096)) is AttentionCGSupport.NEVER


def test_plain_spec_keeps_the_declared_cudagraph_support():
    builder = _builder()
    support = builder.get_cudagraph_support(None, _Spec(2048))
    assert support is builder._cudagraph_support
