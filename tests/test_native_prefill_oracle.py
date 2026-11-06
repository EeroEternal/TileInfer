"""Three-way prefill check on the device: kernel vs reference vs an **independent fp32 oracle**.

Why a third opinion
-------------------
Two implementations that share a misunderstanding will happily agree with each other, and this
repository has already paid for that: before §18.2 the paged-prefill *kernel* and the *reference*
backend were both off by the same head/token transposition, and their mutual difference was small
enough (0.0488) to look like bf16 noise.  What settled it was a third opinion written from the K/V
source tensors, sharing no code with either path: the kernel matched it to 1.7e-3 while the
reference was off by 0.665 **at token 0**, where the answer is simply ``v[0]``.

So this test asserts, for every sampled token, that
``|kernel - oracle|`` and ``|reference - oracle|`` are both within bf16 noise.  It is the
prefill-side counterpart of the logprob comparison C3 requires, and it fails *per implementation*
instead of only when two wrong things differ.

Shapes are the ones that matter: a plain 512-token prefill (the M1 exit-criterion length, head_dim
256 as in Qwen3.6's full-attention layers) and a ragged 300+200 batch (the case the reference got
wrong twice).
"""

from __future__ import annotations

import pytest
import torch

from tileinfer.utils import is_npu_available

try:  # pragma: no cover
    import tilelang.ascend  # noqa: F401

    _HAVE_ASCEND950 = True
except Exception:  # pragma: no cover
    _HAVE_ASCEND950 = False

pytestmark = [
    pytest.mark.npu,
    pytest.mark.skipif(not is_npu_available(), reason="no Ascend NPU available"),
    pytest.mark.skipif(not _HAVE_ASCEND950, reason="tilelang.ascend (>=0.1.15) not importable"),
]

DT = torch.bfloat16
NUM_HEADS, NUM_KV_HEADS, HEAD_DIM = 16, 2, 256
GROUP = NUM_HEADS // NUM_KV_HEADS
ENGINE_BLOCK = 2048
POOL_BLOCKS = 64
#: bf16 through a softmax of up to 512 keys; the kernel's P is bf16, so this is ulp-level, not slack.
TOL = 5e-3


def _build(seq_lens, qo_lens, seed=0):
    """Prime a fused NHD pool with ``seq_lens`` tokens, and return a prefill step over its tail."""
    from tileinfer.serving.buckets import fused_halves
    from tileinfer.vllm.attn.kv_cache import write_kv_cache

    dev = torch.device("npu", 0)
    torch.manual_seed(seed)
    torch.npu.manual_seed_all(seed)
    num_reqs = len(seq_lens)
    max_pages = max((s + ENGINE_BLOCK - 1) // ENGINE_BLOCK for s in seq_lens)
    assert num_reqs * max_pages <= POOL_BLOCKS, "test pool too small"

    table = torch.zeros(num_reqs, max_pages, dtype=torch.int32, device=dev)
    slots = []
    for r, seq_len in enumerate(seq_lens):
        for page in range((seq_len + ENGINE_BLOCK - 1) // ENGINE_BLOCK):
            table[r, page] = r * max_pages + page
        for pos in range(seq_len):
            page, off = divmod(pos, ENGINE_BLOCK)
            slots.append(int(table[r, page]) * ENGINE_BLOCK + off)
    slots = torch.tensor(slots, dtype=torch.int64, device=dev)

    total, num_tokens = sum(seq_lens), sum(qo_lens)
    # The fused pool's K/V factor is outermost in memory; a plain [blocks, 2, ...] allocation is
    # *not* that layout and `fused_halves` refuses it (correctly).
    kv_cache = torch.zeros(2, POOL_BLOCKS, ENGINE_BLOCK, NUM_KV_HEADS, HEAD_DIM,
                           dtype=DT, device=dev).permute(1, 0, 2, 3, 4)
    k_all = torch.randn(total, NUM_KV_HEADS, HEAD_DIM, dtype=DT, device=dev) * 0.2
    v_all = torch.randn(total, NUM_KV_HEADS, HEAD_DIM, dtype=DT, device=dev) * 0.2
    k_half, v_half = fused_halves(kv_cache)
    write_kv_cache(k_all, v_all, k_half, v_half, slots)
    q_all = torch.randn(num_tokens, NUM_HEADS, HEAD_DIM, dtype=DT, device=dev) * 0.2
    return dev, table, slots, kv_cache, k_all, v_all, q_all


def _meta(seq_lens, qo_lens, table, slots, dev):
    from tileinfer.vllm.attn.metadata import TileInferMetadata

    num_reqs = len(seq_lens)
    qo = torch.tensor(qo_lens, dtype=torch.int32)
    qsl = torch.zeros(num_reqs + 1, dtype=torch.int32)
    qsl[1:] = torch.cumsum(qo, 0)
    return TileInferMetadata(
        query_start_loc=qsl.to(dev),
        query_start_loc_cpu=qsl,
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=dev),
        block_table_tensor=table,
        slot_mapping=slots,
        num_reqs=num_reqs,
        num_actual_tokens=int(sum(qo_lens)),
        max_query_len=int(max(qo_lens)),
        bucket=512,
        block_size=ENGINE_BLOCK,
    )


def _oracle(t, start, q_all, k_all, v_all):
    """Plain fp32 causal attention for request starting at ``start``, token ``t``."""
    kk = k_all[start : start + t + 1].float().cpu()
    vv = v_all[start : start + t + 1].float().cpu()
    qq = q_all[start + t].float().cpu()
    out = torch.zeros(NUM_HEADS, HEAD_DIM)
    for h in range(NUM_HEADS):
        kh = h // GROUP
        probs = torch.softmax((kk[:, kh] * qq[h]).sum(-1) * HEAD_DIM**-0.5, 0)
        out[h] = (probs[:, None] * vv[:, kh]).sum(0)
    return out


def _run_both(impl, q_all, k_all, v_all, kv_cache, meta, num_tokens, dev):
    out_k = torch.zeros(num_tokens, NUM_HEADS * HEAD_DIM, dtype=DT, device=dev)
    out_r = torch.zeros_like(out_k)
    impl._prefill_enabled = True
    impl.forward(None, q_all, k_all, v_all, kv_cache, meta, out_k)
    impl._prefill_enabled = False
    impl.forward(None, q_all, k_all, v_all, kv_cache, meta, out_r)
    torch.npu.synchronize()
    return out_k, out_r


def _check(seq_lens, qo_lens):
    from tileinfer.vllm.attn.impl import TileInferAttentionImpl, reset_runtime_state

    dev, table, slots, kv_cache, k_all, v_all, q_all = _build(seq_lens, qo_lens)
    meta = _meta(seq_lens, qo_lens, table, slots, dev)
    impl = TileInferAttentionImpl(NUM_HEADS, HEAD_DIM, HEAD_DIM**-0.5, num_kv_heads=NUM_KV_HEADS)
    try:
        out_k, out_r = _run_both(impl, q_all, k_all, v_all, kv_cache, meta, sum(qo_lens), dev)
        start = 0
        for r, seq_len in enumerate(seq_lens):
            for t in sorted({0, 1, qo_lens[r] - 1}):
                if t < 0 or t >= qo_lens[r]:
                    continue
                want = _oracle(t, start, q_all, k_all, v_all)
                row = sum(qo_lens[:r]) + t
                got_k = out_k[row].float().cpu().reshape(NUM_HEADS, HEAD_DIM)
                got_r = out_r[row].float().cpu().reshape(NUM_HEADS, HEAD_DIM)
                dk = float((got_k - want).abs().max())
                dr = float((got_r - want).abs().max())
                assert dk < TOL, f"kernel vs oracle at req{r} tok{t}: {dk:.5f} >= {TOL}"
                assert dr < TOL, f"reference vs oracle at req{r} tok{t}: {dr:.5f} >= {TOL}"
            start += seq_len
    finally:
        reset_runtime_state()


def test_plain_prefill_512_matches_the_oracle():
    _check([512], [512])


def test_ragged_prefill_matches_the_oracle():
    _check([300, 200], [300, 200])
