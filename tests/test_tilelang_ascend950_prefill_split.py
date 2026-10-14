"""Device tests for the split-KV **prefill** kernels (H19/H20).

The split step must be numerically transparent: every (q-block, split) partial goes through
the LSE-weighted merge, so the merged output must equal both

* the fp32 torch oracle (:func:`reference_attention`) — the absolute contract, and
* the *unsplit* prefill kernel on identical inputs — the equivalence contract (same arithmetic,
  different work partition).

H20 runs two kernels per step (head = fully-visible page chunks with no per-row predicate,
frontier = the plain kernel's RowThr form over ``[P_vis, pages)``), launched from the same
``forward_prefill_split`` public path, so these cases exercise both.  Cases cover: plain
prefill over many pages (no prefix — pure frontier form), append with a partial tail page
(mixed head+frontier), ragged split counts in one batch, page-granular splits, head_dim 256
in the serving configuration (kvh=2, group=8), and the P3 padding paths.
"""

from __future__ import annotations

import pytest
import torch

from tileinfer.testing.reference import reference_attention
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

DTYPE = torch.bfloat16
TOL = 3e-2
BLOCK_Q = 128


def _next_pow2(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def _case(dim, kv_lens, qo_lens, ktp, group=8, kv_heads=2, seed=0, pad_batch=0):
    """Run the split prefill through the public path; return (max diff vs oracle, vs unsplit)."""
    from tileinfer.kernels.attention.paged_decode_ascend950 import (
        declared_pool_cap,
        forward_prefill,
        forward_prefill_split,
        plan_prefill_split_ctx,
    )
    from tileinfer.metadata import RaggedMetadata
    from tileinfer.plan import plan_query_tiles_from_host

    batch = len(kv_lens)
    page_size = 128
    device = torch.device("npu", 0)
    torch.manual_seed(seed)

    indptr, last = [0], []
    for L in kv_lens:
        indptr.append(indptr[-1] + -(-L // page_size))
        last.append(page_size if L % page_size == 0 else L % page_size)
    total = indptr[-1]

    k_cache = torch.randn(total, page_size, kv_heads, dim, dtype=DTYPE, device=device)
    v_cache = torch.randn(total, page_size, kv_heads, dim, dtype=DTYPE, device=device)
    indices = torch.arange(total, dtype=torch.int32, device=device)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    for b in range(batch):
        perm = torch.randperm(indptr[b + 1] - indptr[b], generator=gen)
        indices[indptr[b] : indptr[b + 1]] = indices[indptr[b] : indptr[b + 1]][perm.to(device)]
    cap = declared_pool_cap(total)
    if indices.numel() < cap:
        indices = torch.cat([indices, indices.new_zeros(cap - indices.numel())])

    qo = torch.tensor(qo_lens, dtype=torch.int32, device=device)
    qo_indptr = torch.zeros(batch + 1, dtype=torch.int32, device=device)
    qo_indptr[1:] = torch.cumsum(qo, dim=0).to(torch.int32)
    meta = RaggedMetadata(
        kv_indptr=torch.tensor(indptr, dtype=torch.int32, device=device),
        kv_indices=indices.clone(),
        kv_last_page_len=torch.tensor(last, dtype=torch.int32, device=device),
        page_size=page_size,
        qo_indptr=qo_indptr,
    )

    max_q = max(qo_lens)
    q = torch.randn(batch, kv_heads * group, max_q, dim, dtype=DTYPE, device=device) * 0.2

    pages_host = [indptr[b + 1] - indptr[b] for b in range(batch)]
    lpl_host = list(last)
    qo_rows = [ql * group for ql in qo_lens]
    split_tiles = sum(((r + BLOCK_Q - 1) // BLOCK_Q) * max(1, -(-p // ktp))
                      for r, p in zip(qo_rows, pages_host))
    kernel_batch = _next_pow2(batch) if batch > 1 else 1
    kernel_batch += pad_batch
    kernel_tiles = _next_pow2(split_tiles)

    ctx = plan_prefill_split_ctx(
        list(qo_lens),
        pages_host,
        lpl_host,
        group=group,
        kv_tile_pages=ktp,
        kernel_batch=kernel_batch,
        kernel_tiles=kernel_tiles,
        kv_heads=kv_heads,
        block_q=BLOCK_Q,
        max_qo_tokens=max_q,
        page_size=page_size,
        indices=indices,
        device=device,
    )
    got = forward_prefill_split(
        q, k_cache, v_cache, ctx, group=group
    )
    torch.npu.synchronize()

    # unsplit kernel on the same inputs (equivalence contract)
    ref_sched = plan_query_tiles_from_host(
        qo_rows, pages_host, device=device, block_q=BLOCK_Q, kv_tile_pages=0, load_balance=False
    )
    if kernel_tiles != ref_sched.num_tiles:
        from tileinfer.integrations.vllm_ascend import _pad_tile_schedule

        ref_sched = _pad_tile_schedule(ref_sched, kernel_tiles, kernel_batch - 1)
    # the padded-schedule kernel is built for kernel_batch rows; forward_prefill pads
    # indptr/lpl internally but the caller must pad qo_lens the same way (trash rows = 0)
    if kernel_batch > batch:
        pad = kernel_batch - batch
        qo = torch.cat([qo, torch.zeros(pad, dtype=qo.dtype, device=qo.device)])
    unsplit = forward_prefill(
        q, k_cache, v_cache, meta, ref_sched, qo, group=group, block_q=BLOCK_Q,
        kernel_batch=kernel_batch,
    )
    torch.npu.synchronize()

    host = RaggedMetadata(
        kv_indptr=meta.kv_indptr.cpu(),
        kv_indices=meta.kv_indices.cpu(),
        kv_last_page_len=meta.kv_last_page_len.cpu(),
        page_size=page_size,
        qo_indptr=meta.qo_indptr.cpu(),
    )
    expected = torch.zeros_like(q).cpu()  # the oracle runs on the CPU
    for b in range(batch):
        qb = q[b : b + 1, :, : qo_lens[b], :].cpu().float()
        base = int(host.kv_indptr[b].item())
        end = int(host.kv_indptr[b + 1].item())
        sub = RaggedMetadata(
            kv_indptr=torch.tensor([0, end - base], dtype=torch.int32),
            kv_indices=host.kv_indices[base:end] - base,
            kv_last_page_len=host.kv_last_page_len[b : b + 1],
            page_size=page_size,
            qo_indptr=torch.tensor([0, qo_lens[b]], dtype=torch.int32),
        )
        exp_b = reference_attention(
            qb,
            k_cache[base:end].cpu().float(),
            v_cache[base:end].cpu().float(),
            sub,
            causal=True,
        )
        expected[b, :, : qo_lens[b], :] = exp_b.to(DTYPE)
    live = max(qo_lens)
    d_oracle = (got[:, :, :live, :].cpu().float() - expected[:, :, :live, :].float()).abs().max().item()
    d_unsplit = (got - unsplit).abs().max().item()
    return d_oracle, d_unsplit


@pytest.mark.parametrize(
    "dim,kv_lens,qo_lens,ktp,label",
    [
        (128, [2048], [2048], 4, "plain prefill 16 pages, 4 splits"),
        (128, [2200], [512], 4, "append over 2200-token prefix, partial tail"),
        (128, [1536, 512], [512, 128], 2, "ragged split counts in one batch"),
        (128, [1024], [128], 1, "page-granular splits (8 x 1-page)"),
        (256, [4096], [1024], 8, "d256 serving shape (kvh=2, group=8)"),
    ],
)
def test_prefill_split_matches_oracle_and_unsplit(dim, kv_lens, qo_lens, ktp, label):
    d_oracle, d_unsplit = _case(dim, kv_lens, qo_lens, ktp)
    assert d_oracle < TOL, f"{label}: oracle diff {d_oracle:.4f} exceeds {TOL}"
    assert d_unsplit < TOL, f"{label}: unsplit diff {d_unsplit:.4f} exceeds {TOL}"


def test_prefill_split_with_padding_paths():
    """Batch trash row + pow2 tile padding (P3 canonical variants) stay transparent."""
    d_oracle, d_unsplit = _case(128, [1536, 512], [512, 128], 2, pad_batch=2)
    assert d_oracle < TOL, f"padding paths: oracle diff {d_oracle:.4f}"
    assert d_unsplit < TOL, f"padding paths: unsplit diff {d_unsplit:.4f}"
