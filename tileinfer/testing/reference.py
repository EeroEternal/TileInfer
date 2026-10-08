"""Reference implementation of paged / ragged attention in pure torch.

Two jobs: it is the correctness oracle for every kernel (``tests/``), and it is the executable
specification of the *plan* format — in particular of the split-KV + merge protocol, which the
TileLang kernels must reproduce exactly.  It runs on CPU or NPU; nothing here is Ascend-specific,
and nothing here is meant to be fast.

Terminology matches :mod:`tileinfer.metadata`:

* query      ``[batch, num_qo_heads, qo_len, head_dim]``
* K/V cache  ``[num_pages, page_size, num_kv_heads, head_dim]`` (``NHD``) or
             ``[num_pages, num_kv_heads, page_size, head_dim]`` (``HND``)
* metadata   ``kv_indptr`` / ``kv_indices`` / ``kv_last_page_len`` / ``page_size``

K and V are separate tensors, mirroring how serving engines (vLLM, MindIE) and CANN's FIA
allocate them; a fused ``[2, ...]`` cache is the caller's business, not the kernel's.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from ..metadata import RaggedMetadata
from ..plan import TileSchedule

__all__ = [
    "gather_kv",
    "reference_attention",
    "reference_attention_tiled",
    "merge_partials",
    "expand_kv_heads",
]


def _kv_view(cache: torch.Tensor, layout: str) -> torch.Tensor:
    """Normalise a paged cache to ``[num_pages, page_size, num_kv_heads, head_dim]``."""
    layout = layout.upper()
    if layout == "NHD":
        return cache
    if layout == "HND":
        return cache.permute(0, 2, 1, 3)
    raise ValueError(f"unknown kv layout {layout!r} (expected 'NHD' or 'HND')")


def gather_kv(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    meta: RaggedMetadata,
    request: int,
    layout: str = "NHD",
    page_start: int = 0,
    page_count: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Materialise request ``request``'s ``(K, V)`` as ``[kv_len, num_kv_heads, head_dim]``.

    ``page_start`` / ``page_count`` restrict the gather to a slice of the request's pages — which
    is exactly what a split tile sees.
    """
    k = _kv_view(k_cache, layout)
    v = _kv_view(v_cache, layout)
    page_size = max(meta.page_size, 1)
    start = int(meta.kv_indptr[request].item()) + page_start
    end = int(meta.kv_indptr[request + 1].item())
    if page_count is not None:
        end = min(end, start + page_count)
    pages = meta.kv_indices[start:end].to(torch.long)
    if pages.numel() == 0:
        empty = k.new_zeros((0, k.shape[2], k.shape[3]))
        return empty, empty.clone()

    if page_size == 1:
        return k[pages], v[pages]

    # only the *last* page of a request can be partially filled
    last_len = (
        int(meta.kv_last_page_len[request].item())
        if meta.kv_last_page_len is not None
        else page_size
    )
    is_last_included = end >= int(meta.kv_indptr[request + 1].item())
    # flatten pages into a token sequence first: page 0..n-1 are full pages, only the last one
    # can be short, so the tail is trimmed *after* flattening
    kb = k[pages].reshape(-1, k.shape[2], k.shape[3])
    vb = v[pages].reshape(-1, v.shape[2], v.shape[3])
    if is_last_included and last_len != page_size:
        keep = kb.shape[0] - page_size + last_len
        kb, vb = kb[:keep], vb[:keep]
    return kb, vb


def expand_kv_heads(kv: torch.Tensor, group_size: int) -> torch.Tensor:
    """Repeat KV heads so they line up with the query heads (GQA / MQA)."""
    if group_size == 1:
        return kv
    return kv.repeat_interleave(group_size, dim=1)


def _causal_offset(qo_len: int, kv_len: int, causal: bool) -> int:
    """First KV position visible to query row 0.

    ``0`` for a plain prefill; ``kv_len - qo_len`` for an append step (chunked prefill,
    speculative decode), which is where the causal mask must start.  Getting this wrong is the
    classic chunked-prefill bug.
    """
    if not causal:
        return 0
    return max(0, kv_len - qo_len)


def _mask_scores(
    scores: torch.Tensor,
    qo_len: int,
    kv_len: int,
    causal: bool,
    window_left: int = -1,
    q_offset: int = 0,
) -> torch.Tensor:
    """Apply causal / sliding-window masking to ``[H, qo_len, kv_len]`` scores."""
    device = scores.device
    base = _causal_offset(qo_len, kv_len, causal) + q_offset if causal else 0
    q_pos = torch.arange(qo_len, device=device).unsqueeze(1) + base
    k_pos = torch.arange(kv_len, device=device).unsqueeze(0)
    mask = torch.zeros((qo_len, kv_len), dtype=torch.bool, device=device)
    if causal:
        mask |= k_pos > q_pos
    if window_left >= 0:
        mask |= k_pos < (q_pos - window_left)
    return scores.masked_fill(mask, float("-inf"))


def reference_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    meta: RaggedMetadata,
    *,
    layout: str = "NHD",
    causal: bool = True,
    sm_scale: Optional[float] = None,
    window_left: int = -1,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Straight-line oracle (no scheduler).  ``q`` has a *uniform* ``qo_len`` per request."""
    meta.validate()
    batch, h_q, qo_len, d = q.shape
    h_kv = k_cache.shape[2] if layout.upper() == "NHD" else k_cache.shape[1]
    group = h_q // h_kv
    scale = sm_scale if sm_scale is not None else 1.0 / (d**0.5)
    out = torch.zeros((batch, h_q, qo_len, d), dtype=torch.float32, device=q.device)

    for b in range(batch):
        kb, vb = gather_kv(k_cache, v_cache, meta, b, layout)
        kb = expand_kv_heads(kb.float(), group)
        vb = expand_kv_heads(vb.float(), group)
        kv_len = kb.shape[0]
        if kv_len == 0:
            continue
        qb = q[b].float().permute(1, 0, 2)  # [qo_len, H, D]
        scores = torch.einsum("qhd,khd->hqk", qb, kb) * scale
        scores = _mask_scores(scores, qo_len, kv_len, causal, window_left)
        probs = torch.softmax(scores, dim=-1)
        out[b] = torch.einsum("hqk,khd->qhd", probs, vb).permute(1, 0, 2)

    return out.to(out_dtype or q.dtype)


def reference_attention_tiled(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    meta: RaggedMetadata,
    schedule: TileSchedule,
    *,
    layout: str = "NHD",
    causal: bool = True,
    sm_scale: Optional[float] = None,
    window_left: int = -1,
    out_dtype: Optional[torch.dtype] = None,
    return_partials: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Execute a :class:`TileSchedule` tile by tile, honouring split-KV and merging.

    This is the oracle for the *plan* machinery itself: it consumes exactly the arrays a kernel
    would consume (per-tile ``seq_id`` / ``q_offset`` / ``kv_page_start`` / ``split_id``), so a
    scheduler bug shows up here as a mismatch against :func:`reference_attention`.
    """
    meta.validate()
    batch, h_q, qo_len_total, d = q.shape
    h_kv = k_cache.shape[2] if layout.upper() == "NHD" else k_cache.shape[1]
    group = h_q // h_kv
    scale = sm_scale if sm_scale is not None else 1.0 / (d**0.5)
    device = q.device

    merge = schedule.needs_merge
    if merge and (qo_len_total != 1 or bool((schedule.q_lens != 1).any())):
        raise NotImplementedError(
            "split-KV merge is defined for decode only (one query row per tile)"
        )

    qo_lens_full = (
        (meta.qo_indptr[1:] - meta.qo_indptr[:-1]).tolist() if meta.qo_indptr is not None else None
    )
    assert qo_lens_full is not None, "tiled reference requires qo_indptr"
    page_size = max(meta.page_size, 1)

    num_tiles = schedule.num_tiles
    out = torch.zeros((batch, h_q, qo_len_total, d), dtype=torch.float32, device=device)
    partial_out = (
        torch.zeros((num_tiles, h_q, d), dtype=torch.float32, device=device)
        if (return_partials or merge)
        else None
    )
    partial_lse = (
        torch.full((num_tiles, h_q), float("-inf"), dtype=torch.float32, device=device)
        if (return_partials or merge)
        else None
    )

    for t in range(num_tiles):
        b = int(schedule.seq_ids[t].item())
        q_off = int(schedule.q_offsets[t].item())
        tile_q_len = min(int(schedule.q_lens[t].item()), qo_lens_full[b] - q_off)
        if tile_q_len <= 0:
            continue
        kb, vb = gather_kv(
            k_cache,
            v_cache,
            meta,
            b,
            layout,
            page_start=int(schedule.kv_page_starts[t].item()),
            page_count=int(schedule.kv_page_lens[t].item()),
        )
        kb = expand_kv_heads(kb.float(), group)
        vb = expand_kv_heads(vb.float(), group)
        kv_len_full = int(meta.kv_lens[b].item())
        kv_pos = torch.arange(kb.shape[0], device=device) + int(
            schedule.kv_page_starts[t].item()
        ) * page_size

        qb = q[b, :, q_off : q_off + tile_q_len, :].float().permute(1, 0, 2)  # [S, H, D]
        scores = torch.einsum("qhd,khd->hqk", qb, kb) * scale

        mask = torch.zeros((tile_q_len, kb.shape[0]), dtype=torch.bool, device=device)
        if causal:
            q_pos = torch.arange(tile_q_len, device=device).unsqueeze(1) + (
                kv_len_full - qo_lens_full[b] + q_off
            )
            mask |= kv_pos.unsqueeze(0) > q_pos
        if window_left >= 0:
            mask |= kv_pos.unsqueeze(0) < (q_pos - window_left)
        scores = scores.masked_fill(mask, float("-inf"))

        m = scores.amax(dim=-1)  # [H, S]
        m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
        p = torch.exp(scores - m.unsqueeze(-1))
        denom = p.sum(dim=-1)
        ob = torch.einsum("hqk,khd->qhd", p, vb)  # [S, H, D]
        lse = m + torch.log(torch.clamp(denom, min=1e-30))

        normalised = ob / denom.transpose(0, 1).unsqueeze(-1)  # [S, H, D]

        if merge:
            # contract: partial_out holds the *normalised* output of the tile and partial_lse its
            # log-sum-exp.  Merging is then a weighted average with weights exp(lse_t - lse_all),
            # which sum to exactly 1 because exp(lse_all) = sum_t exp(lse_t).
            partial_out[t] = normalised[0]
            partial_lse[t] = lse[:, 0]
        else:
            out[b, :, q_off : q_off + tile_q_len, :] = normalised.permute(1, 0, 2)

    if merge and partial_out is not None and partial_lse is not None:
        out = merge_partials(partial_out, partial_lse, schedule, meta, h_q)

    return out.to(out_dtype or q.dtype), partial_out, partial_lse


def merge_partials(
    partial_out: torch.Tensor,
    partial_lse: torch.Tensor,
    schedule: TileSchedule,
    meta: RaggedMetadata,
    num_qo_heads: int,
) -> torch.Tensor:
    """Combine split tiles: ``out = Σ_t w_t · partial_out[t]`` with ``w_t = exp(lse_t − lse_all)``.

    ``partial_out[t]`` is the *normalised* output of tile ``t`` (shape ``[H, D]`` for the single
    query row of a decode step) and ``partial_lse[t]`` its log-sum-exp (shape ``[H]``).  The
    weights sum to exactly 1 because ``exp(lse_all) = Σ_t exp(lse_t)``, so the merge is a plain
    weighted average — no second division, which is what makes it cheap in a kernel.

    This is the contract the merge kernel has to implement.
    """
    device = partial_out.device
    d = partial_out.shape[-1]
    per_request: dict = {}
    for t in range(schedule.num_tiles):
        per_request.setdefault(int(schedule.seq_ids[t].item()), []).append(t)

    out = torch.zeros((meta.batch_size, num_qo_heads, 1, d), dtype=partial_out.dtype, device=device)
    for b, tiles in per_request.items():
        lse_stack = torch.stack([partial_lse[t] for t in tiles], dim=0)  # [S, H]
        lse = torch.logsumexp(lse_stack, dim=0)  # [H]
        acc = torch.zeros((num_qo_heads, d), dtype=torch.float32, device=device)
        for t in tiles:
            weight = torch.exp(partial_lse[t] - lse)  # [H], sums to 1 across t
            acc = acc + weight[:, None] * partial_out[t]
        out[b, :, 0, :] = acc
    return out
