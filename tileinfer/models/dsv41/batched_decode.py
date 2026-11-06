"""True batched decode forward (`perf-plan.md` §24 的 C/D 步): R requests in **one** pass.

Why this module exists
----------------------
`_forward_multi` (``tileinfer/vllm/models/dsv41.py``) used to call ``transformer_forward`` once per
request, so a 4-request decode step cost ~4x a single step (measured 3009 ms/step).  This module runs
the 40 layers once with ``bsz = R``: every dense op (norm / linear / hc) naturally batches, the
attention window + compressed stream are gathered per request from the existing ``ModelState``
buffers, and the MoE goes through the grouped path (one expert GEMM per group, not one per request).

Scope (deliberately narrow)
---------------------------
**Decode only** (one token per request, ``seqlen == 1``).  Prefill stays on the serial path: it is a
one-off ~3 s cost, while the P3 target is a per-decode-step metric, and keeping prefill untouched
means the verified S3/S4/S5 path is not re-derived.  Requests are gathered from / scattered back to
their own ``ModelState`` each step, so no new ownership model is introduced: the per-request state
stays the single source of truth and a request can switch between serial and batched steps safely.

Gate: ``VLLM_DSV41_BATCHED_FWD`` (default off).  With the gate off nothing in this module runs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .block import logits_from_hidden, make_identity_pre_mix
from .config import Dsv41Config
from .layer import (
    _E8M0,
    _FP8,
    _linear_fp32,
    attention_forward,
    indexer_forward,
    ops,
    precompute_rope_cos_sin,
    rms_norm,
    select_candidate_blocks,
    window_topk_idxs_batched,
    write_ring_batched,
)
from .modules import Dsv41Attention, Dsv41Compressor, Dsv41Indexer

__all__ = [
    "BatchedShared",
    "LayerScratch",
    "transformer_forward_decode",
    "attention_decode",
    "build_scratch",
    "sync_back",
]

#: ``(compress_rope?, length) -> (cos, sin)``; the tables are position-only so one entry per shape.
_ROPE_CACHE: Dict[Tuple[bool, int], Tuple[torch.Tensor, torch.Tensor]] = {}
_TIMING_PRINTED = 0


def _sync() -> None:
    try:
        torch.npu.synchronize()
    except Exception:  # noqa: BLE001 - CPU test path
        pass


def _rope_tables(cfg: Dsv41Config, compress: bool, length: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
    key = (bool(compress), int(length))
    cached = _ROPE_CACHE.get(key)
    if cached is None or cached[0].device != torch.device(device):
        cached = precompute_rope_cos_sin(
            cfg.rope_head_dim,
            int(length),
            cfg.original_seq_len if compress else 0,
            cfg.compress_rope_theta if compress else cfg.rope_theta,
            cfg.rope_factor,
            cfg.beta_fast,
            cfg.beta_slow,
            device=device,
        )
        _ROPE_CACHE[key] = cached
    return cached


def _rotate_rows(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """RoPE on ``x [R, n, heads, d]`` with one ``(cos, sin)`` row per request (``[R, rd/2]``)."""
    rd = cos.size(-1) * 2
    target = x[..., -rd:]
    pairs = target.float().unflatten(-1, (-1, 2))
    x1, x2 = pairs[..., 0], pairs[..., 1]
    if inverse:
        sin = -sin
    if target.ndim == 3:
        # [R, n, d] -- no head axis (the window KV / compressed latent case)
        c = cos.view(cos.shape[0], 1, -1)
        s = sin.view(sin.shape[0], 1, -1)
    else:
        c = cos.view(cos.shape[0], 1, 1, -1)
        s = sin.view(sin.shape[0], 1, 1, -1)
    out = torch.stack([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1).flatten(-2)
    target.copy_(out.to(target.dtype))
    return x


@dataclass
class BatchedShared:
    """What attention layers publish down the stack (batched version of ``SharedAttentionState``)."""

    compress_kv: Optional[torch.Tensor] = None
    index_k: Optional[torch.Tensor] = None
    topk_idxs: Optional[torch.Tensor] = None
    candidates: Optional[torch.Tensor] = None


@dataclass
class LayerScratch:
    """Per-layer gathered buffers for one batched step; row ``i`` is request ``i``."""

    layer_id: int
    ratio: int
    window_kv: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor
    compress_kv: Optional[torch.Tensor] = None
    index_k: Optional[torch.Tensor] = None
    kv_state: Optional[torch.Tensor] = None
    score_state: Optional[torch.Tensor] = None
    new_latent: Optional[torch.Tensor] = None


@dataclass
class BatchedStep:
    """All scratches of one forward plus the shared publish slot and the input positions."""

    positions: torch.Tensor
    pos_list: List[int] = field(default_factory=list)
    layers: Dict[int, LayerScratch] = field(default_factory=dict)
    shared: BatchedShared = field(default_factory=BatchedShared)


def _layer_rt(state, layer_id: int, cfg: Dsv41Config):
    return state.runtime(layer_id, cfg.n_layers)


def build_scratch(states: Sequence[object], positions: Sequence[int], cfg: Dsv41Config,
                  device) -> BatchedStep:
    """Gather every layer's read-side state from the per-request ``ModelState`` objects.

    Only the prefix a request can actually read (``(pos + 1) // ratio`` compressed entries) is copied,
    so the cost tracks context length rather than the 1M allocation ceiling.
    """
    pos = torch.as_tensor([int(p) for p in positions], dtype=torch.long, device=device)
    pos_list = [int(p) for p in positions]
    R = int(pos.numel())
    need = max(2048, max(pos_list) + 1)
    step = BatchedStep(positions=pos, pos_list=pos_list)
    win = int(cfg.sliding_window)

    for layer_id in range(int(cfg.n_layers)):
        ratio = int(cfg.compress_ratio(layer_id))
        cos, sin = _rope_tables(cfg, ratio > 0, need, device)
        window = torch.stack([_layer_rt(s, layer_id, cfg).window_kv[0] for s in states], dim=0)
        sc = LayerScratch(layer_id=layer_id, ratio=ratio, window_kv=window, cos=cos, sin=sin)
        if cfg.is_kv_source(layer_id):
            comp_len = [(p + 1) // max(ratio, 1) for p in pos_list]
            max_len = max(comp_len) if comp_len else 0
            dim = int(cfg.head_dim)
            idx_dim = int(cfg.index_head_dim)
            # one extra row of capacity when nobody has a compressed entry yet keeps a later write
            # valid (max_len == 0 happens for a 1-token ratio-2 request)
            sc.compress_kv = torch.zeros(R, max(max_len, 1), dim, dtype=torch.bfloat16, device=device)
            sc.index_k = torch.zeros(R, max(max_len, 1), idx_dim, dtype=torch.bfloat16, device=device)
            if ratio > 1:
                sc.kv_state = torch.stack(
                    [_layer_rt(s, layer_id, cfg).kv_state[0] for s in states], dim=0
                ).to(torch.float32)
                sc.score_state = torch.stack(
                    [_layer_rt(s, layer_id, cfg).score_state[0] for s in states], dim=0
                ).to(torch.float32)
            for i, n in enumerate(comp_len):
                if n <= 0:
                    continue
                rt = _layer_rt(states[i], layer_id, cfg)
                sc.compress_kv[i, :n].copy_(rt.compress_kv[0, :n])
                sc.index_k[i, :n].copy_(rt.index_k[0, :n])
        step.layers[layer_id] = sc
    return step


def sync_back(states: Sequence[object], step: BatchedStep, cfg: Dsv41Config) -> None:
    """Write the step's state changes back into each request's own ``ModelState``."""
    win = int(cfg.sliding_window)
    for layer_id, sc in step.layers.items():
        ratio = int(sc.ratio)
        for i, state in enumerate(states):
            rt = _layer_rt(state, layer_id, cfg)
            p = int(step.pos_list[i])
            rt.window_kv[0].copy_(sc.window_kv[i])
            if cfg.is_kv_source(layer_id):
                n = (p + 1) // max(ratio, 1)
                if n > 0:
                    rt.compress_kv[0, :n].copy_(sc.compress_kv[i, :n])
                    rt.index_k[0, :n].copy_(sc.index_k[i, :n])
                if ratio > 1:
                    rt.kv_state[0].copy_(sc.kv_state[i])
                    rt.score_state[0].copy_(sc.score_state[i])


# =============================================================================================
# Attention: one batched decode step
# =============================================================================================


def _compressor_decode(
    comp: Dsv41Compressor,
    x: torch.Tensor,
    pos: torch.Tensor,
    pos_list: Sequence[int],
    sc: LayerScratch,
    cfg: Dsv41Config,
    o,
) -> Optional[torch.Tensor]:
    """``compressor_forward`` for seqlen == 1 per row; returns pooled latents (zero rows unused)."""
    ratio, dtype = int(comp.compress_ratio), x.dtype
    R = int(x.size(0))
    if ratio == 1:
        return rms_norm(comp.norm, o.linear(x, comp.wkv.weight, comp.wkv.scale))

    xf = x.float()
    kv = _linear_fp32(xf, comp.wkv)          # [R,1,dim]
    score = _linear_fp32(xf, comp.wgate)     # [R,1,dim]
    rows = torch.arange(R, device=x.device)
    for r, p in enumerate(pos_list):
        sc.kv_state[r, int(p) % ratio] = kv[r, 0]
        sc.score_state[r, int(p) % ratio] = score[r, 0]

    boundary = [(int(p) + 1) % ratio == 0 for p in pos_list]
    if not any(boundary):
        return None
    pooled = (sc.kv_state * sc.score_state.softmax(dim=1)).sum(dim=1, keepdim=True)
    pooled = rms_norm(comp.norm, pooled.to(dtype))
    return pooled


def _write_compressed(sc: LayerScratch, pos_list: Sequence[int], latent: torch.Tensor,
                      cfg: Dsv41Config, o) -> None:
    """Rope + fp4 quant + scatter of the pooled latent into this layer's compressed stream."""
    import torch as _t

    ratio = int(sc.ratio)
    R = len(pos_list)
    rope_pos = _t.as_tensor([max(0, int(p) + 1 - ratio) for p in pos_list],
                            dtype=_t.long, device=latent.device)
    cos = sc.cos.index_select(0, rope_pos)
    sin = sc.sin.index_select(0, rope_pos)
    # rotate/quantise the whole tensor (only boundary rows are written); in-place rotation must hit
    # the real tensor, not an advanced-indexing copy, or the compressed stream and the index keys
    # would be built from un-rotated values
    _rotate_rows(latent[..., -cfg.rope_head_dim :], cos, sin)
    o.fp4_act_quant(latent, 16, True, scale_dtype=_FP8)
    for r, p in enumerate(pos_list):
        if ratio == 1 or (int(p) + 1) % ratio == 0:
            sc.compress_kv[r, int(p) // ratio] = latent[r, 0]


def _indexer_decode(
    attn: Dsv41Attention,
    x: torch.Tensor,
    qr: torch.Tensor,
    latent: Optional[torch.Tensor],
    pos: torch.Tensor,
    pos_list: Sequence[int],
    sc: LayerScratch,
    shared: BatchedShared,
    cfg: Dsv41Config,
    o,
    offset: int,
) -> torch.Tensor:
    """``indexer_forward`` with per-request positions/compress lengths, top-k done per row."""
    ix: Dsv41Indexer = attn.indexer
    ratio = int(sc.ratio)
    R = int(x.size(0))
    rd = int(cfg.rope_head_dim)
    comp_len = [(int(p) + 1) // ratio for p in pos_list]
    max_len = max(comp_len) if comp_len else 0
    if max_len <= 0:
        idxs = torch.empty(R, 1, 0, dtype=torch.int32, device=x.device)
        shared.topk_idxs = idxs
        return idxs

    if ix.owns_k and latent is not None:
        boundary = [(int(p) + 1) % ratio == 0 for p in pos_list] if ratio > 1 else [True] * R
        if any(boundary):
            k = rms_norm(ix.k_norm, o.linear(latent, ix.wk.weight, ix.wk.scale))
            rope_pos = torch.as_tensor(
                [max(0, int(p) + 1 - ratio) for p in pos_list], dtype=torch.long, device=pos.device
            )
            _rotate_rows(k[..., -rd:], sc.cos.index_select(0, rope_pos),
                         sc.sin.index_select(0, rope_pos))
            o.fp4_act_quant(k, 32, True)
            for r, p in enumerate(pos_list):
                if boundary[r]:
                    sc.index_k[r, int(p) // ratio] = k[r, 0]
        shared.index_k = sc.index_k

    q = o.linear(qr, ix.wq_b.weight, ix.wq_b.scale).unflatten(
        -1, (cfg.index_n_heads, cfg.index_head_dim)
    )
    _rotate_rows(q[..., -rd:], sc.cos.index_select(0, pos), sc.sin.index_select(0, pos))
    o.fp4_act_quant(q, 32, True)

    index_k = shared.index_k[:, :max_len]
    weights = o.linear(x, ix.weights_proj.weight, ix.weights_proj.scale) * (
        cfg.index_head_dim**-0.5 * cfg.index_n_heads**-0.5
    )
    score = torch.einsum("bshd,btd->bsht", q, index_k)
    score = (score.relu() * weights.unsqueeze(-1)).sum(dim=2)   # [R,1,max_len]

    if cfg.is_candidate_source(sc.layer_id):
        width = max_len + (-max_len) % int(cfg.candidate_block_size)
        cand = torch.zeros(R, 1, width, dtype=torch.bool, device=x.device)
        for i in range(R):
            n = int(comp_len[i])
            if n <= 0:
                continue
            cand[i : i + 1, :, :n] = select_candidate_blocks(
                score[i : i + 1, :, :n], n, cfg.candidate_topk_blocks, cfg.candidate_block_size
            )
        shared.candidates = cand
    elif cfg.uses_candidates(sc.layer_id) and shared.candidates is not None:
        cm = shared.candidates[:, :, :max_len]
        score = score.masked_fill(~cm, -math.inf)

    K = min(int(cfg.index_topk), max_len)
    idxs = torch.full((R, 1, K), -1, dtype=torch.int32, device=x.device)
    for i in range(R):
        n = int(comp_len[i])
        if n <= 0:
            continue
        k = min(K, n)
        top = score[i : i + 1, :, :n].topk(k, dim=-1, sorted=False).indices.sort(dim=-1).values
        idxs[i : i + 1, :, :k] = top + offset
    shared.topk_idxs = idxs
    return idxs


def attention_decode(
    attn: Dsv41Attention,
    x: torch.Tensor,
    pos: torch.Tensor,
    pos_list: Sequence[int],
    sc: LayerScratch,
    shared: BatchedShared,
    cfg: Dsv41Config,
    o=None,
) -> torch.Tensor:
    """``attention_forward`` for one token per request (``x [R,1,hidden]``)."""
    o = o or ops()
    R = int(x.size(0))
    win = int(cfg.sliding_window)
    rd = int(cfg.rope_head_dim)
    cos = sc.cos.index_select(0, pos)
    sin = sc.sin.index_select(0, pos)

    qr = rms_norm(attn.q_norm, o.linear(x, attn.wq_a.weight, attn.wq_a.scale))
    q = o.linear(qr, attn.wq_b.weight, attn.wq_b.scale).unflatten(
        -1, (cfg.num_attention_heads, cfg.head_dim)
    )
    _rotate_rows(q[..., -rd:], cos, sin)

    kv = rms_norm(attn.kv_norm, o.linear(x, attn.wkv.weight, attn.wkv.scale))
    _rotate_rows(kv[..., -rd:], cos, sin)
    o.act_quant(kv, 32, "ue8m0", _E8M0, True)
    write_ring_batched(sc.window_kv, list(range(R)), pos_list, kv.squeeze(1), win)
    window_idxs = window_topk_idxs_batched(win, pos_list, device=x.device).unsqueeze(1)

    kv_parts: List[torch.Tensor] = [sc.window_kv]
    idx_parts: List[torch.Tensor] = [window_idxs]
    ratio = int(sc.ratio)
    if ratio > 0:
        comp_len = [(int(p) + 1) // ratio for p in pos_list]
        max_len = max(comp_len) if comp_len else 0
        latent = None
        compress_idxs = None
        if cfg.is_kv_source(sc.layer_id):
            # publish this segment's buffers even when the step produces no new key (ratio > 1 and the
            # position is not on a block boundary): consumers still read the existing stream
            shared.index_k = sc.index_k
            latent = _compressor_decode(attn.compressor, x, pos, pos_list, sc, cfg, o)
            # order matters: the serial path runs the indexer on the **raw** pooled latent and only
            # then rotates/quantises it for the compressed stream (layer.py:_compress_kv).  Doing the
            # write first shifts the fp8 grid of the key and changes one element per row.
            if max_len > 0 and cfg.is_index_source(sc.layer_id):
                compress_idxs = _indexer_decode(
                    attn, x, qr, latent, pos, pos_list, sc, shared, cfg, o, win
                )
            if latent is not None:
                _write_compressed(sc, pos_list, latent, cfg, o)
                sc.new_latent = latent
            shared.compress_kv = sc.compress_kv
        elif max_len > 0 and cfg.is_index_source(sc.layer_id):
            compress_idxs = _indexer_decode(
                attn, x, qr, latent, pos, pos_list, sc, shared, cfg, o, win
            )
        if max_len > 0:
            kv_parts.append(shared.compress_kv[:, :max_len])
            if compress_idxs is None:
                compress_idxs = shared.topk_idxs
            if compress_idxs is None:
                compress_idxs = torch.empty(R, 1, 0, dtype=torch.int32, device=x.device)
            idx_parts.append(compress_idxs)
        else:
            # nothing compressed yet: the index stream is empty this step
            if cfg.is_index_source(sc.layer_id):
                shared.topk_idxs = torch.empty(R, 1, 0, dtype=torch.int32, device=x.device)
            idx_parts.append(shared.topk_idxs)

    kv_all = torch.cat(kv_parts, dim=1)
    idx_all = torch.cat(idx_parts, dim=-1)
    out = o.sparse_attn(q, kv_all, attn.attn_sink, idx_all, cfg.head_dim**-0.5)
    _rotate_rows(out[..., -rd:], cos, sin, True)

    groups = int(cfg.o_groups)
    out = out.view(R, 1, groups, -1)
    wo_a = attn.wo_a.weight.view(groups, cfg.o_lora_rank, -1)
    out = torch.einsum("bsgd,grd->bsgr", out, wo_a)
    return o.linear(out.flatten(2), attn.wo_b.weight, attn.wo_b.scale)


def block_forward_decode(
    block,
    x: torch.Tensor,
    pos: torch.Tensor,
    pos_list: Sequence[int],
    pre_mix: torch.Tensor,
    sc: LayerScratch,
    shared: BatchedShared,
    cfg: Dsv41Config,
    moe_fn=None,
    o=None,
    timings: Optional[dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``block_forward`` with the batched decode attention (same op order as the serial block)."""
    from .block import hc_mixes, hc_post, hc_pre

    residual = x
    attn_pre, attn_post, attn_comb = hc_mixes(
        block, x, block.hc_attn_fn, block.hc_attn_scale, block.hc_attn_base
    )
    # the incoming pre_mix is the previous block's ffn_pre -- using attn_pre here silently drops the
    # cross-sublayer hand-off (the S3 parity tests pin that timing)
    x = hc_pre(x, pre_mix)
    x = rms_norm(block.attn_norm, x)
    if timings is not None:
        import time as _t

        _sync()
        _t0 = _t.perf_counter()
    x = attention_decode(block.attn, x, pos, pos_list, sc, shared, cfg, o)
    if timings is not None:
        _sync()
        timings["attn"] = timings.get("attn", 0.0) + (_t.perf_counter() - _t0)
    x = hc_post(x, residual, attn_post, attn_comb)

    residual = x
    _ffn_pre, ffn_post, ffn_comb = hc_mixes(
        block, x, block.hc_ffn_fn, block.hc_ffn_scale, block.hc_ffn_base
    )
    x = hc_pre(x, attn_pre)
    x = rms_norm(block.ffn_norm, x)
    from .block import moe_forward

    if timings is not None:
        _t0 = _t.perf_counter()
    x = (moe_fn or moe_forward)(block.ffn, x, None)
    if timings is not None:
        _sync()
        timings["moe"] = timings.get("moe", 0.0) + (_t.perf_counter() - _t0)
    x = hc_post(x, residual, ffn_post, ffn_comb)
    return x, _ffn_pre


def _decode_per_row(model, ids, pos, states, cfg, moe_fn, capture=None):
    """Quality/parity mode: per-request attention (exactly the serial kernel shape).

    Dense hc/norm and MoE may still be batched; with ``VLLM_DSV41_BATCHED_MOE=per_row`` the whole
    step uses the single-request math.  Slow, but it is the reference for the 100-prompt gate.
    """
    from .block import engram_forward, hc_mixes, hc_post, hc_pre, moe_forward
    from .layer import ops as _ops

    o = _ops()
    R = int(ids.shape[0])
    h = o.embedding(ids, model.embed.weight)
    h = h.unsqueeze(2).repeat(1, 1, cfg.hc_mult, 1)
    pre_mix = make_identity_pre_mix(h, cfg.hc_mult)

    engram_hashes = None
    if any(block.engram is not None for block in model.layers):
        rows = []
        for i, state in enumerate(states):
            eh = getattr(state, "engram_hash", None)
            if eh is None:
                rows = None
                break
            rows.append(eh.forward(ids[i : i + 1], int(pos[i].item())))
        if rows:
            engram_hashes = torch.cat(rows, dim=0)

    for layer_id, block in enumerate(model.layers):
        if block.engram is not None and engram_hashes is not None:
            h = engram_forward(
                block.engram, h, engram_hashes[:, :, block.engram.layer_hash_index, :], None
            )
        residual = h
        attn_pre, attn_post, attn_comb = hc_mixes(
            block, h, block.hc_attn_fn, block.hc_attn_scale, block.hc_attn_base
        )
        x = rms_norm(block.attn_norm, hc_pre(h, pre_mix))
        outs = [
            attention_forward(
                block.attn, x[r : r + 1], int(pos[r].item()),
                states[r].runtime(layer_id, cfg.n_layers), cfg,
            )
            for r in range(R)
        ]
        x = torch.cat(outs, dim=0)
        x = hc_post(x, residual, attn_post, attn_comb)

        residual = x
        ffn_pre, ffn_post, ffn_comb = hc_mixes(
            block, x, block.hc_ffn_fn, block.hc_ffn_scale, block.hc_ffn_base
        )
        x = rms_norm(block.ffn_norm, hc_pre(x, attn_pre))
        x = (moe_fn or moe_forward)(block.ffn, x, None)
        h = hc_post(x, residual, ffn_post, ffn_comb)
        pre_mix = ffn_pre

    h = hc_pre(h, pre_mix)
    logits = logits_from_hidden(model, h)
    if capture is not None:
        capture["h"] = h.detach()
    return logits, h.detach()


def transformer_forward_decode(
    model,
    input_ids: torch.Tensor,
    positions,
    states: Sequence[object],
    cfg: Optional[Dsv41Config] = None,
    moe_fn=None,
    capture: Optional[dict] = None,
    capture_layers: Optional[dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One batched decode step: ``input_ids`` has one token per request.

    Returns ``(logits [R, vocab], hidden [R, hidden])``.  The per-request state is updated in place
    (window ring, compressed stream, compressor tails) so the next step -- serial or batched -- sees
    exactly what a serial call would have produced.
    """
    from .block import engram_forward
    from .layer import ops as _ops

    o = _ops()
    cfg = cfg or states[0].cfg
    device = model.embed.weight.device
    ids = input_ids.reshape(-1, 1).to(device)
    R = int(ids.size(0))
    pos = torch.as_tensor([int(p) for p in positions], dtype=torch.long, device=device)

    import os as _os_par

    if _os_par.environ.get("VLLM_DSV41_BATCHED_ATTN", "") == "per_row":
        return _decode_per_row(model, ids, pos, states, cfg, moe_fn, capture)

    import os as _os_t
    import time as _time_t

    _timing = _os_t.environ.get("VLLM_DSV41_BATCHED_TIMING", "") not in ("", "0")
    _t_all = _time_t.perf_counter()
    if _timing:
        _sync()
        _t_build = _time_t.perf_counter()
    step = build_scratch(states, pos.tolist(), cfg, device)
    if _timing:
        _sync()
        _t_layers = _time_t.perf_counter()
    shared = step.shared
    timings = {"attn": 0.0, "moe": 0.0} if _timing else None

    engram_hashes = None
    if any(block.engram is not None for block in model.layers):
        rows = []
        for i, state in enumerate(states):
            eh = getattr(state, "engram_hash", None)
            if eh is None:
                rows = None
                break
            rows.append(eh.forward(ids[i : i + 1], int(pos[i].item())))
        if rows:
            engram_hashes = torch.cat(rows, dim=0)

    h = o.embedding(ids, model.embed.weight)
    h = h.unsqueeze(2).repeat(1, 1, cfg.hc_mult, 1)
    pre_mix = make_identity_pre_mix(h, cfg.hc_mult)

    for layer_id, block in enumerate(model.layers):
        if block.engram is not None and engram_hashes is not None:
            h = engram_forward(
                block.engram, h, engram_hashes[:, :, block.engram.layer_hash_index, :], None
            )
        h, pre_mix = block_forward_decode(
            block, h, pos, step.pos_list, pre_mix, step.layers[layer_id], shared, cfg,
            moe_fn=moe_fn, o=o, timings=timings,
        )
        if capture_layers is not None:
            capture_layers[layer_id] = h.detach().mean(dim=2)

    from .block import hc_pre

    h = hc_pre(h, pre_mix)
    logits = logits_from_hidden(model, h)
    if _timing:
        _sync()
        _t_logits = _time_t.perf_counter()
    sync_back(states, step, cfg)
    if _timing:
        _sync()
        _t_done = _time_t.perf_counter()
        global _TIMING_PRINTED
        if _TIMING_PRINTED < 3:
            _TIMING_PRINTED += 1
            print(
                "DSV41_BATCH_TIMING R=%d total=%.0fms build=%.0f layers=%.0f attn=%.0f moe=%.0f "
                "logits=%.0f sync=%.0f pos=%s"
                % (
                    R, (_t_done - _t_all) * 1e3, (_t_layers - _t_build) * 1e3,
                    (_t_logits - _t_layers) * 1e3, timings["attn"] * 1e3,
                    timings["moe"] * 1e3, (_t_logits - _t_layers) * 1e3 - timings["attn"] * 1e3
                    - timings["moe"] * 1e3,
                    (_t_done - _t_logits) * 1e3, list(step.pos_list),
                ),
                flush=True,
            )
    if capture is not None:
        capture["h"] = h.detach()
    return logits, h.detach()
