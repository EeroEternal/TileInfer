"""Chunked indexer prefill (start_pos>0, seqlen>1) parity, CPU-only.

The indexer writes new compressed keys at absolute block positions and scores every query against
all historical compressed blocks.  Splitting a prefill into ratio-aligned chunks must reproduce the
full-prefill keys and top-k indices exactly.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from tileinfer.models.dsv41.layer import indexer_forward, precompute_rope_cos_sin


class _Slot:
    def __init__(self, weight):
        self.weight = weight
        self.scale = None


class _Norm:
    def __init__(self, dim):
        self.weight = torch.ones(dim, dtype=torch.bfloat16)
        self.eps = 1e-6


class _Cfg:
    compress_ratio_value = 2
    index_topk = 4
    index_n_heads = 2
    index_head_dim = 32
    rope_head_dim = 8
    head_dim = 32
    q_lora_rank = 32
    hidden_size = 32
    candidate_topk_blocks = 2
    candidate_block_size = 2

    def compress_ratio(self, layer_id):
        return self.compress_ratio_value

    def is_candidate_source(self, layer_id):
        return False

    def uses_candidates(self, layer_id):
        return False


def _ix(cfg):
    torch.manual_seed(1)
    return SimpleNamespace(
        owns_k=True,
        wq_b=_Slot(torch.randn(cfg.index_n_heads * cfg.index_head_dim, cfg.q_lora_rank, dtype=torch.bfloat16) * 0.1),
        weights_proj=_Slot(torch.randn(cfg.index_n_heads, cfg.hidden_size, dtype=torch.bfloat16) * 0.1),
        wk=_Slot(torch.randn(cfg.index_head_dim, cfg.head_dim, dtype=torch.bfloat16) * 0.1),
        k_norm=_Norm(cfg.index_head_dim),
    )


def _rt(cfg, seqlen=16):
    cos, sin = precompute_rope_cos_sin(cfg.rope_head_dim, seqlen, 0, 10000.0, 1.0, 32, 32)
    index_k = torch.zeros(1, seqlen // cfg.compress_ratio_value, cfg.index_head_dim, dtype=torch.bfloat16)
    return SimpleNamespace(
        layer_id=1,
        cos=cos,
        sin=sin,
        index_k=index_k,
        shared=SimpleNamespace(index_k=index_k, candidates=None, topk_idxs=None),
    )


def test_chunked_indexer_matches_full_prefill():
    cfg = _Cfg()
    ix = _ix(cfg)
    torch.manual_seed(2)
    x = torch.randn(1, 8, cfg.hidden_size, dtype=torch.bfloat16)
    qr = torch.randn(1, 8, cfg.q_lora_rank, dtype=torch.bfloat16)
    latent = torch.randn(1, 4, cfg.head_dim, dtype=torch.bfloat16)

    rt_full = _rt(cfg)
    full = indexer_forward(ix, x, qr, latent, 0, 0, rt_full, cfg)

    rt_chunk = _rt(cfg)
    a = indexer_forward(ix, x[:, :4], qr[:, :4], latent[:, :2], 0, 0, rt_chunk, cfg)
    b = indexer_forward(ix, x[:, 4:], qr[:, 4:], latent[:, 2:], 4, 0, rt_chunk, cfg)
    width = full.size(-1)
    if a.size(-1) < width:
        a = torch.nn.functional.pad(a, (0, width - a.size(-1)), value=-1)
    if b.size(-1) < width:
        b = torch.nn.functional.pad(b, (0, width - b.size(-1)), value=-1)
    chunked = torch.cat([a, b], dim=1)

    assert torch.equal(full, chunked), (full, chunked)
    assert torch.equal(rt_full.index_k, rt_chunk.index_k)


def test_chunked_indexer_ratio1_matches_full_prefill():
    cfg = _Cfg()
    cfg.compress_ratio_value = 1
    ix = _ix(cfg)
    torch.manual_seed(3)
    x = torch.randn(1, 8, cfg.hidden_size, dtype=torch.bfloat16)
    qr = torch.randn(1, 8, cfg.q_lora_rank, dtype=torch.bfloat16)
    latent = torch.randn(1, 8, cfg.head_dim, dtype=torch.bfloat16)

    rt_full = _rt(cfg)
    full = indexer_forward(ix, x, qr, latent, 0, 0, rt_full, cfg)

    rt_chunk = _rt(cfg)
    a = indexer_forward(ix, x[:, :4], qr[:, :4], latent[:, :4], 0, 0, rt_chunk, cfg)
    b = indexer_forward(ix, x[:, 4:], qr[:, 4:], latent[:, 4:], 4, 0, rt_chunk, cfg)
    assert torch.equal(full, torch.cat([a, b], dim=1))
    assert torch.equal(rt_full.index_k, rt_chunk.index_k)
