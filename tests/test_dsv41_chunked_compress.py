"""Chunked prefill of the compressor (ratio > 1), CPU-only.

A chunk that starts/ends on a compression boundary must produce exactly the same latents as one
full prefill.  This is the prerequisite for the block-level KV anchor (P4).
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from tileinfer.models.dsv41.layer import compressor_forward


class _Slot:
    def __init__(self, weight):
        self.weight = weight
        self.scale = None


class _Norm:
    def __init__(self, dim):
        self.weight = torch.ones(dim, dtype=torch.bfloat16)
        self.eps = 1e-6


def _comp(ratio: int, dim: int, hidden: int):
    torch.manual_seed(0)
    comp = SimpleNamespace(
        compress_ratio=ratio,
        norm=_Norm(dim),
        wkv=_Slot(torch.randn(dim, hidden, dtype=torch.bfloat16) * 0.1),
    )
    if ratio > 1:
        comp.wgate = _Slot(torch.randn(dim, hidden, dtype=torch.bfloat16) * 0.1)
    return comp


def _rt(ratio: int, dim: int, max_seq_len: int = 64):
    return SimpleNamespace(
        kv_state=torch.zeros(1, ratio, dim, dtype=torch.bfloat16),
        score_state=torch.zeros(1, ratio, dim, dtype=torch.bfloat16),
    )


def test_chunked_compressor_matches_full_prefill():
    ratio, dim, hidden = 2, 8, 4
    comp = _comp(ratio, dim, hidden)
    x = torch.randn(1, 8, hidden, dtype=torch.bfloat16)

    full = compressor_forward(comp, x, 0, _rt(ratio, dim))
    assert full.shape == (1, 8 // ratio, dim)

    rt = _rt(ratio, dim)
    a = compressor_forward(comp, x[:, :4], 0, rt)
    b = compressor_forward(comp, x[:, 4:], 4, rt)
    chunked = torch.cat([a, b], dim=1)
    assert torch.equal(full, chunked)


def test_chunked_compressor_ratio4():
    ratio, dim, hidden = 4, 8, 4
    comp = _comp(ratio, dim, hidden)
    x = torch.randn(1, 16, hidden, dtype=torch.bfloat16)

    full = compressor_forward(comp, x, 0, _rt(ratio, dim))
    rt = _rt(ratio, dim)
    parts = [compressor_forward(comp, x[:, i : i + 4], i, rt) for i in (0, 4, 8, 12)]
    assert torch.equal(full, torch.cat(parts, dim=1))
