"""CPU round-trip test for the MoE pre-NZ weight packer (no device needed).

Proves the packer's bookkeeping is self-consistent (``pack`` then ``unpack`` restores the input).
This is *not* numerical proof of the device layout -- the authority for that is the device case in
``tests/test_tilelang_ascend950_moe.py``; see :mod:`tileinfer.kernels.moe.prenz`.
"""

from __future__ import annotations

import pytest
import torch

from tileinfer.kernels.moe.prenz import (
    compact_nz_c0,
    pack_nd_to_compact_nz,
    unpack_compact_nz_to_nd,
)


def test_c0_rule():
    assert compact_nz_c0(16) == 16   # bf16 / fp16
    assert compact_nz_c0(32) == 8    # fp32
    with pytest.raises(ValueError):
        compact_nz_c0(24)


@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((64, 512), torch.bfloat16),      # down slice [BN=64, BK=512]
        ((128, 128), torch.bfloat16),
        ((256, 2048, 512), torch.bfloat16),  # a full down weight [E, H, I]
        ((4, 32, 32), torch.float32),
    ],
)
def test_roundtrip(shape, dtype):
    x = torch.randn(shape, dtype=dtype)
    packed = pack_nd_to_compact_nz(x)
    assert packed.shape == x.shape
    restored = unpack_compact_nz_to_nd(packed)
    assert torch.equal(restored, x), "pack/unpack round-trip must be exact"


def test_physical_order_matches_layout_formula():
    """Explicitly check phys_flat = (col//c0)*(rows*c0) + row*c0 + (col%c0)."""
    rows, cols = 4, 32
    c0 = compact_nz_c0(16)
    x = torch.arange(rows * cols, dtype=torch.int64).reshape(rows, cols).to(torch.bfloat16)
    packed = pack_nd_to_compact_nz(x).reshape(-1)
    for r in range(rows):
        for c in range(cols):
            phys = (c // c0) * (rows * c0) + r * c0 + (c % c0)
            assert packed[phys].item() == x[r, c].item(), (r, c, phys)
