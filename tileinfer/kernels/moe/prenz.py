"""Host-side ND -> compact-NZ weight packer for the MoE pre-NZ kernels.

Context (D45): E36/P4 concluded the GM->L1 ``asc_copy_gm2l1_nd2nz`` conversion is the structural
MoE bottleneck and that "pre-NZ direct load is inexpressible in the kernel layer".  A read of the
tilelang compiler (``insert_nd2nz.cc``) plus an offline lower (``benchmarks/probes/moe_prenz_nd2nz.py``)
showed the mechanism *is* expressible: a UB buffer annotated with ``make_ascend_compact_nz_layout``
makes the UB->L1 copy emit a raw DMA instead of the nd2nz conversion.  The weight then has to enter
the kernel already packed in that physical order, which is what this module produces.

Physical order (from ``MakeAscendCompactNZLayout``): for a logical ``[rows, cols]`` bf16 tile with
``c0 = 256 / 16 = 16``,

    phys_flat = (col // c0) * (rows * c0) + row * c0 + (col % c0)

i.e. column-groups of ``c0`` are the outer axis, then the row, then the column inside the group --
the standard Ascend NZ 16xC0 fractal grid, flattened.

Correctness note: this order is derived from the layout the compiler emits, not measured on the
device.  The packer is round-trip tested on the CPU (``pack`` then ``unpack`` restores the input),
which proves the *bookkeeping* is self-consistent; the final authority is a device numerical test
against the torch oracle (``tests/test_tilelang_ascend950_moe.py``).  Do not treat the CPU test as
numerical proof.
"""

from __future__ import annotations

import torch

__all__ = ["compact_nz_c0", "pack_nd_to_compact_nz", "unpack_compact_nz_to_nd"]


def compact_nz_c0(bits: int) -> int:
    """C0 fractal width for a dtype of ``bits`` bits (Ascend rule: ``256 // bits``)."""
    if bits <= 0 or 256 % bits != 0:
        raise ValueError(f"unsupported element bits {bits} for the Ascend NZ layout")
    return 256 // bits


def pack_nd_to_compact_nz(tile: torch.Tensor) -> torch.Tensor:
    """Repack a contiguous ``[..., rows, cols]`` ND tensor into compact-NZ physical order.

    Returns a tensor of the *same shape and dtype* whose contiguous memory holds the NZ order, so
    it can be fed to a kernel argument declared ``[..., rows, cols]`` and loaded through a
    ``make_ascend_compact_nz_layout``-annotated UB buffer.  ``cols`` must be divisible by C0.
    """
    if tile.dim() < 2 or not tile.is_contiguous():
        raise ValueError("pack_nd_to_compact_nz expects a contiguous tensor of rank >= 2")
    c0 = compact_nz_c0(tile.element_size() * 8)
    rows, cols = tile.shape[-2], tile.shape[-1]
    if cols % c0 != 0:
        raise ValueError(f"cols={cols} must be divisible by C0={c0}")
    lead = tile.shape[:-2]
    flat = tile.reshape(-1, rows, cols)
    out = torch.empty_like(flat)
    # phys[..., cg*(rows*c0) + row*c0 + ci] = nd[..., row, cg*c0 + ci]
    src = flat.view(-1, rows, cols // c0, c0)             # [N, row, cg, ci]
    dst = out.view(-1, cols // c0, rows, c0)              # [N, cg, row, ci] == NZ physical order
    dst.copy_(src.permute(0, 2, 1, 3))
    return out.reshape(tile.shape)


def unpack_compact_nz_to_nd(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`pack_nd_to_compact_nz` (used by the CPU round-trip test)."""
    c0 = compact_nz_c0(packed.element_size() * 8)
    rows, cols = packed.shape[-2], packed.shape[-1]
    flat = packed.reshape(-1, rows, cols)
    src = flat.view(-1, cols // c0, rows, c0)
    out = torch.empty_like(flat).view(-1, rows, cols // c0, c0)
    out.copy_(src.permute(0, 2, 1, 3))
    return out.reshape(packed.shape)
