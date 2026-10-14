"""Fused MoE grouped-GEMM kernels (Ascend 950, TileLang)."""

from .grouped_gemm_ascend950 import (
    build_down_kernel,
    build_down_prenz_kernel,
    build_gate_up_kernel,
    build_gate_up_paired_kernel,
    moe_down,
    moe_down_prenz,
    moe_gate_up_swiglu,
    moe_gate_up_swiglu_paired,
    pair_gate_up_rows,
)
from .prenz import pack_nd_to_compact_nz, unpack_compact_nz_to_nd

__all__ = [
    "build_down_kernel",
    "build_down_prenz_kernel",
    "build_gate_up_kernel",
    "build_gate_up_paired_kernel",
    "moe_down",
    "moe_down_prenz",
    "moe_gate_up_swiglu",
    "moe_gate_up_swiglu_paired",
    "pair_gate_up_rows",
    "pack_nd_to_compact_nz",
    "unpack_compact_nz_to_nd",
]
