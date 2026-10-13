"""Fused MoE grouped-GEMM kernels (Ascend 950, TileLang)."""

from .grouped_gemm_ascend950 import (
    build_down_kernel,
    build_gate_up_kernel,
    build_gate_up_paired_kernel,
    moe_down,
    moe_gate_up_swiglu,
    moe_gate_up_swiglu_paired,
    pair_gate_up_rows,
)

__all__ = [
    "build_down_kernel",
    "build_gate_up_kernel",
    "build_gate_up_paired_kernel",
    "moe_down",
    "moe_gate_up_swiglu",
    "moe_gate_up_swiglu_paired",
    "pair_gate_up_rows",
]
