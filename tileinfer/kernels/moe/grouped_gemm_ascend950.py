"""Fused MoE grouped-GEMM kernels for Ascend 950 (TileLang ``ascend`` dialect).

Two kernels cover the routed-MLP body of a Qwen3.6 MoE layer (256 experts,
``moe_intermediate=512``, ``top_k=8``, single card — no cross-card dispatch):

* :func:`moe_gate_up_swiglu` — ``x_slots @ w13[e].T`` with **gate and up
  accumulated in the same block** (two L0C accumulators over two column halves
  of ``w13``) and a SwiGLU epilogue in the vector region, writing the
  ``[rows, inter]`` activation directly.  This removes the intermediate
  ``[rows, 2*inter]`` round-trip *and* the separate ``npu_swiglu`` launch.
* :func:`moe_down` — ``y_slots @ w2[e].T`` back to hidden.

Both kernels consume the *slot layout* (see :mod:`tileinfer.testing.moe_reference`):

``X``        ``[Rslots, K]``  input rows in slot order (pad slots never written back)
``W13``      ``[E, 2*inter, K]``
``ExpertIds`` ``[EM]`` int32  expert of each padded tile (pad tiles: 0)
``SortedIds`` ``[Rslots] int32`` expanded row per slot; sentinel ``num_expanded`` = pad
``Y``        ``[Rslots, inter]`` (kernel 1) / ``Out`` ``[Rslots, hidden]`` (kernel 2)

**Persistent scheduling**: both kernels launch exactly ``num_cores`` blocks
(the 950PR has 28 AI cores) and walk the ``(tile, n-tile)`` work list in
strided rounds.  This matters because a *dense* launch of thousands of tiny
blocks costs ~15 µs per 28-block wave in pure scheduling overhead (measured,
E19: an empty 8192-block kernel takes 4.3 ms), which swamps the arithmetic.
With routing spread over all 256 experts a decode step has ~2048 + 8192
(tile, n-tile) units of real work — as a dense grid that is 5.4 ms of overhead
alone; persistent the same work list is a strided loop with no per-wave launch
gap.

Empty work items (``SortedIds[row0]`` is the sentinel) skip before touching
any weight row — only actually-hit experts read their ~4 MB w13 slice.

Correctness contract: pad slots must not contribute to any valid output row.
The gemm may read pad rows of ``X`` (their product lands in accumulator rows
whose write-back is skipped), and every write-back is guarded per row.

Note: this module must **not** use ``from __future__ import annotations`` (same
``get_type_hints`` constraint as the attention kernels).
"""

import torch
import tilelang
import tilelang.ascend.language as T

# D45: pre-NZ weight path -- the UB staging buffer carries the compact-NZ layout so the UB->L1
# copy lowers to a raw DMA instead of the nd2nz conversion (see benchmarks/probes/moe_prenz_nd2nz.py).
from tilelang.layout import make_ascend_compact_nz_layout
from tilelang.language import annotate_layout

__all__ = [
    "moe_gate_up_swiglu",
    "moe_gate_up_swiglu_paired",
    "moe_down",
    "moe_down_prenz",
    "build_gate_up_kernel",
    "build_gate_up_paired_kernel",
    "build_down_kernel",
    "build_down_prenz_kernel",
    "pair_gate_up_rows",
]

NUM_CORES = 28


def pair_gate_up_rows(w13: torch.Tensor, block_pairs: int) -> torch.Tensor:
    """Reorder ``w13`` ``[E, 2*I, K]`` into paired-block row order ``[E, 2*I, K]``.

    Within each consecutive run of ``2*block_pairs`` rows, the first half is the
    gate rows of one N-block and the second half the matching up rows, so a
    single GEMM produces ``[gate | up]`` side by side and the SwiGLU epilogue
    pairs column ``c`` with column ``c + block_pairs`` with no second pass.
    """
    E, two_i, K = w13.shape
    assert two_i % (2 * block_pairs) == 0
    I = two_i // 2
    ntp = I // block_pairs
    g = w13[:, :I, :].view(E, ntp, block_pairs, K)
    u = w13[:, I:, :].view(E, ntp, block_pairs, K)
    return torch.stack([g, u], dim=2).reshape(E, 2 * I, K).contiguous()


def moe_gate_up_swiglu_paired(
    num_tiles_max: int,
    rows_slots: int,
    num_experts: int,
    inter: int,
    hidden: int,
    block_m: int = 32,
    block_pairs: int = 128,
    block_k: int = 128,
    dtype: str = "bfloat16",
    threads: int = 128,
):
    """Paired variant: one GEMM per (tile, n-block) emits ``[gate | up]`` columns.

    Consumes the :func:`pair_gate_up_rows` weight layout.  Saves one GEMM + one
    L0B copy + one accumulator per block versus the two-pass variant.
    """
    EM = num_tiles_max
    R = rows_slots
    E = num_experts
    I = inter
    K = hidden
    BP = block_pairs
    BK = block_k
    ROWS = block_m // 2
    NT = I // BP
    total = EM * NT
    rounds = (total + NUM_CORES - 1) // NUM_CORES
    assert I % BP == 0 and K % BK == 0 and block_m % 2 == 0

    @T.prim_func
    def main(
        X: T.Tensor((R, K), dtype),
        W13: T.Tensor((E, 2 * I, K), dtype),
        ExpertIds: T.Tensor((EM,), "int32"),
        SortedIds: T.Tensor((R,), "int32"),
        Y: T.Tensor((R, I), dtype),
    ):
        with T.Kernel(NUM_CORES) as bx:
            a_l1 = T.alloc_l1((block_m, BK), dtype)
            w_l1 = T.alloc_l1((2 * BP, BK), dtype)
            ga = T.alloc_l0a((block_m, BK), dtype)
            gb = T.alloc_l0b((2 * BP, BK), dtype)
            acc = T.alloc_l0c((block_m, 2 * BP), "float32")
            o_ub = T.alloc_shared((ROWS, 2 * BP), "float32")
            y_ub = T.alloc_shared((ROWS, BP), dtype)

            for it in T.serial(rounds):
                idx = bx + it * NUM_CORES
                with T.If(idx < total):
                    with T.Then():
                        tile = idx // NT
                        nt = idx % NT
                        eid = ExpertIds[tile]
                        row0 = tile * block_m
                        active = SortedIds[row0] < R
                        with T.If(active):
                            with T.Then():
                                r0 = nt * 2 * BP
                                k0 = 0
                                T.copy(X[row0:row0 + block_m, k0:k0 + BK], a_l1)
                                T.copy(W13[eid, r0:r0 + 2 * BP, k0:k0 + BK], w_l1)
                                T.copy(a_l1, ga)
                                T.copy(w_l1, gb)
                                T.gemm(ga, gb, acc, transpose_B=True, clear_accum=True)
                                for k in T.serial(K // BK - 1):
                                    k0 = (k + 1) * BK
                                    T.copy(X[row0:row0 + block_m, k0:k0 + BK], a_l1)
                                    T.copy(W13[eid, r0:r0 + 2 * BP, k0:k0 + BK], w_l1)
                                    T.copy(a_l1, ga)
                                    T.copy(w_l1, gb)
                                    T.gemm(ga, gb, acc, transpose_B=True)
                                T.dual_copy(acc, o_ub)
                                with T.SimtVF(threads=threads):
                                    for r, c in T.Parallel(ROWS, BP):
                                        g = o_ub[r, c]
                                        sig = T.float32(1) / (T.float32(1) + T.exp(-g))
                                        y_ub[r, c] = T.Cast(dtype, g * sig * o_ub[r, c + BP])
                                n0 = nt * BP
                                T.dual_copy(y_ub[0:ROWS, 0:BP], Y[row0:row0 + block_m, n0:n0 + BP])

    return main


def moe_gate_up_swiglu(
    num_tiles_max: int,
    rows_slots: int,
    num_experts: int,
    inter: int,
    hidden: int,
    block_m: int = 32,
    block_n: int = 128,
    block_k: int = 128,
    dtype: str = "bfloat16",
    threads: int = 128,
):
    """Build the fused gate+up GEMM + SwiGLU epilogue kernel (persistent)."""
    EM = num_tiles_max
    R = rows_slots
    E = num_experts
    I = inter
    K = hidden
    BN = block_n
    BK = block_k
    ROWS = block_m // 2  # per-AIV rows (dual_copy splits the M tile)
    NT = I // BN
    total = EM * NT
    rounds = (total + NUM_CORES - 1) // NUM_CORES
    assert I % BN == 0 and K % BK == 0 and block_m % 2 == 0

    @T.prim_func
    def main(
        X: T.Tensor((R, K), dtype),
        W13: T.Tensor((E, 2 * I, K), dtype),
        ExpertIds: T.Tensor((EM,), "int32"),
        SortedIds: T.Tensor((R,), "int32"),
        Y: T.Tensor((R, I), dtype),
    ):
        with T.Kernel(NUM_CORES) as bx:
            a_l1 = T.alloc_l1((block_m, BK), dtype)
            wg_l1 = T.alloc_l1((BN, BK), dtype)
            wu_l1 = T.alloc_l1((BN, BK), dtype)
            ga = T.alloc_l0a((block_m, BK), dtype)
            gb = T.alloc_l0b((BN, BK), dtype)
            acc_g = T.alloc_l0c((block_m, BN), "float32")
            acc_u = T.alloc_l0c((block_m, BN), "float32")
            g_ub = T.alloc_shared((ROWS, BN), "float32")
            u_ub = T.alloc_shared((ROWS, BN), "float32")
            y_ub = T.alloc_shared((ROWS, BN), dtype)

            for it in T.serial(rounds):
                idx = bx + it * NUM_CORES
                with T.If(idx < total):
                    with T.Then():
                        tile = idx // NT
                        nt = idx % NT
                        eid = ExpertIds[tile]
                        row0 = tile * block_m
                        active = SortedIds[row0] < R
                        with T.If(active):
                            with T.Then():
                                n0 = nt * BN
                                k0 = 0
                                T.copy(X[row0:row0 + block_m, k0:k0 + BK], a_l1)
                                T.copy(W13[eid, n0:n0 + BN, k0:k0 + BK], wg_l1)
                                T.copy(W13[eid, I + n0:I + n0 + BN, k0:k0 + BK], wu_l1)
                                T.copy(a_l1, ga)
                                T.copy(wg_l1, gb)
                                T.gemm(ga, gb, acc_g, transpose_B=True, clear_accum=True)
                                T.copy(wu_l1, gb)
                                T.gemm(ga, gb, acc_u, transpose_B=True, clear_accum=True)
                                for k in T.serial(K // BK - 1):
                                    k0 = (k + 1) * BK
                                    T.copy(X[row0:row0 + block_m, k0:k0 + BK], a_l1)
                                    T.copy(W13[eid, n0:n0 + BN, k0:k0 + BK], wg_l1)
                                    T.copy(W13[eid, I + n0:I + n0 + BN, k0:k0 + BK], wu_l1)
                                    T.copy(a_l1, ga)
                                    T.copy(wg_l1, gb)
                                    T.gemm(ga, gb, acc_g, transpose_B=True)
                                    T.copy(wu_l1, gb)
                                    T.gemm(ga, gb, acc_u, transpose_B=True)
                                T.dual_copy(acc_g, g_ub)
                                T.dual_copy(acc_u, u_ub)
                                with T.SimtVF(threads=threads):
                                    for r, c in T.Parallel(ROWS, BN):
                                        g = g_ub[r, c]
                                        sig = T.float32(1) / (T.float32(1) + T.exp(-g))
                                        y_ub[r, c] = T.Cast(dtype, g * sig * u_ub[r, c])
                                T.dual_copy(y_ub[0:ROWS, 0:BN], Y[row0:row0 + block_m, n0:n0 + BN])

    return main


def moe_down(
    num_tiles_max: int,
    rows_slots: int,
    num_experts: int,
    inter: int,
    hidden: int,
    block_m: int = 32,
    block_n: int = 128,
    block_k: int = 128,
    dtype: str = "bfloat16",
    threads: int = 128,
):
    """Build the down-projection grouped GEMM kernel (persistent)."""
    EM = num_tiles_max
    R = rows_slots
    E = num_experts
    I = inter
    H = hidden
    BN = block_n
    BK = block_k
    ROWS = block_m // 2
    NT = H // BN
    total = EM * NT
    rounds = (total + NUM_CORES - 1) // NUM_CORES
    assert H % BN == 0 and I % BK == 0 and block_m % 2 == 0

    @T.prim_func
    def main(
        Y: T.Tensor((R, I), dtype),
        W2: T.Tensor((E, H, I), dtype),
        ExpertIds: T.Tensor((EM,), "int32"),
        SortedIds: T.Tensor((R,), "int32"),
        Out: T.Tensor((R, H), dtype),
    ):
        with T.Kernel(NUM_CORES) as bx:
            a_l1 = T.alloc_l1((block_m, BK), dtype)
            w_l1 = T.alloc_l1((BN, BK), dtype)
            ga = T.alloc_l0a((block_m, BK), dtype)
            gb = T.alloc_l0b((BN, BK), dtype)
            acc = T.alloc_l0c((block_m, BN), "float32")
            o_ub = T.alloc_shared((ROWS, BN), "float32")
            o_cast = T.alloc_shared((ROWS, BN), dtype)

            for it in T.serial(rounds):
                idx = bx + it * NUM_CORES
                with T.If(idx < total):
                    with T.Then():
                        tile = idx // NT
                        nt = idx % NT
                        eid = ExpertIds[tile]
                        row0 = tile * block_m
                        active = SortedIds[row0] < R
                        with T.If(active):
                            with T.Then():
                                n0 = nt * BN
                                k0 = 0
                                T.copy(Y[row0:row0 + block_m, k0:k0 + BK], a_l1)
                                T.copy(W2[eid, n0:n0 + BN, k0:k0 + BK], w_l1)
                                T.copy(a_l1, ga)
                                T.copy(w_l1, gb)
                                T.gemm(ga, gb, acc, transpose_B=True, clear_accum=True)
                                for k in T.serial(I // BK - 1):
                                    k0 = (k + 1) * BK
                                    T.copy(Y[row0:row0 + block_m, k0:k0 + BK], a_l1)
                                    T.copy(W2[eid, n0:n0 + BN, k0:k0 + BK], w_l1)
                                    T.copy(a_l1, ga)
                                    T.copy(w_l1, gb)
                                    T.gemm(ga, gb, acc, transpose_B=True)
                                T.dual_copy(acc, o_ub)
                                with T.SimtVF(threads=threads):
                                    for r, c in T.Parallel(ROWS, BN):
                                        o_cast[r, c] = T.Cast(dtype, o_ub[r, c])
                                T.dual_copy(o_cast[0:ROWS, 0:BN], Out[row0:row0 + block_m, n0:n0 + BN])

    return main


_KERNEL_CACHE: dict = {}


def moe_down_prenz(
    num_tiles_max: int,
    rows_slots: int,
    num_experts: int,
    inter: int,
    hidden: int,
    block_m: int = 32,
    block_n: int = 64,
    block_k: int = 512,
    dtype: str = "bfloat16",
    threads: int = 128,
):
    """Down-projection grouped GEMM with a **pre-packed NZ** weight (D45, restart路线).

    Identical arithmetic to :func:`moe_down`, but the weight is read from a compact-NZ packed
    tensor ``W2nz`` ``[E * NT, block_n, block_k]`` (one NZ tile per (expert, n-tile); see
    :mod:`tileinfer.kernels.moe.prenz`) through a UB staging buffer annotated with
    ``make_ascend_compact_nz_layout``.  Per the offline probe (``benchmarks/probes/moe_prenz_nd2nz.py``)
    this makes the weight path lower to ``asc_copy_gm2ub_align`` + ``asc_copy_ub2l1`` instead of the
    ``asc_copy_gm2l1_nd2nz`` conversion E36 identified as the structural bottleneck.

    A single K iteration is required (``block_k == inter``) so the packed tile covers the whole
    reduction axis; the caller packs one ``[block_n, inter]`` NZ tile per (expert, n-tile).
    """
    EM = num_tiles_max
    R = rows_slots
    E = num_experts
    I = inter
    H = hidden
    BN = block_n
    BK = block_k
    ROWS = block_m // 2
    NT = H // BN
    total = EM * NT
    rounds = (total + NUM_CORES - 1) // NUM_CORES
    assert H % BN == 0 and block_k == I and block_m % 2 == 0, (
        "pre-NZ down needs block_k == inter (single K iteration) so one packed NZ tile covers K"
    )
    # The weight is packed per (expert, n-tile) as a [BN, I] NZ tile, so the leading axis is E*NT.
    WT = E * NT

    @T.prim_func
    def main(
        Y: T.Tensor((R, I), dtype),
        W2nz: T.Tensor((WT, BN, BK), dtype),
        ExpertIds: T.Tensor((EM,), "int32"),
        SortedIds: T.Tensor((R,), "int32"),
        Out: T.Tensor((R, H), dtype),
    ):
        with T.Kernel(NUM_CORES) as bx:
            w_ub = T.alloc_shared((BN + 1, BK), dtype)  # +1 pad row per the compact-NZ contract
            a_l1 = T.alloc_l1((block_m, BK), dtype)
            w_l1 = T.alloc_l1((BN, BK), dtype)
            ga = T.alloc_l0a((block_m, BK), dtype)
            gb = T.alloc_l0b((BN, BK), dtype)
            acc = T.alloc_l0c((block_m, BN), "float32")
            o_ub = T.alloc_shared((ROWS, BN), "float32")
            o_cast = T.alloc_shared((ROWS, BN), dtype)
            annotate_layout({w_ub: make_ascend_compact_nz_layout(w_ub)})

            for it in T.serial(rounds):
                idx = bx + it * NUM_CORES
                with T.If(idx < total):
                    with T.Then():
                        tile = idx // NT
                        nt = idx % NT
                        eid = ExpertIds[tile]
                        row0 = tile * block_m
                        active = SortedIds[row0] < R
                        with T.If(active):
                            with T.Then():
                                n0 = nt * BN
                                T.copy(Y[row0:row0 + block_m, 0:BK], a_l1)
                                T.copy(W2nz[eid * NT + nt, 0:BN, 0:BK], w_ub[0:BN, 0:BK])
                                T.copy(w_ub[0:BN, 0:BK], w_l1)
                                T.copy(a_l1, ga)
                                T.copy(w_l1, gb)
                                T.gemm(ga, gb, acc, transpose_B=True, clear_accum=True)
                                T.dual_copy(acc, o_ub)
                                with T.SimtVF(threads=threads):
                                    for r, c in T.Parallel(ROWS, BN):
                                        o_cast[r, c] = T.Cast(dtype, o_ub[r, c])
                                T.dual_copy(o_cast[0:ROWS, 0:BN], Out[row0:row0 + block_m, n0:n0 + BN])

    return main


def build_down_prenz_kernel(**spec):
    """Compile (and cache) the pre-NZ down kernel (manual output)."""
    key = ("down_prenz",) + tuple(sorted(spec.items()))
    kernel = _KERNEL_CACHE.get(key)
    if kernel is None:
        kernel = tilelang.compile(
            moe_down_prenz(**spec),
            out_idx=[],
            pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        )
        _KERNEL_CACHE[key] = kernel
    return kernel


def build_gate_up_kernel(**spec):
    """Compile (and cache) the fused gate+up+SwiGLU kernel (manual output)."""
    key = ("gate_up",) + tuple(sorted(spec.items()))
    kernel = _KERNEL_CACHE.get(key)
    if kernel is None:
        kernel = tilelang.compile(
            moe_gate_up_swiglu(**spec),
            out_idx=[],
            pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        )
        _KERNEL_CACHE[key] = kernel
    return kernel


def build_gate_up_paired_kernel(**spec):
    """Compile (and cache) the paired-layout fused kernel (manual output)."""
    key = ("gate_up_paired",) + tuple(sorted(spec.items()))
    kernel = _KERNEL_CACHE.get(key)
    if kernel is None:
        kernel = tilelang.compile(
            moe_gate_up_swiglu_paired(**spec),
            out_idx=[],
            pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        )
        _KERNEL_CACHE[key] = kernel
    return kernel


def build_down_kernel(**spec):
    """Compile (and cache) the down-projection kernel (manual output)."""
    key = ("down",) + tuple(sorted(spec.items()))
    kernel = _KERNEL_CACHE.get(key)
    if kernel is None:
        kernel = tilelang.compile(
            moe_down(**spec),
            out_idx=[],
            pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        )
        _KERNEL_CACHE[key] = kernel
    return kernel
