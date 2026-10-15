"""H26 preNZ align 直载内核(E70/D63):MoE 权重 GM→L1 免 nd2nz 直载变体。

原理:生产内核的权重加载 ``T.copy(GM→L1)`` 被 tilelang 代码gen硬编码为 nd2nz 转换
(~50GB/s/核,E36,自研内核 0.813-0.842x 的根因)。本模块用 raw ``call_extern`` 发射
CANN c_api 的免转换 ``asc_copy_gm2l1_align``,直接平载主机侧预打包的 compact-NZ 权重
(pack 契约见 :mod:`tileinfer.kernels.moe.prenz`;L1 物理序=compact-NZ 字节级等价)。

- 内核几何与生产 :func:`moe_gate_up_swiglu` / :func:`moe_down` 完全一致(固化 spec:
  gate_up BN=256/BK=128、down BN=64/BK=512),仅权重加载路径不同;
- 设备数值与微基准由 benchmarks/probes/h26_moe_prenz_align.py 背书(H26 链:oracle
  TOL 3e-2 相对 + 7 场景 vs aclnn 门槛 decode_b32/prefill_400 ≥1.0x);
- 发射级前置证明见 h26_l1_offset_probe.py(L1 非 0 偏移一致)与
  h26_gm_addr_probe.py(静态/动态 GM 寻址全折进指针)。

权重布局契约(H27/E76 实测修正):
- nd2nz 写入 L1 的真实物理序 = **转置连续序**(物理槽 i ← 语义 (c=i//K, k=i%K);由
  h27_l1_layout_probe.py GEMM 单 hot 反推 32768/32768 确证),E70 的 compact-NZ 静态
  反推被推翻;
- align 直载 = 恒等平载 ⇒ packer 输出 = 每 [BN,K]/[BN,I] slice ``transpose(-2,-1)``
  后平坦化(``pack_gate_up`` / ``pack_down``);
- ``pack_gate_up(w13)``: [E, 2I, K] → flat [E*2*NT*BN*K],strip=(e, h, nt),
  gate strip=(e*2*NT+nt),up strip 同 +NT;
- ``pack_down(w2)``: [E, H, I] → flat [E*NT*BN*I],strip=(e, nt)。
"""

import tilelang
import tilelang.ascend.language as T
from tilelang.language.tir.op import call_extern

NUM_CORES = 28
BURST_BYTES = 4096
GATE_UP_BLOCK_N = 256
DOWN_BLOCK_N = 64


def _slice_bursts(block_n: int, block_k: int) -> tuple:
    total = block_n * block_k * 2
    assert total % BURST_BYTES == 0
    return total // BURST_BYTES, BURST_BYTES


def moe_gate_up_swiglu_align(
    num_tiles_max: int,
    rows_slots: int,
    num_experts: int,
    inter: int,
    hidden: int,
    block_m: int = 32,
    block_n: int = GATE_UP_BLOCK_N,
    block_k: int = 128,
    dtype: str = "bfloat16",
    threads: int = 128,
):
    """gate_up+SwiGLU 融合内核的 align 直载变体。

    权重实参 = ``Wnz`` flat [E*2*NT*BN*K]:strip (e, h, nt) = 生产 ``W13[e, h*I+nt*BN
    : +BN, :]`` 的 [BN, K] compact-NZ 打包;gate strip = eid*2*NT+nt,up strip 同 +NT。
    """
    EM, R = num_tiles_max, rows_slots
    E_, K = num_experts, hidden
    BN, BK = block_n, block_k
    ROWS = block_m // 2
    NT = inter // BN
    total = EM * NT
    rounds = (total + NUM_CORES - 1) // NUM_CORES
    bursts, blen = _slice_bursts(BN, BK)
    strip_elems = BN * K
    up_off = NT * strip_elems

    @T.prim_func
    def main(
        X: T.Tensor((R, K), dtype),
        Wnz: T.Tensor((num_experts * 2 * NT * strip_elems,), dtype),
        ExpertIds: T.Tensor((EM,), "int32"),
        SortedIds: T.Tensor((R,), "int32"),
        Y: T.Tensor((R, inter), dtype),
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
                                base_g = (eid * 2 * NT + nt) * strip_elems
                                k0 = 0
                                T.copy(X[row0:row0 + block_m, k0:k0 + BK], a_l1)
                                T.evaluate(call_extern("void", "asc_set_gm2l1_loop_size", 1, 1))
                                T.evaluate(call_extern("void", "asc_set_gm2l1_loop1_stride", 0, 0))
                                T.evaluate(call_extern("void", "asc_set_gm2l1_loop2_stride", 0, 0))
                                T.evaluate(call_extern(
                                    "void", "asc_copy_gm2l1_align",
                                    T.access_ptr(wg_l1, "r"),
                                    T.access_ptr(Wnz[base_g + k0 * BN], "r"),
                                    bursts, blen, 0, 0, 0, 0, blen, blen))
                                T.evaluate(call_extern(
                                    "void", "asc_copy_gm2l1_align",
                                    T.access_ptr(wu_l1, "r"),
                                    T.access_ptr(Wnz[base_g + up_off + k0 * BN], "r"),
                                    bursts, blen, 0, 0, 0, 0, blen, blen))
                                T.copy(a_l1, ga)
                                T.copy(wg_l1, gb)
                                T.gemm(ga, gb, acc_g, transpose_B=True, clear_accum=True)
                                T.copy(wu_l1, gb)
                                T.gemm(ga, gb, acc_u, transpose_B=True, clear_accum=True)
                                for k in T.serial(K // BK - 1):
                                    k0 = (k + 1) * BK
                                    T.copy(X[row0:row0 + block_m, k0:k0 + BK], a_l1)
                                    T.evaluate(call_extern(
                                        "void", "asc_copy_gm2l1_align",
                                        T.access_ptr(wg_l1, "r"),
                                        T.access_ptr(Wnz[base_g + k0 * BN], "r"),
                                        bursts, blen, 0, 0, 0, 0, blen, blen))
                                    T.evaluate(call_extern(
                                        "void", "asc_copy_gm2l1_align",
                                        T.access_ptr(wu_l1, "r"),
                                        T.access_ptr(Wnz[base_g + up_off + k0 * BN], "r"),
                                        bursts, blen, 0, 0, 0, 0, blen, blen))
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
                                n0 = nt * BN
                                T.dual_copy(y_ub[0:ROWS, 0:BN], Y[row0:row0 + block_m, n0:n0 + BN])

    return main


def moe_down_align(
    num_tiles_max: int,
    rows_slots: int,
    num_experts: int,
    inter: int,
    hidden: int,
    block_m: int = 32,
    block_n: int = DOWN_BLOCK_N,
    block_k: int = 512,
    dtype: str = "bfloat16",
    threads: int = 128,
):
    """down-proj 内核的 align 直载变体,权重 = flat [E*NT*BN*I] compact-NZ。"""
    EM, R = num_tiles_max, rows_slots
    E_, I = num_experts, inter
    BN, BK = block_n, block_k
    ROWS = block_m // 2
    NT = hidden // BN
    total = EM * NT
    rounds = (total + NUM_CORES - 1) // NUM_CORES
    bursts, blen = _slice_bursts(BN, BK)
    strip_elems = BN * I

    @T.prim_func
    def main(
        Y: T.Tensor((R, I), dtype),
        Wnz: T.Tensor((num_experts * NT * strip_elems,), dtype),
        ExpertIds: T.Tensor((EM,), "int32"),
        SortedIds: T.Tensor((R,), "int32"),
        Out: T.Tensor((R, hidden), dtype),
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
                                base = (eid * NT + nt) * strip_elems
                                k0 = 0
                                T.copy(Y[row0:row0 + block_m, k0:k0 + BK], a_l1)
                                T.evaluate(call_extern("void", "asc_set_gm2l1_loop_size", 1, 1))
                                T.evaluate(call_extern("void", "asc_set_gm2l1_loop1_stride", 0, 0))
                                T.evaluate(call_extern("void", "asc_set_gm2l1_loop2_stride", 0, 0))
                                T.evaluate(call_extern(
                                    "void", "asc_copy_gm2l1_align",
                                    T.access_ptr(w_l1, "r"),
                                    T.access_ptr(Wnz[base + k0 * BN], "r"),
                                    bursts, blen, 0, 0, 0, 0, blen, blen))
                                T.copy(a_l1, ga)
                                T.copy(w_l1, gb)
                                T.gemm(ga, gb, acc, transpose_B=True, clear_accum=True)
                                for k in T.serial(I // BK - 1):
                                    k0 = (k + 1) * BK
                                    T.copy(Y[row0:row0 + block_m, k0:k0 + BK], a_l1)
                                    T.evaluate(call_extern(
                                        "void", "asc_copy_gm2l1_align",
                                        T.access_ptr(w_l1, "r"),
                                        T.access_ptr(Wnz[base + k0 * BN], "r"),
                                        bursts, blen, 0, 0, 0, 0, blen, blen))
                                    T.copy(a_l1, ga)
                                    T.copy(w_l1, gb)
                                    T.gemm(ga, gb, acc, transpose_B=True)
                                T.dual_copy(acc, o_ub)
                                with T.SimtVF(threads=threads):
                                    for r, c in T.Parallel(ROWS, BN):
                                        o_cast[r, c] = T.Cast(dtype, o_ub[r, c])
                                n0 = nt * BN
                                T.dual_copy(o_cast[0:ROWS, 0:BN], Out[row0:row0 + block_m, n0:n0 + BN])

    return main


def pack_gate_up(w13, inter: int):
    """[E, 2I, K] → flat [E*2*NT*BN*K](strip=(e,h,nt),h: 0=gate/1=up)。

    H27 修正(E76):nd2nz 实测 L1 物理序=转置连续(列 c 外轴、行 k 内轴,32768/32768),
    非 compact-NZ(E70 静态反推错误)。align 恒等平载 ⇒ GM flat 序 = 每 [BN,K] slice
    转置后平坦化。
    """
    E_, two_i, K = w13.shape
    assert two_i == 2 * inter
    BN = GATE_UP_BLOCK_N
    NT = inter // BN
    packed = w13.view(E_, 2, NT, BN, K).transpose(-2, -1).contiguous()
    return packed.reshape(-1).contiguous()


def pack_down(w2, inter: int):
    """[E, H, I] → flat [E*NT*BN*I](strip=(e,nt));H27 修正=转置连续序(同上)。"""
    E_, Hd, I_ = w2.shape
    assert I_ == inter
    BN = DOWN_BLOCK_N
    NT = Hd // BN
    packed = w2.view(E_, NT, BN, I_).transpose(-2, -1).contiguous()
    return packed.reshape(-1).contiguous()


_KERNEL_CACHE: dict = {}


def build_align_gate_up(**spec):
    key = ("h26_gu_align",) + tuple(sorted(spec.items()))
    if key not in _KERNEL_CACHE:
        _KERNEL_CACHE[key] = tilelang.compile(
            moe_gate_up_swiglu_align(**spec), out_idx=[],
            pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        )
    return _KERNEL_CACHE[key]


def build_align_down(**spec):
    key = ("h26_dn_align",) + tuple(sorted(spec.items()))
    if key not in _KERNEL_CACHE:
        _KERNEL_CACHE[key] = tilelang.compile(
            moe_down_align(**spec), out_idx=[],
            pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        )
    return _KERNEL_CACHE[key]
