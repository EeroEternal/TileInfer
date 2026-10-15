"""H26 设备探针:MoE 两内核权重 GM→L1 免 nd2nz 直载(``asc_copy_gm2l1_align``)。

原理(E70/D63/H26):生产内核的权重加载 ``T.copy(GM→L1)`` 被 tilelang 代码gen硬编码为
nd2nz 转换(~50GB/s/核,E36,是 0.813x 的根因)。本探针用 raw ``call_extern`` 发射 CANN
c_api 的免转换 ``asc_copy_gm2l1_align``,直接平载主机侧预打包的 compact-NZ 权重
(pack 契约见 :mod:`tileinfer.kernels.moe.prenz`;L1 物理序=compact-NZ 字节级等价,
EmitNd2NzPostCopy_ 反推)。

发射级前置(均已离线证明,见 h26_l1_offset_probe.py / h26_gm_addr_probe.py):
- L1 多 buffer 时 ``T.access_ptr`` 正确发射 arena 偏移(big-first 实验,+69632 一致);
- GM 侧静态偏移/动态 strip/动态 K-slice 偏移均折进 emitted 指针表达式。

align 调用语义(c_api cube_datamove.h:1084):burst_count 次 burst_len 字节,
src_stride/dst_stride = burst_len → 平坦连续拷贝。K-slice [BN, BK] 在 compact-NZ strip
内 = 连续 BN*BK*2 字节(列组连续),起点元素偏移 = k0*BN。取 burst_len=4096B(已验证
量级,DeepGEMM fp8 同量级),BN=256/BK=128 与 BN=64/BK=512 的 slice 均 = 65536B = 16 burst。

用法(设备,NPU 空闲窗)::

    source .agent_loop/run/env.sh
    .venv/bin/python benchmarks/probes/h26_moe_prenz_align.py --numeric          # oracle 数值
    .venv/bin/python benchmarks/probes/h26_moe_prenz_align.py --bench            # 7 场景门槛
    .venv/bin/python benchmarks/probes/h26_moe_prenz_align.py --bench --gate     # 仅门槛双场景

门槛(H26 实验设计):decode_b32 与 prefill_400 两内核合计 vs aclnn 链 ratio ≥1.0
(E36b 基线 740.9µs vs 602.1µs / 1191.2µs vs 1002.5µs),数值 fp32 oracle TOL 3e-2(相对)。
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.ascend.language as T
from tilelang.language.tir.op import call_extern
from tileinfer.testing.moe_reference import build_slot_layout, gather_slots

E = 256
H = 2048
I = 512
TOPK = 8
BM = 32
NUM_CORES = 28
BURST_BYTES = 4096
TOL = 3e-2

_KERNEL_CACHE: dict = {}


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
    block_n: int = 256,
    block_k: int = 128,
    dtype: str = "bfloat16",
    threads: int = 128,
):
    """H26 变体:几何与生产 :func:`moe_gate_up_swiglu` 完全一致,仅权重加载换 align 直载。

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
    block_n: int = 64,
    block_k: int = 512,
    dtype: str = "bfloat16",
    threads: int = 128,
):
    """H26 变体 down:几何与生产 :func:`moe_down` 一致,权重 = flat [E*NT*BN*I] compact-NZ。"""
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


def pack_gate_up(w13: torch.Tensor) -> torch.Tensor:
    """[E, 2I, K] → flat [E*2*NT*BN*K],strip=(e,h,nt),h: 0=gate/1=up。

    H27 修正(E76):nd2nz 实测 L1 物理序=转置连续(列 c 外轴、行 k 内轴,32768/32768),
    非 compact-NZ(E70 静态反推错误)。align 恒等平载 ⇒ GM flat 序 = 每 [BN,K] slice
    转置后平坦化。
    """
    E_, two_i, K = w13.shape
    assert two_i == 2 * I
    BN = 256
    NT = I // BN
    packed = w13.view(E_, 2, NT, BN, K).transpose(-2, -1).contiguous()
    return packed.reshape(-1).contiguous()


def pack_down(w2: torch.Tensor) -> torch.Tensor:
    """[E, H, I] → flat [E*NT*BN*I],strip=(e,nt),BN=64;H27 修正=转置连续序(同上)。"""
    E_, Hd, I_ = w2.shape
    BN = 64
    NT = Hd // BN
    packed = w2.view(E_, NT, BN, I_).transpose(-2, -1).contiguous()
    return packed.reshape(-1).contiguous()


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


# ---------------------------------------------------------------- oracle 数值


def run_numeric(dev: str = "npu") -> bool:
    """H26 变体 vs 生产 ND 内核互检(真实 bench 形状)+ vs fp32 oracle(小形状)。"""
    from tileinfer.kernels.moe import build_down_kernel, build_gate_up_kernel

    g = torch.Generator(device="cpu").manual_seed(42)
    w13 = (torch.randn((E, 2 * I, H), dtype=torch.float32, generator=g) * 0.02).to(dev).to(torch.bfloat16)
    w2 = (torch.randn((E, H, I), dtype=torch.float32, generator=g) * 0.02).to(dev).to(torch.bfloat16)
    w13nz = pack_gate_up(w13)
    w2nz = pack_down(w2)
    ok = True

    for name, tokens, skewed in (("decode_b32", 32, False), ("prefill_400", 400, False)):
        if skewed:
            flat = torch.zeros((tokens * TOPK,), dtype=torch.int64)
            flat[: tokens * TOPK // 2] = 0
            flat[tokens * TOPK // 2:] = torch.randint(1, E, (tokens * TOPK // 2,), generator=g)
            topk_ids = flat.view(tokens, TOPK).to(dev)
        else:
            topk_ids = torch.randint(0, E, (tokens, TOPK), generator=g).to(dev)
        x = torch.randn((tokens, H), dtype=torch.float32, generator=g).to(dev).to(torch.bfloat16)

        expanded = x.unsqueeze(1).expand(-1, TOPK, -1).reshape(-1, H)
        lay = build_slot_layout(topk_ids, BM, num_experts=E)
        rows = lay["num_tiles"] * BM
        x_slots = gather_slots(expanded, lay["sorted_ids"], lay["num_expanded"]).contiguous()
        ids = lay["expert_ids"].contiguous()
        sids = lay["sorted_ids"].contiguous()
        y_nd = torch.empty((rows, I), dtype=torch.bfloat16, device=dev)
        out_nd = torch.empty((rows, H), dtype=torch.bfloat16, device=dev)
        y_al = torch.empty_like(y_nd)
        out_al = torch.empty_like(out_nd)

        k1 = build_gate_up_kernel(num_tiles_max=lay["num_tiles"], rows_slots=rows,
                                  num_experts=E, inter=I, hidden=H, block_m=BM, block_n=256)
        k2 = build_down_kernel(num_tiles_max=lay["num_tiles"], rows_slots=rows,
                               num_experts=E, inter=I, hidden=H, block_m=BM, block_n=64, block_k=512)
        a1 = build_align_gate_up(num_tiles_max=lay["num_tiles"], rows_slots=rows,
                                 num_experts=E, inter=I, hidden=H, block_m=BM, block_n=256)
        a2 = build_align_down(num_tiles_max=lay["num_tiles"], rows_slots=rows,
                              num_experts=E, inter=I, hidden=H, block_m=BM, block_n=64, block_k=512)
        torch.npu.synchronize()

        # fp32 oracle(生产内核与 H26 变体都应对齐)
        xf = x.float()
        ref_y = torch.zeros((rows, I), dtype=torch.float32, device=dev)
        ref_o = torch.zeros((rows, H), dtype=torch.float32, device=dev)
        for t in range(lay["num_tiles"]):
            r0, r1 = t * BM, (t + 1) * BM
            if int(lay["sorted_ids"][r0]) >= lay["num_expanded"]:
                continue
            e = int(lay["expert_ids"][t])
            xs = x_slots[r0:r1].float()
            gu = xs @ w13[e].float().T
            gt, up = gu[:, :I], gu[:, I:]
            ref_y[r0:r1] = gt * torch.sigmoid(gt) * up
            ref_o[r0:r1] = ref_y[r0:r1] @ w2[e].float().T

        k1(x_slots, w13, ids, sids, y_nd)
        k2(y_nd, w2, ids, sids, out_nd)
        a1(x_slots, w13nz, ids, sids, y_al)
        a2(y_al, w2nz, ids, sids, out_al)
        torch.npu.synchronize()

        scale_y = ref_y.abs().max().item()
        scale_o = ref_o.abs().max().item()
        nd_y = (y_nd.float() - ref_y).abs().max().item()
        nd_o = (out_nd.float() - ref_o).abs().max().item()
        al_y = (y_al.float() - ref_y).abs().max().item()
        al_o = (out_al.float() - ref_o).abs().max().item()
        rel = lambda d, s: d / max(s, 1e-9)  # noqa: E731
        print(f"[numeric {name}] nd rel_y={rel(nd_y, scale_y):.4g} rel_o={rel(nd_o, scale_o):.4g} | "
              f"h26 rel_y={rel(al_y, scale_y):.4g} rel_o={rel(al_o, scale_o):.4g}")
        if not (rel(al_y, scale_y) < TOL and rel(al_o, scale_o) < TOL):
            ok = False
    print("NUMERIC:", "OK" if ok else "FAIL")
    return ok


# ---------------------------------------------------------------- 微基准


def build_case(tokens: int, seed: int, skewed: bool, dev: str):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn((tokens, H), dtype=torch.float32, generator=g).to(dev).to(torch.bfloat16)
    if skewed:
        flat = torch.zeros((tokens * TOPK,), dtype=torch.int64)
        flat[: tokens * TOPK // 2] = 0
        flat[tokens * TOPK // 2:] = torch.randint(1, E, (tokens * TOPK // 2,), generator=g)
        topk_ids = flat.view(tokens, TOPK).to(dev)
    else:
        topk_ids = torch.randint(0, E, (tokens, TOPK), generator=g).to(dev)
    return x, topk_ids


def bench(fn, warmup=20, iters=100):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    stop = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    stop.record()
    torch.npu.synchronize()
    return start.elapsed_time(stop) / iters


_w = {}


def prepare_baseline(x, topk_ids, dev):
    import torch_npu  # noqa: F811

    flat = topk_ids.reshape(-1).to(torch.int64)
    counts = torch.bincount(flat, minlength=E)
    order = torch.argsort(flat, stable=True)
    expanded = x.unsqueeze(1).expand(-1, TOPK, -1).reshape(-1, H)
    x_compact = expanded.index_select(0, order).contiguous()
    counts_i64 = counts.to(torch.int64)

    def run():
        gu = torch_npu.npu_grouped_matmul(
            x=[x_compact], weight=[_w["w13_ac"]], split_item=2, group_list_type=1,
            group_type=0, group_list=counts_i64)[0]
        a = torch_npu.npu_swiglu(gu)
        return torch_npu.npu_grouped_matmul(
            x=[a], weight=[_w["w2_ac"]], split_item=2, group_list_type=1,
            group_type=0, group_list=counts_i64)[0]

    return run


def prepare_align(x, topk_ids, dev):
    expanded = x.unsqueeze(1).expand(-1, TOPK, -1).reshape(-1, H)
    lay = build_slot_layout(topk_ids, BM, num_experts=E)
    rows = lay["num_tiles"] * BM
    x_slots = gather_slots(expanded, lay["sorted_ids"], lay["num_expanded"]).contiguous()
    k1 = build_align_gate_up(num_tiles_max=lay["num_tiles"], rows_slots=rows,
                             num_experts=E, inter=I, hidden=H, block_m=BM, block_n=256)
    k2 = build_align_down(num_tiles_max=lay["num_tiles"], rows_slots=rows,
                          num_experts=E, inter=I, hidden=H, block_m=BM, block_n=64, block_k=512)
    y = torch.empty((rows, I), dtype=torch.bfloat16, device=dev)
    out = torch.empty((rows, H), dtype=torch.bfloat16, device=dev)
    ids = lay["expert_ids"].contiguous()
    sids = lay["sorted_ids"].contiguous()

    def run():
        k1(x_slots, _w["w13nz"], ids, sids, y)
        k2(y, _w["w2nz"], ids, sids, out)

    return run


def run_bench(out_path: str, gate_only: bool, iters: int = 100, dev: str = "npu") -> int:
    g = torch.Generator(device="cpu").manual_seed(42)
    _w["w13"] = (torch.randn((E, 2 * I, H), dtype=torch.float32, generator=g) * 0.02).to(dev).to(torch.bfloat16)
    _w["w2"] = (torch.randn((E, H, I), dtype=torch.float32, generator=g) * 0.02).to(dev).to(torch.bfloat16)
    _w["w13_ac"] = _w["w13"].transpose(1, 2).contiguous()
    _w["w2_ac"] = _w["w2"].transpose(1, 2).contiguous()
    _w["w13nz"] = pack_gate_up(_w["w13"])
    _w["w2nz"] = pack_down(_w["w2"])

    scenarios = [
        ("decode_b1", 1, False), ("decode_b4", 4, False), ("decode_b16", 16, False),
        ("decode_b32", 32, False), ("prefill_128", 128, False), ("prefill_400", 400, False),
        ("decode_b32_skew", 32, True),
    ]
    if gate_only:
        scenarios = [s for s in scenarios if s[0] in ("decode_b32", "prefill_400")]

    results = {}
    for name, tokens, skewed in scenarios:
        x, topk_ids = build_case(tokens, seed=hash(name) % (2 ** 31), skewed=skewed, dev=dev)
        base = prepare_baseline(x, topk_ids, dev)
        ours = prepare_align(x, topk_ids, dev)
        t0 = time.time()
        tb = bench(base, iters=iters)
        ta = bench(ours, iters=iters)
        results[name] = {
            "tokens": tokens,
            "baseline_ms": round(tb, 4),
            "h26_align_ms": round(ta, 4),
            "speedup": round(tb / ta, 3),
        }
        print(f"{name:16s} tokens={tokens:4d} aclnn={tb:8.4f}ms h26={ta:8.4f}ms "
              f"speedup={tb / ta:6.3f}x (prep {time.time() - t0:.0f}s)")

    gate = all(results[n]["speedup"] >= 1.0 for n in ("decode_b32", "prefill_400")) \
        if all(n in results for n in ("decode_b32", "prefill_400")) else False
    geo = 1.0
    for v in results.values():
        geo *= v["speedup"]
    geo **= 1.0 / len(results)
    summary = {"results": results, "geomean_speedup": round(geo, 4),
               "gate_decode_b32_prefill_400": "PASS" if gate else "FAIL"}
    Path(out_path).write_text(json.dumps(summary, indent=2))
    print(f"SUM geomean={geo:.3f} gate={'PASS' if gate else 'FAIL'} out={out_path}")
    return 0 if gate else 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--numeric", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--gate", action="store_true", help="仅门槛双场景")
    ap.add_argument("--out", default=".agent_loop/run/e71_h26_align_bench.json")
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()
    if not (args.numeric or args.bench):
        args.numeric = True
    rc = 0
    if args.numeric:
        if not run_numeric():
            rc = 1
            if not args.bench:
                return rc
    if args.bench:
        rc = max(rc, run_bench(args.out, gate_only=args.gate, iters=args.iters))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
