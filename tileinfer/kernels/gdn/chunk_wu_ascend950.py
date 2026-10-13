"""GDN chunk 链融合内核:solve_tril + recompute_w_u 一步完成(Ascend 950,TileLang)。

BACKLOG M-GDN 9c / H10(E22 立项取证):在役 chunk 链四核 triton 中,solve_tril
(0.40ms/层@T=128)与 recompute_w_u(0.33ms)为两大单核,且数据依赖相邻
(A_kkt → 块逆 → 乘 k/v)。本内核对每个 (64-token 块 × value head) 单元:

    M  = (I + A64)^-1         # 严格下三角块逆:4 个 16x16 Neumann 级数 + 3 级块合并
    w  = M @ (k·beta·exp(g))  # bf16 cube,fp32 累加
    u  = M @ (v·beta)

数值契约:基线 solve_tril 输出本就是 bf16(output_dtype=k.dtype),recompute 再转
fp32 点积;本内核 M 以 fp32 计算(reducer 求和)、喂 cube 前转 bf16 —— 同一舍入量级,
w/u 对 fp64 oracle TOL 3e-2(同现有内核)。三角形保证有效行输出只依赖有效行/列
(kkt 补齐列本就为 0),尾块无效行不影响有效输出;存储走 trash 行(w/u 多分配一行,
无效行统一写 0,与 recompute_w_u 的 mask-store 语义等价,消除跨序列行覆写竞争)。

结构(遵循 paged_decode_ascend950 house 模式):
* persistent 28 核,任务列表 = NT×HV(NT = varlen 64-token 块数,exact-shape 变体);
* 双 AIV 各持 32 行 band(rows_ub 由 dual_copy(arange) 隐式给基址,SPMD 无 AIV id);
* 两 AIV 冗余计算全 64x64 逆,用 rows_ub[0]//32 选择只落本 band 到 m_bf;
* 16x16 小乘 = shared(UB) 累加器 + serial 内积循环(probe4 验证模式;reducer 在 serial
  循环/多分配点场景会触发多 epoch 或空二进制,见 E23);
* gemm:w = M@kb、u = M@vb(kb/vb 由 dual_copy 按行带拼成全 64 行,transpose_B=False)。

本模块不得使用 ``from __future__ import annotations``(同 attention 内核约束)。
"""

import torch
import tilelang
import tilelang.ascend.language as T

__all__ = ["gdn_chunk_wu_ascend950", "build_gdn_wu_kernel", "gdn_chunk_wu", "NUM_CORES"]

NUM_CORES = 28
BT = 64  # fla chunk size(solve_tril BT 与 cumsum chunk_size 同值)
S = 16  # 子块边长(64 = 4x4 个 16x16 子块)
NEUMANN_STEPS = 15  # 16x16 严格下三角幂零:B^16 = 0,15 步精确


def gdn_chunk_wu_ascend950(
    nt_chunks: int,
    num_seqs: int,
    t_total: int,
    hv_heads: int,
    hk_heads: int,
    dim: int = 128,
    dtype: str = "bfloat16",
    threads: int = 128,
):
    """构建融合内核(每 (NT, N, T, HV, HK) 形状一个变体,exact-shape,manual-out)。"""
    HV = hv_heads
    HK = hk_heads
    HG = hv_heads // hk_heads  # k head 折算:h_v = i_h // HG
    D = dim
    total = nt_chunks * HV
    rounds = (total + NUM_CORES - 1) // NUM_CORES
    hb = BT // 2  # band 行数(双 AIV 各半)

    @T.prim_func
    def main(
        A: T.Tensor((t_total * HV, BT), "float32"),  # kkt 输出平坦 [T*HV, 64](块内严格下三角)
        K: T.Tensor((t_total * HK, D), dtype),  # 已 l2norm,平坦 [T*HK, D]
        V: T.Tensor((t_total * HV, D), dtype),
        Beta: T.Tensor((t_total * HV,), dtype),
        G: T.Tensor((t_total * HV,), "float32"),  # chunk_local_cumsum 输出,平坦 [T*HV]
        CuSeq: T.Tensor((num_seqs + 1,), "int64"),
        ChunkIdx: T.Tensor((nt_chunks, 2), "int64"),  # (i_n, i_t)
        Arange: T.Tensor((BT,), "int32"),  # 常量 arange(64),dual_copy 出 band 基址
        W: T.Tensor((t_total * HV + 1, D), dtype),  # 末行 trash 行(无效行统一写 0)
        U: T.Tensor((t_total * HV + 1, D), dtype),
    ):
        with T.Kernel(NUM_CORES) as bx:
            # ---- UB ----
            a_full = T.alloc_shared((BT, BT), "float32")
            xa_all = T.alloc_shared((4 * S, S), "float32")  # 4 个对角 16x16 逆(拼接)
            xb_all = T.alloc_shared((4 * S, S), "float32")
            blk_top = T.alloc_shared((hb, BT), "float32")
            blk_bot = T.alloc_shared((hb, BT), "float32")
            m_bf = T.alloc_shared((hb, BT), dtype)
            kb_ub = T.alloc_shared((hb, D), dtype)
            vb_ub = T.alloc_shared((hb, D), dtype)
            w_ub = T.alloc_shared((hb, D), "float32")
            u_ub = T.alloc_shared((hb, D), "float32")
            rows_ub = T.alloc_shared((hb,), "int32")  # 本 AIV 的 band 基址(R0+j)
            valid_ub = T.alloc_shared((hb,), "int32")
            row_ub = T.alloc_shared((hb,), "int32")
            # 合并积结果与临时
            x21 = T.alloc_shared((S, S), "float32")
            x31 = T.alloc_shared((S, S), "float32")
            x32 = T.alloc_shared((S, S), "float32")
            x41 = T.alloc_shared((S, S), "float32")
            x42 = T.alloc_shared((S, S), "float32")
            x43 = T.alloc_shared((S, S), "float32")
            p1 = T.alloc_shared((S, S), "float32")
            tmp = T.alloc_shared((4 * S, S), "float32")  # Neumann 步积
            # ---- L1 / L0 ----
            m_l1 = T.alloc_l1((BT, BT), dtype)
            kb_l1 = T.alloc_l1((BT, D), dtype)
            vb_l1 = T.alloc_l1((BT, D), dtype)
            ga = T.alloc_l0a((BT, BT), dtype)
            gb = T.alloc_l0b((BT, D), dtype)
            acc_w = T.alloc_l0c((BT, D), "float32")
            acc_u = T.alloc_l0c((BT, D), "float32")

            T.dual_copy(Arange[0:BT], rows_ub[0:hb])
            trash = t_total * HV

            for it in T.serial(rounds):
                idx = bx + it * NUM_CORES
                with T.If(idx < total):
                    with T.Then():
                        i_tc = idx // HV
                        i_h = idx % HV
                        i_n = T.Cast("int32", ChunkIdx[i_tc, 0])
                        i_t = T.Cast("int32", ChunkIdx[i_tc, 1])
                        bos = T.Cast("int32", CuSeq[i_n])
                        t0 = i_t * BT
                        t_seq = T.Cast("int32", CuSeq[i_n + 1]) - bos
                        khead = i_h // HG

                        # ---- 1. 装载全 64x64 kkt 块(SPMD 冗余)----
                        # 无效行(token >= t_seq)clamp 取址 + select 置 0
                        with T.SimtVF(threads=threads):
                            for r, c in T.Parallel(BT, BT):
                                tok = t0 + r
                                ok = tok < t_seq
                                row = (bos + T.min(tok, t_seq - 1)) * HV + i_h
                                a_full[r, c] = T.if_then_else(ok, A[row, c], T.float32(0))

                        # ---- 2. 四个对角 16x16 块逆:拼成 [64,16] 一次算 ----
                        # X ← I + X@B,B = -A 子块(严格下三角),15 步精确:
                        # (I-B)^-1 = Σ B^k,定点式 X = I + X·B;注意 X ← X + X·B 是
                        # 二项式 (I+B)^n,数值错误(E23 调试实证,禁用)。
                        # shared UB 累加器 + serial 内积(probe4 验证模式,E23)
                        with T.SimtVF(threads=threads):
                            for r, c in T.Parallel(4 * S, S):
                                xa_all[r, c] = T.if_then_else(r % S == c, T.float32(1), T.float32(0))
                            for st in T.serial(NEUMANN_STEPS):
                                for r, c in T.Parallel(4 * S, S):
                                    tmp[r, c] = T.float32(0)
                                for r, c in T.Parallel(4 * S, S):
                                    for kk in T.serial(S):
                                        tmp[r, c] = tmp[r, c] + xa_all[r, kk] * T.if_then_else(
                                            kk > c,
                                            -a_full[(r // S) * S + kk, (r // S) * S + c],
                                            T.float32(0),
                                        )
                                for r, c in T.Parallel(4 * S, S):
                                    xa_all[r, c] = T.if_then_else(
                                        r % S == c, T.float32(1), T.float32(0)
                                    ) + tmp[r, c]

                        # ---- 3. 块合并积(全部 fp32 16x16 小乘,shared UB 累加)----
                        # 前代换(i>j):X_ij = -M_ii·(Σ_{i<k<j} A_ik·X_kj + A_ij·M_jj)
                        # X21 = -M22·A21·M11;X31 = -M33(A31·M11 + A32·X21);
                        # X32 = -M33·A32·M22;X41 = -M44(A41·M11 + A42·X21 + A43·X31);
                        # X42 = -M44(A42·M22 + A43·X32);X43 = -M44·A43·M33
                        # 行块基址:M11=xa_all[0], M22=[S], M33=[2S], M44=[3S];
                        #          A21/A31/A32/A41/A42/A43 行基址 = 16/32/32/48/48/48,
                        #          A32/A42/A43 列基址 = 16/16/32
                        with T.SimtVF(threads=threads):
                            # X21 = -(M22·A21)·M11
                            for r, c in T.Parallel(S, S):
                                p1[r, c] = T.float32(0)
                            for r, c in T.Parallel(S, S):
                                for kk in T.serial(S):
                                    p1[r, c] = p1[r, c] + xa_all[S + r, kk] * a_full[S + kk, c]
                            for r, c in T.Parallel(S, S):
                                x21[r, c] = T.float32(0)
                            for r, c in T.Parallel(S, S):
                                for kk in T.serial(S):
                                    x21[r, c] = x21[r, c] - p1[r, kk] * xa_all[kk, c]
                            # X31 = -M33·(A31·M11 + A32·X21)(P3^-1 在左)
                            for r, c in T.Parallel(S, S):
                                p1[r, c] = T.float32(0)
                            for r, c in T.Parallel(S, S):
                                for kk in T.serial(S):
                                    p1[r, c] = (
                                        p1[r, c]
                                        + a_full[2 * S + r, kk] * xa_all[kk, c]
                                        + a_full[2 * S + r, S + kk] * x21[kk, c]
                                    )
                            for r, c in T.Parallel(S, S):
                                x31[r, c] = T.float32(0)
                            for r, c in T.Parallel(S, S):
                                for kk in T.serial(S):
                                    x31[r, c] = x31[r, c] - xa_all[2 * S + r, kk] * p1[kk, c]
                            # X32 = -M33·A32·M22(A32 = 行 32-47、列 16-31)
                            for r, c in T.Parallel(S, S):
                                p1[r, c] = T.float32(0)
                            for r, c in T.Parallel(S, S):
                                for kk in T.serial(S):
                                    p1[r, c] = p1[r, c] + a_full[2 * S + r, S + kk] * xa_all[S + kk, c]
                            for r, c in T.Parallel(S, S):
                                x32[r, c] = T.float32(0)
                            for r, c in T.Parallel(S, S):
                                for kk in T.serial(S):
                                    x32[r, c] = x32[r, c] - xa_all[2 * S + r, kk] * p1[kk, c]
                            # X41 = -M44·(A41·M11 + A42·X21 + A43·X31)
                            for r, c in T.Parallel(S, S):
                                p1[r, c] = T.float32(0)
                            for r, c in T.Parallel(S, S):
                                for kk in T.serial(S):
                                    p1[r, c] = (
                                        p1[r, c]
                                        + a_full[3 * S + r, kk] * xa_all[kk, c]
                                        + a_full[3 * S + r, S + kk] * x21[kk, c]
                                        + a_full[3 * S + r, 2 * S + kk] * x31[kk, c]
                                    )
                            for r, c in T.Parallel(S, S):
                                x41[r, c] = T.float32(0)
                            for r, c in T.Parallel(S, S):
                                for kk in T.serial(S):
                                    x41[r, c] = x41[r, c] - xa_all[3 * S + r, kk] * p1[kk, c]
                            # X42 = -M44·(A42·M22 + A43·X32)
                            for r, c in T.Parallel(S, S):
                                p1[r, c] = T.float32(0)
                            for r, c in T.Parallel(S, S):
                                for kk in T.serial(S):
                                    p1[r, c] = (
                                        p1[r, c]
                                        + a_full[3 * S + r, S + kk] * xa_all[S + kk, c]
                                        + a_full[3 * S + r, 2 * S + kk] * x32[kk, c]
                                    )
                            for r, c in T.Parallel(S, S):
                                x42[r, c] = T.float32(0)
                            for r, c in T.Parallel(S, S):
                                for kk in T.serial(S):
                                    x42[r, c] = x42[r, c] - xa_all[3 * S + r, kk] * p1[kk, c]
                            # X43 = -M44·A43·M33
                            for r, c in T.Parallel(S, S):
                                p1[r, c] = T.float32(0)
                            for r, c in T.Parallel(S, S):
                                for kk in T.serial(S):
                                    p1[r, c] = p1[r, c] + a_full[3 * S + r, 2 * S + kk] * xa_all[2 * S + kk, c]
                            for r, c in T.Parallel(S, S):
                                x43[r, c] = T.float32(0)
                            for r, c in T.Parallel(S, S):
                                for kk in T.serial(S):
                                    x43[r, c] = x43[r, c] - xa_all[3 * S + r, kk] * p1[kk, c]

                        # ---- 4. 组装 M 两个半带(blk_top = M 行 0..31,blk_bot = 行 32..63)----
                        # 注意:T.fill 不得出现在 If/Then 条件区内(SimtVF 外),否则 codegen
                        # 静默产出空二进制(E23 微二分实证);清零并入 SimtVF Parallel。
                        with T.SimtVF(threads=threads):
                            for j, c in T.Parallel(hb, BT):
                                blk_top[j, c] = T.float32(0)
                                blk_bot[j, c] = T.float32(0)
                            for r, c in T.Parallel(S, S):
                                blk_top[r, c] = xa_all[r, c]
                            for r, c in T.Parallel(S, S):
                                blk_top[S + r, c] = x21[r, c]
                                blk_top[S + r, S + c] = xa_all[S + r, c]
                            for r, c in T.Parallel(S, S):
                                blk_bot[r, c] = x31[r, c]
                                blk_bot[r, S + c] = x32[r, c]
                                blk_bot[r, 2 * S + c] = xa_all[2 * S + r, c]
                            for r, c in T.Parallel(S, S):
                                blk_bot[S + r, c] = x41[r, c]
                                blk_bot[S + r, S + c] = x42[r, c]
                                blk_bot[S + r, 2 * S + c] = x43[r, c]
                                blk_bot[S + r, 3 * S + c] = xa_all[3 * S + r, c]
                            # band 选择(每 AIV 只落本带)+ 行带元数据
                            sel = rows_ub[0] // hb
                            for j, c in T.Parallel(hb, BT):
                                m_bf[j, c] = T.Cast(
                                    dtype, T.if_then_else(sel == 0, blk_top[j, c], blk_bot[j, c])
                                )
                            for j in T.Parallel(hb):
                                tok = t0 + rows_ub[j]
                                valid_ub[j] = T.if_then_else(tok < t_seq, 1, 0)
                                row_ub[j] = (bos + T.min(tok, t_seq - 1)) * HV + i_h
                            # kb = k·beta·exp(g);vb = v·beta(band 装载,dual_copy 拼全 64 行)
                            for j, d in T.Parallel(hb, D):
                                bb = T.Cast("float32", Beta[row_ub[j]])
                                gg = G[row_ub[j]]
                                tk = T.Cast("float32", K[(row_ub[j] // HV) * HK + khead, d])
                                kb_ub[j, d] = T.Cast(
                                    dtype, T.if_then_else(valid_ub[j] == 1, tk * bb * T.exp(gg), T.float32(0))
                                )
                            for j, d in T.Parallel(hb, D):
                                vv = T.Cast("float32", V[row_ub[j], d])
                                bb = T.Cast("float32", Beta[row_ub[j]])
                                vb_ub[j, d] = T.Cast(
                                    dtype, T.if_then_else(valid_ub[j] == 1, vv * bb, T.float32(0))
                                )

                        # ---- 5. cube:w = M@kb,u = M@vb ----
                        T.dual_copy(m_bf[0:hb, 0:BT], m_l1[0:BT, 0:BT])
                        T.dual_copy(kb_ub[0:hb, 0:D], kb_l1[0:BT, 0:D])
                        T.dual_copy(vb_ub[0:hb, 0:D], vb_l1[0:BT, 0:D])
                        T.copy(m_l1[0:BT, 0:BT], ga)
                        T.copy(kb_l1[0:BT, 0:D], gb)
                        T.gemm(ga, gb, acc_w, transpose_B=False, clear_accum=True)
                        T.copy(vb_l1[0:BT, 0:D], gb)
                        T.gemm(ga, gb, acc_u, transpose_B=False, clear_accum=True)
                        T.dual_copy(acc_w, w_ub)
                        T.dual_copy(acc_u, u_ub)

                        # ---- 6. cast + 掩码存储(无效行 → trash 行写 0)----
                        with T.SimtVF(threads=threads):
                            for j, d in T.Parallel(hb, D):
                                wrow = T.if_then_else(valid_ub[j] == 1, row_ub[j], trash)
                                W[wrow, d] = T.if_then_else(
                                    valid_ub[j] == 1, T.Cast(dtype, w_ub[j, d]), T.Cast(dtype, 0)
                                )
                                U[wrow, d] = T.if_then_else(
                                    valid_ub[j] == 1, T.Cast(dtype, u_ub[j, d]), T.Cast(dtype, 0)
                                )

    return main


_KERNEL_CACHE: dict = {}


def build_gdn_wu_kernel(**spec):
    """编译(并缓存)exact-shape 融合内核。"""
    key = ("gdn_wu",) + tuple(sorted(spec.items()))
    kernel = _KERNEL_CACHE.get(key)
    if kernel is None:
        kernel = tilelang.compile(
            gdn_chunk_wu_ascend950(**spec),
            out_idx=[],
            pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        )
        _KERNEL_CACHE[key] = kernel
    return kernel


_ARANGE_BUF: torch.Tensor | None = None


def _arange64(device) -> torch.Tensor:
    global _ARANGE_BUF
    if _ARANGE_BUF is None or _ARANGE_BUF.device != device:
        _ARANGE_BUF = torch.arange(BT, dtype=torch.int32, device=device)
    return _ARANGE_BUF


def gdn_chunk_wu(
    a_kkt: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
):
    """融合 solve_tril + recompute_w_u:返回 (w, u),形状均 [1, T, HV, D] bf16。

    a_kkt: chunk_scaled_dot_kkt_fwd 的输出 [1, T, HV, 64] fp32;
    k/v: [1, T, HK/HV, D] bf16 连续(k 已 l2norm);g_cumsum: [1, T, HV] fp32;
    cu_seqlens: int64 [N+1];chunk_indices: prepare_chunk_indices(cu, 64) [NT, 2] int64。
    """
    _, t_total, hv, dim = v.shape
    hk = k.shape[2]
    ns = int(cu_seqlens.numel()) - 1
    nt = int(chunk_indices.shape[0])
    ker = build_gdn_wu_kernel(
        nt_chunks=nt,
        num_seqs=ns,
        t_total=t_total,
        hv_heads=hv,
        hk_heads=hk,
        dim=dim,
    )
    a3 = a_kkt.reshape(t_total * hv, BT)
    k3 = (k if k.is_contiguous() else k.contiguous()).reshape(t_total * hk, dim)
    v3 = (v if v.is_contiguous() else v.contiguous()).reshape(t_total * hv, dim)
    if beta.dtype != torch.bfloat16:
        b2 = beta.to(torch.bfloat16).contiguous().reshape(t_total * hv)
    else:
        b2 = (beta if beta.is_contiguous() else beta.contiguous()).reshape(t_total * hv)
    g2 = g_cumsum.contiguous().reshape(t_total * hv)
    w = torch.empty((t_total * hv + 1, dim), dtype=v.dtype, device=v.device)
    u = torch.empty((t_total * hv + 1, dim), dtype=v.dtype, device=v.device)
    ker(a3, k3, v3, b2, g2, cu_seqlens, chunk_indices, _arange64(v.device), w, u)
    return w[: t_total * hv].view(1, t_total, hv, dim), u[: t_total * hv].view(1, t_total, hv, dim)
