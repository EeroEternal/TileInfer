"""GDN chunk 链 solve_tril + recompute_w_u 的链级 oracle(BACKLOG M-GDN 9c,H10)。

vllm-ascend 在役 chunk 链(E20/E22)中段两步:
    A  = chunk_scaled_dot_kkt_fwd(k, beta, g_cumsum)   # 严格下三角,BT=64,fp32
    Ai = solve_tril(A)                                  # (I + A)^-1,逐 64x64 块
    w  = Ai @ (k * beta * exp(g_cumsum))                # recompute_w_u
    u  = Ai @ (v * beta)

本文件按数学定义给出 fp32/fp64 参照(不依赖 triton),供设备测试对照。
语义要点(均与 vllm_ascend/ops/triton/fla 对齐):
* A 的非零元仅在每 64-token 块内严格下三角:A[i,j] = beta_i·exp(g_i-g_j)·(k_i·k_j),i>j(块内);
* k 按 head 折算:k head = i_h // (HV // HK)(GQA:HK=16, HV=32);
* 无效行(序列末尾不足 64 的补齐区)不产生输出;三角形保证有效输出只依赖有效行/列。
"""

from __future__ import annotations

import torch

__all__ = ["kkt_reference", "chunk_wu_reference"]


def kkt_reference(
    k: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_size: int = 64,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """kkt 参照:返回 [1, T, HV, chunk_size] fp64(默认)严格下三角 A。"""
    t_total = k.shape[1]
    hv = beta.shape[-1]
    a = torch.zeros(t_total, hv, chunk_size, dtype=dtype, device=k.device)
    hg = hv // k.shape[2]
    kf = k.to(dtype)[0, :, torch.arange(hv, device=k.device) // hg]  # [T, hv, d] 按值头展开
    betaf = beta.to(dtype)
    gf = g_cumsum.to(dtype)
    for n in range(len(cu_seqlens) - 1):
        bos, eos = int(cu_seqlens[n]), int(cu_seqlens[n + 1])
        for t0 in range(bos, eos, chunk_size):
            t1 = min(t0 + chunk_size, eos)
            idx = torch.arange(t0, t1, device=k.device)
            kkb = torch.einsum("ihd,jhd->hij", kf[idx], kf[idx])  # [h, l, l]:k_i·k_j
            gd = gf[0, idx].unsqueeze(1) - gf[0, idx].unsqueeze(0)  # [i, j, h]
            block = betaf[0, idx, :].unsqueeze(1) * torch.exp(gd) * kkb.permute(1, 2, 0)
            # block[i, j, h] = beta_i exp(g_i-g_j) k_i·k_j;只留严格下三角
            tri = torch.tril(torch.ones(t1 - t0, t1 - t0, dtype=dtype, device=k.device), -1)
            full = torch.zeros(t1 - t0, hv, chunk_size, dtype=dtype, device=k.device)
            full[:, :, : t1 - t0] = block.permute(0, 2, 1) * tri.unsqueeze(1)
            a[idx, :] = full
    return a.unsqueeze(0)


def chunk_wu_reference(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor,
    cu_seqlens: torch.Tensor,
    a_kkt: torch.Tensor | None = None,
    chunk_size: int = 64,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """solve_tril + recompute_w_u 的链级参照。

    输入:k [1,T,HK,D]、v [1,T,HV,D](bf16)、beta/g_cumsum [1,T,HV]、cu_seqlens;
    可选 a_kkt(外部提供 kkt 输出,否则按定义自算)。返回 (w, u, a_inv):
    w/u [1,T,HV,D] 与 v 同 dtype;a_inv [1,T,HV,chunk_size](fp32,调试用;
    无效块行补单位阵,与"未存储"语义一致)。
    """
    _, t_total, hv, dim = v.shape
    hk = k.shape[2]
    hg = hv // hk
    if a_kkt is None:
        a_kkt = kkt_reference(k, beta, g_cumsum, cu_seqlens, chunk_size, dtype)
    af = a_kkt.to(dtype)
    kf = k.to(dtype)
    vf = v.to(dtype)
    betaf = beta.to(dtype)
    gf = g_cumsum.to(dtype)
    w = torch.zeros(1, t_total, hv, dim, dtype=dtype, device=v.device)
    u = torch.zeros(1, t_total, hv, dim, dtype=dtype, device=v.device)
    a_inv = torch.zeros(t_total, hv, chunk_size, dtype=dtype, device=v.device)
    for n in range(len(cu_seqlens) - 1):
        bos, eos = int(cu_seqlens[n]), int(cu_seqlens[n + 1])
        for t0 in range(bos, eos, chunk_size):
            t1 = min(t0 + chunk_size, eos)
            idx = torch.arange(t0, t1, device=v.device)
            l = t1 - t0
            blk = af[0, idx][:, :, :l]  # [l, h, l](部分块切到有效列)
            eye = torch.eye(l, dtype=dtype, device=v.device)
            # (I + A)^-1 逐 head:块严格下三角,直接求逆
            inv = torch.linalg.inv(eye.unsqueeze(0) + blk.permute(1, 0, 2))  # [h, l, l]
            ai_blk = torch.zeros(l, hv, chunk_size, dtype=dtype, device=v.device)
            ai_blk[:, :, :l] = inv.permute(1, 0, 2)
            a_inv[idx] = ai_blk
            khead = (torch.arange(hv, device=v.device) // hg).unsqueeze(0)  # [1, h]
            kb = kf[0, idx.unsqueeze(1), khead] * betaf[0, idx].unsqueeze(-1) * torch.exp(
                gf[0, idx]
            ).unsqueeze(-1)
            vb = vf[0, idx] * betaf[0, idx].unsqueeze(-1)
            # w_idx = inv @ kb_idx(块内下三角:只需 j<=i,这里用全乘,块外列无效行恒 0)
            w[0, idx] = torch.einsum("hij,jhd->ihd", inv, kb)
            u[0, idx] = torch.einsum("hij,jhd->ihd", inv, vb)
    return w.to(v.dtype), u.to(v.dtype), a_inv.unsqueeze(0).to(torch.float32)
