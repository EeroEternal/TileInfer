"""GDN chunk 链 oracle(chunk_wu_reference)的 CPU 数值自检。

M-GDN 9c / H10:设备测试以本 oracle 为参照,这里先验证 oracle 自身的数学正确性:
1. kkt 参照与 (I+A)·M = I(块逆定义);
2. w/u 与逐块显式解法一致(独立实现交叉验证);
3. 尾块/多序列的行归属与零填充语义。
"""

from __future__ import annotations

import pytest
import torch

from tileinfer.testing.gdn_reference import chunk_wu_reference, kkt_reference


def _make_inputs(t_total, hk=4, hv=8, d=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(1, t_total, hk, d, generator=g).to(torch.bfloat16)
    k = torch.nn.functional.normalize(k.float(), dim=-1).to(torch.bfloat16)
    v = torch.randn(1, t_total, hv, d, generator=g).to(torch.bfloat16)
    beta = torch.rand(1, t_total, hv, generator=g).to(torch.bfloat16)
    g_cum = -torch.rand(1, t_total, hv, generator=g).cumsum(1).float() * 0.05
    return k, v, beta, g_cum


@pytest.mark.parametrize("lens", [[128], [100], [64], [1], [100, 100, 65], [70, 3]])
def test_kkt_inverse_identity(lens):
    """(I + A_block) @ M_block = I 对每个 (块, head) 成立。"""
    cu = torch.tensor([0] + torch.tensor(lens).cumsum(0).tolist(), dtype=torch.int64)
    t_total = int(cu[-1])
    k, v, beta, g_cum = _make_inputs(t_total)
    a = kkt_reference(k, beta, g_cum, cu)
    _, _, a_inv = chunk_wu_reference(k, v, beta, g_cum, cu, a_kkt=a)
    # 逐块验证
    for n in range(len(cu) - 1):
        bos, eos = int(cu[n]), int(cu[n + 1])
        for t0 in range(bos, eos, 64):
            t1 = min(t0 + 64, eos)
            eye = torch.eye(t1 - t0, dtype=torch.float64)
            blk = a[0, t0:t1, :, : t1 - t0].double().permute(1, 0, 2)  # [h, l, l]
            m = a_inv[0, t0:t1, :, : t1 - t0].double().permute(1, 0, 2)
            prod = torch.einsum("hij,hjk->hik", eye + blk, m)
            torch.testing.assert_close(prod, eye.expand_as(prod), rtol=1e-6, atol=1e-6)


def test_wu_matches_explicit_per_block_solve():
    """w/u 与逐块 torch.linalg.solve 显式解一致(独立交叉验证)。"""
    lens = [100, 70]
    cu = torch.tensor([0] + torch.tensor(lens).cumsum(0).tolist(), dtype=torch.int64)
    t_total = int(cu[-1])
    k, v, beta, g_cum = _make_inputs(t_total, hk=2, hv=4, d=8, seed=3)
    w, u, _ = chunk_wu_reference(k, v, beta, g_cum, cu)
    # 显式解:逐块 solve_triangular(M, I+A)^T …直接解 (I+A) X = kb
    kf, vf, bf, gf = (x.double() for x in (k, v, beta, g_cum))
    hv, hg = 4, 2
    for n in range(len(cu) - 1):
        bos, eos = int(cu[n]), int(cu[n + 1])
        for t0 in range(bos, eos, 64):
            t1 = min(t0 + 64, eos)
            idx = torch.arange(t0, t1)
            a_blk = kkt_reference(k, beta, g_cum, cu)[0, idx, :, : t1 - t0].double().permute(1, 0, 2)
            eye = torch.eye(t1 - t0, dtype=torch.float64).unsqueeze(0)
            khead = (torch.arange(hv) // hg).unsqueeze(0)
            kb = kf[0, idx.unsqueeze(1), khead] * bf[0, idx].unsqueeze(-1) * torch.exp(gf[0, idx]).unsqueeze(-1)
            vb = vf[0, idx] * bf[0, idx].unsqueeze(-1)
            w_exp = torch.linalg.solve(eye + a_blk, kb.permute(1, 0, 2)).permute(1, 0, 2)
            u_exp = torch.linalg.solve(eye + a_blk, vb.permute(1, 0, 2)).permute(1, 0, 2)
            # 两侧都过 bf16 舍入(oracle 返回契约是 bf16)
            torch.testing.assert_close(w[0, idx].double(), w_exp.to(torch.bfloat16).double(), rtol=0, atol=0)
            torch.testing.assert_close(u[0, idx].double(), u_exp.to(torch.bfloat16).double(), rtol=0, atol=0)


def test_wu_dtype_and_shape_contract():
    k, v, beta, g_cum = _make_inputs(96, seed=7)
    cu = torch.tensor([0, 96], dtype=torch.int64)
    w, u, a_inv = chunk_wu_reference(k, v, beta, g_cum, cu)
    assert w.dtype == torch.bfloat16 and w.shape == (1, 96, 8, 16)
    assert u.dtype == torch.bfloat16 and u.shape == (1, 96, 8, 16)
    assert a_inv.dtype == torch.float32 and a_inv.shape == (1, 96, 8, 64)
