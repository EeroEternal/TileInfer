"""H28 shared expert E+1 折叠设备 oracle(bf16/TOL 3e-2,与现役内核同口径)。

验证折叠数学端到端等价:分解路径(routed grouped E + shared 串行链×sigmoid 门控)
vs 折叠路径(routed grouped E+1,恒选末位列 + sigmoid 门控折进 topk 权重)。
GMM 走在役原语 ``torch_npu.npu_grouped_matmul``(与 moe_mlp.py unquant 链同参:
split_item=2/group_list_type=1/group_type=0,权重 [E,H,·] 转置布局)。
"""

from __future__ import annotations

import pytest
import torch

from tileinfer.integrations.shared_expert_fold import (
    append_shared_column,
    concat_shared_weights,
    extend_gate_weight,
    slice_router_logits,
)
from tileinfer.utils import is_npu_available

E = 256
H = 2048
I = 512
TOPK = 8
TOL = 3e-2


def _grouped_mlp(x_compact, counts, w13_ac, w2_ac):
    """在役 unquant 链:npu_grouped_matmul -> npu_swiglu -> npu_grouped_matmul。"""
    import torch_npu

    g = torch_npu.npu_grouped_matmul(
        x=[x_compact], weight=[w13_ac], split_item=2, group_list_type=1,
        group_type=0, group_list=counts,
    )[0]
    a = torch_npu.npu_swiglu(g)
    return torch_npu.npu_grouped_matmul(
        x=[a], weight=[w2_ac], split_item=2, group_list_type=1,
        group_type=0, group_list=counts,
    )[0]


def _aggregate(expert_out, topk_ids, topk_weights, m):
    """Σ_k w[t,k]·out[t,k](expanded 行序 = token 主序)。"""
    return (expert_out.view(m, TOPK + 1, H).to(torch.float32)
            * topk_weights.unsqueeze(-1)).sum(dim=1)


@pytest.mark.npu
@pytest.mark.skipif(not is_npu_available(), reason="no Ascend NPU available")
def test_fold_oracle_grouped_vs_decomposed():
    device = torch.device("npu", 0)
    g = torch.Generator(device="cpu").manual_seed(7)
    w13 = (torch.randn((E, 2 * I, H), generator=g) * 0.02).to(device).to(torch.bfloat16)
    w2 = (torch.randn((E, H, I), generator=g) * 0.02).to(device).to(torch.bfloat16)
    sh_w13 = (torch.randn((2 * I, H), generator=g) * 0.02).to(device).to(torch.bfloat16)
    sh_w2 = (torch.randn((H, I), generator=g) * 0.02).to(device).to(torch.bfloat16)
    gate_w = (torch.randn((E, H), generator=g) * 0.05).to(device).to(torch.float32)
    sh_gate_row = (torch.randn((H,), generator=g) * 0.05).to(device).to(torch.float32)

    for m in (1, 8, 32):
        x = torch.randn((m, H), generator=g).to(device).to(torch.bfloat16)
        topk_ids = torch.randint(0, E, (m, TOPK), generator=g).to(device)
        topk_w = torch.rand((m, TOPK), generator=g).to(device).to(torch.float32)
        # ------- 分解路径:routed grouped(E)+ shared 串行链 × sigmoid(gate_logit)
        gate_logit = torch.nn.functional.linear(x.float(), sh_gate_row.unsqueeze(0))
        flat = topk_ids.reshape(-1)
        counts = torch.bincount(flat, minlength=E).to(torch.int64)
        order = torch.argsort(flat, stable=True)
        x_exp = torch.cat(
            [
                x.unsqueeze(1).expand(-1, TOPK, -1).reshape(-1, H),
                x,  # shared 行置于末尾(shared 独立 GEMM,非 grouped)
            ]
        )
        x_routed = x.unsqueeze(1).expand(-1, TOPK, -1).reshape(-1, H).index_select(0, order).contiguous()
        routed = _grouped_mlp(
            x_routed, counts, w13.transpose(1, 2).contiguous(), w2.transpose(1, 2).contiguous()
        )
        routed_out = torch.zeros((m, H), device=device, dtype=torch.float32)
        w_sorted = topk_w.reshape(-1).index_select(0, order)
        routed_out.index_add_(0, (order // TOPK), routed.to(torch.float32) * w_sorted.unsqueeze(1))
        import torch_npu

        shared = torch.nn.functional.linear(
            torch_npu.npu_swiglu(torch.nn.functional.linear(x, sh_w13)), sh_w2
        )
        ref = routed_out + shared.float() * torch.sigmoid(gate_logit)

        # ------- 折叠路径:E+1 恒选列 + sigmoid 门控折进 topk 权重
        gate_w_ext = extend_gate_weight(gate_w, sh_gate_row)
        logits_ext = torch.nn.functional.linear(x.float(), gate_w_ext)
        logits_r, gate_logit_fold = slice_router_logits(logits_ext, E)
        assert logits_r.shape == (m, E)
        # 扩列 GEMM 的 N 维 tiling 与独立 GEMM 不同 → ULP 级差异(fp32),非逐位契约
        assert torch.allclose(
            logits_r, torch.nn.functional.linear(x.float(), gate_w), rtol=1e-4, atol=1e-5
        )
        assert torch.allclose(gate_logit_fold, gate_logit, rtol=1e-4, atol=1e-5)
        ids_f, w_f = append_shared_column(topk_ids, topk_w, E, gate_logit_fold, torch.float32)
        assert (ids_f[:, -1] == E).all()
        assert torch.allclose(w_f[:, -1], torch.sigmoid(gate_logit[:, 0]), atol=1e-6)
        w13_f, w2_f = concat_shared_weights(w13, w2, sh_w13.unsqueeze(0), sh_w2.unsqueeze(0))
        flat_f = ids_f.reshape(-1)
        counts_f = torch.bincount(flat_f, minlength=E + 1).to(torch.int64)
        order_f = torch.argsort(flat_f, stable=True)
        x_f = x.unsqueeze(1).expand(-1, TOPK + 1, -1).reshape(-1, H).index_select(0, order_f).contiguous()
        fused = _grouped_mlp(x_f, counts_f, w13_f.transpose(1, 2).contiguous(), w2_f.transpose(1, 2).contiguous())
        fold_out = torch.zeros((m, H), device=device, dtype=torch.float32)
        wf_sorted = w_f.reshape(-1).index_select(0, order_f)
        fold_out.index_add_(0, (order_f // (TOPK + 1)), fused.to(torch.float32) * wf_sorted.unsqueeze(1))

        diff = (fold_out - ref).abs().max().item()
        scale = ref.abs().max().item()
        assert diff / max(scale, 1e-6) <= TOL, f"m={m}: rel {diff}/{scale} > {TOL}"


@pytest.mark.npu
@pytest.mark.skipif(not is_npu_available(), reason="no Ascend NPU available")
def test_fold_gate_extension_dtype_and_shape():
    """gate 扩列:fp32 连续、行序保持、末行=shared gate 行。"""
    device = torch.device("npu", 0)
    g = torch.Generator(device="cpu").manual_seed(3)
    w = torch.randn((E, H), generator=g).to(device).to(torch.float32)
    row = torch.randn((H,), generator=g).to(device).to(torch.bfloat16)
    ext = extend_gate_weight(w, row)
    assert ext.shape == (E + 1, H) and ext.is_contiguous() and ext.dtype == torch.float32
    assert torch.equal(ext[:E], w)
    assert torch.equal(ext[E], row.float())
