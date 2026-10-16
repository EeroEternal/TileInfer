"""H34 GDN fused(qkvz+ba)权重 load 期行重排契约测试(tileinfer.kernels.gdn.reorder)。

范围(与 H32 契约同口径):
* CPU:重排置换的合法性与结构(组主序终序);终序切片 vs 上游
  ``fix_query_key_value_ordering``+rearrange+cat 的六路消费语义逐位相等
  (同一 GEMM 输出列置换,纯重排无算术);守卫矩阵继承;懒构建失败缓存独立
  命名(与 H32 属性互不可见)。
* 设备(npu 标记):重排权重 GEMM vs 列置换 GEMM 输出的 TOL 等价(3e-2,不同 N
  的 MatMulV3 tiling 累加序差,与 H32 同口径)。

引擎权威验证链:接线(opt-in token)留待 H32 A/B 收口,随后独立权威口径 A/B。
"""

from __future__ import annotations

import types

import pytest
import torch

from tileinfer.kernels.gdn.reorder import (
    build_reordered_inproj_weight,
    fused_forward_reordered,
    reorder_rows_plan,
    split_reordered,
)
from tileinfer.utils import is_npu_available

HK, HV, DK, DV = 16, 32, 128, 128  # Qwen3.6 GDN 在役形状(TP1)
HIDDEN = 2048
M = 16
N_QKVZ = HK * (DK + DK + (DV + DV) * HV // HK)  # 12288
N_BA = HV * 2  # 64


def _ordering_layer():
    try:
        from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
            QwenGatedDeltaNetAttention,
        )
    except Exception:  # pragma: no cover - vllm 不可导入时跳过
        pytest.skip("vllm qwen_gdn_linear_attn not importable")
    layer = object.__new__(QwenGatedDeltaNetAttention)
    layer.num_k_heads = HK
    layer.num_v_heads = HV
    layer.head_k_dim = DK
    layer.head_v_dim = DV
    layer.tp_size = 1
    return layer


def _interleaved_weights(seed: int):
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(N_QKVZ, HIDDEN, generator=g) * 0.02,
        torch.randn(N_BA, HIDDEN, generator=g) * 0.02,
    )


def test_plan_is_permutation():
    plan = reorder_rows_plan(HK, HV, DK, DV, 1)
    assert plan.numel() == N_QKVZ + N_BA
    assert torch.equal(plan.sort().values, torch.arange(plan.numel()))


def test_plan_structure_matches_consumption_order():
    """结构抽检:q/k/v/z 组主序、ba 组内 [b|a]、组间跨 4 行(vpg=2)。"""
    plan = reorder_rows_plan(HK, HV, DK, DV, 1).tolist()
    group = 2 * DK + 2 * (HV // HK) * DV  # 768
    vpg = HV // HK  # 2
    # q_all: plan[0:HK*DK] = 各组 q 段依次
    expect_q = [g * group + i for g in range(HK) for i in range(DK)]
    assert plan[: HK * DK] == expect_q
    # k_all 紧随
    expect_k = [g * group + DK + i for g in range(HK) for i in range(DK)]
    assert plan[HK * DK: 2 * HK * DK] == expect_k
    # v_all
    expect_v = [
        g * group + 2 * DK + i for g in range(HK) for i in range(vpg * DV)
    ]
    assert plan[2 * HK * DK: 2 * HK * DK + HV * DV] == expect_v
    # z_all
    expect_z = [
        g * group + 2 * DK + vpg * DV + i
        for g in range(HK)
        for i in range(vpg * DV)
    ]
    assert plan[2 * HK * DK + HV * DV: N_QKVZ] == expect_z
    # b_all / a_all:ba 区组步长 2*vpg=4;消费序=组主序(g 外层、j 内层)
    expect_b = [N_QKVZ + 4 * g + j for g in range(HK) for j in range(vpg)]
    expect_a = [N_QKVZ + 4 * g + vpg + j for g in range(HK) for j in range(vpg)]
    assert plan[N_QKVZ: N_QKVZ + HV] == expect_b
    assert plan[N_QKVZ + HV:] == expect_a


def test_reordered_slices_equal_reference_ordering():
    """同一 GEMM 输出按 plan 列置换后,终序切片与上游消费链逐位相等。"""
    layer = _ordering_layer()
    w_qkvz, w_ba = _interleaved_weights(13)
    base = torch.cat([w_qkvz, w_ba], dim=0)
    plan = reorder_rows_plan(HK, HV, DK, DV, 1)
    x = torch.randn(M, HIDDEN)
    y = x @ base.T  # [M, 12352]

    q, k, v, z_ref, b_ref, a_ref = layer.fix_query_key_value_ordering(
        y[:, :N_QKVZ], y[:, N_QKVZ:]
    )
    mixed_qkv_ref = torch.cat(
        [q.reshape(M, -1), k.reshape(M, -1), v.reshape(M, -1)], dim=-1
    )

    y2 = y[:, plan]
    mixed_qkv, z, b, a = split_reordered(y2, HK, HV, DK, DV, 1)
    assert torch.equal(mixed_qkv, mixed_qkv_ref)
    assert torch.equal(z, z_ref)
    assert torch.equal(b, b_ref)
    assert torch.equal(a, a_ref)


def test_split_reordered_are_zero_copy_views():
    fused = torch.randn(M, N_QKVZ + N_BA)
    mixed_qkv, z, b, a = split_reordered(fused, HK, HV, DK, DV, 1)
    assert mixed_qkv.data_ptr() == fused.data_ptr()
    # z 基址=列偏移 2*dk_total+dv_total 处;b 基址=2*dk_total+2*dv_total 处
    dk_total = HK * DK
    dv_total = HV * DV
    assert z.data_ptr() == fused[:, 2 * dk_total + dv_total:].data_ptr()
    assert z.shape == (M, HV, DV)
    assert b.shape == (M, HV) and a.shape == (M, HV)
    assert b.data_ptr() == fused[:, 2 * dk_total + 2 * dv_total:].data_ptr()


def test_build_reordered_weight_guards():
    w_qkvz, w_ba = _interleaved_weights(5)

    def make_layer(**over):
        layer = types.SimpleNamespace(
            gqa_interleaved_layout=True,
            tp_size=1,
            quant_config=None,
            num_k_heads=HK,
            num_v_heads=HV,
            head_k_dim=DK,
            head_v_dim=DV,
            in_proj_qkvz=types.SimpleNamespace(weight=w_qkvz),
            in_proj_ba=types.SimpleNamespace(weight=w_ba),
        )
        for k, v in over.items():
            setattr(layer, k, v)
        return layer

    plan = reorder_rows_plan(HK, HV, DK, DV, 1)
    reordered = build_reordered_inproj_weight(make_layer())
    assert reordered is not None
    assert reordered.shape == (N_QKVZ + N_BA, HIDDEN)
    assert torch.equal(reordered, torch.cat([w_qkvz, w_ba], 0)[plan])

    assert build_reordered_inproj_weight(make_layer(gqa_interleaved_layout=False)) is None
    assert build_reordered_inproj_weight(make_layer(tp_size=2)) is None
    assert build_reordered_inproj_weight(make_layer(quant_config=object())) is None
    assert build_reordered_inproj_weight(make_layer(in_proj_ba=None)) is None
    assert build_reordered_inproj_weight(make_layer(in_proj_ba=types.SimpleNamespace())) is None
    assert build_reordered_inproj_weight(make_layer(num_k_heads=15)) is None  # 整除断裂
    bad_rows = make_layer(in_proj_qkvz=types.SimpleNamespace(weight=torch.randn(100, HIDDEN)))
    assert build_reordered_inproj_weight(bad_rows) is None  # 行数与 plan 不符
    no_heads = types.SimpleNamespace(
        gqa_interleaved_layout=True,
        tp_size=1,
        quant_config=None,
        in_proj_qkvz=types.SimpleNamespace(weight=w_qkvz),
        in_proj_ba=types.SimpleNamespace(weight=w_ba),
    )
    assert build_reordered_inproj_weight(no_heads) is None  # 形状属性缺失


def test_failed_guard_cached_with_distinct_attrs():
    layer = types.SimpleNamespace(  # 守卫失败:非 interleaved
        gqa_interleaved_layout=False,
        tp_size=1,
        quant_config=None,
        in_proj_qkvz=types.SimpleNamespace(weight=torch.randn(N_QKVZ, HIDDEN)),
        in_proj_ba=types.SimpleNamespace(weight=torch.randn(N_BA, HIDDEN)),
    )
    x = torch.randn(M, HIDDEN)
    out = torch.empty_like(x)
    assert fused_forward_reordered(layer, x, out) is False
    assert getattr(layer, "_tileinfer_reordered_inproj_failed", False) is True
    assert fused_forward_reordered(layer, x, out) is False  # 第二次直接短路
    # 与 H32 路径属性互不可见
    assert getattr(layer, "_tileinfer_fused_inproj_failed", False) is False
    assert getattr(layer, "_tileinfer_reordered_inproj", None) is None


def test_plan_rejects_invalid_config():
    with pytest.raises(ValueError):
        reorder_rows_plan(HK, HV, DK, DV, 3)  # num_k_heads % tp != 0
    with pytest.raises(ValueError):
        reorder_rows_plan(HK, 31, DK, DV, 1)  # v heads 不整除


@pytest.mark.npu
@pytest.mark.skipif(not is_npu_available(), reason="no Ascend NPU available")
def test_reordered_gemm_tol_equal_on_device():
    """重排权重 GEMM vs 列置换 GEMM 输出:TOL 3e-2(不同 N tiling 累加序差)。"""
    device = torch.device("npu", 0)
    w_qkvz, w_ba = _interleaved_weights(17)
    base = torch.cat([w_qkvz, w_ba], dim=0).to(device=device, dtype=torch.bfloat16)
    plan = reorder_rows_plan(HK, HV, DK, DV, 1).to(device)
    x = (torch.randn(M, HIDDEN) * 0.5).to(device=device, dtype=torch.bfloat16)

    y_base = torch.nn.functional.linear(x, base).float()
    y_perm = y_base[:, plan]
    y_reord = torch.nn.functional.linear(x, base[plan].contiguous()).float()
    diff = (y_reord - y_perm).abs().max().item()
    scale = y_perm.abs().max().item()
    assert diff / max(scale, 1e-6) <= 3e-2, f"TOL violated: {diff}/{scale}"

    # 同一设备输出的终序切片逻辑与 CPU 逐位一致(纯 view)
    m1, z1, b1, a1 = split_reordered(y_reord, HK, HV, DK, DV, 1)
    m2, z2, b2, a2 = split_reordered(y_perm, HK, HV, DK, DV, 1)
    for u, v in ((m1, m2), (z1, z2), (b1, b2), (a1, a2)):
        assert torch.equal(u.float(), v.float())


def test_ops_all_contains_gdnreorder():
    """接线门:gdnreorder 在 _OPS_ALL 且每步读取语义不受影响(CPU 零设备)。"""
    from tileinfer.integrations.vllm_ascend import _OPS_ALL, ops_enabled

    assert "gdnreorder" in _OPS_ALL
    import os

    old = os.environ.get("TILEINFER_OPS")
    try:
        os.environ["TILEINFER_OPS"] = "gdnreorder"
        assert ops_enabled("gdnreorder") is True
        os.environ["TILEINFER_OPS"] = "decode,prefill,gdnproj"
        assert ops_enabled("gdnreorder") is False
        os.environ["TILEINFER_OPS"] = ""
        assert ops_enabled("gdnreorder") is True
    finally:
        if old is None:
            os.environ.pop("TILEINFER_OPS", None)
        else:
            os.environ["TILEINFER_OPS"] = old


def test_stats_counters_registered():
    """在役探针断言依赖:计数键预置于 _stats(零设备)。"""
    from tileinfer.integrations.vllm_ascend import TileInferDecodeAttention

    stats = TileInferDecodeAttention._stats
    assert "gdnreorder_calls" in stats
    assert "gdnreorder_fallbacks" in stats


def test_install_gdn_proj_reorder_wiring():
    """安装函数接线(类级包装,无设备依赖):覆写生效、幂等、标记属性独立命名。"""
    pytest.importorskip("vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn")
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention as cls,
    )

    from tileinfer.integrations.vllm_ascend import install_gdn_proj_reorder

    original = cls.forward_cuda
    assert install_gdn_proj_reorder() is True
    assert getattr(cls, "_tileinfer_gdnreorder_installed", False)
    assert cls.forward_cuda is not original
    assert install_gdn_proj_reorder() is True  # 幂等
    assert getattr(cls, "_tileinfer_gdnproj_installed", False) is not None
    # 独立失败缓存命名,与 H32 互不可见(契约:对象级属性缺省)
    layer = object.__new__(cls)
    assert not hasattr(layer, "_tileinfer_reordered_inproj")
    assert not hasattr(layer, "_tileinfer_reordered_inproj_failed")
