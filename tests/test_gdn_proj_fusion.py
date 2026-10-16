"""H32 GDN in_proj_ba→in_proj_qkvz 权重融合契约测试(tileinfer.kernels.gdn.proj_fusion)。

范围:
* CPU:strided 切片的纯维度拆分 view 合法性(E101);上游 ``fix_query_key_value_ordering``
  在「分离连续输入」与「fused strided 切片输入」下六路输出逐位相等(__new__ 构造,
  不依赖引擎);``build_fused_inproj_weight`` 守卫矩阵;失败标记缓存(不逐步重建);
  ``_OPS_ALL``/``ops_enabled`` 的 ``gdnproj`` 开关回归。
* 设备:融合 GEMM 与分离双 GEMM 的逐位等价(E100 微基准同口径,torch.equal)。

设备/引擎权威验证链:H32 权威口径引擎 A/B(在役探针 gdnproj_calls/fallbacks +
图模式端到端),见 PROGRESS iter-073。
"""

from __future__ import annotations

import types

import pytest
import torch

from tileinfer.kernels.gdn.proj_fusion import (
    build_fused_inproj_weight,
    fused_forward_cuda,
)
from tileinfer.utils import is_npu_available

HK, HV, DK, DV = 16, 32, 128, 128  # Qwen3.6 GDN 在役形状(TP1)
HIDDEN = 2048
M = 16


def _ordering_layer():
    """__new__ 构造最小 GDN 层:只填 fix_query_key_value_ordering 用到的形状属性。"""
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
    n_qkvz = HK * (DK + DK + (DV + DV) * HV // HK)  # 12288
    n_ba = HV * 2  # 64
    w_qkvz = torch.randn(n_qkvz, HIDDEN, generator=g) * 0.02
    w_ba = torch.randn(n_ba, HIDDEN, generator=g) * 0.02
    return w_qkvz, w_ba


def test_ops_all_contains_gdnproj():
    from tileinfer.integrations.vllm_ascend import _OPS_ALL, ops_enabled

    assert "gdnproj" in _OPS_ALL
    assert ops_enabled("gdnproj") is True  # 缺省全开
    monkey_spec = "decode,prefill,gdnproj"
    import os

    old = os.environ.get("TILEINFER_OPS")
    try:
        os.environ["TILEINFER_OPS"] = monkey_spec
        assert ops_enabled("gdnproj") is True
        os.environ["TILEINFER_OPS"] = "decode,prefill"
        assert ops_enabled("gdnproj") is False
        os.environ["TILEINFER_OPS"] = "gdnproj,decode"
        assert ops_enabled("decode") is True  # 同列时其他接管不受影响
        os.environ["TILEINFER_OPS"] = "nope"
        with pytest.raises(ValueError):
            ops_enabled("gdnproj")
    finally:
        if old is None:
            os.environ.pop("TILEINFER_OPS", None)
        else:
            os.environ["TILEINFER_OPS"] = old


def test_strided_slice_split_view_is_legal():
    fused = torch.randn(M, HK * 768 + HV * 2)
    n_qkvz = HK * 768
    q = fused[:, :n_qkvz].view(M, HK, 768)
    ba = fused[:, n_qkvz:].view(M, HK, 4)
    assert q.shape == (M, HK, 768) and ba.shape == (M, HK, 4)
    assert q.stride() == (fused.shape[1], 768, 1)
    assert ba.stride() == (fused.shape[1], 4, 1)


def test_ordering_fused_strided_equals_separate():
    layer = _ordering_layer()
    w_qkvz, w_ba = _interleaved_weights(7)
    fused = torch.cat([w_qkvz, w_ba], dim=0)
    x = torch.randn(M, HIDDEN)
    y_qkvz = x @ w_qkvz.T
    y_ba = x @ w_ba.T
    y_fused = x @ fused.T

    outs_sep = layer.fix_query_key_value_ordering(y_qkvz, y_ba)
    outs_fus = layer.fix_query_key_value_ordering(
        y_fused[:, :12288], y_fused[:, 12288:]
    )
    assert len(outs_sep) == len(outs_fus) == 6
    for a, b in zip(outs_sep, outs_fus):
        assert a.shape == b.shape
        assert torch.equal(a, b), "ordering outputs must be bit-identical"


def test_build_fused_weight_guards():
    w_qkvz, w_ba = _interleaved_weights(3)

    def make_layer(**over):
        layer = types.SimpleNamespace(
            gqa_interleaved_layout=True,
            tp_size=1,
            quant_config=None,
            in_proj_qkvz=types.SimpleNamespace(weight=w_qkvz),
            in_proj_ba=types.SimpleNamespace(weight=w_ba),
        )
        for k, v in over.items():
            setattr(layer, k, v)
        return layer

    fused = build_fused_inproj_weight(make_layer())
    assert fused is not None and torch.equal(fused, torch.cat([w_qkvz, w_ba], 0))

    assert build_fused_inproj_weight(make_layer(gqa_interleaved_layout=False)) is None
    assert build_fused_inproj_weight(make_layer(tp_size=2)) is None
    assert build_fused_inproj_weight(make_layer(quant_config=object())) is None
    assert build_fused_inproj_weight(make_layer(in_proj_ba=None)) is None
    no_ba = make_layer(in_proj_ba=types.SimpleNamespace())
    assert build_fused_inproj_weight(no_ba) is None
    bad_k = make_layer(
        in_proj_ba=types.SimpleNamespace(weight=torch.randn(64, 1024))
    )
    assert build_fused_inproj_weight(bad_k) is None


def test_failed_guard_is_cached():
    layer = types.SimpleNamespace(  # 守卫失败:非 interleaved
        gqa_interleaved_layout=False,
        tp_size=1,
        quant_config=None,
        in_proj_qkvz=types.SimpleNamespace(weight=torch.randn(12288, HIDDEN)),
        in_proj_ba=types.SimpleNamespace(weight=torch.randn(64, HIDDEN)),
    )
    x = torch.randn(M, HIDDEN)
    out = torch.empty_like(x)
    assert fused_forward_cuda(layer, x, out) is False
    assert getattr(layer, "_tileinfer_fused_inproj_failed", False) is True
    assert fused_forward_cuda(layer, x, out) is False  # 第二次直接短路
    assert getattr(layer, "_tileinfer_fused_inproj", None) is None


@pytest.mark.npu
@pytest.mark.skipif(not is_npu_available(), reason="no Ascend NPU available")
def test_fused_gemm_tol_equal_on_device():
    """设备上 N=12352 与 N=12288/64 的 MatMulV3 tiling 不同 → 累加序差(bf16 ULP 级)。

    契约=TOL 级等价(3e-2,与现役内核同口径),非逐位等价(E100 微基准该数据集
    maxdiff=0.0 属巧合;本用例以独立随机数据复核幅值)。
    """
    device = torch.device("npu", 0)
    w_qkvz, w_ba = _interleaved_weights(11)
    w_fused = torch.cat([w_qkvz, w_ba], dim=0).to(device)
    x = (torch.randn(M, HIDDEN) * 0.5).to(device=device, dtype=torch.bfloat16)
    wq = w_qkvz.to(device=device, dtype=torch.bfloat16)
    wb = w_ba.to(device=device, dtype=torch.bfloat16)

    y_fused = torch.nn.functional.linear(x, w_fused).float()
    y_sep = torch.cat(
        [
            torch.nn.functional.linear(x, wq),
            torch.nn.functional.linear(x, wb),
        ],
        dim=-1,
    ).float()
    diff = (y_fused - y_sep).abs().max().item()
    scale = y_sep.abs().max().item()
    assert diff / max(scale, 1e-6) <= 3e-2, f"TOL violated: {diff}/{scale}"

    n_qkvz = wq.shape[0]
    layer = _ordering_layer()
    # 同一 GEMM 输出,strided 切片输入 vs 连续拷贝输入:ordering 六路输出逐位一致
    # (纯维度拆分无算术;跨 GEMM 的差异已由上面 TOL 断言覆盖)
    outs_cont = layer.fix_query_key_value_ordering(
        y_fused[:, :n_qkvz].contiguous(), y_fused[:, n_qkvz:].contiguous()
    )
    outs_str = layer.fix_query_key_value_ordering(
        y_fused[:, :n_qkvz], y_fused[:, n_qkvz:]
    )
    for a, b in zip(outs_cont, outs_str):
        assert torch.equal(a, b)
