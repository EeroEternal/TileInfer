"""H35 GDN 小核链削减契约测试(tileinfer.kernels.gdn.small_ops)。

范围(CPU,不依赖引擎):
* rearrange cat-free 替身与上游 cat 版逐位相等;c=1 时 q/k/v 为缓冲视图
  (data_ptr 落在原缓冲内,零拷贝);
* l2norm 配对融合:rearrange 垫片 + l2norm(q)+l2norm(k) 调用序下,融合结果与
  两次独立调用逐位相等;不相关张量/ops 关闭时回退独立调用;
* conv1d 包装:非连续权重视图第二次调用被替换为缓存连续张量(值逐位一致),
  连续权重/ops 关闭时直通;
* ``_OPS_ALL``/``ops_enabled`` 的 ``gdnsmall`` 开关回归。

设备/引擎权威验证链:H35 权威口径引擎 A/B(在役探针 gdnsmall_* 计数 +
图模式端到端),见 HYPOTHESES H35。
"""

from __future__ import annotations

import types

import pytest
import torch

from tileinfer.kernels.gdn import small_ops
from tileinfer.utils import is_npu_available

HK, HV, DK, DV = 16, 32, 128, 128  # Qwen3.6 GDN 在役形状(TP1)
Q_DIM = HK * DK  # 2048
K_DIM = HK * DK  # 2048
V_DIM = HV * DV  # 4096


def _ref_rearrange(self, mixed_qkv):
    """上游 vllm 0.23.0 qwen_gdn_linear_attn.py:808 的 cat 版参照。"""
    if mixed_qkv is None:
        return None, None, None
    seq_len = mixed_qkv.shape[0]
    q_dim = self.key_dim // self.tp_size
    k_dim = self.key_dim // self.tp_size
    v_dim = self.value_dim // self.tp_size
    query, key, value = torch.split(mixed_qkv, [q_dim, k_dim, v_dim], dim=-1)
    fused = torch.cat([query.reshape(-1), key.reshape(-1), value.reshape(-1)], dim=0)
    q_size = seq_len * q_dim
    k_size = seq_len * k_dim
    q = fused[0:q_size].view(1, seq_len, -1, self.head_k_dim)
    k = fused[q_size:q_size + k_size].view(1, seq_len, -1, self.head_k_dim)
    v = fused[q_size + k_size:].view(1, seq_len, -1, self.head_v_dim)
    return q, k, v


def _ref_l2norm(x, *a, **k):
    """fla l2norm_fwd 的 CPU 参照(逐末维行归一)。"""
    dtype = k.get("output_dtype", None) or x.dtype
    eps = a[0] if a else k.get("eps", 1e-6)
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return xf.to(dtype)


def _layer():
    """__new__ 构造最小 GDN 层:只填 rearrange/l2norm 用到的形状属性。"""
    try:
        from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
            QwenGatedDeltaNetAttention,
        )
    except Exception:  # pragma: no cover
        pytest.skip("vllm qwen_gdn_linear_attn not importable")
    layer = object.__new__(QwenGatedDeltaNetAttention)
    layer.num_k_heads = HK
    layer.num_v_heads = HV
    layer.head_k_dim = DK
    layer.head_v_dim = DV
    layer.key_dim = HK * DK
    layer.value_dim = HV * DV
    layer.tp_size = 1
    layer._tileinfer_rearrange_orig = _ref_rearrange
    return layer


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    small_ops.reset_state()
    small_ops.reset_counters()
    yield
    small_ops.reset_state()


def test_ops_all_contains_gdnsmall(monkeypatch):
    from tileinfer.integrations.vllm_ascend import _OPS_ALL, ops_enabled

    assert "gdnsmall" in _OPS_ALL
    monkeypatch.setenv("TILEINFER_OPS", "gdnsmall")
    assert ops_enabled("gdnsmall")
    monkeypatch.setenv("TILEINFER_OPS", "decode,prefill")
    assert not ops_enabled("gdnsmall")


def test_rearrange_matches_reference_and_zero_copy_c1(monkeypatch):
    monkeypatch.setenv("TILEINFER_OPS", "gdnsmall")
    layer = _layer()
    g = torch.Generator().manual_seed(7)
    mixed = torch.randn(1, Q_DIM + K_DIM + V_DIM, generator=g)

    qr, kr, vr = _ref_rearrange(layer, mixed)
    q, k, v = small_ops.cat_free_rearrange(layer, mixed)
    assert q.shape == qr.shape and k.shape == kr.shape and v.shape == vr.shape
    assert torch.equal(q, qr) and torch.equal(k, kr) and torch.equal(v, vr)
    # c=1:split 切片全连续 → q/k/v 是原缓冲视图,零拷贝
    base = mixed.data_ptr()
    es = mixed.element_size()
    assert q.data_ptr() == base
    assert k.data_ptr() == base + Q_DIM * es
    assert v.data_ptr() == base + (Q_DIM + K_DIM) * es
    assert small_ops.COUNTERS["gdnsmall_rearrange_calls"] == 1


def test_rearrange_matches_reference_c8(monkeypatch):
    monkeypatch.setenv("TILEINFER_OPS", "gdnsmall")
    layer = _layer()
    g = torch.Generator().manual_seed(11)
    mixed = torch.randn(8, Q_DIM + K_DIM + V_DIM, generator=g)
    qr, kr, vr = _ref_rearrange(layer, mixed)
    q, k, v = small_ops.cat_free_rearrange(layer, mixed)
    assert torch.equal(q, qr) and torch.equal(k, kr) and torch.equal(v, vr)


def test_rearrange_none_passthrough(monkeypatch):
    monkeypatch.setenv("TILEINFER_OPS", "gdnsmall")
    layer = _layer()
    assert small_ops.cat_free_rearrange(layer, None) == (None, None, None)
    assert small_ops.COUNTERS["gdnsmall_rearrange_calls"] == 0


def test_l2norm_paired_fusion_equivalence(monkeypatch):
    monkeypatch.setenv("TILEINFER_OPS", "gdnsmall")
    layer = _layer()
    small_ops._orig_l2norm = _ref_l2norm
    wrapped = small_ops.make_l2norm_wrapper(_ref_l2norm)
    g = torch.Generator().manual_seed(3)
    mixed = torch.randn(1, Q_DIM + K_DIM + V_DIM, generator=g)

    q, k, v = small_ops.cat_free_rearrange(layer, mixed)
    q1 = wrapped(q)
    k1 = wrapped(k)
    # 参照:两次独立调用
    q2, k2 = _ref_l2norm(q), _ref_l2norm(k)
    assert torch.equal(q1, q2)
    assert torch.equal(k1, k2)
    assert small_ops.COUNTERS["gdnsmall_l2norm_fused"] == 1
    # 配对槽已消费;再单独调用走独立路径
    x = torch.randn(2, HK, DK)
    assert torch.equal(wrapped(x), _ref_l2norm(x))
    assert small_ops.COUNTERS["gdnsmall_l2norm_fused"] == 1


def test_l2norm_fallback_unrelated_tensor(monkeypatch):
    monkeypatch.setenv("TILEINFER_OPS", "gdnsmall")
    wrapped = small_ops.make_l2norm_wrapper(_ref_l2norm)
    g = torch.Generator().manual_seed(5)
    x1 = torch.randn(1, HK, DK, generator=g)
    x2 = torch.randn(1, HK, DK, generator=g) + 10.0
    assert torch.equal(wrapped(x1), _ref_l2norm(x1))
    assert torch.equal(wrapped(x2), _ref_l2norm(x2))
    assert small_ops.COUNTERS["gdnsmall_l2norm_fused"] == 0


def test_l2norm_ops_disabled_passthrough(monkeypatch):
    monkeypatch.setenv("TILEINFER_OPS", "decode")
    layer = _layer()
    wrapped = small_ops.make_l2norm_wrapper(_ref_l2norm)
    g = torch.Generator().manual_seed(9)
    mixed = torch.randn(1, Q_DIM + K_DIM + V_DIM, generator=g)
    q, k, v = small_ops.cat_free_rearrange(layer, mixed)
    assert torch.equal(wrapped(q), _ref_l2norm(q))
    assert small_ops.COUNTERS["gdnsmall_l2norm_fused"] == 0


def test_conv_wrapper_caches_noncontig_weight(monkeypatch):
    monkeypatch.setenv("TILEINFER_OPS", "gdnsmall")
    calls = []

    def stub_orig(out, x, w, **kw):
        calls.append(w.is_contiguous())
        return "ok"

    wrapped = small_ops.make_conv1d_wrapper(stub_orig)
    g = torch.Generator().manual_seed(13)
    weight = torch.randn(8192, 4, generator=g)
    w_view = weight.view(8192, 4).transpose(0, 1)  # 每步重造的转置视图
    assert not w_view.is_contiguous()
    out = torch.empty(1, 32, 8192)
    x = torch.randn(1, 32, 8192, generator=g)
    assert wrapped(out, x, w_view) == "ok"
    assert wrapped(out, x, w_view) == "ok"
    assert calls == [True, True]  # 两次都收到连续张量(首次物化,此后命中缓存)
    assert small_ops.COUNTERS["gdnsmall_conv_hits"] == 2
    # 缓存值与视图逐位一致
    cached = list(small_ops._conv_cache.values())[0][1]
    assert torch.equal(cached, w_view.contiguous())


def test_conv_wrapper_passthrough_contig(monkeypatch):
    monkeypatch.setenv("TILEINFER_OPS", "gdnsmall")
    seen = []

    def stub_orig(out, x, w, **kw):
        seen.append(w)
        return "ok"

    wrapped = small_ops.make_conv1d_wrapper(stub_orig)
    w = torch.randn(4, 8192)
    assert wrapped(None, None, w) == "ok"
    assert seen[0] is w
    assert small_ops.COUNTERS["gdnsmall_conv_hits"] == 0


def test_conv_wrapper_ops_disabled(monkeypatch):
    monkeypatch.setenv("TILEINFER_OPS", "moe")
    seen = []

    def stub_orig(out, x, w, **kw):
        seen.append(w)
        return "ok"

    wrapped = small_ops.make_conv1d_wrapper(stub_orig)
    w_view = torch.randn(8192, 4).transpose(0, 1)
    assert wrapped(None, None, w_view) == "ok"
    assert seen[0] is w_view
    assert small_ops.COUNTERS["gdnsmall_conv_hits"] == 0


def test_stats_sink_wiring():
    """安装函数把计数器同步进后端类 _stats(签名存在性回归)。"""
    from tileinfer.integrations.vllm_ascend import TileInferDecodeAttention

    for key in (
        "gdnsmall_rearrange_calls",
        "gdnsmall_conv_hits",
        "gdnsmall_l2norm_fused",
        "gdnsmall_fallbacks",
    ):
        assert key in TileInferDecodeAttention._stats
