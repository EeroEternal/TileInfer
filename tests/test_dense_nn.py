"""dense_nn(H77)契约测试:门控、形状族识别、预排正确性、apply shim 语义。

背景:H77 假设——dense b1 大形状(qkvz/attn_qkv/o_proj)引擎内缺口 1.24-1.3x
(E297=E255×E269 交叉),ND 物理连续 NN 布局(E100 M=16 +11%)是唯一未测正证据轴。
本测试在纯 CPU 面锁死安全契约:
  1) 门控:env 未设/0 → install False、零包装;子集族语法;
  2) 形状族:三族 (N,K) 命中、非目标形状(router/ba/shared/lm_head)零触碰;
  3) 预排:w_nn=[K,N] 连续、数值上 x@w_nn ≡ F.linear(x, w)(fp32 参照);
  4) shim 语义:小 M 走 matmul 路径、大 M 回落原 apply、bias 层跳过、
     TP>1 跳过、幂等(重复 PWAL 不二次 shim);
  5) wrapper 链:与 H73 共用 PWAL 钩子时各自独立包装、原实现必被调用。

运行::

    .venv/bin/python -m pytest tests/test_dense_nn.py -v
"""

from __future__ import annotations

import types

import pytest
import torch

try:
    from tileinfer.integrations import dense_nn as dn

    _HAVE_MOD = True
except Exception:  # pragma: no cover
    _HAVE_MOD = False

pytestmark = pytest.mark.skipif(not _HAVE_MOD, reason="dense_nn module unavailable")


class _FakeQM:
    """最小 quant_method 替身:apply 记录调用并返回标记。"""

    def __init__(self):
        self.calls = []
        self.weight_reads = []

    def apply(self, layer, x, bias=None):
        self.calls.append(("orig", tuple(x.shape)))
        self.weight_reads.append(tuple(layer.weight.shape))
        return torch.full((x.shape[0], layer.weight.shape[0]), 7.0, dtype=x.dtype)


class _FakeLayer:
    def __init__(self, n, k, bias=False, family_weight=None):
        self.prefix = "model.layers.0.self_attn.qkv_proj"
        self.weight = torch.nn.Parameter(torch.randn(n, k, dtype=torch.bfloat16) * 0.02)
        self.bias = torch.nn.Parameter(torch.randn(n, dtype=torch.bfloat16)) if bias else None
        self.quant_method = _FakeQM()


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    monkeypatch.delenv("TILEINFER_DENSE_NN", raising=False)
    monkeypatch.delenv("TILEINFER_DENSE_NN_MAXM", raising=False)
    monkeypatch.delenv("TILEINFER_DENSE_NN_DRYRUN", raising=False)
    monkeypatch.setattr(dn, "_INSTALLED", False)
    for k in dn._STATS:
        dn._STATS[k] = 0
    yield
    # 还原类级包装,避免污染其他测试
    try:
        from vllm.model_executor.layers.linear import UnquantizedLinearMethod

        cur = UnquantizedLinearMethod.process_weights_after_loading
        m = getattr(cur, "_tileinfer_dense_nn_orig", None)
        if m is not None:
            UnquantizedLinearMethod.process_weights_after_loading = m
    except Exception:
        pass


# ---------------------------------------------------------------- 1) 门控

def test_gate_off_by_default(monkeypatch):
    assert dn._enabled_families() is None
    assert dn.install() is False
    assert dn.is_installed() is False


def test_gate_all_and_subset(monkeypatch):
    monkeypatch.setenv("TILEINFER_DENSE_NN", "1")
    assert dn._enabled_families() == {"qkvz", "attnqkv", "oproj"}
    monkeypatch.setenv("TILEINFER_DENSE_NN", "qkvz, oproj")
    assert dn._enabled_families() == {"qkvz", "oproj"}
    monkeypatch.setenv("TILEINFER_DENSE_NN", "bogus")
    assert dn._enabled_families() is None


def test_install_idempotent_and_wraps(monkeypatch):
    monkeypatch.setenv("TILEINFER_DENSE_NN", "1")
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    orig = UnquantizedLinearMethod.process_weights_after_loading
    assert dn.install() is True
    wrapped = UnquantizedLinearMethod.process_weights_after_loading
    assert wrapped is not orig
    assert getattr(wrapped, "_tileinfer_dense_nn_wrapped", False)
    assert dn.install() is True  # 二次 install 幂等
    assert UnquantizedLinearMethod.process_weights_after_loading is wrapped


# ---------------------------------------------------------------- 2) 形状族

def test_family_detection_and_non_targets(monkeypatch):
    monkeypatch.setenv("TILEINFER_DENSE_NN", "1")
    hits = 0
    for n, k, expect in [
        (12288, 2048, "qkvz"), (9216, 2048, "attnqkv"), (2048, 4096, "oproj"),
        (256, 2048, None),      # router
        (64, 2048, None),       # in_proj_ba
        (1024, 2048, None),     # shared gate_up(H28/H73 域,零交集)
        (248320, 2048, None),   # lm_head
    ]:
        dn._STATS["dense_nn_modules"] = 0
        layer = _FakeLayer(n, k)
        got = dn._shim_layer(layer)
        assert got == (expect is not None), (n, k)
        if expect:
            hits += 1
            assert dn._FAMILY_BY_SHAPE[(n, k)] == expect
            assert dn.stats()["dense_nn_modules"] == 1
        else:
            assert dn.stats()["dense_nn_modules"] == 0
            assert not hasattr(layer.quant_method, "_tileinfer_dense_nn")
    assert hits == 3


# ---------------------------------------------------------------- 3) 预排正确性

def test_preformat_numerics(monkeypatch):
    monkeypatch.setenv("TILEINFER_DENSE_NN", "1")
    torch.manual_seed(0)
    n, k = 2048, 4096  # o_proj 族形状(bf16 CPU matmul 快)
    layer = _FakeLayer(n, k)
    w_nd = layer.weight.data
    dn._shim_layer(layer)
    w_nn = layer.quant_method.apply.__defaults__[1]
    assert w_nn.shape == (k, n)
    assert w_nn.is_contiguous()
    x = torch.randn(4, k, dtype=torch.bfloat16)
    ref = torch.nn.functional.linear(x, w_nd)
    out = torch.matmul(x, w_nn)
    torch.testing.assert_close(out.float(), ref.float(), atol=2e-2, rtol=2e-2)


# ---------------------------------------------------------------- 4) shim 语义

def test_shim_small_m_uses_matmul_large_m_falls_back(monkeypatch):
    monkeypatch.setenv("TILEINFER_DENSE_NN", "1")
    layer = _FakeLayer(12288, 2048)
    dn._shim_layer(layer)
    qm = layer.quant_method

    x_small = torch.randn(8, 2048, dtype=torch.bfloat16)
    out = qm.apply(layer, x_small)
    assert torch.allclose(out, torch.matmul(x_small, qm.apply.__defaults__[1]))
    assert dn.stats()["dense_nn_forward"] == 1

    x_big = torch.randn(64, 2048, dtype=torch.bfloat16)
    out2 = qm.apply(layer, x_big)
    assert out2.shape == (64, 12288) and float(out2[0, 0]) == 7.0  # orig 替身标记
    assert dn.stats()["dense_nn_fallback"] == 1


def test_maxm_env_respected(monkeypatch):
    monkeypatch.setenv("TILEINFER_DENSE_NN", "1")
    monkeypatch.setenv("TILEINFER_DENSE_NN_MAXM", "2")
    layer = _FakeLayer(12288, 2048)
    dn._shim_layer(layer)
    assert layer.quant_method.apply.__defaults__[3] == 2
    dn._STATS["dense_nn_forward"] = 0
    layer.quant_method.apply(layer, torch.randn(3, 2048, dtype=torch.bfloat16))
    assert dn.stats()["dense_nn_forward"] == 0
    assert dn.stats()["dense_nn_fallback"] == 1


def test_bias_layer_skipped(monkeypatch):
    monkeypatch.setenv("TILEINFER_DENSE_NN", "1")
    layer = _FakeLayer(12288, 2048, bias=True)
    assert dn._shim_layer(layer) is False
    assert dn.stats()["dense_nn_skipped"] == 1
    assert not hasattr(layer.quant_method, "_tileinfer_dense_nn")


def test_non_bf16_fp16_skipped(monkeypatch):
    monkeypatch.setenv("TILEINFER_DENSE_NN", "1")
    layer = _FakeLayer(12288, 2048)
    layer.weight = torch.nn.Parameter(torch.randn(12288, 2048, dtype=torch.float32))
    assert dn._shim_layer(layer) is False


def test_tp_gt1_skipped(monkeypatch):
    monkeypatch.setenv("TILEINFER_DENSE_NN", "1")
    monkeypatch.setattr(dn, "_tp1", lambda: False)
    layer = _FakeLayer(12288, 2048)
    assert dn._shim_layer(layer) is False
    assert dn.stats()["dense_nn_skipped"] == 1


def test_double_pwali_no_double_shim(monkeypatch):
    monkeypatch.setenv("TILEINFER_DENSE_NN", "1")
    layer = _FakeLayer(12288, 2048)
    assert dn._shim_layer(layer) is True
    qm = layer.quant_method
    shim1 = qm.apply
    assert dn._shim_layer(layer) is False  # 已 shim 的层不再处理
    assert qm.apply is shim1
    assert dn.stats()["dense_nn_modules"] == 1


def test_dryrun_no_shim(monkeypatch):
    monkeypatch.setenv("TILEINFER_DENSE_NN", "1")
    monkeypatch.setenv("TILEINFER_DENSE_NN_DRYRUN", "1")
    layer = _FakeLayer(12288, 2048)
    assert dn._shim_layer(layer) is False
    assert not hasattr(layer.quant_method, "_tileinfer_dense_nn")


# ---------------------------------------------------------------- 5) wrapper 链

def test_pwal_chain_calls_orig_and_shims(monkeypatch):
    monkeypatch.setenv("TILEINFER_DENSE_NN", "1")
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    orig = UnquantizedLinearMethod.process_weights_after_loading
    assert dn.install() is True
    wrapped = UnquantizedLinearMethod.process_weights_after_loading
    assert wrapped is not orig

    qm = _FakeQM()
    layer = _FakeLayer(12288, 2048)
    layer.quant_method = qm
    # 经类方法走一遍(与引擎调用同构)
    wrapped(qm, layer)
    assert hasattr(qm, "_tileinfer_dense_nn")
    # orig(此处为 _FakeQM.apply 替身)仍可经 fallback 触达
    out = qm.apply(layer, torch.randn(64, 2048, dtype=torch.bfloat16))
    assert float(out[0, 0]) == 7.0
    assert dn.stats()["dense_nn_fallback"] == 1


def test_wrapper_marker_distinct_from_h73(monkeypatch):
    """H73 用 _tileinfer_wrapped 标记,H77 用 _tileinfer_dense_nn_wrapped:
    两者互不吞包装(H73 先装后,H77 仍须能装上)。"""
    monkeypatch.setenv("TILEINFER_DENSE_NN", "1")
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    def h73_style_pwal(self, layer, *a, **kw):
        return None

    h73_style_pwal._tileinfer_wrapped = True
    UnquantizedLinearMethod.process_weights_after_loading = h73_style_pwal
    assert dn.install() is True
    cur = UnquantizedLinearMethod.process_weights_after_loading
    assert cur is not h73_style_pwal
    assert getattr(cur, "_tileinfer_dense_nn_wrapped", False)
    # H77 wrapper 内部仍调用 H73 形态的 orig
    qm = _FakeQM()
    layer = _FakeLayer(12288, 2048)
    layer.quant_method = qm
    cur(qm, layer)
    assert hasattr(qm, "_tileinfer_dense_nn")
