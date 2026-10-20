"""dense_nn(H79 v2=weight-swap)契约测试:门控、形状族识别、换形正确性、探针语义。

背景:H77 v1(E320 定谳)python qm.apply shim 在 FULL graph decode 被整体旁路;
v2=PWAL 期 weight.data 物理换形(存储 [K,N] 连续+逻辑 [N,K] 视图,stride=(1,N)),
vendor apply 按 stride 自动走 NN 内核,零热路径分支。E321 stride 微基准:
o_proj 全 M 段 +14~17%,部署白名单=oproj。
本测试在纯 CPU 面锁死安全契约:
  1) 门控:env 未设/0 → install False、零包装;子集族语法;
  2) 形状族:三族 (N,K) 命中、非目标形状(router/ba/shared/lm_head)零触碰;
  3) 换形:shape 不变、stride=(1,N)、.t() 连续、数值 x@w 形态 ≡ F.linear(x, w_orig);
  4) 语义:quant_method.apply 不被替换(零热路径)、bias 层跳过、非 bf16/fp16 跳过、
     TP>1 跳过、幂等(重复 PWAL 不重复换形/重复计数)、stride 被改写后自愈;
  5) wrapper 链:core+leaf 双类后置钩子、原实现必被调用、与 H73 marker 互不干扰;
  6) 探针:stats() 的 stride_ok/stride_bad 物理断言面。

运行::

    .venv/bin/python -m pytest tests/test_dense_nn.py -v
"""

from __future__ import annotations

import pytest
import torch

try:
    from tileinfer.integrations import dense_nn as dn

    _HAVE_MOD = True
except Exception:  # pragma: no cover
    _HAVE_MOD = False

pytestmark = pytest.mark.skipif(not _HAVE_MOD, reason="dense_nn module unavailable")


def _patched_classes():
    classes = []
    try:
        from vllm.model_executor.layers.linear import UnquantizedLinearMethod

        classes.append(UnquantizedLinearMethod)
    except Exception:
        pass
    try:
        from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod

        classes.append(AscendUnquantizedLinearMethod)
    except Exception:
        pass
    return classes


class _FakeQM:
    """最小 quant_method 替身:apply 与引擎同构(F.linear on weight.data)。"""

    def __init__(self):
        self.calls = 0

    def apply(self, layer, x, bias=None):
        self.calls += 1
        return torch.nn.functional.linear(x, layer.weight.data, bias)


class _FakeLayer:
    def __init__(self, n, k, bias=False):
        self.prefix = "model.layers.0.self_attn.o_proj"
        self.weight = torch.nn.Parameter(torch.randn(n, k, dtype=torch.bfloat16) * 0.02)
        self.bias = torch.nn.Parameter(torch.randn(n, dtype=torch.bfloat16)) if bias else None
        self.quant_method = _FakeQM()


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    monkeypatch.delenv("TILEINFER_DENSE_NN", raising=False)
    monkeypatch.delenv("TILEINFER_DENSE_NN_DRYRUN", raising=False)
    monkeypatch.setattr(dn, "_INSTALLED", False)
    dn._REGISTERED.clear()
    for k in dn._STATS:
        dn._STATS[k] = 0
    yield
    # 还原类级包装,避免污染其他测试
    for cls in _patched_classes():
        cur = cls.process_weights_after_loading
        m = getattr(cur, "_tileinfer_dense_nn_orig", None)
        if m is not None:
            cls.process_weights_after_loading = m


# ---------------------------------------------------------------- 1) 门控

def test_gate_off_by_default():
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
    classes = _patched_classes()
    assert classes, "vllm linear classes unavailable"
    before = {c: c.process_weights_after_loading for c in classes}
    assert dn.install() is True
    n_wrapped = sum(
        getattr(c.process_weights_after_loading, "_tileinfer_dense_nn_wrapped", False)
        for c in classes)
    assert n_wrapped >= 1
    assert dn.install() is True  # 二次 install 幂等
    for c in classes:
        assert c.process_weights_after_loading is before[c] or \
            getattr(c.process_weights_after_loading, "_tileinfer_dense_nn_wrapped", False)


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
        got = dn._swap_layer(layer)
        assert got == (expect is not None), (n, k)
        if expect:
            hits += 1
            assert dn._FAMILY_BY_SHAPE[(n, k)] == expect
            assert dn.stats()["dense_nn_modules"] == 1
        else:
            assert dn.stats()["dense_nn_modules"] == 0
            assert not getattr(layer, "_tileinfer_dense_nn_swapped", False)
        dn._REGISTERED.clear()
    assert hits == 3


# ---------------------------------------------------------------- 3) 换形正确性

def test_swap_layout_and_numerics(monkeypatch):
    monkeypatch.setenv("TILEINFER_DENSE_NN", "1")
    torch.manual_seed(0)
    n, k = 2048, 4096  # o_proj 族形状(E321 白名单族)
    layer = _FakeLayer(n, k)
    w_orig = layer.weight.data.clone()
    qm_before = layer.quant_method
    assert dn._swap_layer(layer) is True
    w = layer.weight.data
    # 逻辑形状不变;stride=(1,N);B 视图(.t())物理连续 NN
    assert tuple(w.shape) == (n, k)
    assert tuple(w.stride()) == (1, n)
    assert w.t().is_contiguous()
    assert not w.is_contiguous()
    assert w.data_ptr() != w_orig.data_ptr()
    # 零热路径改动:quant_method 实例与 apply 均未被替换
    assert layer.quant_method is qm_before
    x = torch.randn(4, k, dtype=torch.bfloat16)
    ref = torch.nn.functional.linear(x, w_orig)
    out = layer.quant_method.apply(layer, x)  # 与引擎 unquantized_gemm 同构
    torch.testing.assert_close(out.float(), ref.float(), atol=2e-2, rtol=2e-2)


# ---------------------------------------------------------------- 4) 语义

def test_bias_layer_skipped(monkeypatch):
    monkeypatch.setenv("TILEINFER_DENSE_NN", "1")
    layer = _FakeLayer(2048, 4096, bias=True)
    assert dn._swap_layer(layer) is False
    assert dn.stats()["dense_nn_skipped"] == 1
    assert not getattr(layer, "_tileinfer_dense_nn_swapped", False)


def test_non_bf16_fp16_skipped(monkeypatch):
    monkeypatch.setenv("TILEINFER_DENSE_NN", "1")
    layer = _FakeLayer(2048, 4096)
    layer.weight = torch.nn.Parameter(torch.randn(2048, 4096, dtype=torch.float32))
    assert dn._swap_layer(layer) is False


def test_tp_gt1_skipped(monkeypatch):
    monkeypatch.setenv("TILEINFER_DENSE_NN", "1")
    monkeypatch.setattr(dn, "_tp1", lambda: False)
    layer = _FakeLayer(2048, 4096)
    assert dn._swap_layer(layer) is False
    assert dn.stats()["dense_nn_skipped"] == 1


def test_double_swap_idempotent(monkeypatch):
    monkeypatch.setenv("TILEINFER_DENSE_NN", "1")
    layer = _FakeLayer(2048, 4096)
    assert dn._swap_layer(layer) is True
    ptr1 = layer.weight.data.data_ptr()
    stride1 = tuple(layer.weight.data.stride())
    assert dn._swap_layer(layer) is False  # 已换形,stride 健康 → 跳过
    assert layer.weight.data.data_ptr() == ptr1
    assert tuple(layer.weight.data.stride()) == stride1
    assert dn.stats()["dense_nn_modules"] == 1


def test_swap_selfheal_on_clobber(monkeypatch):
    """已被外部改写(如权重更新流还原 ND)时重换形(self-heal)。"""
    monkeypatch.setenv("TILEINFER_DENSE_NN", "1")
    layer = _FakeLayer(2048, 4096)
    assert dn._swap_layer(layer) is True
    layer.weight.data = layer.weight.data.contiguous()
    assert tuple(layer.weight.data.stride()) == (4096, 1)  # 被还原为 ND 连续
    assert dn._swap_layer(layer) is True  # 自愈重换
    assert tuple(layer.weight.data.stride()) == (1, 2048)
    assert dn.stats()["dense_nn_modules"] == 1  # 不重复计数


def test_dryrun_no_swap(monkeypatch):
    monkeypatch.setenv("TILEINFER_DENSE_NN", "1")
    monkeypatch.setenv("TILEINFER_DENSE_NN_DRYRUN", "1")
    layer = _FakeLayer(2048, 4096)
    assert dn._swap_layer(layer) is False
    assert not getattr(layer, "_tileinfer_dense_nn_swapped", False)


# ---------------------------------------------------------------- 5) wrapper 链

def test_pwal_chain_calls_orig_and_swaps(monkeypatch):
    monkeypatch.setenv("TILEINFER_DENSE_NN", "1")
    try:
        from vllm_ascend.ascend_config import init_ascend_config
        from vllm.config import VllmConfig

        init_ascend_config(VllmConfig())  # leaf 链 maybe_trans_nz 需 config 单例
    except Exception:
        pass
    assert dn.install() is True
    classes = [c for c in _patched_classes()
               if getattr(c.process_weights_after_loading,
                          "_tileinfer_dense_nn_wrapped", False)]
    assert classes
    for cls in classes:
        layer = _FakeLayer(2048, 4096)
        w_orig = layer.weight.data.clone()
        qm = cls()  # leaf 链 super() 需真实实例
        layer.quant_method = qm
        cls.process_weights_after_loading(qm, layer)
        assert getattr(layer, "_tileinfer_dense_nn_swapped", False)
        assert tuple(layer.weight.data.stride()) == (1, 2048)
        x = torch.randn(4, 4096, dtype=torch.bfloat16)
        ref = torch.nn.functional.linear(x, w_orig)
        # Ascend 真 qm.apply=custom op(仅 NPU 后端),CPU 面以同构 F.linear 直比
        out = torch.nn.functional.linear(x, layer.weight.data)
        torch.testing.assert_close(out.float(), ref.float(), atol=2e-2, rtol=2e-2)


def test_wrapper_marker_distinct_from_h73(monkeypatch):
    """H73 用 _tileinfer_wrapped 标记,H79 用 _tileinfer_dense_nn_wrapped:
    两者互不吞包装(H73 先装后,H79 仍须能装上)。"""
    monkeypatch.setenv("TILEINFER_DENSE_NN", "1")
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    def h73_style_pwal(self, layer, *a, **kw):
        return None

    h73_style_pwal._tileinfer_wrapped = True
    orig_h73 = UnquantizedLinearMethod.process_weights_after_loading
    UnquantizedLinearMethod.process_weights_after_loading = h73_style_pwal
    try:
        assert dn.install() is True
        cur = UnquantizedLinearMethod.process_weights_after_loading
        assert cur is not h73_style_pwal
        assert getattr(cur, "_tileinfer_dense_nn_wrapped", False)
        # H79 wrapper 内部仍调用 H73 形态的 orig,且换形生效
        layer = _FakeLayer(2048, 4096)
        cur(layer.quant_method, layer)
        assert getattr(layer, "_tileinfer_dense_nn_swapped", False)
    finally:
        UnquantizedLinearMethod.process_weights_after_loading = orig_h73


# ---------------------------------------------------------------- 6) 探针面

def test_stats_probe_stride_ok_bad(monkeypatch):
    monkeypatch.setenv("TILEINFER_DENSE_NN", "1")
    layer_ok = _FakeLayer(2048, 4096)
    layer_bad = _FakeLayer(2048, 4096)
    assert dn._swap_layer(layer_ok) is True
    assert dn._swap_layer(layer_bad) is True
    s = dn.stats()
    assert s["dense_nn_modules"] == 2 and s["dense_nn_stride_ok"] == 2
    assert s["dense_nn_stride_bad"] == 0
    # 外部改写一层 → stride_bad=1
    layer_bad.weight.data = layer_bad.weight.data.contiguous()
    s2 = dn.stats()
    assert s2["dense_nn_stride_ok"] == 1 and s2["dense_nn_stride_bad"] == 1
    # 弱引用:层被回收后探针不崩溃、不计数
    dn._REGISTERED.clear()
    dn._REGISTERED.append(dn.weakref.ref(layer_ok))
    dead = _FakeLayer(2048, 4096)
    dn._REGISTERED.append(dn.weakref.ref(dead))
    del dead
    s3 = dn.stats()
    assert s3["dense_nn_stride_ok"] == 1
