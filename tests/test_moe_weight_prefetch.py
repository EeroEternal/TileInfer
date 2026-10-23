"""H37 MoE 权重 prefetch 阈值对齐契约测试(tileinfer.kernels.moe.prefetch)。

范围(CPU,不依赖引擎/设备):
* ``moeprefetch`` 开关注册与 ``ops_enabled`` 回归;
* 安装期阈值 patch(96→1)与幂等;开关未列 moeprefetch 时不安装;
* 惰性替位对象:ADDCF enabled 配置优先、自建 fallback 配置(attn 置零/moe 使能);
  物化失败 → 永久 falsy + fallbacks 计数,不破引擎;
* ``_arm``:moe 臂强制使能、attn 臂置零(单变量纪律)、MoE hook 计数包装
  (active/fallbacks),异常原样上抛;
* 委托面:bool/方法/常量属性经替位对象达到真实 method。

设备/引擎权威验证链:H37 权威口径引擎 A/B(在役探针 moeprefetch_active +
图模式端到端),见 HYPOTHESES H37;950PR L2-bypass 风险以实测裁决(E109)。
"""

from __future__ import annotations

import pytest
import torch

from tileinfer.kernels.moe import prefetch as pf


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    """隔离全局态:安装标记/计数器/sink/vllm_ascend 全局。"""
    monkeypatch.setattr(pf, "_installed", False)
    monkeypatch.setattr(pf, "COUNTERS", {k: 0 for k in pf.COUNTERS})
    monkeypatch.setattr(pf, "_stats_sink", None)
    vutils = pytest.importorskip("vllm_ascend.utils")
    wp = pytest.importorskip("vllm_ascend.ops.weight_prefetch")
    monkeypatch.setattr(vutils, "_WEIGHT_PREFETCH_METHOD", None, raising=False)
    monkeypatch.setattr(wp, "MOE_PREFETCH_TOKEN_THRESHOLD", 96, raising=False)
    yield


def _sink() -> dict[str, int]:
    seen: dict[str, int] = {}
    pf._stats_sink = lambda name: seen.__setitem__(name, seen.get(name, 0) + 1)
    return seen


def test_ops_switch_registered(monkeypatch):
    from tileinfer.integrations.vllm_ascend import _OPS_ALL, ops_enabled

    assert "moeprefetch" in _OPS_ALL
    monkeypatch.setenv("TILEINFER_OPS", "decode,gdnproj,moeprefetch")
    assert ops_enabled("moeprefetch")
    monkeypatch.setenv("TILEINFER_OPS", "decode,gdnproj")
    assert not ops_enabled("moeprefetch")


def test_install_patches_threshold_and_idempotent(monkeypatch):
    pytest.importorskip("vllm_ascend.ops.weight_prefetch")
    wp = pytest.importorskip("vllm_ascend.ops.weight_prefetch")
    vutils = pytest.importorskip("vllm_ascend.utils")
    monkeypatch.setenv("TILEINFER_OPS", "moeprefetch")
    assert pf.install_moe_weight_prefetch() is True
    assert wp.MOE_PREFETCH_TOKEN_THRESHOLD == 1
    assert isinstance(vutils._WEIGHT_PREFETCH_METHOD, pf._LazyPrefetchMethod)
    # 幂等:二次安装直接 True,不重复替换
    holder = vutils._WEIGHT_PREFETCH_METHOD
    assert pf.install_moe_weight_prefetch() is True
    assert vutils._WEIGHT_PREFETCH_METHOD is holder
    assert pf._installed


def test_install_off_switch_noop(monkeypatch):
    pytest.importorskip("vllm_ascend.ops.weight_prefetch")
    wp = pytest.importorskip("vllm_ascend.ops.weight_prefetch")
    vutils = pytest.importorskip("vllm_ascend.utils")
    monkeypatch.setenv("TILEINFER_OPS", "decode,gdnproj")
    assert pf.install_moe_weight_prefetch() is False
    assert wp.MOE_PREFETCH_TOKEN_THRESHOLD == 96
    assert vutils._WEIGHT_PREFETCH_METHOD is None
    assert not pf._installed


def _stub_real(monkeypatch, enabled: bool = True):
    """给 wp 模块打桩,使 WeightPrefetchMethod 构造不依赖 vllm 全局 config。"""
    wp = pytest.importorskip("vllm_ascend.ops.weight_prefetch")
    ac = pytest.importorskip("vllm_ascend.ascend_config")

    class _StubCfg:
        def __init__(self, d):
            self.enabled = d.get("enabled", False)
            self.prefetch_ratio = d.get("prefetch_ratio", {})

    monkeypatch.setattr(wp, "get_current_vllm_config", lambda: object(), raising=False)
    monkeypatch.setattr(wp, "is_moe_model", lambda cfg: True, raising=False)
    # 前向上下文在 CPU 契约测试中不存在 → hook 内部短路返回(stock 行为)
    monkeypatch.setattr(wp, "get_forward_context", lambda: None, raising=False)
    monkeypatch.setattr(ac, "WeightPrefetchConfig", _StubCfg, raising=False)
    return wp, ac


def test_lazy_materialization_fallback_cfg(monkeypatch):
    _stub_real(monkeypatch)
    ac = pytest.importorskip("vllm_ascend.ascend_config")

    def _no_cfg():
        raise RuntimeError("ascend config not initialized")

    monkeypatch.setattr(ac, "get_ascend_config", _no_cfg, raising=False)
    vutils = pytest.importorskip("vllm_ascend.utils")
    monkeypatch.setenv("TILEINFER_OPS", "moeprefetch")
    assert pf.install_moe_weight_prefetch() is True
    lazy = vutils._WEIGHT_PREFETCH_METHOD
    assert bool(lazy) is True
    real = lazy._materialize()
    assert real.moe.enable is True
    assert real.moe.prefetch_ratio == {"gate_up": 0.8}
    # 单变量纪律:attn 臂恒置零
    assert real.attn.enable is False
    assert real.mlp.enable is False  # MoE 模型下 mlp 臂天然失活
    # 计数包装已挂:真实 method 的 preprocess 已被替换
    assert real.maybe_prefetch_moe_weight_preprocess.__name__ == "_pre"


def test_lazy_materialization_prefers_addcfg(monkeypatch):
    wp, ac = _stub_real(monkeypatch)
    vutils = pytest.importorskip("vllm_ascend.utils")

    class _AddCfg:
        enabled = True
        prefetch_ratio = {"moe": {"gate_up": 0.5}, "attn": {"qkv": 1.0, "o": 1.0}}

    monkeypatch.setattr(
        ac, "get_ascend_config", lambda: type("AC", (), {"weight_prefetch_config": _AddCfg()})(),
        raising=False,
    )
    monkeypatch.setenv("TILEINFER_OPS", "moeprefetch")
    assert pf.install_moe_weight_prefetch() is True
    real = vutils._WEIGHT_PREFETCH_METHOD._materialize()
    assert real.moe.prefetch_ratio == {"gate_up": 0.5}
    # ADDCFG 显式给了 attn 非零比例 → _arm 单变量纪律强制置零
    assert real.attn.enable is False


def test_lazy_dead_path_falsy_and_counts(monkeypatch):
    wp, ac = _stub_real(monkeypatch)

    def _boom():
        raise RuntimeError("no config")

    monkeypatch.setattr(ac, "get_ascend_config", _boom, raising=False)
    # 物化本身失败:WeightPrefetchMethod 换成构造即炸的替身
    def _bad_cfg(_d):
        raise RuntimeError("bad cfg")

    monkeypatch.setattr(ac, "WeightPrefetchConfig", _bad_cfg, raising=False)
    seen = _sink()
    vutils = pytest.importorskip("vllm_ascend.utils")
    monkeypatch.setenv("TILEINFER_OPS", "moeprefetch")
    assert pf.install_moe_weight_prefetch() is True
    lazy = vutils._WEIGHT_PREFETCH_METHOD
    assert bool(lazy) is False
    assert pf.COUNTERS["moeprefetch_fallbacks"] == 1
    assert seen.get("moeprefetch_fallbacks") == 1
    # 永久 falsy:再次访问不再重试
    assert bool(lazy) is False
    assert pf.COUNTERS["moeprefetch_fallbacks"] == 1


def test_arm_existing_method_counts(monkeypatch):
    """worker init 先行时序:install 就地武装既有 method 并挂计数。"""
    wp = pytest.importorskip("vllm_ascend.ops.weight_prefetch")
    vutils = pytest.importorskip("vllm_ascend.utils")
    seen = _sink()

    class _Arm:
        def __init__(self):
            self.moe = type("M", (), {"enable": False, "prefetch_ratio": {"gate_up": 0.8}, "is_active_this_forward": False})()
            self.attn = type("A", (), {"enable": True, "prefetch_ratio": {"qkv": 1.0, "o": 1.0}})()
            self.mlp = type("L", (), {"enable": False})()

        def maybe_prefetch_moe_weight_preprocess(self, hs, prefix):
            # 门槛逻辑与上游一致:模块全局阈值在调用期读取
            self.moe.is_active_this_forward = hs.shape[0] >= wp.MOE_PREFETCH_TOKEN_THRESHOLD

        def maybe_prefetch_moe_weight_postprocess(self, flag):
            pass

    fake = _Arm()
    monkeypatch.setattr(vutils, "_WEIGHT_PREFETCH_METHOD", fake, raising=False)
    monkeypatch.setenv("TILEINFER_OPS", "moeprefetch")
    assert pf.install_moe_weight_prefetch() is True
    assert wp.MOE_PREFETCH_TOKEN_THRESHOLD == 1
    # 阈值对齐后:M=8(此前被 96 排除)现在激活
    fake.maybe_prefetch_moe_weight_preprocess(torch.zeros(8, 4), "gate_up")
    assert fake.moe.is_active_this_forward is True
    assert pf.COUNTERS["moeprefetch_active"] == 1
    assert seen.get("moeprefetch_active") == 1
    fake.maybe_prefetch_moe_weight_preprocess(torch.zeros(8, 4), "gate_up")
    assert pf.COUNTERS["moeprefetch_active"] == 2
    # 武装副作用:moe 使能、attn 置零
    assert fake.moe.enable is True
    assert fake.attn.enable is False
    # 异常上抛语义:被包装的原实现炸 → fallbacks 计数后原样抛出(不吞)
    def _boom(hs, prefix):
        raise RuntimeError("x")

    # 独立 fake 走完整 install→call 路径(闭包捕获的 orig 即会炸的 _boom)
    fake2 = _Arm()
    fake2.maybe_prefetch_moe_weight_preprocess = _boom
    pf._installed = False
    monkeypatch.setattr(vutils, "_WEIGHT_PREFETCH_METHOD", fake2, raising=False)
    assert pf.install_moe_weight_prefetch() is True
    with pytest.raises(RuntimeError):
        fake2.maybe_prefetch_moe_weight_preprocess(torch.zeros(8, 4), "gate_up")
    assert pf.COUNTERS["moeprefetch_fallbacks"] == 1


def test_lazy_delegation_surface(monkeypatch):
    """委托面:方法/常量属性经替位对象达到真实 method(线性 op MLP hook 等引用)。"""
    wp, ac = _stub_real(monkeypatch)
    vutils = pytest.importorskip("vllm_ascend.utils")
    monkeypatch.setenv("TILEINFER_OPS", "moeprefetch")
    assert pf.install_moe_weight_prefetch() is True
    lazy = vutils._WEIGHT_PREFETCH_METHOD
    real = lazy._materialize()
    assert lazy.MLP_GATE_UP == real.MLP_GATE_UP
    assert lazy.MLP_DOWN == real.MLP_DOWN
    assert lazy.mla_sfa_prefetch_enable == real.mla_sfa_prefetch_enable
    # 未物化失败场景下,方法调用透传(参数原样;preprocess 在 forward context
    # 短路返回,不触达设备 op——postprocess 的设备 op 属 NPU 域,不进 CPU 契约)
    lazy.maybe_prefetch_moe_weight_preprocess(torch.zeros(32, 4), "gate_up")
    assert pf.COUNTERS["moeprefetch_active"] == 1


def test_runner_binding_constructs_in_ctx_and_arms(monkeypatch):
    """E409 修复:runner init 绑定接管——构造交还 ctx 时点+就地武装(单事实源)。

    引擎时序:worker_base.init_device 的 set_current_vllm_config 包裹 runner
    构造(model_runner_v1:338 在其中);绑定包装在该时点被调用,构造出真实
    武装 method 并落位,惰性替位对象不再被触碰。
    """
    wp, ac = _stub_real(monkeypatch)
    vutils = pytest.importorskip("vllm_ascend.utils")
    import sys
    import types

    vllm_cfg_mod = pytest.importorskip("vllm.config")
    ctx_holder: dict = {"cfg": object()}
    entered: list = []
    monkeypatch.setattr(
        vllm_cfg_mod, "get_current_vllm_config_or_none", lambda: ctx_holder["cfg"], raising=False
    )

    def _fake_set(cfg):
        entered.append(cfg)
        import contextlib

        @contextlib.contextmanager
        def _cm():
            yield

        return _cm()

    monkeypatch.setattr(vllm_cfg_mod, "set_current_vllm_config", _fake_set, raising=False)

    fake_mrv1 = types.ModuleType("vllm_ascend.worker.model_runner_v1")
    stock_calls: list = []

    def _orig_set(cfg):
        stock_calls.append(cfg)

    fake_mrv1.set_weight_prefetch_method = _orig_set
    fake_worker = types.ModuleType("vllm_ascend.worker")
    fake_worker.model_runner_v1 = fake_mrv1
    monkeypatch.setitem(sys.modules, "vllm_ascend.worker", fake_worker)
    monkeypatch.setitem(sys.modules, "vllm_ascend.worker.model_runner_v1", fake_mrv1)

    monkeypatch.setenv("TILEINFER_OPS", "moeprefetch")
    assert pf.install_moe_weight_prefetch() is True
    # 绑定接管:mrv1 绑定被替换为 force-construct 包装(打标幂等)
    assert getattr(fake_mrv1.set_weight_prefetch_method, "_tileinfer_force_construct", False) is True
    # runner init 时点调用:构造自持(ctx 在位→ctx 包装),槽位落真实武装 method
    fake_mrv1.set_weight_prefetch_method(None)
    real = vutils._WEIGHT_PREFETCH_METHOD
    assert real is not None and not isinstance(real, pf._LazyPrefetchMethod)
    assert real.moe.enable is True
    assert entered, "ctx 包装未发生"
    assert stock_calls == [], "stock 原函数不应被调用(构造权自持)"
    assert real.maybe_prefetch_moe_weight_preprocess.__name__ == "_pre"
    assert real.attn.enable is False  # 单变量纪律
    # 幂等:二次 install 走 _installed 早退,绑定不重复包装
    binding = fake_mrv1.set_weight_prefetch_method
    assert pf.install_moe_weight_prefetch() is True
    assert fake_mrv1.set_weight_prefetch_method is binding
