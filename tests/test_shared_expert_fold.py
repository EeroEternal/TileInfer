"""H28 shared expert 恒选 E+1 折叠契约测试(CPU,无设备依赖)。

覆盖:纯函数数学语义(切片/扩列/拼接/扩行)、安装接线(包装标记、默认关零行为)、
折叠时序(quant PWAL 前置、H73 融合形态剔除 gate 行)、select 包装的扩列与降级、
bypass 零张量语义。
"""

from __future__ import annotations

import os

import pytest
import torch

from tileinfer.integrations import shared_expert_fold as fold


@pytest.fixture(autouse=True)
def _reset():
    fold._INSTALLED = False
    fold._OWNER_REGISTRY.clear()
    fold._STATS.update(dict.fromkeys(fold._STATS, 0))
    fold._DISPATCHER_BUMPED.update({"e": None, "k": None})
    old = {k: os.environ.get(k) for k in ("TILEINFER_SHARED_EXPERT_FOLD", "TILEINFER_SHARED_EXPERT_FOLD_DRYRUN")}
    yield
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


# ---------------------------------------------------------------- 纯函数


def test_slice_router_logits():
    logits = torch.randn(4, 9)
    routed, gate = fold.slice_router_logits(logits, 8)
    assert routed.shape == (4, 8) and gate.shape == (4, 1)
    assert torch.equal(routed, logits[:, :8]) and torch.equal(gate, logits[:, 8:])


def test_append_shared_column_math():
    ids = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
    ws = torch.tensor([[0.3, 0.4], [0.5, 0.6]], dtype=torch.float32)
    logit = torch.tensor([[1.0], [-2.0]])
    ids2, ws2 = fold.append_shared_column(ids, ws, 777, logit, torch.float32)
    assert ids2.shape == (2, 3) and ws2.shape == (2, 3)
    assert (ids2[:, 2] == 777).all()
    assert torch.allclose(ws2[:, 2], torch.sigmoid(logit.squeeze(1)))
    assert torch.allclose(ws2[:, :2], ws) and (ids2[:, :2] == ids).all()


def test_append_shared_column_after_renorm_is_independent():
    # renorm 在 select_experts 内部完成;追加列不受其污染(语义红线)。
    ws = torch.tensor([[1.0, 3.0]]) / 4.0  # 已 renormalize 的 routed 权重
    logit = torch.tensor([[0.0]])
    _, ws2 = fold.append_shared_column(
        torch.tensor([[0, 1]]), ws, 2, logit, torch.float32
    )
    assert abs(ws2[0, 0].item() - 0.25) < 1e-6 and abs(ws2[0, 2].item() - 0.5) < 1e-6


def test_concat_shared_weights_shape_and_values():
    w13 = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    w2 = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    s13 = torch.full((1, 4, 3), 100.0)
    s2 = torch.full((1, 3, 4), 200.0)
    n13, n2 = fold.concat_shared_weights(w13, w2, s13, s2)
    assert n13.shape == (3, 4, 3) and n2.shape == (3, 3, 4)
    assert torch.equal(n13[2], s13[0]) and torch.equal(n2[2], s2[0])
    assert torch.equal(n13[:2], w13) and torch.equal(n2[:2], w2)
    with pytest.raises(ValueError):
        fold.concat_shared_weights(w13, w2, torch.full((1, 5, 3), 1.0), s2)


def test_extend_gate_weight():
    g = torch.ones(8, 5)
    row = torch.full((5,), 2.0)
    g2 = fold.extend_gate_weight(g, row)
    assert g2.shape == (9, 5)
    assert torch.equal(g2[8], row) and torch.equal(g2[:8], g)


# ---------------------------------------------------------------- 接线


class _FakeMoe:
    """模拟 AscendFusedMoE 的折叠相关面。

    direct=True 模拟 0.23.0 在役结构(层本体持 w13_weight/w2_weight,pwals 以层本体
    调用);direct=False 模拟非 0.23.0 分支(routed_experts 子模块持权重)。
    """

    def __init__(self, fused_gate=False, direct=False):
        self.routed_experts = type("R", (), {})()
        if direct:
            self.w13_weight = torch.nn.Parameter(torch.randn(8, 4, 3))
            self.w2_weight = torch.nn.Parameter(torch.randn(8, 3, 4))
        self.routed_experts.w13_weight = torch.nn.Parameter(torch.randn(8, 4, 3))
        self.routed_experts.w2_weight = torch.nn.Parameter(torch.randn(8, 3, 4))
        self._shared_experts = type("M", (), {})()
        gw13 = torch.randn(4, 3)
        grow = torch.randn(1, 3)
        if fused_gate:
            self._shared_experts.gate_up_proj = type(
                "L", (), {"weight": torch.nn.Parameter(torch.cat([gw13, grow]))}
            )()
            self._shared_experts._tileinfer_gate_fused = True
        else:
            self._shared_experts.gate_up_proj = type("L", (), {"weight": torch.nn.Parameter(gw13)})()
        self._shared_experts.down_proj = type("L", (), {"weight": torch.nn.Parameter(torch.randn(3, 4))})()
        self._shared_experts.expert_gate = type("G", (), {"weight": torch.nn.Parameter(grow)})()
        self.gate = type("Gt", (), {"weight_fp32": torch.nn.Parameter(torch.randn(8, 3))})()
        self.top_k = 8
        self._tileinfer_fold_done = False


def test_fold_weights_concat_and_expert_id(monkeypatch):
    monkeypatch.setattr(fold, "_bump_dispatcher_expert_num", lambda e: 1)
    monkeypatch.setattr(fold, "_bump_dispatcher_top_k", lambda k: 1)
    owner = _FakeMoe(fused_gate=True)
    assert fold._fold_weights(owner) is True
    # H73 形态剔 gate 行后 shared gate_up 仍 4 行;原 8 专家拼 1 → 9。
    assert owner._tileinfer_fold_expert_id == 8
    assert owner.routed_experts.w13_weight.shape[0] == 9
    assert owner.routed_experts.w2_weight.shape[0] == 9
    assert fold.stats()["fold_layers"] == 1


def test_fold_weights_direct_holder_023(monkeypatch):
    """0.23.0 形态:层本体持权重,折叠改写层本体,routed 子模块不动。"""
    monkeypatch.setattr(fold, "_bump_dispatcher_expert_num", lambda e: 1)
    monkeypatch.setattr(fold, "_bump_dispatcher_top_k", lambda k: 1)
    owner = _FakeMoe(direct=True)
    assert fold._weight_holder(owner) is owner
    assert fold._fold_weights(owner) is True
    assert owner.w13_weight.shape[0] == 9 and owner.w2_weight.shape[0] == 9
    assert owner.routed_experts.w13_weight.shape[0] == 8  # 非 0.23 载体不被误改


def test_fold_weights_aborts_when_dispatcher_not_bumped(monkeypatch):
    """E309:expert_num 未补升 ⇒ 越界 id 被丢弃,折叠必须判失败不生效。"""
    monkeypatch.setattr(fold, "_bump_dispatcher_expert_num", lambda e: 0)
    monkeypatch.setattr(fold, "_bump_dispatcher_top_k", lambda k: 1)
    owner = _FakeMoe(direct=True)
    assert fold._fold_weights(owner) is False
    assert fold.stats()["fold_skipped"] >= 1
    assert fold.stats()["fold_layers"] == 0
    # 权重已被替换但 owner 未置 done ⇒ 下次 PWAL 可重试;本断言锁定不置 done。
    assert getattr(owner, "_tileinfer_fold_done", False) is False


def test_moe_init_registers_layer_and_routed():
    fold._OWNER_REGISTRY.clear()
    calls = []
    orig = lambda self, *a, **k: calls.append(self)
    wrapped = fold._wrap_moe_init(orig)
    owner = _FakeMoe(direct=True)
    wrapped(owner)
    assert fold._OWNER_REGISTRY[id(owner)] is owner
    assert fold._OWNER_REGISTRY[id(owner.routed_experts)] is owner


def test_quant_pwal_wraps_before_orig(monkeypatch):
    monkeypatch.setattr(fold, "_bump_dispatcher_expert_num", lambda e: 1)
    monkeypatch.setattr(fold, "_bump_dispatcher_top_k", lambda k: 1)
    owner = _FakeMoe()
    fold._OWNER_REGISTRY[id(owner)] = owner
    calls = []
    qm = type("QM", (), {})()

    def orig_pwals(self, layer, *a, **k):
        # orig(=pad/NZ)执行时折叠必须已完成(前置时序契约)。
        calls.append(getattr(owner, "_tileinfer_fold_done", False))

    qm.process_weights_after_loading = fold._wrap_quant_pwals(orig_pwals)
    os.environ["TILEINFER_SHARED_EXPERT_FOLD"] = "1"
    qm.process_weights_after_loading(qm, owner)  # 0.23.0:pwals 以层本体调用
    assert calls == [True]
    assert owner.routed_experts.w13_weight.shape[0] == 9


def test_select_wrapper_slices_and_appends():
    owner = _FakeMoe()
    fold._fold_weights(owner)
    owner._tileinfer_fold_ext = True

    captured = {}

    def orig_select(*args, **kwargs):
        captured["logits"] = kwargs["router_logits"]
        captured["num_experts"] = kwargs["num_experts"]
        return torch.tensor([[0.9, 0.1]]), torch.tensor([[2, 5]])

    sel = fold._wrap_select_experts(orig_select)
    logits = torch.randn(1, 9)
    fold._CTX.owner = owner
    try:
        ws, ids = sel(hidden_states=torch.randn(1, 3), router_logits=logits, num_experts=8)
    finally:
        fold._CTX.owner = None
    assert captured["logits"].shape == (1, 8) and captured["num_experts"] == 8
    assert ids.shape == (1, 3) and int(ids[0, 2]) == 8
    assert ws.shape == (1, 3)


def test_select_wrapper_passthrough_without_ctx():
    def orig_select(*args, **kwargs):
        return torch.tensor([[1.0]]), torch.tensor([[3]])

    sel = fold._wrap_select_experts(orig_select)
    ws, ids = sel(router_logits=torch.randn(1, 8), num_experts=8)
    assert ids.shape == (1, 1)  # 无上下文(非折叠层)完全透传


def test_shared_forward_bypass_zero_and_ctx():
    owner = _FakeMoe()
    fold._fold_weights(owner)
    owner._tileinfer_fold_done = True  # 真实流程由 PWAL 包装置位
    seen = []

    def orig_impl(self, hidden, logits):
        seen.append(fold._wrap_select_experts.__name__ and getattr(fold._CTX, "owner", None))
        return hidden + 1

    wrapped = fold._wrap_shared_forward(orig_impl)
    os.environ["TILEINFER_SHARED_EXPERT_FOLD"] = "1"
    out = wrapped(owner, torch.zeros(2, 3), None)
    assert torch.equal(out, torch.ones(2, 3))
    assert seen == [owner]
    assert getattr(fold._CTX, "owner", None) is None  # 退出后上下文清理

    bypass = fold._wrap_shared_bypass(lambda self, h, ev: (_ for _ in ()).throw(AssertionError("orig called")))
    z = bypass(owner, torch.randn(2, 3), None)
    assert z.shape == (2, 3) and float(z.abs().sum()) == 0.0


def test_gate_extension_lazy_and_once():
    owner = _FakeMoe()
    fold._fold_weights(owner)
    fold._ensure_gate_extended(owner)
    assert owner._tileinfer_fold_ext is True
    assert owner.gate.weight_fp32.shape == (9, 3)
    first = owner.gate.weight_fp32
    fold._ensure_gate_extended(owner)
    assert owner.gate.weight_fp32 is first  # 二次调用不再更换对象(capture 安全)


def test_select_wrapper_plain_logits_gemv_fallback():
    """非 internal router(在役 Qwen3.6 形态):logits [M,E] 原样透传 + expert_gate
    GEMV 现算门控(E310:扩列判据改用 fold_expert_id,降级分支真正可达)。"""
    owner = _FakeMoe()
    fold._fold_weights(owner)  # expert_id = 8
    owner._tileinfer_fold_ext = False  # gate 未扩列

    calls = {}

    def gate_forward(hidden):
        calls["hidden_shape"] = tuple(hidden.shape)
        return (torch.tensor([[1.5]]), None)

    owner._shared_experts.expert_gate = gate_forward

    def orig_select(*args, **kwargs):
        assert tuple(kwargs["router_logits"].shape) == (1, 8)  # 原形态透传不切片
        return torch.tensor([[0.9, 0.1]]), torch.tensor([[2, 5]])

    sel = fold._wrap_select_experts(orig_select)
    fold._CTX.owner = owner
    try:
        ws, ids = sel(
            hidden_states=torch.randn(1, 3), router_logits=torch.randn(1, 8), num_experts=8
        )
    finally:
        fold._CTX.owner = None
    assert calls["hidden_shape"] == (1, 3)
    assert ids.shape == (1, 3) and int(ids[0, 2]) == 8
    expect = float(torch.sigmoid(torch.tensor(1.5)))
    assert abs(float(ws[0, 2]) - expect) < 1e-6
    assert fold.stats()["fold_forward"] == 1


def test_select_wrapper_passthrough_shape_mismatch():
    """列数既非 E 也非 E+1:透传且不计数(E310 双态判据的负例)。"""
    owner = _FakeMoe()
    fold._fold_weights(owner)

    def orig_select(*args, **kwargs):
        return torch.tensor([[1.0]]), torch.tensor([[3]])

    sel = fold._wrap_select_experts(orig_select)
    fold._CTX.owner = owner
    try:
        ws, ids = sel(router_logits=torch.randn(1, 5), num_experts=8)
    finally:
        fold._CTX.owner = None
    assert ids.shape == (1, 1)
    assert fold.stats()["fold_forward"] == 0


def test_install_idempotent_and_gates_off():
    assert fold.is_installed() is False
    r1 = fold.install()
    if r1:  # vllm_ascend 在役时:幂等且状态可查
        assert fold.is_installed() is True
        assert fold.install() is True


# ---------------------------------------------------------------- dispatcher top_k 补升(EZ0002)


def _fake_mcm(monkeypatch, top_k, num_experts_local=256):
    """向真实 moe_comm_method 单例表注入伪实现(CPU 可导入,单例表设备期才注册)。"""
    class _TD:
        def __init__(self):
            self.top_k = top_k
            self.num_experts_local = num_experts_local

    td = _TD()
    impl = type("I", (), {})()
    impl.token_dispatcher = td
    from vllm_ascend.ops.fused_moe import moe_comm_method as mcm

    monkeypatch.setattr(mcm, "_MoECommMethods", {"topk": impl})
    return td


def test_bump_dispatcher_top_k_math_and_idempotent(monkeypatch):
    td = _fake_mcm(monkeypatch, top_k=8)
    assert fold._bump_dispatcher_top_k(8) == 1
    assert td.top_k == 9
    # 幂等:guard 命中即成功返回,不再重复补升
    assert fold._bump_dispatcher_top_k(8) == 1
    assert td.top_k == 9


def test_bump_dispatcher_top_k_mismatch_noop(monkeypatch):
    td = _fake_mcm(monkeypatch, top_k=4)
    # 单例 top_k 与折叠层 k 不一致:不补升、返回 0(fail-closed)
    assert fold._bump_dispatcher_top_k(8) == 0
    assert td.top_k == 4
    # 失败不落 guard,修正后可重试
    assert fold._bump_dispatcher_top_k(4) == 1
    assert td.top_k == 5


def test_fold_weights_aborts_when_top_k_not_bumped(monkeypatch):
    """EZ0002(iter-213):top_k 未补升 ⇒ active_num 失配,折叠必须判失败。"""
    monkeypatch.setattr(fold, "_bump_dispatcher_expert_num", lambda e: 1)
    monkeypatch.setattr(fold, "_bump_dispatcher_top_k", lambda k: 0)
    owner = _FakeMoe()
    assert fold._fold_weights(owner) is False
    assert fold.stats()["fold_skipped"] == 1
    assert fold.stats()["fold_layers"] == 0
