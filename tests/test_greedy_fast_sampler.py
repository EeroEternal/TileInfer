"""H36 sampler greedy 快路径契约测试(tileinfer.kernels.sampling.greedy)。

范围(CPU,不依赖引擎/vllm_ascend):
* 数值等价:守卫全满足时,包装 forward 与 vLLM 基类 Sampler.forward(bf16→fp32
  cast + argmax)采样 token id 逐位一致——含并列极值(同一值集 + 同 tie-break)
  与 fp32/bf16 两种 dtype;SamplerOutput 形状 [B,1]、int32;
* 守卫矩阵:9 条守卫任一不满足(predict_bonus_token / 非 all_greedy /
  max_num_logprobs / logprob_token_ids / penalties / bad_words /
  allowed_token_ids_mask / non_argmax_invariant 非空 / thinking_budget 追踪)
  均回退原 forward,且回退输出与纯基类输出一致;
* 开关:``TILEINFER_OPS`` 不含 ``greedy`` 时直通原实现;异常时计数 fallbacks
  并回退;``_OPS_ALL``/``ops_enabled("greedy")`` 回归;
* 计数器:命中 bump greedy_fast_calls,经 _stats_sink 同步。

设备/引擎权威验证链:H36 权威口径引擎 A/B(在役探针 greedy_fast_calls +
图模式端到端),见 HYPOTHESES H36。
"""

from __future__ import annotations

import types

import pytest
import torch
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler as BaseSampler

from tileinfer.kernels.sampling import greedy
from tileinfer.utils import is_npu_available

B, V = 8, 1024


def _md(**overrides) -> SamplingMetadata:
    """构造全守卫通过的 SamplingMetadata(all_greedy/无 logprobs/无惩罚/无 mask)。"""
    fields = dict(
        temperature=torch.zeros(B),
        all_greedy=True,
        all_random=False,
        top_p=None,
        top_k=None,
        generators={},
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.zeros(B),
        presence_penalties=torch.zeros(B),
        repetition_penalties=torch.zeros(B),
        output_token_ids=[[] for _ in range(B)],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=types.SimpleNamespace(non_argmax_invariant=[], argmax_invariant=[]),
    )
    fields.update(overrides)
    return SamplingMetadata(**fields)


def _sentinel_orig(*args, **kwargs):
    """回退哨兵:记录被调用并返回可识别标记。"""
    _sentinel_orig.called += 1
    _sentinel_orig.args = args
    return "ORIG"


_sentinel_orig.called = 0


@pytest.fixture(autouse=True)
def _clean():
    greedy.reset_state()
    _sentinel_orig.called = 0
    yield
    greedy.reset_state()


def _wrap(orig):
    return greedy.make_greedy_forward(orig)


def test_equivalence_matches_base_forward_bitwise():
    """快路径与基类 forward 逐位一致(bf16,含并列极值)。"""
    torch.manual_seed(7)
    logits = torch.randn(B, V, dtype=torch.bfloat16)
    # 人工并列极值:每行两个相等最大值 → 检验 tie-break 一致
    logits[:, 0] = 4.0
    logits[:, V // 2] = 4.0
    base = BaseSampler()
    fast = _wrap(BaseSampler.forward)
    out_base = BaseSampler.forward(base, logits, _md())
    out_fast = fast(base, logits, _md())
    assert out_fast is not None and out_fast != "ORIG"
    torch.testing.assert_close(
        out_fast.sampled_token_ids, out_base.sampled_token_ids
    )
    assert out_fast.sampled_token_ids.dtype == torch.int32
    assert out_fast.sampled_token_ids.shape == (B, 1)
    assert out_fast.logprobs_tensors is None
    assert greedy.COUNTERS["greedy_fast_calls"] == 1
    # 并列极值行为两次路径一致(首个最大索引)
    assert out_fast.sampled_token_ids[:, 0].tolist() == [0] * B


def test_equivalence_fp32_and_shapes():
    """fp32 logits 同样逐位一致;B=1 单请求形状正确。"""
    torch.manual_seed(11)
    logits = torch.randn(1, V, dtype=torch.float32)
    base = BaseSampler()
    fast = _wrap(BaseSampler.forward)
    out_base = BaseSampler.forward(base, logits, _md(temperature=torch.zeros(1)))
    out_fast = fast(base, logits, _md(temperature=torch.zeros(1)))
    torch.testing.assert_close(out_fast.sampled_token_ids, out_base.sampled_token_ids)


@pytest.mark.parametrize(
    "override",
    [
        {"all_greedy": False},
        {"max_num_logprobs": 5},
        {"logprob_token_ids": {0: [1, 2]}},
        {"no_penalties": False},
        {"bad_words_token_ids": {0: [[9, 9]]}},
        {"allowed_token_ids_mask": torch.zeros(B, V, dtype=torch.bool)},
        {
            "logitsprocs": types.SimpleNamespace(
                non_argmax_invariant=[object()], argmax_invariant=[]
            )
        },
    ],
)
def test_guard_violations_fall_back_to_orig(override):
    """任一守卫不满足 → 回退原 forward,不产生快路径计数。"""
    logits = torch.randn(B, V, dtype=torch.bfloat16)
    fast = _wrap(_sentinel_orig)
    out = fast(object(), logits, _md(**override))
    assert out == "ORIG"
    assert _sentinel_orig.called == 1
    assert greedy.COUNTERS["greedy_fast_calls"] == 0


def test_guard_predict_bonus_token_fall_back():
    logits = torch.randn(B, V, dtype=torch.bfloat16)
    fast = _wrap(_sentinel_orig)
    assert fast(object(), logits, _md(), True) == "ORIG"
    assert _sentinel_orig.called == 1
    assert greedy.COUNTERS["greedy_fast_calls"] == 0


def test_guard_skips_counted_with_reason():
    """守卫不满足(E402 可观测面):guard_skips 计数,原因逐条可诊断,契约不变。"""
    logits = torch.randn(B, V, dtype=torch.bfloat16)
    fast = _wrap(_sentinel_orig)
    assert fast(object(), logits, _md(all_greedy=False)) == "ORIG"
    assert greedy.COUNTERS["greedy_fast_guard_skips"] == 1
    assert greedy.COUNTERS["greedy_fast_calls"] == 0
    # 逐条守卫原因(与 greedy_fast_applicable 布尔契约单一事实源)
    cases = [
        (_md(), True, "predict_bonus_token"),
        (_md(all_greedy=False), False, "not_all_greedy"),
        (_md(max_num_logprobs=5), False, "logprobs_requested"),
        (_md(logprob_token_ids={0: [1]}), False, "logprob_token_ids"),
        (_md(no_penalties=False), False, "penalties_present"),
        (_md(bad_words_token_ids={0: [[9]]}), False, "bad_words"),
        (_md(allowed_token_ids_mask=torch.zeros(B, V, dtype=torch.bool)), False, "allowed_token_ids_mask"),
        (_md(logitsprocs=None), False, "non_argmax_invariant_procs"),
        (_md(logitsprocs=types.SimpleNamespace(non_argmax_invariant=[object()])), False, "non_argmax_invariant_procs"),
        (_md(), False, "logits_shape"),
    ]
    for md, bonus, reason in cases[:-1]:
        assert greedy._first_guard_fail(logits, md, bonus) == reason, reason
    # logits 形状守卫:3D 张量
    assert greedy._first_guard_fail(torch.randn(B, V, V), _md(), False) == "logits_shape"
    # 全满足 → None,且 applicable 布尔与之一致
    assert greedy._first_guard_fail(logits, _md(), False) is None
    assert greedy.greedy_fast_applicable(logits, _md(), False) is True


def test_guard_thinking_budget_tracked_fall_back():
    class _Holder:
        def has_tracked_requests(self):
            return True

    logits = torch.randn(B, V, dtype=torch.bfloat16)
    fast = _wrap(_sentinel_orig)
    assert fast(object(), logits, _md(thinking_budget_state_holder=_Holder())) == "ORIG"

    class _Empty:
        def has_tracked_requests(self):
            return False

    # 无追踪 holder 不阻塞快路径
    out = fast(BaseSampler(), logits, _md(thinking_budget_state_holder=_Empty()))
    assert out != "ORIG"
    assert greedy.COUNTERS["greedy_fast_calls"] == 1


def test_fallback_output_matches_pure_base():
    """守卫失败回退路径的输出与纯基类 forward 逐位一致(以 logprobs 场景为例)。"""
    torch.manual_seed(3)
    logits = torch.randn(B, V, dtype=torch.bfloat16)
    md = _md(max_num_logprobs=5)
    base = BaseSampler()
    fast = _wrap(BaseSampler.forward)
    out_pure = BaseSampler.forward(base, logits, md)
    out_fast = fast(base, logits, md)
    torch.testing.assert_close(
        out_fast.sampled_token_ids, out_pure.sampled_token_ids
    )
    torch.testing.assert_close(
        out_fast.logprobs_tensors.logprobs, out_pure.logprobs_tensors.logprobs
    )
    assert greedy.COUNTERS["greedy_fast_calls"] == 0


def test_ops_gate_off_passes_through(monkeypatch):
    """TILEINFER_OPS 不含 greedy → 直通原实现(开关每步读取)。"""
    monkeypatch.setenv("TILEINFER_OPS", "decode,prefill")
    logits = torch.randn(B, V, dtype=torch.bfloat16)
    fast = _wrap(_sentinel_orig)
    assert fast(object(), logits, _md()) == "ORIG"
    assert _sentinel_orig.called == 1
    assert greedy.COUNTERS["greedy_fast_calls"] == 0

    monkeypatch.setenv("TILEINFER_OPS", "greedy")
    out = fast(BaseSampler(), logits, _md())
    assert out != "ORIG"
    assert greedy.COUNTERS["greedy_fast_calls"] == 1


def test_exception_falls_back_and_counts():
    """快路径内部异常 → 计 fallbacks 并回退原实现,绝不抛出。"""
    logits = torch.randn(B, V, dtype=torch.bfloat16)
    orig_raise = _wrap(_sentinel_orig)

    def _boom(*a, **k):
        raise RuntimeError("boom")

    monkey_target = greedy
    saved = monkey_target.greedy_fast_forward
    monkey_target.greedy_fast_forward = _boom
    try:
        assert orig_raise(object(), logits, _md()) == "ORIG"
    finally:
        monkey_target.greedy_fast_forward = saved
    assert greedy.COUNTERS["greedy_fast_fallbacks"] == 1
    assert _sentinel_orig.called == 1


def test_stats_sink_sync():
    """_stats_sink 安装后计数同步进外部 stats(探针读取路径)。"""
    stats: dict = {}
    greedy._stats_sink = lambda name: stats.__setitem__(name, stats.get(name, 0) + 1)
    logits = torch.randn(B, V, dtype=torch.bfloat16)
    fast = _wrap(BaseSampler.forward)
    fast(BaseSampler(), logits, _md())
    assert stats.get("greedy_fast_calls") == 1


def test_ops_all_contains_greedy():
    """``_OPS_ALL``/``ops_enabled`` 对 greedy 开关的回归。"""
    from tileinfer.integrations.vllm_ascend import _OPS_ALL, ops_enabled

    assert "greedy" in _OPS_ALL
    monkey_enabled = is_npu_available()
    # ops_enabled 行为:未设置 → 全开;显式列表 → 精确匹配
    import os

    saved = os.environ.get("TILEINFER_OPS")
    try:
        os.environ.pop("TILEINFER_OPS", None)
        assert ops_enabled("greedy") is True
        os.environ["TILEINFER_OPS"] = "greedy"
        assert ops_enabled("greedy") is True
        os.environ["TILEINFER_OPS"] = "gdnsmall"
        assert ops_enabled("greedy") is False
    finally:
        if saved is None:
            os.environ.pop("TILEINFER_OPS", None)
        else:
            os.environ["TILEINFER_OPS"] = saved
    assert monkey_enabled is not None


@pytest.mark.skipif(not is_npu_available(), reason="需要 NPU:vllm_ascend 导入与设备 A/B")
def test_install_greedy_fast_sampler_wiring():
    """安装函数接线(设备侧):类级覆写生效、保留原实现引用、幂等。"""
    from vllm.v1.sample.sampler import Sampler as _base

    from tileinfer.integrations.vllm_ascend import install_greedy_fast_sampler
    from vllm_ascend.sample.sampler import AscendSampler

    assert install_greedy_fast_sampler() is True
    assert getattr(AscendSampler, "_tileinfer_greedy_fast_installed", False)
    assert AscendSampler._tileinfer_greedy_forward_orig is _base.forward
    assert AscendSampler.forward is not _base.forward
    assert install_greedy_fast_sampler() is True  # 幂等
