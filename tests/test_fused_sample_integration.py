"""fused_sample 集成层守卫契约(纯 CPU,不编译内核)。

守卫是**唯一**决定是否接管的地方,任何一条出错都会导致:
- 漏接管 → 静默无收益(不易发现);
- 误接管 → 采样错误(严重)。

所以逐条锁死。
"""

from __future__ import annotations

import pytest
import torch

from tileinfer.integrations import fused_sample as integ

V = 248320  # 真实词表


class _MD:
    """最小采样元数据替身(模拟 vLLM 的 SamplingMetadata 字段)。"""

    def __init__(self, **kw):
        self.max_num_logprobs = None
        self.logprob_token_ids = []
        self.no_penalties = True
        self.bad_words_token_ids = None
        self.allowed_token_ids_mask = None
        self.logitsprocs = None
        self.temperature = torch.tensor([1.0])
        self.top_p = torch.tensor([0.9])
        self.top_k = torch.tensor([0])
        self.vocab_size = V
        for k, v in kw.items():
            setattr(self, k, v)


def _logits(batch=integ.MIN_BATCH_FOR_FUSED_SAMPLE, vocab=V):
    # 默认 batch 取门槛值:守卫要求 batch >= MIN_BATCH_FOR_FUSED_SAMPLE
    # (设备实测交叉点,低于它融合更慢),专测门槛的用例自己传小 batch。
    return torch.randn(batch, vocab, dtype=torch.float32)


# ------------------------------------------------------------ ops 开关


def test_ops_disabled_blocks(monkeypatch):
    monkeypatch.setattr(integ, "_ops_enabled", lambda: False)
    assert integ.fused_sample_applicable(_logits(), _MD(), False) is True  # 守卫本身不看 ops
    # 但 make_fused_forward 会检查 ops


def test_guard_basic_ok():
    assert integ.fused_sample_applicable(_logits(), _MD(), False) is True


# ------------------------------------------------------------ 逐条守卫


def test_rejects_predict_bonus():
    assert integ.fused_sample_applicable(_logits(), _MD(), True) is False


def test_rejects_non_2d():
    assert integ.fused_sample_applicable(torch.randn(V), _MD(), False) is False
    assert integ.fused_sample_applicable(torch.randn(2, 2, V), _MD(), False) is False


def test_accepts_low_precision_logits():
    """E367 根因①:引擎内 lm_head 直出 bf16,守卫必须放行(wrapper 内 cast fp32)。"""
    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        assert integ.fused_sample_applicable(_logits().to(dtype), _MD(), False) is True


def test_rejects_non_float_dtype():
    assert integ.fused_sample_applicable(_logits().double(), _MD(), False) is False


def test_rejects_non_contiguous():
    base = _logits(2)
    sliced = base[:, ::2]
    assert not sliced.is_contiguous()
    assert integ.fused_sample_applicable(sliced, _MD(), False) is False


@pytest.mark.parametrize("vocab", [1000, 8192, 16320, 16384])
def test_rejects_small_or_unaligned_vocab(vocab):
    """vocab 必须 ≥16384 且是 64 的倍数(batch 取门槛值,隔离 vocab 这一个变量)。"""
    logits = torch.randn(integ.MIN_BATCH_FOR_FUSED_SAMPLE, vocab, dtype=torch.float32)
    md = _MD(vocab_size=vocab)
    expected = vocab >= 16384 and vocab % 64 == 0
    assert integ.fused_sample_applicable(logits, md, False) is expected


def test_rejects_below_batch_threshold():
    """batch 低于门槛必须回退:设备实测该区间融合更慢(batch=8 慢 2.9×)。"""
    for batch in (1, 8, integ.MIN_BATCH_FOR_FUSED_SAMPLE - 1):
        assert integ.fused_sample_applicable(_logits(batch), _MD(), False) is False
    assert integ.fused_sample_applicable(
        _logits(integ.MIN_BATCH_FOR_FUSED_SAMPLE), _MD(), False
    ) is True


def test_rejects_logprobs_request():
    assert integ.fused_sample_applicable(_logits(), _MD(max_num_logprobs=5), False) is False
    assert integ.fused_sample_applicable(_logits(), _MD(logprob_token_ids=[1, 2]), False) is False


def test_rejects_penalties():
    assert integ.fused_sample_applicable(_logits(), _MD(no_penalties=False), False) is False


def test_rejects_bad_words():
    assert integ.fused_sample_applicable(_logits(), _MD(bad_words_token_ids=[7]), False) is False


def test_rejects_allowed_mask():
    mask = torch.zeros(2, V, dtype=torch.bool)
    assert integ.fused_sample_applicable(_logits(), _MD(allowed_token_ids_mask=mask), False) is False


def test_accepts_non_argmax_invariant_procs():
    """E367 根因②:LogitBias/MinTokens 处理器在 vLLM 初始化期即登记进
    non_argmax_invariant(与请求参数无关),恒非空——不得作为拒绝理由;
    语义改为 wrapper 内实际应用处理器(见 test_forward_applies_processors)。"""
    import types

    md = _MD(logitsprocs=types.SimpleNamespace(non_argmax_invariant=[object()]))
    assert integ.fused_sample_applicable(_logits(), md, False) is True


def test_forward_applies_processors(monkeypatch):
    """接管路径必须先 cast fp32 并应用 processors,再喂内核;失败回退用原始入参。"""
    monkeypatch.setattr(integ, "_ops_enabled", lambda: True)
    integ.reset_state()

    seen = {}

    class _Self:
        def apply_logits_processors(self, logits, md, pbt):
            seen["dtype"] = logits.dtype
            seen["applied"] = True
            return logits

    def fake_kernel(fp32_logits, temps, tp, tk, seed, out):
        seen["kernel_dtype"] = fp32_logits.dtype
        seen["kernel_batch"] = int(out.shape[0])
        out.fill_(7)

    monkeypatch.setattr(integ, "_get_kernel", lambda *a: fake_kernel)

    import types

    md = _MD(logitsprocs=types.SimpleNamespace(non_argmax_invariant=[object()]))
    bf16 = _logits().to(torch.bfloat16)
    fwd = integ.make_fused_forward(lambda self, *a, **k: pytest.fail("must not fall back"))
    out = fwd(_Self(), bf16, md, False, None)
    assert seen == {
        "dtype": torch.float32,
        "applied": True,
        "kernel_dtype": torch.float32,
        "kernel_batch": integ.MIN_BATCH_FOR_FUSED_SAMPLE,
    }
    assert integ.COUNTERS["fused_sample_calls"] == 1
    assert torch.all(out.sampled_token_ids == 7)


def test_forward_falls_back_if_processors_raise(monkeypatch):
    """processors 应用抛异常 → 计 fallbacks 并用**原始入参**回退原生 forward。"""
    monkeypatch.setattr(integ, "_ops_enabled", lambda: True)
    integ.reset_state()

    sentinel = object()
    orig_logits = {}

    class _Self:
        def apply_logits_processors(self, logits, md, pbt):
            raise RuntimeError("synthetic processor failure")

    def orig(self, logits, md, pbt=False, lmo=None):
        orig_logits["dtype"] = logits.dtype
        return sentinel

    import types

    fwd = integ.make_fused_forward(orig)
    bf16 = _logits().to(torch.bfloat16)
    md = _MD(logitsprocs=types.SimpleNamespace(non_argmax_invariant=[object()]))
    out = fwd(_Self(), bf16, md, False, None)
    assert out is sentinel
    assert orig_logits["dtype"] == torch.bfloat16  # 原始入参,未被 cast 污染
    assert integ.COUNTERS["fused_sample_fallbacks"] == 1
    assert integ.COUNTERS["fused_sample_calls"] == 0


def test_rejects_pure_greedy_batch():
    """全 greedy 交给 H36(它更轻),本内核不接管。"""
    md = _MD(top_p=torch.tensor([1.0]), top_k=torch.tensor([0]))
    assert integ.fused_sample_applicable(_logits(), md, False) is False


def test_accepts_top_p_only():
    md = _MD(top_p=torch.tensor([0.85]), top_k=torch.tensor([0]))
    assert integ.fused_sample_applicable(_logits(), md, False) is True


def test_accepts_top_k_only():
    md = _MD(top_p=torch.tensor([1.0]), top_k=torch.tensor([16]))
    assert integ.fused_sample_applicable(_logits(), md, False) is True


# ------------------------------------------------------------ 回退语义


def test_forward_falls_back_on_exception(monkeypatch):
    """内核抛异常时必须回退原 forward,并 bump 计数器——绝不抛出。"""
    monkeypatch.setattr(integ, "_ops_enabled", lambda: True)
    integ.reset_state()

    sentinel = object()

    def orig(self, logits, md, pbt=False, lmo=None):
        return sentinel

    def boom(*a, **k):
        raise RuntimeError("synthetic kernel failure")

    monkeypatch.setattr(integ, "_get_kernel", boom)

    import types

    fwd = integ.make_fused_forward(orig)
    out = fwd(types.SimpleNamespace(), _logits(), _MD(), False, None)
    assert out is sentinel
    assert integ.COUNTERS["fused_sample_fallbacks"] == 1
    assert integ.COUNTERS["fused_sample_calls"] == 0


def test_forward_skips_when_ops_off(monkeypatch):
    monkeypatch.setattr(integ, "_ops_enabled", lambda: False)
    integ.reset_state()
    sentinel = object()

    def orig(self, logits, md, pbt=False, lmo=None):
        return sentinel

    import types

    fwd = integ.make_fused_forward(orig)
    out = fwd(types.SimpleNamespace(), _logits(), _MD(), False, None)
    assert out is sentinel
    assert integ.COUNTERS["fused_sample_calls"] == 0
    assert integ.COUNTERS["fused_sample_fallbacks"] == 0


def test_forward_falls_back_when_guard_rejects(monkeypatch):
    monkeypatch.setattr(integ, "_ops_enabled", lambda: True)
    integ.reset_state()
    sentinel = object()

    def orig(self, logits, md, pbt=False, lmo=None):
        return sentinel

    import types

    fwd = integ.make_fused_forward(orig)
    # 全 greedy → 守卫拒绝 → 回退
    md = _MD(top_p=torch.tensor([1.0]), top_k=torch.tensor([0]))
    out = fwd(types.SimpleNamespace(), _logits(), md, False, None)
    assert out is sentinel
    assert integ.COUNTERS["fused_sample_calls"] == 0


def test_param_extraction_broadcasts_scalar():
    """单元素参数必须广播到 batch,不能形状不匹配。"""
    md = _MD(temperature=torch.tensor([0.7]), top_p=torch.tensor([0.8]), top_k=torch.tensor([5]))
    temps, tp, tk, seed = integ._extract_params(md, 4, torch.device("cpu"))
    assert temps.shape == (4,)
    assert float(temps[0]) == pytest.approx(0.7)
    assert tp.shape == (4,)
    assert tk.shape == (4,)
    assert seed.shape == (4,)


def test_param_extraction_clamps_zero_temperature():
    """temperature=0 表示 greedy;内核里 1/temp 会爆,必须钳到 1e-6。"""
    md = _MD(temperature=torch.tensor([0.0]))
    temps, _, _, _ = integ._extract_params(md, 2, torch.device("cpu"))
    assert float(temps[0]) == pytest.approx(1e-6)
    assert torch.isfinite(1.0 / temps).all()
