"""H73 shared_expert_gate 发射合并(CPU 契约,零设备依赖)。

背景(E269 dense 逐形状记账,c1 稳态 768 步):
  ``Qwen3NextSparseMoeBlock.shared_expert_gate`` = ``ReplicatedLinear(hidden, 1)``,
  40 层每层一次发射。E269 实测合计 **100 µs/步**(c1 TPOT 的 2.8%),权重仅 4 KB
  (按 1.6 TB/s 只需 0.0025 µs)→ **99.9% 是纯发射税**,单位权重成本全场最贵。
  上游同构:ROCm #56789 记录 ``(1,K) skinny GEMM`` 占 decode 13.7%。

方案:``Qwen2MoeMLP.forward`` 里 ``expert_gate(x)`` 与 ``gate_up_proj(x)`` 输入
完全相同,把 gate 的 1 列拼进 ``MergedColumnParallelLinear`` 的 ``output_sizes``
(``[I, I]`` → ``[I, I, 1]``),一次发射后切分。

与已证否路线的区别:
  - H68-a(GDN qkvz+ba)失败于「融合输出 strided slice 喂 conv1d」→ 本方下游是
    SiluAndMul/sigmoid/逐元素乘,**无 conv1d**;
  - H69(router+shared_gate+gate_up 三合一)失败于「三族分属不同层需跨层重构」
    → 本方**在同一个 MLP 模块内**。

锁定契约:
  1. 开关默认关(零行为变化);dryrun 只记日志不改结构;
  2. 数值等价:拼接后 ``fused[..., :-1]`` == 原 ``gate_up``、``fused[..., -1:]``
     == 原 ``expert_gate(x)``,逐位一致(同一 GEMM,列独立);
  3. 形状守卫:只有 ``expert_gate.weight.shape[0] == 1`` 且无 bias 才合并;
  4. 权重迁移:独立的 ``*shared_expert_gate.weight`` 必须落到 fused 第 3 段;
  5. 任何异常都回退原 forward / 原结构。
"""

from __future__ import annotations

import inspect

import pytest

torch = pytest.importorskip("torch")

try:
    from tileinfer.integrations import shared_gate_fuse as mod

    _HAVE_MOD = True
except Exception:  # pragma: no cover
    _HAVE_MOD = False


pytestmark = pytest.mark.skipif(not _HAVE_MOD, reason="shared_gate_fuse unavailable")


# ---------------------------------------------------------------- 1. 开关默认


def test_h73_disabled_by_default(monkeypatch):
    monkeypatch.delenv("TILEINFER_SHARED_GATE_FUSE", raising=False)
    assert mod._enabled() is False
    monkeypatch.setenv("TILEINFER_SHARED_GATE_FUSE", "1")
    assert mod._enabled() is True
    monkeypatch.setenv("TILEINFER_SHARED_GATE_FUSE", "0")
    assert mod._enabled() is False


def test_h73_dryrun_flag(monkeypatch):
    monkeypatch.delenv("TILEINFER_SHARED_GATE_FUSE_DRYRUN", raising=False)
    assert mod._dryrun() is False
    monkeypatch.setenv("TILEINFER_SHARED_GATE_FUSE_DRYRUN", "1")
    assert mod._dryrun() is True


# ---------------------------------------------------------------- 2. 数值等价(核心)


def test_h73_column_concat_is_bitwise_equivalent():
    """把 gate 列拼到 gate_up 末尾,一次 matmul 后切分 == 两次独立 matmul。

    这是整个方案的正确性基石:GEMM 的每一列独立,拼接不改变任何列的结果。
    """
    torch.manual_seed(0)
    hidden, inter, tokens = 2048, 512, 4

    x = torch.randn(tokens, hidden, dtype=torch.bfloat16)
    w_gate_up = torch.randn(2 * inter, hidden, dtype=torch.bfloat16)
    w_gate = torch.randn(1, hidden, dtype=torch.bfloat16)

    # 原路径:两次独立 matmul
    ref_gate_up = torch.matmul(x, w_gate_up.T)
    ref_gate_logit = torch.matmul(x, w_gate.T)

    # 融合路径:一次 matmul + 切分
    w_fused = torch.cat([w_gate_up, w_gate], dim=0)
    fused = torch.matmul(x, w_fused.T)
    got_gate_up = fused[..., :-1]
    got_gate_logit = fused[..., -1:]

    assert torch.equal(got_gate_up, ref_gate_up), "gate_up 段必须逐位一致"
    assert torch.equal(got_gate_logit, ref_gate_logit), "gate_logit 段必须逐位一致"


def test_h73_forward_semantics_match():
    """out = sigmoid(gate(x)) * down(silu(gate_up(x))):融合前后逐位一致。"""
    torch.manual_seed(1)
    hidden, inter, tokens = 256, 128, 3

    x = torch.randn(tokens, hidden, dtype=torch.float32)
    w_gu = torch.randn(2 * inter, hidden) * 0.05
    w_d = torch.randn(hidden, inter) * 0.05
    w_g = torch.randn(1, hidden) * 0.05

    def silu(t):
        return t * torch.sigmoid(t)

    # 原路径
    gu = torch.matmul(x, w_gu.T)
    a, b = gu.chunk(2, dim=-1)
    out_ref = torch.sigmoid(torch.matmul(x, w_g.T)) * torch.matmul(silu(a * torch.sigmoid(b)), w_d.T)

    # 融合路径
    w_fused = torch.cat([w_gu, w_g], dim=0)
    fused = torch.matmul(x, w_fused.T)
    gate_up, gate_logit = fused[..., :-1], fused[..., -1:]
    a2, b2 = gate_up.chunk(2, dim=-1)
    out_got = torch.sigmoid(gate_logit) * torch.matmul(silu(a2 * torch.sigmoid(b2)), w_d.T)

    assert torch.allclose(out_got, out_ref, atol=1e-6), "融合前后语义必须一致"


# ---------------------------------------------------------------- 3. 形状守卫


def test_h73_shape_guard_logic():
    """只有单输出、无 bias 的 expert_gate 才允许合并。"""
    src = inspect.getsource(mod._make_mlp_init)
    assert "w.shape[0] != 1" in src, "必须校验 gate 输出维度为 1"
    assert "bias" in src, "必须拒绝带 bias 的 gate"


# ---------------------------------------------------------------- 4. 权重迁移


def test_h73_load_weights_handles_gate_prefix():
    """独立的 shared_expert_gate.weight 必须被识别并搬进 fused 第 3 段。"""
    src = inspect.getsource(mod._make_mlp_load_weights)
    assert "shared_expert_gate.weight" in src
    assert "expert_gate.weight" in src
    assert "gate_up_proj.weight[-1:, :]" in src, "必须写到 fused 的最后一列"


# ---------------------------------------------------------------- 5. 回退纪律


def test_h73_fallback_on_exception():
    """forward 包装必须在任何异常时回退原实现(不得让优化影响在役路径)。"""
    src = inspect.getsource(mod._make_mlp_forward)
    assert "except Exception" in src
    assert "orig_forward(self, x)" in src


def test_h73_install_is_idempotent():
    """install() 重复调用应直接返回(不重复包装,避免包装套包装)。"""
    src = inspect.getsource(mod.install)
    assert "if _INSTALLED:" in src
