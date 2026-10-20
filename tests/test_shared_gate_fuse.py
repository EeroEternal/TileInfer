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


def test_h73_bind_must_happen_before_nz_conversion():
    """绑定必须在权重转 FRACTAL_NZ **之前**完成。

    2026-10-20 设备实测:惰性在首次 forward 绑定,此时 vllm-ascend 已把
    ``gate_up_proj.weight`` 转成 NZ(ndim != 2),``[-1:, :]`` 不再是"最后一行",
    就地拷贝直接写坏内存 → EngineCore 静默死亡。dryrun(不改结构)臂则 5/5 正常,
    证明崩溃来自结构改造而非融合算术。
    """
    install_src = inspect.getsource(mod.install)
    assert "_patch_linear_pwal" in install_src, "必须在 install 期挂 NZ 前钩子"

    pwal_src = inspect.getsource(mod._patch_linear_pwal)
    assert "process_weights_after_loading" in pwal_src
    assert "_bind_for_linear" in pwal_src

    # 兜底路径必须拒绝在 NZ(ndim != 2)张量上切片
    bind_src = inspect.getsource(mod._bind_gate_row)
    assert "ndim != 2" in bind_src, "NZ 张量上绝不能按行切片拷贝"

    # 无 NZ 时正常绑定,且只绑一次
    torch.manual_seed(5)
    hidden, inter = 8, 4

    class _Gate(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(1, hidden) * 0.05)

    holder = torch.nn.Module()
    holder.expert_gate = _Gate()
    holder.gate_up_proj = torch.nn.Module()
    holder.gate_up_proj.weight = torch.nn.Parameter(torch.zeros(2 * inter + 1, hidden))
    holder._tileinfer_gate_fused = True
    holder._tileinfer_gate_bound = False
    mod._FUSED_MLPS.append(holder)
    try:
        mod._bind_for_linear(holder.gate_up_proj)
        assert holder._tileinfer_gate_bound is True
        assert torch.equal(
            holder.gate_up_proj.weight.data[-1:, :], holder.expert_gate.weight.data
        )
        holder.gate_up_proj.weight.data[-1:, :].zero_()
        mod._bind_for_linear(holder.gate_up_proj)
        assert torch.count_nonzero(holder.gate_up_proj.weight.data[-1:, :]) == 0, "不得重复拷贝"
    finally:
        mod._FUSED_MLPS.remove(holder)

    # NZ 形态(ndim=4)必须拒绝兜底绑定
    class _Nz(torch.nn.Module):
        pass

    nz = _Nz()
    nz.weight = torch.nn.Parameter(torch.zeros(1, 1, 16, 16))
    nz_holder = torch.nn.Module()
    nz_holder.expert_gate = _Gate()
    nz_holder.gate_up_proj = nz
    nz_holder._tileinfer_gate_fused = True
    nz_holder._tileinfer_gate_bound = False
    assert mod._bind_gate_row(nz_holder) is False, "NZ 张量必须拒绝绑定"


def test_h73_install_must_not_require_load_weights():
    """回归锁定:install() 不得无条件访问 ``Qwen2MoeMLP.load_weights``。

    旧实现直接取该属性;本栈类无此方法 → AttributeError 被上层吞成 warning,
    融合**从未安装**(2026-10-20 设备 serve 日志实证)。这使 H73 首次 A/B 作废。
    """
    src = inspect.getsource(mod.install)
    code = "\n".join(
        line for line in src.splitlines() if not line.lstrip().startswith("#")
    )
    assert "Qwen2MoeMLP.load_weights =" not in code, (
        "install() 不得依赖 Qwen2MoeMLP.load_weights(本栈不存在该方法)"
    )


# ---------------------------------------------------------------- 5. 回退纪律


def test_h73_fallback_on_exception():
    """forward 包装必须在任何异常时回退原实现(不得让优化影响在役路径)。"""
    src = inspect.getsource(mod._make_mlp_forward)
    assert "except Exception" in src
    assert "orig_forward(self, x)" in src


def test_h73_fused_slice_must_be_contiguous():
    """``fused[..., :-1]`` 的行 stride 是 ``2I+1``(非连续),不得直接喂 vendor 激活。

    H68-a 的教训:融合输出以 strided slice 喂 AscendC 算子会走慢路径,把收益吃光。
    这里必须显式 ``contiguous()`` 后再进 ``SiluAndMul``(npu_swiglu)。
    """
    src = inspect.getsource(mod._make_mlp_forward)
    assert "fused[..., :-1].contiguous()" in src, "融合切分后必须连续化再喂 npu_swiglu"


def test_h73_must_patch_ascend_shared_part2():
    """shared expert 在 vllm-ascend 里走 part1/part2,**不调用** Qwen2MoeMLP.forward。

    只挂 forward 时 part2 会把融合后的 2I+1 列直接喂 ``npu_swiglu``,实测抛
    ``Invalid_Argument_Tensor_Shape [2048, 1025]``(split 轴必须是偶数)。
    因此必须同时包装 ``AscendFusedMoE._shared_experts_part2``。
    """
    src = inspect.getsource(mod._make_part2)
    assert "_shared_experts_part2" in src
    assert "shared_gate_up[..., :-1].contiguous()" in src

    install_src = inspect.getsource(mod.install)
    assert "_patch_ascend_shared_parts" in install_src, "install() 必须挂 part2 包装"


def test_h73_install_is_idempotent():
    """install() 重复调用应直接返回(不重复包装,避免包装套包装)。"""
    src = inspect.getsource(mod.install)
    assert "if _INSTALLED:" in src
