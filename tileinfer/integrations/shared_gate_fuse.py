"""H73 shared_expert_gate 发射合并(opt-in):把 hidden→1 的瘦 GEMV 并进 gate_up。

背景(E269 dense 逐形状记账,c1 稳态 768 步):
  Qwen3NextSparseMoeBlock 的 ``shared_expert_gate`` 是 ``ReplicatedLinear(hidden, 1)``,
  40 层每层一次发射。E269 实测 ``shared_gate`` 合计 **100 µs/步**(占 c1 TPOT 2.8%),
  而它的权重只有 ``2048 × 1 × 2B = 4 KB``——按 1.6 TB/s 峰值只需 0.0025 µs。
  **99.9% 是纯发射税**,单位权重成本是全部 GEMM 里最贵的(2.5 µs / 4 KB)。

  上游同构先例:ROCm 的 ``#56789`` 记录 ``(1,K) skinny GEMM`` 占 decode step 13.7%,
  当时的本地判定是"950PR 无 skinny 缺失问题,零行动"——但 E269 的实测数字
  (effBW≈0.0)推翻了这条判定。

方案:
  ``Qwen2MoeMLP.forward`` 里 ``expert_gate(x)`` 与 ``gate_up_proj(x)`` **输入完全相同**,
  因此把 gate 的 1 列权重拼到 gate_up 的列维(``MergedColumnParallelLinear`` 的
  ``output_sizes`` 从 ``[I, I]`` 扩成 ``[I, I, 1]``),一次发射得到
  ``[gate_act | up | gate_logit]``,再按原语义切分。

  与已证否的两条路线的关系:
  - **H68-a**(GDN qkvz+ba):失败原因是融合输出以 strided slice 喂 conv1d 走了
    AscendC 慢路径。本方案的下游是 ``SiluAndMul``/``sigmoid``/逐元素乘,
    **没有 conv1d**,不触发该慢路径。
  - **H69**(router+shared_gate+shared_gate_up 三合一):失败原因是三族分属不同层
    (router 在块 forward,shared 族在 FusedMoE runner 深处),需跨层重构;且
    shared 族已有 separate-stream overlap。本方案**在同一个 MLP 模块内**,
    只合并 ``expert_gate`` 与 ``gate_up_proj`` 这一对,不跨边界。

语义:
  ``out = sigmoid(expert_gate(x)) * down(silu(gate_up(x)))`` → 权重拼接后
  ``g = fused(x)``,``gate_up = g[..., :-1]``、``gate_logit = g[..., -1:]``,
  算术逐位一致(同一 GEMM,只是 N 多了 1 列;列独立,无跨列依赖)。

开关:
  ``TILEINFER_SHARED_GATE_FUSE=1``  启用(默认关 = 零行为变化)
  ``TILEINFER_SHARED_GATE_FUSE_DRYRUN=1``  只记日志不实际改权重(部署前自检)

注意:权重拼接发生在 ``load_weights`` 之前对模块结构的改造,因此本模块必须在
**模型构建期**生效(与 gdnproj 的 install 时序不同)。install() 挂在校准点
``Qwen2MoeMLP.__init__`` 的包装上,由 backend 的 install 触发。
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_INSTALLED = False
_STATS: dict[str, int] = {
    "shared_gate_fuse_modules": 0,  # 被改造成三段 gate_up 的 MLP 数
    "shared_gate_fuse_forward": 0,  # 走融合 forward 的次数
    "shared_gate_fuse_skipped": 0,  # 因条件不满足而跳过(保留原路径)的次数
}


def _enabled() -> bool:
    return os.environ.get("TILEINFER_SHARED_GATE_FUSE", "").strip() not in (
        "",
        "0",
        "false",
        "False",
    )


def _dryrun() -> bool:
    return os.environ.get("TILEINFER_SHARED_GATE_FUSE_DRYRUN", "").strip() not in (
        "",
        "0",
        "false",
        "False",
    )


def stats() -> dict[str, int]:
    return dict(_STATS)


def is_installed() -> bool:
    return _INSTALLED


def _make_mlp_init(orig_init):
    """包装 ``Qwen2MoeMLP.__init__``:有多余的 expert_gate 时把列并进 gate_up。

    注意:改结构后**必须**同时改 ``load_weights`` 的映射(见 :func:`install`),
    否则 checkpoint 里独立的 ``shared_expert_gate.weight`` 无处可去。
    """

    def __init__(self, hidden_size, intermediate_size, hidden_act, *args, **kwargs):
        expert_gate = kwargs.get("expert_gate", None)
        orig_init(self, hidden_size, intermediate_size, hidden_act, *args, **kwargs)
        if not _enabled() or expert_gate is None:
            return
        try:
            from vllm.model_executor.layers.linear import MergedColumnParallelLinear

            # 只在形状吻合(单输出 gate)时合并;多输出或带 bias 的形态保持原样。
            w = getattr(expert_gate, "weight", None)
            if w is None or w.shape[0] != 1 or getattr(expert_gate, "bias", None) is not None:
                _STATS["shared_gate_fuse_skipped"] += 1
                return
            if _dryrun():
                logger.info(
                    "H73 dryrun: would fuse shared_expert_gate(hidden=%d) into gate_up",
                    hidden_size,
                )
                return

            qc = kwargs.get("quant_config", None)
            sp = kwargs.get("is_sequence_parallel", False)
            prefix = kwargs.get("prefix", "")
            fused = MergedColumnParallelLinear(
                hidden_size,
                [intermediate_size] * 2 + [1],
                bias=False,
                quant_config=qc,
                disable_tp=sp,
                prefix=f"{prefix}.gate_up_proj",
            )
            self.gate_up_proj = fused
            # 保留原模块:load_weights 用它接住 `...shared_expert_gate.weight`,
            # 再搬进 fused 的第 3 段(见 _copy_gate_into_fused)。
            self.expert_gate = expert_gate
            self._tileinfer_gate_fused = True
            self._tileinfer_gate_hidden = hidden_size
            _STATS["shared_gate_fuse_modules"] += 1
        except Exception:  # pragma: no cover - 任何异常都退回原结构
            _STATS["shared_gate_fuse_skipped"] += 1
            logger.warning("H73: shared gate fuse skipped for %s", kwargs.get("prefix", "?"))

    return __init__


def _make_mlp_forward(orig_forward):
    """包装 ``Qwen2MoeMLP.forward``:融合形态下切分一次 GEMM 的结果。"""

    def forward(self, x):
        if not getattr(self, "_tileinfer_gate_fused", False):
            return orig_forward(self, x)
        try:
            import torch.nn.functional as F

            fused, _ = self.gate_up_proj(x)  # [..., 2I + 1]
            gate_up = fused[..., :-1]
            gate_logit = fused[..., -1:]
            out = self.act_fn(gate_up)
            out, _ = self.down_proj(out)
            out = F.sigmoid(gate_logit) * out
            _STATS["shared_gate_fuse_forward"] += 1
            return out
        except Exception:  # pragma: no cover
            _STATS["shared_gate_fuse_skipped"] += 1
            return orig_forward(self, x)

    return forward


def _make_mlp_load_weights(orig_load_weights):
    """包装 ``Qwen2MoeMLP.load_weights``:接受独立的 ``shared_expert_gate.weight``。

    原实现只认 ``gate_up_proj`` / ``down_proj`` 两个前缀;融合后 gate 的权重
    仍以原前缀出现在 checkpoint 里,这里把它**暂存**下来,等 load 结束时
    由 :func:`_copy_gate_into_fused` 搬进 fused 的第 3 段。
    """

    def load_weights(self, weights):
        pending_gate = None
        rest = []
        for name, w in weights:
            if name.endswith("expert_gate.weight") or name.endswith("shared_expert_gate.weight"):
                pending_gate = w
            else:
                rest.append((name, w))
        result = orig_load_weights(self, rest) if rest else None
        if pending_gate is not None and getattr(self, "_tileinfer_gate_fused", False):
            try:
                import torch

                with torch.no_grad():
                    self.gate_up_proj.weight[-1:, :].copy_(
                        pending_gate.reshape(1, -1).to(self.gate_up_proj.weight.dtype)
                    )
            except Exception:  # pragma: no cover
                logger.warning("H73: failed to place fused gate column")
        return result

    return load_weights


def install() -> bool:
    """在模型构建前挂包装。开关在每步读,装后 ``=0`` 仍零行为变化。"""
    global _INSTALLED
    if _INSTALLED:
        return True
    try:
        from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP
    except Exception:
        logger.warning("H73 not installed: qwen2_moe unavailable")
        return False

    Qwen2MoeMLP.__init__ = _make_mlp_init(Qwen2MoeMLP.__init__)
    Qwen2MoeMLP.forward = _make_mlp_forward(Qwen2MoeMLP.forward)
    Qwen2MoeMLP.load_weights = _make_mlp_load_weights(Qwen2MoeMLP.load_weights)
    _INSTALLED = True
    logger.info(
        "TileInfer shared_gate fuse installed (H73, governed by TILEINFER_SHARED_GATE_FUSE; "
        "dryrun=%s)", _dryrun(),
    )
    return True

