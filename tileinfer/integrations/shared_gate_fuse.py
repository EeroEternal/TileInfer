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

import inspect
import logging
import os

logger = logging.getLogger(__name__)

_INSTALLED = False
_STATS: dict[str, int] = {
    "shared_gate_fuse_modules": 0,  # 被改造成三段 gate_up 的 MLP 数
    "shared_gate_fuse_forward": 0,  # 走融合 forward 的次数
    "shared_gate_fuse_skipped": 0,  # 因条件不满足而跳过(保留原路径)的次数
    "shared_gate_fuse_bound": 0,  # 已把独立 gate 权重搬进 fused 末行的模块数
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

    权重落点见 :func:`_patch_linear_pwal` —— 本栈 ``Qwen2MoeMLP`` **没有**
    ``load_weights``,checkpoint 里独立的 ``shared_expert_gate.weight`` 由模型级
    ``named_parameters()`` 映射到保留的 ``expert_gate`` 模块上,再在 NZ 转换前
    搬进 fused 末行。
    """

    # `Qwen2MoeMLP.__init__` 的签名:vLLM 目前全用关键字调用,但上游随时可能改成
    # 位置传参;只读 kwargs 会在那种情况下**静默漏融合**(不报错、不生效)。
    # 这里按实际签名 bind 取值,两种调用方式都正确。
    _sig = inspect.signature(orig_init)

    def __init__(self, *args, **kwargs):
        bound = _sig.bind(self, *args, **kwargs)
        bound.apply_defaults()
        expert_gate = bound.arguments.get("expert_gate", None)
        hidden_size = bound.arguments.get("hidden_size", 0)
        orig_init(self, *args, **kwargs)
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

            qc = bound.arguments.get("quant_config", None)
            sp = bound.arguments.get("is_sequence_parallel", False)
            prefix = bound.arguments.get("prefix", "")
            intermediate_size = bound.arguments["intermediate_size"]
            fused = MergedColumnParallelLinear(
                hidden_size,
                [intermediate_size] * 2 + [1],
                bias=False,
                quant_config=qc,
                disable_tp=sp,
                prefix=f"{prefix}.gate_up_proj",
            )
            self.gate_up_proj = fused
            # 保留原模块:checkpoint 里的 `...mlp.shared_expert_gate.weight` 仍按原
            # 参数名落到这里,由 `_bind_one_layer` 在 NZ 转换前搬进 fused 末行。
            self.expert_gate = expert_gate
            self._tileinfer_gate_fused = True
            self._tileinfer_gate_bound = False
            self._tileinfer_gate_hidden = hidden_size
            _STATS["shared_gate_fuse_modules"] += 1
            _FUSED_MLPS.append(self)
            # 惰性挂 part2:此刻模型正在构建,fused_moe 必然已导入且注册完成;
            # 在插件 install() 早期主动 import 它会打乱注册顺序。
            _patch_ascend_shared_parts()
        except Exception:  # pragma: no cover - 任何异常都退回原结构
            _STATS["shared_gate_fuse_skipped"] += 1
            logger.warning("H73: shared gate fuse skipped for %s", kwargs.get("prefix", "?"))

    return __init__


def _bind_gate_row(module) -> bool:
    """查询/兜底绑定:仅在权重仍是 2-D ND 时才允许就地拷贝。

    **正常路径是 NZ 转换前**(见 :func:`_patch_linear_pwal`)。这里只处理
    "钩子没触发"的兜底:一旦权重已被转成 FRACTAL_NZ(ndim != 2),
    按 ``[-1:, :]`` 切片已不是"最后一行",绝不能拷贝,直接返回 False
    让 forward 走原实现(数值仍正确,只是不省那一次发射)。
    """
    if getattr(module, "_tileinfer_gate_bound", False):
        return True
    w = getattr(getattr(module, "gate_up_proj", None), "weight", None)
    if w is None or w.data.ndim != 2:
        return False
    _bind_one_layer(module)
    return bool(getattr(module, "_tileinfer_gate_bound", False))


def _make_mlp_forward(orig_forward):
    """包装 ``Qwen2MoeMLP.forward``:融合形态下切分一次 GEMM 的结果。"""

    def forward(self, x):
        if not getattr(self, "_tileinfer_gate_fused", False):
            return orig_forward(self, x)
        try:
            import torch.nn.functional as F

            if not _bind_gate_row(self):
                return orig_forward(self, x)
            fused, _ = self.gate_up_proj(x)  # [..., 2I + 1]
            # `fused[..., :-1]` 的行 stride 是 2I+1 → 非连续。vendor 的
            # `npu_swiglu` 是 AscendC 算子,非连续输入会走慢路径(H68-a 的同一坑)。
            # decode 时行数只有 1-8,这里的连续化是 KB 级 D2D 拷贝,远小于省下的
            # 一次瘦 GEMV 发射(100 µs/步)。
            gate_up = fused[..., :-1].contiguous()
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


def _make_part2(orig_part2):
    """包装 ``AscendFusedMoE._shared_experts_part2``。

    这是本栈 shared expert 的**实际执行路径**(vllm-ascend 把共享专家拆成
    part1=gate_up GEMM、part2=激活+down_proj+gate 三步,以支持流水 overlap),
    它**不调用** ``Qwen2MoeMLP.forward``。只挂 forward 会让融合后的 2I+1 列
    直接进 ``npu_swiglu`` → 非法形状(实测 ``[2048, 1025]``,EZ0009)。
    """

    def _shared_experts_part2(self, hidden_states, shared_gate_up):
        mlp = getattr(self, "_shared_experts", None)
        if not getattr(mlp, "_tileinfer_gate_fused", False):
            return orig_part2(self, hidden_states, shared_gate_up)
        try:
            import torch.nn.functional as F

            if not _bind_gate_row(mlp):
                return orig_part2(self, hidden_states, shared_gate_up)
            # 非连续切片不得直接喂 AscendC 激活(见 _make_mlp_forward 的同款注释)。
            gate_up = shared_gate_up[..., :-1].contiguous()
            gate_logit = shared_gate_up[..., -1:]
            shared_act = mlp.act_fn(gate_up)
            out, _ = mlp.down_proj(shared_act)
            out = F.sigmoid(gate_logit) * out
            _STATS["shared_gate_fuse_forward"] += 1
            return out
        except Exception:  # pragma: no cover
            _STATS["shared_gate_fuse_skipped"] += 1
            return orig_part2(self, hidden_states, shared_gate_up)

    return _shared_experts_part2


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
    # 注意:本栈的 `Qwen2MoeMLP` **没有** `load_weights`(权重由模型级
    # `stacked_params_mapping` + `named_parameters()` 映射)。旧实现在这里无条件
    # 访问 `Qwen2MoeMLP.load_weights` 会抛 AttributeError,而 install() 的调用方
    # 把异常吞成一条 warning —— 结果融合从未安装(H73 首次设备 A/B 由此作废)。
    _patch_linear_pwal()
    _patch_ascend_shared_parts()
    _INSTALLED = True
    logger.info(
        "TileInfer shared_gate fuse installed (H73, governed by TILEINFER_SHARED_GATE_FUSE; "
        "dryrun=%s)", _dryrun(),
    )
    return True


def _bind_one_layer(module) -> None:
    """把某层 fused gate_up 的末行填成独立 gate 的权重(NZ 转换前调用)。"""
    if not getattr(module, "_tileinfer_gate_fused", False):
        return
    if getattr(module, "_tileinfer_gate_bound", False):
        return
    try:
        import torch

        gate_w = module.expert_gate.weight
        with torch.no_grad():
            module.gate_up_proj.weight[-1:, :].copy_(gate_w.reshape(1, -1))
        module._tileinfer_gate_bound = True
        _STATS["shared_gate_fuse_bound"] += 1
    except Exception:  # pragma: no cover
        logger.warning("H73: failed to bind fused gate row")


def _patch_linear_pwal() -> None:
    """在 ``UnquantizedLinearMethod.process_weights_after_loading`` 入口完成绑定。

    vllm-ascend 的 ``AscendUnquantizedLinearMethod`` 在此方法里把权重转 FRACTAL_NZ;
    绑定必须在其**之前**发生,否则按 ``[-1:, :]`` 切片已不再是"最后一行"。
    """
    try:
        from vllm.model_executor.layers.linear import UnquantizedLinearMethod

        orig = UnquantizedLinearMethod.process_weights_after_loading
        if getattr(orig, "_tileinfer_wrapped", False):
            return

        def process_weights_after_loading(self, layer, *args, **kwargs):
            # layer 是 MergedColumnParallelLinear;其父模块是 Qwen2MoeMLP。
            # 这里直接在该 linear 上注册钩子拿不到父模块,故用反向遍历:
            # 绑定只按 "本层 gate_up 是否已融合" 判断,与父模块无关。
            _bind_for_linear(layer)
            return orig(self, layer, *args, **kwargs)

        process_weights_after_loading._tileinfer_wrapped = True
        UnquantizedLinearMethod.process_weights_after_loading = process_weights_after_loading
    except Exception:  # pragma: no cover - 优化模块不得影响在役路径
        logger.warning("H73: failed to patch linear process_weights_after_loading")


_FUSED_MLPS: list = []


def _bind_for_linear(layer) -> None:
    """若 ``layer`` 是某个融合 MLP 的 gate_up,则在其 NZ 转换前完成绑定。"""
    for mlp in _FUSED_MLPS:
        if getattr(mlp, "gate_up_proj", None) is layer:
            _bind_one_layer(mlp)
            return


def _patch_ascend_shared_parts() -> None:
    """给 vllm-ascend 的 ``AscendFusedMoE`` 挂 part2 包装(shared expert 的真路径)。

    两个实现文件并存(``fused_moe.py`` / ``fused_moe_0_23_0.py``),在役版本以
    导入顺序决定;两者都有 ``_shared_experts_part2`` 就都挂,缺一个不影响另一个。
    """
    for mod_name in (
        "vllm_ascend.ops.fused_moe.fused_moe_0_23_0",
        "vllm_ascend.ops.fused_moe.fused_moe",
    ):
        try:
            module = __import__(mod_name, fromlist=["AscendFusedMoE"])
            cls = getattr(module, "AscendFusedMoE", None)
            if cls is None or not hasattr(cls, "_shared_experts_part2"):
                continue
            if getattr(cls._shared_experts_part2, "_tileinfer_wrapped", False):
                continue
            wrapped = _make_part2(cls._shared_experts_part2)
            wrapped._tileinfer_wrapped = True
            cls._shared_experts_part2 = wrapped
        except Exception:  # pragma: no cover - 优化模块不得影响在役路径
            logger.warning("H73: failed to patch %s shared-expert part2", mod_name)

