"""H27 preNZ MoE 引擎接入(D70):接管 vLLM-Ascend ``unified_apply_mlp``。

patch 点 = ``vllm_ascend.ops.fused_moe.moe_comm_method.unified_apply_mlp``(模块全局名,
``MoECommMethod._apply_mlp`` 直接按名调用,monkey-patch 生效;与 install_gdn_* 同型)。

接管条件(其余一律回落原实现,保证通用正确性):
- 未量化 bf16(``quant.is_quant`` 为假)、无 LoRA、``activation == SWIGLU``、
  ``need_trans`` 任意(原始 w13 布局 [E,2I,K] 语义固定:0..I-1=gate、I..2I-1=up);
- TP=1 单卡(E 维完整,group_list cumsum 全局)。

链路(unquant 等价替换):
  compact [R,K] --scatter--> X_slot [rows_static,K]
  → gate_up+SwiGLU align 内核(权重=加载期预打包 Wnz,免 nd2nz 直载,E76 转置连续序)
  → Y_slot [rows_static,I] → down align 内核 → Out_slot [rows_static,H]
  --gather(slot_pos)--> [R,H](行序与输入一致,combine 无感)。

opt-in:``TILEINFER_OPS`` 显式含 ``moe``(BACKLOG 纪律:内核线 ≥1.0x aclnn 门槛
(H27 bench)通过后才默认提供;此处仅注册安装路径)。

权重打包:懒触发(首次 forward,``id``+``_version`` 校验缓存;~GB 级一次性,被
引擎 warmup 吸收)。内核实例按 ``(rows_static, em_static)`` 缓存,首次出现的新几何
触发 tilelang 编译(同样被 warmup 吸收)。
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)

_BLOCK_M = 32


def _swiglu_allowed(activation) -> bool:
    """仅标准 SwiGLU(silu(gate)*up,门/升各占 w13 输出前/后半)与内核 epilogue 一致。"""
    try:
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    except Exception:  # noqa: BLE001 - 版本差异时保守回落
        return False
    try:
        return activation == MoEActivation.SWIGLU
    except Exception:  # noqa: BLE001
        return False


class _PrenzMoeState:
    """每 (几何) 的内核实例与缓冲;每 (权重对象) 的预打包 Wnz 缓存。"""

    def __init__(self, inter: int, hidden: int, num_experts: int):
        self.inter = inter
        self.hidden = hidden
        self.num_experts = num_experts
        self.rows_static = 0
        self.em_static = 0
        self._contract_checked = False
        self._kernels = {}  # (rows_static, em_static) -> (k_gate_up, k_down)
        self._bufs = {}  # rows_static -> (X_slot, Y_slot, Out_slot)
        self._wcache = {}  # id(weight) -> (version, packed)

    # ---- 权重懒 pack(H27 转置连续契约,E76) ------------------------------
    def pack_weights(self, w13: torch.Tensor, w2: torch.Tensor):
        from tileinfer.kernels.moe.align import pack_down, pack_gate_up

        key = (id(w13), id(w2))
        hit = self._wcache.get(key)
        ver = (w13._version, w2._version)
        if hit is not None and hit[0] == ver:
            return hit[1], hit[2]
        packed13 = pack_gate_up(w13.detach(), inter=self.inter)
        packed2 = pack_down(w2.detach(), inter=self.inter)
        self._wcache.clear()  # 单层单套权重;旧权重对象释放
        self._wcache[key] = (ver, packed13, packed2)
        return packed13, packed2

    # ---- 缓冲按需增长(rows_static 只增不减,分配在 capture 外) -----------
    def buffers(self, rows_static: int, device: torch.device):
        buf = self._bufs.get(rows_static)
        if buf is None:
            opts = {"dtype": torch.bfloat16, "device": device}
            buf = (
                torch.empty((rows_static, self.hidden), **opts),
                torch.empty((rows_static, self.inter), **opts),
                torch.empty((rows_static, self.hidden), **opts),
            )
            self._bufs[rows_static] = buf
        return buf

    # ---- 内核实例(按编译期几何缓存) --------------------------------------
    def kernels(self, rows_static: int, em_static: int):
        key = (rows_static, em_static)
        kern = self._kernels.get(key)
        if kern is None:
            from tileinfer.kernels.moe.align import (
                build_align_down,
                build_align_gate_up,
            )

            k1 = build_align_gate_up(
                num_tiles_max=em_static, rows_slots=rows_static,
                num_experts=self.num_experts, inter=self.inter,
                hidden=self.hidden, block_m=_BLOCK_M,
                block_n=256, block_k=128,
            )
            k2 = build_align_down(
                num_tiles_max=em_static, rows_slots=rows_static,
                num_experts=self.num_experts, inter=self.inter,
                hidden=self.hidden, block_m=_BLOCK_M,
                block_n=64, block_k=512,
            )
            kern = (k1, k2)
            self._kernels[key] = kern
        return kern


_STATE: dict | None = None
_patched_module = None
_original_ref = None


def _get_state(w13: torch.Tensor) -> _PrenzMoeState:
    global _STATE
    if _STATE is None:
        num_experts, two_i, hidden = w13.shape
        _STATE = _PrenzMoeState(inter=two_i // 2, hidden=hidden, num_experts=num_experts)
    return _STATE


def tileinfer_apply_mlp(*, mlp_compute_input):
    """TileInfer preNZ MoE 路径;不满足接管条件时回落原 ``unified_apply_mlp``。"""
    from vllm_ascend.ops.fused_moe.moe_mlp import unified_apply_mlp as original

    quant = mlp_compute_input.quant
    if (
        quant.is_quant
        or mlp_compute_input.lora_context is not None
        or not _swiglu_allowed(mlp_compute_input.activation)
        or mlp_compute_input.topk_scales is not None
        or mlp_compute_input.swiglu_limit
    ):
        return original(mlp_compute_input=mlp_compute_input)

    x = mlp_compute_input.hidden_states
    w13 = mlp_compute_input.weights.w1
    w2 = mlp_compute_input.weights.w2
    if x.dtype != torch.bfloat16 or x.ndim != 2 or not x.is_contiguous():
        return original(mlp_compute_input=mlp_compute_input)
    if w13.dtype != torch.bfloat16 or w2.dtype != torch.bfloat16:
        return original(mlp_compute_input=mlp_compute_input)

    group_list = mlp_compute_input.group_list
    if group_list.dtype == torch.int64:
        group_list = group_list.to(torch.int32)
    if mlp_compute_input.group_list_type == 1:  # counts → cumsum(type=0)
        group_list = torch.cumsum(group_list, 0).to(torch.int32)

    state = _get_state(w13)
    num_rows = int(x.shape[0])
    if num_rows == 0:
        return x.new_empty((0, state.hidden))
    if group_list.shape[0] != state.num_experts:
        # 多卡 EP/切片:形态超出本路径假设(TP=1,E 维完整),回落
        return original(mlp_compute_input=mlp_compute_input)
    if not state._contract_checked:
        # 布局契约(group_list cumsum 闭合于 R)只在首次调用同步校验一次,
        # 之后信任 dispatch 契约,主路径零主机同步
        if int(group_list[-1]) != num_rows:
            logger.warning(
                "TileInfer MoE preNZ: group_list[-1]=%s != num_rows=%s, takeover disabled",
                int(group_list[-1]), num_rows,
            )
            mcm = _patched_module
            if mcm is not None:
                mcm.unified_apply_mlp = _original_ref
            return original(mlp_compute_input=mlp_compute_input)
        state._contract_checked = True

    from tileinfer.kernels.moe.slot_layout import (
        em_upper_bound,
        slot_layout_from_group_list,
    )

    # slot 行数上界 = EM_static*BM(每 expert 至多一个不完整 tile):探针链同公式
    # (rows = num_tiles*BM);缓冲按该上界分配,rows_static 只增不减
    em_static = em_upper_bound(num_rows, state.num_experts, _BLOCK_M)
    rows_static = em_static * _BLOCK_M
    if rows_static > state.rows_static:
        logger.info("TileInfer MoE preNZ geometry grows to rows_static=%d em_static=%d",
                    rows_static, em_static)
        state.rows_static = rows_static
        state.em_static = em_static
    else:
        em_static = state.em_static
        rows_static = state.rows_static

    lay = slot_layout_from_group_list(
        group_list, num_rows, rows_static, em_static, state.num_experts, _BLOCK_M
    )
    x_slot, y_slot, out_slot = state.buffers(rows_static, x.device)
    pos64 = lay["slot_pos"].long()
    x_slot.index_copy_(0, pos64, x)

    packed13, packed2 = state.pack_weights(w13, w2)
    k1, k2 = state.kernels(rows_static, em_static)
    k1(x_slot, packed13, lay["expert_ids"], lay["sorted_ids"], y_slot)
    k2(y_slot, packed2, lay["expert_ids"], lay["sorted_ids"], out_slot)
    return out_slot.index_select(0, pos64)


def install_moe_prenz() -> bool:
    """接管 vLLM-Ascend MoE MLP(显式 opt-in:``TILEINFER_OPS`` 含 ``moe``)。"""
    spec = (os.environ.get("TILEINFER_OPS") or "").strip().lower()
    if "moe" not in {s.strip() for s in spec.split(",") if s.strip()}:
        logger.info("TileInfer MoE preNZ takeover left off (TILEINFER_OPS does not list moe)")
        return False
    try:
        import vllm_ascend.ops.fused_moe.moe_comm_method as mcm
    except Exception as exc:  # noqa: BLE001
        logger.warning("TileInfer MoE preNZ takeover unavailable (no moe_comm_method): %s", exc)
        return False
    if getattr(mcm, "_tileinfer_prenz_patched", False):
        return True
    global _patched_module, _original_ref
    _patched_module = mcm
    _original_ref = mcm.unified_apply_mlp
    mcm.unified_apply_mlp = tileinfer_apply_mlp
    mcm._tileinfer_prenz_patched = True
    logger.info("TileInfer MoE preNZ takeover installed (unified_apply_mlp -> tileinfer_apply_mlp)")
    return True
