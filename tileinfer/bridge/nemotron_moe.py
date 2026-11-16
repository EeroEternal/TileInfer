"""Bridge adapter for vLLM/vllm-ascend non-gated FusedMoE support (v3).

vLLM 0.23's ``FusedMoE.__init__`` refuses ``is_act_and_mul=False`` on anything but CUDA/XPU
(``layer.py:367-372``) before the OOT quant method is selected.  The non-gated expert MLP itself
is implemented in ``tileinfer.providers.moe.fused_experts``; the guard is a platform allow-list, not
an algorithm check.  This module swaps the module-level ``current_platform`` object for the
duration of that one constructor call: every attribute is forwarded, only ``is_cuda()`` answers
True.  The vendored source is not edited and no other call in the process sees the proxy.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_MARKER = "__tileinfer_non_gated_guard__"


class _CudaGuardPlatform:
    def __init__(self, real: Any) -> None:
        self._real = real

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)

    def is_cuda(self) -> bool:
        return True

    def is_cuda_alike(self) -> bool:
        # layer.py:368 uses this spelling, not is_cuda().
        return True


def install_non_gated_guard() -> bool:
    """Allow ``FusedMoE(is_act_and_mul=False)`` to be constructed on this platform."""
    try:
        from vllm.model_executor.layers.fused_moe import layer as moe_layer
    except Exception as exc:  # noqa: BLE001 - CPU/non-vLLM environments are fine
        logger.debug("TileInfer OOT MoE: non-gated guard not installed (%r)", exc)
        return False

    original = moe_layer.FusedMoE.__init__
    if getattr(original, _MARKER, False):
        return True

    def patched(self: Any, *args: Any, **kwargs: Any) -> None:
        if kwargs.get("is_act_and_mul", True) is not False:
            return original(self, *args, **kwargs)
        real = moe_layer.current_platform
        moe_layer.current_platform = _CudaGuardPlatform(real)
        try:
            return original(self, *args, **kwargs)
        finally:
            moe_layer.current_platform = real

    setattr(patched, _MARKER, True)
    moe_layer.FusedMoE.__init__ = patched
    logger.info(
        "TileInfer OOT MoE: non-gated FusedMoE guard bypassed for this platform (relu2/no_mul)"
    )
    return True


#: The Ascend platform subclass does not go through ``UnquantizedFusedMoEMethod.apply``: it
#: installs ``AscendUnquantizedFusedMoEMethod`` in ``AscendFusedMoE.__init__`` and then calls
#: ``moe_comm_method.fused_experts`` (grouped-matmul).  That kernel only implements gated MLPs and
#: rejects Nemotron's ``[E, inter, hidden]`` / relu2 layout at the ACL level.  Replace only the
#: execution and weight-postprocessing methods for the non-gated case; keep the vendor routing
#: selector so the router semantics (sigmoid / grouped top-k / bias) stay identical.
_ASCEND_NON_GATED_MARKER = "__tileinfer_ascend_non_gated_experts__"


def install_ascend_non_gated_experts() -> bool:
    """Route non-gated experts through the torch implementation on the Ascend platform."""
    try:
        from vllm_ascend.ops.fused_moe.fused_moe import AscendUnquantizedFusedMoEMethod
        from vllm_ascend.ops.fused_moe.experts_selector import select_experts
        from vllm_ascend.ops.fused_moe.moe_comm_method import FusedExpertsResult
        from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
            UnquantizedFusedMoEMethod,
        )
        from tileinfer.providers.moe import fused_experts
    except Exception as exc:  # noqa: BLE001 - CPU / non-Ascend environments are fine
        logger.debug("TileInfer OOT MoE: Ascend non-gated hook not installed (%r)", exc)
        return False

    if getattr(AscendUnquantizedFusedMoEMethod.apply, _ASCEND_NON_GATED_MARKER, False):
        return True

    original_apply = AscendUnquantizedFusedMoEMethod.apply
    original_process = AscendUnquantizedFusedMoEMethod.process_weights_after_loading

    def process_weights_after_loading(self: Any, layer: Any) -> None:
        if getattr(getattr(layer, "moe_config", None), "is_act_and_mul", True) is False:
            # Keep the checkpoint's [E, inter, hidden] and [E, hidden, inter] layout; the vendor
            # transpose is for its grouped GEMM, which we are not calling for this layer.
            UnquantizedFusedMoEMethod.process_weights_after_loading(self, layer)
            return
        return original_process(self, layer)

    def apply(self: Any, layer: Any, x: Any, use_grouped_topk: bool, top_k: int,
              router_logits: Any, renormalize: bool, topk_group: Any = None,
              num_expert_group: Any = None, custom_routing_function: Any = None,
              scoring_func: str = "softmax", routed_scaling_factor: float = 1.0,
              e_score_correction_bias: Any = None, num_experts: int = -1,
              expert_map: Any = None, apply_router_weight_on_input: bool = False,
              activation: str = "silu", enable_force_load_balance: bool = False,
              log2phy: Any = None, global_redundant_expert_num: int = 0,
              pertoken_scale: Any = None, mc2_mask: Any = None):
        if getattr(getattr(layer, "moe_config", None), "is_act_and_mul", True) is not False:
            return original_apply(
                self, layer, x, use_grouped_topk, top_k, router_logits, renormalize,
                topk_group, num_expert_group, custom_routing_function, scoring_func,
                routed_scaling_factor, e_score_correction_bias, num_experts, expert_map,
                apply_router_weight_on_input, activation, enable_force_load_balance, log2phy,
                global_redundant_expert_num, pertoken_scale, mc2_mask,
            )

        topk_weights, topk_ids = select_experts(
            hidden_states=x,
            router_logits=router_logits,
            top_k=top_k,
            use_grouped_topk=use_grouped_topk,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
            num_experts=num_experts,
        )
        if log2phy is not None:
            topk_ids = log2phy[topk_ids]
        if expert_map is not None:
            # ``fused_experts`` takes the global-to-local map itself.
            out = fused_experts(
                x, layer.w13_weight, layer.w2_weight, topk_weights, topk_ids,
                activation=activation,
                apply_router_weight_on_input=apply_router_weight_on_input,
                global_num_experts=num_experts,
                expert_map=expert_map,
            )
        else:
            out = fused_experts(
                x, layer.w13_weight, layer.w2_weight, topk_weights, topk_ids,
                activation=activation,
                apply_router_weight_on_input=apply_router_weight_on_input,
                global_num_experts=num_experts,
            )
        return FusedExpertsResult(routed_out=out)

    setattr(apply, _ASCEND_NON_GATED_MARKER, True)
    AscendUnquantizedFusedMoEMethod.apply = apply
    AscendUnquantizedFusedMoEMethod.process_weights_after_loading = process_weights_after_loading
    logger.info(
        "TileInfer OOT MoE: Ascend non-gated expert path installed (torch fused_experts + relu2)"
    )
    return True


def install_nemotron_shared_alias() -> bool:
    """Give Nemotron's non-gated shared MLP the name the Ascend shared-expert path probes.

    ``NemotronHMLP`` (vLLM models/nemotron_h.py:91-128) has ``up_proj`` -> relu2 -> ``down_proj``;
    the Ascend ``_forward_shared_experts`` path assumes every shared expert has ``gate_up_proj``
    and reads it before choosing the unquantised branch (fused_moe_0_23_0.py:565-575).  The alias
    returns ``up_proj``, so the unquantised branch calls the same projection, applies the module's
    own ``act_fn`` once, and then ``down_proj`` -- exactly NemotronHMLP.forward.  No kernel or
    weight layout changes.
    """
    try:
        from vllm.model_executor.models.nemotron_h import NemotronHMLP
    except Exception as exc:  # noqa: BLE001 - non-Nemotron environments are fine
        logger.debug("TileInfer: Nemotron shared-MLP alias not installed (%r)", exc)
        return False
    if getattr(NemotronHMLP, "_tileinfer_gate_up_alias", False):
        return True
    NemotronHMLP.gate_up_proj = property(lambda self: self.up_proj)
    NemotronHMLP._tileinfer_gate_up_alias = True
    logger.info("TileInfer: NemotronHMLP.gate_up_proj aliased to up_proj for Ascend shared experts")
    return True
