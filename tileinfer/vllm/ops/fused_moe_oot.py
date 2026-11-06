"""The out-of-tree (OOT) ``UnquantizedFusedMoEMethod`` vLLM 0.23 expects, built on our MoE port.

Why this module has to exist
----------------------------
On an OOT platform (``TileInferPlatform.is_out_of_tree()``) vLLM 0.23 never builds a routed-expert
kernel, and it *knows* it does not:

::

    oracle/unquantized.py:168-169    if current_platform.is_out_of_tree():
                                         return UnquantizedMoeBackend.OOT, None     # experts_cls = None
    unquantized_fused_moe_method.py:55      self.unquantized_backend, self.experts_cls = select_...
    unquantized_fused_moe_method.py:217-222 if self.unquantized_backend in [TPU, OOT]:
                                                return                              # no kernel, ever
    fused_moe_method_base.py:154-161        def is_monolithic(...):
                                                if self.moe_kernel is None:
                                                    if hasattr(self, "experts_cls"):
                                                        return self.experts_cls.is_monolithic()
                                                    ...
                                                return self.moe_kernel.is_monolithic

"OOT handles internally" means *the platform is expected to register an OOT subclass* - the
documented extension point, and vLLM's own docstring names this exact class
(``custom_op.py:322-353``: ``@UnquantizedFusedMoEMethod.register_oot class HPUUnquantizedFusedMoEMethod``).
Without one, ``FusedMoE.maybe_init_modular_kernel`` (``layer.py:457``) reads
``quant_method.is_monolithic`` and raises ``AttributeError: 'NoneType' object has no attribute
'is_monolithic'`` **while the model is loading** (the call chain is ``gpu_model_runner.py:5193`` ->
``prepare_communication_buffer_for_model`` -> ``layer.py:451-457``). No token is ever produced.

What we supply instead of vLLM's kernel
---------------------------------------
Three small objects, each with exactly the members the local vLLM source reads - nothing invented:

============================  =========================================================  ==============================
member                        read at                                                     what ours answers
============================  =========================================================  ==============================
``experts_cls`` (an object)   ``fused_moe_method_base.py:157-158`` (``hasattr`` + call)  :class:`TorchExperts`
``experts_cls.is_monolithic()``  ``fused_moe_method_base.py:158``                         ``False`` (a *method*)
``moe_kernel`` (an object)    ``fused_moe_method_base.py:35, 41, 48, 137-138, 156-161``   :class:`TorchMoEKernel`
``moe_kernel.is_monolithic``  ``fused_moe_method_base.py:161``                           ``False`` (an *attribute*)
``moe_kernel.can_overlap_shared_experts``  ``fused_moe_method_base.py:48``               ``False``
``moe_kernel.prepare_finalize.topk_indices_dtype()``  ``fused_moe_method_base.py:137-138``  ``None``
``moe_kernel.output_is_reduced()``  ``moe_runner.py:372-373``                            ``False``
``moe_kernel.apply(**kwargs)``  ``unquantized_fused_moe_method.py:311-324``              our ``moe.fused_experts``
``moe_kernel.apply_monolithic(...)``  ``unquantized_fused_moe_method.py:373-374``        refuses (never monolithic)
============================  =========================================================  ==============================

Note the asymmetry in that table, because it *is* the trap: the base class calls
``experts_cls.is_monolithic()`` **as a method** and reads ``moe_kernel.is_monolithic`` **as an
attribute**.  A stand-in that answers with a plain truthy attribute where the caller calls it is a
silent wrong branch; ``tests/test_fused_moe_oot.py`` therefore pins both shapes and proves that the
plain-truthy variant fails.

``moe_kernel`` is set in ``__init__`` (there is no ``_setup_kernel`` call for OOT -
``unquantized_fused_moe_method.py:217-222`` returns first), which has three consequences we want and
one we must guard:

* ``supports_internal_mk`` becomes ``True`` (``fused_moe_method_base.py:38-41``), so
  ``layer.py:457`` returns immediately: ``maybe_init_modular_kernel`` is never entered, and the
  stock ``process_weights_after_loading`` keeps returning before ``_setup_kernel`` - our weights stay
  in the checkpoint layout the port consumes.
* per step the runner takes the modular branch (``moe_runner.py:516`` -> ``:524-539``:
  ``router.select_experts`` -> ``quant_method.apply`` -> ``forward`` -> ``forward_oot`` (OOT ladder,
  ``custom_op.py:204-205``) -> the stock ``forward_native`` -> ``self.moe_kernel.apply(...)``).  We
  deliberately do **not** rewrite ``forward_native``: vLLM's own body does the argument marshalling
  (``unquantized_fused_moe_method.py:311-324``), so there is no copy of it here to drift.
* the guard: ``moe_runner.py:682-686`` disables the naive DP/EP dispatch when ``supports_internal_mk``
  is true, because a real kernel dispatches inside itself.  Ours does not, so DP/EP/PCP > 1 is
  **refused** in :func:`check_unquantized_config` instead of silently computing on local tokens.

What we refuse, loudly
----------------------
There is no silent branch in this file.  :func:`check_unquantized_config` refuses, at construction
time (i.e. at model load, before any weight is read): a quantised config, an expert bias, LoRA, a
non-gated MLP (``is_act_and_mul=False``), any DP/EP/PCP > 1, and any storage dtype outside
bf16/fp16/fp32.  :class:`TorchMoEKernel` repeats the dtype check on the weights it is handed, so a
quantised checkpoint is caught even if a config object claims otherwise.  The only thing a
non-matching backend gets is a hard error: ``register_oot`` is unconditional, so if this class is
ever instantiated on a platform whose ``select_unquantized_moe_backend`` returns something other than
``OOT`` (CUDA/ROCm/XPU/CPU), running the block would mean *upstream's* kernel format conversion on
top of our torch port - wrong numerics instead of a crash.  It raises instead.

What a real upstream fix removes
--------------------------------
Every docstring above is a citation of things that would disappear: the day
``select_unquantized_moe_backend`` returns a kernel class for an OOT platform (or
``unquantized_fused_moe_method`` stops returning early for ``UnquantizedMoeBackend.OOT``), this whole
module - and the ``unquantized_fused_moe`` entry in ``families/gdn_moe.py`` - is deleted, not
adapted.  The ``Patch.remove_when`` for the binding says exactly that.

Import policy: **no ``vllm`` import, at module level or anywhere else.**  The base class arrives as
an argument to :func:`make_oot_method`, so this module imports on a CPU box (it needs torch, because
:func:`tileinfer.vllm.ops.moe.fused_experts` does), which is what lets ``tileinfer doctor`` and the
CPU test suite exercise the whole mechanism.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch

# The *same* object the ``fused_experts`` binding in ``families/gdn_moe.py`` wraps as its baseline.
# Reused by identity on purpose: the OOT kernel is a second call site for one implementation, not a
# second implementation (``tests/test_fused_moe_oot.py`` pins the identity).
from tileinfer.vllm.ops.moe import _QUANT_BOOL_ATTRS, _QUANT_VALUE_ATTRS, fused_experts

__all__ = [
    "OOT_CLASS_NAME",
    "TorchExperts",
    "TorchMoEKernel",
    "TorchPrepareFinalize",
    "check_unquantized_config",
    "make_oot_method",
]

logger = logging.getLogger("tileinfer.vllm.ops.fused_moe_oot")

#: The class name we publish.  ``register_oot`` registers it under the *in-tree* name
#: (``custom_op.py:333-337``), so ``CustomOp.__new__`` substitutes it on every instantiation
#: (``custom_op.py:109-128``, reached from ``layer.py:359``).
OOT_CLASS_NAME = "TorchUnquantizedFusedMoEMethod"

#: The dtypes this port can compute.  bf16/fp16 are the serving contract (plan §0); fp32 is accepted
#: because the port's arithmetic is fp32 and the CPU rehearsal runs in it.  Everything else - int8,
#: uint8, fp8, fp4 - is a quantised storage format, i.e. a quantised config, and is refused.
_ALLOWED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)

#: One INFO line per process, not one per MoE layer (Qwen3.6 has dozens).
_ANNOUNCED = False


def _announce(moe: Any, backend_name: str) -> None:
    """Say once which implementation the expert MLP runs through; debug on every layer after."""
    global _ANNOUNCED
    message = "TileInfer OOT MoE: %s installed for %s experts (backend=%s, in_dtype=%s); expert " \
        "compute is tileinfer.vllm.ops.moe.fused_experts"
    args = (
        OOT_CLASS_NAME,
        getattr(moe, "num_experts", "?"),
        backend_name,
        getattr(moe, "in_dtype", "?"),
    )
    if _ANNOUNCED:
        logger.debug(message, *args)
        return
    _ANNOUNCED = True
    logger.info(message, *args)


# =============================================================================================
# Refusals: the config shapes this method must not claim
# =============================================================================================


def check_unquantized_config(moe: Any) -> None:
    """Refuse anything but an unquantised, single-rank, gated-MLP bf16/fp16 MoE config.

    Attribute names are read defensively (``getattr`` with the upstream default), so this works on
    the real ``FusedMoEConfig`` (``.../fused_moe/config.py:1235-1269``) and on a stub in a test.  The
    quantisation markers are ``moe._check_unquantized``'s own constants: one list, not two.

    Raise ``NotImplementedError`` (a *refusal*, not a degradation) for:

    * ``is_act_and_mul=False`` - ``layer.py:367-372`` already refuses a non-gated MLP on a
      non-CUDA/non-XPU platform before any kernel runs, and our port is the gated SwiGLU MLP only;
    * ``has_bias`` - upstream folds the expert bias into the quant config
      (``unquantized_fused_moe_method.py:275-282``); ``fused_experts`` has no bias argument, so
      accepting the config would silently drop it;
    * ``is_lora_enabled`` - LoRA on the routed experts is a different kernel contract;
    * DP / EP / PCP > 1 - see the module docstring: ``supports_internal_mk`` is true, so
      ``moe_runner.py:682-686`` skips the naive dispatch/combine and *our* kernel would have to do
      it.  It does not.  TP is fine (the runner all-reduces the combined output itself,
      ``moe_runner.py:396-418``, because :meth:`TorchMoEKernel.output_is_reduced` is ``False``);
    * any ``in_dtype`` outside bf16/fp16/fp32, and any quantisation marker that is set
      (``quant_dtype``/``weight_quant_dtype``, or any of ``moe._QUANT_BOOL_ATTRS`` /
      ``moe._QUANT_VALUE_ATTRS``) - that is a quantised config.
    """
    if getattr(moe, "is_act_and_mul", True) is not True:
        raise NotImplementedError(
            "TileInfer OOT MoE: moe.is_act_and_mul=False (the non-gated MLP) is not implemented; "
            "vLLM refuses it on a non-CUDA/non-XPU platform itself "
            "(model_executor/layers/fused_moe/layer.py:367-372) and this port only implements the "
            "gated SwiGLU expert MLP (tileinfer/vllm/ops/moe.py: fused_experts)."
        )
    if getattr(moe, "has_bias", False):
        raise NotImplementedError(
            "TileInfer OOT MoE: moe.has_bias=True is not implemented; upstream folds w13_bias/"
            "w2_bias into the quant config (unquantized_fused_moe_method.py:275-282) and "
            "tileinfer.vllm.ops.moe.fused_experts has no bias argument, so running this config "
            "would silently drop the expert bias. Fail here instead."
        )
    if getattr(moe, "is_lora_enabled", False):
        raise NotImplementedError(
            "TileInfer OOT MoE: moe.is_lora_enabled=True is not implemented; LoRA on the routed "
            "experts is a different kernel contract (oracle/unquantized.py:171-174 routes it to the "
            "Triton backend, which no OOT platform has)."
        )

    parallel = getattr(moe, "moe_parallel_config", None)
    for name in ("dp_size", "ep_size", "pcp_size"):
        size = getattr(parallel, name, None)
        if size is not None and int(size) > 1:
            raise NotImplementedError(
                f"TileInfer OOT MoE: moe.moe_parallel_config.{name}={int(size)} is not implemented. "
                "Our moe_kernel reports can_overlap_shared_experts=False and stands in for a real "
                "FusedMoEKernel, so FusedMoEMethodBase.supports_internal_mk is True, which makes "
                "MoERunner skip its naive dispatch/combine (runner/moe_runner.py:682-706) on the "
                "assumption that the kernel does it. It does not: v0 runs TP-only (no DP/EP/PCP)."
            )

    dtype = getattr(moe, "in_dtype", None)
    if dtype is not None and dtype not in _ALLOWED_DTYPES:
        raise NotImplementedError(
            f"TileInfer OOT MoE: moe.in_dtype={dtype} is not an unquantised bf16/fp16 (or fp32 "
            f"rehearsal) dtype; this port implements the unquantised expert MLP only. Allowed: "
            f"{tuple(str(item) for item in _ALLOWED_DTYPES)}."
        )

    markers = [
        f"{name}={getattr(moe, name)!r}"
        for name in _QUANT_BOOL_ATTRS
        if getattr(moe, name, None)
    ]
    markers += [
        f"{name}={getattr(moe, name)!r}"
        for name in _QUANT_VALUE_ATTRS
        if getattr(moe, name, None) is not None and getattr(moe, name) is not False
    ]
    markers += [
        f"{name}={getattr(moe, name)!r}"
        for name in ("quant_dtype", "weight_quant_dtype")
        if getattr(moe, name, None) is not None
    ]
    if markers:
        raise NotImplementedError(
            "TileInfer OOT MoE: the moe config carries quantisation markers ("
            + ", ".join(markers)
            + "); this port is the unquantised bf16/fp16 expert MLP only "
            "(docs/plan-vllm-native-backend.md §0). A quantised call must fail here rather than "
            "compute in the wrong precision."
        )


def _check_expert_weight_dtype(w1: Any, w2: Any) -> None:
    """Refuse quantised expert *weights* (int8/fp8/...) even if the config claimed nothing.

    ``fused_experts`` would happily cast an int8 tensor to fp32 and produce numbers, so the
    checkpoint - not only the config - is checked.  Called per forward: two ``.dtype`` reads, no host
    synchronisation.
    """
    for name, weight in (("w1", w1), ("w2", w2)):
        dtype = getattr(weight, "dtype", None)
        if dtype is not None and dtype not in _ALLOWED_DTYPES:
            raise NotImplementedError(
                f"TileInfer OOT MoE: {name} has dtype {dtype}, which is not an unquantised "
                f"bf16/fp16 (or fp32) expert weight. Allowed: "
                f"{tuple(str(item) for item in _ALLOWED_DTYPES)}. A quantised checkpoint must not "
                "be cast into the torch port: that computes in the wrong precision instead of "
                "failing."
            )


# =============================================================================================
# The stand-ins: one member per source line, nothing more
# =============================================================================================


class TorchExperts:
    """Stands in for the ``mk.FusedMoEExperts`` subclass the OOT oracle returns as ``None``.

    ``oracle/unquantized.py:168-169`` returns ``(UnquantizedMoeBackend.OOT, None)``, and the base
    class then calls ``self.experts_cls.is_monolithic()`` (``fused_moe_method_base.py:156-158``) -
    the ``AttributeError`` of the whole gap.  ``is_monolithic()`` is the **only** member the local
    source reads from ``experts_cls``; the other uses of an ``experts_cls`` in the tree pass it to
    ``make_unquantized_moe_kernel`` (``oracle/unquantized.py:329-360``), which the OOT branch of
    ``process_weights_after_loading`` never reaches (``unquantized_fused_moe_method.py:217-222``).
    So this class implements exactly one method and says so, instead of pretending to be a kernel.

    It answers ``False`` because our port is *not* monolithic: the runner computes the top-k itself
    (``moe_runner.py:524-528``) and hands us ``topk_weights``/``topk_ids``.  ``True`` would send the
    step to ``apply_monolithic`` with raw ``router_logits`` (``moe_runner.py:516-522``), i.e. a
    different function.  Note ``is_monolithic`` is a **method** here: the caller writes
    ``experts_cls.is_monolithic()``.
    """

    def is_monolithic(self) -> bool:  # noqa: D401 - the name is upstream's
        """``fused_moe_method_base.py:158`` - called with parentheses."""
        return False

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return "TorchExperts(is_monolithic=False)"


class TorchPrepareFinalize:
    """The one member of ``mk.FusedMoEPrepareAndFinalizeModular`` the method reads.

    ``fused_moe_method_base.py:137-138``: ``return self.moe_kernel.prepare_finalize.
    topk_indices_dtype()``.  ``None`` means "do not recast ``topk_ids``" - the router's own int dtype
    is what our port accepts (it casts to int64 internally), so there is nothing to pin.  A real
    prepare/finalize also owns the all2all; v0 is TP-only, which
    :func:`check_unquantized_config` enforces.
    """

    def topk_indices_dtype(self) -> Optional[torch.dtype]:
        return None

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return "TorchPrepareFinalize(topk_indices_dtype=None)"


class TorchMoEKernel:
    """Stands in for the ``mk.FusedMoEKernel`` the OOT branch never builds.

    It is *not* a kernel and does not pretend to be one: every other attribute of ``FusedMoEKernel``
    (routing tables, all2all, fused output reduction, ...) is deliberately absent, because the local
    source never reads it on this path.  The four members below are read at these exact lines:

    * ``is_monolithic`` (attribute, **no parentheses**) - ``fused_moe_method_base.py:161``;
    * ``can_overlap_shared_experts`` (attribute) - ``fused_moe_method_base.py:43-49``, which decides
      ``SharedExpertsOrder`` in ``runner/shared_experts.py:92-112``; ``False`` is what makes the
      runner apply the shared expert itself (``moe_runner.py:512-514``) with
      ``NO_OVERLAP``, before and outside our call;
    * ``output_is_reduced()`` - ``moe_runner.py:369-374`` and ``:396-418``; ``False`` because this
      kernel returns unreduced output, so the runner's late all-reduce still runs when TP > 1;
    * ``prepare_finalize.topk_indices_dtype()`` - ``fused_moe_method_base.py:137-138``.

    ``apply`` takes the keyword set the stock ``forward_native`` passes
    (``unquantized_fused_moe_method.py:312-324``) and computes with
    :func:`tileinfer.vllm.ops.moe.fused_experts`.
    """

    #: Read WITHOUT parentheses, ``fused_moe_method_base.py:161``.  (``experts_cls.is_monolithic``
    #: is the opposite convention - see :class:`TorchExperts`.)
    is_monolithic = False

    #: Read, ``fused_moe_method_base.py:48``.
    can_overlap_shared_experts = False

    def __init__(self) -> None:
        self.prepare_finalize = TorchPrepareFinalize()

    def output_is_reduced(self) -> bool:
        """``moe_runner.py:372-373``: our output is not all-reduced, the runner does it."""
        return False

    def apply(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: Any = "silu",
        apply_router_weight_on_input: bool = False,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        *,
        shared_experts: Any = None,
        shared_experts_input: Any = None,
    ) -> torch.Tensor:
        """The routed expert MLP - ``moe.fused_experts``, reached through the kernel slot.

        Positional order is ``fused_experts``' own (``.../fused_moe/fused_moe.py:1474-1485``); the
        stock ``forward_native`` passes every argument by keyword, so both call styles land
        correctly.

        ``shared_experts``/``shared_experts_input`` are accepted and **ignored on purpose**, not
        dropped: the runner always passes them (``moe_runner.py:532-539``) and applies them itself
        in ``NO_OVERLAP`` order *before* this call (``moe_runner.py:512-514`` ->
        ``shared_experts.py:92-112``), then adds the output (``moe_runner.py:673-676``).  Adding
        them here would double-count the dense expert.  ``mk_can_overlap_shared_experts`` is
        ``False``, so ``MK_INTERNAL_OVERLAPPED`` - the one order that would hand them to the kernel
        - can never be selected.
        """
        _check_expert_weight_dtype(w1, w2)
        return fused_experts(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            # The OOT method IS the unquantised path (layer.py:355-359 only builds it when there is
            # no quant_config, or get_quant_method() declined), and the storage dtype was checked
            # above; there is no FusedMoEQuantConfig to pass down.
            quant_config=None,
        )

    def apply_monolithic(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Never reached: :attr:`is_monolithic` is ``False`` (``moe_runner.py:516-522``)."""
        raise NotImplementedError(
            "TileInfer OOT MoE: apply_monolithic is not implemented. This kernel answers "
            "is_monolithic=False, so the runner routes through router.select_experts + "
            "quant_method.apply (runner/moe_runner.py:524-539) and never takes the monolithic "
            "branch. Reaching this means the runner's branch changed - read "
            "fused_moe_method_base.py:154-161 again before implementing it."
        )

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return "TorchMoEKernel(is_monolithic=False, can_overlap_shared_experts=False)"


# =============================================================================================
# The method class itself (a factory: the base class arrives as an argument)
# =============================================================================================


def make_oot_method(base_cls: type) -> type:
    """Build the OOT subclass of ``base_cls`` (``UnquantizedFusedMoEMethod``).

    This is the ``impl`` of the family's ``kind="custom_op_oot"`` binding: ``install.py`` calls
    ``impl(base_cls)`` and then registers the result through the **class-level** extension point,

    ::

        base_cls.register_oot(TorchUnquantizedFusedMoEMethod)      # custom_op.py:331-353

    after which ``CustomOp.__new__`` (``custom_op.py:109-128``) substitutes it for every
    ``UnquantizedFusedMoEMethod(...)`` - which ``layer.py:359`` performs for the unquantised path.
    Registering the class is ``install.py``'s job, not this function's: the *mechanism* stays in one
    place, and this module never needs to know whether the family switch was on.

    Why a factory at all: this module must import on a box without vLLM (the CPU suite and
    ``tileinfer doctor`` run there), so the base class is a parameter instead of an import.

    The subclass overrides four things and nothing else:

    * ``__init__`` - keep vLLM's selection result, then replace the ``None`` it returned under
      ``UnquantizedMoeBackend.OOT`` with :class:`TorchExperts`, and attach
      :class:`TorchMoEKernel` (see the module docstring for what that changes);
    * ``maybe_make_prepare_finalize`` -> ``None`` - belt and braces.  ``layer.py:457`` short-circuits
      today because ``supports_internal_mk`` is ``True``, but if a future vLLM consulted it anyway,
      the inherited body raises (``unquantized_fused_moe_method.py:70-78``: "should not be called");
      ``None`` is what ``FusedMoEMethodBase.maybe_make_prepare_finalize`` returns when there is no
      all2all backend (``fused_moe_method_base.py:105-115``), so it is the correct answer here;
    * ``supports_eplb`` -> ``False`` - the inherited ``True`` (``:66-68``) advertises EPLB, but our
      kernel has no EPLB interface.  ``layer.py:374`` then refuses ``enable_eplb`` with a clear
      message instead of an ``AttributeError`` deep in a captured graph;
    * ``process_weights_after_loading`` - call upstream's (whose OOT early return,
      ``:217-222``, is what keeps the checkpoint weight layout) and then *assert* our kernel is
      still the one in place, so a future upstream change to that early return is one loud line
      instead of silently shuffling weights into a format the port does not read.
    """
    if not isinstance(base_cls, type):
        raise TypeError(
            f"TileInfer OOT MoE: make_oot_method expects the in-tree UnquantizedFusedMoEMethod "
            f"class, got {base_cls!r}. install.py resolves it from the binding's target; a string "
            "or an instance here means the binding table or the patch body is wrong."
        )

    class TorchUnquantizedFusedMoEMethod(base_cls):  # noqa: N801 - named after what it replaces
        """``UnquantizedFusedMoEMethod`` for an out-of-tree platform, on the torch MoE port."""

        def __init__(self, moe: Any) -> None:
            super().__init__(moe)
            backend = getattr(self, "unquantized_backend", None)
            backend_name = str(getattr(backend, "name", backend))
            if backend_name != "OOT":
                raise RuntimeError(
                    "TileInfer OOT MoE: this class only implements the OOT path, but "
                    f"select_unquantized_moe_backend chose the {backend_name!r} backend "
                    "(oracle/unquantized.py:153-175). Registering it there would run our torch port "
                    "on top of *that* backend's weight-format conversion "
                    "(unquantized_fused_moe_method.py:157-202) - wrong numerics instead of a crash. "
                    "Set TILEINFER_OPS so the moe family is off on this platform, or remove the "
                    "custom_op_oot binding."
                )
            check_unquantized_config(moe)
            # The two members the base class and the runner read (see the module docstring table).
            self.experts_cls = TorchExperts()
            self.moe_kernel = TorchMoEKernel()
            _announce(moe, backend_name)

        def maybe_make_prepare_finalize(self, routing_tables: Any = None) -> None:
            """No all2all prepare/finalize on OOT (see :func:`make_oot_method`)."""
            return None

        @property
        def supports_eplb(self) -> bool:
            """No EPLB: the stand-in kernel has no expert-load-balancing interface."""
            return False

        def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
            """Upstream's OOT early return, then prove nobody replaced our kernel.

            ``unquantized_fused_moe_method.py:204-222`` returns for ``UnquantizedMoeBackend.OOT``
            *before* ``_setup_kernel``, so no weight shuffle happens and ``moe_kernel`` keeps
            pointing at :class:`TorchMoEKernel`.  If a future vLLM drops that branch, the inherited
            body would call ``convert_to_unquantized_kernel_format`` + ``make_unquantized_moe_kernel``
            with our stand-in ``experts_cls`` - the error below names the cause instead.
            """
            super().process_weights_after_loading(layer)
            if not isinstance(self.moe_kernel, TorchMoEKernel):
                raise RuntimeError(
                    "TileInfer OOT MoE: process_weights_after_loading replaced the stand-in "
                    f"moe_kernel with {type(self.moe_kernel).__name__!r}. Upstream no longer takes "
                    "the OOT early return (unquantized_fused_moe_method.py:217-222), so it built a "
                    "real kernel over weights in a converted layout while our port still expects "
                    "the checkpoint layout. This is the moment to delete this module: see "
                    "Patch.remove_when ('upstream builds an OOT moe_kernel')."
                )

    TorchUnquantizedFusedMoEMethod.__name__ = OOT_CLASS_NAME
    TorchUnquantizedFusedMoEMethod.__qualname__ = OOT_CLASS_NAME
    return TorchUnquantizedFusedMoEMethod
