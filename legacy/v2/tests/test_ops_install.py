"""Contract tests for ``tileinfer.vllm.ops.install`` - the takeovers, with **no vLLM** on the box.

Why a fake ``vllm`` package (the pattern of ``tests/test_attn_contract.py``)
---------------------------------------------------------------------------
The installer resolves vLLM names lazily and never imports vLLM itself, so the module under test
imports on a CPU laptop.  What cannot be exercised without an engine is the *resolution*: which of
the names in the binding table exist.  The fake below therefore reproduces the exact module layout
the local vLLM subset proves (``reference/vllm-0.23/``), with one "kernel" object per call site and
each of them raising if it is ever called - so a test that accidentally runs vLLM's own
implementation fails loudly instead of silently passing.

The fake deliberately reproduces the module layout the local vLLM subset proves
(``reference/vllm-0.23/``), including the two MoE call sites that are now in it:
``vllm._custom_ops`` (``vllm___custom_ops.py:2346-2363`` and ``:2494-2509``) and
``vllm.model_executor.layers.fused_moe.fused_moe`` (``.../fused_moe/fused_moe.py:1474-1515``).  Each
kernel raises if it is ever called, so a test that accidentally runs vLLM's own implementation fails
loudly instead of silently passing - and each fake carries the **real signature**, so a port whose
argument order drifted cannot bind and then be called successfully.

``EXPECTED_BOUND`` / ``EXPECTED_UNBOUND`` are asserted exactly: when someone transcribes a call site
(or a vLLM release renames one) this list has to change in the same commit - which is the point of
asserting it exactly instead of asserting "at least one op is unbound".

The fake is installed per test (not at import time) and torn down afterwards, and it only adds the
call-site modules that are still absent from ``sys.modules`` - so it *augments* the shared fake in
``tests/fake_vllm.py`` (used by ``tests/test_attn_contract.py`` and
``tests/test_worker_contract.py``) instead of replacing it, whatever order the contract tests run
in.  On a machine that really has vLLM the fake is not installed and the exact-list assertions
skip: they are a statement about *our* transcription, not about the installed engine.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import logging
import pathlib
import sys
import types
from typing import Any, Dict, List, Tuple

import pytest
import torch

from tileinfer.vllm.compat import patches
from tileinfer.vllm.ops import install as install_mod
from tileinfer.vllm.ops import switch

# ---------------------------------------------------------------------------------------
# What the local vLLM subset proves, and what it does not
# ---------------------------------------------------------------------------------------

_GDN = "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"
_FLA = "vllm.model_executor.layers.fla.ops"
_MAMBA = "vllm.model_executor.layers.mamba.ops"

#: The ops that must bind against the transcribed ABI (GDN + MoE families, defaults on).
EXPECTED_BOUND = [
    "chunk_gated_delta_rule",
    "l2norm_fwd",
    "causal_conv1d_fn",
    "causal_conv1d_update",
    "fused_sigmoid_gating_delta_rule_update",
    "fused_recurrent_gated_delta_rule_packed_decode",
    "fused_post_conv_prep",
    "topk_softmax",
    "moe_align_block_size",
    "fused_experts",
    "unquantized_fused_moe",
]

#: The exact unbound list, in table order.  ``chunk_local_cumsum`` is a *reported* miss (ABI
#: mismatch: upstream takes ``chunk_indices``/``head_first``/``output_dtype`` and 3-D/4-D layouts,
#: ``fla_cumsum.py:243-252``), so it stays here deliberately.  The three MoE ops used to sit here
#: ("vLLM name not verified offline"); ``vllm___custom_ops.py`` and ``.../fused_moe/fused_moe.py``
#: settled the names, so they moved to ``EXPECTED_BOUND`` and the fake grew those modules.  A vLLM
#: rename shows up as a name moving between the two lists, which is the reviewed change this
#: exact-list assertion exists to force.
EXPECTED_UNBOUND = [
    "chunk_local_cumsum",
]

#: ``dense`` is off by default, so its only op never even tries to bind.
EXPECTED_DISABLED = ["silu_and_mul"]

_MISSING = object()


def _real_vllm_installed() -> bool:
    """True when ``vllm`` on ``sys.path`` is a real distribution (then the fake is not used)."""
    module = sys.modules.get("vllm")
    if module is not None:
        return getattr(module, "__spec__", None) is not None
    try:
        return importlib.util.find_spec("vllm") is not None
    except (ImportError, ValueError):  # a broken install must not look like presence
        return False


REAL_VLLM = _real_vllm_installed()


# ---------------------------------------------------------------------------------------
# The fake vllm: one module per call site the GDN layer imports from
# ---------------------------------------------------------------------------------------


def _pkg(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []  # type: ignore[attr-defined]
    return module


def _originals() -> Dict[str, Any]:
    """The vLLM-side objects; each raises if run, so a missed takeover cannot pass silently."""

    def l2norm_fwd(x, eps=1e-6, output_dtype=None):  # noqa: ANN001
        raise AssertionError("the fake vLLM l2norm_fwd kernel must never run")

    def causal_conv1d_fn(*args, **kwargs):
        raise AssertionError("the fake vLLM causal_conv1d_fn kernel must never run")

    def causal_conv1d_update(*args, **kwargs):
        raise AssertionError("the fake vLLM causal_conv1d_update kernel must never run")

    def fused_sigmoid_gating_delta_rule_update(*args, **kwargs):
        raise AssertionError("the fake vLLM sigmoid-gating kernel must never run")

    def fused_recurrent_gated_delta_rule_packed_decode(*args, **kwargs):
        raise AssertionError("the fake vLLM packed-decode kernel must never run")

    def chunk_local_cumsum(g, chunk_size, reverse=False, cu_seqlens=None, chunk_indices=None,
                           head_first=False, output_dtype=torch.float, **kwargs):
        raise AssertionError("the fake vLLM chunk_local_cumsum kernel must never run")

    def fused_post_conv_prep(*args, **kwargs):
        raise AssertionError("the fake vLLM fused_post_conv_prep kernel must never run")

    def chunk_gated_delta_rule(*args, **kwargs):
        raise AssertionError("the fake vLLM FLA chunk chain must never run")

    # --- the MoE call sites, with the *real* signatures --------------------------------
    # vllm/_custom_ops.py:2494-2509.  The real default for `renormalize` is False and the bias is
    # None; both are positional-or-keyword there, and the port must accept them the same way.
    def topk_softmax(
        topk_weights,
        topk_ids,
        token_expert_indices,
        gating_output,
        renormalize=False,
        e_score_correction_bias=None,
    ):
        raise AssertionError("the fake vLLM topk_softmax kernel must never run")

    # vllm/_custom_ops.py:2346-2363.  num_experts BEFORE block_size - the order this test now pins.
    def moe_align_block_size(
        topk_ids,
        num_experts,
        block_size,
        sorted_token_ids,
        experts_ids,
        num_tokens_post_pad,
        expert_map=None,
    ):
        raise AssertionError("the fake vLLM moe_align_block_size kernel must never run")

    # .../fused_moe/fused_moe.py:1474-1485.  The real default is `MoEActivation.SILU`; a string
    # stands in here because the fake cannot import vLLM's enum, and every real call passes it.
    def fused_experts(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation="silu",
        apply_router_weight_on_input=False,
        global_num_experts=-1,
        expert_map=None,
        quant_config=None,
    ):
        raise AssertionError("the fake vLLM fused_experts kernel must never run")

    return {
        "l2norm_fwd": l2norm_fwd,
        "causal_conv1d_fn": causal_conv1d_fn,
        "causal_conv1d_update": causal_conv1d_update,
        "fused_sigmoid_gating_delta_rule_update": fused_sigmoid_gating_delta_rule_update,
        "fused_recurrent_gated_delta_rule_packed_decode":
            fused_recurrent_gated_delta_rule_packed_decode,
        "chunk_local_cumsum": chunk_local_cumsum,
        "fused_post_conv_prep": fused_post_conv_prep,
        "chunk_gated_delta_rule": chunk_gated_delta_rule,
        "topk_softmax": topk_softmax,
        "moe_align_block_size": moe_align_block_size,
        "fused_experts": fused_experts,
    }


def _fake_modules() -> Tuple[Dict[str, types.ModuleType], Dict[str, Any]]:
    """``(name -> module, originals)`` for the transcribed vLLM layout."""
    original = _originals()

    # --- the third kind: ``CustomOp.register_oot`` (custom_op.py:109-128, :322-353) ----------
    # A *class-level* extension point, so the fake has to reproduce the two class methods that make
    # it work, not just a name: ``__new__`` substitutes the registered class for every
    # instantiation, and ``register_oot`` files it under the in-tree name.  The body is the stock
    # OOT state on purpose - ``experts_cls``/``moe_kernel`` are None, so ``is_monolithic`` raises the
    # real ``AttributeError`` until a subclass is registered.
    oot_registry: Dict[str, Any] = {}

    class UnquantizedFusedMoEMethod:
        """``unquantized_fused_moe_method.py:48`` + the ``CustomOp`` members the OOT path uses."""

        name = "unquantized_fused_moe"

        def __new__(cls, *args, **kwargs):
            # custom_op.py:109-128: the in-tree name decides which class is instantiated.
            return object.__new__(oot_registry.get(cls.__name__, cls))

        def __init__(self, moe):
            self.moe = moe
            self.moe_quant_config = None
            self.moe_kernel = None  # :217-222 - the OOT branch builds nothing
            self.experts_cls = None  # oracle/unquantized.py:168-169 - (OOT, None)
            self._forward_method = self.forward_native  # custom_op.py:204-205 (oot -> native)
            self.unquantized_backend = "OOT"

        @property
        def supports_internal_mk(self):
            return self.moe_kernel is not None  # fused_moe_method_base.py:38-41

        @property
        def is_monolithic(self):
            # fused_moe_method_base.py:154-161, verbatim - the call/attribute asymmetry included.
            if self.moe_kernel is None:
                if hasattr(self, "experts_cls"):
                    return self.experts_cls.is_monolithic()
                return False
            return self.moe_kernel.is_monolithic

        def maybe_make_prepare_finalize(self, routing_tables=None):
            raise ValueError("this class should not be called (:70-78)")

        def forward_native(self, layer, x, topk_weights, topk_ids, shared_experts=None,
                           shared_experts_input=None):
            assert self.moe_kernel is not None, "the OOT branch built no kernel (:311)"
            return self.moe_kernel.apply(
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=layer.activation,
                apply_router_weight_on_input=layer.apply_router_weight_on_input,
                global_num_experts=layer.global_num_experts,
                expert_map=layer.expert_map,
                shared_experts=shared_experts,
                shared_experts_input=shared_experts_input,
            )

        def forward(self, *args, **kwargs):
            return self._forward_method(*args, **kwargs)

        def apply(self, layer, x, topk_weights, topk_ids, shared_experts=None,
                  shared_experts_input=None):
            return self.forward(layer=layer, x=x, topk_weights=topk_weights, topk_ids=topk_ids,
                                shared_experts=shared_experts,
                                shared_experts_input=shared_experts_input)

        @classmethod
        def register_oot(cls, _decorated_op_cls=None, name=None):
            # custom_op.py:331-353, reduced: the duplicate assert and the registry write are the
            # behaviour our idempotence depends on.
            def decorator(op_cls):
                reg_name = name if name is not None else cls.__name__
                assert reg_name not in oot_registry, f"Duplicate op name: {reg_name}"
                op_cls.name = reg_name
                oot_registry[reg_name] = op_cls
                return op_cls

            if _decorated_op_cls is None:
                return decorator
            if isinstance(_decorated_op_cls, type):
                return decorator(_decorated_op_cls)
            raise TypeError("Decorator can only be applied to classes.")

    class ChunkGatedDeltaRule:
        """``qwen_gdn_linear_attn.py:290-416``, reduced to the dispatch-relevant part.

        ``__init__`` re-pins ``self._forward_method`` after ``super().__init__()`` - the single most
        important fact about this takeover: it captures a **bound method** at construction time, so
        patching ``forward_oot`` alone (the ladder path) would never be used on a non-CUDA platform.
        """

        gdn_prefill_backend = "triton"

        def __init__(self) -> None:
            self._forward_method = self.forward_native  # :306-311 for a non-CUDA platform

        def forward_native(self, q, k, v, g, beta, initial_state, output_final_state,
                           cu_seqlens=None, chunk_indices=None, chunk_offsets=None,
                           use_qk_l2norm_in_kernel=True, core_attn_out=None):
            raise AssertionError("the fake vLLM FLA chunk chain must never run")

    modules: Dict[str, types.ModuleType] = {}

    root = _pkg("vllm")
    model_executor = _pkg("vllm.model_executor")
    layers = _pkg("vllm.model_executor.layers")
    fla = _pkg("vllm.model_executor.layers.fla")
    fla_ops = _pkg(_FLA)
    l2norm = types.ModuleType(f"{_FLA}.l2norm")
    chunk = types.ModuleType(f"{_FLA}.chunk")
    cumsum = types.ModuleType(f"{_FLA}.cumsum")
    sigmoid_gating = types.ModuleType(f"{_FLA}.fused_sigmoid_gating")
    recurrent = types.ModuleType(f"{_FLA}.fused_recurrent")
    post_conv = types.ModuleType(f"{_FLA}.fused_gdn_prefill_post_conv")
    mamba = _pkg("vllm.model_executor.layers.mamba")
    mamba_ops = _pkg(_MAMBA)
    causal_conv1d = types.ModuleType(f"{_MAMBA}.causal_conv1d")
    gdn_pkg = _pkg("vllm.model_executor.layers.mamba.gdn")
    qwen = types.ModuleType(_GDN)

    # --- the MoE modules (vllm___custom_ops.py, .../fused_moe/fused_moe.py) -------------
    custom_ops = types.ModuleType("vllm._custom_ops")
    fused_moe_pkg = _pkg("vllm.model_executor.layers.fused_moe")
    fused_moe_mod = types.ModuleType("vllm.model_executor.layers.fused_moe.fused_moe")
    # .../fused_moe/unquantized_fused_moe_method.py - the OOT extension point's module.  Both the
    # definition module and the package re-export (:39) carry the class, so the second target in the
    # binding resolves as well.
    unquantized_mod = types.ModuleType(
        "vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method"
    )

    # --- the definition modules ---------------------------------------------------------
    l2norm.l2norm_fwd = original["l2norm_fwd"]
    chunk.l2norm_fwd = original["l2norm_fwd"]  # `from .l2norm import l2norm_fwd` (:17)
    chunk.chunk_gated_delta_rule = original["chunk_gated_delta_rule"]
    cumsum.chunk_local_cumsum = original["chunk_local_cumsum"]
    sigmoid_gating.fused_sigmoid_gating_delta_rule_update = original[
        "fused_sigmoid_gating_delta_rule_update"
    ]
    recurrent.fused_recurrent_gated_delta_rule_packed_decode = original[
        "fused_recurrent_gated_delta_rule_packed_decode"
    ]
    post_conv.fused_post_conv_prep = original["fused_post_conv_prep"]
    causal_conv1d.causal_conv1d_fn = original["causal_conv1d_fn"]
    causal_conv1d.causal_conv1d_update = original["causal_conv1d_update"]

    # --- fla/ops/__init__.py re-exports (:9-25) -----------------------------------------
    fla_ops.chunk_gated_delta_rule = original["chunk_gated_delta_rule"]
    fla_ops.fused_post_conv_prep = original["fused_post_conv_prep"]
    fla_ops.fused_recurrent_gated_delta_rule_packed_decode = original[
        "fused_recurrent_gated_delta_rule_packed_decode"
    ]
    fla_ops.fused_sigmoid_gating_delta_rule_update = original[
        "fused_sigmoid_gating_delta_rule_update"
    ]

    # --- the consumer module: names imported by value (:24-49) --------------------------
    qwen.ChunkGatedDeltaRule = ChunkGatedDeltaRule
    qwen.fla_chunk_gated_delta_rule = original["chunk_gated_delta_rule"]
    qwen.l2norm_fwd = original["l2norm_fwd"]
    qwen.causal_conv1d_fn = original["causal_conv1d_fn"]
    qwen.causal_conv1d_update = original["causal_conv1d_update"]
    qwen.fused_sigmoid_gating_delta_rule_update = original[
        "fused_sigmoid_gating_delta_rule_update"
    ]
    qwen.fused_recurrent_gated_delta_rule_packed_decode = original[
        "fused_recurrent_gated_delta_rule_packed_decode"
    ]
    qwen.fused_post_conv_prep = original["fused_post_conv_prep"]

    # --- the MoE call sites -------------------------------------------------------------
    # `vllm._custom_ops` is a plain module at the package root, not a package: the two wrappers
    # are attributes of it, exactly as `vllm___custom_ops.py` defines them.
    custom_ops.topk_softmax = original["topk_softmax"]
    custom_ops.moe_align_block_size = original["moe_align_block_size"]
    fused_moe_mod.fused_experts = original["fused_experts"]
    unquantized_mod.UnquantizedFusedMoEMethod = UnquantizedFusedMoEMethod
    unquantized_mod.op_registry_oot = oot_registry  # custom_op.py:22 - what the test inspects
    fused_moe_pkg.UnquantizedFusedMoEMethod = UnquantizedFusedMoEMethod  # __init__.py:39

    for name, module in (
        ("vllm", root),
        ("vllm.model_executor", model_executor),
        ("vllm.model_executor.layers", layers),
        ("vllm.model_executor.layers.fla", fla),
        (_FLA, fla_ops),
        (f"{_FLA}.l2norm", l2norm),
        (f"{_FLA}.chunk", chunk),
        (f"{_FLA}.cumsum", cumsum),
        (f"{_FLA}.fused_sigmoid_gating", sigmoid_gating),
        (f"{_FLA}.fused_recurrent", recurrent),
        (f"{_FLA}.fused_gdn_prefill_post_conv", post_conv),
        ("vllm.model_executor.layers.mamba", mamba),
        (_MAMBA, mamba_ops),
        (f"{_MAMBA}.causal_conv1d", causal_conv1d),
        ("vllm.model_executor.layers.mamba.gdn", gdn_pkg),
        (_GDN, qwen),
        ("vllm._custom_ops", custom_ops),
        ("vllm.model_executor.layers.fused_moe", fused_moe_pkg),
        ("vllm.model_executor.layers.fused_moe.fused_moe", fused_moe_mod),
        (
            "vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method",
            unquantized_mod,
        ),
    ):
        modules[name] = module
    return modules, original


def _install_modules(modules: Dict[str, types.ModuleType]) -> Dict[str, Any]:
    """Put ``modules`` into ``sys.modules`` (only the absent ones) and wire parent attributes."""
    created: Dict[str, types.ModuleType] = {}
    for name, module in modules.items():
        if name in sys.modules:
            continue
        sys.modules[name] = module
        created[name] = module
    saved: List[Tuple[Any, str, Any]] = []
    for name in created:
        parent_name, _, attribute = name.rpartition(".")
        parent = sys.modules.get(parent_name)
        if parent is None:
            continue
        previous = getattr(parent, attribute, _MISSING)
        saved.append((parent, attribute, previous))
        setattr(parent, attribute, created[name])
    return {"created": created, "saved": saved}


def _uninstall_modules(state: Dict[str, Any]) -> None:
    for parent, attribute, previous in reversed(state["saved"]):
        if previous is _MISSING:
            try:
                delattr(parent, attribute)
            except AttributeError:  # pragma: no cover - already gone
                pass
        else:
            setattr(parent, attribute, previous)
    for name in state["created"]:
        sys.modules.pop(name, None)


@pytest.fixture(autouse=True)
def fake_vllm(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Fresh fake vLLM + a fresh install state for every test (no cross-test leakage)."""
    monkeypatch.delenv("TILEINFER_OPS", raising=False)  # the suite must not inherit a legacy spec
    if REAL_VLLM:
        install_mod._reset()
        try:
            yield None
        finally:
            install_mod._reset()
        return
    modules, original = _fake_modules()
    state = _install_modules(modules)
    install_mod._reset()
    try:
        yield original
    finally:
        install_mod._reset()
        _uninstall_modules(state)


def _skip_if_real() -> None:
    if REAL_VLLM:
        pytest.skip(
            "a real vLLM is installed: the exact bound/unbound lists describe the transcribed "
            "fake in this file, not the engine on this machine"
        )


def _fake(module: str) -> types.ModuleType:
    return sys.modules[module]


def _is_takeover(obj: Any) -> bool:
    return bool(getattr(obj, "__tileinfer_takeover__", False))


# ---------------------------------------------------------------------------------------
# 1. the patch records themselves
# ---------------------------------------------------------------------------------------


def test_every_takeover_is_a_patch_with_complete_metadata(fake_vllm):
    _skip_if_real()
    install_mod.install_ops()

    ours = {target for binding in install_mod.bindings() for target in binding.targets + binding.aliases}
    registered = [patch for patch in patches.PATCHES if patch.target in ours]
    assert registered, "no patch was registered: the fake vLLM did not expose the targets"

    for patch in registered:
        assert patch.target, "a patch with an empty target"
        assert patch.why, f"{patch.target}: no `why`"
        assert patch.since, f"{patch.target}: no `since`"
        assert patch.remove_when, f"{patch.target}: no `remove_when`"
        assert patch.upstream_ref, f"{patch.target}: no `upstream_ref`"
        assert callable(patch.apply), f"{patch.target}: apply is not callable"
        assert patch.as_dict()["target"] == patch.target
    # One patch per attribute write, and the consumer aliases are their own writes.
    targets = {patch.target for patch in registered}
    assert f"{_FLA}.chunk.l2norm_fwd" in targets
    assert f"{_GDN}.l2norm_fwd" in targets
    assert f"{_GDN}.causal_conv1d_update" in targets


# ---------------------------------------------------------------------------------------
# 2. what binds, what does not - the exact lists
# ---------------------------------------------------------------------------------------


def test_install_binds_the_gdn_family(fake_vllm):
    _skip_if_real()
    report = install_mod.install_ops()

    assert report.bound == EXPECTED_BOUND
    assert report.families == {"gdn": True, "moe": True, "dense": False}
    # `chunk_local_cumsum` is reported as unbound (its ABI is not ours to replace), so the report
    # must not claim `ok` - that is the honest vote, not a bug.
    assert report.ok is False
    assert report.patches is not None and report.patches.applied


def test_install_reports_the_exact_unbound_list(fake_vllm):
    _skip_if_real()
    report = install_mod.install_ops()

    # Exact, not "at least one": a vLLM rename (or a new port in this package) has to move a name
    # between these lists, and that is a deliberate, reviewed change.
    assert report.unbound == EXPECTED_UNBOUND
    assert report.disabled == EXPECTED_DISABLED
    # The single remaining miss says *why*, and names the target it declined.
    assert "not a drop-in" in report.details["chunk_local_cumsum"]
    assert "vllm.model_executor.layers.fla.ops.cumsum.chunk_local_cumsum" in (
        report.details["chunk_local_cumsum"]
    )
    # The three MoE ops bind at the names the real 0.23 sources carry - one name each, no candidate
    # list left to fall through.
    assert report.targets["topk_softmax"] == "vllm._custom_ops.topk_softmax"
    assert report.targets["moe_align_block_size"] == "vllm._custom_ops.moe_align_block_size"
    assert report.targets["fused_experts"] == (
        "vllm.model_executor.layers.fused_moe.fused_moe.fused_experts"
    )
    for op in ("topk_softmax", "moe_align_block_size", "fused_experts"):
        assert report.details[op] == f"bound at {report.targets[op]}"
    # ... and the op that *used* to be reported as unported is bound too, with its exact target.
    assert report.details["fused_post_conv_prep"].startswith(
        f"bound at {_FLA}.fused_gdn_prefill_post_conv.fused_post_conv_prep"
    )


def test_unbound_targets_are_logged_with_their_names(fake_vllm, caplog):
    _skip_if_real()
    with caplog.at_level(logging.WARNING, logger="tileinfer.vllm.ops.install"):
        install_mod.install_ops()
    text = caplog.text
    # The one ABI miss is warned about, with the precise target that was declined.
    assert "TileInfer op unbound: chunk_local_cumsum" in text
    assert "vllm.model_executor.layers.fla.ops.cumsum.chunk_local_cumsum" in text
    # The MoE ops are bound now, so they must NOT be reported as misses any more.
    for op in ("topk_softmax", "moe_align_block_size", "fused_experts", "fused_post_conv_prep"):
        assert f"TileInfer op unbound: {op}" not in text


# ---------------------------------------------------------------------------------------
# 3. the takeover actually reaches our code
# ---------------------------------------------------------------------------------------


def test_attribute_takeover_routes_to_our_implementation(fake_vllm):
    _skip_if_real()
    original = fake_vllm["l2norm_fwd"]
    install_mod.install_ops()

    from tileinfer.vllm.ops.gdn import l2norm_fwd as ours

    module = _fake(f"{_FLA}.l2norm")
    assert _is_takeover(module.l2norm_fwd)
    assert module.l2norm_fwd.__tileinfer_original__ is original
    assert module.l2norm_fwd.__tileinfer_impl__ is ours

    x = torch.randn(4, 8)
    assert torch.equal(module.l2norm_fwd(x), ours(x))
    # Both consumer names too: the GDN layer imported them by value (:32, :46-49).
    assert _is_takeover(_fake(f"{_FLA}.chunk").l2norm_fwd)
    assert _is_takeover(_fake(_GDN).l2norm_fwd)
    assert _is_takeover(_fake(_GDN).causal_conv1d_update)


def _moe_bound(module: str):
    """The bound name inside a fake MoE module (``vllm._custom_ops`` etc.)."""
    return _fake(module)


def test_moe_takeovers_take_the_real_argument_order(fake_vllm):
    """The three MoE takeovers are reached with the argument shape the real 0.23 call sites use.

    Binding the right *name* is only half of it: the port is called by vLLM, so its positional
    arguments must mean the same thing.  Each block below calls the rebound fake module exactly as
    the real call site does (positional, in the real order) and compares with our own function -
    so an argument-order regression (`num_experts`/`block_size`, or
    `expert_map`/`global_num_experts`, which *were* swapped here before the source was read) fails.
    """
    _skip_if_real()
    install_mod.install_ops()

    from tileinfer.vllm.ops import moe as ours

    # --- topk_softmax: vllm/_custom_ops.py:2494-2501 --------------------------------
    logits = torch.randn(6, 8, generator=torch.Generator().manual_seed(3))
    out_a = (
        torch.empty(6, 2),
        torch.empty(6, 2, dtype=torch.int32),
        torch.empty(6, 2, dtype=torch.int32),
    )
    out_b = (out_a[0].clone(), out_a[1].clone(), out_a[2].clone())
    # Six positionals: the two trailing ones (`renormalize`, `e_score_correction_bias`) are
    # positional-or-keyword upstream, so a keyword-only port would fail here.
    _moe_bound("vllm._custom_ops").topk_softmax(*out_a, logits, False, None)
    ours.topk_softmax(*out_b, logits, renormalize=False)
    for got, want in zip(out_a, out_b):
        assert torch.equal(got, want)
    # `renormalize` really is the 5th positional and really normalises.
    _moe_bound("vllm._custom_ops").topk_softmax(*out_a, logits, True)
    assert torch.allclose(out_a[0].sum(-1), torch.ones(6), atol=1e-6)
    # The bias is part of the real signature and is refused with a reason, never ignored.
    with pytest.raises(NotImplementedError, match="e_score_correction_bias"):
        _moe_bound("vllm._custom_ops").topk_softmax(*out_a, logits, False, torch.zeros(8))

    # --- moe_align_block_size: num_experts BEFORE block_size (:2346-2354) -----------
    ids = torch.tensor([[0, 1], [1, 1], [2, 0], [3, 3]], dtype=torch.int32)

    def _slots(n: int):
        return (
            torch.empty(n, dtype=torch.int32),
            torch.empty(n, dtype=torch.int32),
            torch.empty(1, dtype=torch.int32),
        )

    got, want = _slots(16), _slots(16)
    # 8 pairs (4 tokens x top_k 2) -> counts 2/3/1/2 over the 4 experts -> tiles 1/2/1/1 with
    # block_size 2 -> 5 tiles -> 10 slots.  Every number here depends on reading the two scalars in
    # the real order (num_experts, then block_size).
    _moe_bound("vllm._custom_ops").moe_align_block_size(ids, 4, 2, *got)
    assert int(got[2]) == 10, "num_experts/block_size are not in the real order"
    ours.moe_align_block_size(ids, 4, 2, *want)
    for got_t, want_t in zip(got, want):
        assert torch.equal(got_t, want_t)
    # The order is discriminated: swapping the two scalars would read expert ids 0..3 as an
    # out-of-range top_k for num_experts=2 (the port validates instead of writing garbage).
    with pytest.raises(ValueError, match="num_experts"):
        _moe_bound("vllm._custom_ops").moe_align_block_size(ids, 2, 4, *_slots(16))
    # expert_map is part of the real signature and is refused, never ignored.
    with pytest.raises(NotImplementedError, match="expert_map"):
        _moe_bound("vllm._custom_ops").moe_align_block_size(
            ids, 4, 2, *_slots(16), torch.arange(4, dtype=torch.int32)
        )

    # --- fused_experts: the wrapper's ten positionals (:1474-1485) -------------------
    gen = torch.Generator().manual_seed(11)
    x = torch.randn(3, 8, generator=gen)
    w1 = torch.randn(4, 6, 8, generator=gen)
    w2 = torch.randn(4, 8, 3, generator=gen)
    weights = torch.rand(3, 2, generator=gen)
    routed = torch.tensor([[0, 1], [2, 3], [1, 0]])
    expect = ours.fused_experts(x, w1, w2, weights, routed)

    call = _moe_bound("vllm.model_executor.layers.fused_moe.fused_moe").fused_experts
    # activation, apply_router_weight_on_input, global_num_experts, expert_map, quant_config.
    assert torch.equal(call(x, w1, w2, weights, routed, "silu", False, -1, None, None), expect)

    # `global_num_experts` is the 8th positional and `expert_map` the 9th - the order that used to
    # be reversed here.  An identity map must be a no-op, and it only is if they land correctly.
    identity = torch.arange(4, dtype=torch.int32)
    assert torch.equal(call(x, w1, w2, weights, routed, "silu", False, 4, identity, None), expect)

    # The real wrapper hands us the *enum*; the inner op gets `activation.value` (a string).
    class _Silu:
        value = "silu"

    assert torch.equal(call(x, w1, w2, weights, routed, _Silu(), False, -1, None, None), expect)

    # quant_config: an unquantised config is accepted, a quantised or unclassifiable one is refused.
    class _Unquantized:
        use_fp8_w8a8 = use_int8_w8a8 = use_int8_w8a16 = use_int4_w4a16 = False
        per_act_token_quant = per_channel_quant = False
        ocp_mx_scheme = block_shape = None
        w1_scale = w2_scale = w1_zp = w2_zp = None
        a1_scale = a2_scale = w1_bias = w2_bias = None

    assert torch.equal(call(x, w1, w2, weights, routed, "silu", False, -1, None, _Unquantized()),
                       expect)

    class _Fp8(_Unquantized):
        use_fp8_w8a8 = True

    with pytest.raises(NotImplementedError, match="use_fp8_w8a8"):
        call(x, w1, w2, weights, routed, "silu", False, -1, None, _Fp8())
    with pytest.raises(NotImplementedError, match="exposes none of"):
        call(x, w1, w2, weights, routed, "silu", False, -1, None, object())


def _gdn_inputs(dtype=torch.float32, total=3, k_heads=2, v_heads=2, dim=4):
    gen = torch.Generator().manual_seed(7)
    k = torch.nn.functional.normalize(torch.randn(1, total, k_heads, dim, generator=gen), dim=-1)
    q = torch.randn(1, total, k_heads, dim, generator=gen)
    v = torch.randn(1, total, v_heads, dim, generator=gen)
    inc = (-torch.rand(1, total, v_heads, generator=gen) * 0.05).to(torch.float32)
    beta = (0.1 + 0.5 * torch.rand(1, total, v_heads, generator=gen)).to(torch.float32)
    state = torch.zeros(1, v_heads, dim, dim, dtype=torch.float32)
    return q.to(dtype), k.to(dtype), v.to(dtype), inc, beta, state


def test_custom_op_takeover_hijacks_the_pinned_forward_method(fake_vllm):
    """The load-bearing mechanism test: ``__init__`` pins ``forward_native``, so that is what runs."""
    _skip_if_real()
    install_mod.install_ops()

    from tileinfer.vllm.ops.fla import chunk_gated_delta_rule

    cls = _fake(_GDN).ChunkGatedDeltaRule
    assert _is_takeover(cls.forward_native)
    assert cls.forward_oot is cls.forward_native  # the ladder path, custom_op.py:204-205

    q, k, v, g, beta, state = _gdn_inputs()
    expected_o, expected_state = chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        initial_state=state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
    )

    instance = cls()  # __init__ captures self._forward_method = self.forward_native
    o, final_state = instance._forward_method(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=state,
        output_final_state=True,
        cu_seqlens=None,
        chunk_indices=None,
        chunk_offsets=None,
        use_qk_l2norm_in_kernel=False,
    )
    assert torch.equal(o, expected_o)
    assert torch.equal(final_state, expected_state)


def test_custom_op_takeover_fills_core_attn_out_when_asked(fake_vllm):
    _skip_if_real()
    install_mod.install_ops()
    cls = _fake(_GDN).ChunkGatedDeltaRule
    q, k, v, g, beta, state = _gdn_inputs()
    buffer = torch.zeros(1, 3, 2, 4)
    o, _ = cls.forward_native(
        None,
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
        core_attn_out=buffer,
    )
    assert torch.equal(buffer.reshape(-1), o.reshape(-1))  # qwen_gdn_linear_attn.py:339-343


# ---------------------------------------------------------------------------------------
# 3b. the third kind: a class-level ``register_oot`` (no attribute is rebound)
# ---------------------------------------------------------------------------------------

_OOT_MODULE = "vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method"
_OOT_TARGET = f"{_OOT_MODULE}.UnquantizedFusedMoEMethod"


def _moe_config():
    """A ``FusedMoEConfig`` stub with the members ``fused_moe_oot`` reads (``config.py:1235-1269``)."""
    return types.SimpleNamespace(
        num_experts=4,
        in_dtype=torch.bfloat16,
        is_act_and_mul=True,
        has_bias=False,
        is_lora_enabled=False,
        moe_parallel_config=types.SimpleNamespace(
            tp_size=1, dp_size=1, ep_size=1, pcp_size=1
        ),
    )


def _oot_layer(dtype=torch.bfloat16):
    """The four layer attributes the stock ``forward_native`` reads (:312-324)."""
    gen = torch.Generator().manual_seed(31)
    return types.SimpleNamespace(
        w13_weight=torch.randn(4, 6, 8, generator=gen).to(dtype),
        w2_weight=torch.randn(4, 8, 3, generator=gen).to(dtype),
        activation="silu",
        apply_router_weight_on_input=False,
        global_num_experts=4,
        expert_map=None,
    )


def test_custom_op_oot_registers_a_class_instead_of_rebinding_a_method(fake_vllm):
    """The load-bearing mechanism test for the third kind.

    ``kind="custom_op_oot"`` is not "an attribute write with a different value": it is vLLM's
    *documented* extension point (``custom_op.py:326-329``), where ``CustomOp.__new__``
    (:109-128) substitutes the registered class for every ``UnquantizedFusedMoEMethod(...)`` -
    which is what ``layer.py:359`` does while the model is loading.  So the assertions are:
    the registry holds our subclass under the in-tree name, instantiating the in-tree class builds
    it, the load-time read that used to raise now answers ``False``, and the in-tree class itself
    was **not** modified (unlike ``custom_op``, which writes two methods onto the class).
    """
    _skip_if_real()
    module = _fake(_OOT_MODULE)
    stock = module.UnquantizedFusedMoEMethod
    forward_before = stock.forward_native
    moe_config = _moe_config()

    # The stock OOT state is the bug: no kernel, experts_cls is None, and is_monolithic calls it.
    assert module.op_registry_oot == {}
    with pytest.raises(AttributeError, match="is_monolithic"):
        stock(moe_config).is_monolithic

    report = install_mod.install_ops()
    assert "unquantized_fused_moe" in report.bound
    assert report.targets["unquantized_fused_moe"] == _OOT_TARGET

    # 1. the mechanism: register_oot holds our subclass under the *in-tree* name ...
    (registered,) = module.op_registry_oot.values()
    from tileinfer.vllm.ops import fused_moe_oot

    assert registered.__name__ == fused_moe_oot.OOT_CLASS_NAME
    assert issubclass(registered, stock)
    assert registered is not stock

    # 2. ... so instantiating the in-tree class builds the subclass (custom_op.py:109-128) ...
    instance = stock(moe_config)  # exactly what layer.py:359 does
    assert type(instance) is registered
    # 3. ... and the two reads the loader makes now answer instead of raising.
    assert instance.is_monolithic is False
    assert instance.supports_internal_mk is True  # fused_moe_method_base.py:38-41

    # 4. the class attribute was NOT rebound: this kind is class-level only.
    assert stock.forward_native is forward_before
    assert not _is_takeover(stock.forward_native)

    # 5. the Patch record carries the metadata a future maintainer needs.
    patch = next(item for item in patches.PATCHES if item.target == _OOT_TARGET)
    assert patch.why and patch.since and patch.upstream_ref
    assert patch.remove_when == "upstream builds an OOT moe_kernel"
    assert "register_oot" in patch.why

    # 6. idempotence: a reinstall after `_reset()` must not stack a second registration (the fake's
    #    register_oot asserts on a duplicate, as upstream's does).
    install_mod._reset()
    assert "unquantized_fused_moe" in install_mod.install_ops().bound
    assert len(module.op_registry_oot) == 1


def test_custom_op_oot_routes_the_step_through_the_moe_port(fake_vllm):
    """After registration the whole step chain reaches ``moe.fused_experts`` - and only there.

    ``attach -> MoERunner._apply_quant_method`` calls ``quant_method.apply``, which is
    ``forward`` -> ``_forward_method`` (the OOT ladder's ``forward_oot`` -> ``forward_native``,
    ``custom_op.py:204-205``) -> the stock ``forward_native`` (:311-324) -> ``moe_kernel.apply``.
    The fake's ``forward_native`` asserts ``moe_kernel is not None`` (the stock assertion at :311),
    so this fails loudly if the subclass ever stops filling the kernel slot.
    """
    _skip_if_real()
    install_mod.install_ops()

    from tileinfer.vllm.ops import moe as ours

    stock = _fake(_OOT_MODULE).UnquantizedFusedMoEMethod
    instance = stock(_moe_config())
    layer = _oot_layer(torch.float32)
    gen = torch.Generator().manual_seed(41)
    x = torch.randn(3, 8, generator=gen)
    weights = torch.rand(3, 2, generator=gen)
    ids = torch.tensor([[0, 1], [2, 3], [1, 1]])

    got = instance.apply(
        layer=layer,
        x=x,
        topk_weights=weights,
        topk_ids=ids,
        shared_experts=object(),  # the runner always passes them; they are added by the runner
        shared_experts_input=x,
    )
    want = ours.fused_experts(
        x, layer.w13_weight, layer.w2_weight, weights, ids, activation=layer.activation
    )
    assert torch.equal(got, want)


def test_custom_op_oot_registers_nothing_when_the_family_is_off(monkeypatch, fake_vllm):
    """A switched-off family leaves upstream byte-identical (the stock crash is then vLLM's)."""
    _skip_if_real()
    monkeypatch.setenv("TILEINFER_OPS", "gdn")  # moe off
    report = install_mod.install_ops()

    assert "unquantized_fused_moe" in report.disabled
    module = _fake(_OOT_MODULE)
    assert module.op_registry_oot == {}
    with pytest.raises(AttributeError, match="is_monolithic"):
        module.UnquantizedFusedMoEMethod(_moe_config()).is_monolithic


# ---------------------------------------------------------------------------------------
# 4. the switch
# ---------------------------------------------------------------------------------------


def test_switch_defaults_are_the_families_the_model_needs(monkeypatch):
    monkeypatch.delenv("TILEINFER_OPS", raising=False)
    assert switch.ops_enabled("gdn") is True
    assert switch.ops_enabled("moe") is True
    assert switch.ops_enabled("dense") is False
    assert set(switch.OPS_DEFAULT_ON) == {"gdn", "moe"}


def test_switch_honours_the_environment(monkeypatch):
    monkeypatch.setenv("TILEINFER_OPS", "dense")
    assert switch.ops_enabled("gdn") is False
    assert switch.ops_enabled("moe") is False
    assert switch.ops_enabled("dense") is True

    monkeypatch.setenv("TILEINFER_OPS", "gdn,moe")
    assert switch.ops_enabled("gdn") is True and switch.ops_enabled("dense") is False

    monkeypatch.setenv("TILEINFER_OPS", "  GDN  ,  moe ")  # case/whitespace, as the legacy switch
    assert switch.ops_enabled("gdn") is True

    monkeypatch.setenv("TILEINFER_OPS", "")  # blank means "default", not "nothing"
    assert switch.ops_enabled("gdn") is True


def test_switch_raises_on_a_typo_but_install_does_not(monkeypatch, fake_vllm):
    _skip_if_real()
    monkeypatch.setenv("TILEINFER_OPS", "gdn,typo")
    with pytest.raises(ValueError, match="typo"):
        switch.ops_enabled("gdn")
    # The tolerant form falls back to the defaults: a typo must not take the model down.
    assert switch.enabled_families() == set(switch.OPS_DEFAULT_ON)

    report = install_mod.install_ops()  # no raise
    assert report.families == {"gdn": True, "moe": True, "dense": False}


def test_install_skips_the_disabled_family(monkeypatch, fake_vllm):
    _skip_if_real()
    monkeypatch.setenv("TILEINFER_OPS", "moe")  # gdn off
    report = install_mod.install_ops()

    assert "chunk_gated_delta_rule" in report.disabled
    assert "causal_conv1d_update" in report.disabled
    assert not _is_takeover(_fake(_GDN).causal_conv1d_update)  # nothing was rebound
    assert not _is_takeover(_fake(_GDN).ChunkGatedDeltaRule.forward_native)
    # The MoE family is what is on, so exactly its four ops bind (and no GDN op does).
    assert report.bound == [
        "topk_softmax",
        "moe_align_block_size",
        "fused_experts",
        "unquantized_fused_moe",
    ]
    assert not _is_takeover(_fake(f"{_FLA}.l2norm").l2norm_fwd)


def test_disabled_family_falls_back_to_the_original_at_call_time(monkeypatch, fake_vllm):
    """The per-call guard: flipping the switch off hands the call back to vLLM's own object."""
    _skip_if_real()
    import os

    original = fake_vllm["l2norm_fwd"]
    from tileinfer.vllm.ops.gdn import l2norm_fwd as ours

    x = torch.randn(2, 4)
    os.environ["TILEINFER_OPS"] = "gdn"  # applied while the switch says on
    try:
        install_mod.install_ops()
        module = _fake(f"{_FLA}.l2norm")
        assert torch.equal(module.l2norm_fwd(x), ours(x))
        os.environ["TILEINFER_OPS"] = "moe"  # now off, per call, no reinstall
        with pytest.raises(AssertionError, match="must never run"):
            module.l2norm_fwd(x)
        assert module.l2norm_fwd.__tileinfer_original__ is original
    finally:
        os.environ.pop("TILEINFER_OPS", None)


# ---------------------------------------------------------------------------------------
# 5. idempotence and a missing target
# ---------------------------------------------------------------------------------------


def test_install_is_idempotent(fake_vllm):
    _skip_if_real()
    first = install_mod.install_ops()
    registered = len(patches.PATCHES)

    second = install_mod.install_ops()
    assert second is first
    assert (second.bound, second.unbound, second.disabled) == (
        first.bound,
        first.unbound,
        first.disabled,
    )
    assert len(patches.PATCHES) == registered

    # A fresh install after `_reset()` must not stack a wrapper on a wrapper.
    install_mod._reset()
    third = install_mod.install_ops()
    assert third.bound == first.bound
    assert _fake(f"{_FLA}.l2norm").l2norm_fwd.__tileinfer_original__ is fake_vllm["l2norm_fwd"]


def test_a_missing_target_does_not_raise(monkeypatch, fake_vllm):
    _skip_if_real()
    install_mod.install_ops()
    assert "fused_recurrent_gated_delta_rule_packed_decode" in install_mod.install_ops().bound

    # Make the definition module and the consumer alias both vanish, as a vLLM rename would.
    monkeypatch.delitem(sys.modules, f"{_FLA}.fused_recurrent")
    monkeypatch.delattr(_fake(_FLA), "fused_recurrent")
    monkeypatch.delattr(_fake(_GDN), "fused_recurrent_gated_delta_rule_packed_decode")
    install_mod._reset()

    report = install_mod.install_ops()  # must not raise
    assert "fused_recurrent_gated_delta_rule_packed_decode" in report.unbound
    assert "no vLLM target resolves" in report.details[
        "fused_recurrent_gated_delta_rule_packed_decode"
    ]
    assert "chunk_gated_delta_rule" in report.bound  # the rest still binds


def test_install_without_any_vllm_does_not_raise(monkeypatch):
    """No vLLM at all (the CPU box): every op is unbound, nothing raises, the report is honest."""
    _skip_if_real()
    for name in [name for name in list(sys.modules) if name == "vllm" or name.startswith("vllm.")]:
        monkeypatch.delitem(sys.modules, name)
    install_mod._reset()
    report = install_mod.install_ops()
    assert report.bound == []
    assert set(report.unbound) | set(report.disabled) == {
        binding.op for binding in install_mod.bindings()
    }
    assert report.patches is not None and not report.patches.applied


# ---------------------------------------------------------------------------------------
# 6. the package surface, the plugin edit, and the import guards
# ---------------------------------------------------------------------------------------


def test_the_ops_are_exported_from_the_package():
    import tileinfer.vllm.ops as ops
    import tileinfer.vllm.ops.fla as fla

    for name in ("l2norm_fwd", "chunk_local_cumsum", "topk_softmax", "fused_experts",
                 "moe_align_block_size", "silu_and_mul"):
        assert name in ops.__all__, f"ops.{name} is not exported"
        assert callable(getattr(ops, name))
    for name in ("chunk_gated_delta_rule", "chunk_gated_delta_rule_fwd", "FLA_CHUNK_SIZE",
                 "solve_tril", "causal_conv1d_fn", "causal_conv1d_update",
                 "fused_sigmoid_gating_delta_rule_update",
                 "fused_recurrent_gated_delta_rule_packed_decode", "chunk_fwd_o",
                 "chunk_gated_delta_rule_fwd_h", "chunk_scaled_dot_kkt_fwd", "recompute_w_u_fwd"):
        assert name in fla.__all__, f"fla.{name} is not exported"
        assert getattr(fla, name) is not None


def test_plugin_runs_the_operator_wiring_as_its_last_step():
    from tileinfer.vllm import plugin

    source = inspect.getsource(plugin.install)
    order = [
        source.index("_install_patches"),
        source.index("_install_backend"),
        source.index("_install_graph_bridge"),
        source.index("_install_ops"),
    ]
    assert order == sorted(order), "install_ops() must be the last step (it patches model modules)"
    assert plugin.ENV_OPS == "TILEINFER_OPS"
    assert "ops" in plugin.status()  # status() reports the operator wiring, and must not raise
    assert isinstance(plugin._install_ops(), dict)  # never raises, even with no vLLM targets


def _module_level_imports(tree: ast.Module) -> List[Tuple[str, int]]:
    found: List[Tuple[str, int]] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            found.extend((alias.name, node.lineno) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if base:
                found.append((base, node.lineno))
            found.extend(
                (f"{base}.{alias.name}" if base else alias.name, node.lineno)
                for alias in node.names
                if alias.name
            )
    return found


def test_install_and_switch_never_import_vllm_at_module_level():
    """``tileinfer.vllm.ops`` must stay importable on a box without vLLM / torch_npu."""
    root = pathlib.Path(__file__).resolve().parents[1] / "tileinfer" / "vllm" / "ops"
    for path in (root / "install.py", root / "switch.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        offenders = [
            f"{module} (line {line})"
            for module, line in _module_level_imports(tree)
            if module == "vllm" or module.startswith("vllm.") or module.startswith("torch_npu")
            or module.startswith("vllm_ascend")
        ]
        assert not offenders, f"{path.name} imports engine code at module level: {offenders}"
    # The switch is a reimplementation of the convention, not a re-export of the legacy adapter:
    # the only place the legacy module may be named is a comment/docstring, never an import.
    tree = ast.parse((root / "switch.py").read_text(encoding="utf-8"), filename="switch.py")
    imported = [module for module, _line in _all_imports(tree)]
    assert not [module for module in imported if "vllm_ascend" in module]


def _all_imports(tree: ast.Module) -> List[Tuple[str, int]]:
    found: List[Tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((alias.name, node.lineno) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if base:
                found.append((base, node.lineno))
            found.extend(
                (f"{base}.{alias.name}" if base else alias.name, node.lineno)
                for alias in node.names
                if alias.name
            )
    return found
