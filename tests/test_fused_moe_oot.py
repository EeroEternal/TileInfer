"""Contract tests for ``tileinfer.vllm.ops.fused_moe_oot`` - the OOT MoE method, with no vLLM.

What is under test, and why it can be tested on a CPU box
--------------------------------------------------------
The gap this module closes is a **Python-level** contract, not a numeric one: vLLM 0.23's OOT branch
returns ``(UnquantizedMoeBackend.OOT, None)`` (``oracle/unquantized.py:168-169``), never builds a
kernel (``unquantized_fused_moe_method.py:217-222``), and the base class then calls
``None.is_monolithic()`` (``fused_moe_method_base.py:156-158``) while the model is loading.  So the
things worth pinning are all observable without hardware:

1. the stand-ins answer **exactly** the attribute accesses the source shows - and note that the two
   are different shapes (``experts_cls.is_monolithic()`` is *called*, ``moe_kernel.is_monolithic`` is
   *read*).  A plain truthy attribute where the caller calls it must fail, and
   :func:`test_a_plain_truthy_is_monolithic_would_still_crash_the_base_class` proves the test can
   tell the difference;
2. the subclass fills the kernel slot the OOT branch leaves empty, with every member the method and
   the runner read (``output_is_reduced``, ``prepare_finalize.topk_indices_dtype``, ...);
3. the compute path is ``tileinfer.vllm.ops.moe.fused_experts`` - the *same object* the
   ``fused_experts`` binding wraps - reached through the ``moe_kernel.apply`` slot, byte for byte;
4. a quantised / biased / LoRA / non-gated / DP-EP-PCP config is refused loudly at construction, and
   a quantised *weight* at call time;
5. the family record carries the metadata a future maintainer needs, with
   ``remove_when="upstream builds an OOT moe_kernel"``.

``_StockBase`` below is a reduced transcription of ``UnquantizedFusedMoEMethod`` +
``FusedMoEMethodBase``: every method carries the line number it comes from, so a reader can check
the claim against ``runtime/vllm-api/`` instead of trusting this file.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
from typing import Any, Dict, Optional

import pytest
import torch

from tileinfer.vllm.ops import fused_moe_oot
from tileinfer.vllm.ops import moe as moe_ops
from tileinfer.vllm.ops.families import gdn_moe

ROOT = pathlib.Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------------------
# The reduced stock base class: one method per line the OOT path touches
# ---------------------------------------------------------------------------------------


class _ParallelConfig:
    """The three sizes ``check_unquantized_config`` refuses above 1 (``config.py:999-1012``)."""

    def __init__(self, tp_size: int = 1, dp_size: int = 1, ep_size: int = 1, pcp_size: int = 1):
        self.tp_size = tp_size
        self.dp_size = dp_size
        self.ep_size = ep_size
        self.pcp_size = pcp_size


class _Config:
    """``FusedMoEConfig`` reduced to the members this port reads (``config.py:1235-1269``)."""

    def __init__(self, **overrides: Any) -> None:
        self.num_experts = 4
        self.in_dtype = torch.bfloat16  # :1248
        self.is_act_and_mul = True  # :1261
        self.has_bias = False  # :1260
        self.is_lora_enabled = False  # :1262
        self.moe_parallel_config = _ParallelConfig()  # :1245
        for name, value in overrides.items():
            setattr(self, name, value)


class _StockBase:
    """``UnquantizedFusedMoEMethod`` + ``FusedMoEMethodBase``, reduced to what OOT touches.

    The stock OOT state is the point: ``experts_cls`` is ``None`` because the oracle returns
    ``(OOT, None)`` (``oracle/unquantized.py:168-169``) and ``moe_kernel`` stays ``None`` because
    ``process_weights_after_loading`` returns before ``_setup_kernel``
    (``unquantized_fused_moe_method.py:217-222``).
    """

    def __init__(self, moe: Any) -> None:
        self.moe = moe  # fused_moe_method_base.py:33
        self.moe_quant_config: Optional[Any] = None  # :34
        self.moe_kernel: Optional[Any] = None  # :35
        self.unquantized_backend = "OOT"  # enum member in vLLM; the name is all we read
        self.experts_cls: Optional[Any] = None  # oracle/unquantized.py:168-169
        # custom_op.py:204-205: on OOT the ladder ends at forward_oot -> forward_native.
        self._forward_method = self.forward_native

    # -- FusedMoEMethodBase ------------------------------------------------------------------

    @property
    def supports_internal_mk(self) -> bool:
        return self.moe_kernel is not None  # :38-41

    @property
    def mk_can_overlap_shared_experts(self) -> bool:
        return self.moe_kernel is not None and self.moe_kernel.can_overlap_shared_experts  # :43-49

    @property
    def topk_indices_dtype(self) -> Optional[torch.dtype]:
        if self.moe_kernel is not None:
            return self.moe_kernel.prepare_finalize.topk_indices_dtype()  # :137-138
        return None

    @property
    def is_monolithic(self) -> bool:
        # :154-161, verbatim shape - including the call/attribute asymmetry.
        if self.moe_kernel is None:
            if hasattr(self, "experts_cls"):
                return self.experts_cls.is_monolithic()
            else:
                return False
        return self.moe_kernel.is_monolithic

    def maybe_make_prepare_finalize(self, routing_tables: Any = None) -> Any:
        # unquantized_fused_moe_method.py:70-78: "this function should not be called".
        raise ValueError(f"{type(self).__name__} uses the new modular kernel initialization logic")

    def get_fused_moe_quant_config(self, layer: Any) -> None:
        return None  # unquantized_fused_moe_method.py:275-282 (unbiased branch)

    # -- the step ---------------------------------------------------------------------------

    def forward_native(
        self,
        layer: Any,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: Any = None,
        shared_experts_input: Any = None,
    ) -> torch.Tensor:
        assert self.moe_kernel is not None  # unquantized_fused_moe_method.py:311
        return self.moe_kernel.apply(  # :312-324, every argument by keyword
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

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self._forward_method(*args, **kwargs)  # custom_op.py:135-136

    def apply(
        self,
        layer: Any,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: Any = None,
        shared_experts_input: Any = None,
    ) -> torch.Tensor:
        return self.forward(  # unquantized_fused_moe_method.py:284-300
            layer=layer,
            x=x,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )

    def process_weights_after_loading(self, layer: Any) -> None:
        # unquantized_fused_moe_method.py:217-222: the OOT early return.  No kernel is built, no
        # weight format is converted.
        return None


class _Layer:
    """The two weight tensors and the four flags the stock ``forward_native`` reads."""

    def __init__(
        self,
        *,
        dtype: torch.dtype = torch.bfloat16,
        seed: int = 5,
        num_experts: int = 4,
        hidden: int = 8,
        inter: int = 3,
    ) -> None:
        gen = torch.Generator().manual_seed(seed)
        self.w13_weight = torch.randn(num_experts, 2 * inter, hidden, generator=gen).to(dtype)
        self.w2_weight = torch.randn(num_experts, hidden, inter, generator=gen).to(dtype)
        self.activation = "silu"
        self.apply_router_weight_on_input = False
        self.global_num_experts = num_experts
        self.expert_map = None


def _build(**overrides: Any):
    """A method built by the public factory, exactly as ``install.py`` builds it."""
    return fused_moe_oot.make_oot_method(_StockBase)(_Config(**overrides))


# ---------------------------------------------------------------------------------------
# 1. the stand-ins answer exactly what the base class reads
# ---------------------------------------------------------------------------------------


def test_the_experts_stand_in_is_called_and_answers_no() -> None:
    """``fused_moe_method_base.py:157-158``: ``hasattr`` then ``experts_cls.is_monolithic()``.

    The OOT oracle returns ``None`` for this slot (``oracle/unquantized.py:168-169``), which is the
    ``AttributeError`` the whole gap is about.  Our stand-in answers ``False`` - the runner must take
    the modular branch (``router.select_experts`` -> ``apply``, ``moe_runner.py:516-539``), not
    ``apply_monolithic``.
    """
    experts = fused_moe_oot.TorchExperts()

    assert hasattr(experts, "is_monolithic"), "the base class checks hasattr before calling"
    assert callable(experts.is_monolithic), "the base class writes is_monolithic() - it must be callable"
    assert experts.is_monolithic() is False, "True would send the step to apply_monolithic"
    # A *method*, not a property and not an attribute: `type(experts).__dict__` is where the base
    # class's `experts_cls.is_monolithic()` looks.
    assert inspect.isfunction(type(experts).__dict__["is_monolithic"])
    # And it is the only member the local source reads from experts_cls; the other uses pass it to
    # make_unquantized_moe_kernel (oracle/unquantized.py:329-360), which OOT never reaches.
    public = {name for name in vars(fused_moe_oot.TorchExperts) if not name.startswith("_")}
    assert public == {"is_monolithic"}


def test_the_kernel_stand_in_is_read_and_answers_no() -> None:
    """``moe_kernel``'s members, each with the line that reads it - note: no parentheses here."""
    kernel = fused_moe_oot.TorchMoEKernel()

    # fused_moe_method_base.py:161 reads the ATTRIBUTE (contrast with experts_cls above).
    assert kernel.is_monolithic is False
    assert not callable(kernel.is_monolithic), "base:161 does not call it: a method here is wrong too"
    assert not inspect.isroutine(type(kernel).__dict__["is_monolithic"])

    assert kernel.can_overlap_shared_experts is False  # base:43-49 -> shared_experts.py:92-112
    assert kernel.output_is_reduced() is False  # moe_runner.py:369-374
    assert kernel.prepare_finalize.topk_indices_dtype() is None  # base:137-138


def test_the_base_class_reads_both_shapes_and_a_plain_truthy_would_still_crash_it() -> None:
    """The discrimination proof: the stock crash, our answer, and why an attribute is not enough.

    Three states of the same base-class read:

    1. stock OOT (``experts_cls is None``) -> ``AttributeError`` - the bug;
    2. ours -> ``False`` through *both* branches (the kernel one, and - with the kernel removed -
       the ``experts_cls`` one, so the stand-in is not decorative);
    3. a stand-in whose ``is_monolithic`` is a plain truthy non-callable -> ``TypeError``.  This is
       the shape an ``attribute`` patch would have produced (rebinding a property with a value),
       and it is why the ledger rejected that route: a test that cannot tell 2 from 3 is worthless.
    """
    # 1. the stock OOT state, exactly as the model loader meets it (layer.py:457).
    stock = _StockBase(_Config())
    with pytest.raises(AttributeError, match="is_monolithic"):
        stock.is_monolithic

    # 2. through the kernel slot (base:161) ...
    method = _build()
    assert method.is_monolithic is False

    # ... and through the experts_cls slot (base:156-158), which is what we would still answer if a
    # future vLLM read the property before attaching a kernel.
    method.moe_kernel = None
    assert method.is_monolithic is False
    assert isinstance(method.experts_cls, fused_moe_oot.TorchExperts)

    # 3. a plain truthy value must NOT be accepted - the base class calls it.
    class _PlainTruthy:
        is_monolithic = True

    stock.experts_cls = _PlainTruthy()
    with pytest.raises(TypeError, match="not callable"):
        stock.is_monolithic


def test_the_method_fills_the_kernel_slot_the_oot_branch_leaves_empty() -> None:
    """Every member the method/runner read through ``moe_kernel``, on the built class."""
    method = _build()

    assert isinstance(method.moe_kernel, fused_moe_oot.TorchMoEKernel)
    # The consequence we want: layer.py:457 returns immediately, so maybe_init_modular_kernel is
    # never entered (its next statement - maybe_make_prepare_finalize - raises on this class).
    assert method.supports_internal_mk is True
    assert method.mk_can_overlap_shared_experts is False
    assert method.topk_indices_dtype is None
    # Belt and braces, documented in make_oot_method: if it *is* called, "no all2all" is the answer,
    # not the inherited ValueError (unquantized_fused_moe_method.py:70-78).
    assert method.maybe_make_prepare_finalize() is None
    assert method.maybe_make_prepare_finalize(routing_tables=None) is None
    # No EPLB interface on the stand-in kernel: advertise that instead of the inherited True
    # (unquantized_fused_moe_method.py:66-68), so layer.py:374 refuses loudly.
    assert method.supports_eplb is False


def test_process_weights_after_loading_keeps_the_stand_in_and_detects_a_real_kernel() -> None:
    """The OOT early return is load-bearing: it is what keeps the checkpoint weight layout."""
    method = _build()
    method.process_weights_after_loading(_Layer())  # no raise, no shuffle
    assert isinstance(method.moe_kernel, fused_moe_oot.TorchMoEKernel)

    class _FutureBase(_StockBase):
        """A vLLM that stopped returning early for OOT and built a real kernel."""

        def process_weights_after_loading(self, layer: Any) -> None:
            self.moe_kernel = object()

    future = fused_moe_oot.make_oot_method(_FutureBase)(_Config())
    with pytest.raises(RuntimeError, match="remove_when|upstream builds an OOT moe_kernel|no longer"):
        future.process_weights_after_loading(_Layer())


# ---------------------------------------------------------------------------------------
# 2. the compute path is the existing MoE port, through the kernel slot
# ---------------------------------------------------------------------------------------


def test_the_oot_method_reuses_the_moe_port_by_identity() -> None:
    """Reuse, not a reimplementation: the object the ``fused_experts`` binding wraps, verbatim."""
    assert fused_moe_oot.fused_experts is moe_ops.fused_experts
    record = {binding.op: binding for binding in gdn_moe.bind()}["fused_experts"]
    assert record.impl is fused_moe_oot.fused_experts
    # ... and the OOT binding is a *different* op name (a different mechanism), so neither record
    # shadows the other in the report.
    ops = [binding.op for binding in gdn_moe.bind()]
    assert ops.count("fused_experts") == 1 and ops.count("unquantized_fused_moe") == 1


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_the_compute_path_equals_fused_experts(dtype: torch.dtype) -> None:
    """``apply`` (the runner's call) == ``moe.fused_experts`` with the same arguments, bit for bit."""
    method = _build()
    layer = _Layer(dtype=dtype, seed=11)
    gen = torch.Generator().manual_seed(11)
    x = torch.randn(3, 8, generator=gen).to(dtype)
    weights = torch.rand(3, 2, generator=gen)
    # A duplicated id (token 0 takes expert 1 twice) is legal routing and must accumulate.
    ids = torch.tensor([[0, 1], [2, 3], [1, 1]])

    want = moe_ops.fused_experts(
        x, layer.w13_weight, layer.w2_weight, weights, ids, activation=layer.activation
    )
    got = method.apply(
        layer=layer,
        x=x,
        topk_weights=weights,
        topk_ids=ids,
        # Qwen3.6 always has a shared expert, so the runner always passes these (moe_runner.py:532).
        shared_experts=object(),
        shared_experts_input=x,
    )
    assert torch.equal(got, want)
    assert got.dtype == want.dtype

    # The kernel slot itself, called the way the stock forward_native calls it (all keywords).
    direct = method.moe_kernel.apply(
        hidden_states=x,
        w1=layer.w13_weight,
        w2=layer.w2_weight,
        topk_weights=weights,
        topk_ids=ids,
        activation=layer.activation,
        apply_router_weight_on_input=layer.apply_router_weight_on_input,
        global_num_experts=layer.global_num_experts,
        expert_map=layer.expert_map,
        shared_experts=None,
        shared_experts_input=None,
    )
    assert torch.equal(direct, want)

    # Shared experts are the runner's job on OOT (NO_OVERLAP, moe_runner.py:512-514): passing them
    # to the kernel must not change one bit, or the dense expert would be counted twice.
    assert torch.equal(direct, got)


def test_the_monolithic_slot_refuses_instead_of_computing_something_else() -> None:
    """``apply_monolithic`` is unreachable (``is_monolithic`` is False) and says so."""
    with pytest.raises(NotImplementedError, match="is_monolithic=False"):
        fused_moe_oot.TorchMoEKernel().apply_monolithic()


# ---------------------------------------------------------------------------------------
# 3. refusals: quantised / unsupported configs, loudly
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides, needle",
    [
        ({"in_dtype": torch.int8}, "in_dtype"),
        ({"in_dtype": torch.float8_e4m3fn}, "in_dtype"),
        ({"has_bias": True}, "has_bias"),
        ({"is_lora_enabled": True}, "is_lora_enabled"),
        ({"is_act_and_mul": False}, "is_act_and_mul"),
        ({"moe_parallel_config": _ParallelConfig(ep_size=2)}, "ep_size"),
        ({"moe_parallel_config": _ParallelConfig(dp_size=2)}, "dp_size"),
        ({"moe_parallel_config": _ParallelConfig(pcp_size=2)}, "pcp_size"),
        ({"quant_dtype": torch.float8_e4m3fn}, "quantisation markers"),
        ({"block_shape": [128, 128]}, "quantisation markers"),
        ({"per_act_token_quant": True}, "quantisation markers"),
    ],
)
def test_a_quantized_or_unsupported_config_is_refused(overrides: Dict[str, Any], needle: str) -> None:
    """At construction time - i.e. at model load, before a weight is read - and by name."""
    with pytest.raises(NotImplementedError, match=needle):
        _build(**overrides)


def test_the_same_refusals_fire_through_the_public_helper() -> None:
    """``check_unquantized_config`` is public so ``tileinfer doctor``/a probe can ask the question."""
    fused_moe_oot.check_unquantized_config(_Config())  # the good one: no raise
    with pytest.raises(NotImplementedError, match="has_bias"):
        fused_moe_oot.check_unquantized_config(_Config(has_bias=True))


def test_a_quantized_weight_is_refused_at_call_time() -> None:
    """The config is not the only source of truth: the checkpoint's dtype is checked too."""
    method = _build()
    layer = _Layer(dtype=torch.float32)
    layer.w13_weight = layer.w13_weight.to(torch.int8)  # a quantised checkpoint
    gen = torch.Generator().manual_seed(3)
    with pytest.raises(NotImplementedError, match="w1 has dtype torch.int8"):
        method.apply(
            layer=layer,
            x=torch.randn(2, 8, generator=gen),
            topk_weights=torch.rand(2, 1, generator=gen),
            topk_ids=torch.tensor([[0], [1]]),
        )


def test_the_class_only_claims_the_oot_backend() -> None:
    """``register_oot`` is unconditional, so a non-OOT selection must fail instead of mis-computing.

    On any other backend ``process_weights_after_loading`` does *not* return early: it shuffles the
    weights into that backend's kernel format (``unquantized_fused_moe_method.py:157-202``) and our
    port would then read a converted layout.  That is wrong numerics, not a crash - hence the
    refusal.
    """

    class _Triton(_StockBase):
        def __init__(self, moe: Any) -> None:
            super().__init__(moe)
            self.unquantized_backend = "TRITON"

    with pytest.raises(RuntimeError, match="TRITON"):
        fused_moe_oot.make_oot_method(_Triton)(_Config())


def test_the_factory_rejects_a_non_class() -> None:
    with pytest.raises(TypeError, match="make_oot_method"):
        fused_moe_oot.make_oot_method("UnquantizedFusedMoEMethod")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------------------
# 4. the family record: the patch, its metadata, and its mechanism
# ---------------------------------------------------------------------------------------


def test_the_family_records_the_third_kind_with_complete_metadata() -> None:
    record = {binding.op: binding for binding in gdn_moe.bind()}["unquantized_fused_moe"]

    assert record.kind == "custom_op_oot", "the class-level mechanism is the whole point"
    assert record.family == "moe"
    assert record.impl is fused_moe_oot.make_oot_method
    # The target is the class layer.py:359 instantiates; the definition module comes first and the
    # package re-export second, so a vLLM that moves the import still binds.
    assert record.targets[0] == (
        "vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method.UnquantizedFusedMoEMethod"
    )
    assert record.targets[1] == (
        "vllm.model_executor.layers.fused_moe.UnquantizedFusedMoEMethod"
    )
    # Metadata, exactly as a Patch will carry it.
    assert record.why and "register_oot" in record.why and "AttributeError" in record.why
    assert record.remove_when == "upstream builds an OOT moe_kernel"
    assert "oracle/unquantized.py" in record.upstream_ref
    assert "custom_op.py" in record.upstream_ref
    assert record.abi_ok is True and record.abi_note == "" and record.aliases == ()
    # No candidate set: this impl is a class factory, not a per-call function (see
    # gdn_moe._with_candidates). The callable-with-a-shape record is `fused_experts`.
    assert record.candidates is None
    # The family promises it, so install.py checks the table against required_names() and the
    # coverage ledger has to document it.
    assert "unquantized_fused_moe" in gdn_moe.required_names()


def test_the_oot_module_never_imports_vllm() -> None:
    """The base class is an argument on purpose: this module must import on a CPU box."""
    path = ROOT / "tileinfer" / "vllm" / "ops" / "fused_moe_oot.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offenders = []
    for node in ast.walk(tree):
        modules = []
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules = [node.module]
        offenders.extend(
            f"{module} (line {node.lineno})"
            for module in modules
            if module == "vllm"
            or module.startswith("vllm.")
            or module.startswith(("torch_npu", "vllm_ascend"))
        )
    assert not offenders, f"fused_moe_oot.py imports engine code: {offenders}"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
