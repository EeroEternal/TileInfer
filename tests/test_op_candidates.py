"""Contract tests for the per-operator candidate seam (``tileinfer.vllm.ops.candidates``).

What this file is defending
---------------------------
The seam exists so performance work can add a second implementation per vLLM ABI name (a TileLang
kernel later) without changing the ABI, without changing what Qwen3.6 does today, and without any
way for an unmeasured kernel to become the default.  So the tests here are about the *gate* and the
*cheapness*, not about a kernel:

1. a set with only its baseline returns the baseline implementation itself - the pre-candidate
   behaviour, not a new function that happens to call it;
2. a candidate is selected by a **shape** predicate (reading
   ``.shape``/``.dtype``/``is_contiguous()`` and nothing else), and ``select`` never touches a
   tensor value - pinned with a tensor stand-in whose every value access raises;
3. priority order is registration order, and the first *eligible* candidate wins;
4. a recorded ``measured_ratio`` below the 1.0x contract line is not eligible even when its flag
   says on - the repo's own cautionary example (the MoE grouped-GEMM kernel at 0.86x,
   ``docs/status.md`` E19/H7) must not become the default by accident;
5. ``TILEINFER_OPS_CANDIDATES=off`` disarms every non-baseline candidate, and a typo fails safe;
6. the wrapper ``install.py`` binds keeps positional **and** keyword calling (``CustomOp`` forwards
   by keyword, module attributes by position), and introspection still names the baseline impl;
7. the family still promises exactly the same ``required_names()``, one candidate set per name, and
   the one registered non-baseline candidate is equivalent to its baseline, off by default, and
   never selected for a production-shaped call.

The test-local second implementation
------------------------------------
``_alt_l2norm`` is the "trivially-equivalent alternative implementation in the test": the seam's
second implementation has to come from somewhere, and ``tileinfer/kernels/`` (the MoE grouped-GEMM
kernels, the GDN chunk-wu kernel) imports ``tilelang`` at module level, so it cannot be imported on
this CPU box - see the note on ``_MOE_REFERENCE_NOTE`` in ``gdn_moe.py`` for the recorded 0.86x
result that keeps the real kernel candidate disabled.  ``_alt_l2norm`` is written from
``ops/gdn.py``'s documented formula, so the equivalence assertions are about maths, not identity.
"""

from __future__ import annotations

import json
from typing import Any, List

import pytest
import torch

from tileinfer.vllm.ops import candidates as candidates_mod
from tileinfer.vllm.ops import install as install_mod
from tileinfer.vllm.ops.candidates import (
    DEFAULT_THRESHOLD,
    ENV_CANDIDATES,
    CallSpec,
    Candidate,
    CandidateSet,
    always,
)
from tileinfer.vllm.ops.families import gdn_moe
from tileinfer.vllm.ops.gdn import l2norm_fwd

#: The repo's recorded negative result, reused verbatim: the MoE grouped-GEMM kernel measured 0.86x
#: of the aclnn baseline on device after three optimisation rounds (docs/status.md, E19/H7), i.e.
#: below the 1.0x contract line.  It must never become the default, so it is the gate's test case.
RECORDED_RATIO = 0.86


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test inherits a switch from the shell (the candidate kill switch or the family switch)."""
    monkeypatch.delenv(ENV_CANDIDATES, raising=False)
    monkeypatch.delenv("TILEINFER_OPS", raising=False)


# ---------------------------------------------------------------------------------------
# A second implementation of a real op, defined here on purpose (see the module docstring)
# ---------------------------------------------------------------------------------------


def _alt_l2norm(
    x: torch.Tensor, eps: float = 1e-6, output_dtype: Any = None
) -> torch.Tensor:
    """``x / sqrt(sum(x^2, -1) + eps)`` from ``ops/gdn.py``'s documented formula.

    Deliberately *not* a call into :func:`tileinfer.vllm.ops.gdn.l2norm_fwd`: it has to be a second
    implementation for the seam to prove anything, and it has to be equivalent for the selection
    tests to be able to compare outputs.
    """
    work = x.to(torch.float32)
    denom = torch.sqrt(torch.sum(work * work, dim=-1, keepdim=True) + float(eps))
    return (work / denom).to(output_dtype or x.dtype)


def _rectangular(spec: CallSpec) -> bool:
    """A shape/dtype predicate of the kind a kernel would carry: 2-D, last dim 8, contiguous."""
    x = spec.arg(0, "x")
    shape = getattr(x, "shape", ())
    if len(shape) != 2 or shape[1] != 8:
        return False
    return str(getattr(x, "dtype", None)) == "torch.float32" and bool(x.is_contiguous())


def _selected_set() -> CandidateSet:
    """The baseline plus :func:`_alt_l2norm`, selected for the rectangular shape only."""
    chosen = CandidateSet("l2norm_fwd", Candidate("torch", l2norm_fwd, always, True))
    chosen.add(
        Candidate(
            name="rectangular",
            impl=_alt_l2norm,
            predicate=_rectangular,
            enabled_by_default=True,
            note="test-local equivalent implementation; shape-gated",
        )
    )
    return chosen


def _recorded_set() -> CandidateSet:
    """The baseline plus the recorded 0.86x candidate, flagged **on** so the gate blocks it."""
    chosen = CandidateSet("fused_experts", Candidate("torch", l2norm_fwd, always, True))
    chosen.add(
        Candidate(
            name="moe_grouped_gemm",
            impl=_alt_l2norm,
            predicate=always,
            enabled_by_default=True,
            note="0.86x on device, not enabled",
            measured_ratio=RECORDED_RATIO,
        )
    )
    return chosen


def _spec(x: Any, op: str = "l2norm_fwd") -> CallSpec:
    return CallSpec(args=(x,), kwargs={}, op=op)


def _marker(*_args: Any, **_kwargs: Any) -> str:
    """An implementation that only says who it is, for the priority-order tests."""
    return "marker"


def _other_marker(*_args: Any, **_kwargs: Any) -> str:
    return "other"


# ---------------------------------------------------------------------------------------
# 1. baseline only: today's behaviour, by construction
# ---------------------------------------------------------------------------------------


def test_a_baseline_only_set_is_the_pre_candidate_behaviour() -> None:
    chosen = CandidateSet("l2norm_fwd", Candidate("torch", l2norm_fwd, always, True))
    x = torch.randn(4, 8)

    assert len(chosen) == 1
    assert list(chosen) == [chosen.baseline]
    assert chosen.baseline.impl is l2norm_fwd
    assert chosen.baseline.enabled_by_default is True
    # `select` hands back the implementation object itself - not a wrapper, not a copy.
    assert chosen.select(_spec(x)) is l2norm_fwd
    assert chosen.select_candidate(_spec(x)) is chosen.baseline
    # ... and the dispatcher is bit-identical to calling the implementation directly.
    assert torch.equal(chosen.dispatched()(x), l2norm_fwd(x))
    assert torch.equal(chosen.dispatched()(x, 1e-5, torch.float64),
                       l2norm_fwd(x, 1e-5, torch.float64))


def test_an_unmeasured_candidate_is_opt_in_by_its_flag() -> None:
    candidate = Candidate("alt", _alt_l2norm, always, True, note="no measurement yet")
    assert candidate.measured_ratio is None
    assert candidate.eligible() is True
    off = Candidate("alt", _alt_l2norm, always, False, note="no measurement yet")
    assert off.eligible() is False


# ---------------------------------------------------------------------------------------
# 2. selection by a shape predicate, in both calling styles
# ---------------------------------------------------------------------------------------


def test_a_shape_predicate_selects_the_candidate() -> None:
    chosen = _selected_set()
    rectangular = torch.randn(4, 8)
    assert chosen.select(_spec(rectangular)) is _alt_l2norm
    # The shape is the discriminator, not the name of the caller: a (4, 7) call is the baseline.
    assert chosen.select(_spec(torch.randn(4, 7))) is l2norm_fwd
    assert chosen.select(_spec(torch.randn(4, 8, dtype=torch.bfloat16))) is l2norm_fwd

    # The CustomOp takeover forwards by keyword (`forward_native(q=..., k=...)`), so the same
    # predicate has to see a keyword call too - that is what `CallSpec.arg(index, name)` is for.
    by_keyword = CallSpec(args=(), kwargs={"x": torch.randn(4, 8)}, op="l2norm_fwd")
    assert chosen.select(by_keyword) is _alt_l2norm


def test_the_two_implementations_are_equivalent_where_the_candidate_is_selected() -> None:
    """A candidate selectable for a shape must compute the same function there."""
    chosen = _selected_set()
    x = torch.randn(4, 8)
    got = chosen.dispatched()(x)
    assert torch.allclose(got, l2norm_fwd(x), atol=1e-6, rtol=1e-5)
    assert got.dtype == x.dtype


# ---------------------------------------------------------------------------------------
# 3. priority order
# ---------------------------------------------------------------------------------------


def test_priority_order_is_registration_order() -> None:
    chosen = CandidateSet("l2norm_fwd", Candidate("torch", l2norm_fwd, always, True))
    first = chosen.add(Candidate("first", _marker, always, True))
    second = chosen.add(Candidate("second", _other_marker, always, True))

    x = torch.randn(2, 4)
    assert chosen.select(_spec(x)) is _marker  # the first registered candidate wins
    assert [candidate.name for candidate in chosen] == ["first", "second", "torch"]
    # The baseline is last in iteration order, and is not one of the candidates.
    assert chosen.baseline is not first and chosen.baseline is not second
    assert chosen.select(_spec(x)) is not _other_marker


def test_a_disabled_high_priority_candidate_does_not_block_the_next() -> None:
    chosen = CandidateSet("l2norm_fwd", Candidate("torch", l2norm_fwd, always, True))
    chosen.add(
        Candidate("slow", _marker, always, True, measured_ratio=0.5, note="below the line")
    )
    chosen.add(Candidate("fast", _other_marker, always, True))

    assert chosen.select(_spec(torch.randn(2, 4))) is _other_marker


# ---------------------------------------------------------------------------------------
# 4. the measured-ratio gate: the 0.86x case
# ---------------------------------------------------------------------------------------


def test_a_recorded_ratio_below_the_threshold_stays_disabled() -> None:
    assert DEFAULT_THRESHOLD == 1.0  # the contract line: parity with the baseline
    chosen = _recorded_set()
    candidate = chosen.candidates[0]
    spec = _spec(torch.randn(4, 8), op="fused_experts")

    # The flag says ON - this is exactly the "someone flips it" case the gate has to survive.
    assert candidate.enabled_by_default is True
    assert candidate.measured_ratio == RECORDED_RATIO
    assert candidate.eligible() is False
    assert chosen.select(spec) is l2norm_fwd  # the baseline, i.e. the aclnn-equivalent path
    assert chosen.select_candidate(spec) is chosen.baseline

    # Parity (and anything above it) is eligible: the gate is the number, not the mood.
    at_parity = CandidateSet("fused_experts", chosen.baseline, threshold=1.0)
    at_parity.add(
        Candidate("grouped", _alt_l2norm, always, True, note="1.00x", measured_ratio=1.0)
    )
    assert at_parity.candidates[0].eligible() is True
    assert at_parity.select(spec) is _alt_l2norm

    # The recorded number is reported verbatim - the reason has to survive to the doctor.
    row = chosen.describe()["candidates"][0]
    assert row["measured_ratio"] == RECORDED_RATIO
    assert row["status"].startswith("disabled")
    assert "0.86" in row["status"] and "1.00" in row["status"]


def test_the_threshold_is_per_set_so_a_shape_can_have_its_own_contract() -> None:
    baseline = Candidate("torch", l2norm_fwd, always, True)
    strict = CandidateSet("l2norm_fwd", baseline, threshold=1.10)
    strict.add(Candidate("grouped", _alt_l2norm, always, True, measured_ratio=1.05))
    assert strict.select(_spec(torch.randn(2, 4))) is l2norm_fwd  # 1.05x does not clear 1.10x


# ---------------------------------------------------------------------------------------
# 5. the global kill switch
# ---------------------------------------------------------------------------------------


def test_the_kill_switch_disarms_every_non_baseline_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chosen = _selected_set()
    x = torch.randn(4, 8)

    assert chosen.select(_spec(x)) is _alt_l2norm

    monkeypatch.setenv(ENV_CANDIDATES, "off")
    assert candidates_mod.candidates_enabled() is False
    assert chosen.select(_spec(x)) is l2norm_fwd
    assert chosen.select_candidate(_spec(x)) is chosen.baseline
    # The baseline path is untouched by the switch: still the implementation's own result.
    assert torch.equal(chosen.dispatched()(x), l2norm_fwd(x))

    # Read per call, mirroring TILEINFER_OPS: flipping back needs no reinstall.
    monkeypatch.setenv(ENV_CANDIDATES, "ON")
    assert chosen.select(_spec(x)) is _alt_l2norm
    monkeypatch.delenv(ENV_CANDIDATES, raising=False)
    assert chosen.select(_spec(x)) is _alt_l2norm

    # ... and through the wrapper install.py binds (the real call path), per call, with no rebuild.
    call = chosen.dispatched()
    assert torch.allclose(call(x), _alt_l2norm(x))
    monkeypatch.setenv(ENV_CANDIDATES, "off")
    assert torch.equal(call(x), l2norm_fwd(x))
    monkeypatch.delenv(ENV_CANDIDATES, raising=False)
    assert torch.allclose(call(x), _alt_l2norm(x))


def test_an_unrecognised_kill_switch_value_fails_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A typo must never *enable* an unmeasured kernel: it reads as off, and says so once."""
    monkeypatch.setenv(ENV_CANDIDATES, "maybe")
    assert candidates_mod.candidates_enabled() is False
    monkeypatch.setenv(ENV_CANDIDATES, "")  # blank = "no override", not "off"
    assert candidates_mod.candidates_enabled() is True


def test_the_kill_switch_never_changes_the_baseline_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chosen = CandidateSet("l2norm_fwd", Candidate("torch", l2norm_fwd, always, True))
    x = torch.randn(3, 5)
    monkeypatch.setenv(ENV_CANDIDATES, "off")
    assert chosen.select(_spec(x)) is l2norm_fwd
    assert torch.equal(chosen.dispatched()(x), l2norm_fwd(x))


# ---------------------------------------------------------------------------------------
# 6. `select` never touches tensor values
# ---------------------------------------------------------------------------------------


class _ValueOnlyTensor:
    """A tensor stand-in that answers the three inspected attributes and raises for anything else.

    A predicate may read ``.shape`` / ``.dtype`` / ``.is_contiguous()`` - all free.  Reading a
    *value* (``.item()``, ``.max()``, indexing, iteration, arithmetic) would be a host sync inside a
    forward, which ``select`` is not allowed to do per call, so every such access fails the test.
    """

    shape = (4, 8)
    dtype = "torch.float32"  # a *string*: predicates compare str(dtype), so no torch import either
    ndim = 2

    def is_contiguous(self) -> bool:
        return True

    def _forbidden(self, *_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError(
            "select() must not read a tensor value: only .shape/.dtype/.is_contiguous() are free"
        )

    item = max = min = sum = tolist = numel = _forbidden
    __getitem__ = __iter__ = __float__ = __bool__ = __len__ = _forbidden


def test_select_never_touches_tensor_values() -> None:
    seen: List[Any] = []

    def predicate(spec: CallSpec) -> bool:
        x = spec.arg(0, "x")
        seen.append((tuple(x.shape), x.dtype, x.is_contiguous()))
        return x.shape[0] == 4 and x.is_contiguous()

    def exploding_baseline(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("select() must not call an implementation")

    def exploding_candidate(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("select() must not call an implementation")

    chosen = CandidateSet("l2norm_fwd", Candidate("torch", exploding_baseline, always, True))
    chosen.add(Candidate("shape_gated", exploding_candidate, predicate, True))

    # Both implementations explode if called: only `select` runs.
    assert chosen.select(_spec(_ValueOnlyTensor())) is exploding_candidate
    assert seen == [((4, 8), "torch.float32", True)]
    assert chosen.select(_spec(_ValueOnlyTensor())) is exploding_candidate  # still no raise
    assert chosen.baseline.impl is exploding_baseline


def test_a_predicate_that_raises_is_logged_once_and_falls_back(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def broken(_spec: CallSpec) -> bool:
        raise RuntimeError("predicate bug")

    chosen = CandidateSet("l2norm_fwd", Candidate("torch", l2norm_fwd, always, True))
    chosen.add(Candidate("broken", always, broken, True))
    spec = _spec(torch.randn(2, 4))

    with caplog.at_level("WARNING", logger="tileinfer.vllm.ops.candidates"):
        assert chosen.select(spec) is l2norm_fwd
        assert chosen.select(spec) is l2norm_fwd
    assert caplog.text.count("predicate raised") == 1  # once per candidate, not once per call


# ---------------------------------------------------------------------------------------
# 7. the wrapper keeps the ABI (positional *and* keyword), and reports the baseline
# ---------------------------------------------------------------------------------------


def test_the_dispatcher_wrapper_preserves_positional_and_keyword_arguments() -> None:
    calls: List[Any] = []

    def impl(x: Any, eps: float = 1e-6, output_dtype: Any = None) -> str:
        calls.append((x, eps, output_dtype))
        return "ok"

    chosen = CandidateSet("l2norm_fwd", Candidate("torch", impl, always, True))
    call = chosen.dispatched()

    assert call(1, 2, 3) == "ok"
    assert calls[-1] == (1, 2, 3)  # positional arguments arrive in place
    assert call(1, eps=2) == "ok"
    assert calls[-1] == (1, 2, None)  # a trailing keyword fills the default
    assert call(x=1, eps=2) == "ok"
    assert calls[-1] == (1, 2, None)  # all-keyword, as the CustomOp path calls it

    # The markers install.py reports: the baseline (never the dispatcher) and the set itself.
    assert call.__tileinfer_baseline__ is impl
    assert call.__tileinfer_candidates__ is chosen
    assert call.__name__.endswith("__candidates")


def test_the_wrapper_dispatches_by_shape_in_either_calling_style() -> None:
    call = _selected_set().dispatched()
    x = torch.randn(4, 8)
    assert torch.allclose(call(x), _alt_l2norm(x))
    assert torch.allclose(call(x=x), _alt_l2norm(x))
    narrow = torch.randn(4, 7)
    assert torch.equal(call(narrow), l2norm_fwd(narrow))


# ---------------------------------------------------------------------------------------
# 8. describe(): the report the doctor/probe prints, including the reason
# ---------------------------------------------------------------------------------------


def test_describe_reports_every_candidate_with_its_gate_and_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chosen = _recorded_set()
    described = chosen.describe()

    assert described["op"] == "fused_experts"
    assert described["threshold"] == DEFAULT_THRESHOLD
    assert described["baseline"]["name"] == "torch"
    assert described["baseline"]["status"] == "baseline"
    rows = {row["name"]: row for row in described["candidates"]}
    assert set(rows) == {"moe_grouped_gemm"}
    assert rows["moe_grouped_gemm"]["note"] == "0.86x on device, not enabled"
    assert rows["moe_grouped_gemm"]["status"].startswith("disabled")
    assert "measured 0.86x" in rows["moe_grouped_gemm"]["status"]

    monkeypatch.setenv(ENV_CANDIDATES, "off")
    described = chosen.describe()
    assert described["switch"]["enabled"] is False
    assert "kill switch" in described["candidates"][0]["status"]
    # The report is plain data: it is printed as JSON and stored in a log, never introspected.
    assert json.loads(json.dumps(described))["op"] == "fused_experts"


def test_a_candidate_name_identifies_it_uniquely() -> None:
    chosen = CandidateSet("l2norm_fwd", Candidate("torch", l2norm_fwd, always, True))
    chosen.add(Candidate("alt", _alt_l2norm, always, True))
    with pytest.raises(ValueError, match="duplicate candidate name"):
        chosen.add(Candidate("alt", _alt_l2norm, always, True))
    with pytest.raises(ValueError, match="baseline"):
        chosen.add(Candidate("torch", _alt_l2norm, always, True))
    with pytest.raises(TypeError):
        chosen.add("not a candidate")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------------------
# 9. the family: same names, one set per name, torch impl still the baseline
# ---------------------------------------------------------------------------------------


def test_the_family_still_promises_exactly_the_same_names() -> None:
    assert gdn_moe.required_names() == (
        "chunk_gated_delta_rule",
        "l2norm_fwd",
        "causal_conv1d_fn",
        "causal_conv1d_update",
        "fused_sigmoid_gating_delta_rule_update",
        "fused_recurrent_gated_delta_rule_packed_decode",
        "chunk_local_cumsum",
        "fused_post_conv_prep",
        "topk_softmax",
        "moe_align_block_size",
        "fused_experts",
        # The OOT MoE method (kind="custom_op_oot"): a class registration, not an operator call.
        # Its compute body is the `fused_experts` entry above, which is where the candidate set is.
        "unquantized_fused_moe",
        "silu_and_mul",
    )
    records = gdn_moe.bind()
    assert [binding.op for binding in records] == list(gdn_moe.required_names())


def test_every_binding_keeps_its_torch_impl_and_gains_one_candidate_set() -> None:
    for binding in gdn_moe.bind():
        assert binding.impl is not None
        if binding.kind == "custom_op_oot":
            # A class factory (impl(base_cls) -> subclass) has no per-call shape for a predicate to
            # dispatch on: the candidate seam lives on the entry whose impl is a real call.
            assert binding.candidates is None, f"{binding.op}: the OOT kind takes no candidate set"
            continue
        assert binding.candidates is not None, f"{binding.op}: no candidate set"
        assert binding.candidates.op == binding.op
        assert binding.candidates.baseline.name == gdn_moe.BASELINE_NAME
        assert binding.candidates.baseline.impl is binding.impl
        assert binding.candidates.baseline.enabled_by_default is True
        assert binding.candidates.baseline.eligible() is True


def test_the_family_dispatch_is_byte_identical_to_the_baseline() -> None:
    """The one thing that must not change for Qwen3.6: what the bound function computes."""
    record = {binding.op: binding for binding in gdn_moe.bind()}["fused_experts"]
    gen = torch.Generator().manual_seed(17)
    x = torch.randn(3, 8, generator=gen)
    w1 = torch.randn(4, 6, 8, generator=gen)
    w2 = torch.randn(4, 8, 3, generator=gen)
    weights = torch.rand(3, 2, generator=gen)
    ids = torch.tensor([[0, 1], [2, 3], [1, 0]])

    call = record.candidates.dispatched()
    assert torch.equal(call(x, w1, w2, weights, ids), record.impl(x, w1, w2, weights, ids))
    assert torch.equal(
        call(hidden_states=x, w1=w1, w2=w2, topk_weights=weights, topk_ids=ids),
        record.impl(x, w1, w2, weights, ids),
    )


# ---------------------------------------------------------------------------------------
# 10. the one registered second candidate: real, equivalent, off, and never selected
# ---------------------------------------------------------------------------------------


def test_the_second_candidate_is_off_with_its_recorded_reason() -> None:
    records = {binding.op: binding for binding in gdn_moe.bind()}
    extra = [
        candidate
        for binding in records.values()
        if binding.candidates is not None
        for candidate in binding.candidates.candidates
    ]
    assert [candidate.name for candidate in extra] == ["moe_reference_oracle"]
    (oracle,) = extra

    assert oracle.enabled_by_default is False
    assert oracle.eligible() is False
    assert oracle.measured_ratio is None  # the oracle is not a speed-up: nothing to record
    # The recorded reason is the repo's own result, not a placeholder.
    assert "0.86x" in oracle.note and "1.0x" in oracle.note and "E19/H7" in oracle.note
    assert "tileinfer/testing/moe_reference" in oracle.note


def test_the_second_candidate_is_equivalent_to_the_baseline() -> None:
    records = {binding.op: binding for binding in gdn_moe.bind()}
    record = records["fused_experts"]
    (oracle,) = record.candidates.candidates

    gen = torch.Generator().manual_seed(23)
    x = torch.randn(3, 8, generator=gen)
    w1 = torch.randn(4, 6, 8, generator=gen)
    w2 = torch.randn(4, 8, 3, generator=gen)
    weights = torch.rand(3, 2, generator=gen)
    ids = torch.tensor([[0, 1], [2, 3], [1, 0]])

    want = record.impl(x, w1, w2, weights, ids)
    got = oracle.impl(x, w1, w2, weights, ids)  # the baseline's positional order
    assert got.shape == want.shape and got.dtype == want.dtype
    assert torch.allclose(got, want, atol=1e-5, rtol=1e-4)

    # It refuses what the oracle cannot express instead of computing a different function.
    with pytest.raises(NotImplementedError, match="activation"):
        oracle.impl(x, w1, w2, weights, ids, activation="gelu")
    with pytest.raises(NotImplementedError, match="expert_map"):
        oracle.impl(x, w1, w2, weights, ids, expert_map=torch.arange(4, dtype=torch.int32))


def test_the_second_candidate_is_never_selected_for_a_production_call() -> None:
    record = {binding.op: binding for binding in gdn_moe.bind()}["fused_experts"]
    (oracle,) = record.candidates.candidates
    gen = torch.Generator().manual_seed(29)
    w1 = torch.randn(4, 6, 8, generator=gen)
    w2 = torch.randn(4, 8, 3, generator=gen)
    weights = torch.rand(2, 2, generator=gen)
    ids = torch.tensor([[0, 1], [2, 3]])

    # The predicate's own shape (fp32, <= 2 tokens) would match this call ...
    tiny = torch.randn(1, 8, generator=gen)
    spec = CallSpec(args=(tiny, w1, w2, weights[:1], ids[:1]), op="fused_experts")
    assert oracle.predicate(spec) is True
    # ... and the flag still keeps it out, so the baseline runs.
    assert record.candidates.select(spec) is record.impl

    # A serving-shaped call (bf16, many tokens) does not even match the predicate.
    serving = CallSpec(
        args=(torch.zeros(512, 32, dtype=torch.bfloat16), w1, w2, weights, ids),
        op="fused_experts",
    )
    assert oracle.predicate(serving) is False
    assert record.candidates.select(serving) is record.impl


# ---------------------------------------------------------------------------------------
# 11. install.py: the dispatcher is what gets bound, the baseline is what is reported
# ---------------------------------------------------------------------------------------


def test_install_binds_the_dispatcher_and_reports_the_baseline_impl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = {binding.op: binding for binding in gdn_moe.bind()}["l2norm_fwd"]

    def vllm_original(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("vLLM's own kernel must never run while the gdn family is on")

    takeover = install_mod._wrap(
        record.candidates.dispatched(), vllm_original, "gdn", "fake.l2norm_fwd"
    )
    # Identity, not a wrapper: the doctor line and every test that pins the impl keep working.
    assert takeover.__tileinfer_takeover__ is True
    assert takeover.__tileinfer_impl__ is record.impl
    assert takeover.__tileinfer_candidates__ is record.candidates
    assert takeover.__tileinfer_original__ is vllm_original

    x = torch.randn(3, 5)
    assert torch.equal(takeover(x), l2norm_fwd(x))  # positional
    assert torch.equal(takeover(x=x, eps=1e-5), l2norm_fwd(x, 1e-5))  # keyword

    # The family switch still owns the call: switched off, vLLM's object runs instead.
    monkeypatch.setenv("TILEINFER_OPS", "moe")
    with pytest.raises(AssertionError, match="must never run"):
        takeover(x)


def test_the_install_exposes_the_candidate_report_for_the_doctor() -> None:
    report = install_mod.candidate_report()
    # One entry per binding that *has* a set: the OOT method record is a class factory, so it takes
    # none (see gdn_moe._with_candidates) and cannot appear here.
    assert set(report) == {
        binding.op for binding in install_mod.bindings() if binding.candidates is not None
    }
    assert "unquantized_fused_moe" not in report
    assert report["l2norm_fwd"]["candidates"] == []
    assert [row["name"] for row in report["fused_experts"]["candidates"]] == [
        "moe_reference_oracle"
    ]
    assert json.loads(json.dumps(report))["fused_experts"]["op"] == "fused_experts"
