"""Contract tests for ``tileinfer doctor`` (``tileinfer/vllm/doctor.py``) - no vLLM, no NPU.

Why the fake ``vllm``
---------------------
``doctor`` is a *diagnostic*: its whole value is that it prints a checklist on a box where some of
the imports are broken.  The checks that need an engine are therefore gated on
``plugin._vllm_probe()``, and this file exercises both sides of that gate:

* with **no** ``vllm`` at all (the CPU CI box) the engine-side checks must degrade to ``SKIP`` with
  a fix hint, not raise;
* with a fake ``vllm`` module in ``sys.modules`` (the same pattern as
  ``tests/test_platform_contract.py``) the plugin can be told to install, so ``plugin.status`` and
  ``ops.binding`` are exercised through their real code paths - against modules that only provide
  the names, never the engine.

As in the other contract suites the fake is installed through ``monkeypatch`` (never left behind,
never replacing a fake another module already registered) and the *real* disk tree is untouched: the
``runner_source`` check is read-only, and ``tilelang_cache`` writes one probe file into the
directory the environment names (skipped by the check when the variable is unset).

What is asserted
----------------
* the registry is the documented one (every requirement has a named check, and ``PASS``/``FAIL``/
  ``SKIP`` are the only statuses);
* the printed report contains every check name, and every failure/skip line carries a fix hint;
* a hard failure returns non-zero, a soft failure does not, and the tail line names the step to
  start at;
* the natural invocation is not a usage error: the optional ``doctor`` word (and its ``check`` /
  ``status`` aliases) runs the same checklist as no argument, while an unknown word still exits 2;
* nothing raises - not a check that raises, not one that returns a malformed result, and not
  ``main()`` itself (that is the property the command exists for).
"""

from __future__ import annotations

import json
import sys
import types
from typing import Any, List, Tuple

import pytest

from tileinfer.vllm import doctor

#: The checks this command is specified to print, in order.  Pinned literally (not just derived from
#: ``doctor.CHECKS``) so deleting one is a failing test, not a silently shorter checklist.
EXPECTED_CHECKS: Tuple[str, ...] = (
    "python",
    "torch",
    "torch_npu",
    "cann_env",
    "device",
    "tilelang",
    "tilelang_cache",
    "vllm_version",
    "plugin.status",
    "ops.binding",
    "runner_source",
    "graphs.capture_path_ready",
)


# ---------------------------------------------------------------------------------------
# fixtures: a fake vllm (only when it is safe) and a restored plugin / ops-install state
# ---------------------------------------------------------------------------------------


def _fake_vllm(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Reuse an existing fake, refuse to hide a real vLLM, otherwise install a minimal one."""
    existing = sys.modules.get("vllm")
    if existing is not None and getattr(existing, "__file__", None):
        pytest.skip("a real vLLM is installed; injecting a fake would hide it from the checks")
    if existing is not None:
        return existing  # another contract test's fake: shared, never replaced
    module = types.ModuleType("vllm")
    module.__path__ = []  # type: ignore[attr-defined]
    module.__version__ = "0.23.0"
    monkeypatch.setitem(sys.modules, "vllm", module)
    return module


@pytest.fixture
def fake_vllm(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    return _fake_vllm(monkeypatch)


@pytest.fixture
def clean_state() -> Any:
    """Snapshot ``plugin._STATE`` and reset the ops installer, so a check cannot leak into the suite.

    ``doctor`` deliberately calls ``plugin.install()`` / ``install_ops()`` (a doctor that did not
    would only ever report "the plugin did not run in this process"), and both cache their result
    process-wide.  Restoring the snapshot is what keeps these tests order-independent.
    """
    from tileinfer.vllm import plugin
    from tileinfer.vllm.ops import install as ops_install

    snapshot = dict(plugin._STATE)
    ops_install._reset()
    try:
        yield plugin
    finally:
        ops_install._reset()
        plugin._STATE.clear()
        plugin._STATE.update(snapshot)


def _spec(name: str) -> doctor.CheckSpec:
    return next(spec for spec in doctor._CHECKS if spec.name == name)


def _last_line(out: str) -> str:
    return [line for line in out.splitlines() if line.strip()][-1]


# ---------------------------------------------------------------------------------------
# 1. the registry is the documented checklist
# ---------------------------------------------------------------------------------------


def test_the_check_inventory_is_the_documented_one():
    assert tuple(doctor.CHECKS) == EXPECTED_CHECKS
    assert doctor._CHECKS and all(spec.name and callable(spec.fn) for spec in doctor._CHECKS)
    # Steps 1, 2, 4 and 5 of docs/hardware-checklist.md are mirrored; step 3 is a pytest run and is
    # deliberately not doctor's job (the tail line still names it when everything passes).
    assert {spec.step for spec in doctor._CHECKS} == {1, 2, 4, 5}
    # Only the graph verdict is soft: not-ready means the documented enforce_eager fallback.
    assert doctor.SOFT_CHECKS == ("graphs.capture_path_ready",)
    assert doctor.PASS != doctor.FAIL != doctor.SKIP != doctor.PASS  # three distinct spellings


# ---------------------------------------------------------------------------------------
# 2. the printed report
# ---------------------------------------------------------------------------------------


def test_the_report_contains_every_check_name(fake_vllm, clean_state, capsys):
    code = doctor.main([])
    out = capsys.readouterr().out

    for name in EXPECTED_CHECKS:
        assert name in out, f"doctor's report never mentions the {name!r} check"
    assert doctor.PASS in out, "no PASS line: the report must use the three status words"
    assert "fix:" in out or code == 0, "a failing run must print at least one fix hint"
    assert "docs/hardware-checklist.md" in out
    assert code in (0, 1)


def test_the_tail_line_always_names_a_checklist_step(fake_vllm, clean_state, capsys):
    code = doctor.main([])
    out = capsys.readouterr().out
    tail = _last_line(out)
    assert "docs/hardware-checklist.md step" in tail
    assert code in (0, 1)  # a run either has a hard failure or it does not; nothing else


@pytest.mark.parametrize("spec", doctor._CHECKS, ids=lambda spec: spec.name)
def test_every_check_is_self_contained(spec, clean_state):
    """Run each real check on this box: a status, a detail, and a fix hint whenever it is not PASS."""
    result = doctor._run_check(spec)
    assert result.name == spec.name and result.step == spec.step
    assert result.status in (doctor.PASS, doctor.FAIL, doctor.SKIP)
    assert result.detail, f"{spec.name} produced no detail line"
    if result.status != doctor.PASS:
        assert result.hint, f"{spec.name} fails/skips without a fix hint"


def test_a_broken_check_becomes_a_printed_fail(clean_state):
    """The wrapper is the whole no-traceback guarantee: a raising check is a FAIL, not an exception."""

    def boom():
        raise RuntimeError("simulated broken probe")

    spec = doctor.CheckSpec("broken_probe", 1, True, boom)
    result = doctor._run_check(spec)  # must not raise

    assert result.status == doctor.FAIL
    assert "simulated broken probe" in result.detail
    assert "tileinfer doctor" in result.hint  # points at the tool, not at the environment


@pytest.mark.parametrize(
    "returned",
    [
        ("PASS",),  # too short
        ("PASS", "detail", "hint", "extra"),  # too long
        ("MAYBE", "detail", "hint"),  # unknown status
        "PASS",  # not a tuple at all
    ],
)
def test_a_malformed_check_result_becomes_a_printed_fail(returned, clean_state):
    spec = doctor.CheckSpec("malformed", 1, True, lambda: returned)
    result = doctor._run_check(spec)
    assert result.status == doctor.FAIL
    assert result.hint


# ---------------------------------------------------------------------------------------
# 3. exit codes: hard vs soft
# ---------------------------------------------------------------------------------------


def _all_pass(*names: Tuple[str, int, bool]) -> Tuple[doctor.CheckSpec, ...]:
    return tuple(
        doctor.CheckSpec(
            name, step, hard, (lambda label=name: (doctor.PASS, f"{label} is fine", ""))
        )
        for name, step, hard in names
    )


def test_a_hard_failure_returns_non_zero(monkeypatch, clean_state, capsys):
    def boom():
        raise RuntimeError("simulated device failure")

    monkeypatch.setattr(
        doctor,
        "_CHECKS",
        _all_pass(("python", 1, True)) + (doctor.CheckSpec("device", 1, True, boom),),
    )
    code = doctor.main([])
    out = capsys.readouterr().out

    assert code != 0
    assert f"[{doctor.FAIL}] device" in out
    assert "simulated device failure" in out
    assert "step 1" in _last_line(out), "the tail line must name the step with the hard failure"


def test_a_soft_failure_keeps_the_exit_code_zero(monkeypatch, clean_state, capsys):
    """Capture-path not ready is the documented eager fallback: reported, but not a hard failure."""

    def not_ready():
        return (doctor.FAIL, "no accelerator is available", "expected: eager is the fallback")

    monkeypatch.setattr(
        doctor,
        "_CHECKS",
        _all_pass(("python", 1, True))
        + (doctor.CheckSpec("graphs.capture_path_ready", 5, False, not_ready),),
    )
    code = doctor.main([])
    out = capsys.readouterr().out

    assert code == 0
    assert f"[{doctor.FAIL}] graphs.capture_path_ready" in out
    assert "step 5" in _last_line(out)


def test_the_tail_line_points_at_the_first_failing_step(monkeypatch, clean_state, capsys):
    def fail():
        return (doctor.FAIL, "broken", "fix it")

    monkeypatch.setattr(
        doctor,
        "_CHECKS",
        _all_pass(("python", 1, True))
        + (doctor.CheckSpec("vllm_version", 2, True, fail),)
        + (doctor.CheckSpec("runner_source", 4, True, fail),),
    )
    assert doctor.main([]) == 1
    tail = _last_line(capsys.readouterr().out)
    assert "step 2" in tail and "step 4" not in tail, (
        "the tail must name the *first* broken step, because every later one depends on it"
    )


def test_everything_passing_sends_the_reader_to_step_three(monkeypatch, clean_state, capsys):
    monkeypatch.setattr(
        doctor,
        "_CHECKS",
        _all_pass(("python", 1, True), ("vllm_version", 2, True), ("runner_source", 4, True)),
    )
    assert doctor.main([]) == 0
    tail = _last_line(capsys.readouterr().out)
    assert "step 3" in tail and "pytest" in tail


# ---------------------------------------------------------------------------------------
# 4. never raises, and the two output modes
# ---------------------------------------------------------------------------------------


def test_main_never_raises_even_when_every_check_does(monkeypatch, clean_state, capsys):
    def boom():
        raise RuntimeError("every probe is broken")

    monkeypatch.setattr(
        doctor,
        "_CHECKS",
        tuple(doctor.CheckSpec(name, 1, True, boom) for name in EXPECTED_CHECKS),
    )
    code = doctor.main([])  # must not raise
    out = capsys.readouterr().out

    assert code == 1
    for name in EXPECTED_CHECKS:
        assert name in out
    assert "every probe is broken" in out


def test_json_mode_is_machine_readable(monkeypatch, clean_state, capsys):
    monkeypatch.setattr(doctor, "_CHECKS", _all_pass(("python", 1, True)))
    assert doctor.main(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["exit_code"] == 0
    assert payload["checks"][0]["name"] == "python"
    assert payload["checks"][0]["status"] == doctor.PASS
    assert "next" in payload and "step 3" in payload["next"]


def test_unknown_arguments_are_a_usage_error_not_a_traceback(capsys):
    assert doctor.main(["--not-a-flag"]) == 2
    out = capsys.readouterr().out
    assert "--json" in out  # the usage line names what is supported
    assert doctor.main(["--help"]) == 0
    assert "hardware-checklist" in capsys.readouterr().out


# ---------------------------------------------------------------------------------------
# 5. the two checks whose verdict is driven by another module's answer
# ---------------------------------------------------------------------------------------


def test_vllm_version_check_fails_outside_the_supported_range(monkeypatch, clean_state):
    """The version gate is a hard failure: the platform ABC is transcribed from 0.23 only."""
    from tileinfer.vllm import plugin

    monkeypatch.setattr(
        plugin,
        "_vllm_probe",
        lambda: {"available": True, "version": "0.24.1", "supported": False, "detail": "vLLM 0.24.1"},
    )
    result = doctor._run_check(_spec("vllm_version"))
    assert result.status == doctor.FAIL
    assert "0.24.1" in result.detail
    assert plugin.SUPPORTED_VLLM_RANGE in result.detail
    assert result.hint


def test_vllm_version_unknown_is_a_skip_not_a_failure(monkeypatch, clean_state):
    """A source checkout carries no distribution metadata: that must not fail the bring-up."""
    from tileinfer.vllm import plugin

    monkeypatch.setattr(
        plugin,
        "_vllm_probe",
        lambda: {
            "available": True,
            "version": "",
            "supported": False,
            "detail": "vLLM (already imported; distribution version unknown)",
        },
    )
    result = doctor._run_check(_spec("vllm_version"))
    assert result.status == doctor.SKIP
    assert result.hint


def test_capture_path_not_ready_is_reported_without_failing_the_run(monkeypatch, clean_state):
    """The check reuses ``graphs.capture_path_ready``'s reason instead of inventing its own."""
    from tileinfer.vllm import graphs

    monkeypatch.setattr(graphs, "capture_path_ready", lambda: (False, "torch.npu lacks CUDAGraph"))
    result = doctor._run_check(_spec("graphs.capture_path_ready"))
    assert result.status == doctor.FAIL
    assert result.hard is False
    assert "torch.npu lacks CUDAGraph" in result.detail
    assert "enforce_eager" in result.hint


def test_ops_binding_tolerates_the_one_declined_op(monkeypatch, clean_state, fake_vllm):
    """``chunk_local_cumsum`` is unbound on purpose (``abi_ok=False``); that must not be a failure."""
    from tileinfer.vllm.ops import install as ops_install

    class _Report:
        bound = ["l2norm_fwd"]
        unbound = ["chunk_local_cumsum"]
        disabled: List[str] = []

        def summary(self) -> str:
            return "ops: 1 bound, 1 unbound (no vLLM name / no implementation), 0 off by switch"

    monkeypatch.setattr(ops_install, "install_ops", lambda: _Report())
    result = doctor._run_check(_spec("ops.binding"))
    assert result.status == doctor.PASS, result.detail
    assert "declined on purpose" in result.detail


def test_ops_binding_fails_on_an_unexpected_miss(monkeypatch, clean_state, fake_vllm):
    """An op we *do* port failing to bind is a rename or a missing port - that is hard."""
    from tileinfer.vllm.ops import install as ops_install

    class _Report:
        bound: List[str] = []
        unbound = ["l2norm_fwd"]
        disabled: List[str] = []

        def summary(self) -> str:
            return "ops: 0 bound, 1 unbound (no vLLM name / no implementation), 0 off by switch"

    monkeypatch.setattr(ops_install, "install_ops", lambda: _Report())
    result = doctor._run_check(_spec("ops.binding"))
    assert result.status == doctor.FAIL
    assert "l2norm_fwd" in result.detail
    assert "tests/test_ops_install.py" in result.hint


def test_ops_binding_reports_candidate_gate_status(monkeypatch, clean_state):
    """The candidate seam is visible in the ``ops.binding`` line, *with* its recorded reason.

    Two properties are asserted, because both are easy to lose later:

    * the segment appears whenever a non-baseline candidate is registered - the candidate table is
      TileInfer's own data, so it is printed even when the check itself skips (no vLLM on this box);
    * it is **additive**: the status (and therefore the exit code) stays the binding report's, and a
      missing candidate table just removes the tail.
    """
    from tileinfer.vllm.ops import install as ops_install

    # The shipped family registers one candidate set, with one non-baseline candidate, off by default.
    described = ops_install.candidate_report()
    assert described["fused_experts"]["candidates"], "expected a registered candidate to report"

    seen = doctor._run_check(_spec("ops.binding"))
    assert "candidates:" in seen.detail
    # The disabled candidate's *status text* is what a reader needs, not just its name.
    assert "fused_experts[moe_reference_oracle=disabled: off by default" in seen.detail

    # Same verdict without the segment: this is a diagnostic line, not a check of its own.
    monkeypatch.setattr(ops_install, "candidate_report", lambda: {})
    without = doctor._run_check(_spec("ops.binding"))
    assert without.status == seen.status
    assert "candidates:" not in without.detail


# ---------------------------------------------------------------------------------------
# 6. the invocation: an optional subcommand word, and what is still a usage error
# ---------------------------------------------------------------------------------------


def test_the_natural_subcommand_runs_the_checklist(fake_vllm, clean_state, capsys):
    """``tileinfer doctor`` is what people type: it must be the checklist, not a usage error."""
    code = doctor.main(["doctor"])
    out = capsys.readouterr().out

    assert code in (0, 1), "the subcommand must return a checklist verdict, never 2"
    assert "unknown argument" not in out
    assert "docs/hardware-checklist.md" in out


@pytest.mark.parametrize("alias", ["check", "status"])
def test_the_subcommand_aliases_run_the_checklist(alias, fake_vllm, clean_state, capsys):
    assert doctor.main([alias]) in (0, 1)
    out = capsys.readouterr().out
    assert "unknown argument" not in out
    assert "docs/hardware-checklist.md" in out


def test_no_arguments_behaves_exactly_as_before(fake_vllm, clean_state, capsys):
    """``main([])`` is unchanged: the subcommand adds a spelling, not a second report."""
    bare = doctor.main([])
    bare_report = capsys.readouterr().out

    assert bare in (0, 1)
    assert doctor.main(["doctor"]) == bare
    assert capsys.readouterr().out == bare_report


def test_the_subcommand_combines_with_the_flags(monkeypatch, clean_state, capsys):
    monkeypatch.setattr(doctor, "_CHECKS", _all_pass(("python", 1, True)))

    assert doctor.main(["doctor", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["checks"][0]["name"] == "python"

    assert doctor.main(["--json", "status"]) == 0
    assert "checks" in json.loads(capsys.readouterr().out)


def test_an_unknown_word_is_still_a_usage_error(capsys):
    """A subcommand is accepted only when it is one of the three documented spellings."""
    assert doctor.main(["doctorx"]) == 2
    out = capsys.readouterr().out
    assert "unknown argument" in out
    assert "doctor" in out, "the usage line must name the accepted spellings"

    assert doctor.main(["doctor", "--not-a-flag"]) == 2
    assert "unknown argument" in capsys.readouterr().out


def test_a_repeated_subcommand_is_a_usage_error(capsys):
    assert doctor.main(["doctor", "check"]) == 2
    assert "twice" in capsys.readouterr().out
