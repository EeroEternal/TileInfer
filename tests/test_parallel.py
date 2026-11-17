"""Phase 0 of multi-card support: parallel is a declaration, and selection fails closed.

The tests run on CPU with no engine.  They pin the three properties that make sharding safe
to add later without changing the architecture:

* the environment records tp/ep/dp/pp and turns them into a closed axis vocabulary;
* a provider declares which axes it has been validated for;
* the decision refuses a single-only provider under any active sharding axis instead of
  running it on locally sharded tensors (wrong numbers, no exception).
"""

from __future__ import annotations

import pytest

from tileinfer.contracts.capabilities import (
    ProviderSpec,
    Capability,
    normalize_parallel,
    parallel_supports,
)
from tileinfer.contracts.plan import Plan
from tileinfer.manifests.lfm3_1_7 import LFM3_1_7
from tileinfer.manifests.nemotron_3_5 import NEMOTRON_3_5_DRAFT
from tileinfer.manifests.qwen3_6 import QWEN3_6
from tileinfer.resolve.decide import decide
from tileinfer.resolve.probe import Environment, probe


def _vendor_env(**parallel: int) -> Environment:
    return Environment(
        vllm="0.23.0",
        vllm_ascend="0.23.0",
        torch="2.10.0+cpu",
        python="3.12.14",
        device_type="A5",
        has_vendor=True,
        **parallel,
    )


def _provider(**kwargs) -> ProviderSpec:
    base = dict(
        capability=Capability.MOE_ROUTED,
        name="x",
        impl="x.y",
        source="tileinfer",
        graphable=True,
        evidence="bench/x",
    )
    base.update(kwargs)
    return ProviderSpec(**base)


def test_environment_axes_default_to_single() -> None:
    assert Environment().parallel_axes() == ("single",)


def test_environment_axes_record_every_active_dimension() -> None:
    env = _vendor_env(tp_size=8, ep_size=2)
    assert env.parallel_axes() == ("tp", "ep")


def test_environment_rejects_non_positive_sizes() -> None:
    with pytest.raises(ValueError, match="tp_size"):
        Environment(tp_size=0)


def test_parallel_vocabulary_is_closed_and_deduplicated() -> None:
    assert normalize_parallel(("single",)) == ("single",)
    assert normalize_parallel(("single", "tp")) == ("single", "tp")
    assert normalize_parallel(("tp", "ep", "tp")) == ("tp", "ep")
    with pytest.raises(ValueError, match="unknown parallel modes"):
        normalize_parallel(("tensor",))
    with pytest.raises(ValueError, match="at least one mode"):
        normalize_parallel(())


def test_parallel_supports_requires_every_live_axis() -> None:
    assert parallel_supports(("single",), ())
    assert parallel_supports(("single", "tp"), ("tp",))
    assert not parallel_supports(("single",), ("tp",))
    assert not parallel_supports(("tp",), ("tp", "ep"))


def test_provider_validates_its_parallel_declaration() -> None:
    _provider(parallel=("single", "tp")).validate()
    with pytest.raises(ValueError, match="unknown parallel modes"):
        _provider(parallel=("bogus",)).validate()


def test_vendor_selection_survives_tp_and_the_plan_records_it() -> None:
    plan, report = decide(QWEN3_6, _vendor_env(tp_size=2))
    assert report.servable
    assert plan.parallel == ("tp",)
    assert all(selection.source == "vendor" for selection in plan.selections)


def test_single_only_provider_is_refused_under_tp() -> None:
    # LFM3's short-conv state provider is declared single-card only (honest today), so a
    # TP=2 plan must report the capability as a gap instead of installing it.
    plan, report = decide(LFM3_1_7, _vendor_env(tp_size=2))
    assert report.servable is False
    assert [gap.capability for gap in report.gaps] == ["state.linear"]
    assert plan.selection("state.linear") is None


def test_candidates_are_refused_under_tp_and_the_reason_says_why() -> None:
    plan, report = decide(
        NEMOTRON_3_5_DRAFT, _vendor_env(ep_size=2), allow_candidates=True
    )
    # The Mamba2 candidate cannot be selected under a sharding axis it never declared.
    assert plan.selection("state.linear") is None
    gap = next(g for g in report.gaps if g.capability == "state.linear")
    assert "no support for parallel" in gap.reason
    assert "tileinfer.mamba2_state" in gap.reason


def test_plan_round_trips_its_parallel_axes() -> None:
    plan, _ = decide(QWEN3_6, _vendor_env(tp_size=2, pp_size=2))
    data = plan.as_dict()
    assert data["parallel"] == ["tp", "pp"]
    restored = Plan.from_dict(data)
    assert restored.parallel == ("tp", "pp")
    assert restored.plan_id == plan.plan_id


def test_old_plan_without_parallel_loads_as_single() -> None:
    plan, _ = decide(QWEN3_6, _vendor_env())
    data = plan.as_dict()
    data.pop("parallel")
    restored = Plan.from_dict(data)
    assert restored.parallel == ("single",)


def test_plan_rejects_an_invalid_parallel_declaration() -> None:
    plan, _ = decide(QWEN3_6, _vendor_env())
    bad = plan.as_dict()
    bad["parallel"] = ["tensor"]
    with pytest.raises(ValueError, match="unknown parallel modes"):
        Plan.from_dict(bad)


def test_probe_reads_parallel_sizes_from_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("TILEINFER_TP_SIZE", "4")
    monkeypatch.setenv("TILEINFER_EP_SIZE", "2")
    monkeypatch.delenv("TILEINFER_DP_SIZE", raising=False)
    monkeypatch.delenv("TILEINFER_PP_SIZE", raising=False)
    env = probe()
    assert (env.tp_size, env.ep_size, env.dp_size, env.pp_size) == (4, 2, 1, 1)
    assert env.parallel_axes() == ("tp", "ep")


def test_probe_explicit_sizes_override_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("TILEINFER_TP_SIZE", "4")
    env = probe(tp_size=1)
    assert env.parallel_axes() == ("single",)
