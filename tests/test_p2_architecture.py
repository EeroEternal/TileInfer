"""Architecture contracts added after P1: registry, result type, mechanisms, runtime gate."""

from __future__ import annotations

import sys
import types

import pytest

from tileinfer.bridge.results import InstallResult
from tileinfer.bridge.runtime_gate import check_plan
from tileinfer.bridge.seams import install_model_table
from tileinfer.contracts.capabilities import Capability, ProviderSpec
from tileinfer.contracts.manifest import SupportSpec
from tileinfer.contracts.plan import PLAN_MECHANISMS, Plan
from tileinfer.manifests import QWEN3_6, LFM3_1_7
from tileinfer.providers import GLOBAL_PROVIDERS, find_provider, records_for
from tileinfer.resolve.decide import decide
from tileinfer.resolve.probe import Environment


def _env():
    return Environment(vllm="0.23.0", vllm_ascend="0.23.0", python="3.12", has_vendor=True)


def test_registry_sees_all_origins_and_runtime_global_entries(monkeypatch) -> None:
    test_spec = ProviderSpec(
        capability=Capability.NORM_RMS,
        name="tileinfer.test_global",
        impl="tileinfer.providers.state.short_conv",
        source="tileinfer",
        graphable=False,
        evidence="test",
    )
    monkeypatch.setitem(GLOBAL_PROVIDERS, Capability.NORM_RMS, test_spec)
    assert find_provider("tileinfer.test_global") is test_spec
    origins = {record.origin for record in records_for(Capability.STATE_LINEAR)}
    assert {"vendor", "route_o", "candidate"} <= origins


def test_support_spec_rejects_unknown_status_and_keeps_list_shape() -> None:
    with pytest.raises(ValueError):
        SupportSpec({"text": "almost"})
    support = SupportSpec({"text": "verified", "tp": ["1", "2"]})
    assert support["tp"] == ["1", "2"]
    assert support.as_dict()["text"] == "verified"


def test_install_result_normalises_a_false_actor_return() -> None:
    failed = InstallResult.from_mapping("actor", {"registered": False})
    assert failed.ok is False
    ok = InstallResult.from_mapping("actor", {"registered": True})
    assert ok.ok is True


def test_model_table_fails_closed_when_an_actor_reports_failure(monkeypatch) -> None:
    module = types.ModuleType("fake_failed_model")
    module.register_models = lambda: {"registered": False, "detail": "nope"}
    monkeypatch.setitem(sys.modules, "fake_failed_model", module)

    plan, _ = decide(QWEN3_6, _env())
    payload = {**plan.as_dict(), "model_impl": "fake_failed_model", "plan_id": ""}
    failed_plan = Plan.from_dict(payload)
    result = install_model_table(failed_plan)
    assert result.ok is False
    assert "reported failure" in result.detail


def test_lfm3_plan_records_real_mechanisms() -> None:
    plan, _ = decide(LFM3_1_7, _env())
    assert "model_register" in plan.mechanisms
    assert "model_call" in plan.mechanisms
    assert set(plan.mechanisms) <= PLAN_MECHANISMS


def test_runtime_gate_refuses_a_parallel_mismatch() -> None:
    plan, _ = decide(QWEN3_6, _env())
    reason = check_plan(plan, {}, parallel_axes=("tp",), check_versions=False)
    assert reason is not None
    assert "parallel gate" in reason


def test_provider_install_actions_are_declared() -> None:
    mamba = find_provider("tileinfer.mamba2_state")
    moe = find_provider("tileinfer.nemotron_moe")
    assert mamba is not None and moe is not None
    assert [action.kind for action in mamba.install_actions] == ["impl_bind"]
    assert [action.kind for action in moe.install_actions] == [
        "vendor_patch",
        "vendor_patch",
        "vendor_patch",
    ]


def test_model_adapter_pipeline_is_ordered_and_cached(monkeypatch) -> None:
    from tileinfer.bridge.model_adapters import ModelAdapters, reset_model_adapters

    calls: list[str] = []
    config = types.ModuleType("fake_cfg")
    config.register = lambda: calls.append("config") or {"registered": True}
    model = types.ModuleType("fake_model")
    model.register_config = lambda: calls.append("model_config") or {"registered": True}
    model.register_models = lambda: calls.append("model_models") or {"registered": True}
    monkeypatch.setitem(sys.modules, "fake_cfg", config)
    monkeypatch.setitem(sys.modules, "fake_model", model)
    reset_model_adapters()

    adapters = ModelAdapters(config="fake_cfg", model="fake_model")
    result = adapters.install()
    assert result.ok
    assert calls == ["config", "model_config", "model_models"]
    cached = adapters.install()
    assert cached.ok
    assert calls == ["config", "model_config", "model_models"]
