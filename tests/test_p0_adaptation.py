"""P0 adaptation-layer contracts: provider lifecycle, config adapter, support metadata."""

from __future__ import annotations

import dataclasses
import sys
import types

from tileinfer.bridge.config_adapter import install_config_adapter, reset_config_adapters
from tileinfer.bridge import seams
from tileinfer.contracts.capabilities import Capability, ProviderSpec
from tileinfer.contracts.plan import Plan
from tileinfer.manifests import LFM3_1_7, QWEN3_6
from tileinfer.manifests.nemotron_3_5 import NEMOTRON_3_5_DRAFT
from tileinfer.providers import find_provider
from tileinfer.resolve.decide import decide
from tileinfer.resolve.probe import Environment


def test_provider_spec_lifecycle_fields_are_explicit() -> None:
    spec = ProviderSpec(
        capability=Capability.STATE_LINEAR,
        name="tileinfer.example",
        impl="tileinfer.providers.example",
        source="tileinfer",
        graphable=True,
        evidence="bench/example",
        status="candidate",
        scope="model",
        install="model_impl",
        graph_safe=True,
        evidence_level="device-c3",
    )
    spec.validate()
    data = spec.as_dict()
    assert data["status"] == "candidate"
    assert data["scope"] == "model"
    assert data["install"] == "model_impl"
    assert data["graph_safe"] is True
    assert data["evidence_level"] == "device-c3"


def test_lfm3_short_conv_is_model_scoped_and_consumed_by_the_model() -> None:
    spec = find_provider("tileinfer.short_conv")
    assert spec is not None
    assert spec.scope == "model"
    assert spec.install == "model_impl"
    assert spec.status == "default"
    assert spec.graph_safe is True


def test_impl_table_skips_a_model_consumed_provider_without_hardcoding_its_name() -> None:
    plan, _ = decide(
        LFM3_1_7,
        Environment(vllm="0.23.0", vllm_ascend="0.23.0", python="3.12", has_vendor=True),
    )
    result = seams.install_impl_table(plan)
    assert result.ok is True
    assert "model_impl" in result.detail


def test_config_adapter_runs_a_module_actor_and_is_idempotent(monkeypatch) -> None:
    module = types.ModuleType("fake_config_adapter")
    calls = []

    def register():
        calls.append(1)
        return {"normalized": True}

    module.register = register
    monkeypatch.setitem(sys.modules, "fake_config_adapter", module)
    reset_config_adapters()

    first = install_config_adapter("fake_config_adapter")
    second = install_config_adapter("fake_config_adapter")
    assert first["ok"] is True
    assert first["result"] == {"normalized": True}
    assert second["ok"] is True
    assert len(calls) == 1


def test_config_impl_is_carried_by_the_plan_and_changes_its_identity() -> None:
    plan, _ = decide(
        QWEN3_6,
        Environment(vllm="0.23.0", vllm_ascend="0.23.0", python="3.12", has_vendor=True),
    )
    payload = {**plan.as_dict(), "config_impl": "fake.config:register", "plan_id": ""}
    claimed = Plan.from_dict(payload)
    assert claimed.config_impl == "fake.config:register"
    assert claimed.plan_id != plan.plan_id
    assert Plan.from_dict(claimed.as_dict()).plan_id == claimed.plan_id


def test_support_metadata_does_not_change_the_content_addressed_manifest() -> None:
    changed = dataclasses.replace(QWEN3_6, support={"text": "verified", "mtp": "failed"})
    assert changed.fingerprint == QWEN3_6.fingerprint
    assert changed.as_dict()["support"]["mtp"] == "failed"


def test_nemotron_declares_config_adapter_and_partial_support() -> None:
    assert NEMOTRON_3_5_DRAFT.config_impl.endswith("register_nemotron_h_alias")
    assert NEMOTRON_3_5_DRAFT.support["text"] == "partial"
    assert NEMOTRON_3_5_DRAFT.support["mtp"] == "failed"
