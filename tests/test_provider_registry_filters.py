"""Registry helpers filter by lifecycle, not just by origin."""

from __future__ import annotations

from tileinfer.contracts.capabilities import Capability, ProviderSpec
from tileinfer.providers.catalog import ROUTE_O_PROVIDERS
from tileinfer.providers.global_ import GLOBAL_PROVIDERS
from tileinfer.providers.registry import (
    candidate_providers,
    global_provider,
    records_for,
    route_providers,
    vendor_provider,
)


def _spec(name: str, status: str, capability: Capability = Capability.STATE_LINEAR) -> ProviderSpec:
    return ProviderSpec(
        capability=capability,
        name=name,
        impl="tileinfer.providers.state.short_conv:ShortConvState",
        source="tileinfer",
        graphable=False,
        covers={"mixer": "short_conv"},
        status=status,
        evidence="unit-test-only",
        delete_when="test",
    )


def test_global_provider_only_returns_defaults(monkeypatch) -> None:
    monkeypatch.setitem(GLOBAL_PROVIDERS, Capability.STATE_LINEAR, _spec("test.candidate", "candidate"))
    assert global_provider(Capability.STATE_LINEAR) is None

    default = _spec("test.default", "default")
    monkeypatch.setitem(GLOBAL_PROVIDERS, Capability.STATE_LINEAR, default)
    assert global_provider(Capability.STATE_LINEAR) is default


def test_route_providers_skip_candidate_entries(monkeypatch) -> None:
    monkeypatch.setitem(
        ROUTE_O_PROVIDERS,
        "test.route_candidate",
        _spec("test.route_candidate", "candidate"),
    )
    names = {provider.name for provider in route_providers(Capability.STATE_LINEAR)}
    assert "tileinfer.short_conv" in names
    assert "test.route_candidate" not in names


def test_candidates_are_visible_but_not_selectable() -> None:
    candidates = candidate_providers(Capability.STATE_LINEAR)
    assert candidates
    assert all(provider.status == "candidate" for provider in candidates)
    records = {record.name: record for record in records_for(Capability.STATE_LINEAR)}
    assert records["tileinfer.mamba2_state"].selectable is False


def test_vendor_provider_returns_the_baseline() -> None:
    provider = vendor_provider(Capability.ATTENTION_FULL)
    assert provider is not None
    assert provider.name == "vllm_ascend.attention"
