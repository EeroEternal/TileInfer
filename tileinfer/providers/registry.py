"""One registry for every provider declaration.

Four tables still exist because they have different lifetimes (vendor data, global defaults,
route-O model-scoped declarations, candidates), but callers no longer need to know which one
to consult.  ``find_provider``/``providers_for`` return the same ``ProviderSpec`` regardless
of where it was declared, and ``origin`` makes the source explicit for audit and doctor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from ..contracts.capabilities import Capability, ProviderSpec
from .catalog import ROUTE_O_PROVIDERS
from .global_ import GLOBAL_PROVIDERS
from .nemotron import CANDIDATES
from .vendor import VENDOR_PROVIDERS


@dataclass(frozen=True)
class ProviderRecord:
    spec: ProviderSpec
    origin: str
    selectable: bool

    @property
    def name(self) -> str:
        return self.spec.name


def records() -> Tuple[ProviderRecord, ...]:
    out: List[ProviderRecord] = []
    out.extend(
        ProviderRecord(spec=spec, origin="vendor", selectable=True)
        for spec in VENDOR_PROVIDERS.values()
    )
    out.extend(
        ProviderRecord(spec=spec, origin="global", selectable=spec.status == "default")
        for spec in GLOBAL_PROVIDERS.values()
    )
    out.extend(
        ProviderRecord(spec=spec, origin="route_o", selectable=spec.status == "default")
        for spec in ROUTE_O_PROVIDERS.values()
    )
    out.extend(
        ProviderRecord(spec=spec, origin="candidate", selectable=False)
        for spec in CANDIDATES
    )
    return tuple(out)


def provider_specs() -> Dict[str, ProviderSpec]:
    return {record.name: record.spec for record in records()}


def records_for(capability: Capability) -> Tuple[ProviderRecord, ...]:
    return tuple(record for record in records() if record.spec.capability is capability)


def providers_for(capability: Capability) -> Tuple[ProviderSpec, ...]:
    return tuple(record.spec for record in records_for(capability))


def find_provider(name: str) -> ProviderSpec | None:
    for record in records():
        if record.name == name:
            return record.spec
    return None


def global_provider(capability: Capability) -> ProviderSpec | None:
    return GLOBAL_PROVIDERS.get(capability)


def route_providers(capability: Capability) -> Tuple[ProviderSpec, ...]:
    return tuple(
        record.spec
        for record in records_for(capability)
        if record.origin == "route_o"
    )


def candidate_providers(capability: Capability) -> Tuple[ProviderSpec, ...]:
    return tuple(
        record.spec
        for record in records_for(capability)
        if record.origin == "candidate"
    )
