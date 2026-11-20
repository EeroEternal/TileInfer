"""L4 - implementation providers.

Providers here are implementations plus their declarations.  The unified registry lives in
:mod:`tileinfer.providers.registry`; callers should use ``find_provider`` / ``providers_for``
instead of consulting a particular table.  The four source tables remain because vendor data,
global defaults, route-O declarations and candidates have different lifetimes.
"""

from __future__ import annotations

from .catalog import ROUTE_O_PROVIDERS
from .global_ import GLOBAL_PROVIDERS
from .nemotron import CANDIDATES, MAMBA2_STATE, NEMOTRON_MOE
from .registry import (
    ProviderRecord,
    candidate_providers,
    find_provider,
    global_provider,
    provider_specs,
    providers_for,
    records,
    records_for,
    route_providers,
    vendor_provider,
)
from .vendor import VENDOR_PROVIDERS

#: Declared candidates, not defaults.  A candidate is selectable only through an explicit
#: bring-up (``allow_candidates``); see ``docs/extending.md`` §3.
CANDIDATE_PROVIDERS = CANDIDATES

#: Compatibility name for code that only needs a name -> declaration map.  Prefer
#: ``find_provider`` in new code.
PROVIDER_SPECS = provider_specs()

__all__ = [
    "VENDOR_PROVIDERS",
    "ROUTE_O_PROVIDERS",
    "GLOBAL_PROVIDERS",
    "CANDIDATE_PROVIDERS",
    "MAMBA2_STATE",
    "NEMOTRON_MOE",
    "PROVIDER_SPECS",
    "ProviderRecord",
    "find_provider",
    "providers_for",
    "provider_specs",
    "records",
    "records_for",
    "global_provider",
    "vendor_provider",
    "route_providers",
    "candidate_providers",
]
