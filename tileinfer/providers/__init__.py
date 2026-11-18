"""L4 - implementation providers.

Qwen3.6 needs no provider: every capability it declares is served by vllm-ascend.  LFM3 is
the first model with a real gap - its short convolution state is not GDN - so
``state/short_conv.py`` exists and is declared as the route-O model's own state provider.

A provider is written only when the vendor cannot use a capability at all, cannot run it
correctly, or cannot run it at the shape we need (``docs/extending.md`` §3.0).  Serving a
capability the vendor already serves is not a gap, and re-implementing it is the cost v2
already paid.

Providers here are implementations plus their five declarations (coverage, graphability,
parallelism, evidence, delete condition).  They import torch and the capability contracts,
never vLLM or vllm-ascend; installation belongs to ``tileinfer.bridge``.

The v2 operator tree was moved to ``legacy/v2/tileinfer/vllm/ops`` on 2026-11-16.  It is not
importable and has no compatibility shims in the installed package; v3 code must not reference
``tileinfer.vllm`` at all.
"""

from __future__ import annotations

from .catalog import ROUTE_O_PROVIDERS
from .nemotron import CANDIDATES, MAMBA2_STATE, NEMOTRON_MOE

#: Declared candidates, not defaults.  ``resolve.decide.TILEINFER_PROVIDERS`` stays empty until
#: a candidate has target real-weight C1/C2 evidence; see ``docs/extending.md`` §3.
CANDIDATE_PROVIDERS = CANDIDATES

#: Every declaration the bridge may need to resolve by name.  This is deliberately a name
#: lookup, not a selection: the plan still decides which provider is installed.
PROVIDER_SPECS = {
    **{provider.name: provider for provider in ROUTE_O_PROVIDERS.values()},
    **{provider.name: provider for provider in CANDIDATE_PROVIDERS},
}


def find_provider(name: str):
    """Return the declaration for ``name``, or ``None`` for an unknown/vendor name."""
    return PROVIDER_SPECS.get(name)


__all__ = [
    "MAMBA2_STATE",
    "NEMOTRON_MOE",
    "CANDIDATE_PROVIDERS",
    "PROVIDER_SPECS",
    "find_provider",
]
