"""Global default providers (currently empty).

Candidates and route-O providers are model-scoped and must not become defaults for
unrelated declarations.  The dict is kept mutable because tests and bring-up tools register a
temporary default to exercise the decision layer.
"""

from __future__ import annotations

from typing import Dict

from ..contracts.capabilities import Capability, ProviderSpec

GLOBAL_PROVIDERS: Dict[Capability, ProviderSpec] = {}

__all__ = ["GLOBAL_PROVIDERS"]
