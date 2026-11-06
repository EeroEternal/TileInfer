"""Which operator families :func:`tileinfer.vllm.ops.install.install_ops` may wire.

The convention is copied from ``ops_enabled`` in the legacy adapter under ``legacy/`` - the
**function only**, that module is deliberately not imported here because importing it pulls in the
vendor package (the repo-wide ratchet in ``tests/`` pins that hard).  One environment variable,
``TILEINFER_OPS``,
holds a comma-separated list of family names, and it is read **per call**, so flipping the variable
(+ a restart) is the whole fallback story - no config file, no recompile.

Differences from the legacy switch, and why
-------------------------------------------
* **The default set is explicit.**  The legacy switch treats unset/empty as "all on", which is right
  for opt-in optimisations whose fallback is the stock CUDA/Triton path.  Here the fallback does not
  exist for the target model: without the GDN and MoE takeovers the model cannot run on Ascend at
  all (``docs/plan-vllm-native-backend.md`` §4: "6-10 是唯一没有 native 兜底的族").  So unset means
  :data:`OPS_DEFAULT_ON` - ``gdn`` and ``moe`` on, ``dense`` off - not "everything".
* **``dense`` is off by default** because its takeover names are still [unverified] offline: no
  module in ``runtime/vllm-api/`` shows where vLLM 0.23 calls ``silu_and_mul`` (see ``install.py``),
  so enabling it by default would only add unbound warnings on every boot.
* **An unknown name still raises** from :func:`ops_enabled` - that is the legacy contract and the
  only way a typo is ever noticed (``tests/test_gdn_proj_fusion.py`` pins the same behaviour).
  :func:`enabled_families` is the tolerant form the installer uses instead: it logs the offending
  names and falls back to the defaults, because a ``vllm.general_plugins`` entry point must never
  raise (it runs in every process of the server, including the ones that only serve the API).
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional, Set, Tuple

__all__ = [
    "ENV_OPS",
    "OPS_ALL",
    "OPS_DEFAULT_ON",
    "enabled_families",
    "ops_enabled",
    "parse_spec",
    "describe",
]

logger = logging.getLogger("tileinfer.vllm.ops.switch")

#: The environment variable, spelled exactly as the legacy adapter spells it.
ENV_OPS = "TILEINFER_OPS"

#: Every family the installer understands.  ``gdn`` = the FLA/GDN operator family,
#: ``moe`` = routing + expert MLP, ``dense`` = the shared/dense activation ops.
OPS_ALL: Tuple[str, ...] = ("gdn", "moe", "dense")

#: What an unset/empty ``TILEINFER_OPS`` means.  See the module docstring: unlike the legacy
#: switch this is *not* "all on" - the target model cannot run without the first two.
OPS_DEFAULT_ON: Tuple[str, ...] = ("gdn", "moe")


def parse_spec(spec: Optional[str] = None) -> Tuple[Set[str], Tuple[str, ...]]:
    """``(enabled_families, unknown_names)`` for ``spec`` (default: the environment variable).

    ``spec=None`` reads ``TILEINFER_OPS``; an unset **or blank** value returns
    :data:`OPS_DEFAULT_ON` (the legacy switch has the same "blank means default" rule, see the
    legacy adapter under ``legacy/``).  Unknown names are returned rather than dropped, so
    the caller decides whether they are fatal (``ops_enabled``) or a logged nuisance
    (``enabled_families``).
    """
    if spec is None:
        spec = os.environ.get(ENV_OPS)
    if spec is None or not spec.strip():
        return set(OPS_DEFAULT_ON), ()
    names = {chunk.strip().lower() for chunk in spec.split(",") if chunk.strip()}
    return names & set(OPS_ALL), tuple(sorted(names - set(OPS_ALL)))


def ops_enabled(op: str, *, spec: Optional[str] = None) -> bool:
    """Is the takeover for ``op`` switched on?  Raises on a name this module does not know.

    This is the strict, per-call form (the legacy adapter's contract): an unknown name is a
    configuration error and has to be visible, not silently ignored.  The installer and the
    per-call guards inside the takeovers use :func:`enabled_families` instead, which cannot raise.
    """
    family = op.strip().lower()
    enabled, unknown = parse_spec(spec)
    if unknown:
        raise ValueError(
            f"{ENV_OPS} contains unknown ops {list(unknown)}; valid: {sorted(OPS_ALL)}"
        )
    if family not in OPS_ALL:
        raise ValueError(
            f"{family!r} is not a takeover family; valid: {sorted(OPS_ALL)}"
        )
    return family in enabled


def enabled_families(*, spec: Optional[str] = None) -> Set[str]:
    """The tolerant form of :func:`ops_enabled`: never raises, logs an unusable spec.

    Unknown names fall back to :data:`OPS_DEFAULT_ON` *entirely*, on purpose: a typo such as
    ``TILEINFER_OPS=gend`` would otherwise disable every takeover (the parsed set would be empty)
    and take the model down, whereas falling back to the defaults keeps a runnable engine and puts
    the mistake in the log at WARNING.  (The per-call :func:`ops_enabled` still raises for callers
    that want to fail fast.)
    """
    enabled, unknown = parse_spec(spec)
    if unknown:
        logger.warning(
            "%s contains unknown ops %s (valid: %s); falling back to the default set %s. "
            "Fix the variable and restart to take effect.",
            ENV_OPS,
            list(unknown),
            sorted(OPS_ALL),
            sorted(OPS_DEFAULT_ON),
        )
        return set(OPS_DEFAULT_ON)
    return enabled


def describe(*, spec: Optional[str] = None) -> Dict[str, object]:
    """The switch as plain data (``tileinfer doctor`` and the install log line print this)."""
    raw = os.environ.get(ENV_OPS)
    enabled, unknown = parse_spec(spec)
    return {
        "env": ENV_OPS,
        "raw": raw,
        "enabled": sorted(enabled),
        "disabled": sorted(set(OPS_ALL) - enabled),
        "unknown": list(unknown),
        "default_on": sorted(OPS_DEFAULT_ON),
        "all": sorted(OPS_ALL),
    }
