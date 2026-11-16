"""Several implementations per operator name, chosen per call - the slot performance work needs.

Today every vLLM ABI name in :mod:`tileinfer.vllm.ops.families.gdn_moe` has exactly one
implementation: the torch port, bound directly by :mod:`tileinfer.vllm.ops.install`.  A TileLang
kernel cannot simply replace it in place, for three reasons that all have to be represented here:
it is only faster for some shapes, it must not become the default before a *device* run proves it,
and a bad kernel has to be disarmable without a code change (and without a restart).

What "candidate" means
----------------------
* :class:`Candidate` - one implementation plus *when* it may run: a ``predicate`` over the call's
  arguments, ``enabled_by_default``, and the ``measured_ratio`` a benchmark recorded for it.
* :class:`CallSpec` - the actual call's positional and keyword arguments as plain data, so a
  predicate can inspect ``.shape`` / ``.dtype`` / ``.is_contiguous()`` of the real tensors without
  calling the implementation and **without running a tensor op** (no ``.item()``, no ``.max()``, no
  arithmetic: reading those values would be a host sync, and the whole point of a predicate is that
  it is free).
* :class:`CandidateSet` - the candidates for one operator, in **priority order**, around a
  ``baseline`` that is always present.  :meth:`CandidateSet.select` is the per-call entry point and
  is deliberately tiny: no allocation, no tensor work, and the baseline is what comes back whenever
  nothing else qualifies (or the global kill switch is off).

The gate, in order
------------------
1. ``TILEINFER_OPS_CANDIDATES=off`` - the global kill switch.  It disarms **every** non-baseline
   candidate process-wide and is read per call (cheap: a dict hit), mirroring ``TILEINFER_OPS``.
2. ``enabled_by_default=False`` - the explicit "this is not a default yet" flag.
3. ``measured_ratio`` - the recorded speed-up over the baseline (``1.0`` = parity, ``0.86`` = 14%
   slower).  Below :data:`DEFAULT_THRESHOLD` a candidate is **not eligible even if the flag is on**,
   which is how this repo's cautionary example is encoded: the MoE grouped-GEMM kernel measured
   0.86x of the aclnn baseline on device after three optimisation rounds (``docs/status.md``,
   E19/H7) and must never become the default by accident.
4. the candidate's own ``predicate`` - shape / dtype / contiguity only.

Everything that falls through to the baseline is the pre-candidate behaviour, byte for byte.

What this module is *not*
-------------------------
No autotuning and no timing: a ``measured_ratio`` is a number a person measured offline and wrote
down (with the reason in ``note``).  No fallback that can change results: a predicate that raises is
logged once and treated as "no match" for the rest of the process, because a broken measurement must
not take a forward down.  And a candidate is never even inspected unless it is statically eligible
(flag + threshold) first - so a disabled entry costs one attribute test per call, nothing else.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, Mapping, Optional, Set, Tuple

__all__ = [
    "DEFAULT_THRESHOLD",
    "ENV_CANDIDATES",
    "CallSpec",
    "Candidate",
    "CandidateSet",
    "Predicate",
    "always",
    "candidates_enabled",
    "describe_switch",
]

logger = logging.getLogger("tileinfer.vllm.ops.candidates")

#: The kill switch, spelled like the family switch (``TILEINFER_OPS``): one variable, read per call.
ENV_CANDIDATES = "TILEINFER_OPS_CANDIDATES"

#: The contract line a recorded measurement has to reach: ``1.0`` = parity with the baseline.
#: Below it the candidate is not eligible whatever its flag says (see the module docstring).
DEFAULT_THRESHOLD = 1.0

#: Accepted spellings.  Anything else is a typo, and a typo must never *enable* an unmeasured
#: kernel: it fails safe to the baseline (the same "log once, keep going" convention as
#: ``switch.enabled_families``, but in the direction that cannot make a run worse).
_ON_VALUES = frozenset({"on", "1", "true", "yes", "enable", "enabled", "all"})
_OFF_VALUES = frozenset({"off", "0", "false", "no", "disable", "disabled", "none"})

#: Parsed kill switch, keyed by the **raw** value (``None`` = unset), so every distinct setting is
#: parsed once and the per-call path is a dict lookup - no ``strip()``/``lower()`` allocation, and a
#: flip of the variable is still seen immediately.  Bounded by the number of distinct values the
#: variable ever holds in this process.
_SWITCH: Dict[Optional[str], bool] = {None: True}

#: Values already complained about, so a malformed kill switch logs once per distinct value.
_WARNED: Set[str] = set()


def _parse_switch(raw: str) -> bool:
    """``raw`` -> enabled?  Unknown values are logged once and read as "off" (fail safe)."""
    value = raw.strip().lower()
    if value in _ON_VALUES:
        return True
    if value in _OFF_VALUES:
        return False
    if value not in _WARNED:
        _WARNED.add(value)
        logger.warning(
            "%s=%r is not a recognised value (on: %s | off: %s); treating candidate selection as "
            "OFF so only the baseline implementations run. Fix the variable and restart to take "
            "effect.",
            ENV_CANDIDATES,
            raw,
            ", ".join(sorted(_ON_VALUES)),
            ", ".join(sorted(_OFF_VALUES)),
        )
    return False


def candidates_enabled(spec: Optional[str] = None) -> bool:
    """Is non-baseline candidate selection allowed right now?  Never raises.

    ``spec=None`` reads :data:`ENV_CANDIDATES`; an unset or blank value means "no override"
    (enabled), and only an explicit off-ish value disarms the candidates.  The answer is cached per
    raw value, so calling this from a forward costs one dict lookup and allocates nothing.
    """
    raw = os.environ.get(ENV_CANDIDATES) if spec is None else spec
    try:
        return _SWITCH[raw]
    except KeyError:
        pass
    answer = True if raw is None or not str(raw).strip() else _parse_switch(raw)
    _SWITCH[raw] = answer
    return answer


def describe_switch(*, spec: Optional[str] = None) -> Dict[str, Any]:
    """The kill switch as plain data (``tileinfer doctor`` / the probe print this)."""
    raw = os.environ.get(ENV_CANDIDATES) if spec is None else spec
    return {
        "env": ENV_CANDIDATES,
        "raw": raw,
        "enabled": candidates_enabled(spec),
        "on_values": sorted(_ON_VALUES),
        "off_values": sorted(_OFF_VALUES),
        "default": "enabled (unset means no override)",
    }


# =============================================================================================
# What a predicate sees: the call, as data
# =============================================================================================


@dataclass(frozen=True)
class CallSpec:
    """One call's arguments, as data - all a predicate is allowed to look at.

    ``args``/``kwargs`` are the *actual* objects the caller passed (no copy, no unwrapping), so a
    predicate can read ``spec.arg(0, "hidden_states").shape`` and get the real shape.  Nothing here
    computes with a tensor: :meth:`arg` is a tuple index plus a dict lookup, which is what keeps
    ``select`` allocation-free apart from this one small object per call.
    """

    args: Tuple[Any, ...] = ()
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    op: str = ""

    def arg(self, index: int, name: str, default: Any = None) -> Any:
        """The ``index``-th positional argument, or the ``name`` keyword, or ``default``.

        Both spellings have to work: module-attribute call sites pass positionally
        (``l2norm_fwd(x, eps)``) while the ``CustomOp`` takeover forwards by keyword
        (``forward_native(q=..., k=...)``), and one predicate serves both.
        """
        if index < len(self.args):
            return self.args[index]
        return self.kwargs.get(name, default)

    def passed(self, index: int, name: str) -> bool:
        """Was the argument supplied at all (positionally or by keyword)?  No attribute access."""
        return index < len(self.args) or name in self.kwargs


#: A predicate: cheap, side-effect-free, and only shape/dtype/contiguity of the real arguments.
Predicate = Callable[[CallSpec], bool]


def always(_call_spec: CallSpec) -> bool:
    """The baseline's predicate: the fallback is never consulted, but it has to have the same
    callable shape as every other predicate so a set can be built uniformly."""
    return True


# =============================================================================================
# One candidate, and the set around the baseline
# =============================================================================================


def _describe_impl(impl: Any) -> str:
    """``module.qualname`` for a report line, without assuming the object is a function."""
    module = getattr(impl, "__module__", "") or "?"
    name = getattr(impl, "__qualname__", None) or getattr(impl, "__name__", "") or repr(impl)
    return f"{module}.{name}"


@dataclass(frozen=True)
class Candidate:
    """One implementation of one operator, plus the gate that decides whether it may run.

    ``measured_ratio`` is a *recorded* measurement, not something this module takes: the candidate's
    speed-up over the baseline it is registered against (``1.0`` = parity).  ``None`` means "not
    measured as a candidate" and leaves the decision to ``enabled_by_default``; a number below
    :data:`DEFAULT_THRESHOLD` makes the candidate ineligible even when the flag is on.
    """

    name: str
    impl: Callable[..., Any]
    predicate: Predicate
    enabled_by_default: bool
    note: str = ""
    measured_ratio: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("a Candidate needs a non-empty name (it is what describe() reports)")
        if not callable(self.impl):
            raise TypeError(f"candidate {self.name!r}: impl is not callable ({self.impl!r})")
        if not callable(self.predicate):
            raise TypeError(
                f"candidate {self.name!r}: predicate is not callable ({self.predicate!r})"
            )
        if self.measured_ratio is not None and not isinstance(self.measured_ratio, (int, float)):
            raise TypeError(
                f"candidate {self.name!r}: measured_ratio must be a number or None, "
                f"got {self.measured_ratio!r}"
            )

    def eligible(self, threshold: float = DEFAULT_THRESHOLD) -> bool:
        """The *static* half of the gate: the flag and the recorded measurement.

        No environment read and no call inspection, so this is the cheap filter ``select`` applies
        before it looks at a call at all.
        """
        if not self.enabled_by_default:
            return False
        return self.measured_ratio is None or self.measured_ratio >= threshold

    def as_dict(self, threshold: float = DEFAULT_THRESHOLD) -> Dict[str, Any]:
        """This candidate as plain data (notes included: the *reason* is part of the record)."""
        return {
            "name": self.name,
            "impl": _describe_impl(self.impl),
            "predicate": getattr(self.predicate, "__name__", repr(self.predicate)),
            "enabled_by_default": self.enabled_by_default,
            "measured_ratio": self.measured_ratio,
            "threshold": threshold,
            "eligible": self.eligible(threshold),
            "note": self.note,
        }


class CandidateSet:
    """Every implementation of one operator: candidates in priority order, plus a baseline.

    ``baseline`` is always present and is what :meth:`select` returns when no candidate qualifies,
    so the pre-candidate behaviour is the default by construction rather than by remembering to add
    it.  :meth:`add` appends, so **registration order is priority order**: the first candidate whose
    predicate matches wins.
    """

    __slots__ = ("op", "_baseline", "_candidates", "threshold", "_warned_predicates")

    def __init__(
        self,
        op: str,
        baseline: Candidate,
        *,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        if not isinstance(baseline, Candidate):
            raise TypeError(f"baseline must be a Candidate, got {baseline!r}")
        if not str(op):
            raise ValueError("a CandidateSet needs the operator name it belongs to")
        self.op = str(op)
        self._baseline = baseline
        self._candidates: Tuple[Candidate, ...] = ()
        self.threshold = float(threshold)
        #: candidate names whose predicate already raised, so the warning is logged once
        self._warned_predicates: Set[str] = set()

    # -- the surface -------------------------------------------------------------------------

    @property
    def baseline(self) -> Candidate:
        """The always-present fallback (the pre-candidate implementation)."""
        return self._baseline

    @property
    def candidates(self) -> Tuple[Candidate, ...]:
        """The non-baseline candidates, highest priority first."""
        return self._candidates

    def add(self, candidate: Candidate) -> Candidate:
        """Register a candidate at the **lowest** priority so far; returns it."""
        if not isinstance(candidate, Candidate):
            raise TypeError(f"add() takes a Candidate, got {candidate!r}")
        if candidate.name == self._baseline.name:
            raise ValueError(
                f"{self.op}: candidate name {candidate.name!r} is already the baseline's; names "
                "identify the implementation in describe() and in the log"
            )
        for existing in self._candidates:
            if existing.name == candidate.name:
                raise ValueError(
                    f"{self.op}: duplicate candidate name {candidate.name!r}; names identify the "
                    "implementation in describe() and in the log"
                )
        self._candidates = self._candidates + (candidate,)
        return candidate

    def __len__(self) -> int:
        return 1 + len(self._candidates)

    def __iter__(self) -> Iterator[Candidate]:
        """Candidates in priority order, then the baseline last (nothing jumps the queue)."""
        return iter(self._candidates + (self._baseline,))

    def __repr__(self) -> str:
        names = ", ".join(candidate.name for candidate in self._candidates)
        return (
            f"CandidateSet({self.op!r}, baseline={self._baseline.name!r}, "
            f"candidates=[{names}], threshold={self.threshold})"
        )

    # -- the per-call decision ---------------------------------------------------------------

    def select(self, call_spec: CallSpec) -> Callable[..., Any]:
        """The implementation to call for this call: first eligible candidate whose predicate
        matches, else the baseline.

        Cheap by contract: it reads only the candidate records and whatever the predicate reads
        (``.shape``/``.dtype``/``.is_contiguous()``), allocates nothing, and never runs the
        implementation or a tensor op.  It is called once per operator call, from the wrapper
        :meth:`dispatched` returns, so it must stay that way.
        """
        return self._choose(call_spec).impl

    def select_candidate(self, call_spec: CallSpec) -> Candidate:
        """:meth:`select` without the final ``.impl`` load - the winner, for probes and tests."""
        return self._choose(call_spec)

    def _choose(self, call_spec: CallSpec) -> Candidate:
        candidates = self._candidates
        if not candidates:  # the shipped case: nothing to decide, no env read, no predicate call
            return self._baseline
        for candidate in candidates:
            if not candidate.enabled_by_default:
                continue
            ratio = candidate.measured_ratio
            if ratio is not None and ratio < self.threshold:
                continue
            if not candidates_enabled():  # the global kill switch, read once per call at most
                return self._baseline
            if self._matches(candidate, call_spec):
                return candidate
        return self._baseline

    def _matches(self, candidate: Candidate, call_spec: CallSpec) -> bool:
        try:
            return bool(candidate.predicate(call_spec))
        except Exception as exc:  # noqa: BLE001 - a broken predicate must not fail a forward
            if candidate.name not in self._warned_predicates:
                self._warned_predicates.add(candidate.name)
                logger.warning(
                    "TileInfer op %s: candidate %r's predicate raised (%r); treating it as "
                    "'no match' for the rest of the process (the baseline still runs).",
                    self.op,
                    candidate.name,
                    exc,
                )
            return False

    # -- what install.py binds ---------------------------------------------------------------

    def dispatched(self) -> Callable[..., Any]:
        """A thin wrapper that calls :meth:`select` first; install.py binds **this**.

        The wrapper's only job is to turn a call into a :class:`CallSpec` and to forward the
        original positional/keyword arguments unchanged to the winner, so the vLLM ABI is exactly
        the implementation's own (no signature rewriting, no argument coercion).  It also carries
        the two markers install.py reports:

        ``__tileinfer_baseline__``   the baseline impl, so ``__tileinfer_impl__`` on the takeover
                                     keeps naming the implementation rather than this seam;
        ``__tileinfer_candidates__`` the set itself, for ``describe()`` / the probe.
        """
        op = self.op
        baseline = self._baseline.impl

        def dispatch(*args: Any, **kwargs: Any) -> Any:
            return self.select(CallSpec(args, kwargs, op))(*args, **kwargs)

        name = getattr(baseline, "__name__", None) or op
        dispatch.__name__ = f"{name}__candidates"
        dispatch.__qualname__ = dispatch.__name__
        dispatch.__doc__ = (
            f"{name} through {op}'s CandidateSet: {len(self._candidates)} candidate(s) over the "
            "baseline, selected per call (tileinfer.vllm.ops.candidates)."
        )
        dispatch.__tileinfer_candidates__ = self  # type: ignore[attr-defined]
        dispatch.__tileinfer_baseline__ = baseline  # type: ignore[attr-defined]
        return dispatch

    # -- the report --------------------------------------------------------------------------

    def describe(self) -> Dict[str, Any]:
        """The set as plain data: every candidate with its gate status and recorded reason.

        This is what ``tileinfer doctor`` / a probe prints, so a candidate that is off says *why*
        (the flag, the measured ratio, the kill switch) instead of silently not being used.
        """
        switch = describe_switch()
        global_on = bool(switch["enabled"])
        rows = [dict(self._baseline.as_dict(self.threshold), status="baseline")]
        for candidate in self._candidates:
            reasons = []
            if not candidate.enabled_by_default:
                reasons.append("off by default")
            ratio = candidate.measured_ratio
            if ratio is not None and ratio < self.threshold:
                reasons.append(
                    f"measured {float(ratio):.2f}x < {self.threshold:.2f}x threshold"
                )
            if not global_on:
                reasons.append(f"kill switch {ENV_CANDIDATES}=off")
            rows.append(
                dict(
                    candidate.as_dict(self.threshold),
                    status="eligible" if not reasons else "disabled: " + "; ".join(reasons),
                )
            )
        return {
            "op": self.op,
            "threshold": self.threshold,
            "switch": switch,
            "baseline": rows[0],
            "candidates": rows[1:],
        }
