"""The eight top-level capability contracts, plus the Mamba2 sub-contract.

The eight ``Capability`` values stay closed (the decision layer and its tests depend on
that).  ``Mamba2Contract`` is a sub-contract selected through ``state.linear``; it is not a
ninth capability, so adding it does not change plan routing.

Why protocols and not base classes
----------------------------------
The decision layer only needs to know *what a capability looks like* so it can match a
manifest against the environment.  It must never need an implementation object, because it
runs at deploy time on a machine that may have no engine and no device.  ``typing.Protocol``
gives us that structural contract without an import edge from the contract to any provider.

The signatures below are the vocabulary.  A provider is allowed to add keyword arguments;
it is not allowed to change the meaning of the ones listed here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence, Tuple, runtime_checkable


class Capability(str, Enum):
    """The closed set of capabilities a model declaration may require."""

    ATTENTION_FULL = "attention.full"
    ATTENTION_SLIDING = "attention.sliding"
    ATTENTION_LINEAR = "attention.linear"
    MOE_ROUTED = "moe.routed"
    NORM_RMS = "norm.rms"
    ROTARY = "rotary"
    SAMPLING_TOPK = "sampling.topk"
    STATE_LINEAR = "state.linear"
    QUANT_VERIFY = "quant.verify"


#: Stable iteration order; used by the plan writer so two runs produce identical files.
CAPABILITIES: Tuple[Capability, ...] = (
    Capability.ATTENTION_FULL,
    Capability.ATTENTION_SLIDING,
    Capability.ATTENTION_LINEAR,
    Capability.MOE_ROUTED,
    Capability.NORM_RMS,
    Capability.ROTARY,
    Capability.SAMPLING_TOPK,
    Capability.STATE_LINEAR,
    Capability.QUANT_VERIFY,
)

#: Closed vocabulary for ``ProviderSpec.parallel``.  ``single`` means every parallel axis is
#: 1; ``tp``/``ep``/``dp``/``pp`` mean "correct when that axis is larger than 1".  A provider
#: that runs both single-card and under TP declares ``("single", "tp")``; one that has only
#: been validated single-card declares ``("single",)`` and is refused under any sharding.  This
#: is data, not a new access point: adding a parallel dimension must never grow the mechanism
#: budget.
PARALLEL_MODES: Tuple[str, ...] = ("single", "tp", "ep", "dp", "pp")


def normalize_parallel(axes: Sequence[str]) -> Tuple[str, ...]:
    """Validate and de-duplicate a parallel declaration (order preserved)."""
    values = tuple(dict.fromkeys(str(axis) for axis in axes))
    if not values:
        raise ValueError("parallel must name at least one mode")
    unknown = sorted(set(values) - set(PARALLEL_MODES))
    if unknown:
        raise ValueError(
            f"unknown parallel modes {unknown}; known: {list(PARALLEL_MODES)}"
        )
    return values


def parallel_supports(supported: Sequence[str], required: Sequence[str]) -> bool:
    """Whether ``supported`` covers every axis in ``required`` (empty means single-card)."""
    needed = set(required) or {"single"}
    return needed <= set(supported)


@runtime_checkable
class AttentionContract(Protocol):
    """Full attention over a paged KV cache (prefill and decode)."""

    def plan(self, metadata: Any, **kwargs: Any) -> Any:
        """Turn pure batch metadata into a plan; no device allocation happens here."""

    def run(self, plan: Any, *args: Any, **kwargs: Any) -> Any:
        """Execute a plan built by :meth:`plan` against the engine's KV cache."""


@runtime_checkable
class LinearAttentionContract(Protocol):
    """Linear attention (GDN): a chunked prefill form and a recurrent decode form."""

    def chunk(self, q: Any, k: Any, v: Any, g: Any, beta: Any, **kwargs: Any) -> Any:
        """Chunked (prefill) form; returns the layer output and the new state."""

    def recurrent(self, q: Any, k: Any, v: Any, g: Any, beta: Any, state: Any, **kwargs: Any) -> Any:
        """Single-step (decode) form; updates ``state`` in place."""


@runtime_checkable
class MoEContract(Protocol):
    """Routed experts: routing, permutation and the expert matmuls."""

    def routed(self, hidden_states: Any, router_logits: Any, **kwargs: Any) -> Any:
        """Return the routed-expert contribution for a batch of tokens."""


@runtime_checkable
class NormContract(Protocol):
    """RMSNorm, standalone and fused with the residual add."""

    def rms_norm(self, x: Any, weight: Any, eps: float, **kwargs: Any) -> Any:
        """Normalise ``x`` in place-free fashion and return the result."""


@runtime_checkable
class RotaryContract(Protocol):
    """Rotary position embedding, including the interleaved MRoPE variant."""

    def apply(self, positions: Any, q: Any, k: Any, **kwargs: Any) -> Any:
        """Rotate ``q``/``k`` at ``positions`` and return them in the engine's layout."""


@runtime_checkable
class SamplingContract(Protocol):
    """The sampling primitives that sit between logits and the next token."""

    def topk_softmax(self, logits: Any, k: int, **kwargs: Any) -> Any:
        """Return ``(weights, token_ids)`` for the top-``k`` entries of ``logits``."""


@runtime_checkable
class StateContract(Protocol):
    """The recurrent state of a linear-attention layer (conv + ssm)."""

    def state_shape(self, spec: Any) -> Any:
        """Return the shapes the engine must allocate for this layer's state."""

    def update(self, state: Any, *args: Any, **kwargs: Any) -> Any:
        """Advance the state by one step and return it."""


@runtime_checkable
class Mamba2Contract(Protocol):
    """Mamba2 mixer compute, a sub-contract of ``state.linear``.

    Kept separate from :class:`LinearAttentionContract` (GDN's q/k/v/g/beta) because the
    two have different math and state shapes.  The decision layer still selects this through
    the existing ``state.linear`` capability; the provider's ``covers`` map declares
    ``{"mixer": "mamba2"}`` so a GDN implementation cannot be mistaken for it.
    """

    def forward(self, hidden_states: Any, mup_vector: Any | None = None) -> Any:
        """Run the conv/SSM/gated-norm/out-proj path for this mixer."""

    def state_shape(self, spec: Any) -> Any:
        """Return the shapes the engine must allocate for conv + SSM state."""


@runtime_checkable
class QuantConfigContract(Protocol):
    """What a provider must be able to answer about a checkpoint's quantization."""

    def verify(self, quant_config: Any) -> bool:
        """Return whether this provider can execute the given quantization scheme."""


# The quant contract is named ``QuantContract`` in the public surface; keep the protocol
# name readable in isolation while exporting the shorter alias.
QuantContract = QuantConfigContract


@dataclass(frozen=True)
class InstallAction:
    """One declared side effect of installing a provider."""

    kind: str  # impl_bind | vendor_patch | model_call
    actor: str = ""
    hook: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "actor": self.actor, "hook": self.hook}


@dataclass(frozen=True)
class ProviderSpec:
    """A provider's declaration (L4).  Five fields are mandatory by project rule.

    A provider exists only for a gap the vendor cannot serve (``docs/extending.md`` §3.0):
    when the vendor runs the capability correctly and graphably, the decision selects the
    vendor entry and no provider is written at all.  ``covers`` is symbolic on purpose: a
    provider says which shapes/precisions it handles (``{"head_dim": 256, "dtype":
    ["bf16"]}``), never a concrete tensor.  ``measured_ratio`` is the speed ratio against the
    arm this provider replaces - the vendor path, or the vendor's own fallback - and it only
    exists when that arm runs at all.  When the vendor cannot deploy the capability there is
    nothing to divide by: ``measured_ratio`` stays ``None`` and ``evidence`` carries the
    absolute numbers plus the reference-implementation agreement instead.  Evidence is
    mandatory either way; a provider without one is a candidate, never a default.

    ``source`` still accepts ``vendor`` because the same shape describes the vendor entries
    in the plan table (``providers/vendor.py``); those are *selections*, not providers in
    ``providers/``.
    """

    capability: Capability
    name: str
    impl: str
    source: str  # vendor | tileinfer | vendored | reference
    graphable: bool
    covers: Mapping[str, Any] = field(default_factory=dict)
    parallel: Tuple[str, ...] = ("single",)
    measured_ratio: Optional[float] = None
    evidence: str = ""
    delete_when: str = ""
    notes: str = ""
    #: Selection lifecycle.  ``candidate`` may only enter a plan through an explicit
    #: bring-up (``allow_candidates``) and keeps the plan unservable; ``default`` is eligible
    #: for normal vendor-first selection.
    status: str = "default"
    #: Where the declaration applies.  ``global`` is the aspect/state provider table;
    #: ``model`` only applies when the declaration ships a ``model_impl`` and consumes the
    #: provider inside that model (LFM3 short-conv).
    scope: str = "global"
    #: How the bridge installs it: an operator-table entry, a provider consumed by our model
    #: implementation, or a vendor hook (platform guard / monkeypatch).
    install: str = "impl_table"
    #: Optional legacy single binder.  Prefer ``install_actions`` for new providers.
    install_actor: str = ""
    #: Declared side effects.  A provider may bind an operator, install one or more vendor
    #: patches, and/or be consumed directly by the model implementation.
    install_actions: Tuple[InstallAction, ...] = ()
    #: Measured under ACL graph capture (replay, no host sync, no fallback), not merely
    #: "this implementation is graph-shaped".
    graph_safe: bool = False
    #: Machine-readable evidence maturity, independent of the free-form ``evidence`` path.
    evidence_level: str = "declared"

    def validate(self) -> None:
        if not self.name or not self.impl:
            raise ValueError("a provider needs a name and an importable implementation path")
        if self.source not in {"vendor", "tileinfer", "vendored", "reference"}:
            raise ValueError(f"unknown provider source {self.source!r}")
        if self.status not in {"default", "candidate"}:
            raise ValueError(f"unknown provider status {self.status!r}")
        if self.scope not in {"global", "model"}:
            raise ValueError(f"unknown provider scope {self.scope!r}")
        if self.install not in {"impl_table", "model_impl", "vendor_hook"}:
            raise ValueError(f"unknown provider install mode {self.install!r}")
        for action in self.install_actions:
            if action.kind not in {"impl_bind", "vendor_patch", "model_call"}:
                raise ValueError(f"unknown install action {action.kind!r}")
        if self.evidence_level not in {
            "declared",
            "candidate",
            "cpu",
            "device-c1",
            "device-c2",
            "device-c3",
            "device-c4",
            "verified",
        }:
            raise ValueError(f"unknown provider evidence level {self.evidence_level!r}")
        normalize_parallel(self.parallel)
        if self.measured_ratio is not None and not self.evidence:
            raise ValueError(
                f"provider {self.name!r} declares a measured ratio without an evidence id"
            )
        if not self.evidence:
            raise ValueError(
                f"provider {self.name!r} has no evidence id; a ratio is optional (only when "
                "the replaced arm runs), evidence is not - without it this is a candidate, "
                "not a default"
            )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "capability": self.capability.value,
            "name": self.name,
            "impl": self.impl,
            "source": self.source,
            "graphable": self.graphable,
            "covers": dict(self.covers),
            "parallel": list(self.parallel),
            "measured_ratio": self.measured_ratio,
            "evidence": self.evidence,
            "delete_when": self.delete_when,
            "notes": self.notes,
            "status": self.status,
            "scope": self.scope,
            "install": self.install,
            "install_actor": self.install_actor,
            "install_actions": [action.as_dict() for action in self.install_actions],
            "graph_safe": self.graph_safe,
            "evidence_level": self.evidence_level,
        }
