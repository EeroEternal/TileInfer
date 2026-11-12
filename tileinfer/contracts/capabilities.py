"""The eight capability contracts, and the shape of a provider declaration.

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
    Capability.ATTENTION_LINEAR,
    Capability.MOE_ROUTED,
    Capability.NORM_RMS,
    Capability.ROTARY,
    Capability.SAMPLING_TOPK,
    Capability.STATE_LINEAR,
    Capability.QUANT_VERIFY,
)


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
class QuantConfigContract(Protocol):
    """What a provider must be able to answer about a checkpoint's quantization."""

    def verify(self, quant_config: Any) -> bool:
        """Return whether this provider can execute the given quantization scheme."""


# The quant contract is named ``QuantContract`` in the public surface; keep the protocol
# name readable in isolation while exporting the shorter alias.
QuantContract = QuantConfigContract


@dataclass(frozen=True)
class ProviderSpec:
    """A provider's declaration (L4).  Five fields are mandatory by project rule.

    ``covers`` is symbolic on purpose: a provider says which shapes/precisions it handles
    (``{"head_dim": 256, "dtype": ["bf16"]}``), never a concrete tensor.  ``measured_ratio``
    is the *measured* speed ratio against the vendor implementation and must have an
    evidence id; a provider without one may only be a candidate, never a default.
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

    def validate(self) -> None:
        if not self.name or not self.impl:
            raise ValueError("a provider needs a name and an importable implementation path")
        if self.source not in {"vendor", "tileinfer", "vendored", "reference"}:
            raise ValueError(f"unknown provider source {self.source!r}")
        if self.measured_ratio is not None and not self.evidence:
            raise ValueError(
                f"provider {self.name!r} declares a measured ratio without an evidence id"
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
        }
