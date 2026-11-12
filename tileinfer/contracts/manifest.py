"""The model declaration: what a model *needs*, never who provides it.

A declaration is the input to the decision layer and must be computable without loading
weights or touching a device.  It therefore describes structure and requirements only:

* the layer sequence (full attention vs linear attention),
* the per-flavour parameters (heads, head dims, convolution width, ...),
* the KV grouping and page constraints,
* the quantization scheme,
* the capability list the decision layer must match,
* one benchmark sample with the expected first-token top-5, so "runs correctly" has a
  model-local oracle.

It must **not** say "use vllm_ascend's backend" - that is the plan's job.  Keeping policy
out of the declaration is what lets the same model run on a different provider later.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .capabilities import CAPABILITIES, Capability

MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class AttentionSpec:
    """Full-attention parameters (one per model, not per layer)."""

    kind: str
    num_heads: int
    num_kv_heads: int
    head_dim: int
    output_gate: bool = False
    rope: str = "mrope"
    partial_rotary_factor: float = 1.0

    def validate(self) -> None:
        if self.num_heads <= 0 or self.num_kv_heads <= 0 or self.head_dim <= 0:
            raise ValueError("attention head counts and head_dim must be positive")
        if self.num_heads % self.num_kv_heads:
            raise ValueError("num_heads must be a multiple of num_kv_heads (GQA grouping)")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "output_gate": self.output_gate,
            "rope": self.rope,
            "partial_rotary_factor": self.partial_rotary_factor,
        }


@dataclass(frozen=True)
class LinearSpec:
    """Linear-attention (GDN) parameters."""

    kind: str = "gdn"
    num_k_heads: int = 16
    num_v_heads: int = 32
    head_k_dim: int = 128
    head_v_dim: int = 128
    conv_kernel_dim: int = 4
    state_dtype: str = "float32"

    def validate(self) -> None:
        if min(self.num_k_heads, self.num_v_heads, self.head_k_dim, self.head_v_dim) <= 0:
            raise ValueError("linear-attention head counts and head dims must be positive")
        if self.num_v_heads % self.num_k_heads:
            raise ValueError("num_v_heads must be a multiple of num_k_heads (GDN grouping)")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "num_k_heads": self.num_k_heads,
            "num_v_heads": self.num_v_heads,
            "head_k_dim": self.head_k_dim,
            "head_v_dim": self.head_v_dim,
            "conv_kernel_dim": self.conv_kernel_dim,
            "state_dtype": self.state_dtype,
        }


@dataclass(frozen=True)
class MoESpec:
    """Routed-expert parameters."""

    num_experts: int = 256
    top_k: int = 8
    intermediate_size: int = 512
    shared_experts: int = 0

    def validate(self) -> None:
        if self.num_experts <= 0 or not (0 < self.top_k <= self.num_experts):
            raise ValueError("MoE needs num_experts > 0 and 0 < top_k <= num_experts")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "intermediate_size": self.intermediate_size,
            "shared_experts": self.shared_experts,
        }


@dataclass(frozen=True)
class ModelManifest:
    """One model, one declaration (single source of truth)."""

    name: str
    manifest_id: str
    arch: str
    model_type: str
    num_layers: int
    layer_types: Tuple[str, ...]
    attention: AttentionSpec
    linear: Optional[LinearSpec] = None
    moe: Optional[MoESpec] = None
    quantization: str = "none"
    kv: Dict[str, Any] = field(default_factory=dict)
    required_capabilities: Tuple[str, ...] = ()
    benchmark: Dict[str, Any] = field(default_factory=dict)
    source: str = "docs/notes/op-coverage-qwen36.md"
    schema_version: int = MANIFEST_SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != MANIFEST_SCHEMA_VERSION:
            raise ValueError("unsupported manifest schema version")
        if not self.name or not self.manifest_id or not self.arch:
            raise ValueError("manifest needs name, manifest_id and arch")
        if len(self.layer_types) != self.num_layers:
            raise ValueError(
                f"layer_types has {len(self.layer_types)} entries for {self.num_layers} layers"
            )
        unknown = sorted(set(self.layer_types) - {"full_attention", "linear_attention"})
        if unknown:
            raise ValueError(f"unknown layer types {unknown}")
        if self.linear is not None and "linear_attention" not in self.layer_types:
            raise ValueError("linear spec declared but no linear_attention layer")
        if self.moe is not None and "moe" not in self.required_capabilities and (
            Capability.MOE_ROUTED.value not in self.required_capabilities
        ):
            raise ValueError("MoE spec declared but moe.routed is not required")
        self.attention.validate()
        if self.linear is not None:
            self.linear.validate()
        if self.moe is not None:
            self.moe.validate()
        caps = set(self.required_capabilities)
        valid = {c.value for c in CAPABILITIES}
        unknown_caps = sorted(caps - valid)
        if unknown_caps:
            raise ValueError(f"manifest requires unknown capabilities {unknown_caps}")
        if not caps:
            raise ValueError("a manifest must declare at least one required capability")
        if self.quantization not in {"none", "fp8", "fp4", "w4a8"}:
            raise ValueError(f"unknown quantization scheme {self.quantization!r}")
        for key in ("prompt", "max_tokens", "top5"):
            if key not in self.benchmark:
                raise ValueError(f"benchmark sample is missing {key!r}")

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":")).encode()
        import hashlib

        return hashlib.sha256(payload).hexdigest()[:16]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "manifest_id": self.manifest_id,
            "arch": self.arch,
            "model_type": self.model_type,
            "num_layers": self.num_layers,
            "layer_types": list(self.layer_types),
            "attention": self.attention.as_dict(),
            "linear": self.linear.as_dict() if self.linear else None,
            "moe": self.moe.as_dict() if self.moe else None,
            "quantization": self.quantization,
            "kv": dict(self.kv),
            "required_capabilities": list(self.required_capabilities),
            "benchmark": dict(self.benchmark),
            "source": self.source,
        }

    def write(self, path: str) -> None:
        self.validate()
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.as_dict(), fh, ensure_ascii=False, indent=2)
            fh.write("\n")
