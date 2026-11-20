"""The model declaration: what a model *needs*, never who provides it.

A declaration is the input to the decision layer and must be computable without loading
weights or touching a device.  It therefore describes structure and requirements only:

* the layer sequence (full attention, linear attention, Mamba, MLP, or MoE),
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

from .capabilities import CAPABILITIES, Capability, normalize_parallel

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
    #: Sliding-window width (tokens visible to a query, including itself).  Set when the
    #: layer sequence contains ``sliding_attention`` layers; ``None`` means every attention
    #: layer sees the whole history.
    sliding_window: Optional[int] = None
    #: Per-layer-type RoPE parameters (LFM3 has two rotary dims: full and sliding layers).
    rope_by_layer: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    #: Layer indices with no rotary embedding (iRoPE: the global layers of MuseGlimmer are
    #: NoPE).  Empty for ordinary models.
    rope_disabled_layers: Tuple[int, ...] = ()

    def validate(self) -> None:
        if self.num_heads <= 0 or self.num_kv_heads <= 0 or self.head_dim <= 0:
            raise ValueError("attention head counts and head_dim must be positive")
        if self.num_heads % self.num_kv_heads:
            raise ValueError("num_heads must be a multiple of num_kv_heads (GQA grouping)")
        if self.sliding_window is not None and self.sliding_window < 1:
            raise ValueError("sliding_window must be a positive token count")
        for layer_type, params in self.rope_by_layer.items():
            if not isinstance(params, Mapping) or "rope_theta" not in params:
                raise ValueError(f"rope_by_layer[{layer_type!r}] needs a rope_theta")

    def as_dict(self) -> Dict[str, Any]:
        data = {
            "kind": self.kind,
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "output_gate": self.output_gate,
            "rope": self.rope,
            "partial_rotary_factor": self.partial_rotary_factor,
            "sliding_window": self.sliding_window,
            "rope_disabled_layers": list(self.rope_disabled_layers),
        }
        if self.sliding_window is not None:
            data["sliding_window"] = self.sliding_window
        if self.rope_by_layer:
            data["rope_by_layer"] = {
                key: dict(value) for key, value in self.rope_by_layer.items()
            }
        return data


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
class StateSpec:
    """Recurrent state carried by a non-attention layer (for example a short convolution)."""

    kind: str = "short_conv"
    dim: int = 2048
    kernel: int = 7
    bias: bool = False

    def validate(self) -> None:
        if self.kind not in {"short_conv"}:
            raise ValueError(f"unknown state kind {self.kind!r}")
        if self.dim <= 0 or self.kernel <= 0:
            raise ValueError("state dim and kernel must be positive")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "dim": self.dim,
            "kernel": self.kernel,
            "bias": self.bias,
        }


@dataclass(frozen=True)
class MambaSpec:
    """Mamba2 mixer parameters (not GDN; keep the two separate)."""

    kind: str = "mamba2"
    num_heads: int = 128
    head_dim: int = 64
    state_size: int = 128
    n_groups: int = 8
    conv_kernel: int = 4
    expand: int = 2
    state_dtype: str = "float32"

    def validate(self) -> None:
        if self.kind != "mamba2":
            raise ValueError(f"unsupported mamba kind {self.kind!r}")
        fields = (
            self.num_heads,
            self.head_dim,
            self.state_size,
            self.n_groups,
            self.conv_kernel,
            self.expand,
        )
        if min(fields) <= 0:
            raise ValueError("mamba head/state/group/conv/expand values must be positive")
        if self.num_heads % self.n_groups:
            raise ValueError("mamba num_heads must be a multiple of n_groups")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "state_size": self.state_size,
            "n_groups": self.n_groups,
            "conv_kernel": self.conv_kernel,
            "expand": self.expand,
            "state_dtype": self.state_dtype,
        }


@dataclass(frozen=True)
class MoESpec:
    """Routed-expert parameters."""

    num_experts: int = 256
    top_k: int = 8
    intermediate_size: int = 512
    shared_experts: int = 0
    shared_intermediate_size: int = 0

    def validate(self) -> None:
        if self.num_experts <= 0 or not (0 < self.top_k <= self.num_experts):
            raise ValueError("MoE needs num_experts > 0 and 0 < top_k <= self.num_experts")
        if self.shared_experts < 0 or self.shared_intermediate_size < 0:
            raise ValueError("MoE shared expert counts/sizes must be non-negative")
        if self.shared_experts and self.shared_intermediate_size <= 0:
            raise ValueError("MoE with shared experts needs shared_intermediate_size > 0")

    def as_dict(self) -> Dict[str, Any]:
        data = {
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "intermediate_size": self.intermediate_size,
            "shared_experts": self.shared_experts,
        }
        # Qwen's shared_experts=0 keeps the old byte-identical shape; Nemotron records the
        # 3712-wide shared MLP that the checkpoint actually contains.
        if self.shared_intermediate_size:
            data["shared_intermediate_size"] = self.shared_intermediate_size
        return data


SUPPORT_STATUSES = {"verified", "partial", "failed", "not_applicable", "not_implemented", "not_verified"}
_SUPPORT_KEYS = {
    "text",
    "vision",
    "audio",
    "mtp",
    "quant",
    "tp",
    "tp_note",
    "max_context_verified",
    "notes",
}


class SupportSpec(dict):
    """Validated verification scope copied into plans and the doctor report."""

    def __init__(self, value: Optional[Mapping[str, Any]] = None) -> None:
        data = dict(value or {})
        unknown = sorted(set(data) - _SUPPORT_KEYS)
        if unknown:
            raise ValueError(f"unknown support keys {unknown}; known: {sorted(_SUPPORT_KEYS)}")
        for key in ("text", "vision", "audio", "mtp"):
            if key in data:
                status = str(data[key])
                if status not in SUPPORT_STATUSES:
                    raise ValueError(
                        f"support[{key!r}]={status!r}; known: {sorted(SUPPORT_STATUSES)}"
                    )
                data[key] = status
        for key in ("quant", "tp"):
            if key in data:
                data[key] = list(data[key])
        if "max_context_verified" in data and data["max_context_verified"] is not None:
            data["max_context_verified"] = int(data["max_context_verified"])
        super().__init__(data)

    def as_dict(self) -> Dict[str, Any]:
        return dict(self)

    def validate(self) -> None:
        SupportSpec(self)


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
    state: Optional[StateSpec] = None
    mamba: Optional[MambaSpec] = None
    moe: Optional[MoESpec] = None
    quantization: str = "none"
    kv: Dict[str, Any] = field(default_factory=dict)
    required_capabilities: Tuple[str, ...] = ()
    benchmark: Dict[str, Any] = field(default_factory=dict)
    #: Dotted module path of the model implementation this repository supplies when the
    #: vendor has no architecture for it (``register_config`` / ``register_models`` actors).
    #: Empty means "the vendor already serves the model".
    model_impl: str = ""
    #: Parallel axes this model implementation has been validated under.  It is enforced
    #: only when ``model_impl`` is set (route O: the model class is ours).  A vendor-served
    #: manifest leaves sharding to the vendor, so the decision must not block its TP path.
    #: A route-O model that has only been validated single-card keeps ``("single",)`` and is
    #: refused under any sharding axis; ``("single", "tp")`` is a claim that must be backed
    #: by ``benchmark.tolerance_tp`` (cross-TP C2) and the verification matrix.
    parallel: Tuple[str, ...] = ("single",)
    #: Dotted module path + actor that normalises/registers a checkpoint config before
    #: ``ModelConfig`` is built.  Empty means the vendor config path is sufficient.  The
    #: actor defaults to ``register``; ``module:actor`` selects a different entry point.
    config_impl: str = ""
    #: CLI name and ``module:Class`` path of the reasoning/chat-format parser, when the
    #: checkpoint speaks a private channel protocol (``--reasoning-parser <name>``).
    reasoning_parser: str = ""
    reasoning_parser_impl: str = ""
    #: Machine-readable verification scope (text/vision/mtp/quant/tp/context), so a plan,
    #: doctor report, or evidence page cannot turn "C1 ran" into "fully supported".
    support: SupportSpec = field(default_factory=SupportSpec)
    #: Compat-hook names this declaration needs.  Kept out of the content-addressed manifest
    #: fingerprint (like ``support``) but copied into Plan.hooks for the deployment audit.
    hooks: Tuple[str, ...] = ()
    source: str = "docs/notes/op-coverage-qwen36.md"
    schema_version: int = MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.support, SupportSpec):
            object.__setattr__(self, "support", SupportSpec(self.support))

    def validate(self) -> None:
        if self.schema_version != MANIFEST_SCHEMA_VERSION:
            raise ValueError("unsupported manifest schema version")
        if not self.name or not self.manifest_id or not self.arch:
            raise ValueError("manifest needs name, manifest_id and arch")
        normalize_parallel(self.parallel)
        if len(self.layer_types) != self.num_layers:
            raise ValueError(
                f"layer_types has {len(self.layer_types)} entries for {self.num_layers} layers"
            )
        unknown = sorted(
            set(self.layer_types)
            - {
                "full_attention",
                "sliding_attention",
                "linear_attention",
                "conv",
                "mamba",
                "mlp",
                "moe",
            }
        )
        if unknown:
            raise ValueError(f"unknown layer types {unknown}")
        if self.linear is not None and "linear_attention" not in self.layer_types:
            raise ValueError("linear spec declared but no linear_attention layer")
        if self.state is not None and "conv" not in self.layer_types:
            raise ValueError("state spec declared but no conv layer")
        if "conv" in self.layer_types and self.state is None:
            raise ValueError("conv layer declared but no state spec")
        if self.mamba is not None and "mamba" not in self.layer_types:
            raise ValueError("mamba spec declared but no mamba layer")
        if "mamba" in self.layer_types and self.mamba is None:
            raise ValueError("mamba layer declared but no mamba spec")
        if (
            self.mamba is not None
            or self.linear is not None
            or self.state is not None
        ) and Capability.STATE_LINEAR.value not in self.required_capabilities:
            raise ValueError("recurrent mixer declared but state.linear is not required")
        if self.moe is not None and "moe" not in self.required_capabilities and (
            Capability.MOE_ROUTED.value not in self.required_capabilities
        ):
            raise ValueError("MoE spec declared but moe.routed is not required")
        self.attention.validate()
        has_sliding = "sliding_attention" in self.layer_types
        if has_sliding and self.attention.sliding_window is None:
            raise ValueError("sliding_attention layers declared but attention.sliding_window is unset")
        if not has_sliding and self.attention.sliding_window is not None:
            raise ValueError("attention.sliding_window set but no sliding_attention layer")
        if has_sliding and Capability.ATTENTION_SLIDING.value not in self.required_capabilities:
            raise ValueError(
                "sliding_attention layers require the attention.sliding capability"
            )
        if self.linear is not None:
            self.linear.validate()
        if self.state is not None:
            self.state.validate()
        if self.mamba is not None:
            self.mamba.validate()
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
        # A declaration may ship before its C2 sample is captured; it then cannot be used for
        # numeric acceptance, but it can still drive deployment (this is how a brand-new model
        # is onboarded).  A partially filled sample is an error, never a silent pass.
        self.support.validate()
        if self.benchmark:
            for key in ("prompt", "max_tokens", "top5"):
                if key not in self.benchmark:
                    raise ValueError(f"benchmark sample is missing {key!r}")

    @property
    def mixer_kinds(self) -> Tuple[str, ...]:
        """Recurrent mixer kinds a ``state.linear`` provider must cover.

        A GDN provider and a Mamba2 provider both speak ``state.linear``; the manifest's
        mixer kind is what keeps them from being mistaken for one another.
        """
        kinds = []
        if self.linear is not None:
            kinds.append(self.linear.kind)
        if self.mamba is not None:
            kinds.append(self.mamba.kind)
        if self.state is not None:
            kinds.append(self.state.kind)
        return tuple(kinds)

    @property
    def fingerprint(self) -> str:
        data = dict(self.as_dict())
        # Support is evidence metadata, not a deployment fact: recording that MTP failed
        # must not change a content-addressed plan id (and therefore the plan itself).
        data.pop("support", None)
        data.pop("hooks", None)
        payload = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
        import hashlib

        return hashlib.sha256(payload).hexdigest()[:16]

    def as_dict(self) -> Dict[str, Any]:
        # Keep the pre-Mamba shape byte-identical when a model has no Mamba layers:
        # plan ids are content-addressed and must not churn for Qwen3.6 just because the
        # schema learned a new optional field.
        data = {
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
            "model_impl": self.model_impl,
            "reasoning_parser": self.reasoning_parser,
            "reasoning_parser_impl": self.reasoning_parser_impl,
            "support": self.support.as_dict(),
            "hooks": list(self.hooks),
            "source": self.source,
        }
        if self.config_impl:
            data["config_impl"] = self.config_impl
        # Omit the default so a vendor-served / single-card declaration keeps its old
        # byte-identical shape (and plan id); a sharded claim is part of the model identity.
        if self.parallel != ("single",):
            data["parallel"] = list(self.parallel)
        if self.state is not None:
            data["state"] = self.state.as_dict()
        if self.mamba is not None:
            data["mamba"] = self.mamba.as_dict()
        return data

    def write(self, path: str) -> None:
        self.validate()
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.as_dict(), fh, ensure_ascii=False, indent=2)
            fh.write("\n")
