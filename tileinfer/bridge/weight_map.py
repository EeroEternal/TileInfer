"""Declarative checkpoint -> vLLM parameter mapping (bridge layer).

Every route-O model repeats the same four operations: rename a substring, merge a fused
parameter from several checkpoint tensors, skip prefixes the model does not use (vision,
MTP), and remap expert indices.  This module makes those operations data, so a model payload
declares a :class:`WeightMapSpec` and calls :func:`load_weights` instead of hand-writing the
same loop (and the same missing/shape bugs) again.

It lives in L5 because the actual loaders are vLLM ``Parameter`` attributes; the declaration
contains no engine objects.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

import torch

from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import is_pp_missing_parameter


@dataclass(frozen=True)
class StackedWeight:
    """One checkpoint tensor merged into a fused vLLM parameter."""

    target: str
    source: str
    shard_id: Any


@dataclass(frozen=True)
class WeightMapSpec:
    """Declarative name/mapping rules for one model's ``load_weights``."""

    renames: Tuple[Tuple[str, str], ...] = ()
    stacked: Tuple[StackedWeight, ...] = ()
    skip_prefixes: Tuple[str, ...] = ()
    skip_substrings: Tuple[str, ...] = ()
    expert_remap: Optional[Mapping[int, int]] = None
    #: Names whose parameters must stay FP32 (routers, mHC, ...).  The loader records them;
    #: model-specific post-processing can use ``fp32_parameter_names`` without a second scan.
    fp32_prefixes: Tuple[str, ...] = ()
    notes: str = ""

    def _skip(self, name: str) -> bool:
        return any(name.startswith(prefix) for prefix in self.skip_prefixes) or any(
            substring in name for substring in self.skip_substrings
        )

    def _remap_expert(self, name: str) -> str:
        if not self.expert_remap:
            return name
        mapping = self.expert_remap

        def replace(match: re.Match[str]) -> str:
            expert = int(match.group(1))
            return f"experts.{mapping.get(expert, expert)}."

        return re.sub(r"experts\.(\d+)\.", replace, name)

    def map_name(self, name: str) -> Optional[str]:
        """Apply skips/renames/expert mapping; ``None`` means "drop this tensor"."""
        if self._skip(name):
            return None
        mapped = name
        for old, new in self.renames:
            mapped = mapped.replace(old, new, 1)
        mapped = self._remap_expert(mapped)
        if self._skip(mapped):
            return None
        return mapped

    def fp32_parameter_names(self, module: torch.nn.Module) -> Tuple[str, ...]:
        return tuple(
            name
            for name, _ in module.named_parameters()
            if any(name.startswith(prefix) for prefix in self.fp32_prefixes)
        )


LFM3_WEIGHT_MAP = WeightMapSpec(
    renames=((".conv.", ".short_conv."),),
    stacked=(
        StackedWeight(".qkv_proj", ".q_proj", "q"),
        StackedWeight(".qkv_proj", ".k_proj", "k"),
        StackedWeight(".qkv_proj", ".v_proj", "v"),
        StackedWeight(".w13", ".w1", 0),
        StackedWeight(".w13", ".w3", 1),
    ),
    skip_prefixes=("lm_head.",),
)


def load_weights(
    module: torch.nn.Module,
    weights: Iterable[tuple[str, torch.Tensor]],
    spec: WeightMapSpec,
) -> set[str]:
    """Load ``weights`` into ``module`` using ``spec`` and return loaded parameter names."""
    params = dict(module.named_parameters())
    loaded: set[str] = set()
    for raw_name, tensor in weights:
        name = spec.map_name(raw_name)
        if name is None:
            continue
        for stacked in spec.stacked:
            if stacked.source + "." not in name:
                continue
            name = name.replace(stacked.source + ".", stacked.target + ".", 1)
            if is_pp_missing_parameter(name, module):
                break
            param = params.get(name)
            if param is None:
                raise KeyError(f"checkpoint tensor {raw_name!r} maps to unknown parameter {name!r}")
            weight_loader = getattr(param, "weight_loader", None)
            if weight_loader is None:
                raise ValueError(
                    f"parameter {name!r} needs a weight_loader for shard {stacked.shard_id!r}"
                )
            weight_loader(param, tensor, stacked.shard_id)
            loaded.add(name)
            break
        else:
            if is_pp_missing_parameter(name, module):
                continue
            param = params.get(name)
            if param is None:
                raise KeyError(f"checkpoint tensor {raw_name!r} maps to unknown parameter {name!r}")
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, tensor)
            loaded.add(name)
    return loaded


__all__ = ["StackedWeight", "WeightMapSpec", "LFM3_WEIGHT_MAP", "load_weights"]
