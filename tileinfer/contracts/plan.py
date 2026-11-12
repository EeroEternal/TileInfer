"""The plan file: the single artefact deployment hands to the bridge.

One decision, one file.  It records *which* implementation serves each capability, *why*,
which access points are allowed to be installed, and the version triangle it was computed
against.  The bridge refuses to inject when the live versions no longer match - the failure
mode of a version mismatch is wrong numbers, not a crash, so this must be a hard gate.

The format is JSON and is validated on both write and read.  Unknown keys are rejected on
read so a typo in a hand-edited plan cannot silently turn into "vendor default".
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional

SCHEMA_VERSION = 1

#: The only mechanisms allowed to be recorded as access points.  The budget is 8; the
#: project uses 6.  Keeping the closed list here means a plan cannot invent a seventh.
ACCESS_POINTS: Dict[str, str] = {
    "plugin_entry": "vllm.general_plugins entry point that calls the bridge installer",
    "impl_table": "implementation replacement table (operator/layer dispatch)",
    "model_table": "model replacement table (route O)",
    "attn_select": "attention backend selection (wrap the platform hook)",
    "method_wrap": "method wrapping for ordinary functions/methods",
    "graph_hook": "capture-time hook: bucket check before, fallback count after",
}

MAX_ACCESS_POINTS = 8


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class ModelRef:
    """The model a plan was computed for."""

    name: str
    manifest: str
    arch: str
    fingerprint: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "manifest": self.manifest,
            "arch": self.arch,
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelRef":
        return cls(
            name=str(data["name"]),
            manifest=str(data["manifest"]),
            arch=str(data["arch"]),
            fingerprint=str(data.get("fingerprint", "")),
        )


@dataclass(frozen=True)
class VersionTriangle:
    """The three versions a plan is valid for, plus the runtime version for diagnosis."""

    vllm: str
    vllm_ascend: str
    python: str
    torch: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "vllm": self.vllm,
            "vllm_ascend": self.vllm_ascend,
            "python": self.python,
            "torch": self.torch,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VersionTriangle":
        return cls(
            vllm=str(data["vllm"]),
            vllm_ascend=str(data["vllm_ascend"]),
            python=str(data["python"]),
            torch=str(data.get("torch", "")),
        )


@dataclass(frozen=True)
class Selection:
    """One row of the decision: capability -> provider."""

    capability: str
    provider: str
    impl: str
    reason: str
    source: str = "vendor"
    graphable: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return {
            "capability": self.capability,
            "provider": self.provider,
            "impl": self.impl,
            "reason": self.reason,
            "source": self.source,
            "graphable": self.graphable,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Selection":
        return cls(
            capability=str(data["capability"]),
            provider=str(data["provider"]),
            impl=str(data["impl"]),
            reason=str(data["reason"]),
            source=str(data.get("source", "vendor")),
            graphable=bool(data.get("graphable", True)),
        )


@dataclass
class Plan:
    """The content-addressed deployment decision."""

    model: ModelRef
    versions: VersionTriangle
    selections: List[Selection] = field(default_factory=list)
    access_points: List[str] = field(default_factory=list)
    instrumentation: Dict[str, Any] = field(default_factory=dict)
    fallback_policy: str = "forbid"
    plan_id: str = ""
    created_at: str = ""
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.plan_id:
            self.plan_id = self.compute_id()
        if not self.created_at:
            self.created_at = _utc_now()

    def compute_id(self) -> str:
        """Deterministic id: model + versions + selections, not a timestamp."""
        payload = {
            "schema_version": self.schema_version,
            "model": self.model.as_dict(),
            "versions": self.versions.as_dict(),
            "selections": [s.as_dict() for s in self.selections],
            "access_points": sorted(self.access_points),
            "fallback_policy": self.fallback_policy,
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"plan schema_version={self.schema_version} is not {SCHEMA_VERSION}"
            )
        if not self.model.name or not self.model.manifest:
            raise ValueError("plan model needs a name and a manifest id")
        for key in ("vllm", "vllm_ascend", "python"):
            if not getattr(self.versions, key):
                raise ValueError(f"plan version triangle is missing {key!r}")
        unknown = [ap for ap in self.access_points if ap not in ACCESS_POINTS]
        if unknown:
            raise ValueError(
                f"plan names unknown access points {unknown}; allowed: {sorted(ACCESS_POINTS)}"
            )
        if len(self.access_points) > MAX_ACCESS_POINTS:
            raise ValueError(
                f"plan uses {len(self.access_points)} access points, budget is {MAX_ACCESS_POINTS}"
            )
        if len(set(self.access_points)) != len(self.access_points):
            raise ValueError("plan lists an access point twice")
        seen = set()
        for selection in self.selections:
            if selection.capability in seen:
                raise ValueError(f"plan decides {selection.capability!r} twice")
            seen.add(selection.capability)
            if not selection.impl:
                raise ValueError(f"selection {selection.capability!r} has no implementation")
        if self.fallback_policy not in {"forbid", "count"}:
            raise ValueError("fallback_policy must be 'forbid' or 'count'")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "created_at": self.created_at,
            "model": self.model.as_dict(),
            "versions": self.versions.as_dict(),
            "selections": [s.as_dict() for s in self.selections],
            "access_points": list(self.access_points),
            "instrumentation": dict(self.instrumentation),
            "fallback_policy": self.fallback_policy,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Plan":
        known = {
            "schema_version",
            "plan_id",
            "created_at",
            "model",
            "versions",
            "selections",
            "access_points",
            "instrumentation",
            "fallback_policy",
        }
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"plan has unknown keys {unknown}")
        plan = cls(
            model=ModelRef.from_dict(data["model"]),
            versions=VersionTriangle.from_dict(data["versions"]),
            selections=[Selection.from_dict(s) for s in data.get("selections", [])],
            access_points=list(data.get("access_points", [])),
            instrumentation=dict(data.get("instrumentation", {})),
            fallback_policy=str(data.get("fallback_policy", "forbid")),
            plan_id=str(data.get("plan_id", "")),
            created_at=str(data.get("created_at", "")),
            schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
        )
        plan.validate()
        return plan

    def write(self, path: str) -> None:
        self.validate()
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.as_dict(), fh, ensure_ascii=False, indent=2, sort_keys=False)
            fh.write("\n")
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> "Plan":
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def selection(self, capability: str) -> Optional[Selection]:
        for item in self.selections:
            if item.capability == capability:
                return item
        return None

    def summary(self) -> str:
        counts: Dict[str, int] = {}
        for selection in self.selections:
            counts[selection.provider] = counts.get(selection.provider, 0) + 1
        providers = ", ".join(f"{name}={n}" for name, n in sorted(counts.items())) or "none"
        return (
            f"plan={self.plan_id} model={self.model.name} "
            f"selections={len(self.selections)} ({providers}) "
            f"access_points={len(self.access_points)} fallback={self.fallback_policy}"
        )
