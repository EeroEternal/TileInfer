"""The gap report: what the declaration needs and the environment cannot supply.

This is the artefact that makes "cost of adding a model" predictable.  It is produced by the
same pure decision that produces the plan, from the same inputs, and it is deliberately
separate from the plan: a plan can exist with zero gaps (nothing to do), and a report can
exist with no plan at all (the model cannot be served here).

A required gap means ``servable = False``.  A non-required gap is recorded but does not stop
deployment - the capability has a vendor fallback that is known to work.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping


@dataclass(frozen=True)
class Gap:
    """One missing capability, with the candidates that could fill it."""

    capability: str
    required: bool
    reason: str
    candidates: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "capability": self.capability,
            "required": self.required,
            "reason": self.reason,
            "candidates": list(self.candidates),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Gap":
        return cls(
            capability=str(data["capability"]),
            required=bool(data["required"]),
            reason=str(data["reason"]),
            candidates=list(data.get("candidates", [])),
        )


@dataclass
class GapReport:
    """All gaps for one model on one machine."""

    model: str
    manifest_id: str
    servable: bool
    gaps: List[Gap] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "manifest_id": self.manifest_id,
            "servable": self.servable,
            "gaps": [g.as_dict() for g in self.gaps],
            "notes": list(self.notes),
        }

    def write(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.as_dict(), fh, ensure_ascii=False, indent=2)
            fh.write("\n")

    @classmethod
    def load(cls, path: str) -> "GapReport":
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return cls(
            model=str(data["model"]),
            manifest_id=str(data["manifest_id"]),
            servable=bool(data["servable"]),
            gaps=[Gap.from_dict(g) for g in data.get("gaps", [])],
            notes=list(data.get("notes", [])),
        )

    def summary(self) -> str:
        required = sum(1 for g in self.gaps if g.required)
        return (
            f"model={self.model} servable={self.servable} "
            f"gaps={len(self.gaps)} (required={required})"
        )
