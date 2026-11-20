"""One result type for every install action.

The bridge has several layers (config adapter, model actor, parser, provider binder,
compat hook, seam) but they all answer the same question: did this action succeed, what did
it install, what did it skip, and what failed?  Keeping that in one type stops a failed
actor from being reported as OK merely because it returned instead of raising.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional


@dataclass
class InstallResult:
    name: str
    ok: bool = True
    installed: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    failed: Dict[str, str] = field(default_factory=dict)
    detail: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        data = {
            "name": self.name,
            "ok": bool(self.ok),
            "installed": list(self.installed),
            "skipped": list(self.skipped),
            "failed": dict(self.failed),
            "detail": self.detail,
        }
        data.update(self.extra)
        return data

    def summary(self) -> str:
        return (
            f"{self.name}: {len(self.installed)} installed, "
            f"{len(self.failed)} failed, {len(self.skipped)} skipped"
        )

    # Compatibility with the dict-shaped results used before this type existed.
    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.as_dict().get(key, default)

    def __contains__(self, key: object) -> bool:
        return key in self.as_dict()

    @classmethod
    def from_mapping(cls, name: str, value: Any, detail: str = "") -> "InstallResult":
        """Normalise an actor's legacy dict/bool return into an InstallResult."""
        if isinstance(value, InstallResult):
            return value
        if isinstance(value, Mapping):
            failed = dict(value.get("failed") or {})
            installed = list(value.get("installed") or [])
            skipped = list(value.get("skipped") or [])
            ok = value.get("ok")
            if ok is None:
                for key in ("registered", "success", "installed_ok"):
                    if key in value:
                        ok = value[key] is not False
                        break
            if ok is None:
                ok = not failed
            return cls(
                name=name,
                ok=bool(ok) and not failed,
                installed=installed,
                skipped=skipped,
                failed=failed,
                detail=detail or str(value.get("detail", "")),
            )
        if value is False:
            return cls(name=name, ok=False, detail=detail or "actor returned False")
        return cls(name=name, ok=True, detail=detail)


__all__ = ["InstallResult"]
