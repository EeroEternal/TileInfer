"""Ratchet: ``vllm_ascend`` is imported by **one** layer, and only that layer.

v2 banned the vendor package everywhere because it was a standalone platform.  v3 reverses
that premise: the platform, worker, memory, KV pool and graph capture are the vendor's, so
the bridge must be able to call into them.  What must not change is the *direction* of the
dependency: only ``tileinfer/bridge/`` may import upstream.  Contracts, resolution,
declarations, providers and kernels stay engine-free, which is what keeps them testable on a
CPU laptop and what makes "the vendor moved" a one-layer problem.

The check is still a ratchet: a new import outside ``bridge/`` fails, and a scan that finds
the string only in a comment is not counted (it parses imports, not text).
"""

from __future__ import annotations

import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]

SCAN_GLOBS = (
    "tileinfer/**/*.py",
    "tests/**/*.py",
    "scripts/**/*.py",
    "examples/**/*.py",
)

#: The only layer allowed to touch the vendor platform.  ``legacy/`` is not scanned at all.
ALLOWED_PREFIXES = ("tileinfer/bridge/",)

#: Kept for compatibility with the old test signature; empty because the carve-out is now a
#: directory rule, not a file allowlist.
ALLOWED: set[str] = set()


def _imported_modules(path: pathlib.Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
            # `from ...integrations import vllm_ascend` has the name, not the module
            modules.extend(alias.name for alias in node.names)
    return modules


def _scan() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for pattern in SCAN_GLOBS:
        for path in ROOT.glob(pattern):
            if not path.is_file():
                continue
            relative = path.relative_to(ROOT).as_posix()
            modules = [m for m in _imported_modules(path) if "vllm_ascend" in m]
            if modules:
                found[relative] = sorted(set(modules))
    return found


def _format(found: dict[str, list[str]]) -> str:
    return "; ".join(f"{path} -> {modules}" for path, modules in sorted(found.items()))


def test_only_the_bridge_imports_vllm_ascend() -> None:
    offenders = {
        path: modules
        for path, modules in _scan().items()
        if not any(path.startswith(prefix) for prefix in ALLOWED_PREFIXES)
    }
    assert not offenders, (
        "vllm_ascend imported outside tileinfer/bridge/: "
        + _format(offenders)
        + ". Move the upstream-facing code into the bridge; the decision/contract layers must "
        "stay engine-free (docs/architecture.md, L3/L5)."
    )


def test_the_bridge_actually_uses_the_seam() -> None:
    """The carve-out must not be theoretical: the bridge has to import the vendor package."""
    found = _scan()
    bridge_modules = [
        modules for path, modules in found.items() if path.startswith("tileinfer/bridge/")
    ]
    assert bridge_modules, (
        "the layered guard allows tileinfer/bridge/ to import vllm_ascend, but nothing there "
        "does - either the seam was removed or the rule is stale"
    )
