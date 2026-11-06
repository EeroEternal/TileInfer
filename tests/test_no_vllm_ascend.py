"""Ratchet: nothing outside ``legacy/`` may reference ``vllm_ascend``.

The migration target used to be "no new dependencies" and the guard pinned an exact allowlist of
modules that were still allowed to import the vendor package.  That list is gone: the legacy adapter,
the opt-in patches built on its internals, and the tests that exercised them were moved into
``legacy/`` (a frozen snapshot - see ``legacy/README.md``).  The contract is now absolute: the live
tree contains **zero** references.

``SCAN_GLOBS`` covers the live tree only; ``legacy/`` is the carve-out the goal names, and nothing in
it is packaged or tested.  A new file that imports ``vllm_ascend`` fails this test, and so does
re-exporting it by name (``from ... import vllm_ascend``), because the scan looks at import targets
as well as module names.
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

#: Empty on purpose - the ratchet has closed.  The name stays so a regression is a one-line diff
#: instead of a rewrite, and so the intent of the check is still readable.
ALLOWED: set[str] = set()

#: The new backend layers must never reference it either, not even transiently.
CLEAN_PATHS = ("tileinfer/vllm", "tileinfer/serving")


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


def test_no_vllm_ascend_dependencies() -> None:
    found = _scan()
    assert found == {}, (
        "vllm_ascend referenced outside legacy/: "
        + _format(found)
        + ". Move the module into legacy/ (keeping its repo-relative path) - the live tree must not "
        "depend on the vendor package at all."
    )


def test_backend_layers_are_free_of_the_legacy_adapter() -> None:
    offenders = {
        path: modules
        for path, modules in _scan().items()
        if any(path.startswith(prefix + "/") for prefix in CLEAN_PATHS)
    }
    assert not offenders, f"the new layers must not reference vllm_ascend: {offenders}"
