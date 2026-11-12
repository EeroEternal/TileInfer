"""Static guard for the layer split (L1/L2 must stay engine-independent).

The whole reason ``tileinfer.serving`` exists is that the batching/planning/prefill logic stays
testable on a CPU laptop, without vLLM, without ``torch_npu`` and without a device.  A single stray
``import vllm`` in that layer silently destroys that property - the module still imports on the
box, so nothing fails until someone tries to run the CPU suite - which is exactly the kind of
rot a test has to hold down.

The check is intentionally conservative:

* ``vllm`` / ``vllm_ascend`` are forbidden in the engine-free paths at **any** nesting level,
  including ``if TYPE_CHECKING:`` blocks (use a ``Protocol`` or a string annotation instead);
* ``torch_npu`` is allowed only *inside* a function, where it stays an optional capability probe
  (``tileinfer.utils.is_npu_available`` is the existing example).
"""

from __future__ import annotations

import ast
import pathlib
from typing import Iterable, List, Tuple

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "tileinfer"

#: Engine-free paths (relative to ``tileinfer/``).  ``vllm/`` is deliberately absent: it is the
#: one layer allowed to import vLLM.
ENGINE_FREE_PATHS: Tuple[str, ...] = (
    "serving",
    "attention",
    "contracts",
    "manifests",
    "resolve",
    "providers",
    "aot",
    "observ",
    "cli",
    "metadata.py",
    "plan.py",
    "utils.py",
    "testing",
)

FORBIDDEN_ANY_LEVEL = ("vllm", "vllm_ascend")
FORBIDDEN_AT_MODULE_LEVEL = ("torch_npu",)


def _iter_files() -> Iterable[pathlib.Path]:
    for entry in ENGINE_FREE_PATHS:
        path = PACKAGE / entry
        if path.is_file():
            yield path
        elif path.is_dir():
            yield from sorted(path.rglob("*.py"))


def _module_level_imports(tree: ast.Module) -> List[Tuple[str, int]]:
    """(module, lineno) for imports executed at import time (module scope only)."""
    found: List[Tuple[str, int]] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            found.extend((alias.name, node.lineno) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            # `from . import x` inside a package still names x; resolve nothing - names are
            # checked only for the absolute forms that matter here.
            if base:
                found.append((base, node.lineno))
            for alias in node.names:
                if alias.name:
                    found.append((f"{base}.{alias.name}" if base else alias.name, node.lineno))
    return found


def _all_imports(tree: ast.Module) -> List[Tuple[str, int]]:
    found: List[Tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((alias.name, node.lineno) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if base:
                found.append((base, node.lineno))
            for alias in node.names:
                if alias.name:
                    found.append((f"{base}.{alias.name}" if base else alias.name, node.lineno))
    return found


def _matches(module: str, forbidden: str) -> bool:
    return module == forbidden or module.startswith(forbidden + ".")


@pytest.mark.parametrize("path", list(_iter_files()), ids=lambda p: p.relative_to(ROOT).as_posix())
def test_engine_free_layers_never_import_the_engine(path: pathlib.Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offenders = [
        f"{module} (line {line})"
        for module, line in _all_imports(tree)
        if any(_matches(module, forbidden) for forbidden in FORBIDDEN_ANY_LEVEL)
    ]
    assert not offenders, (
        f"{path.relative_to(ROOT)} imports engine code: {offenders}. "
        "Move it into tileinfer/vllm/ or behind a Protocol."
    )


@pytest.mark.parametrize("path", list(_iter_files()), ids=lambda p: p.relative_to(ROOT).as_posix())
def test_engine_free_layers_keep_torch_npu_optional(path: pathlib.Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offenders = [
        f"{module} (line {line})"
        for module, line in _module_level_imports(tree)
        if any(_matches(module, forbidden) for forbidden in FORBIDDEN_AT_MODULE_LEVEL)
    ]
    assert not offenders, (
        f"{path.relative_to(ROOT)} imports torch_npu at module level: {offenders}. "
        "Import it inside the function that probes for a device."
    )


def test_serving_and_attention_do_not_reach_into_the_legacy_adapters() -> None:
    """The new layers must not depend on the old ``integrations`` patch modules."""
    offenders = []
    for entry in ("serving", "attention", "contracts", "manifests", "resolve", "providers", "aot", "observ", "cli"):
        for path in (PACKAGE / entry).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for module, line in _all_imports(tree):
                if "integrations" in module:
                    offenders.append(f"{path.relative_to(ROOT)}:{line} -> {module}")
    assert not offenders, f"new layers depend on the legacy adapters: {offenders}"
