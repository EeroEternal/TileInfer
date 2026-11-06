"""Every module that reads ``os.environ`` must import ``os``.

This has bitten this project six times (``patches.py``, ``moe.py``, ``recurrent.py``, ``install.py``,
``tests/test_gdn_recurrent.py``, ``platform.py``), each time as a ``NameError`` **at runtime** — and
twice only on the device, where a round costs ten minutes.  A static check catches all of it in a
second, so it belongs in the suite rather than in a reviewer's memory.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path("tileinfer")


def _modules():
    return sorted(p for p in ROOT.rglob("*.py") if p.is_file())


def _uses_name(tree: ast.AST, name: str) -> bool:
    """Whether ``tree`` loads the bare name ``name`` (``os.environ`` -> ``os``, ``os.path`` -> ``os``)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Load):
            return True
        if isinstance(node, ast.Attribute):
            base = node
            while isinstance(base, ast.Attribute):
                base = base.value
            if isinstance(base, ast.Name) and base.id == name:
                return True
    return False


def _imports_name(tree: ast.AST, name: str) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == name or alias.asname == name:
                    return True
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == name or alias.asname == name:
                    return True
    return False


@pytest.mark.parametrize("path", _modules(), ids=lambda p: str(p))
def test_os_is_imported_where_it_is_used(path: Path):
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as exc:  # pragma: no cover - a broken file fails its own tests
        pytest.skip(f"{path} does not parse: {exc}")
    if not _uses_name(tree, "os"):
        return
    assert _imports_name(tree, "os"), (
        f"{path} uses `os` but never imports it; this has been a runtime NameError six times"
    )
