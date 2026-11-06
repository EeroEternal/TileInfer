"""The measurement channel: file-watched switches (``tileinfer.serving.flags``).

This module exists only to make a **paired** A/B possible inside one server process (§27: restarting
per arm has a 45% noise floor).  Two properties matter and are pinned here:

* when the watch is off, ``flag`` reads the environment and **touches no filesystem** — a production
  process must not pay for a measurement feature;
* when it is on, a file override wins over the environment, and clearing it restores the default.

The directory is redirected to ``tmp_path`` in every test so a leftover ``/tmp/tileinfer_flag`` on a
measurement box can never influence a test run.
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pytest

from tileinfer.serving import flags


@pytest.fixture()
def watched(monkeypatch):
    """A workspace-local flag dir (the sandbox forbids pytest's %TEMP% ``tmp_path``)."""
    root = Path(".work") / f"flags-test-{uuid.uuid4().hex[:8]}"
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("TILEINFER_FLAG_WATCH", "1")
    monkeypatch.setenv("TILEINFER_FLAG_DIR", str(root))
    monkeypatch.setattr(flags, "FLAG_DIR", str(root))
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_watch_is_off_by_default(monkeypatch):
    monkeypatch.delenv("TILEINFER_FLAG_WATCH", raising=False)
    assert flags.watch_enabled() is False
    monkeypatch.setenv("TILEINFER_FLAG_WATCH", "0")
    assert flags.watch_enabled() is False
    monkeypatch.setenv("TILEINFER_FLAG_WATCH", "yes")
    assert flags.watch_enabled() is True


def test_environment_alone_decides_when_not_watching(monkeypatch, watched):
    monkeypatch.delenv("TILEINFER_FLAG_WATCH", raising=False)
    # A stray override file must be ignored while the channel is closed.
    (watched / "thing=1").write_text("", encoding="utf-8")
    assert flags.flag("thing", "SOME_UNSET_FLAG", False) is False
    monkeypatch.setenv("SOME_UNSET_FLAG", "1")
    assert flags.flag("thing", "SOME_UNSET_FLAG", False) is True


def test_file_override_wins_and_can_be_cleared(watched):
    assert flags.flag("thing", "SOME_UNSET_FLAG", True) is True  # default
    flags.set_switch("thing", False)
    assert flags.flag("thing", "SOME_UNSET_FLAG", True) is False
    flags.set_switch("thing", True)
    assert flags.flag("thing", "SOME_UNSET_FLAG", False) is True
    flags.set_switch("thing", None)
    assert flags.flag("thing", "SOME_UNSET_FLAG", False) is False  # back to the default


def test_set_switch_replaces_rather_than_stacks(watched):
    flags.set_switch("thing", True)
    flags.set_switch("thing", False)
    present = sorted(p.name for p in watched.iterdir())
    assert present == ["thing=0"], "the old override must be removed, not left beside the new one"


def test_file_name_carries_the_value(watched):
    flags.set_switch("a", True)
    flags.set_switch("b", False)
    assert sorted(p.name for p in watched.iterdir()) == ["a=1", "b=0"]
