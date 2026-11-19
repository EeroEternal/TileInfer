"""The host HCCL rootinfo preflight: explain the failure before a sharded run starts.

The 235.16 box had ``/etc/hccl_rootinfo.json`` as an empty directory; HCCL reads that path as
a rankTable file, so TP>1 died with ``RootInfoDetect failed`` after a multi-minute wait.
``doctor`` now names the cause.  These tests pin the pure classifier.
"""

from __future__ import annotations

import pathlib

from tileinfer.cli.doctor import hccl_rootinfo_problem


def test_missing_rootinfo_is_healthy(tmp_path: pathlib.Path) -> None:
    assert hccl_rootinfo_problem(str(tmp_path / "absent.json")) == ""


def test_directory_rootinfo_is_reported(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "hccl_rootinfo.json"
    target.mkdir()
    problem = hccl_rootinfo_problem(str(target))
    assert "directory" in problem
    assert "RootInfoDetect" in problem


def test_empty_file_is_reported(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "hccl_rootinfo.json"
    target.write_text("")
    assert "empty file" in hccl_rootinfo_problem(str(target))


def test_invalid_json_is_reported(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "hccl_rootinfo.json"
    target.write_text("{not json")
    assert "not valid JSON" in hccl_rootinfo_problem(str(target))


def test_valid_json_is_healthy(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "hccl_rootinfo.json"
    target.write_text('{"version": "1.0"}')
    assert hccl_rootinfo_problem(str(target)) == ""
