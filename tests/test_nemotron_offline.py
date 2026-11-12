"""Hardware-free checks for the Nemotron M5 offline phase.

These tests only read the vendored vLLM/vllm-ascend source tree.  They must stay runnable
without torch_npu, without a card and without a server; if the vendored tree is absent they
skip rather than pretend the audit passed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PREFIX = ROOT / ".venv_v3/lib/python3.12/site-packages"


@pytest.mark.skipif(not PREFIX.is_dir(), reason="vendored v3 site-packages not present")
def test_nemotron_static_audit_is_read_only_and_reports_unknowns(tmp_path: Path) -> None:
    out = tmp_path / "nemotron.json"
    subprocess.run(
        [sys.executable, "tools/nemotron/static_audit.py", "--out", str(out)],
        cwd=ROOT,
        check=True,
    )
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["hardware_used"] is False
    assert data["static_only"] is True
    assert data["static_servable"] is False
    assert data["unknown"], "a static prior must list what it could not prove"
    checks = {c["id"]: c for c in data["checks"]}
    assert checks["vllm_model_registry"]["ok"] is True
    assert checks["vllm_mtp_registry"]["ok"] is True
    assert checks["ascend_spec_patch"]["ok"] is True
    # This is the gap, not a pass: the audit records it as False until proven otherwise.
    assert checks["ascend_model_registration"]["ok"] is False
