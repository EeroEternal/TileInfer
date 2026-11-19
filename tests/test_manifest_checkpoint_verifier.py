"""Run the generic manifest/checkpoint verifier on the local checkpoints when present."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tileinfer.manifests import find_manifest
from tileinfer.manifests.nemotron_3_5 import NEMOTRON_3_5_DRAFT
from tileinfer.manifests.verify import verify_manifest

CASES = [
    ("lfm3_1_7", "/home/models/LFM3-1.7B"),
    ("muse_glimmer", "/home/models/Muse-Glimmer-30B"),
]
NEMOTRON = ("nemotron", "/home/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16")


@pytest.mark.parametrize(("key", "path"), CASES)
def test_declaration_matches_the_checkpoint(key: str, path: str) -> None:
    config_path = Path(os.environ.get(f"{key.upper()}_MODEL_DIR", path)) / "config.json"
    if not config_path.exists():
        pytest.skip(f"checkpoint config not present: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = find_manifest(key)
    assert verify_manifest(manifest, config) == []


def test_nemotron_declaration_matches_the_checkpoint() -> None:
    config_path = Path(NEMOTRON[1]) / "config.json"
    if not config_path.exists():
        pytest.skip(f"checkpoint config not present: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert verify_manifest(NEMOTRON_3_5_DRAFT, config) == []
