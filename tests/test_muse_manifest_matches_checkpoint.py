"""Pin the MuseGlimmer declaration to the real checkpoint config when it is present.

The manifest is the only place the model's facts live (docs/architecture.md), but the
checkpoint's ``config.json`` is what the model class actually reads at load time.  If the two
drift, deployment still starts and silently uses different layer geometry.  This test skips
on machines without the weights; the device CI job runs it where they exist.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tileinfer.manifests.muse_glimmer import (
    FULL_ATTENTION_LAYERS,
    LAYER_TYPES,
    MUSE_GLIMMER,
    SLIDING_LAYERS,
)

MODEL_DIR = Path(os.environ.get("MUSE_MODEL_DIR", "/home/models/Muse-Glimmer-30B"))
CONFIG = MODEL_DIR / "config.json"

pytestmark = pytest.mark.skipif(
    not CONFIG.exists(), reason=f"checkpoint config not present: {CONFIG}"
)


def _text_config() -> dict:
    raw = json.loads(CONFIG.read_text())
    return raw.get("text_config", raw)


def test_layer_geometry_matches_the_checkpoint() -> None:
    cfg = _text_config()
    assert MUSE_GLIMMER.num_layers == cfg["num_hidden_layers"] == 52
    assert len(MUSE_GLIMMER.layer_types) == cfg["num_hidden_layers"]
    assert tuple(MUSE_GLIMMER.layer_types) == tuple(cfg["layer_types"])
    assert MUSE_GLIMMER.attention.sliding_window == cfg["sliding_window"] == 2048
    assert cfg["num_attention_heads"] == 32
    assert cfg["num_key_value_heads"] == 2
    assert cfg["head_dim"] == 128


def test_full_and_sliding_counts_match_the_declaration() -> None:
    cfg = _text_config()
    full = tuple(i for i, t in enumerate(cfg["layer_types"]) if t == "full_attention")
    sliding = tuple(i for i, t in enumerate(cfg["layer_types"]) if t == "sliding_attention")
    assert full == tuple(range(3, 52, 4)) == FULL_ATTENTION_LAYERS
    assert sliding == tuple(i for i in range(52) if i not in full) == SLIDING_LAYERS
    assert len(full) == 13 and len(sliding) == 39
    assert LAYER_TYPES == tuple(cfg["layer_types"])


def test_no_rope_layers_match_the_declaration() -> None:
    cfg = _text_config()
    theta = cfg.get("layer_rope_theta")
    if not isinstance(theta, list):
        pytest.skip("checkpoint does not expose per-layer rope theta")
    # The checkpoint encodes NoPE layers as rope theta == 0 (iRoPE).
    checkpoint = tuple(i for i, value in enumerate(theta) if not value)
    assert tuple(MUSE_GLIMMER.attention.rope_disabled_layers) == checkpoint
    assert checkpoint == FULL_ATTENTION_LAYERS
    assert all(theta[i] for i in SLIDING_LAYERS)
