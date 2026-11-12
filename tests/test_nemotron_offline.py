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

from tileinfer.contracts.capabilities import Mamba2Contract
from tileinfer.contracts.manifest import AttentionSpec, MambaSpec, ModelManifest
from tileinfer.manifests import QWEN3_6

ROOT = Path(__file__).resolve().parents[1]
PREFIX = ROOT / ".venv_v3/lib/python3.12/site-packages"


def _mamba_manifest(*, layer_types: tuple[str, ...], mamba: MambaSpec | None) -> ModelManifest:
    return ModelManifest(
        name="nemotron-synthetic",
        manifest_id="nemotron-synthetic",
        arch="NemotronHForCausalLM",
        model_type="nemotron_h",
        num_layers=len(layer_types),
        layer_types=layer_types,
        attention=AttentionSpec(kind="gqa", num_heads=32, num_kv_heads=8, head_dim=128),
        mamba=mamba,
        required_capabilities=("attention.full", "state.linear"),
        benchmark={"prompt": "hi", "max_tokens": 1, "top5": []},
    )


class _Mamba2Stub:
    def forward(self, hidden_states, mup_vector=None):
        return hidden_states

    def state_shape(self, spec):
        return (1,)


def test_mamba2_provider_contract_is_structural() -> None:
    assert isinstance(_Mamba2Stub(), Mamba2Contract)


def test_qwen_manifest_shape_is_unchanged_by_the_optional_mamba_field() -> None:
    assert "mamba" not in QWEN3_6.as_dict()


def test_manifest_represents_mamba_and_mlp_layers() -> None:
    m = _mamba_manifest(
        layer_types=("mamba", "mlp", "full_attention", "mlp"),
        mamba=MambaSpec(num_heads=128, n_groups=8),
    )
    m.validate()
    assert m.as_dict()["mamba"]["kind"] == "mamba2"


def test_manifest_rejects_mamba_layer_without_spec() -> None:
    m = _mamba_manifest(layer_types=("mamba", "full_attention"), mamba=None)
    with pytest.raises(ValueError, match="no mamba spec"):
        m.validate()


def test_manifest_rejects_mamba_spec_without_layer() -> None:
    m = _mamba_manifest(
        layer_types=("full_attention", "mlp"),
        mamba=MambaSpec(num_heads=128, n_groups=8),
    )
    with pytest.raises(ValueError, match="no mamba layer"):
        m.validate()



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
