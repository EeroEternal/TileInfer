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

from tileinfer.contracts.capabilities import CAPABILITIES, Capability, Mamba2Contract, ProviderSpec
from tileinfer.contracts.manifest import AttentionSpec, MambaSpec, ModelManifest, MoESpec
from tileinfer.manifests import MANIFESTS, QWEN3_6
from tileinfer.manifests.nemotron_3_5 import BENCHMARK_PENDING, NEMOTRON_3_5_DRAFT
from tileinfer.resolve.decide import TILEINFER_PROVIDERS, decide
from tileinfer.resolve.probe import Environment

ROOT = Path(__file__).resolve().parents[1]
PREFIX = ROOT / ".venv_v3/lib/python3.12/site-packages"
VLLM_PY = ROOT / ".venv_v3/bin/python"


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


def test_manifest_represents_an_explicit_moe_layer() -> None:
    m = ModelManifest(
        name="nemotron-synthetic",
        manifest_id="nemotron-synthetic",
        arch="NemotronHForCausalLM",
        model_type="nemotron_h",
        num_layers=4,
        layer_types=("mamba", "moe", "full_attention", "mlp"),
        attention=AttentionSpec(kind="gqa", num_heads=32, num_kv_heads=8, head_dim=128),
        mamba=MambaSpec(num_heads=128, n_groups=8),
        moe=MoESpec(num_experts=64, top_k=4),
        required_capabilities=("attention.full", "state.linear", "moe.routed"),
        benchmark={"prompt": "hi", "max_tokens": 1, "top5": []},
    )
    m.validate()
    assert "moe" in m.layer_types


def test_moe_with_shared_expert_requires_a_shared_width() -> None:
    with pytest.raises(ValueError, match="shared_intermediate_size"):
        MoESpec(num_experts=8, top_k=1, shared_experts=1).validate()


def test_qwen_moe_does_not_gain_the_shared_width_key() -> None:
    assert "shared_intermediate_size" not in QWEN3_6.as_dict()["moe"]


def test_manifest_rejects_recurrent_mixer_without_state_linear() -> None:
    m = _mamba_manifest(
        layer_types=("mamba", "full_attention"),
        mamba=MambaSpec(num_heads=128, n_groups=8),
    )
    object.__setattr__(m, "required_capabilities", ("attention.full",))
    with pytest.raises(ValueError, match="state.linear is not required"):
        m.validate()


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
def _config_dict(pattern: str = "M-*E") -> dict:
    return {
        "architectures": ["NemotronHForCausalLM"],
        "model_type": "nemotron_h",
        "num_hidden_layers": len(pattern),
        "hybrid_override_pattern": pattern,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "mamba_num_heads": 128,
        "mamba_head_dim": 64,
        "ssm_state_size": 128,
        "mamba_n_groups": 8,
        "mamba_d_conv": 4,
        "mamba_expand": 2,
        "max_position_embeddings": 131072,
        "num_nextn_predict_layers": 1,
        "n_routed_experts": 64,
        "num_experts_per_tok": 4,
        "moe_intermediate_size": 512,
    }


def test_config_to_manifest_maps_layer_characters(tmp_path: Path) -> None:
    cfg = tmp_path / "config.json"
    out = tmp_path / "draft.json"
    cfg.write_text(json.dumps(_config_dict("M-*E")), encoding="utf-8")
    subprocess.run(
        [sys.executable, "tools/nemotron/config_to_manifest.py", "--config", str(cfg), "--out", str(out)],
        cwd=ROOT,
        check=True,
    )
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["layer_types"] == ["mamba", "mlp", "full_attention", "moe"]
    assert data["mamba"]["n_groups"] == 8
    assert data["moe"]["top_k"] == 4
    assert data["num_nextn_predict_layers"] == 1
    assert data["unresolved"]


def test_config_to_manifest_rejects_unknown_pattern_character() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "config_to_manifest", ROOT / "tools/nemotron/config_to_manifest.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    with pytest.raises(ValueError, match="unknown pattern"):
        mod.parse_config(_config_dict("M?"))


def test_config_to_manifest_rejects_bad_gqa_and_mamba_grouping() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "config_to_manifest", ROOT / "tools/nemotron/config_to_manifest.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    bad_gqa = _config_dict()
    bad_gqa["num_key_value_heads"] = 7
    with pytest.raises(ValueError, match="divisible"):
        mod.parse_config(bad_gqa)
    bad_mamba = _config_dict()
    bad_mamba["mamba_n_groups"] = 7
    with pytest.raises(ValueError, match="divisible"):
        mod.parse_config(bad_mamba)


def test_nemotron_declaration_matches_the_real_config_snapshot() -> None:
    m = NEMOTRON_3_5_DRAFT
    m.validate()
    assert m.num_layers == 52
    assert m.layer_types.count("mamba") == 23
    assert m.layer_types.count("moe") == 23
    assert m.layer_types.count("full_attention") == 6
    assert m.attention.num_heads == 32 and m.attention.num_kv_heads == 2
    assert m.mamba is not None and m.mamba.num_heads == 64 and m.mamba.n_groups == 8
    assert m.moe is not None and m.moe.num_experts == 128 and m.moe.top_k == 6
    assert m.kv["max_position_embeddings"] == 262144
    assert m.kv["num_nextn_predict_layers"] == 1
    assert "found" in m.kv["mtp_weights"]
    assert BENCHMARK_PENDING is True
    assert m.benchmark["top5"] == []
    assert m.manifest_id not in MANIFESTS


def test_mtp_probe_artifact_confirms_weights_are_present() -> None:
    data = json.loads((ROOT / "docs/notes/m5-nemotron-mtp.json").read_text(encoding="utf-8"))
    assert data["mtp_weights_found"] is True
    assert data["named_mtp_count"] == 270
    assert data["max_layer_index"] == 51


def test_config_normalizer_accepts_the_checkpoint_schema() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "config_to_manifest", ROOT / "tools/nemotron/config_to_manifest.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    raw = json.loads((ROOT / "docs/notes/m5-nemotron-config.json").read_text(encoding="utf-8"))
    norm = mod.normalize_config(raw)
    assert "layers_block_type" not in norm
    assert "mtp_layers_block_type" not in norm
    assert norm["hybrid_override_pattern"].count("*") == 6
    assert norm["hybrid_override_pattern"].count("M") == 23
    assert norm["hybrid_override_pattern"].count("E") == 23
    assert norm["mtp_hybrid_override_pattern"] == "*E"
    draft = mod.parse_config(raw)
    assert draft["num_layers"] == 52
    assert draft["max_position_embeddings"] == 262144
    assert draft["num_nextn_predict_layers"] == 1


@pytest.mark.skipif(not VLLM_PY.is_file(), reason="vendored v3 interpreter not present")
def test_normalized_config_loads_in_vllm_without_npu(tmp_path: Path) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "config_to_manifest", ROOT / "tools/nemotron/config_to_manifest.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    raw = json.loads((ROOT / "docs/notes/m5-nemotron-config.json").read_text(encoding="utf-8"))
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(mod.normalize_config(raw)), encoding="utf-8")
    code = (
        "import json,sys; from vllm.transformers_utils.configs.nemotron_h import NemotronHConfig; "
        f"c=NemotronHConfig(**json.load(open({str(cfg)!r}))); "
        "print(c.hybrid_override_pattern); print(c.mtp_hybrid_override_pattern)"
    )
    got = subprocess.run([str(VLLM_PY), "-c", code], cwd=ROOT, text=True, capture_output=True)
    assert got.returncode == 0, got.stderr
    pattern, mtp = got.stdout.splitlines()[-2:]
    assert pattern.count("*") == 6 and pattern.count("M") == 23 and pattern.count("E") == 23
    assert mtp == "*E"


def test_real_nemotron_draft_reports_its_seven_required_gaps() -> None:
    env = Environment(
        vllm="0.23.0", vllm_ascend=None, torch="2.10.0+cpu", device_type="A5", has_vendor=False
    )
    plan, report = decide(NEMOTRON_3_5_DRAFT, env, policy="vendor_first")
    assert report.servable is False
    assert len(report.gaps) == len(NEMOTRON_3_5_DRAFT.required_capabilities) == 7
    assert report.gaps[-1].required
    assert plan.selections == []


def _mtp_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("mtp_probe", ROOT / "tools/nemotron/mtp_probe.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_mtp_probe_detects_extra_layer_and_absent_mtp() -> None:
    mod = _mtp_module()
    base = {f"backbone.layers.{i}.mixer.in_proj.weight": "m.safetensors" for i in range(52)}
    assert mod.inspect_index({"weight_map": base}, 52)["mtp_weights_found"] is False
    base["backbone.layers.52.mixer.in_proj.weight"] = "m.safetensors"
    found = mod.inspect_index({"weight_map": base}, 52)
    assert found["mtp_weights_found"] is True
    assert found["max_layer_index"] == 52
    assert found["extra_layer_count"] == 1


def test_preflight_detects_incomplete_shards(tmp_path: Path) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("preflight_model", ROOT / "tools/nemotron/preflight_model.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "model-00001.safetensors"}}), encoding="utf-8"
    )
    first = mod.check(tmp_path)
    assert first["ready_to_load"] is False
    assert first["ready_for_dummy"] is True
    assert first["shards_missing"] == ["model-00001.safetensors"]
    (tmp_path / "model-00001.safetensors").write_bytes(b"x")
    second = mod.check(tmp_path)
    assert second["ready_to_load"] is True
    assert second["ready_for_dummy"] is True
    (tmp_path / "tokenizer.json").unlink()
    assert mod.check(tmp_path)["ready_for_dummy"] is False


def test_static_gap_snapshot_is_vendor_gap_only_for_state_linear() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "gap_snapshot", ROOT / "tools/nemotron/gap_snapshot.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    payload = mod.build()
    assert payload["hardware_used"] is False
    assert payload["weights_read"] is False
    present = payload["scenarios"]["vendor-present-static"]
    assert present["servable"] is False
    assert [g["capability"] for g in present["gap_report"]["gaps"] if g["required"]] == [
        "state.linear"
    ]
    absent = payload["scenarios"]["vendor-absent"]
    assert absent["servable"] is False
    assert len(absent["gap_report"]["gaps"]) == len(NEMOTRON_3_5_DRAFT.required_capabilities)


def test_make_shim_replaces_only_config(tmp_path: Path) -> None:
    model = tmp_path / "model"
    shim = tmp_path / "shim"
    model.mkdir()
    cfg = {
        "architectures": ["NemotronHForCausalLM"],
        "model_type": "nemotron_h",
        "num_hidden_layers": 2,
        "layers_block_type": ["mamba", "moe"],
        "num_attention_heads": 32,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "mamba_num_heads": 64,
        "mamba_head_dim": 64,
        "ssm_state_size": 128,
        "n_groups": 8,
        "conv_kernel": 4,
        "expand": 2,
        "max_position_embeddings": 262144,
        "num_nextn_predict_layers": 1,
        "mtp_layers_block_type": ["attention", "moe"],
        "n_routed_experts": 128,
        "num_experts_per_tok": 6,
        "moe_intermediate_size": 1856,
    }
    (model / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model / "model-1.safetensors").write_bytes(b"x")
    subprocess.run(
        [sys.executable, "tools/nemotron/make_shim.py", "--model", str(model), "--shim", str(shim)],
        cwd=ROOT,
        check=True,
    )
    normalized = json.loads((shim / "config.json").read_text(encoding="utf-8"))
    assert normalized["hybrid_override_pattern"] == "ME"
    assert normalized["mtp_hybrid_override_pattern"] == "*E"
    assert (shim / "tokenizer.json").is_symlink()
    assert (shim / "model-1.safetensors").is_symlink()
    assert json.loads((model / "config.json").read_text(encoding="utf-8"))["layers_block_type"] == [
        "mamba",
        "moe",
    ]


def test_vendor_gdn_state_does_not_cover_a_mamba2_declaration() -> None:
    env = Environment(
        vllm="0.23.0",
        vllm_ascend="0.23.0",
        python="3.12",
        torch="2.10.0",
        device_type="A5",
        has_vendor=True,
    )
    plan, report = decide(NEMOTRON_3_5_DRAFT, env, policy="vendor_first")
    assert not any(s.capability == Capability.STATE_LINEAR.value for s in plan.selections)
    assert any(g.capability == Capability.STATE_LINEAR.value for g in report.gaps)
    assert report.servable is False


def test_qwen_still_selects_the_vendor_gdn_state() -> None:
    env = Environment(
        vllm="0.23.0",
        vllm_ascend="0.23.0",
        python="3.12",
        torch="2.10.0",
        device_type="A5",
        has_vendor=True,
    )
    plan, report = decide(QWEN3_6, env, policy="vendor_first")
    selected = plan.selection(Capability.STATE_LINEAR.value)
    assert selected is not None
    assert selected.provider == "vllm_ascend.gdn_state"
    assert report.servable is True


def test_a_mamba2_provider_with_the_right_covers_can_fill_state_linear(monkeypatch) -> None:
    env = Environment(
        vllm="0.23.0",
        vllm_ascend="0.23.0",
        python="3.12",
        torch="2.10.0",
        device_type="A5",
        has_vendor=True,
    )
    provider = ProviderSpec(
        capability=Capability.STATE_LINEAR,
        name="tileinfer.mamba2_state",
        impl="tileinfer.providers.mamba2:Mamba2State",
        source="tileinfer",
        graphable=False,
        covers={"mixer": "mamba2"},
        evidence="unit-test-only",
        delete_when="vendor Mamba2 state lands",
    )
    monkeypatch.setitem(TILEINFER_PROVIDERS, Capability.STATE_LINEAR, provider)
    plan, report = decide(NEMOTRON_3_5_DRAFT, env, policy="vendor_first")
    selected = plan.selection(Capability.STATE_LINEAR.value)
    assert selected is not None
    assert selected.source == "tileinfer"
    assert report.servable is True


def test_no_vendor_decision_reports_all_gaps_without_hardware(tmp_path: Path) -> None:
    m = ModelManifest(
        name="nemotron-synthetic",
        manifest_id="nemotron-synthetic",
        arch="NemotronHForCausalLM",
        model_type="nemotron_h",
        num_layers=4,
        layer_types=("mamba", "mlp", "full_attention", "mlp"),
        attention=AttentionSpec(kind="gqa", num_heads=32, num_kv_heads=8, head_dim=128),
        mamba=MambaSpec(num_heads=128, n_groups=8),
        moe=MoESpec(num_experts=64, top_k=4),
        required_capabilities=tuple(c.value for c in CAPABILITIES),
        benchmark={"prompt": "hi", "max_tokens": 1, "top5": []},
    )
    m.validate()
    env = Environment(
        vllm="0.23.0", vllm_ascend=None, torch="2.10.0+cpu", device_type="A5", has_vendor=False
    )
    plan, report = decide(m, env, policy="vendor_first")
    assert report.servable is False
    assert len(report.gaps) == len(CAPABILITIES)
    assert all(g.required for g in report.gaps)
    assert plan.selections == []
    plan.validate()

    target = tmp_path / "gaps.json"
    report.write(str(target))
    from tileinfer.contracts.gaps import GapReport

    restored = GapReport.load(str(target))
    assert restored.as_dict() == report.as_dict()


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
    assert checks["ascend_hybrid_mamba_config"]["ok"] is True
    assert checks["ascend_causal_conv1d_patch"]["ok"] is True
    # These are gaps, not passes: recorded False until proven otherwise.
    assert checks["ascend_model_registration"]["ok"] is False
    assert checks["ascend_mamba2_mixer_patch"]["ok"] is False


def test_nemotron_h_config_alias_is_resolvable_by_autoconfig(tmp_path: Path) -> None:
    from tileinfer.vllm.config_compat import register_nemotron_h_alias

    info = register_nemotron_h_alias()
    if not info["registered"]:
        pytest.skip(f"vLLM/Transformers unavailable: {info['detail']}")

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "config_to_manifest_alias", ROOT / "tools/nemotron/config_to_manifest.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    raw = json.loads((ROOT / "docs/notes/m5-nemotron-config.json").read_text(encoding="utf-8"))
    norm = mod.normalize_config(raw)
    (tmp_path / "config.json").write_text(json.dumps(norm), encoding="utf-8")

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(str(tmp_path), local_files_only=True)
    assert type(config).__name__ == "NemotronHConfig"
    assert config.num_hidden_layers == 52
    assert config.model_type == "nemotron_h"


def test_raw_layers_block_type_is_normalised_before_vllm_config_init(tmp_path: Path) -> None:
    """Nemotron checkpoints export a list, vLLM exposes a read-only property derived from a string.

    This is the shape that made a raw 30B config fail with
    ``AttributeError: property 'layers_block_type' ... has no setter``; the adapter must convert it
    before the vendor class sees it.
    """
    from tileinfer.vllm.config_compat import register_nemotron_h_alias

    info = register_nemotron_h_alias()
    if not info["registered"]:
        pytest.skip(f"vLLM/Transformers unavailable: {info['detail']}")

    raw = json.loads((ROOT / "docs/notes/m5-nemotron-config.json").read_text(encoding="utf-8"))
    raw["layers_block_type"] = ["mamba", "moe", "mamba", "attention"] * 13
    raw["num_hidden_layers"] = len(raw["layers_block_type"])
    raw["mtp_layers_block_type"] = ["attention", "moe"]
    raw.pop("hybrid_override_pattern", None)
    (tmp_path / "config.json").write_text(json.dumps(raw), encoding="utf-8")

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(str(tmp_path), local_files_only=True)
    assert type(config).__name__ == "NemotronHConfig"
    assert config.hybrid_override_pattern == "MEM*" * 13
    assert config.mtp_hybrid_override_pattern == "*E"
    assert config.layers_block_type[:4] == ["mamba", "moe", "mamba", "attention"]
