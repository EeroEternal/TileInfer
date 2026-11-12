"""Unit tests for the v3 artefacts: declaration -> decision -> plan / gap report.

These run on a CPU laptop with no engine and no device: that is the property the layer split
is for, and it is asserted here by importing the modules in a subprocess with the vendor
package hidden.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest

from tileinfer.contracts.capabilities import CAPABILITIES, Capability, ProviderSpec
from tileinfer.contracts.gaps import Gap, GapReport
from tileinfer.contracts.manifest import ModelManifest
from tileinfer.contracts.plan import ACCESS_POINTS, Plan, Selection
from tileinfer.manifests import MANIFESTS, QWEN3_6, find_manifest
from tileinfer.resolve.decide import decide
from tileinfer.resolve.probe import Environment

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _vendor_env() -> Environment:
    return Environment(
        vllm="0.23.0",
        vllm_ascend="0.23.0",
        torch="2.10.0+cpu",
        python="3.12.14",
        device_type="A5",
        has_vendor=True,
    )


def test_eight_capabilities_and_their_order_is_stable() -> None:
    assert len(CAPABILITIES) == 8
    assert CAPABILITIES[0] is Capability.ATTENTION_FULL
    assert [c.value for c in CAPABILITIES] == [
        "attention.full",
        "attention.linear",
        "moe.routed",
        "norm.rms",
        "rotary",
        "sampling.topk",
        "state.linear",
        "quant.verify",
    ]


def test_qwen36_declaration_matches_the_checkpoint_facts() -> None:
    m = QWEN3_6
    m.validate()
    assert m.num_layers == 40
    assert m.layer_types.count("full_attention") == 10
    assert m.layer_types.count("linear_attention") == 30
    assert m.attention.head_dim == 256
    assert m.attention.num_heads == 16 and m.attention.num_kv_heads == 2
    assert m.linear is not None and m.linear.num_v_heads == 32
    assert m.moe is not None and m.moe.num_experts == 256 and m.moe.top_k == 8
    assert len(m.required_capabilities) == 8
    assert m.benchmark["first_token"] == "Here"


def test_registry_accepts_the_three_natural_keys_and_rejects_unknown() -> None:
    for key in (QWEN3_6.manifest_id, QWEN3_6.arch, QWEN3_6.name):
        assert find_manifest(key) is QWEN3_6
    assert set(MANIFESTS) >= {QWEN3_6.manifest_id, QWEN3_6.arch, QWEN3_6.name}
    with pytest.raises(KeyError):
        find_manifest("not-a-model")


def test_manifest_validation_rejects_a_layer_count_mismatch() -> None:
    payload = QWEN3_6.as_dict()
    payload["layer_types"] = payload["layer_types"][:-1]
    with pytest.raises(ValueError):
        ModelManifest(**{**payload, "layer_types": tuple(payload["layer_types"])}).validate()


def test_provider_needs_evidence_for_a_measured_ratio() -> None:
    spec = ProviderSpec(
        capability=Capability.MOE_ROUTED,
        name="x",
        impl="x.y",
        source="tileinfer",
        graphable=True,
        measured_ratio=1.2,
    )
    with pytest.raises(ValueError):
        spec.validate()
    spec2 = ProviderSpec(
        capability=Capability.MOE_ROUTED,
        name="x",
        impl="x.y",
        source="tileinfer",
        graphable=True,
        measured_ratio=1.2,
        evidence="bench/x",
    )
    spec2.validate()


def test_decision_with_vendor_selects_every_capability_and_reports_no_gap() -> None:
    plan, report = decide(QWEN3_6, _vendor_env(), policy="vendor_first")
    assert report.servable is True
    assert report.gaps == []
    assert len(plan.selections) == 8
    assert {s.capability for s in plan.selections} == set(QWEN3_6.required_capabilities)
    assert all(s.source == "vendor" for s in plan.selections)
    assert plan.access_points == ["plugin_entry", "attn_select", "method_wrap", "graph_hook"]
    assert plan.fallback_policy == "forbid"


def test_decision_without_vendor_is_not_servable_and_names_every_gap() -> None:
    env = Environment(vllm="0.23.0", vllm_ascend=None, python="3.12.14", has_vendor=False)
    plan, report = decide(QWEN3_6, env)
    assert report.servable is False
    assert len(report.gaps) == 8
    assert all(g.required for g in report.gaps)
    assert plan.selections == []


def test_policy_is_closed() -> None:
    with pytest.raises(ValueError):
        decide(QWEN3_6, _vendor_env(), policy="fastest")


def test_plan_round_trips_and_its_id_is_content_addressed() -> None:
    plan, _ = decide(QWEN3_6, _vendor_env())
    data = plan.as_dict()
    restored = Plan.from_dict(data)
    assert restored.as_dict() == data
    assert restored.plan_id == plan.plan_id
    # The id ignores the timestamp, so two runs in different seconds agree.
    plan2 = Plan(
        model=plan.model,
        versions=plan.versions,
        selections=plan.selections,
        access_points=plan.access_points,
        instrumentation=plan.instrumentation,
        fallback_policy=plan.fallback_policy,
    )
    assert plan2.plan_id == plan.plan_id


def test_plan_rejects_unknown_keys_unknown_access_points_and_duplicates() -> None:
    plan, _ = decide(QWEN3_6, _vendor_env())
    bad = plan.as_dict()
    bad["typo"] = 1
    with pytest.raises(ValueError, match="unknown keys"):
        Plan.from_dict(bad)

    bad2 = plan.as_dict()
    bad2["access_points"] = ["plugin_entry", "not_a_mechanism"]
    with pytest.raises(ValueError, match="unknown access points"):
        Plan.from_dict(bad2)

    bad3 = plan.as_dict()
    bad3["access_points"] = ["plugin_entry", "plugin_entry"]
    with pytest.raises(ValueError, match="twice"):
        Plan.from_dict(bad3)


def test_plan_rejects_over_budget_and_duplicate_selections() -> None:
    plan, _ = decide(QWEN3_6, _vendor_env())
    bad = plan.as_dict()
    bad["access_points"] = [
        "plugin_entry", "attn_select", "method_wrap", "graph_hook",
        "model_table", "impl_table", "plugin_entry", "attn_select", "method_wrap",
    ]
    with pytest.raises(ValueError, match="budget"):
        Plan.from_dict(bad)

    dup = plan.as_dict()
    dup["selections"] = dup["selections"] + [dup["selections"][0]]
    with pytest.raises(ValueError, match="twice"):
        Plan.from_dict(dup)


def test_gap_report_round_trips(tmp_path: pathlib.Path) -> None:
    report = GapReport(
        model="m",
        manifest_id="m",
        servable=False,
        gaps=[Gap("attention.full", True, "none", ["p"])],
        notes=["n"],
    )
    out = tmp_path / "gaps.json"
    report.write(str(out))
    assert GapReport.load(str(out)).as_dict() == report.as_dict()


def test_pyproject_registers_only_the_general_plugin() -> None:
    """The platform slot belongs to vllm-ascend; a second claimant breaks startup."""
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '[project.entry-points."vllm.platform_plugins"]' not in text
    assert 'tileinfer = "tileinfer.bridge.install:install"' in text
    assert 'tileinfer = "tileinfer.vllm.plugin:install"' not in text


def test_engine_free_layers_import_in_a_clean_subprocess(tmp_path: pathlib.Path) -> None:
    """No vLLM, no torch_npu, no vendor package: the decision path must still import."""
    code = (
        "import sys;\n"
        "import tileinfer.contracts, tileinfer.manifests, tileinfer.resolve, tileinfer.cli.plan;\n"
        "assert 'vllm' not in sys.modules;\n"
        "assert 'vllm_ascend' not in sys.modules;\n"
        "print('clean')\n"
    )
    env = {
        "PYTHONPATH": str(ROOT),
        "PATH": "/usr/bin:/bin",
        "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
    }
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "clean" in proc.stdout
