"""The op audit's bookkeeping: what counts as vendor, what counts as a silent fallback.

Engine-free by design - the registry knows nothing about vLLM, and the bridge is what fills it.
"""

from __future__ import annotations

import json

from tileinfer.observ import opaudit


def _record(name: str, *, module: str, forward_module: str, base: str = "custom_op") -> None:
    opaudit.record(
        name,
        base=base,
        module=module,
        forward="forward_oot" if base == "custom_op" else "forward",
        forward_module=forward_module,
    )


def test_counts_instances_per_class() -> None:
    opaudit.reset()
    _record("RMSNorm", module="vllm_ascend.ops.layernorm", forward_module="vllm_ascend.ops.layernorm")
    _record("RMSNorm", module="vllm_ascend.ops.layernorm", forward_module="vllm_ascend.ops.layernorm")
    _record("GELU", module="vllm.model_executor.layers.activation", forward_module="vllm.model_executor.custom_op")

    snap = opaudit.snapshot()
    by_name = {row["name"]: row for row in snap["ops"]}
    assert by_name["RMSNorm"]["count"] == 2
    assert by_name["RMSNorm"]["impl"] == "vendor"
    assert by_name["GELU"]["impl"] == "torch"
    assert snap["totals"]["instances"] == 3
    assert snap["totals"]["vendor"] == 2
    assert snap["totals"]["torch"] == 1


def test_vendor_class_that_falls_through_to_torch_is_flagged() -> None:
    """The vendor registered the class name but the forward that runs is still the default."""
    opaudit.reset()
    _record(
        "AscendThing",
        module="vllm_ascend.ops.thing",
        forward_module="vllm.model_executor.custom_op",
    )
    row = opaudit.snapshot()["ops"][0]
    assert row["impl"] == "torch"
    assert row["suspicious"] is True


def test_pluggable_layer_default_is_not_called_torch() -> None:
    opaudit.reset()
    _record("FusedMoE", module="vllm.model_executor.layers.fused_moe.layer", forward_module="vllm.model_executor.layers.fused_moe.layer", base="pluggable")
    row = opaudit.snapshot()["ops"][0]
    assert row["impl"] == "vllm"
    assert row["suspicious"] is False


def test_dump_skips_a_process_that_built_nothing(tmp_path) -> None:
    opaudit.reset()
    target = tmp_path / "ops.json"
    assert opaudit.dump(str(target)) is None
    assert not target.exists()


def test_dump_writes_json_and_is_reloadable(tmp_path) -> None:
    opaudit.reset()
    _record("SiluAndMul", module="vllm_ascend.ops.activation", forward_module="vllm_ascend.ops.activation")
    target = tmp_path / "ops.json"
    assert opaudit.dump(str(target)) == str(target)
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["ops"][0]["name"] == "SiluAndMul"
    assert data["totals"]["vendor"] == 1


def test_audit_is_off_unless_the_env_asks_for_it(monkeypatch) -> None:
    """The audit is a diagnostic: an ordinary deployment must not pay for it."""
    from tileinfer.bridge.install import AUDIT_DEFAULT_PATH, _audit_path

    monkeypatch.delenv("TILEINFER_AUDIT", raising=False)
    assert _audit_path() is None
    monkeypatch.setenv("TILEINFER_AUDIT", "1")
    assert _audit_path() == AUDIT_DEFAULT_PATH
    monkeypatch.setenv("TILEINFER_AUDIT", "/tmp/ops.json")
    assert _audit_path() == "/tmp/ops.json"
