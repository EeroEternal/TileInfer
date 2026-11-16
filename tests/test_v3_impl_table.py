from __future__ import annotations

import sys
import types
from types import SimpleNamespace

from tileinfer.bridge import impl_table


def _plan(*providers: str, access=("impl_table",)) -> SimpleNamespace:
    return SimpleNamespace(
        selections=[
            SimpleNamespace(provider=name, source="tileinfer", capability="state.linear")
            for name in providers
        ],
        access_points=list(access),
    )


def test_no_non_vendor_selection_is_a_noop() -> None:
    report = impl_table.install(SimpleNamespace(selections=[], access_points=[]))
    assert report.ok is True
    assert report.installed == []
    assert "no non-vendor selections" in report.skipped[0]


def test_non_vendor_selection_requires_impl_table_access_point() -> None:
    report = impl_table.install(_plan("tileinfer.mamba2_state", access=("plugin_entry",)))
    assert report.ok is False
    assert "impl_table" in report.failed["plan"]


def test_unservable_plan_refuses_non_vendor_provider(monkeypatch) -> None:
    monkeypatch.delenv("TILEINFER_ALLOW_UNSERVABLE", raising=False)
    plan = _plan("tileinfer.mamba2_state")
    plan.servable = False
    report = impl_table.install(plan)
    assert report.ok is False
    assert "servable=false" in report.failed["plan"]


def test_unknown_provider_is_refused_not_guessed() -> None:
    report = impl_table.install(_plan("tileinfer.not_a_provider"))
    assert report.ok is False
    assert "no v3 provider binder" in report.failed["tileinfer.not_a_provider"]


def test_mamba2_binder_writes_only_the_selected_targets(monkeypatch) -> None:
    # The binder imports vLLM modules; give it a fake module tree and pin that only the
    # selected provider's names are written.
    targets = {
        "vllm.model_executor.layers.mamba.ops.causal_conv1d": "old_conv",
        "vllm.model_executor.layers.mamba.ops.ssd_combined": "old_chunk",
        "vllm.model_executor.layers.mamba.mamba_mixer2": "old_chunk",
        "vllm.model_executor.layers.mamba.ops.ssu_dispatch": "old_ssu",
    }
    # mamba_mixer2 is one module for both names; create the hierarchy explicitly.
    modules: dict[str, types.ModuleType] = {}
    for name in (
        "vllm",
        "vllm.model_executor",
        "vllm.model_executor.layers",
        "vllm.model_executor.layers.mamba",
        "vllm.model_executor.layers.mamba.ops",
        "vllm.model_executor.layers.mamba.ops.causal_conv1d",
        "vllm.model_executor.layers.mamba.ops.ssd_combined",
        "vllm.model_executor.layers.mamba.ops.ssu_dispatch",
        "vllm.model_executor.layers.mamba.mamba_mixer2",
    ):
        module = types.ModuleType(name)
        module.__path__ = []
        modules[name] = module
    modules[
        "vllm.model_executor.layers.mamba.ops.causal_conv1d"
    ].causal_conv1d_fn = targets["vllm.model_executor.layers.mamba.ops.causal_conv1d"]
    modules["vllm.model_executor.layers.mamba.mamba_mixer2"].causal_conv1d_fn = targets[
        "vllm.model_executor.layers.mamba.ops.causal_conv1d"
    ]
    modules[
        "vllm.model_executor.layers.mamba.ops.causal_conv1d"
    ].causal_conv1d_update = targets["vllm.model_executor.layers.mamba.ops.causal_conv1d"]
    modules["vllm.model_executor.layers.mamba.mamba_mixer2"].causal_conv1d_update = targets[
        "vllm.model_executor.layers.mamba.ops.causal_conv1d"
    ]
    modules[
        "vllm.model_executor.layers.mamba.ops.ssd_combined"
    ].mamba_chunk_scan_combined_varlen = targets[
        "vllm.model_executor.layers.mamba.ops.ssd_combined"
    ]
    modules[
        "vllm.model_executor.layers.mamba.mamba_mixer2"
    ].mamba_chunk_scan_combined_varlen = targets[
        "vllm.model_executor.layers.mamba.ops.ssd_combined"
    ]
    modules["vllm.model_executor.layers.mamba.ops.ssu_dispatch"].selective_state_update = targets[
        "vllm.model_executor.layers.mamba.ops.ssu_dispatch"
    ]
    modules["vllm.model_executor.layers.mamba.mamba_mixer2"].selective_state_update = targets[
        "vllm.model_executor.layers.mamba.ops.ssu_dispatch"
    ]
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    report = impl_table.install(_plan("tileinfer.mamba2_state"))
    assert report.ok is True, report.failed
    assert len(report.installed) == 9  # 4 primary + 4 aliases + provider marker
    from tileinfer.providers.nemotron import mamba2

    assert (
        modules[
            "vllm.model_executor.layers.mamba.ops.causal_conv1d"
        ].causal_conv1d_fn is mamba2.causal_conv1d_fn
    )
    assert (
        modules[
            "vllm.model_executor.layers.mamba.ops.causal_conv1d"
        ].causal_conv1d_update is mamba2.causal_conv1d_update
    )
    assert (
        modules[
            "vllm.model_executor.layers.mamba.ops.ssd_combined"
        ].mamba_chunk_scan_combined_varlen is mamba2.mamba_chunk_scan_combined_varlen
    )
    assert (
        modules["vllm.model_executor.layers.mamba.mamba_mixer2"].selective_state_update
        is mamba2.selective_state_update
    )


def test_bridge_does_not_import_the_v2_operator_installer() -> None:
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "tileinfer/bridge/install.py").read_text(
        encoding="utf-8"
    )
    assert "tileinfer.vllm.ops.install" not in source
    assert "from tileinfer.vllm.ops" not in source


def test_v3_provider_sources_do_not_import_the_v2_tree() -> None:
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "tileinfer/providers"
    offenders = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(
                name == "tileinfer.vllm" or name.startswith("tileinfer.vllm.")
                for name in names
            ):
                offenders.append(str(path.relative_to(root)))
    assert offenders == []


def test_install_all_fails_closed_without_impl_table() -> None:
    from tileinfer.bridge import seams

    plan = SimpleNamespace(
        selections=[SimpleNamespace(provider="tileinfer.mamba2_state", source="tileinfer")],
        access_points=["plugin_entry"],
    )
    report = seams.install_all(plan)
    assert report["ok"] is False
    row = next(item for item in report["seams"] if item["name"] == "impl_table")
    assert row["ok"] is False
    assert "no impl_table access point" in row["detail"]
