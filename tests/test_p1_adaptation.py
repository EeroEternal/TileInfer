"""P1 shared-tool contracts: state layout, weight maps, compat-hook registry."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from tileinfer.bridge.compat_hooks import HOOKS, install_compat_hook, list_compat_hooks
from tileinfer.bridge.weight_map import StackedWeight, WeightMapSpec, load_weights
from tileinfer.models.state_layout import short_conv_state_layout, state_layout_fits_page


def test_short_conv_state_layout_matches_the_ascend_page_geometry() -> None:
    layout = short_conv_state_layout(
        dim=2048, kernel=7, head_dim=128, num_kv_heads=4, dtype_size=2
    )
    assert layout.state_len == 6
    assert layout.target_page_bytes == 131072
    assert layout.padded_len == 32
    assert layout.shape(dim_first=True) == (2048, 32)
    assert layout.shape(dim_first=False) == (32, 2048)
    assert state_layout_fits_page(layout) is True


def test_weight_map_renames_skips_and_remaps_experts() -> None:
    spec = WeightMapSpec(
        renames=((".conv.", ".short_conv."),),
        skip_prefixes=("lm_head.",),
        skip_substrings=("rotary_emb.inv_freq",),
        expert_remap={3: 7},
    )
    assert spec.map_name("model.layers.0.conv.weight") == "model.layers.0.short_conv.weight"
    assert spec.map_name("lm_head.weight") is None
    assert spec.map_name("model.rotary_emb.inv_freq") is None
    assert spec.map_name("model.experts.3.up_proj") == "model.experts.7.up_proj"


def test_weight_map_reports_fp32_parameter_prefixes() -> None:
    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.router = nn.Module()
            self.router.weight = nn.Parameter(torch.zeros(2))
            self.proj = nn.Module()
            self.proj.weight = nn.Parameter(torch.zeros(2))

    spec = WeightMapSpec(fp32_prefixes=("router.",))
    assert spec.fp32_parameter_names(Model()) == ("router.weight",)


def test_weight_map_loads_a_fused_parameter_through_weight_loader() -> None:
    pytest.importorskip("vllm")
    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.qkv_proj = nn.Module()
            self.qkv_proj.weight = nn.Parameter(torch.zeros(6))

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer = Layer()

    model = Model()
    calls: list[tuple[object, int]] = []

    def loader(param, tensor, shard_id):
        calls.append((param, shard_id))

    model.layer.qkv_proj.weight.weight_loader = loader
    spec = WeightMapSpec(
        stacked=(StackedWeight(".qkv_proj", ".q_proj", "q"),),
    )
    loaded = load_weights(
        model,
        [("layer.q_proj.weight", torch.ones(2))],
        spec,
    )
    assert loaded == {"layer.qkv_proj.weight"}
    assert len(calls) == 1
    assert calls[0][1] == "q"


def test_compat_hooks_are_declared_with_owner_and_delete_condition() -> None:
    names = {hook["name"] for hook in list_compat_hooks()}
    assert {"lfm3_rope_global_buffer", "nemotron_non_gated_guard"} <= names
    for hook in HOOKS.values():
        assert hook.delete_when
        assert hook.evidence


def test_compat_hook_disable_switch_is_honoured(monkeypatch) -> None:
    monkeypatch.setenv("TILEINFER_DISABLE_LFM3_ROPE_PATCH", "1")
    result = install_compat_hook("lfm3_rope_global_buffer")
    assert result["ok"] is True
    assert result["disabled"] is True


def test_weight_map_module_imports_without_vllm() -> None:
    import subprocess
    import sys

    code = (
        "import sys; sys.modules['vllm'] = None; "
        "import tileinfer.bridge.weight_map; print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
