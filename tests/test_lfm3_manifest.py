"""LFM3 declaration, route-O plan row and short-conv provider math."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tileinfer.manifests import find_manifest
from tileinfer.providers.state.short_conv import ShortConvState
from tileinfer.resolve.decide import decide
from tileinfer.resolve.probe import Environment


def test_lfm3_manifest_and_route_o_plan() -> None:
    manifest = find_manifest("lfm3_1_7")
    manifest.validate()
    assert manifest.arch == "Lfm3ForCausalLM"
    assert manifest.layer_types.count("conv") == 13
    assert manifest.layer_types.count("full_attention") == 6
    assert manifest.layer_types.count("sliding_attention") == 5
    assert manifest.model_impl == "tileinfer.bridge.models.lfm3"

    plan, report = decide(
        manifest,
        Environment(vllm="0.23.0", vllm_ascend="0.23.0", python="3.12", has_vendor=True),
    )
    by_capability = {selection.capability: selection for selection in plan.selections}
    assert by_capability["state.linear"].provider == "tileinfer.short_conv"
    assert "model.arch" not in by_capability
    assert plan.model_impl == manifest.model_impl
    assert "model_table" in plan.access_points
    assert "impl_table" in plan.access_points
    assert report.servable


def test_qwen_gdn_state_stays_vendor_served() -> None:
    qwen = find_manifest("qwen3_6")
    plan, _ = decide(
        qwen,
        Environment(vllm="0.23.0", vllm_ascend="0.23.0", python="3.12", has_vendor=True),
    )
    state = next(s for s in plan.selections if s.capability == "state.linear")
    assert state.source == "vendor"


def test_short_conv_prefill_and_step_match_torch_conv() -> None:
    torch.manual_seed(0)
    tokens, dim, kernel = 5, 8, 7
    x = torch.randn(tokens, dim)
    weight = torch.randn(dim, 1, kernel)

    out, state = ShortConvState.prefill(x, weight)
    padded = F.pad(x.transpose(0, 1).unsqueeze(0), (kernel - 1, 0))
    reference = F.conv1d(padded, weight, groups=dim)[0].transpose(0, 1)
    assert torch.allclose(out, reference, atol=1e-5)
    expected_state = F.pad(x.transpose(0, 1), (kernel - 1, 0))[:, -(kernel - 1) :]
    assert torch.equal(state, expected_state)

    nxt = torch.randn(1, dim)
    step, state2 = ShortConvState.step(nxt, weight, state)
    padded2 = F.pad(
        torch.cat([x, nxt], dim=0).transpose(0, 1).unsqueeze(0), (kernel - 1, 0)
    )
    expected = F.conv1d(padded2, weight, groups=dim)[0].transpose(0, 1)[-1:]
    assert torch.allclose(step, expected, atol=1e-5)
    expected_state2 = torch.cat([state, nxt.transpose(0, 1)], dim=-1)[:, -(kernel - 1) :]
    assert torch.equal(state2, expected_state2)


def test_short_conv_batch_prefill_matches_serial() -> None:
    torch.manual_seed(1)
    dim, kernel = 8, 7
    x = torch.randn(11, dim)
    weight = torch.randn(dim, 1, kernel)
    starts = torch.tensor([0, 4])
    ends = torch.tensor([4, 11])
    state = torch.randn(2, dim, kernel - 1)

    batched, new_state = ShortConvState.prefill_batch(
        x, weight, state, starts, ends
    )
    for index in range(2):
        serial, serial_state = ShortConvState.prefill(
            x[starts[index] : ends[index]], weight, state[index]
        )
        assert torch.allclose(
            batched[starts[index] : ends[index]], serial, atol=1e-5
        )
        assert torch.allclose(new_state[index], serial_state, atol=1e-5)
