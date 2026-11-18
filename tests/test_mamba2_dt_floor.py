import torch

from tileinfer.providers.nemotron.mamba2 import _softplus_clamp


def test_decode_dt_respects_time_step_min():
    # HF clamps softplus(dt + dt_bias) to time_step_min (1e-3 for NemotronH).
    # A very negative dt would otherwise produce a near-zero step and drift.
    dt = torch.full((4,), -20.0)
    bias = torch.zeros(4)
    out = _softplus_clamp(dt, bias, True)
    assert torch.all(out >= 1e-3)
