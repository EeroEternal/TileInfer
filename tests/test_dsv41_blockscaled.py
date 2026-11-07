"""Ascend MXFP4 block-scaled GEMM parity (NPU-only).

The kernel is the fused replacement for ``unpack_fp4_lut`` + broadcast scale + ``bmm``:
packed uint8 FP4 weights and e8m0 scales are consumed directly by the Cube unit.
"""

import pytest
import torch


def _probe():
    pytest.importorskip("torch_npu")
    pytest.importorskip("tilelang")
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    from tools.dsv41 import probe_blockscaled_gemm as probe

    return probe


def test_fp4_blockscaled_exact():
    probe = _probe()
    assert probe.check_fp4_fp4() < 1e-6


def test_npu_mx_quant_plus_blockscaled_exact():
    probe = _probe()
    assert probe.check_npu_quant_gemm(32, 2304, 5120) < 1e-6


def test_grouped_w2_matches_per_expert():
    probe = _probe()
    from tileinfer.models.dsv41.block import _blockscaled_linear, _blockscaled_linear_grouped

    E, N, K = 2, 64, 256
    torch.manual_seed(5)
    act = torch.randn(E, K, device="npu", dtype=torch.bfloat16)
    packed, scales = [], []
    for _ in range(E):
        p, s = probe._pack(torch.randn(N, K) * 0.1)
        packed.append(p.npu())
        scales.append(s.npu())
    w = torch.stack([p.view(torch.uint8) for p in packed]).view(torch.float4_e2m1fn_x2)
    sc = torch.stack([s.view(torch.uint8) for s in scales]).view(torch.float8_e8m0fnu)
    ref = torch.stack(
        [_blockscaled_linear(act[i : i + 1], w[i], sc[i], N, K)[0] for i in range(E)]
    )
    out = _blockscaled_linear_grouped(act, w, sc, N, K)
    assert (out - ref).abs().max().item() < 1e-5
