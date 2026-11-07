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
