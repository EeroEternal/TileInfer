"""fused_sample 纯 CPU 契约:分块规划、边界、参数校验。

设备数值不在本文件(tilelang 编译需 NPU,见 `tests/test_fused_sample_device.py`)。
"""

from __future__ import annotations

import math

import pytest

from tileinfer.kernels.sampling import fused_sample as fs


# ---------------------------------------------------------------- 规划


def test_plan_real_vocab():
    """真实词表(V=248320)的分块必须能整除、且不超 UB 上限。"""
    p = fs.fused_sample_plan(248320, 8)
    assert 248320 % p["blk"] == 0
    assert p["blk"] <= 16384
    assert p["blk"] % 64 == 0
    assert p["n_chunks"] == 248320 // p["blk"]
    assert p["n_vec"] == p["blk"] // 64


@pytest.mark.parametrize("vocab", [16384, 32768, 65536, 131072, 248320])
def test_plan_divides_exactly(vocab):
    p = fs.fused_sample_plan(vocab, 1)
    assert vocab % p["blk"] == 0
    assert p["blk"] >= 64


def test_plan_num_cores_capped():
    """核数应为 min(batch, 40),不是 batch 本身。"""
    assert fs.fused_sample_plan(16384, 1)["num_cores"] == 1
    assert fs.fused_sample_plan(16384, 8)["num_cores"] == 8
    assert fs.fused_sample_plan(16384, 64)["num_cores"] == 40


def test_plan_rejects_unaligned_vocab():
    """词表必须是 64 的整数倍(内核按 64-lane 向量扫描)。"""
    with pytest.raises(ValueError, match="64"):
        fs.fused_sample_plan(1000, 1)
    with pytest.raises(ValueError, match="64"):
        fs.fused_sample_plan(32, 1)


def test_plan_rejects_bad_batch():
    with pytest.raises(ValueError, match="batch"):
        fs.fused_sample_plan(16384, 0)


def test_plan_small_ub_cap_falls_back_to_64():
    """UB 上限极小时退化为 64 分块,而不是报错。"""
    p = fs.fused_sample_plan(16384, 1, ub_cap=64)
    assert p["blk"] == 64
    assert p["n_chunks"] == 256


def test_block_size_prefers_largest_divisor():
    """应尽量选大块(减少扫描轮数),而非随便一个小块。"""
    assert fs._block_size(16384, 16384) == 16384
    # 248320 = 12416 × 20;12416 是从 16384 往下递减 64 找到的第一个因子
    assert fs._block_size(248320, 16384) == 12416
    assert 248320 % 12416 == 0


# ---------------------------------------------------------------- 常量


def test_constants_sanity():
    assert fs.MIN_VOCAB_FOR_FUSED_SAMPLE == 16384
    # 16 轮三分位二分 ≈ 4^-16 ≈ 2.3e-10,足以定位 float32 阈值
    assert fs.MAX_SEARCH_ROUNDS == 16
    assert 4.0 ** (-fs.MAX_SEARCH_ROUNDS) < 1e-9
    assert fs._VL == 64
    assert math.isclose(fs._MIN_PROB, 2.0**-32)


def test_real_vocab_meets_min():
    """本模型词表必须超过门槛,否则这个内核根本不会被启用。"""
    assert 248320 >= fs.MIN_VOCAB_FOR_FUSED_SAMPLE
