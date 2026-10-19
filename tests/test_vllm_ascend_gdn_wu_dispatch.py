"""GDN chunk-wu 长度自适应分派设备测试(阶段 3 P1/H11)。

用例:
* ``ops_enabled`` 接受 ``gdnwu`` 开关(CPU 可跑,防 _OPS_ALL 回归);
* 分派语义:短上下文(T≤阈值)走融合核(gdn_wu_calls 递增)、长上下文回退原 triton 链
  (gdn_wu_long_skips 递增且 calls 不变);
* 链级数值 parity:同一输入,强制原链(阈值=0)与分派融合(缺省阈值)两条路整链输出
  (o / final_state)对齐,TOL 3e-2(与现有内核一致);
* ``TILEINFER_GDNWU_MAX_TOKENS`` 阈值覆盖生效。

运行::

    source .agent_loop/run/env.sh && .venv/bin/python -m pytest tests/test_vllm_ascend_gdn_wu_dispatch.py -v
"""

from __future__ import annotations

import pytest
import torch

from tileinfer.utils import is_npu_available

HK, HV, D = 16, 32, 128  # Qwen3.6 GDN 在役形状
TOL_RTOL, TOL_ATOL = 3e-2, 3e-2

try:  # pragma: no cover
    import vllm_ascend.ops.gdn  # noqa: F401  先拉起完整链(device_op 反向依赖 ops)

    _HAVE_VLLM_ASCEND = True
except Exception:  # pragma: no cover
    _HAVE_VLLM_ASCEND = False


def _init_once():
    """独立测试进程的初始化链(E22 探针同款);重复调用安全(全量套件会话内只生效一次)。"""
    import os

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29637")
    torch.npu.set_device(0)
    from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

    init_device_properties_triton()
    # 链尾 AscendC 算子(torch.ops._C_ascend.*)注册(E20 坑位备忘:独立进程需显式
    # bootstrap + import;libcust 预加载由 conftest 自举完成)
    from vllm_ascend.utils import bootstrap_custom_op_env

    bootstrap_custom_op_env()
    import vllm_ascend.vllm_ascend_C  # noqa: F401
    import conftest as _cf  # pytest 以 rootdir 方式导入,同名模块即 tests/conftest.py

    _cf.mark_vllm_ascend_C_loaded()
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm.distributed import parallel_state as _ps

    # D141 后 wiring 类测试经 install 链提前拉起 worker patch 家族,distributed
    # 全局态可能已部分建立(world/TP/PCP 各步独立探测补齐,不再整体 try/except
    # —— 整体吞错会把 PCP 缺失掩盖成后续 get_pcp_group 断言崩)。
    with set_current_vllm_config(VllmConfig()):
        try:
            if getattr(_ps, "_WORLD", None) is None:
                init_distributed_environment(world_size=1, rank=0, local_rank=0,
                                             distributed_init_method="env://")
        except Exception:  # noqa: BLE001
            pass
        try:
            if getattr(_ps, "_TP", None) is None:
                initialize_model_parallel(tensor_model_parallel_size=1)
        except Exception:  # noqa: BLE001
            pass
        try:
            if getattr(_ps, "_PCP", None) is None and getattr(_ps, "_WORLD", None) is not None:
                _ps._PCP = _ps.init_model_parallel_group(
                    [[0]], _ps.get_world_group().local_rank, None, group_name="pcp"
                )
        except Exception:  # noqa: BLE001
            pass


@pytest.fixture(autouse=True)
def _vllm_cfg_ctx():
    """测试体统一处于默认 config 上下文。

    iter-204(D141)后 install 链在 wiring 类测试里提前拉起 worker patch 家族,
    部分 CustomOp 转为构造期读 config(vllm/config/vllm.py:2224 断言);无上下文
    的测试体随即报 "Current vLLM config is not set"。此 fixture 使本模块测试体
    (不含 _init_once 自身,其自带 with 块)恒有 config。
    """
    from vllm.config import VllmConfig, set_current_vllm_config

    with set_current_vllm_config(VllmConfig()):
        yield


def _make(t_total, seed=0, cu_dtype=torch.int64):
    import torch.nn.functional as F

    g = torch.Generator(device="cpu").manual_seed(seed)
    k = torch.randn(1, t_total, HK, D, generator=g).to(torch.bfloat16)
    k = torch.nn.functional.normalize(k.float(), dim=-1).to(torch.bfloat16).npu()
    q = torch.randn(1, t_total, HK, D, generator=g).to(torch.bfloat16).npu()
    v = (torch.randn(1, t_total, HV, D, generator=g) * 0.2).to(torch.bfloat16).npu()
    beta = torch.rand(1, t_total, HV, generator=g).to(torch.bfloat16).npu()
    gating = F.logsigmoid(torch.rand(1, t_total, HV, generator=g)).float().npu()
    cu = torch.tensor([0, t_total], dtype=cu_dtype).npu()
    h0 = torch.randn(1, HV, D, D, generator=g).float().npu()
    return q, k, v, beta, gating, cu, h0


def _call_chain(t_total, seed=0, cu_dtype=torch.int64):
    """经装好分派的 chunk 模块函数整链调用(与在役同参,varlen 分支)。

    原链读取 ``get_forward_context()``,进程内无 worker 时用官方 context manager 置空上下文。
    """
    import vllm_ascend.ops.triton.fla.chunk as chunk_mod
    from vllm.config import VllmConfig
    from vllm.forward_context import set_forward_context

    q, k, v, beta, gating, cu, h0 = _make(t_total, seed, cu_dtype)
    with set_forward_context(None, vllm_config=VllmConfig()):
        return chunk_mod.chunk_gated_delta_rule_fwd(
            q, k, v, gating, beta, D**-0.5, h0, True, cu_seqlens=cu, prebuilt_meta=None,
        ), (q, k, v, beta, gating, cu, h0)


def test_ops_switch_accepts_gdnwu():
    """TILEINFER_OPS 必须接受 gdnwu 且语义正确(历史缺陷:_OPS_ALL 缺项会在役抛 ValueError)。"""
    from tileinfer.integrations.vllm_ascend import ops_enabled

    import os

    old = os.environ.get("TILEINFER_OPS")
    try:
        os.environ.pop("TILEINFER_OPS", None)
        assert ops_enabled("gdnwu") is True
        os.environ["TILEINFER_OPS"] = "decode,prefill,moe,gdnwu"
        assert ops_enabled("gdnwu") is True
        os.environ["TILEINFER_OPS"] = "decode,prefill,moe"
        assert ops_enabled("gdnwu") is False
    finally:
        if old is None:
            os.environ.pop("TILEINFER_OPS", None)
        else:
            os.environ["TILEINFER_OPS"] = old


@pytest.mark.npu
@pytest.mark.skipif(not (is_npu_available() and _HAVE_VLLM_ASCEND), reason="requires Ascend NPU")
def test_dispatch_short_fused_long_original():
    """T=128 走融合核、T=256 回退原链;计数器反映分流真实发生(P1 验收的在役口径)。"""
    _init_once()
    import os

    from tileinfer.integrations.vllm_ascend import (
        TileInferDecodeAttention,
        install_gdn_chunk_wu,
    )

    assert install_gdn_chunk_wu()
    old = os.environ.get("TILEINFER_OPS")
    os.environ["TILEINFER_OPS"] = "decode,prefill,moe,gdnwu"
    try:
        stats = TileInferDecodeAttention._stats
        calls = stats.get("gdn_wu_calls", 0)
        skips = stats.get("gdn_wu_long_skips", 0)
        (g64, o, _, fs, w, _, _), _ = _call_chain(128, seed=1, cu_dtype=torch.int32)
        assert stats["gdn_wu_calls"] == calls + 1, "短上下文必须走融合核"
        assert stats.get("gdn_wu_long_skips", 0) == skips
        assert torch.isfinite(o).all()

        (g64, o2, _, fs2, w2, _, _), _ = _call_chain(256, seed=2)
        assert stats["gdn_wu_calls"] == calls + 1, "T=256 超阈值不得走融合核(E26/E26b)"
        assert stats["gdn_wu_long_skips"] == skips + 1, "长上下文必须计长度分流跳过"
        assert torch.isfinite(o2).all()
    finally:
        if old is None:
            os.environ.pop("TILEINFER_OPS", None)
        else:
            os.environ["TILEINFER_OPS"] = old


@pytest.mark.npu
@pytest.mark.skipif(not (is_npu_available() and _HAVE_VLLM_ASCEND), reason="requires Ascend NPU")
def test_chain_parity_fused_vs_original_t128():
    """同输入整链 parity:阈值=0(强制原链)vs 缺省阈值(融合中段)——P1 的数值 oracle。"""
    _init_once()
    import os

    from tileinfer.integrations.vllm_ascend import (
        TileInferDecodeAttention,
        install_gdn_chunk_wu,
    )

    assert install_gdn_chunk_wu()
    old = os.environ.get("TILEINFER_OPS")
    old_thr = os.environ.get("TILEINFER_GDNWU_MAX_TOKENS")
    os.environ["TILEINFER_OPS"] = "decode,prefill,moe,gdnwu"
    try:
        os.environ["TILEINFER_GDNWU_MAX_TOKENS"] = "0"
        (g64, o_ref, _, fs_ref, _, _, _), inputs = _call_chain(128, seed=3)
        stats = TileInferDecodeAttention._stats
        calls = stats.get("gdn_wu_calls", 0)

        os.environ["TILEINFER_GDNWU_MAX_TOKENS"] = ""  # 回到缺省阈值 128 → 走融合
        (q, k, v, beta, gating, cu, h0) = inputs
        import vllm_ascend.ops.triton.fla.chunk as chunk_mod
        from vllm.config import VllmConfig
        from vllm.forward_context import set_forward_context

        with set_forward_context(None, vllm_config=VllmConfig()):
            got = chunk_mod.chunk_gated_delta_rule_fwd(
                q, k, v, gating, beta, D**-0.5, h0, True, cu_seqlens=cu, prebuilt_meta=None,
            )
        assert TileInferDecodeAttention._stats["gdn_wu_calls"] == calls + 1
        _, o_got, _, fs_got, _, _, _ = got
        torch.testing.assert_close(o_got.float(), o_ref.float(), rtol=TOL_RTOL, atol=TOL_ATOL)
        torch.testing.assert_close(fs_got.float(), fs_ref.float(), rtol=TOL_RTOL, atol=TOL_ATOL)
    finally:
        if old is None:
            os.environ.pop("TILEINFER_OPS", None)
        else:
            os.environ["TILEINFER_OPS"] = old
        if old_thr is None:
            os.environ.pop("TILEINFER_GDNWU_MAX_TOKENS", None)
        else:
            os.environ["TILEINFER_GDNWU_MAX_TOKENS"] = old_thr


@pytest.mark.npu
@pytest.mark.skipif(not (is_npu_available() and _HAVE_VLLM_ASCEND), reason="requires Ascend NPU")
def test_serving_contract_int32_meta():
    """在役契约(E27/D34):引擎 metadata 的 cu_seqlens 与 prebuilt_meta.chunk_indices_chunk64
    均为 int32,分派必须完成到内核 int64 签名的转换(首轮 A/B 100% fallback 的根因)。"""
    _init_once()
    import os

    from tileinfer.integrations.vllm_ascend import (
        TileInferDecodeAttention,
        _build_gdn_chunk_meta,
        install_gdn_chunk_wu,
    )

    assert install_gdn_chunk_wu()
    old = os.environ.get("TILEINFER_OPS")
    os.environ["TILEINFER_OPS"] = "decode,prefill,moe,gdnwu"
    try:
        q, k, v, beta, gating, cu32, h0 = _make(128, seed=6, cu_dtype=torch.int32)
        meta = _build_gdn_chunk_meta(tuple(cu32.to(torch.int64).tolist()), cu32.device, HV)
        meta.chunk_indices_chunk64 = meta.chunk_indices_chunk64.to(torch.int32)
        import vllm_ascend.ops.triton.fla.chunk as chunk_mod
        from vllm.config import VllmConfig
        from vllm.forward_context import set_forward_context

        stats = TileInferDecodeAttention._stats
        calls = stats.get("gdn_wu_calls", 0)
        with set_forward_context(None, vllm_config=VllmConfig()):
            g64, o, _, fs, w, _, _ = chunk_mod.chunk_gated_delta_rule_fwd(
                q, k, v, gating, beta, D**-0.5, h0, True, cu_seqlens=cu32, prebuilt_meta=meta,
            )
        assert stats["gdn_wu_calls"] == calls + 1, "int32 meta 契约下必须正常走融合核"
        assert torch.isfinite(o).all()
    finally:
        if old is None:
            os.environ.pop("TILEINFER_OPS", None)
        else:
            os.environ["TILEINFER_OPS"] = old


@pytest.mark.npu
@pytest.mark.skipif(not (is_npu_available() and _HAVE_VLLM_ASCEND), reason="requires Ascend NPU")
def test_threshold_env_override():
    """TILEINFER_GDNWU_MAX_TOKENS 覆盖:调到 256 后 T=256 也走融合核(阈值可运维调整)。"""
    _init_once()
    import os

    from tileinfer.integrations.vllm_ascend import (
        TileInferDecodeAttention,
        _gdnwu_max_tokens,
        install_gdn_chunk_wu,
    )

    assert install_gdn_chunk_wu()
    old = os.environ.get("TILEINFER_OPS")
    old_thr = os.environ.get("TILEINFER_GDNWU_MAX_TOKENS")
    os.environ["TILEINFER_OPS"] = "decode,prefill,moe,gdnwu"
    try:
        os.environ["TILEINFER_GDNWU_MAX_TOKENS"] = "256"
        assert _gdnwu_max_tokens() == 256
        stats = TileInferDecodeAttention._stats
        calls = stats.get("gdn_wu_calls", 0)
        _, _ = _call_chain(256, seed=4)
        assert stats["gdn_wu_calls"] == calls + 1, "阈值覆盖为 256 后 T=256 必须走融合核"

        os.environ["TILEINFER_GDNWU_MAX_TOKENS"] = "not-a-number"
        assert _gdnwu_max_tokens() == 128, "非法阈值回退缺省 128"
    finally:
        if old is None:
            os.environ.pop("TILEINFER_OPS", None)
        else:
            os.environ["TILEINFER_OPS"] = old
        if old_thr is None:
            os.environ.pop("TILEINFER_GDNWU_MAX_TOKENS", None)
        else:
            os.environ["TILEINFER_GDNWU_MAX_TOKENS"] = old_thr
