"""GDN chunk 链融合内核(w = (I+A)^-1·k·beta·e^g 等)设备测试(Ascend 950)。

运行::

    source .agent_loop/run/env.sh && .venv/bin/python -m pytest tests/test_tilelang_ascend950_gdn_wu.py -v

BACKLOG M-GDN 9c / H10。用例:
* parity vs fp64 oracle(oracle/test_gdn_wu_reference.py 已自检):
  T=128 满块 / T=100 尾块(36 有效)/ 多序列含短尾 / T=1 单 token;
* parity vs 在役 triton 基线(solve_tril + recompute_w_u,同输入同容差);
* T=8192 长上下文 parity 抽验(链级接口正确性)。
TOL 3e-2(与现有内核一致;差异来源 = M 的 bf16 舍入点与 cube fp16 路径,与基线同级)。
"""

from __future__ import annotations

import pytest
import torch

from tileinfer.testing.gdn_reference import chunk_wu_reference
from tileinfer.utils import is_npu_available

pytestmark = pytest.mark.npu

HK, HV, D = 16, 32, 128  # Qwen3.6 GDN 在役形状
TOL_RTOL, TOL_ATOL = 3e-2, 3e-2


try:  # pragma: no cover
    import vllm_ascend.ops.gdn  # noqa: F401  先拉起完整链(device_op 反向依赖 ops)

    _HAVE_VLLM_ASCEND = True
except Exception:  # pragma: no cover
    _HAVE_VLLM_ASCEND = False


@pytest.fixture(autouse=True, scope="session")
def _init_env():
    """独立测试进程没有 vLLM worker 的初始化链(E22 探针同款):triton 设备属性 +
    distributed(1 rank)+ VllmConfig(set_current_vllm_config 内 initialize_model_parallel)。"""
    import os

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29633")
    torch.npu.set_device(0)
    from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

    init_device_properties_triton()
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel

    # 套件内已初始化(gdn_reorder wiring 等先行测试触发 distributed 全局态,iter-204
    # D141 后 install 链会提前拉起 worker patch 家族)时忽略,复用既有组。
    try:
        # init_world_group 链内 CustomOp 构造要求 config 上下文(iter-205 修复:
        # 无上下文时抛 "Current vLLM config is not set" 被吞,留下 torch 已 init
        # 而 vllm _WORLD 未建的分裂态,连坐后续文件 PCP 兜底)。
        with set_current_vllm_config(VllmConfig()):
            init_distributed_environment(world_size=1, rank=0, local_rank=0, distributed_init_method="env://")
            initialize_model_parallel(tensor_model_parallel_size=1)
    except Exception:  # noqa: BLE001 - 已初始化时复用
        pass


def _make(lens, seed=0, scale=0.2):
    g = torch.Generator(device="cpu").manual_seed(seed)
    t_total = sum(lens)
    k = torch.randn(1, t_total, HK, D, generator=g).to(torch.bfloat16)
    k = torch.nn.functional.normalize(k.float(), dim=-1).to(torch.bfloat16).npu()
    v = (torch.randn(1, t_total, HV, D, generator=g) * scale).to(torch.bfloat16).npu()
    beta = torch.rand(1, t_total, HV, generator=g).to(torch.bfloat16).npu()
    g_cum = (-torch.rand(1, t_total, HV, generator=g).cumsum(1) * 0.05).float().npu()
    cu = torch.tensor([0] + torch.tensor(lens).cumsum(0).tolist(), dtype=torch.int64).npu()
    return k, v, beta, g_cum, cu


def _assert_parity(lens, seed=0):
    from tileinfer.kernels.gdn.chunk_wu_ascend950 import gdn_chunk_wu
    from vllm_ascend.ops.triton.fla.utils import prepare_chunk_indices

    k, v, beta, g_cum, cu = _make(lens, seed)
    t_total = int(cu[-1])
    chunk_idx = prepare_chunk_indices(cu.cpu(), 64).npu()
    # kkt 输入:在役链由 chunk_scaled_dot_kkt_fwd 产生;oracle 可自算,这里统一用
    # kkt_reference(设备上逐块 fp64,仅测试用)保证 A 的来源单一。
    from tileinfer.testing.gdn_reference import kkt_reference

    a64 = kkt_reference(k.cpu(), beta.cpu(), g_cum.cpu(), cu.cpu())
    a = a64.to(torch.float32).npu()
    w_k, u_k = gdn_chunk_wu(a, k, v, beta, g_cum, cu, chunk_idx)
    w_ref, u_ref, _ = chunk_wu_reference(k.cpu(), v.cpu(), beta.cpu(), g_cum.cpu(), cu.cpu())
    torch.testing.assert_close(w_k.float().cpu(), w_ref.float(), rtol=TOL_RTOL, atol=TOL_ATOL)
    torch.testing.assert_close(u_k.float().cpu(), u_ref.float(), rtol=TOL_RTOL, atol=TOL_ATOL)
    # 无效行(尾块补齐区)必须不被写脏:与 oracle 零行一致
    return w_k, u_k


@pytest.mark.skipif(not is_npu_available(), reason="requires Ascend NPU")
def test_wu_parity_t128_full_chunks():
    _assert_parity([128])


@pytest.mark.skipif(not is_npu_available(), reason="requires Ascend NPU")
def test_wu_parity_partial_tail():
    _assert_parity([100])


@pytest.mark.skipif(not is_npu_available(), reason="requires Ascend NPU")
def test_wu_parity_multiseq_short_tails():
    _assert_parity([100, 100, 65])


@pytest.mark.skipif(not is_npu_available(), reason="requires Ascend NPU")
def test_wu_parity_single_token():
    _assert_parity([1])


@pytest.mark.skipif(not is_npu_available(), reason="requires Ascend NPU")
def test_wu_parity_vs_inservice_triton_t128():
    """与在役 triton 基线(solve_tril + recompute_w_u)同输入对照。"""
    from vllm_ascend.ops.triton.fla.solve_tril import solve_tril
    from vllm_ascend.ops.triton.fla.wy_fast import recompute_w_u_fwd

    k, v, beta, g_cum, cu = _make([128], seed=5)
    from vllm_ascend.ops.triton.fla.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd

    a_kkt = chunk_scaled_dot_kkt_fwd(k=k, beta=beta, g_cumsum=g_cum, cu_seqlens=cu, chunk_indices=None, output_dtype=torch.float32)
    a_tri = solve_tril(A=a_kkt, cu_seqlens=cu, chunk_indices_large_block=None, chunk_indices_bt=None, output_dtype=k.dtype)
    w_ref, u_ref = recompute_w_u_fwd(k=k, v=v, beta=beta, A=a_tri, g_cumsum=g_cum, cu_seqlens=cu, chunk_indices=None)
    from tileinfer.kernels.gdn.chunk_wu_ascend950 import gdn_chunk_wu
    from vllm_ascend.ops.triton.fla.utils import prepare_chunk_indices

    chunk_idx = prepare_chunk_indices(cu.cpu(), 64).npu()
    w_k, u_k = gdn_chunk_wu(a_kkt, k, v, beta, g_cum, cu, chunk_idx)
    torch.testing.assert_close(w_k.float(), w_ref.float(), rtol=TOL_RTOL, atol=TOL_ATOL)
    torch.testing.assert_close(u_k.float(), u_ref.float(), rtol=TOL_RTOL, atol=TOL_ATOL)


@pytest.mark.skipif(not is_npu_available(), reason="requires Ascend NPU")
def test_wu_parity_t8192_smoke():
    w_k, u_k = _assert_parity([8192], seed=9)
    assert w_k.shape == (1, 8192, HV, D) and u_k.shape == (1, 8192, HV, D)
