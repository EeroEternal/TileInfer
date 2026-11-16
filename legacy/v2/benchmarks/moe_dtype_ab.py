"""MoE candidate: fp32 matmuls (historical) vs input-dtype matmuls, same process.

C4 wants a same-process comparison, and ``_matmul_dtype`` reads ``TILEINFER_MOE_FP32`` per call, so
one process can measure both.  C3 wants the numbers checked, not just timed: both modes are compared
against ``testing.moe_reference.reference_moe`` (straight-line fp32 oracle).

Shapes are Qwen3.6-35B-A3B's MoE vocabulary: hidden 2048, 256 experts, top-8, bf16 weights.
"""

import os
import sys
import time

import torch

import torch_npu  # noqa: F401

DEV = torch.device("npu", 0)
DT = torch.bfloat16
HIDDEN, INTER, EXPERTS, TOPK = 2048, 768, 256, 8


def _make(tokens, seed=0):
    torch.manual_seed(seed)
    w13 = torch.randn(EXPERTS, 2 * INTER, HIDDEN, dtype=DT, device=DEV) * 0.02
    w2 = torch.randn(EXPERTS, HIDDEN, INTER, dtype=DT, device=DEV) * 0.02
    x = torch.randn(tokens, HIDDEN, dtype=DT, device=DEV) * 0.2
    ids = torch.stack([torch.randperm(EXPERTS, device=DEV)[:TOPK] for _ in range(tokens)]).to(
        torch.int32
    )
    w = torch.rand(tokens, TOPK, dtype=torch.float32, device=DEV)
    w = w / w.sum(dim=1, keepdim=True)
    return x, w13, w2, ids, w


def _run(x, w13, w2, ids, w):
    from tileinfer.vllm.ops.moe import fused_experts

    return fused_experts(x, w13, w2, w, ids, activation="silu")


def _mode(env):
    for key, value in env.items():
        os.environ[key] = value


def correctness():
    from tileinfer.testing.moe_reference import reference_moe

    x, w13, w2, ids, w = _make(24)
    want = reference_moe(x, w13, w2, ids, w)
    rows = []
    variants = (
        ("fp32+loop", {"TILEINFER_MOE_FP32": "1", "TILEINFER_MOE_SORTED": "0"}),
        ("input+loop", {"TILEINFER_MOE_FP32": "0", "TILEINFER_MOE_SORTED": "0"}),
        ("fp32+sorted", {"TILEINFER_MOE_FP32": "1", "TILEINFER_MOE_SORTED": "1"}),
        ("input+sorted", {"TILEINFER_MOE_FP32": "0", "TILEINFER_MOE_SORTED": "1"}),
    )
    for label, env in variants:
        _mode(env)
        got = _run(x, w13, w2, ids, w).float()
        diff = (got - want).abs()
        rows.append((label, float(diff.max()), float(diff.mean()), float(want.abs().max())))
    print("### correctness vs fp32 oracle (24 tokens, top-8)")
    for label, mx, mean, scale in rows:
        print(f"###   {label:14s} max_abs={mx:.6f} mean_abs={mean:.8f} oracle_max={scale:.4f}")
    return rows


def timing(tokens=512, iters=5):
    x, w13, w2, ids, w = _make(tokens)
    print(f"### timing: {tokens} tokens, top-8, {EXPERTS} experts, {iters} iters")
    out = {}
    variants = (
        ("fp32+loop", {"TILEINFER_MOE_FP32": "1", "TILEINFER_MOE_SORTED": "0"}),
        ("input+loop", {"TILEINFER_MOE_FP32": "0", "TILEINFER_MOE_SORTED": "0"}),
        ("input+sorted", {"TILEINFER_MOE_FP32": "0", "TILEINFER_MOE_SORTED": "1"}),
    )
    for label, env in variants:
        _mode(env)
        _run(x, w13, w2, ids, w)  # warm (allocator, JIT)
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            _run(x, w13, w2, ids, w)
        torch.npu.synchronize()
        ms = (time.perf_counter() - t0) * 1000.0 / iters
        out[label] = ms
        print(f"###   {label:14s} {ms:8.2f} ms/call")
    for label, ms in out.items():
        if label != "fp32+loop" and ms > 0:
            print(f"### measured_ratio {label} vs fp32+loop = {out['fp32+loop'] / ms:.3f}")
    return out


if __name__ == "__main__":
    print(f"### device={torch.npu.get_device_name(0)} torch={torch.__version__}", flush=True)
    correctness()
    timing()
    print("### done")
    sys.exit(0)
