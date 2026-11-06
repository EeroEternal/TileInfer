"""Does batching all experts into one ``bmm`` pay off?  Device, ~30 s.

Claim being tested (§53): at prefill shapes the expert path is 8192 kernel launches per step
(256 experts x 32 layers) and the arithmetic is ~0.3% of the time, so replacing the per-expert loop
with two ``bmm`` calls over a padded ``[E, max_n, hidden]`` tensor should collapse that.

Measured: correctness against ``reference_moe`` at a small shape (both paths), then wall time for
``fused_experts`` at 512 tokens / top-8 / 256 experts with the switch off and on.  Also at decode
(1 token), where the shape gate deliberately keeps the loop -- the numbers there show the gate is
doing its job rather than being a hidden regression.
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


def _mode(bmm: bool):
    os.environ["TILEINFER_MOE_BMM"] = "1" if bmm else "0"
    os.environ.pop("TILEINFER_FLAG_WATCH", None)


def correctness():
    from tileinfer.testing.moe_reference import reference_moe

    x, w13, w2, ids, w = _make(24)
    want = reference_moe(x, w13, w2, ids, w)
    print("### correctness vs fp32 oracle (24 tokens, all 256 experts hit)")
    for bmm in (False, True):
        _mode(bmm)
        got = _run(x, w13, w2, ids, w).float()
        d = (got - want).abs()
        print(
            f"###   bmm={int(bmm)}: max_abs={float(d.max()):.6f} "
            f"mean_abs={float(d.mean()):.8f} oracle_max={float(want.abs().max()):.4f}"
        )


def timing(tokens, iters=5):
    x, w13, w2, ids, w = _make(tokens)
    out = {}
    for bmm in (False, True):
        for fp32 in (True, False):
            os.environ["TILEINFER_MOE_BMM"] = "1" if bmm else "0"
            os.environ["TILEINFER_MOE_FP32"] = "1" if fp32 else "0"
            os.environ.pop("TILEINFER_FLAG_WATCH", None)
            _run(x, w13, w2, ids, w)
            torch.npu.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                _run(x, w13, w2, ids, w)
            torch.npu.synchronize()
            ms = (time.perf_counter() - t0) * 1000.0 / iters
            out[(bmm, fp32)] = ms
            print(f"### tokens={tokens:4d} bmm={int(bmm)} fp32={int(fp32)}: {ms:8.2f} ms/call")
    base = out[(False, True)]
    for key, ms in sorted(out.items()):
        if ms > 0:
            print(f"### tokens={tokens} bmm={int(key[0])} fp32={int(key[1])}: "
                  f"ratio(loop+fp32 / this) = {base / ms:.3f}")
    os.environ.pop("TILEINFER_MOE_BMM", None)
    os.environ.pop("TILEINFER_MOE_FP32", None)


if __name__ == "__main__":
    print(f"### device={torch.npu.get_device_name(0)}")
    correctness()
    timing(512)
    timing(1, iters=20)
    sys.exit(0)
