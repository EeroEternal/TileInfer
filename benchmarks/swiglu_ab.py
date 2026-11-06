"""Fused SwiGLU vs the torch form: numeric agreement and per-call time.

``torch_npu.npu_swiglu`` computes the whole ``silu(gate) * up`` in one kernel; this module's torch
form is ``silu`` on a slice plus a multiply (two device ops).  The expert loop calls it once per hit
expert, so the op saving multiplies by 256 experts x 32 layers on a prefill step.

Checked for agreement first: the fused kernel works in the tensor's dtype while the torch form
computes in fp32 and casts back, so a real precision difference is expected and must be measured
rather than assumed away.
"""

import os
import sys
import time

import torch

import torch_npu  # noqa: F401

DEV = torch.device("npu", 0)
DT = torch.bfloat16
HIDDEN, INTER, EXPERTS = 2048, 768, 256
FAILED = []


def main() -> int:
    from tileinfer.vllm.ops.moe import silu_and_mul, _swiglu
    from tileinfer.testing.moe_reference import reference_gate_up_swiglu_slots

    fn = getattr(torch_npu, "npu_swiglu", None)
    print(f"### npu_swiglu present: {fn is not None}")
    if fn is None:
        return 1

    for tokens in (8, 512):
        torch.manual_seed(0)
        x = torch.randn(tokens, 2 * INTER, dtype=DT, device=DEV) * 0.6
        want = silu_and_mul(x).float()
        got = fn(x).float()
        diff = (got - want).abs()
        scale = want.abs().max()
        ok = float(diff.max()) <= 0.02 * float(scale)
        print(
            f"### tokens={tokens}: max_abs={float(diff.max()):.6f} "
            f"mean_abs={float(diff.mean()):.7f} ref_max={float(scale):.4f} "
            f"bit_equal={torch.equal(got, want)} -> {'PASS' if ok else 'FAIL'}"
        )
        if not ok:
            FAILED.append(f"numeric tokens={tokens}")

    # per-call time at both shapes the serving path uses
    import os as _os

    _os.environ["TILEINFER_FLAG_WATCH"] = "1"
    _os.environ["TILEINFER_FLAG_DIR"] = "/tmp/tileinfer_flag"

    for tokens in (8, 512):
        torch.manual_seed(0)
        x = torch.randn(tokens, 2 * INTER, dtype=DT, device=DEV) * 0.6
        for label, watch in (("torch form", False), ("fused", True)):
            from tileinfer.serving.flags import set_switch

            set_switch("moe_swiglu", watch)
            _swiglu(x)
            torch.npu.synchronize()
            t0 = time.perf_counter()
            for _ in range(200):
                _swiglu(x)
            torch.npu.synchronize()
            us = (time.perf_counter() - t0) * 1e6 / 200
            print(f"### tokens={tokens} {label:11s}: {us:7.2f} us/call")
    set_switch("moe_swiglu", None)
    print("### RESULT:", "PASS" if not FAILED else f"FAIL {FAILED}")
    return 0 if not FAILED else 1


if __name__ == "__main__":
    sys.exit(main())
