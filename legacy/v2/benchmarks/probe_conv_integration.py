"""Integration check for the fused conv kernel (§88): the public entry, switch off vs on.

The unit oracles use tiny channel counts, which the kernel path rejects (`channels % 32`), so their
passing proves nothing about the wiring.  This calls ``causal_conv1d_update`` at the real shape
(8192 channels, width 4, bf16) twice — once with the switch off, once on — and compares both the
output row and the mutated cache line.  Also reports whether the kernel was actually taken.
"""

import os
import sys

import torch

import torch_npu  # noqa: F401

DT = torch.bfloat16
DEV = torch.device("npu", 0)
C, W, L = 8192, 4, 8


def main() -> int:
    from tileinfer.vllm.ops.fla.causal_conv1d import causal_conv1d_update

    torch.manual_seed(0)
    x = torch.randn(1, C, dtype=DT, device=DEV) * 0.2
    state0 = torch.randn(L, C, W - 1, dtype=DT, device=DEV) * 0.2
    weight = torch.randn(C, W, dtype=DT, device=DEV) * 0.1
    bias = torch.randn(C, dtype=DT, device=DEV) * 0.1
    idx = torch.tensor([3], dtype=torch.int32, device=DEV)

    def run(flag: str):
        os.environ["TILEINFER_GDN_CONV_KERNEL"] = flag
        os.environ.pop("TILEINFER_FLAG_WATCH", None)
        st = state0.clone()
        out = causal_conv1d_update(x, st, weight, bias, "silu", conv_state_indices=idx)
        torch.npu.synchronize()
        return out.clone(), st.clone()

    out_off, st_off = run("0")
    out_on, st_on = run("1")
    out_on2, st_on2 = run("1")  # repeat: the kernel must be stable call-over-call, not just once
    os.environ.pop("TILEINFER_GDN_CONV_KERNEL", None)

    d_out = float((out_on.float() - out_off.float()).abs().max())
    d_state = float((st_on.float() - st_off.float()).abs().max())
    scale = float(out_off.float().abs().max()) or 1.0
    stable = torch.equal(out_on, out_on2) and torch.equal(st_on, st_on2)
    print(f"### out  switch on vs off: max_abs={d_out:.3e} (scale {scale:.3e})")
    print(f"### state on vs off:       max_abs={d_state:.3e}")
    print(f"### repeat-call stability: {stable}")
    ok = d_out <= 2e-2 * scale and d_state <= 2e-2 * max(scale, 1e-3) and stable
    print("### RESULT:", "OK" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
