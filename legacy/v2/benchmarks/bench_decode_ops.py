"""Where does one decode step's ~103 ms go?  Micro-benchmarks at serving shapes, no server.

The phase table's columns are ambiguous (§60/§62) and its totals include one-off compile time, so
this measures each decode-path operator directly at the shape serving actually uses (1 token, batch
1, Qwen3.6 linear-attention geometry) and compares the **sum** against the measured TPOT.  That
comparison is the useful part: if the parts add up to ~103 ms the budget is fully attributed; if they
add up to much less, the remainder is dispatch and only a graph could remove it.

Layout per GDN layer: conv update (16x256 for the qkvz part) then the recurrent delta rule with
16 qk heads x 128 and 32 v heads x 128.
"""

import sys
import time

import torch

import torch_npu  # noqa: F401

DT = torch.bfloat16
DEV = torch.device("npu", 0)
ITERS = 20

# Qwen3.6-35B-A3B
HIDDEN = 2048
CONV_DIM = 8192          # qkvz channels the causal conv sees
HG, HV, K, V = 16, 32, 128, 128
WIDTH = 4
EXPERTS, INTER, TOPK = 256, 768, 8
NH, NKV, HEAD_DIM = 16, 2, 256


def _bench(label, fn):
    try:
        fn()
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(ITERS):
            fn()
        torch.npu.synchronize()
        ms = (time.perf_counter() - t0) * 1000.0 / ITERS
        print(f"### {label:44s} {ms:7.2f} ms/call")
        return ms
    except Exception as exc:  # noqa: BLE001
        print(f"### {label:44s}       -  ({type(exc).__name__}: {exc})")
        return None


def main() -> int:
    from tileinfer.vllm.ops.fla.causal_conv1d import causal_conv1d_update
    from tileinfer.vllm.ops.fla.recurrent import fused_recurrent_gated_delta_rule_packed_decode
    from tileinfer.vllm.ops.moe import fused_experts

    torch.manual_seed(0)
    parts = {}

    # ---- GDN: conv update + recurrent delta rule (one layer) ----
    conv_state = torch.randn(4, CONV_DIM, WIDTH - 1, dtype=DT, device=DEV) * 0.2
    weight = torch.randn(CONV_DIM, WIDTH, dtype=DT, device=DEV) * 0.1
    bias = torch.randn(CONV_DIM, dtype=DT, device=DEV) * 0.1
    x = torch.randn(1, CONV_DIM, dtype=DT, device=DEV) * 0.2
    idx = torch.tensor([1], dtype=torch.int32, device=DEV)
    parts["conv"] = _bench(
        "gdn: causal_conv1d_update",
        lambda: causal_conv1d_update(x, conv_state, weight, bias, "silu", conv_state_indices=idx),
    )

    mixed = torch.randn(1, 2 * HG * K + HV * V, dtype=DT, device=DEV) * 0.2
    a = torch.randn(1, HV, dtype=DT, device=DEV) * 0.2
    b = torch.randn(1, HV, dtype=DT, device=DEV) * 0.2
    A_log = torch.randn(HV, dtype=torch.float32, device=DEV) * 0.1
    dt_bias = torch.randn(HV, dtype=torch.float32, device=DEV) * 0.1
    state = torch.randn(4, HV, V, K, dtype=DT, device=DEV) * 0.1
    out = torch.zeros(1, 1, HV, V, dtype=DT, device=DEV)
    parts["rec"] = _bench(
        "gdn: recurrent packed decode",
        lambda: fused_recurrent_gated_delta_rule_packed_decode(
            mixed, a, b, A_log, dt_bias, K**-0.5, state, out, idx
        ),
    )

    # ---- MoE (one layer, decode) ----
    w13 = torch.randn(EXPERTS, 2 * INTER, HIDDEN, dtype=DT, device=DEV) * 0.02
    w2 = torch.randn(EXPERTS, HIDDEN, INTER, dtype=DT, device=DEV) * 0.02
    mx = torch.randn(1, HIDDEN, dtype=DT, device=DEV) * 0.2
    ids = torch.stack([torch.randperm(EXPERTS, device=DEV)[:TOPK]]).to(torch.int32)
    wts = torch.rand(1, TOPK, dtype=torch.float32, device=DEV)
    wts = wts / wts.sum(dim=1, keepdim=True)
    parts["moe"] = _bench(
        "moe: fused_experts (decode, 1 token)",
        lambda: fused_experts(mx, w13, w2, wts, ids, activation="silu"),
    )

    print("### ---- per-step totals (Qwen3.6: 30 GDN + 32 MoE + 10 attn) ----")
    total = 0.0
    if parts.get("conv") and parts.get("rec"):
        gdn = (parts["conv"] + parts["rec"]) * 30
        total += gdn
        print(f"### GDN  x30 layers                          {gdn:7.2f} ms/step")
    if parts.get("moe"):
        moe = parts["moe"] * 32
        total += moe
        print(f"### MoE  x32 layers                          {moe:7.2f} ms/step")
    print(f"### attributed so far                        {total:7.2f} ms/step")
    print("### measured TPOT is ~103 ms/step; attention decode and dispatch are the remainder")
    return 0


if __name__ == "__main__":
    sys.exit(main())
