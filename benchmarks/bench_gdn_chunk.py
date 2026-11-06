"""Where the GDN chunk chain spends its time, at prefill shape, plus the sub-steps.

§56 left ~0.72 s of marginal prefill cost and the per-op table showed the *installed* GDN ops at
~1 ms each — so the cost is in the chain that is not individually labelled.  This times the public
entry point and then each stage of the chain, so the next optimisation targets the real block rather
than the first plausible one (the mistake §51 nearly made).

Qwen3.6 linear-attention shape: 16 qk heads x 128, 32 v heads x 128, chunk 64.
"""

import sys
import time

import torch

import torch_npu  # noqa: F401

DT = torch.bfloat16
B, T = 1, 512
HG, H, K, V = 16, 32, 128, 128
ITERS = 3


def _bench(label, fn):
    fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        fn()
    torch.npu.synchronize()
    ms = (time.perf_counter() - t0) * 1000.0 / ITERS
    print(f"### {label:38s} {ms:8.2f} ms")
    return ms


def main() -> int:
    from tileinfer.vllm.ops.fla.chunk import chunk_gated_delta_rule, chunk_gated_delta_rule_fwd
    from tileinfer.vllm.ops.fla.delta_h import chunk_gated_delta_rule_fwd_h
    from tileinfer.vllm.ops.fla.kkt import chunk_scaled_dot_kkt_fwd
    from tileinfer.vllm.ops.fla.solve_tril import solve_tril
    from tileinfer.vllm.ops.fla.wu import recompute_w_u_fwd

    dev = torch.device("npu", 0)
    torch.manual_seed(0)
    q = torch.randn(B, T, HG, K, dtype=DT, device=dev) * 0.2
    k = torch.randn(B, T, HG, K, dtype=DT, device=dev) * 0.2
    v = torch.randn(B, T, H, V, dtype=DT, device=dev) * 0.2
    g = torch.randn(B, T, H, dtype=torch.float32, device=dev) * 0.1
    beta = torch.rand(B, T, H, dtype=torch.float32, device=dev)

    total = _bench("chunk_gated_delta_rule (public)", lambda: chunk_gated_delta_rule(q, k, v, g, beta))
    _bench("chunk_gated_delta_rule_fwd", lambda: chunk_gated_delta_rule_fwd(q, k, v, g, beta))

    # Stage-level: the helpers take the shapes the chain builds internally.
    try:
        _bench(
            "chunk_scaled_dot_kkt_fwd (kkt)",
            lambda: chunk_scaled_dot_kkt_fwd(k=k, beta=beta, cu_seqlens=None, chunk_size=64),
        )
    except Exception as exc:  # noqa: BLE001 - shape drift is expected across refactors
        print(f"### kkt stage not timed: {type(exc).__name__}: {exc}")
    try:
        _bench("solve_tril", lambda: solve_tril(torch.randn(B, HG, T, T, dtype=DT, device=dev)))
    except Exception as exc:  # noqa: BLE001
        print(f"### solve_tril not timed: {type(exc).__name__}: {exc}")
    try:
        _bench(
            "recompute_w_u_fwd (wu)",
            lambda: recompute_w_u_fwd(k=k, v=v, beta=beta, A=None, cu_seqlens=None, chunk_size=64),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"### wu not timed: {type(exc).__name__}: {exc}")
    try:
        _bench(
            "chunk_gated_delta_rule_fwd_h (delta_h)",
            lambda: chunk_gated_delta_rule_fwd_h(k, w=k, u=v, g=g, chunk_size=64),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"### delta_h not timed: {type(exc).__name__}: {exc}")
    print(f"### per-layer total: {total:.2f} ms  -> 30 layers = {total * 30:.0f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
