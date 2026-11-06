"""The GDN chunk chain, stage by stage, at prefill shape — no server needed.

§57 pinned 16.15 ms per layer on ``chunk_gated_delta_rule`` (x30 layers = 484 ms, 67% of the
marginal prefill cost) but found the *installed ops* at ~1 ms each, so the time is inside the chain.
``chunk_gated_delta_rule_fwd`` is the six-step composition (chunk.py:288-339), so this replays those
steps with the real signatures and reports each one.  Deliberately not guessing which step dominates:
§51 and §57.1 both showed the intuition was wrong.
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
    try:
        out = fn()
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(ITERS):
            out = fn()
        torch.npu.synchronize()
        ms = (time.perf_counter() - t0) * 1000.0 / ITERS
        print(f"### {label:34s} {ms:8.3f} ms")
        return ms, out
    except Exception as exc:  # noqa: BLE001 - a stage may reject these shapes; report, do not crash
        print(f"### {label:34s}       -  ({type(exc).__name__}: {exc})")
        return None, None


def main() -> int:
    from tileinfer.vllm.ops.fla.chunk import chunk_fwd_o, chunk_gated_delta_rule, _chunk_cumsum
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
    scale = K**-0.5

    total, _ = _bench("chunk_gated_delta_rule (whole)", lambda: chunk_gated_delta_rule(q, k, v, g, beta))
    _bench("0 _chunk_cumsum", lambda: _chunk_cumsum(g, None))
    g_cumsum = _chunk_cumsum(g, None)
    a_ms, A = _bench(
        "1 kkt (chunk_scaled_dot_kkt_fwd)",
        lambda: chunk_scaled_dot_kkt_fwd(k=k, beta=beta, g=g_cumsum, cu_seqlens=None),
    )
    s_ms, Ainv = _bench(
        "2 solve_tril", lambda: solve_tril(A=A, cu_seqlens=None, output_dtype=DT)
    )
    wu_ms, wu = _bench(
        "3 wu (recompute_w_u_fwd)",
        lambda: recompute_w_u_fwd(k=k, v=v, beta=beta, A=Ainv, g_cumsum=g_cumsum, cu_seqlens=None),
    )
    if wu is not None:
        w, u = wu
        h_ms, hv = _bench(
            "4 delta_h (fwd_h)",
            lambda: chunk_gated_delta_rule_fwd_h(
                k=k, w=w, u=u, g=g_cumsum, initial_state=None,
                output_final_state=False, cu_seqlens=None,
            ),
        )
        if hv is not None:
            h, v_new = hv[0], hv[1]
            o_ms, _ = _bench(
                "5 chunk_fwd_o",
                lambda: chunk_fwd_o(q=q, k=k, v=v_new, h=h, g=g_cumsum, scale=scale, cu_seqlens=None),
            )
            parts = [x for x in (a_ms, s_ms, wu_ms, h_ms, o_ms) if x is not None]
            if total and parts:
                print(
                    f"### sum of stages: {sum(parts):.3f} ms of {total:.3f} ms "
                    f"({100.0 * sum(parts) / total:.0f}%)"
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
