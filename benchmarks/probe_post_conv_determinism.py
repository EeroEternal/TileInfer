"""Is ``fused_post_conv_prep`` bit-deterministic?  ~30 s, no server.

The last operator on our decode path that had not been probed (§85).  It takes the packed conv
output and produces ``(q, k, v, g, beta)`` — several reductions (L2 norms, the gate transform) over
``[tokens, heads, dim]`` tensors, which is exactly the shape of op where a non-deterministic
reduction order could hide.

If this is deterministic too, the 0.036 nats self-jitter (§73) is very likely on vLLM's side
(norm/rope/projections), which is outside this repository — and then §52's cross-implementation
criterion is the honest way to record it.
"""

import sys

import torch

import torch_npu  # noqa: F401

DT = torch.bfloat16
DEV = torch.device("npu", 0)
HK, HV, K, V = 16, 32, 128, 128
TRIALS = 20


def main() -> int:
    from tileinfer.vllm.ops.fla.post_conv import fused_post_conv_prep

    torch.manual_seed(0)
    total_bad = 0
    for tokens in (1, 8):
        conv_out = torch.randn(tokens, 2 * HK * K + HV * V, dtype=DT, device=DEV) * 0.2
        a = torch.randn(tokens, HV, dtype=DT, device=DEV) * 0.2
        b = torch.randn(tokens, HV, dtype=DT, device=DEV) * 0.2
        A_log = torch.randn(HV, dtype=torch.float32, device=DEV) * 0.1
        dt_bias = torch.randn(HV, dtype=torch.float32, device=DEV) * 0.1

        def run():
            return fused_post_conv_prep(
                conv_out, a, b, A_log, dt_bias, HK, K, V, apply_l2norm=True
            )

        first = run()
        bad = 0
        worst = 0.0
        for _ in range(TRIALS):
            got = run()
            for x0, x1 in zip(first, got):
                if x0 is None or x1 is None:
                    continue
                if not torch.equal(x0, x1):
                    bad += 1
                    worst = max(worst, float((x0.float() - x1.float()).abs().max()))
        total_bad += bad
        n_out = sum(1 for x in first if x is not None)
        print(
            f"### tokens={tokens}: {bad}/{TRIALS} trials differ across {n_out} outputs "
            f"(worst_abs={worst:.3e})"
        )

    print("### RESULT:", "SAFE" if total_bad == 0 else "NON-DETERMINISTIC")
    return 0


if __name__ == "__main__":
    sys.exit(main())
