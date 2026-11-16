"""Correctness + in-process ratio for the experimental fused conv-update kernel.  ~1 min on device.

Compares ``build_conv_update_kernel`` against the torch ``causal_conv1d_update`` on identical inputs
(out and the mutated state), then times both **inside one process** so the ratio is usable under the
§83.4 rule.  The kernel is default-off; nothing here changes a default.
"""

import sys
import time

import torch

import torch_npu  # noqa: F401

DT = torch.bfloat16
DEV = torch.device("npu", 0)
C, W, L = 8192, 4, 8
ITERS = 20
ROUNDS = 4


def main() -> int:
    from tileinfer.kernels.gdn.conv_update_ascend950 import (
        build_conv_update_kernel,
        gdn_conv_update_kernel_available,
    )
    from tileinfer.vllm.ops.fla.causal_conv1d import causal_conv1d_update

    if not gdn_conv_update_kernel_available():
        print("### RESULT: NOT APPLICABLE (no tilelang)")
        return 0

    torch.manual_seed(0)
    x = torch.randn(C, dtype=DT, device=DEV) * 0.2
    state0 = torch.randn(L, C, W - 1, dtype=DT, device=DEV) * 0.2
    weight = torch.randn(C, W, dtype=DT, device=DEV) * 0.1
    bias = torch.randn(C, dtype=torch.float32, device=DEV) * 0.1
    idx = torch.tensor([3], dtype=torch.int32, device=DEV)

    try:
        import tileinfer.kernels.gdn.conv_update_ascend950 as _k

        _k.__dict__.update(
            C=C, W=W, L=L, taps=W - 1, cores=32,
            per=C // 32, dtype="bfloat16",
        )
        kernel = build_conv_update_kernel(
            channels=C, width=W, lines=L, dtype="bfloat16"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"### BUILD FAILED: {type(exc).__name__}: {exc}")
        return 1

    # ---- correctness ----
    state_k = state0.clone()
    out_k = torch.empty(C, dtype=DT, device=DEV)
    try:
        kernel(x, state_k, weight, bias, 3, out_k)
        torch.npu.synchronize()
    except Exception as exc:  # noqa: BLE001
        print(f"### RUN FAILED: {type(exc).__name__}: {exc}")
        return 1

    state_t = state0.clone()
    out_t = causal_conv1d_update(
        x.unsqueeze(0), state_t, weight, bias, "silu", conv_state_indices=idx
    ).reshape(-1)
    torch.npu.synchronize()

    d_out = float((out_k.float() - out_t.float()).abs().max())
    d_state = float((state_k.float() - state_t.float()).abs().max())
    scale = float(out_t.float().abs().max()) or 1.0
    print(
        f"### correctness: out max_abs={d_out:.3e} (scale {scale:.3e})  "
        f"state max_abs={d_state:.3e}"
    )
    ok = d_out <= 2e-2 * scale and d_state <= 2e-2 * max(scale, 1e-3)
    print(f"### correctness verdict: {'OK' if ok else 'MISMATCH'}")

    # ---- timing, alternating inside one process ----
    def time_kernel() -> float:
        st = state0.clone()
        o = torch.empty(C, dtype=DT, device=DEV)
        kernel(x, st, weight, bias, 3, o)
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(ITERS):
            kernel(x, st, weight, bias, 3, o)
        torch.npu.synchronize()
        return (time.perf_counter() - t0) * 1000.0 / ITERS

    def time_torch() -> float:
        st = state0.clone()
        fn = lambda: causal_conv1d_update(
            x.unsqueeze(0), st, weight, bias, "silu", conv_state_indices=idx
        )
        fn()
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(ITERS):
            fn()
        torch.npu.synchronize()
        return (time.perf_counter() - t0) * 1000.0 / ITERS

    k_times, t_times = [], []
    for r in range(ROUNDS):
        if r % 2 == 0:
            k_times.append(time_kernel())
            t_times.append(time_torch())
        else:
            t_times.append(time_torch())
            k_times.append(time_kernel())
        print(f"[pair{r}] kernel={k_times[-1]:.3f}  torch={t_times[-1]:.3f} ms", flush=True)

    import statistics

    k_med, t_med = statistics.median(k_times), statistics.median(t_times)
    print(
        f"### kernel median {k_med:.3f} ms   torch median {t_med:.3f} ms\n"
        f"### measured_ratio kernel/torch (lower is better for torch) = "
        f"{t_med / k_med:.3f}x  ( >1 means the kernel is faster )"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
