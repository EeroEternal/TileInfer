"""Same-process A/B of the conv state-update store pattern (§82).  ~20 s on device.

The 0.72 -> 0.26 ms result was measured across two separate script runs, and this machine's absolute
speed drifts by 2x between windows (§35).  A ratio from two runs is therefore not evidence.  This
alternates the two modes inside one process, several rounds, and reports the paired difference — the
same discipline as ``paired_tpot.py``.
"""

import os
import statistics
import sys
import time

import torch

import torch_npu  # noqa: F401

DT = torch.bfloat16
DEV = torch.device("npu", 0)
CONV_DIM = 8192
WIDTH = 4
ROUNDS = 5
ITERS = 20


def main() -> int:
    from tileinfer.vllm.ops.fla.causal_conv1d import causal_conv1d_update

    torch.manual_seed(0)
    state = torch.randn(4, CONV_DIM, WIDTH - 1, dtype=DT, device=DEV) * 0.2
    weight = torch.randn(CONV_DIM, WIDTH, dtype=DT, device=DEV) * 0.1
    bias = torch.randn(CONV_DIM, dtype=DT, device=DEV) * 0.1
    x = torch.randn(1, CONV_DIM, dtype=DT, device=DEV) * 0.2
    idx = torch.tensor([1], dtype=torch.int32, device=DEV)

    def measure(mode: str) -> float:
        os.environ["TILEINFER_GDN_CONV_STATE1"] = mode
        os.environ.pop("TILEINFER_FLAG_WATCH", None)
        fn = lambda: causal_conv1d_update(x, state, weight, bias, "silu", conv_state_indices=idx)
        fn()
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(ITERS):
            fn()
        torch.npu.synchronize()
        return (time.perf_counter() - t0) * 1000.0 / ITERS

    on, off, diffs = [], [], []
    for r in range(ROUNDS):
        order = ("1", "0") if r % 2 == 0 else ("0", "1")
        got = {}
        for mode in order:
            got[mode] = measure(mode)
        on.append(got["1"])
        off.append(got["0"])
        diffs.append(got["0"] - got["1"])
        print(
            f"[pair{r}] order={order[0]}->{order[1]}  on={got['1']:.3f}  off={got['0']:.3f} ms  "
            f"(off-on)={diffs[-1]:+.3f}",
            flush=True,
        )
    os.environ.pop("TILEINFER_GDN_CONV_STATE1", None)

    on_med, off_med = statistics.median(on), statistics.median(off)
    print(
        f"### on median {on_med:.3f} ms   off median {off_med:.3f} ms\n"
        f"### paired difference median {statistics.median(diffs):+.3f} ms "
        f"({statistics.median(diffs) / on_med * 100:+.1f}%)\n"
        f"### measured_ratio on/off = {off_med / on_med:.3f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
