"""Is ``index_copy_`` bit-deterministic on this NPU?  ~30 s, no server.

§45 probed ``index_add_`` and ``scatter_add_`` (both non-deterministic on NPU) and ``reshape+sum``
(deterministic), and concluded that ``index_add_`` was the MoE culprit.  ``index_copy_`` was never
probed — yet it is now the write-back primitive of **both** the MoE decode path (§70) and the
capture-safe conv update (§72).  If it is atomic too, both paths produce run-to-run variation and
graph capture would only make that permanent.

The distinction that matters: ``index_add_`` **accumulates** (order matters), ``index_copy_``
**overwrites** (order must not matter if the destination is unique).  This measures whether the
hardware honours that with a permutation as index.
"""

import sys

import torch

import torch_npu  # noqa: F401

DEV = torch.device("npu", 0)
ROWS, HIDDEN = 64, 256
TRIALS = 50


def main() -> int:
    torch.manual_seed(0)
    src = torch.randn(ROWS, HIDDEN, dtype=torch.float32, device=DEV)
    perm = torch.randperm(ROWS, device=DEV)

    def run(device_index: bool):
        out = torch.zeros(ROWS, HIDDEN, dtype=torch.float32, device=DEV)
        out.index_copy_(0, perm, src)
        return out

    first = run(True)
    differing = 0
    worst = 0.0
    for _ in range(TRIALS):
        got = run(True)
        if not torch.equal(got, first):
            differing += 1
            worst = max(worst, float((got - first).abs().max()))
    verdict = "DETERMINISTIC" if differing == 0 else f"NON-DETERMINISTIC in {differing}/{TRIALS}"
    print(f"### index_copy_ (permutation): {verdict}  worst_abs={worst:.3e}")

    # Same data through a scatter-like write with duplicate destinations: that one *must* be odd
    # (undefined order), so it is a control that the probe can actually detect a difference.
    dup = torch.zeros(ROWS, dtype=torch.int64, device=DEV)

    def run_dup():
        out = torch.zeros(8, HIDDEN, dtype=torch.float32, device=DEV)
        out.index_copy_(0, dup[: src.shape[0]], src)
        return out

    try:
        ctrl = run_dup()
        ctrl_bad = sum(1 for _ in range(10) if not torch.equal(run_dup(), ctrl))
        print(f"### control (duplicate index, last-writer-wins): {ctrl_bad}/10 differ")
    except Exception as exc:  # noqa: BLE001
        print(f"### control skipped: {type(exc).__name__}: {exc}")

    print("### RESULT:", "SAFE" if differing == 0 else "UNSAFE — replace index_copy_ writes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
