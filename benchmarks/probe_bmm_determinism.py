"""Is the padded ``bmm`` that §70 introduced bit-deterministic on this NPU?  ~30 s, no server.

§74 cleared ``index_copy_`` (deterministic with a permutation; the probe's control detected
differences), so the remaining suspicion for the 0.036 nats self-jitter (§73) is the *arithmetic*
§70 changed: one ``bmm`` over ``[256, max_n, hidden] x [256, hidden, 2*inter]`` instead of a
per-expert matmul.  Large batched GEMMs use split-K reductions on several backends, and a split-K
atomic accumulate is exactly the ``index_add_`` class of problem.

Three shapes, because the answer may depend on ``max_n``:
  max_n = 1   (decode, one token)
  max_n = 8   (decode, eight tokens — the gate's boundary)
  max_n = 32  (prefill-like padding, what the host-read path uses)
"""

import sys

import torch

import torch_npu  # noqa: F401

DEV = torch.device("npu", 0)
DT = torch.bfloat16
EXPERTS, HIDDEN, INTER = 256, 2048, 768
TRIALS = 20


def _check(label, fn, reference=None):
    first = fn()
    differing = 0
    worst = 0.0
    for _ in range(TRIALS):
        got = fn()
        if not torch.equal(got, first):
            differing += 1
            worst = max(worst, float((got.float() - first.float()).abs().max()))
    verdict = "DETERMINISTIC" if differing == 0 else f"NON-DETERMINISTIC {differing}/{TRIALS}"
    extra = ""
    if reference is not None:
        delta = float((first.float() - reference.float()).abs().max())
        extra = f"  |bmm - matmul|={delta:.3e}"
    print(f"### {label:34s} {verdict}  worst_abs={worst:.3e}{extra}")
    return differing


def main() -> int:
    torch.manual_seed(0)
    w = torch.randn(EXPERTS, 2 * INTER, HIDDEN, dtype=DT, device=DEV) * 0.02
    wt = w.transpose(1, 2)
    total = 0
    for max_n in (1, 8, 32):
        x = torch.randn(EXPERTS, max_n, HIDDEN, dtype=DT, device=DEV) * 0.2
        total += _check(f"bmm  [256,{max_n},2048]x[...]", lambda x=x: torch.bmm(x, wt))
        # Reference: the same arithmetic through per-expert matmul, to show the two forms agree to
        # fp32-reassociation level (a large gap would mean a real difference, not just ordering).
        ref = torch.stack([x[e] @ wt[e] for e in range(0, EXPERTS, 32)], dim=0)
        got = torch.stack([torch.bmm(x[e : e + 1], wt[e : e + 1])[0] for e in range(0, EXPERTS, 32)], dim=0)
        delta = float((got.float() - ref.float()).abs().max())
        print(f"###   max_n={max_n}: bmm vs matmul max_abs={delta:.3e}")
    print("### RESULT:", "SAFE" if total == 0 else "UNSAFE — the padded bmm is not reproducible")
    return 0


if __name__ == "__main__":
    sys.exit(main())
