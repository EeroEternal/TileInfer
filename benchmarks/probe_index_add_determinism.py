"""Is ``index_add_`` bit-deterministic on this NPU?

§44: our arm was non-deterministic between two identical requests (same server, greedy, 24 tokens),
while the vllm-ascend arm was bit-exact (delta 0 on all four prompts).  Our MoE combine is the only
place in our path that accumulates with duplicate indices -- an atomic add on most accelerators, and
atomics have no defined summation order.  So test the primitive directly rather than bisecting the
model (turning our takeovers off is not possible: vLLM's own GDN/MoE need Triton).

Controls alongside it: ``scatter_add_`` (also atomic), and a reshape+``sum`` over the same rows (a
reduction with a fixed order, expected to be deterministic).
"""

import torch

import torch_npu  # noqa: F401

DEV = torch.device("npu", 0)
TOKENS, TOP_K, HIDDEN = 4, 8, 256
TRIALS = 50


def main() -> int:
    torch.manual_seed(0)
    slots = TOKENS * TOP_K
    src = torch.randn(slots, HIDDEN, dtype=torch.float32, device=DEV) * 0.1
    idx = torch.arange(TOKENS, device=DEV, dtype=torch.int64).repeat_interleave(TOP_K)

    def run_index_add():
        out = torch.zeros(TOKENS, HIDDEN, dtype=torch.float32, device=DEV)
        out.index_add_(0, idx, src)
        return out

    def run_scatter_add():
        out = torch.zeros(TOKENS, HIDDEN, dtype=torch.float32, device=DEV)
        out.scatter_add_(0, idx.unsqueeze(1).expand(slots, HIDDEN), src)
        return out

    def run_reshape_sum():
        return src.view(TOKENS, TOP_K, HIDDEN).sum(dim=1)

    for name, fn in (
        ("index_add_", run_index_add),
        ("scatter_add_", run_scatter_add),
        ("reshape+sum", run_reshape_sum),
    ):
        first = fn()
        worst = 0.0
        differing = 0
        for _ in range(TRIALS):
            got = fn()
            if not torch.equal(got, first):
                differing += 1
                worst = max(worst, float((got - first).abs().max()))
        verdict = "DETERMINISTIC" if differing == 0 else f"NON-DETERMINISTIC in {differing}/{TRIALS}"
        print(f"### {name:12s} {verdict}  worst_abs={worst:.3e}", flush=True)

    # And the real question: does the difference survive into a token decision?
    ref = run_reshape_sum()
    worst_gap = 0.0
    for _ in range(20):
        got = run_index_add()
        gap = float((got - ref).abs().max())
        worst_gap = max(worst_gap, gap)
    print(f"### index_add_ vs reshape+sum: worst_abs={worst_gap:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
