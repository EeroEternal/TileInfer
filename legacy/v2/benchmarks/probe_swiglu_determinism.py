"""Is the fused ``npu_swiglu`` bit-deterministic on this NPU?  ~30 s, no server.

The last unprobed operator on the default decode path (§73/§84.4).  §74 and §76 cleared
``index_copy_`` and the padded ``bmm``, yet the default state still fails the 1e-6 self-jitter check
(0.036 nats), so the remaining candidate is the fused activation: ``_swiglu`` routes to
``torch_npu.npu_swiglu`` whenever the symbol exists and the tensor is not on CPU.

A fused SiLU-mul kernel is a plausible place for a non-deterministic reduction order, and it would
also explain why graph capture amplified the difference (§44).

Probe shape is the MoE gate/up layout: ``[E, n, 2*inter]`` — the same tensor the expert path feeds it.
"""

import sys

import torch

import torch_npu  # noqa: F401

DEV = torch.device("npu", 0)
DT = torch.bfloat16
TRIALS = 20


def main() -> int:
    from tileinfer.vllm.ops.moe import _swiglu, _npu_swiglu_fn, silu_and_mul

    fn = _npu_swiglu_fn()
    print("### npu_swiglu available:", fn is not None)
    if fn is None:
        print("### RESULT: NOT APPLICABLE (no fused symbol; the torch port is used)")
        return 0

    torch.manual_seed(0)
    total_bad = 0
    for shape in ((8, 1, 1536), (8, 8, 1536), (256, 1, 1536)):
        x = torch.randn(*shape, dtype=DT, device=DEV) * 0.2
        fused_first = _swiglu(x)
        port_first = silu_and_mul(x)
        bad_fused = 0
        bad_port = 0
        worst_f = 0.0
        worst_p = 0.0
        for _ in range(TRIALS):
            got = _swiglu(x)
            if not torch.equal(got, fused_first):
                bad_fused += 1
                worst_f = max(worst_f, float((got.float() - fused_first.float()).abs().max()))
            gotp = silu_and_mul(x)
            if not torch.equal(gotp, port_first):
                bad_port += 1
                worst_p = max(worst_p, float((gotp.float() - port_first.float()).abs().max()))
        cross = float((fused_first.float() - port_first.float()).abs().max())
        total_bad += bad_fused + bad_port
        print(
            f"### shape {shape}: fused {bad_fused}/{TRIALS} differ (worst {worst_f:.3e}) | "
            f"torch port {bad_port}/{TRIALS} differ (worst {worst_p:.3e}) | fused vs port {cross:.3e}"
        )

    print("### RESULT:", "SAFE" if total_bad == 0 else "NON-DETERMINISTIC — switch the fused swiglu off")
    return 0


if __name__ == "__main__":
    sys.exit(main())
