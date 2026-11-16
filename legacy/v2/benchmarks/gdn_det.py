"""Is the GDN decode chain bit-reproducible?  Same shape of test as ``decode_det.py``, ~30 s.

§44/§49: a single decode attention step is bit-identical over 5 runs, and ``fused_experts`` is
bit-identical over 20.  So the non-reproducibility of the serving arm is *not* in either of those.
The remaining pieces of a Qwen3.6 decode step are the GDN chain (conv update + recurrent delta rule)
and the sampler.  This probes the GDN pair with fixed inputs and a state cloned from the same
template for every run -- both mutate in place, so the clone is what makes the comparison meaningful.
"""

import sys

import torch

import torch_npu  # noqa: F401

DT = torch.bfloat16
H, HV, K, V = 16, 32, 128, 128          # Qwen3.6 linear-attention config
ROWS = 8
WIDTH = 4
RUNS = 20


def main() -> int:
    from tileinfer.vllm.ops.fla.causal_conv1d import causal_conv1d_update
    from tileinfer.vllm.ops.fla.recurrent import fused_recurrent_gated_delta_rule_packed_decode

    dev = torch.device("npu", 0)
    torch.manual_seed(0)

    # ---- conv update -------------------------------------------------------------------------
    conv_state = torch.randn(ROWS, 8192, WIDTH - 1, dtype=DT, device=dev) * 0.2
    weight = torch.randn(8192, WIDTH, dtype=DT, device=dev) * 0.1
    bias = torch.randn(8192, dtype=DT, device=dev) * 0.1
    x = torch.randn(1, 8192, dtype=DT, device=dev) * 0.2
    idx = torch.tensor([3], dtype=torch.int32, device=dev)
    conv_template = conv_state.clone()

    outs = []
    for _ in range(RUNS):
        state = conv_template.clone()
        outs.append(causal_conv1d_update(x, state, weight, bias, "silu", conv_state_indices=idx).clone())
    torch.npu.synchronize()
    bad = sum(1 for o in outs[1:] if not torch.equal(o, outs[0]))
    worst = max((float((o.float() - outs[0].float()).abs().max()) for o in outs[1:]), default=0.0)
    print(f"### causal_conv1d_update: {bad}/{RUNS - 1} differ, worst_abs={worst:.3e}")

    # ---- recurrent delta rule ----------------------------------------------------------------
    mixed = torch.randn(1, 2 * H * K + HV * V, dtype=DT, device=dev) * 0.2
    a = torch.randn(1, HV, dtype=DT, device=dev) * 0.2
    b = torch.randn(1, HV, dtype=DT, device=dev) * 0.2
    A_log = torch.randn(HV, dtype=torch.float32, device=dev) * 0.1
    dt_bias = torch.randn(HV, dtype=torch.float32, device=dev) * 0.1
    st = torch.randn(ROWS, HV, V, K, dtype=DT, device=dev) * 0.1
    sidx = torch.tensor([3], dtype=torch.int32, device=dev)
    state_template = st.clone()

    rots = []
    states = []
    for _ in range(RUNS):
        s = state_template.clone()
        out = torch.zeros(1, 1, HV, V, dtype=DT, device=dev)
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed, a, b, A_log, dt_bias, K**-0.5, s, out, sidx
        )
        rots.append(out.clone())
        states.append(s.clone())
    torch.npu.synchronize()
    bad_o = sum(1 for o in rots[1:] if not torch.equal(o, rots[0]))
    bad_s = sum(1 for o in states[1:] if not torch.equal(o, states[0]))
    worst_o = max((float((o.float() - rots[0].float()).abs().max()) for o in rots[1:]), default=0.0)
    print(f"### recurrent decode: out {bad_o}/{RUNS - 1} differ (worst_abs={worst_o:.3e}), "
          f"state {bad_s}/{RUNS - 1} differ")
    print("### RESULT:", "DETERMINISTIC" if (bad == 0 and bad_o == 0 and bad_s == 0) else "NON-DETERMINISTIC")
    return 0


if __name__ == "__main__":
    sys.exit(main())
