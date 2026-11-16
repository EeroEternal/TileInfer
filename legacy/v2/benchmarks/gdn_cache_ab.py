"""GDN decode: cached per-layer constants vs rebuilt every call.

Decode is dispatch-bound (§22), and this operator rebuilds three per-layer constants on every call:
``arange(HV) // (HV // H)``, ``A_log.to(fp32)`` and ``dt_bias.to(fp32)``.  The cache is keyed on the
source tensors' identity *and holds strong references to them*, so an entry cannot outlive its layer
and the key can never alias a recycled address.

Checked for equality as well as speed: the cached path must produce bit-identical output and state.
"""

import os
import sys
import time

import torch

import torch_npu  # noqa: F401

DEV = torch.device("npu", 0)
DT = torch.bfloat16
H, HV, K, V = 16, 32, 128, 128          # Qwen3.6-35B-A3B linear-attention config
ROWS = 256
SCALE = K**-0.5
FAILED = []


def _inputs(seed=0):
    torch.manual_seed(seed)
    mixed = torch.randn(1, 2 * H * K + HV * V, dtype=DT, device=DEV) * 0.2
    a = torch.randn(1, HV, dtype=DT, device=DEV) * 0.2
    b = torch.randn(1, HV, dtype=DT, device=DEV) * 0.2
    A_log = torch.randn(HV, dtype=torch.float32, device=DEV) * 0.1
    dt_bias = torch.randn(HV, dtype=torch.float32, device=DEV) * 0.1
    state = torch.randn(ROWS, HV, V, K, dtype=DT, device=DEV) * 0.1
    idx = torch.zeros(1, dtype=torch.int32, device=DEV)  # slot 0 is the "skip" slot (line 296-299)
    idx[0] = 7
    out = torch.zeros(1, 1, HV, V, dtype=DT, device=DEV)
    return mixed, a, b, A_log, dt_bias, state, idx, out


def _call(mixed, a, b, A_log, dt_bias, state, idx, out):
    from tileinfer.vllm.ops.fla.recurrent import fused_recurrent_gated_delta_rule_packed_decode

    return fused_recurrent_gated_delta_rule_packed_decode(
        mixed, a, b, A_log, dt_bias, SCALE, state, out, idx
    )


def correctness():
    mixed, a, b, A_log, dt_bias, state, idx, out = _inputs()
    results = {}
    for label, cache in (("cache=0", "0"), ("cache=1", "1")):
        os.environ["TILEINFER_GDN_CACHE"] = cache
        st = state.clone()
        o = torch.zeros_like(out)
        got, _ = _call(mixed, a, b, A_log, dt_bias, st, idx, o)
        torch.npu.synchronize()
        results[label] = (got.clone(), st.clone())
    same_out = torch.equal(results["cache=0"][0], results["cache=1"][0])
    same_state = torch.equal(results["cache=0"][1], results["cache=1"][1])
    print(f"### bit-identical output={same_out} state={same_state}")
    if not (same_out and same_state):
        FAILED.append("cache changes results")


def timing(iters=200):
    mixed, a, b, A_log, dt_bias, state, idx, out = _inputs()
    for label, cache in (("cache=0", "0"), ("cache=1", "1")):
        os.environ["TILEINFER_GDN_CACHE"] = cache
        _call(mixed, a, b, A_log, dt_bias, state, idx, out)
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            _call(mixed, a, b, A_log, dt_bias, state, idx, out)
        torch.npu.synchronize()
        us = (time.perf_counter() - t0) * 1e6 / iters
        globals().setdefault("_t", {})[label] = us
        print(f"### {label}: {us:8.1f} us/call")
    base, cached = _t["cache=0"], _t["cache=1"]
    print(f"### measured_ratio = {base / cached:.3f}")


if __name__ == "__main__":
    print(f"### device={torch.npu.get_device_name(0)}")
    correctness()
    timing()
    print("### RESULT:", "PASS" if not FAILED else f"FAIL {FAILED}")
    sys.exit(0 if not FAILED else 1)
