"""Minimal cube -> vector hand-off reproducer, for the PTO target on Ascend 950PR.

This exists because it is the smallest program that reproduces the failure that currently blocks
*every* attention kernel on our reference machine — including the two upstream PTO attention
examples in `tile-ai/tilelang-ascend`:

    GEMM (cube) -> L0C -> GM workspace -> UB (vector) -> add -> output
    => `AclrtSynchronizeDeviceWithTimeout ... 507014` (aicore timeout)

Cube-only kernels (e.g. upstream `examples/gemm/example_gemm_pto_developer.py`) and vector-only
kernels pass, so the failure is specific to the cube→vector hand-off under the PTO target.

Run it on an Ascend machine with the source-built toolchain:

    source <env with tilelang-ascend on PYTHONPATH>
    python benchmarks/probes/pto_cv_handoff.py

Exit code 0 means the hand-off works, 1 means it hung (or produced wrong numbers) — which makes it
usable as a canary in CI on a device runner.
"""

from __future__ import annotations

import sys

import torch

try:
    import torch_npu  # noqa: F401  (registers the device backend)
except Exception as exc:  # pragma: no cover
    print(f"torch_npu unavailable: {exc}")
    sys.exit(2)

import tilelang as tl  # noqa: E402
from tilelang import language as T  # noqa: E402

PASS_CONFIGS = {
    tl.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tl.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tl.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tl.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tl.jit(out_idx=[2], workspace_idx=[3], target="pto", pass_configs=PASS_CONFIGS)
def cv_handoff(m: int, n: int, k: int):
    """``C = A @ B + 1`` — the ``+1`` must happen on the vector core, after the cube wrote L0C."""

    @T.prim_func
    def main(
        A: T.Tensor([m, k], "float16"),  # type: ignore
        B: T.Tensor([k, n], "float16"),  # type: ignore
        C: T.Tensor([m, n], "float16"),  # type: ignore
        ws: T.Tensor([m, n], "float32"),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_l1 = T.alloc_L1([m, k], "float16")
            b_l1 = T.alloc_L1([k, n], "float16")
            c_l0c = T.alloc_L0C([m, n], "float32")
            u = T.alloc_ub([m, n], "float32")
            o = T.alloc_ub([m, n], "float16")

            T.copy(A, a_l1)
            T.copy(B, b_l1)
            T.gemm_v0(a_l1, b_l1, c_l0c, init=True)
            T.copy(c_l0c, ws)  # cube -> GM
            T.copy(ws, u)  # GM -> vector
            T.tile.add(u, u, 1.0)
            T.copy(u, o)
            T.copy(o, C)

    return main


def main() -> int:
    m = n = k = 64
    torch.manual_seed(0)
    a = torch.randn(m, k, dtype=torch.float16, device="npu")
    b = torch.randn(k, n, dtype=torch.float16, device="npu")
    try:
        kernel = cv_handoff(m, n, k)
        c = kernel(a, b)
        torch.npu.synchronize()
    except Exception as exc:  # noqa: BLE001 - we want the message, not a traceback
        message = next((l for l in str(exc).splitlines() if "error" in l.lower()), str(exc)[:120])
        print(f"FAIL: cube->vector hand-off did not complete: {message.strip()[:160]}")
        return 1

    expected = (a.float() @ b.float()) + 1.0
    diff = (c.float() - expected).abs().max().item()
    status = "PASS" if diff < 0.5 else "FAIL"
    print(f"{status}: cube->vector hand-off max|diff| = {diff:.4f}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
