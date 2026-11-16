"""Kernel attention vs reference attention on the SAME inputs, step by step.

§50: our arm is deterministic and differs from vllm-ascend by up to 0.21 nats, with a token flip at
step 22.  Two candidate sources:

* our kernel stores the softmax probabilities as **bf16** before the PV product (``p_ub`` is the
  compute dtype) while the torch reference keeps fp32 throughout -- a plausible origin for a slow
  drift that eventually flips a near-tie;
* something outside attention.

This isolates the first question without a server: build *both* attention implementations in one
process, feed them identical inputs for 24 decode steps, and print the per-step max |delta| in the
hidden state.  What matters is the *shape* of the curve: a constant tiny delta is rounding, a growing
one is drift in the same direction (which is what turns into a different token).
"""

import os
import sys

import torch

import torch_npu  # noqa: F401

DT = torch.bfloat16
NH, NKV, D = 16, 2, 256
BLK = 2048
BLOCKS = 32
SEQ = 512
STEPS = 24


def build_impl(path_switch: str):
    """One impl for the kernel path, one for the torch reference, from the *same* process."""
    if path_switch == "kernel":
        os.environ["TILEINFER_ATTN_BACKEND"] = "tilelang-ascend950"
    else:
        os.environ["TILEINFER_ATTN_BACKEND"] = "reference"
    from importlib import reload

    import tileinfer.vllm.attn.impl as impl_mod

    reload(impl_mod)  # the backend name is read at construction/import time
    return impl_mod.TileInferAttentionImpl(NH, D, D**-0.5, num_kv_heads=NKV), impl_mod


def main() -> int:
    from tileinfer.serving.buckets import fused_halves
    from tileinfer.vllm.attn.kv_cache import write_kv_cache
    from tileinfer.vllm.attn.metadata import TileInferMetadata

    dev = torch.device("npu", 0)
    torch.manual_seed(0)

    max_pages = (SEQ + BLK - 1) // BLK
    table = torch.zeros(1, max_pages, dtype=torch.int32, device=dev)
    for p in range(max_pages):
        table[0, p] = p
    slots = torch.tensor(
        [int(table[0, p // BLK]) * BLK + (p % BLK) for p in range(SEQ)],
        dtype=torch.int64,
        device=dev,
    )
    empty_pool = torch.zeros(2, BLOCKS, BLK, NKV, D, dtype=DT, device=dev).permute(1, 0, 2, 3, 4)
    k_all = torch.randn(SEQ, NKV, D, dtype=DT, device=dev) * 0.2
    v_all = torch.randn(SEQ, NKV, D, dtype=DT, device=dev) * 0.2

    # two identical pools: each implementation writes its own step's K/V, so they must not share
    pools = []
    for _ in range(2):
        pool = empty_pool.clone()
        kh, vh = fused_halves(pool)
        write_kv_cache(k_all, v_all, kh, vh, slots)
        pools.append(pool)
    torch.npu.synchronize()

    kernel_impl, kernel_mod = build_impl("kernel")
    reference_impl, reference_mod = build_impl("reference")
    if kernel_impl._backend_name != "tilelang-ascend950":
        print("### WARNING: kernel impl built with backend", kernel_impl._backend_name)

    qs = [torch.randn(1, NH, D, dtype=DT, device=dev) * 0.2 for _ in range(STEPS)]
    ks = [torch.randn(1, NKV, D, dtype=DT, device=dev) * 0.2 for _ in range(STEPS)]
    vs = [torch.randn(1, NKV, D, dtype=DT, device=dev) * 0.2 for _ in range(STEPS)]

    deltas = []
    for step in range(STEPS):
        pos = SEQ + step
        slot = int(table[0, pos // BLK]) * BLK + (pos % BLK)
        qsl = torch.tensor([0, 1], dtype=torch.int32, device=dev)
        meta = TileInferMetadata(
            query_start_loc=qsl,
            query_start_loc_cpu=qsl.cpu(),
            seq_lens=torch.tensor([pos + 1], dtype=torch.int32, device=dev),
            block_table_tensor=table,
            slot_mapping=torch.tensor([slot], dtype=torch.int64, device=dev),
            num_reqs=1,
            num_actual_tokens=1,
            max_query_len=1,
            bucket=1,
            block_size=BLK,
        )
        out_k = torch.zeros(1, NH * D, dtype=DT, device=dev)
        out_r = torch.zeros_like(out_k)
        kernel_impl.forward(None, qs[step], ks[step], vs[step], pools[0], meta, out_k)
        reference_impl.forward(None, qs[step], ks[step], vs[step], pools[1], meta, out_r)
        torch.npu.synchronize()
        d = float((out_k.float() - out_r.float()).abs().max())
        deltas.append(d)
        if step < 4 or step % 8 == 0 or step == STEPS - 1:
            print(f"### step {step:2d}: max|kernel - reference| = {d:.6f}")
    print(f"### first/last/max delta: {deltas[0]:.6f} / {deltas[-1]:.6f} / {max(deltas):.6f}")
    growing = deltas[-1] > deltas[0] * 2
    print("### RESULT:", "DRIFT (grows)" if growing else "constant rounding level")
    return 0


if __name__ == "__main__":
    sys.exit(main())
