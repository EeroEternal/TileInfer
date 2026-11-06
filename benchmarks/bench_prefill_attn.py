"""Prefill attention in isolation: kernel path vs reference path vs full torch, 512 tokens.

§59.4 put attention prefill at ~327 ms of the 662 ms marginal cost — now the largest single block,
ahead of the GDN chain (253 ms).  Before changing anything: how much of that is the kernel itself,
how much the host-side path around it, and how does the reference implementation compare on the same
shapes?  The answer decides whether the work is in the tilelang kernel, in `PrefillPath`'s per-step
schedule, or not in attention at all.
"""

import os
import sys
import time

import torch

import torch_npu  # noqa: F401

DT = torch.bfloat16
NH, NKV, D = 16, 2, 256
BLK = 2048
BLOCKS = 32
T = 512
ITERS = 3


def _bench(label, fn):
    try:
        fn()
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(ITERS):
            out = fn()
        torch.npu.synchronize()
        ms = (time.perf_counter() - t0) * 1000.0 / ITERS
        print(f"### {label:44s} {ms:8.2f} ms")
        return ms
    except Exception as exc:  # noqa: BLE001
        print(f"### {label:44s}       -  ({type(exc).__name__}: {exc})")
        return None


def build(switch: str):
    os.environ.pop("TILEINFER_ATTN_BACKEND", None)
    os.environ.pop("TILEINFER_PREFILL", None)
    if switch == "kernel":
        pass
    elif switch == "ref_prefill":
        os.environ["TILEINFER_PREFILL"] = "0"
    elif switch == "ref_attn":
        os.environ["TILEINFER_ATTN_BACKEND"] = "reference"
    from importlib import reload

    import tileinfer.vllm.attn.impl as impl_mod

    reload(impl_mod)
    return impl_mod.TileInferAttentionImpl(NH, D, D**-0.5, num_kv_heads=NKV), impl_mod


def main() -> int:
    from tileinfer.vllm.attn.metadata import TileInferMetadata

    dev = torch.device("npu", 0)
    torch.manual_seed(0)
    pages = (T + BLK - 1) // BLK
    table = torch.zeros(1, pages, dtype=torch.int32, device=dev)
    for p in range(pages):
        table[0, p] = p
    slots = torch.tensor(
        [int(table[0, i // BLK]) * BLK + (i % BLK) for i in range(T)],
        dtype=torch.int64,
        device=dev,
    )
    kv_cache = torch.zeros(2, BLOCKS, BLK, NKV, D, dtype=DT, device=dev).permute(1, 0, 2, 3, 4)
    q = torch.randn(T, NH, D, dtype=DT, device=dev) * 0.2
    k = torch.randn(T, NKV, D, dtype=DT, device=dev) * 0.2
    v = torch.randn(T, NKV, D, dtype=DT, device=dev) * 0.2
    qsl = torch.tensor([0, T], dtype=torch.int32, device=dev)
    meta = TileInferMetadata(
        query_start_loc=qsl,
        query_start_loc_cpu=qsl.cpu(),
        seq_lens=torch.tensor([T], dtype=torch.int32, device=dev),
        block_table_tensor=table,
        slot_mapping=slots,
        num_reqs=1,
        num_actual_tokens=T,
        max_query_len=T,
        bucket=1,
        block_size=BLK,
    )

    results = {}
    for switch in ("kernel", "ref_prefill", "ref_attn"):
        impl, mod = build(switch)
        results[switch] = _bench(
            f"prefill T={T} [{switch}]",
            lambda impl=impl: impl.forward(None, q, k, v, kv_cache, meta, torch.zeros(T, NH * D, dtype=DT, device=dev)),
        )
        del impl

    # What a 512-token attention layer *should* cost, for scale: 2 * T^2/2 * H * D * 2 flops.
    flops = 2 * (T * (T + 1) // 2) * NH * D * 2
    print(f"### causal attention math: {flops / 1e9:.2f} GFLOP per layer")
    for name, ms in results.items():
        if ms:
            print(f"### {name}: {ms:.2f} ms -> {flops / (ms * 1e-3) / 1e12:.2f} TFLOP/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
