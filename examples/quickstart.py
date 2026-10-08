"""TileInfer quickstart — decode attention over a paged KV cache.

Runs anywhere: on a CPU-only machine it uses the portable ``reference`` backend (slow but
correct); on Ascend with a TileLang toolchain it uses the ``tilelang`` kernel.

    python examples/quickstart.py
    python examples/quickstart.py --device npu --backend tilelang
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tileinfer import BatchAttention  # noqa: E402
from tileinfer.metadata import PageTable  # noqa: E402
from tileinfer.testing.cache import build_paged_cache, random_q  # noqa: E402
from tileinfer.testing.reference import reference_attention  # noqa: E402
from tileinfer.utils import device_name, resolve_device  # noqa: E402

# A small Llama-ish GQA configuration.
BATCH = 4
NUM_QO_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
PAGE_SIZE = 128
KV_LENS = [2048, 130, 1024, 512]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="auto", help="auto | tilelang | reference")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--load-balance", action="store_true", help="enable LPT scheduling")
    parser.add_argument("--kv-tile-pages", type=int, default=0, help="split-KV tile size in pages")
    args = parser.parse_args(argv)

    device = resolve_device(args.device)
    dtype = torch.float16
    print(f"device      : {device_name(device)}")
    print(f"backends    : {BatchAttention.available_backends(available_only=True)}")

    # ---------------------------------------------------------------- build a batch
    max_kv = max(KV_LENS)
    k = torch.randn(BATCH, NUM_KV_HEADS, max_kv, HEAD_DIM, dtype=dtype, device=device)
    v = torch.randn_like(k)
    k_cache, v_cache, meta = build_paged_cache(
        k, v, page_size=PAGE_SIZE, shuffle_pages=True, seed=0, device=device
    )
    # trim each request to its own length: ragged, and not a multiple of the page size
    indptr = [0]
    for length in KV_LENS:
        indptr.append(indptr[-1] + -(-length // PAGE_SIZE))
    meta = type(meta)(
        kv_indptr=torch.tensor(indptr, dtype=torch.int32, device=device),
        kv_indices=meta.kv_indices,
        kv_last_page_len=torch.tensor(
            [PAGE_SIZE if L % PAGE_SIZE == 0 else L % PAGE_SIZE for L in KV_LENS],
            dtype=torch.int32,
            device=device,
        ),
        page_size=PAGE_SIZE,
    )
    q = random_q(BATCH, NUM_QO_HEADS, 1, HEAD_DIM, dtype=dtype, device=device)

    # ---------------------------------------------------------------- plan (once)
    attn = BatchAttention(backend=args.backend, device=device, dtype=dtype)
    plan = attn.plan(
        kv_indptr=meta.kv_indptr,
        kv_indices=meta.kv_indices,
        kv_last_page_len=meta.kv_last_page_len,
        page_size=PAGE_SIZE,
        num_qo_heads=NUM_QO_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        load_balance=args.load_balance,
        kv_tile_pages=args.kv_tile_pages,
        k_cache=k_cache,
        v_cache=v_cache,
    )
    print(f"backend     : {attn.backend_name}")
    print(f"plan        : {plan.summary()}")

    # ---------------------------------------------------------------- run (many times)
    out = attn.run(q, k_cache, v_cache, plan=plan)
    print(f"output      : shape={tuple(out.shape)} dtype={out.dtype}")

    if args.backend != "reference":
        expected = reference_attention(q, k_cache, v_cache, plan.meta, causal=False)
        err = (out.float() - expected.float()).abs().max().item()
        print(f"max abs err : {err:.3e} (vs torch reference)")
        assert err < 5e-2, "kernel output disagrees with the reference"

    # A dense engine-side page table is accepted just as well.
    table: PageTable = plan.meta.to_page_table()
    plan_from_table = BatchAttention(backend=args.backend, device=device).plan_from_page_table(
        table.table,
        table.seq_lens,
        page_size=PAGE_SIZE,
        num_qo_heads=NUM_QO_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        causal=False,
    )
    out2 = attn.run(q, k_cache, v_cache, plan=plan_from_table)
    same = torch.allclose(out.float(), out2.float(), atol=1e-3)
    print(f"page-table  : ragged and dense metadata agree: {same}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
