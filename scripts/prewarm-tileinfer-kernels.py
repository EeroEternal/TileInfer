#!/usr/bin/env python3
"""Pre-compile the decode kernels before a server starts, so the engine only ever sees cache hits.

Why this exists: a TileLang compile takes ~70 s and **holds the Python interpreter**.  Doing it inside
a serving process therefore blocks that process - in vLLM it stops the EngineCore from answering, which
looks like the engine hanging.  Compiling here (a separate process, before the server is launched)
populates `TILELANG_CACHE_DIR`; the server's own compile call then hits the cache and returns at once.

The kernel key includes the KV pool capacity, so pass the capacity the server will use.  The pool size
is printed by vLLM-Ascend at startup ("GPU KV cache size"), or it can be read from a previous run's log.

    TILELANG_CACHE_DIR=/tmp/tl_cache_vllm \\
        python scripts/prewarm-tileinfer-kernels.py \\
        --kv-heads 2 --group 6 --dim 128 --page-size 128 --num-pages-cap 9425 --buckets 1 2 4
"""
import argparse
import time

from tileinfer.kernels.attention.paged_decode_ascend950 import build_decode_kernel


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kv-heads", type=int, required=True)
    ap.add_argument("--group", type=int, required=True, help="query heads per KV head")
    ap.add_argument("--dim", type=int, default=128, help="head dimension")
    ap.add_argument("--page-size", type=int, default=128)
    ap.add_argument("--num-pages-cap", type=int, required=True, help="the server's KV pool, in pages")
    ap.add_argument("--buckets", type=int, nargs="+", default=[1, 2, 4], help="batch sizes to compile")
    args = ap.parse_args()

    import os

    print(f"TILELANG_CACHE_DIR={os.environ.get('TILELANG_CACHE_DIR', '(unset)')}")
    for batch in args.buckets:
        t0 = time.time()
        build_decode_kernel(
            batch=batch,
            kv_heads=args.kv_heads,
            group=args.group,
            dim=args.dim,
            page_size=args.page_size,
            num_pages_cap=args.num_pages_cap,
        )
        print(f"bucket={batch} compiled in {time.time() - t0:.0f}s", flush=True)
    print("pre-warm done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
