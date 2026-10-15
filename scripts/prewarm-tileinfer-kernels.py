#!/usr/bin/env python3
"""Pre-compile the TileInfer kernels before a server starts, so the engine only ever sees cache hits.

Why this exists: a TileLang compile takes ~70-100 s and **holds the Python interpreter**.  Doing it
inside a serving process therefore blocks that process - in vLLM it stops the EngineCore from
answering, which looks like the engine hanging.  Compiling here (a separate process, before the
server is launched) populates `TILELANG_CACHE_DIR`; the server's own compile call then hits the
cache and returns at once.

T6: the "what to compile" set is derived from the *server config* by
``tileinfer.integrations.prewarm`` - the same pure functions the engine uses - so the pre-warm can
no longer drift from the engine (it used to be: decode buckets hand-typed, prefill not covered at
all, the pool size copied out of a log).  Pass the same values the server will use; both sides then
agree by construction.  Target contract: **zero compiles after start-up**.

Pool capacity: the kernel is compiled for the ABI's *declared* pool capacity, which is quantized up
from the value you pass (next power of two; or pin it exactly with ``TILEINFER_POOL_CAP``).  This is
deliberate (L2 pool-agnostic ABI): the server's pool comes from a vLLM memory snapshot and drifts
run to run (9425 vs 9899 pages), so pre-warm only needs to land in the same capacity bucket as the
server -- pass either run's pool size, or pin ``TILEINFER_POOL_CAP`` to the same value on both sides
for an exact match.

Decode example (Qwen2.5-1.5B):

    TILELANG_CACHE_DIR=/tmp/tl_cache_vllm \\
        python scripts/prewarm-tileinfer-kernels.py \\
        --kv-heads 2 --group 6 --dim 128 --num-engine-pages 9425 --max-num-reqs 32

Prefill too (Qwen3.6-35B-A3B, mixed GDN + d256, block 2048):

    TILELANG_CACHE_DIR=/tmp/tl_cache_vllm \\
        python scripts/prewarm-tileinfer-kernels.py \\
        --kv-heads 2 --group 8 --dim 256 --block-size 2048 \\
        --num-engine-pages 2288 --max-num-reqs 32 \\
        --max-num-batched-tokens 8192 --max-model-len 32768 --prefill

Add ``--prefill-kv-tile-pages 64`` to also build the split-prefill ladder (H19/H20).
"""
import argparse
import os
import time

from tileinfer.integrations.prewarm import (
    KERNEL_PAGE_SIZE,
    build_prewarm_plan,
    prewarm_plan_to_dict,
    sub_pages,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kv-heads", type=int, required=True)
    ap.add_argument("--group", type=int, required=True, help="query heads per KV head")
    ap.add_argument("--dim", type=int, default=128, help="head dimension")
    ap.add_argument("--page-size", type=int, default=KERNEL_PAGE_SIZE)
    ap.add_argument("--block-size", type=int, default=KERNEL_PAGE_SIZE,
                    help="the engine's block size (tokens); a multiple of 128 (GDN models use 2048)")
    ap.add_argument(
        "--num-engine-pages",
        type=int,
        required=True,
        help="the server's KV pool in *engine* pages (quantized up to the ABI's declared capacity)",
    )
    ap.add_argument(
        "--buckets",
        type=int,
        nargs="+",
        default=None,
        help="batch buckets to compile (default: the engine's BATCH_BUCKETS up to --max-num-reqs)",
    )
    # prefill ladder (opt-in; off by default so the decode-only recipe keeps working unchanged)
    ap.add_argument("--prefill", action="store_true", help="also compile the prefill variant ladder")
    ap.add_argument("--max-num-reqs", type=int, default=0, help="scheduler max_num_seqs (buckets/ladder limit)")
    ap.add_argument("--max-num-batched-tokens", type=int, default=0)
    ap.add_argument("--max-model-len", type=int, default=0)
    ap.add_argument("--prefill-kv-tile-pages", type=int, default=0,
                    help="split-prefill kv_tile_pages (0 = plain ladder only)")
    ap.add_argument("--prefill-max-tiles", type=int, default=4096)
    ap.add_argument("--prefill-split-ms", type=int, default=16,
                    help="the split path's compile-time max_splits (SPLIT_MS_COMPILE)")
    ap.add_argument(
        "--dtype",
        default="bfloat16",
        help="the tensor dtype the engine will use; part of the compile key, so it must match",
    )
    ap.add_argument(
        "--plan-only",
        action="store_true",
        help="do not compile; just print the derived pre-warm plan (CPU-only, for CI/ops diffing)",
    )
    ap.add_argument("--json", action="store_true", help="with --plan-only, emit the plan as JSON")
    args = ap.parse_args()

    from tileinfer.integrations.vllm_ascend import BATCH_BUCKETS

    buckets = tuple(args.buckets) if args.buckets else BATCH_BUCKETS
    pin = os.environ.get("TILEINFER_POOL_CAP", "").strip() or None
    plan = build_prewarm_plan(
        num_kv_heads=args.kv_heads,
        group_size=args.group,
        head_size=args.dim,
        dtype=args.dtype,
        block_size=args.block_size,
        engine_pages=args.num_engine_pages,
        max_num_reqs=args.max_num_reqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=args.max_model_len,
        buckets=buckets,
        prefill_max_tiles=args.prefill_max_tiles,
        prefill_kv_tile_pages=args.prefill_kv_tile_pages if args.prefill else 0,
        split_ms_compile=args.prefill_split_ms,
        pool_pages_override=int(pin) if pin else None,
    )

    print(f"TILELANG_CACHE_DIR={os.environ.get('TILELANG_CACHE_DIR', '(unset)')}")
    print(f"sub_pages={sub_pages(args.block_size)} {plan.summary()}")

    if args.plan_only:
        import json

        if args.json:
            print(json.dumps(prewarm_plan_to_dict(plan), indent=2, sort_keys=True))
        else:
            print("decode buckets:", plan.decode)
            print("prefill variants:", len(plan.prefill),
                  f"(plain {sum(1 for s in plan.prefill if len(s) == 9)}"
                  f" / split {sum(1 for s in plan.prefill if len(s) == 10)})")
            for s in plan.prefill:
                print("  prefill spec:", s)
        print("plan-only: no compilation performed")
        return 0

    from tileinfer.kernels.attention.paged_decode_ascend950 import build_decode_kernel

    for batch in plan.decode:
        t0 = time.time()
        build_decode_kernel(
            batch=batch,
            kv_heads=args.kv_heads,
            group=args.group,
            dim=args.dim,
            page_size=args.page_size,
            num_pages_cap=plan.pool_pages,
            dtype=args.dtype,
        )
        print(f"decode bucket={batch} compiled in {time.time() - t0:.0f}s", flush=True)

    if args.prefill:
        from tileinfer.kernels.attention.paged_decode_ascend950 import (
            build_prefill_kernel,
            build_prefill_split_frontier_kernel,
            build_prefill_split_head_kernel,
        )

        for spec in plan.prefill:
            t0 = time.time()
            if len(spec) == 9:  # plain prefill variant
                (batch, kvh, grp, dim, page, pages, tiles, block_q, dt) = spec
                build_prefill_kernel(
                    batch=batch, kv_heads=kvh, group=grp, dim=dim, page_size=page,
                    num_pages_cap=pages, num_tiles=tiles, block_q=block_q, dtype=dt,
                )
                tag = f"tiles={tiles}"
            else:  # split-KV prefill pair (head + frontier)
                (batch, kvh, grp, dim, page, pages, tiles, block_q, dt, max_splits) = spec
                build_prefill_split_head_kernel(
                    batch=batch, kv_heads=kvh, group=grp, dim=dim, page_size=page,
                    num_pages_cap=pages, num_tiles=tiles, max_splits=max_splits,
                    num_slots=tiles + 1, block_q=block_q, dtype=dt,
                )
                build_prefill_split_frontier_kernel(
                    batch=batch, kv_heads=kvh, group=grp, dim=dim, page_size=page,
                    num_pages_cap=pages, num_tiles=tiles, max_splits=max_splits,
                    num_slots=tiles + 1, block_q=block_q, dtype=dt,
                )
                tag = f"tiles={tiles} split(head+frontier,ms={max_splits})"
            print(f"prefill {tag} compiled in {time.time() - t0:.0f}s", flush=True)

    print("pre-warm done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
