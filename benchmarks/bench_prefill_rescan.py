"""Micro-benchmark: prefill KV-rescan truncation (A/B against the full-scan baseline).

The prefill kernel truncates each query tile's page loop to its causal frontier, so the trailing
(masked) pages are never read.  This harness times the shipped (truncated) path and checks it
against the torch reference on every shape; set ``TILEINFER_PREFILL_NOSCAN_OFF=1`` (a debug-only
override that forces the full page scan) to reproduce the baseline the truncation replaced.

Usage::

    source .agent_loop/run/env.sh
    python benchmarks/bench_prefill_rescan.py                     # truncated (shipped)
    TILEINFER_PREFILL_NOSCAN_OFF=1 python benchmarks/bench_prefill_rescan.py   # baseline
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tileinfer.kernels.attention.paged_decode_ascend950 import forward_prefill  # noqa: E402
from tileinfer.metadata import RaggedMetadata  # noqa: E402
from tileinfer.plan import plan_query_tiles  # noqa: E402
from tileinfer.testing.reference import reference_attention  # noqa: E402
from tileinfer.utils import is_npu_available  # noqa: E402

DTYPE = torch.bfloat16
PAGE = 128
BLOCK_Q = 128
GROUP = 8

SHAPES = [
    ("8K prefill", [8192], [8192], 128),
    ("16K prefill", [16384], [16384], 128),
    ("32K prefill", [32768], [32768], 128),
    ("chunk 512 over 8192", [8704], [512], 128),
    ("chunk 2048 over 8192", [10240], [2048], 128),
]


def _build(batch, kv_heads, kv_lens, qo_lens, dim, seed=0):
    dev = torch.device("npu", 0)
    torch.manual_seed(seed)
    indptr, last = [0], []
    for L in kv_lens:
        indptr.append(indptr[-1] + -(-L // PAGE))
        last.append(PAGE if L % PAGE == 0 else L % PAGE)
    total = indptr[-1]
    k_cache = torch.randn(total, PAGE, kv_heads, dim, dtype=DTYPE, device=dev)
    v_cache = torch.randn(total, PAGE, kv_heads, dim, dtype=DTYPE, device=dev)
    indices = torch.arange(total, dtype=torch.int32, device=dev)
    qo = torch.tensor(qo_lens, dtype=torch.int32, device=dev)
    meta = RaggedMetadata(
        kv_indptr=torch.tensor(indptr, dtype=torch.int32, device=dev),
        kv_indices=indices,
        kv_last_page_len=torch.tensor(last, dtype=torch.int32, device=dev),
        page_size=PAGE,
        qo_indptr=torch.zeros(batch + 1, dtype=torch.int32, device=dev),
    )
    q = torch.randn(batch, kv_heads * GROUP, max(qo_lens), dim, dtype=DTYPE, device=dev) * 0.2
    return q, k_cache, v_cache, meta, qo


def _timed(fn, iters=10, warmup=3):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.npu.synchronize()
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples)


def run_case(label, kv_lens, qo_lens, dim, batch=1, kv_heads=2):
    q, k_cache, v_cache, meta, qo = _build(batch, kv_heads, kv_lens, qo_lens, dim)
    page_counts = (meta.kv_indptr[1:] - meta.kv_indptr[:-1]).to(torch.int64)
    schedule = plan_query_tiles(
        (qo * GROUP).to(torch.int64), page_counts, block_q=BLOCK_Q, load_balance=False
    )
    got = forward_prefill(q, k_cache, v_cache, meta, schedule, qo, group=GROUP, block_q=BLOCK_Q)
    torch.npu.synchronize()

    host = RaggedMetadata(
        kv_indptr=meta.kv_indptr.cpu(),
        kv_indices=meta.kv_indices.cpu(),
        kv_last_page_len=meta.kv_last_page_len.cpu(),
        page_size=PAGE,
        qo_indptr=torch.zeros(batch + 1, dtype=torch.int32),
    )
    expected = torch.zeros_like(q).cpu()
    for b in range(batch):
        base, end = int(host.kv_indptr[b]), int(host.kv_indptr[b + 1])
        sub = RaggedMetadata(
            kv_indptr=torch.tensor([0, end - base], dtype=torch.int32),
            kv_indices=host.kv_indices[base:end] - base,
            kv_last_page_len=host.kv_last_page_len[b : b + 1],
            page_size=PAGE,
            qo_indptr=torch.tensor([0, qo_lens[b]], dtype=torch.int32),
        )
        eb = reference_attention(
            q[b : b + 1, :, : qo_lens[b], :].cpu().float(),
            k_cache[base:end].cpu().float(),
            v_cache[base:end].cpu().float(),
            sub,
            causal=True,
        )
        expected[b, :, : qo_lens[b], :] = eb.to(DTYPE)
    diff = (
        got[:, :, : max(qo_lens), :].cpu().float() - expected[:, :, : max(qo_lens), :].float()
    ).abs().max().item()

    latency = _timed(
        lambda: forward_prefill(q, k_cache, v_cache, meta, schedule, qo, group=GROUP, block_q=BLOCK_Q)
    )
    tag = "full-scan baseline" if __import__("os").environ.get("TILEINFER_PREFILL_NOSCAN_OFF") else "truncated"
    print(f"{label:<24} [{tag:<18}] diff={diff:.4f}  {latency:>9.3f} ms  tiles={schedule.num_tiles}")


def main():
    if not is_npu_available():
        print("no NPU available; this benchmark needs the device")
        return 1
    for label, kv_lens, qo_lens, dim in SHAPES:
        run_case(label, kv_lens, qo_lens, dim)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
