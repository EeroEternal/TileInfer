"""Device canary for the Ascend 950 paged-decode kernel (official TileLang backend).

Run it on the machine that has the NPU, with the working stack active:

    source <CANN 9.3.x>/set_env.sh
    python benchmarks/probes/ascend950_paged_decode.py

It compiles a handful of shapes and compares every one against the torch reference in
``tileinfer.testing.reference``, which is the only oracle the project trusts.  Exit code 0 means
every case matched.

Why these cases: each of them broke or nearly broke during the port, and they are one line each.

* ``full page``      — the trivial path, kept as a control;
* ``partial tail``   — a page that is not full: the classic silent paged-attention bug;
* ``single token``   — the heaviest possible masking;
* ``shuffled pages`` — indirection through ``kv_indices`` rather than arithmetic on the request id;
* ``ragged batch``   — several requests with different lengths in one launch;
* ``group == M``     — no padding waste (the kernel pads the M tile to a multiple of 32).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tileinfer.metadata import RaggedMetadata  # noqa: E402
from tileinfer.testing.reference import reference_attention  # noqa: E402

DTYPE = torch.bfloat16
TOL = 3e-2  # bf16 kernel vs fp32 torch reference


def _run(tag, kv_lens, batch, kv_heads, group, page_size=128, dim=128, seed=0):
    from tileinfer.kernels.attention.paged_decode_ascend950 import build_decode_kernel

    device = torch.device("npu", 0)
    torch.manual_seed(seed)

    indptr, last_page_len = [0], []
    for length in kv_lens:
        indptr.append(indptr[-1] + -(-length // page_size))
        last_page_len.append(page_size if length % page_size == 0 else length % page_size)
    total_pages = indptr[-1]

    k_cache = torch.randn(total_pages, page_size, kv_heads, dim, dtype=DTYPE, device=device)
    v_cache = torch.randn(total_pages, page_size, kv_heads, dim, dtype=DTYPE, device=device)
    indices = torch.arange(total_pages, dtype=torch.int32, device=device)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    for b in range(batch):
        start, end = indptr[b], indptr[b + 1]
        perm = torch.randperm(end - start, generator=gen) + start
        indices[start:end] = indices[start:end][perm.to(device)]

    meta = RaggedMetadata(
        kv_indptr=torch.tensor(indptr, dtype=torch.int32, device=device),
        kv_indices=indices,
        kv_last_page_len=torch.tensor(last_page_len, dtype=torch.int32, device=device),
        page_size=page_size,
    )
    meta.qo_indptr = torch.arange(batch + 1, dtype=torch.int32, device=device)
    q = torch.randn(batch, kv_heads * group, 1, dim, dtype=DTYPE, device=device) * 0.2

    from tileinfer import BatchAttention

    attn = BatchAttention(backend="tilelang-ascend950", device=device, dtype=DTYPE)
    t0 = time.time()
    plan = attn.plan(
        kv_indptr=meta.kv_indptr,
        kv_indices=meta.kv_indices,
        kv_last_page_len=meta.kv_last_page_len,
        page_size=page_size,
        num_qo_heads=kv_heads * group,
        num_kv_heads=kv_heads,
        head_dim=dim,
        causal=False,
        k_cache=k_cache,
        v_cache=v_cache,
    )
    compile_s = time.time() - t0
    out = attn.run(q, k_cache, v_cache, plan=plan)
    torch.npu.synchronize()

    host = RaggedMetadata(
        kv_indptr=meta.kv_indptr.cpu(),
        kv_indices=meta.kv_indices.cpu(),
        kv_last_page_len=meta.kv_last_page_len.cpu(),
        page_size=page_size,
    )
    host.qo_indptr = torch.arange(batch + 1, dtype=torch.int32)
    expected = reference_attention(
        q.cpu().float(), k_cache.cpu().float(), v_cache.cpu().float(), host
    ).float()
    diff = (out.cpu().float() - expected).abs().max().item()
    ok = diff < TOL
    print(f"  {'PASS' if ok else 'FAIL'}  {tag:22s} max|diff|={diff:.5f}  (compile {compile_s:.0f}s)")
    return ok


def main() -> int:
    print("Ascend 950 paged decode — kernel vs torch reference")
    results = [
        _run("full page", [128], 1, 2, 8),
        _run("partial tail", [200], 1, 2, 8),
        _run("single token", [1], 1, 2, 8),
        _run("ragged batch", [128, 256], 2, 2, 8),
        _run("group == M tile", [256], 1, 1, 32),
    ]
    print(f"{sum(results)}/{len(results)} cases passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
