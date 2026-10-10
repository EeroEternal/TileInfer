"""KI-1 investigation probe - caller buffers unaligned (refuted)

Run with the Ascend 950 environment (docs/architecture.md).  Kept in the tree because each of these
refuted a plausible cause of KI-1 in about a minute of wall clock, and because the same probes are
the natural starting point if the bug resurfaces.

Original notes follow.
"""

"""KI-1 hypothesis: do unaligned caller buffers fault the kernel?

The kernel binary is fixed (T.Buffer layouts are compile-time), so the only thing that differs
between a lean process (clean) and a fragmented pool / vLLM EngineCore (fault) is the *addresses* of
the tensors we hand it.  This allocates one big buffer per argument and slices every argument out of
it at a chosen element offset, then runs the same kernel and compares with the reference.
"""
import sys, torch, torch_npu  # noqa: F401
sys.path.insert(0, "/home/lipi/TileInfer")
from tileinfer.kernels.attention.paged_decode_ascend950 import build_decode_kernel, padded_rows
from tileinfer.testing.reference import reference_attention
from tileinfer.metadata import RaggedMetadata

dev = torch.device("npu", 0); DT = torch.bfloat16
page, dim, kvh, g = 128, 128, 2, 8
BR = padded_rows(g)

def case(offset_elems, label, shuffle=True):
    torch.manual_seed(0)
    n_pages, kv_len, batch = 2, 256, 1
    # source data (aligned) ----------------------------------------------------------------
    kb = torch.randn(n_pages * page * kvh * dim + 4096, dtype=DT, device=dev)
    vb = torch.randn(n_pages * page * kvh * dim + 4096, dtype=DT, device=dev)
    qb = torch.randn(batch * kvh * BR * dim + 4096, dtype=DT, device=dev)
    ob = torch.randn(batch * kvh * BR * dim + 4096, dtype=DT, device=dev)
    idb = torch.zeros(n_pages + 4096, dtype=torch.int32, device=dev)
    ipb = torch.zeros(batch + 1 + 4096, dtype=torch.int32, device=dev)
    lpb = torch.zeros(batch + 4096, dtype=torch.int32, device=dev)
    k_dense = torch.randn(batch, kvh, kv_len, dim, dtype=DT, device=dev)
    v_dense = torch.randn(batch, kvh, kv_len, dim, dtype=DT, device=dev)

    o = offset_elems
    k_cache = kb[o:o + n_pages * page * kvh * dim].view(n_pages, page, kvh, dim)
    v_cache = vb[o:o + n_pages * page * kvh * dim].view(n_pages, page, kvh, dim)
    # fill the caches through the views (so the *data* is right regardless of alignment)
    for b in range(batch):
        for p in range(n_pages):
            s, e = p * page, min((p + 1) * page, kv_len)
            k_cache[b * n_pages + p, :e - s] = k_dense[b, :, s:e].permute(1, 0, 2)
            v_cache[b * n_pages + p, :e - s] = v_dense[b, :, s:e].permute(1, 0, 2)
    ids = torch.arange(n_pages, dtype=torch.int32, device=dev)
    if shuffle:
        ids = ids[torch.randperm(n_pages, device=dev)]
    kv_indices = idb[o:o + n_pages].copy_(ids)
    kv_indptr = ipb[o:o + 2].copy_(torch.tensor([0, n_pages], dtype=torch.int32, device=dev))
    kv_last = lpb[o:o + 1].copy_(torch.tensor([page], dtype=torch.int32, device=dev))

    q = torch.randn(batch, kvh * g, 1, dim, dtype=DT, device=dev) * 0.2
    q_pad = qb[o:o + batch * kvh * BR * dim].view(batch, kvh * BR, dim).zero_()
    q_pad.view(batch, kvh, BR, dim)[:, :, :g] = q.view(batch, kvh, g, dim)
    out_pad = ob[o:o + batch * kvh * BR * dim].view(batch, kvh * BR, dim)

    try:
        kern = build_decode_kernel(batch=batch, kv_heads=kvh, group=g, dim=dim,
                                   page_size=page, num_pages_cap=n_pages)
        res = kern(q_pad, k_cache, v_cache, kv_indptr, kv_indices, kv_last)
        torch.npu.synchronize()
        got = res.view(batch, kvh, BR, dim)[:, :, :g].reshape(batch, kvh * g, dim)
        m2 = RaggedMetadata(kv_indptr=kv_indptr.cpu(), kv_indices=kv_indices.cpu(),
                            kv_last_page_len=kv_last.cpu(), page_size=page)
        m2.qo_indptr = torch.arange(batch + 1, dtype=torch.int32)
        exp = reference_attention(q.cpu().float(), k_cache.cpu().float(), v_cache.cpu().float(), m2).float()
        d = (got.cpu().float().view_as(exp) - exp).abs().max().item()
        print(f"  {label:34s} max|diff|={d:.4f} {'OK' if d < 3e-2 else 'WRONG'}", flush=True)
    except Exception as e:
        msg = next((l for l in str(e).splitlines() if "error code" in l or "Error" in l), str(e)[:90])
        print(f"  {label:34s} FAILED {msg.strip()[:100]}", flush=True)

print("alignment of every caller buffer (element offset into a fresh tensors):")
case(0, "aligned (control)")
case(1, "offset 1 elem (2B)")
case(8, "offset 8 elem (16B)")
case(64, "offset 64 elem (128B)")
case(256, "offset 256 elem (512B)")
