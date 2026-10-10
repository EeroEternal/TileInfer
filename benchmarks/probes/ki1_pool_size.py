"""KI-1 investigation probe - KV pool above 2 GiB overflows addressing (refuted)

Run with the Ascend 950 environment (docs/architecture.md).  Kept in the tree because each of these
refuted a plausible cause of KI-1 in about a minute of wall clock, and because the same probes are
the natural starting point if the bug resurfaces.

Original notes follow.
"""

"""KI-1 hypothesis 2: does a large KV pool overflow the kernel's address arithmetic?

vLLM allocates one big paged cache (tens of GB).  Our kernel is compiled with the *whole pool* as its
declared tensor shape, so its element/byte offsets grow with the pool even though a step only touches
a few pages.  If the addressing is 32-bit, a pool above the int32 byte limit (~2 GiB, i.e. ~32k pages
at 128x128x2) produces garbage addresses - which is exactly the shape of a vector-core exception.
"""
import sys, torch, torch_npu  # noqa: F401
sys.path.insert(0, "/home/lipi/TileInfer")
from tileinfer.kernels.attention.paged_decode_ascend950 import build_decode_kernel, padded_rows
from tileinfer.testing.reference import reference_attention
from tileinfer.metadata import RaggedMetadata

dev = torch.device("npu", 0); DT = torch.bfloat16
page, dim, kvh, g = 128, 128, 2, 8
BR = padded_rows(g)

def case(cap_pages, used_pages=2, label=""):
    torch.manual_seed(0)
    torch.npu.empty_cache() if hasattr(torch.npu, "empty_cache") else None
    kv_len = used_pages * page
    try:
        k_cache = torch.randn(cap_pages, page, kvh, dim, dtype=DT, device=dev)
        v_cache = torch.randn(cap_pages, page, kvh, dim, dtype=DT, device=dev)
    except Exception as e:
        print(f"  {label}: allocation failed ({str(e)[:60]})", flush=True); return
    # the ABI wants the declared capacity: pad with zeros, as the wrapper does (unused slots are
    # never dereferenced because every access is bounded by kv_indptr)
    ids = torch.cat([torch.arange(used_pages, dtype=torch.int32, device=dev),
                     torch.zeros(cap_pages - used_pages, dtype=torch.int32, device=dev)])
    indptr = torch.tensor([0, used_pages], dtype=torch.int32, device=dev)
    last = torch.tensor([page], dtype=torch.int32, device=dev)
    q = torch.randn(1, kvh * g, 1, dim, dtype=DT, device=dev) * 0.2
    q_pad = torch.zeros(1, kvh * BR, dim, dtype=DT, device=dev)
    q_pad.view(1, kvh, BR, dim)[:, :, :g] = q.view(1, kvh, g, dim)
    gib = cap_pages * page * kvh * dim * 2 / 2**30
    try:
        kern = build_decode_kernel(batch=1, kv_heads=kvh, group=g, dim=dim,
                                   page_size=page, num_pages_cap=cap_pages)
        res = kern(q_pad, k_cache, v_cache, indptr, ids, last)
        torch.npu.synchronize()
    except Exception as e:
        msg = next((l for l in str(e).splitlines() if "error code" in l or "Error" in l), str(e)[:90])
        print(f"  {label} (pool {gib:.1f} GiB): FAILED {msg.strip()[:90]}", flush=True); return
    got = res.view(1, kvh, BR, dim)[:, :, :g].reshape(1, kvh * g, dim)
    m2 = RaggedMetadata(kv_indptr=indptr.cpu(), kv_indices=ids.cpu(), kv_last_page_len=last.cpu(), page_size=page)
    m2.qo_indptr = torch.tensor([0, 1], dtype=torch.int32)
    exp = reference_attention(q.cpu().float(), k_cache[:used_pages].cpu().float(),
                              v_cache[:used_pages].cpu().float(), m2).float()
    d = (got.cpu().float().view_as(exp) - exp).abs().max().item()
    print(f"  {label} (pool {gib:.1f} GiB): max|diff|={d:.4f} {'OK' if d < 3e-2 else 'WRONG'}", flush=True)
    del k_cache, v_cache
    try: torch.npu.empty_cache()
    except Exception: pass

print("pool size sweep (2 pages of real data, the rest unused):")
case(2, 2, "tiny pool")
case(8_000, 2, "8k pages")      # 0.5 GiB
case(40_000, 2, "40k pages")    # 2.5 GiB - just past the int32 byte limit
case(120_000, 2, "120k pages")  # 7.5 GiB
