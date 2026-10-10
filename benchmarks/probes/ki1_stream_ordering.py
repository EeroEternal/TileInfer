"""KI-1 investigation probe - the kernel ignores the caller's NPU stream (refuted)

Run with the Ascend 950 environment (docs/architecture.md).  Kept in the tree because each of these
refuted a plausible cause of KI-1 in about a minute of wall clock, and because the same probes are
the natural starting point if the bug resurfaces.

Original notes follow.
"""

"""KI-1 hypothesis 3: does the kernel honour the caller's NPU stream?

Inside vLLM the cache write (torch_npu's reshape_and_cache) and our kernel launch are adjacent.  If
the TileLang launch does not respect torch's current stream, the kernel can read the page before the
write lands - a race that would look like an occasional device error.

The test writes a distinctive pattern into the page the kernel will read, does so on a *non-default*
stream, then calls the kernel immediately (with the default stream as the current one, exactly as a
careless caller would) and asks whether the kernel saw the write.
"""
import sys, torch, torch_npu  # noqa: F401
sys.path.insert(0, "/home/lipi/TileInfer")
from tileinfer.kernels.attention.paged_decode_ascend950 import build_decode_kernel, padded_rows
from tileinfer.testing.reference import reference_attention
from tileinfer.metadata import RaggedMetadata

dev = torch.device("npu", 0); DT = torch.bfloat16
page, dim, kvh, g = 128, 128, 2, 8
BR = padded_rows(g)

k_cache = torch.zeros(2, page, kvh, dim, dtype=DT, device=dev)
v_cache = torch.zeros(2, page, kvh, dim, dtype=DT, device=dev)
k_new = torch.randn(1, kvh, page, dim, dtype=DT, device=dev)
v_new = torch.randn(1, kvh, page, dim, dtype=DT, device=dev)
q = torch.randn(1, kvh * g, 1, dim, dtype=DT, device=dev) * 0.2
q_pad = torch.zeros(1, kvh * BR, dim, dtype=DT, device=dev)
q_pad.view(1, kvh, BR, dim)[:, :, :g] = q.view(1, kvh, g, dim)
ids = torch.tensor([0, 0], dtype=torch.int32, device=dev)   # ABI: one entry per pool page
indptr = torch.tensor([0, 1], dtype=torch.int32, device=dev)
last = torch.tensor([page], dtype=torch.int32, device=dev)
kern = build_decode_kernel(batch=1, kv_heads=kvh, group=g, dim=dim, page_size=page, num_pages_cap=2)

def expected_from(cache_k, cache_v):
    m = RaggedMetadata(kv_indptr=indptr.cpu(), kv_indices=ids.cpu(), kv_last_page_len=last.cpu(), page_size=page)
    m.qo_indptr = torch.tensor([0, 1], dtype=torch.int32)
    return reference_attention(q.cpu().float(), cache_k.cpu().float(), cache_v.cpu().float(), m).float()

# 1) control: write on the default stream, then run  -> must see the write
k_cache[0] = k_new[0].permute(1, 0, 2)
v_cache[0] = v_new[0].permute(1, 0, 2)
res = kern(q_pad, k_cache, v_cache, indptr, ids, last); torch.npu.synchronize()
got = res.view(1, kvh, BR, dim)[:, :, :g].reshape(1, kvh * g, dim)
d1 = (got.cpu().float().view_as(expected_from(k_cache, v_cache)) - expected_from(k_cache, v_cache)).abs().max().item()
print(f"  control (write then kernel, default stream): max|diff|={d1:.4f}", flush=True)

# 2) the real test: write on a *non-default* stream, run the kernel right after
k_cache.zero_(); v_cache.zero_(); torch.npu.synchronize()
s = torch.npu.Stream()
with torch.npu.stream(s):
    k_cache[0] = k_new[0].permute(1, 0, 2)     # issued on stream s
    v_cache[0] = v_new[0].permute(1, 0, 2)
# no sync: the caller's next line launches the kernel (default stream)
res2 = kern(q_pad, k_cache, v_cache, indptr, ids, last)
torch.npu.synchronize()
got2 = res2.view(1, kvh, BR, dim)[:, :, :g].reshape(1, kvh * g, dim)
exp_written = expected_from(k_cache, v_cache)                  # oracle with the write visible
zeros = torch.zeros_like(k_cache), torch.zeros_like(v_cache)
exp_old = expected_from(*zeros)                                # oracle as if the write had not landed
d_written = (got2.cpu().float().view_as(exp_written) - exp_written).abs().max().item()
d_old = (got2.cpu().float().view_as(exp_old) - exp_old).abs().max().item()
verdict = "SAW the write (ordering respected)" if d_written < d_old else "SAW THE OLD CACHE (ordering violated!)"
print(f"  non-default-stream write then kernel: d(written)={d_written:.4f} d(old)={d_old:.4f} -> {verdict}", flush=True)
