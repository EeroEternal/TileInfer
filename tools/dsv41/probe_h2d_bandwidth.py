"""Measure H2D bandwidth from pageable mmap / pageable anon / pinned host."""

import os
import sys
import time

os.environ.setdefault("VLLM_PLUGINS", "ascend")

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, "/home/userlist/zxs/TileInfer/.venv_ascend30/lib/python3.12/site-packages")
from vllm_ascend.model_executor.offloader.prefetch import (  # noqa: E402
    _ACL_MEMCPY_HOST_TO_DEVICE,
    _acl_memcpy_async,
    _acl_sync,
    _get_acl_rt,
)

N = 512 * 1024 * 1024
src_mm = torch.frombuffer(__import__("mmap").mmap(-1, N), dtype=torch.uint8, count=N)
src_anon = torch.empty(N, dtype=torch.uint8)
src_pin = torch.empty(N, dtype=torch.uint8, pin_memory=True)
dst = torch.empty(N, dtype=torch.uint8, device="npu")
src_mm.fill_(1)
src_anon.fill_(1)
src_pin.fill_(1)


def bench_acl(src, name, iters=3):
    _acl_memcpy_async(dst.data_ptr(), src.data_ptr(), N, _ACL_MEMCPY_HOST_TO_DEVICE)
    _acl_sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        _acl_memcpy_async(dst.data_ptr(), src.data_ptr(), N, _ACL_MEMCPY_HOST_TO_DEVICE)
    _acl_sync()
    dt = (time.perf_counter() - t0) / iters
    print(f"acl  {name:10s} {N/dt/1e9:7.1f} GB/s  ({dt*1e3:.1f} ms)")


def bench_torch(src, name, iters=3):
    dst.copy_(src, non_blocking=True)
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        dst.copy_(src, non_blocking=True)
    torch.npu.synchronize()
    dt = (time.perf_counter() - t0) / iters
    print(f"torch{name:10s} {N/dt/1e9:7.1f} GB/s  ({dt*1e3:.1f} ms)")


bench_acl(src_pin, "pinned")
bench_acl(src_anon, "pageable")
bench_acl(src_mm, "mmap")
bench_torch(src_pin, "pinned")
bench_torch(src_anon, "pageable")
bench_torch(src_mm, "mmap")

# correctness of pageable acl copy
src_mm.zero_()
src_mm[12345] = 77
_acl_memcpy_async(dst.data_ptr(), src_mm.data_ptr(), N, _ACL_MEMCPY_HOST_TO_DEVICE)
_acl_sync()
print("pageable acl correct:", int(dst[12345].item()) == 77)
