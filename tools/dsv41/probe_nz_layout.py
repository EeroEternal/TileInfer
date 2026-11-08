"""Probe the physical size / expert-block layout of FP4 NZ_C0_16 weights.

Answers two questions needed for the native host-offload design:

1. Does the NZ_C0_16 cast pack FP4 (halving host bytes) or keep one byte per
   ND element?
2. Is an expert block inside a full [E, M, K] cast byte-identical to a
   standalone [1, M, K] cast?  If yes we can cast only the selected experts
   per token instead of keeping a full host-NZ copy.
"""

import os
import time

os.environ.setdefault("VLLM_PLUGINS", "ascend")

import torch
import torch_npu  # noqa: F401

from vllm_ascend.model_executor.offloader.prefetch import (
    _ACL_MEMCPY_DEVICE_TO_HOST,
    _acl_memcpy_async,
    _acl_sync,
    _cast_nd_to_nz,
)

E, M, K = 384, 4608, 5120
x = torch.empty((E, M, K), dtype=torch.uint8, device="npu")
nz = _cast_nd_to_nz(x, 50)
total = nz.untyped_storage().nbytes()
print(f"ND bytes      = {x.numel()} ({x.numel()/1e9:.3f} GB)")
print(f"NZ bytes      = {total} ({total/1e9:.3f} GB)  ratio={total/x.numel():.3f}")
print(f"NZ shape      = {tuple(nz.shape)} strides={nz.stride()} contiguous={nz.is_contiguous()}")
print(f"per-expert    = {total/E:.1f} bytes (remainder {total % E})")

host = torch.empty(total, dtype=torch.uint8, pin_memory=True)
_acl_memcpy_async(host.data_ptr(), nz.data_ptr(), total, _ACL_MEMCPY_DEVICE_TO_HOST)
_acl_sync()

per = total // E
for e in (0, 1, 7, 383):
    one = x[e : e + 1].clone()
    nz1 = _cast_nd_to_nz(one, 50)
    b1 = nz1.untyped_storage().nbytes()
    h1 = torch.empty(b1, dtype=torch.uint8, pin_memory=True)
    _acl_memcpy_async(h1.data_ptr(), nz1.data_ptr(), b1, _ACL_MEMCPY_DEVICE_TO_HOST)
    _acl_sync()
    same_offset = torch.equal(h1, host[e * per : e * per + b1]) if e * per + b1 <= total else False
    print(f"expert {e}: single bytes={b1} per_full={per} equal@offset={same_offset}")
    del nz1, h1, one

# Cost of casting a single expert ND->NZ on device.
one = x[3:4].clone()
for _ in range(2):
    _cast_nd_to_nz(one, 50)
_acl_sync()
t0 = time.perf_counter()
for _ in range(20):
    z = _cast_nd_to_nz(one, 50)
_acl_sync()
dt = (time.perf_counter() - t0) / 20
print(f"single-expert cast = {dt*1e3:.2f} ms ({one.numel()/dt/1e9:.1f} GB/s)")
