import time
import torch
try:
    import torch_npu  # noqa: F401
except Exception as exc:
    print("torch_npu import failed:", exc)
    raise SystemExit(1)

dev = "npu:0"
free, total = torch.npu.mem_get_info()
print(f"NPU HBM free={free/2**30:.1f} GiB total={total/2**30:.1f} GiB")
print(f"{'MB':>6} {'pinned':>10} {'pageable':>10}   GB/s")
for mb in (1, 4, 32, 128, 512):
    n = mb * 1024 * 1024 // 2
    dst = torch.empty(n, dtype=torch.bfloat16, device=dev)
    row = []
    for pinned in (True, False):
        src = torch.empty(n, dtype=torch.bfloat16)
        if pinned:
            try:
                src = src.pin_memory()
            except Exception:
                row.append(float("nan"))
                continue
        dst.copy_(src)
        torch.npu.synchronize()
        reps = 30 if mb <= 32 else (10 if mb <= 128 else 4)
        t0 = time.perf_counter()
        for _ in range(reps):
            dst.copy_(src)
        torch.npu.synchronize()
        dt = time.perf_counter() - t0
        row.append(reps * mb / 1024.0 / dt)
        del src
    print(f"{mb:6d} {row[0]:10.2f} {row[1]:10.2f}")
    del dst
