"""Microbenchmark: where do the 2.28 ms of the batched w13 GEMM go?"""

import time

import torch
import torch_npu

from tools.dsv41 import ref_cpu

dev = "npu:0"
torch.manual_seed(0)
k, out_f, in_f = 6, 2304, 5120
packed = torch.randint(0, 256, (k, out_f, in_f // 2), dtype=torch.uint8, device=dev)
sbits = torch.randint(120, 135, (k, out_f, in_f // 32), dtype=torch.uint8, device=dev)
scale = sbits.view(torch.float8_e8m0fnu)
x = torch.randn(k, 1, in_f, dtype=torch.bfloat16, device=dev)
s = ref_cpu.scale_to_float(scale).to(torch.bfloat16)


def bench(name, fn, n=20):
    for _ in range(3):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.npu.synchronize()
    print("%-14s %.3f ms" % (name, (time.perf_counter() - t0) / n * 1e3))


def unpack_scaled():
    w = ref_cpu.unpack_fp4_lut(packed, k * out_f, in_f, dtype=torch.bfloat16)
    w = w.reshape(k, out_f, in_f // 32, 32)
    return (w * s.unsqueeze(-1)).reshape(k, out_f, in_f)


xq, xs = ref_cpu.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu, False)
xf = (xq.float() * ref_cpu.scale_to_float(xs).repeat_interleave(32, -1)).to(torch.bfloat16)
wf = unpack_scaled()

bench("unpack", lambda: ref_cpu.unpack_fp4_lut(packed, k * out_f, in_f, dtype=torch.bfloat16))
bench("unpack+scale", unpack_scaled)
bench("act_quant", lambda: ref_cpu.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu, False))
bench("bmm", lambda: torch.bmm(xf, wf.transpose(1, 2)))
bench(
    "full",
    lambda: torch.bmm(
        (ref_cpu.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu, False)[0].float()
         * ref_cpu.scale_to_float(ref_cpu.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu, False)[1])
         .repeat_interleave(32, -1)).to(torch.bfloat16),
        unpack_scaled().transpose(1, 2),
    ),
)
