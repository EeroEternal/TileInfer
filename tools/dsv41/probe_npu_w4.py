"""Probe: does npu_weight_quant_batchmatmul support MXFP4 (e2m1 + e8m0, group 32)?"""

import torch
import torch_npu

from tools.dsv41 import ref_cpu

torch.manual_seed(0)
dev = "npu:0"
x = torch.randn(1, 128, dtype=torch.bfloat16, device=dev)
w = torch.randn(64, 128, dtype=torch.float32)
packed, scale = ref_cpu.quantize_weight_fp4_block32(w)
print("packed", tuple(packed.shape), packed.dtype, "scale", tuple(scale.shape), scale.dtype)
packed = packed.to(dev)
scale = scale.to(dev)
wf = packed
print("wf", tuple(wf.shape), wf.dtype)

ref_w = ref_cpu.unpack_fp4_lut(packed, 64, 128, dtype=torch.bfloat16).float()
ref_w = ref_w * ref_cpu.scale_to_float(scale).repeat_interleave(32, dim=-1)
ref = (x.float() @ ref_w.float().t()).to(torch.bfloat16)

for name, sc in (("scale", scale), ("scale_t", scale.t().contiguous())):
    try:
        out = torch_npu.npu_weight_quant_batchmatmul(
            x, wf, sc, antiquant_group_size=32,
            weight_dtype=torch_npu.float4_e2m1fn_x2,
        )
        d = (out.float() - ref.float()).abs().max().item()
        print(name, "OK", tuple(out.shape), out.dtype, "maxdiff", d)
    except Exception as e:  # noqa: BLE001
        print(name, "ERR", type(e).__name__, str(e)[:300])

