"""Probe: npu_grouped_matmul with packed uint8 weight + weight_dtype=fp4 (MXFP4)."""

import torch
import torch_npu

from tools.dsv41 import ref_cpu

torch.manual_seed(0)
dev = "npu:0"
x = torch.randn(1, 128, dtype=torch.bfloat16, device=dev)
w = torch.randn(64, 128, dtype=torch.float32)
packed, scale = ref_cpu.quantize_weight_fp4_block32(w)
packed = packed.to(dev)
scale = scale.to(dev)
print("packed", tuple(packed.shape), "scale", tuple(scale.shape), scale.dtype)

ref_w = ref_cpu.unpack_fp4_lut(packed, 64, 128, dtype=torch.bfloat16).float()
ref_w = ref_w * ref_cpu.scale_to_float(scale).repeat_interleave(32, dim=-1)
ref = (x.float() @ ref_w.float().t()).to(torch.bfloat16)

for name, s in (("scale", scale), ("scale_t", scale.t().contiguous())):
    for split in (0, 2):
        try:
            out = torch_npu.npu_grouped_matmul(
                [x], [packed], antiquant_scale=[s],
                weight_dtype=torch_npu.float4_e2m1fn_x2,
                split_item=split, output_dtype=torch.bfloat16,
            )
            o = out[0] if isinstance(out, (list, tuple)) else out
            d = (o.float() - ref.float()).abs().max().item()
            print(name, "split", split, "OK", tuple(o.shape), "maxdiff", d)
        except Exception as e:  # noqa: BLE001
            print(name, "split", split, "ERR", type(e).__name__, str(e)[:180])
