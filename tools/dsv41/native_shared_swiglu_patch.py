#!/usr/bin/env python3
"""Patch the two shared-expert npu_swiglu_group_quant calls to torch built-ins."""

from __future__ import annotations

import pathlib

p = pathlib.Path(
    "/home/userlist/zxs/TileInfer/.venv_ascend30/lib/python3.12/site-packages/"
    "vllm_ascend/ops/fused_moe/shared_experts.py"
)
text = p.read_text()

old1 = '''            quantized_x, swiglu_out_scale, _ = torch.ops._C_ascend.npu_swiglu_group_quant(
                submitted_gate_up.gate_up,
                topk_weight=None,
                group_index=None,
                dst_type=torch.float8_e4m3fn,
                quant_mode=2,
                clamp_value=self.swiglu_limit,
            )
'''
new1 = '''            quantized_x, swiglu_out_scale, _ = _torch_swiglu_group_quant(
                submitted_gate_up.gate_up, self.swiglu_limit
            )
'''
old2 = '''            quantized_x, scale, _ = torch.ops._C_ascend.npu_swiglu_group_quant(
                gate_up,
                topk_weight=None,
                group_index=None,
                dst_type=torch.float8_e4m3fn,
                quant_mode=SWIGLU_MX_QUANT_MODE,
                clamp_value=0.0,
            )
'''
new2 = '''            quantized_x, scale, _ = _torch_swiglu_group_quant(gate_up, 0.0)
'''
for old, new, tag in ((old1, new1, "shared 607"), (old2, new2, "shared 651")):
    if new in text:
        print(f"[skip] {tag}")
    elif old in text:
        text = text.replace(old, new, 1)
        print(f"[patch] {tag}")
    else:
        raise SystemExit(f"[fail] {tag} anchor")

helper = '''

def _torch_swiglu_group_quant(x, clamp_value=0.0):
    """SwiGLU + MXFP8 quant via torch_npu built-ins (custom op ABI fallback)."""
    if clamp_value is not None and clamp_value > 0:
        x = torch.clamp(x, max=clamp_value)
    y = torch_npu.npu_swiglu(x, dim=-1)
    q, scale = torch_npu.npu_dynamic_mx_quant(y, dst_type=torch.float8_e4m3fn)
    from vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4 import (
        maybe_normalize_mxfp_scale_layout,
    )

    return q, maybe_normalize_mxfp_scale_layout(scale), y
'''
if "_torch_swiglu_group_quant" not in text or "def _torch_swiglu_group_quant" not in text:
    text += helper
    print("[patch] helper appended")
p.write_text(text)
print("done")
