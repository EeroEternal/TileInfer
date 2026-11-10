#!/usr/bin/env python3
"""Cast torch_npu.npu_dynamic_mx_quant uint8 scale to e8m0 dtype for GMM."""

from __future__ import annotations

import pathlib

base = pathlib.Path(
    "/home/userlist/zxs/TileInfer/.venv_ascend30/lib/python3.12/site-packages/vllm_ascend"
)

p = base / "quantization/methods/w4a8/w4a8_mxfp4.py"
text = p.read_text()
old = """        hidden_states, out_scale = torch_npu.npu_dynamic_mx_quant(
            hidden_states, dst_type=torch.float8_e4m3fn
        )
        return hidden_states, maybe_normalize_mxfp_scale_layout(out_scale)
"""
new = """        hidden_states, out_scale = torch_npu.npu_dynamic_mx_quant(
            hidden_states, dst_type=torch.float8_e4m3fn
        )
        if out_scale.dtype != torch_npu.float8_e8m0fnu:
            out_scale = out_scale.view(torch_npu.float8_e8m0fnu)
        return hidden_states, maybe_normalize_mxfp_scale_layout(out_scale)
"""
if new in text:
    print("[skip] w4a8 scale cast")
elif old in text:
    p.write_text(text.replace(old, new, 1))
    print("[patch] w4a8 scale cast")
else:
    raise SystemExit("[fail] w4a8 anchor")

p = base / "ops/fused_moe/shared_experts.py"
text = p.read_text()
old = """    q, scale = torch_npu.npu_dynamic_mx_quant(y, dst_type=torch.float8_e4m3fn)
"""
new = """    q, scale = torch_npu.npu_dynamic_mx_quant(y, dst_type=torch.float8_e4m3fn)
    if scale.dtype != torch_npu.float8_e8m0fnu:
        scale = scale.view(torch_npu.float8_e8m0fnu)
"""
if new in text:
    print("[skip] shared scale cast")
elif old in text:
    p.write_text(text.replace(old, new, 1))
    print("[patch] shared scale cast")
else:
    raise SystemExit("[fail] shared anchor")
