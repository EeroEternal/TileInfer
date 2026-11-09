#!/usr/bin/env python3
"""Torch fallback for npu_swiglu_group_quant when the CANN custom op ABI mismatches.

The wheel's torch binding and the installed CANN 9.3.0 built-in
``aclnnSwigluGroupQuant`` disagree (yOut null).  SwiGLU + MXFP8 quant is
expressible with torch_npu built-ins, so use those for bring-up.
"""

from __future__ import annotations

import pathlib

p = pathlib.Path(
    "/home/userlist/zxs/TileInfer/.venv_ascend30/lib/python3.12/site-packages/"
    "vllm_ascend/quantization/methods/w4a8/w4a8_mxfp4.py"
)
text = p.read_text()

old = '''        hidden_states, out_scale, _ = torch.ops._C_ascend.npu_swiglu_group_quant(
            hidden_states,
            topk_weight=None,
            group_index=group_index,
            dst_type=torch.float8_e4m3fn,
            quant_mode=2,
            clamp_value=mlp_compute_input.swiglu_limit,
        )
        return hidden_states, maybe_normalize_mxfp_scale_layout(out_scale)
'''
new = '''        # Torch fallback for the CANN 9.3.0 / wheel aclnnSwigluGroupQuant ABI
        # mismatch: built-in SwiGLU + MXFP8 dynamic quant.
        limit = mlp_compute_input.swiglu_limit
        if limit is not None and limit > 0:
            hidden_states = torch.clamp(hidden_states, max=limit)
        hidden_states = torch_npu.npu_swiglu(hidden_states, dim=-1)
        hidden_states, out_scale = torch_npu.npu_dynamic_mx_quant(
            hidden_states, dst_type=torch.float8_e4m3fn
        )
        return hidden_states, maybe_normalize_mxfp_scale_layout(out_scale)
'''
if new in text:
    print("[skip] swiglu fallback already present")
else:
    if old not in text:
        raise SystemExit("[fail] swiglu call anchor not found")
    p.write_text(text.replace(old, new, 1))
    print("[patch] swiglu_group_quant torch fallback")
