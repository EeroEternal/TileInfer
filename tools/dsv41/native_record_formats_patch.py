#!/usr/bin/env python3
"""Convert NZ expert weights back to ND inside device_loading_context.

With ``--quantization deepseek_v4_fp8`` the MXFP4 expert weights become an
internal NZ format during ``process_weights_after_loading``.  vLLM's
``device_loading_context`` then tries ``torch.empty_like(...).copy_(p.data)`` to
move them back to CPU, which the NPU rejects for internal-format tensors.

``vllm_ascend`` ships ``record_offloaded_param_formats()`` for exactly this: it
records the NZ format and restores the ND tensor while the parameter is still on
device.  The only missing piece is calling it inside the loading context.
"""

from __future__ import annotations

import argparse
import pathlib
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site-packages", required=True)
    args = ap.parse_args()
    path = pathlib.Path(args.site_packages).resolve() / "vllm/model_executor/model_loader/utils.py"
    text = path.read_text()

    old = '''            with loading_context:
                quant_method.process_weights_after_loading(module)
'''
    new = '''            with loading_context:
                quant_method.process_weights_after_loading(module)
                # --- restore NZ -> ND before device_loading_context copies back ---
                try:
                    from vllm_ascend.model_executor.offloader.prefetch import (
                        record_offloaded_param_formats,
                    )

                    record_offloaded_param_formats()
                except ImportError:
                    pass
'''
    if new in text:
        print("[skip] record formats hook already present")
        return 0
    if old not in text:
        raise SystemExit("[fail] process_weights_after_loading anchor not found")
    path.write_text(text.replace(old, new, 1))
    print("[patch] record_offloaded_param_formats inside device_loading_context")
    return 0


if __name__ == "__main__":
    sys.exit(main())
