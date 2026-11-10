#!/usr/bin/env python3
"""Wire the selective per-expert loader into vllm-ascend's routed MoE path.

``vllm_ascend/ops/fused_moe/routed_experts.py`` calls ``self._select_experts``
and ``self.quant_method.apply`` directly, bypassing vLLM's MoERunner where the
TileInfer hook lives.  Without this, the decoder-layer hook sets
``_selective_forward`` (and skips ``wait_prefetch``) but no expert is loaded.
"""

from __future__ import annotations

import argparse
import pathlib


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site-packages", required=True)
    args = ap.parse_args()
    p = (
        pathlib.Path(args.site_packages)
        / "vllm_ascend/ops/fused_moe/routed_experts.py"
    )
    text = p.read_text()
    old = """        self.ascend_pertoken_scale = pertoken_scale
        self.ascend_mc2_mask = mc2_mask
"""
    new = """        # TileInfer selective expert load: this path bypasses MoERunner.
        try:
            from vllm.model_executor.offloader.base import get_offloader

            _load_selected = getattr(get_offloader(), "maybe_load_selected_experts", None)
            if _load_selected is not None:
                _load_selected(topk_ids)
        except ImportError:
            pass
        self.ascend_pertoken_scale = pertoken_scale
        self.ascend_mc2_mask = mc2_mask
"""
    if new in text:
        print("[skip] routed hook already present")
        return 0
    if old not in text:
        raise SystemExit("[fail] routed_experts anchor not found")
    p.write_text(text.replace(old, new, 1))
    print("[patch] routed selective hook")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
