"""Launch vLLM with the TileInfer backend registered.

    python scripts/vllm_tileinfer_launch.py serve <model> --attention-backend CUSTOM ...

Why a launcher instead of the `vllm` CLI: the backend has to be registered *before* vLLM resolves
`--attention-backend`, and an out-of-tree package can only do that by importing itself first.
`vllm_ascend` has to be importable already (it is, once the model's platform is selected) - importing
it here as well makes the ordering explicit.
"""

from __future__ import annotations

import sys


def main() -> int:
    import os

    import vllm_ascend  # noqa: F401  (its platform registers the Ascend backends)

    if os.environ.get("TILEINFER_DISABLE"):
        # the baseline: vLLM-Ascend's own (FIA) path, for an A/B comparison of the same prompt
        print("[tileinfer] TILEINFER_DISABLE is set - serving with the stock Ascend backends")
    else:
        from tileinfer.integrations import vllm_ascend as ti

        if not ti.install():
            raise SystemExit("TileInfer could not register its attention backend")
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        print("[tileinfer] CUSTOM backend resolves to:", AttentionBackendEnum.CUSTOM.get_class())

    from vllm.entrypoints.cli.main import main as vllm_main

    return vllm_main()


if __name__ == "__main__":
    raise SystemExit(main())
