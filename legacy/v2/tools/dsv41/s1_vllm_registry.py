#!/usr/bin/env python3
"""S1 evidence inside a real vLLM process: registration resolves and vLLM can parse the config.

``tileinfer.vllm.plugin`` registers ``DeepseekV41ForCausalLM`` at import time; this script checks that
through *vLLM's own* registry (not our module's view of it), and then asks vLLM's config loader to read
``/home/models/DeepSeek-V4.1-Flash/config.json`` -- the two things a ``vllm serve`` invocation does
before it ever touches a weight.

    python tools/dsv41/s1_vllm_registry.py
"""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

M = "/home/models/DeepSeek-V4.1-Flash"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=M)
    args = parser.parse_args(argv)

    ok = True
    print("vllm:", end=" ")
    try:
        import vllm

        print(vllm.__version__)
    except Exception as exc:  # pragma: no cover - only on a broken env
        print(f"import failed: {exc}")
        return 1

    # 1. our plugin registers the architecture
    from tileinfer.vllm import plugin
    from tileinfer.vllm.config import register_config

    print("transformers config:", register_config())
    reg = plugin.install(strict=False)
    print("plugin.install ->", reg)
    st = plugin.status()
    print("plugin status:", {k: st[k] for k in st if k in ("models", "platform", "ops")} if isinstance(st, dict) else st)

    # 2. vLLM's registry must resolve the name to our class
    from vllm.model_executor.models.registry import ModelRegistry

    arch = "DeepseekV41ForCausalLM"
    try:
        archs = ModelRegistry.get_supported_archs()
        hit = arch in archs
        print(f"vLLM get_supported_archs contains {arch}: {hit}" +
              (f" (of {len(archs)} archs)" if hasattr(archs, "__len__") else ""))
        table = getattr(ModelRegistry, "mapping", None) or getattr(ModelRegistry, "_architectures", None)
        if isinstance(table, dict):
            print(f"  registry entry: {table.get(arch)}")
        ok = ok and hit
    except Exception as exc:
        print(f"registry lookup failed: {exc}")
        ok = False

    # 3. vLLM's config loader must accept the checkpoint
    try:
        from vllm.transformers_utils.config import get_config

        cfg = get_config(args.model_dir, trust_remote_code=True)
        print(
            f"vLLM get_config -> {type(cfg).__name__} "
            f"hidden={getattr(cfg, 'hidden_size', '?')} layers={getattr(cfg, 'num_hidden_layers', '?')} "
            f"experts={getattr(cfg, 'n_routed_experts', '?')}"
        )
    except Exception as exc:
        print(f"get_config failed: {exc}")
        ok = False

    print("RESULT:", "OK" if ok else "PROBLEM")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
