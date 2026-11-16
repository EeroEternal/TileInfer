"""vLLM integration layer.

This is the **only** package in TileInfer that is allowed to ``import vllm`` (or ``torch_npu``,
or vLLM's internals).  Everything below it - ``tileinfer.core`` and ``tileinfer.serving`` - stays
engine-independent so it can be tested on a CPU laptop and reused by another engine later.

Layout (filled in as the port lands):

``platform.py``
    ``TileInferPlatform``: device discovery, capability bits, backend/worker selection, config
    validation.  Registered through vLLM's ``vllm.platform_plugins`` entry point.
``worker.py`` / ``mem.py``
    The device worker and the KV-pool / device-memory plumbing.
``attn/``
    The attention backend: metadata builder, impl, KV-cache write, layer selection.
``ops/``
    Operator takeovers over vLLM's ``CustomOp`` dispatch (dense, GDN, MoE, sampling).
``compat/``
    The controlled patch registry and the missing-operator audit mode.
``plugin.py``
    ``install()``: the ``vllm.general_plugins`` entry point.

Importing this package must stay cheap and side-effect free: it does **not** import vLLM, and the
entry-point functions import what they need when vLLM calls them.
"""

from __future__ import annotations

__all__: list[str] = []
