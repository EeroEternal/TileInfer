"""vLLM-facing model adapters (L3).

One module per architecture: it owns the ``nn.Module`` that vLLM's ``ModelRegistry`` resolves, the
``load_weights`` protocol, and nothing else — the math lives in ``tileinfer/models/<arch>`` so it can
be tested on a box with no vLLM (``docs/plan-vllm-native-backend.md`` §2).
"""

__all__ = ["dsv41"]
