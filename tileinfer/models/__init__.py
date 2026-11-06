"""Model implementations that are **not** vLLM-aware (L1-adjacent).

Each subpackage is one architecture, written against plain ``torch`` so it can be constructed,
loaded and unit-tested on a CPU box with no vLLM and no NPU:

* ``tileinfer.models.dsv41`` — DeepSeek-V4.1-Flash (Compressor/Indexer sparse MLA, Engram,
  Hyper-Connections, DSpark MTP, ViT).  Facts: ``docs/notes/dsv41-facts.md``.

Hard rule (mirrors ``docs/plan-vllm-native-backend.md`` §2): nothing under ``tileinfer/models``
imports ``vllm``, ``torch_npu`` or ``vllm_ascend``.  The vLLM adapter lives in
``tileinfer/vllm/models/<arch>.py`` and imports *this* package, never the other way round.
"""

__all__ = ["dsv41"]
