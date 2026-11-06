"""``tileinfer.models.dsv41`` — DeepSeek-V4.1-Flash, pure ``torch``.

* :class:`~tileinfer.models.dsv41.config.Dsv41Config` — every shape fact (HF *and* flat spellings).
* :mod:`~tileinfer.models.dsv41.modules` — the parameter tree; ``named_parameters()`` is exactly the
  checkpoint's name set.
* :mod:`~tileinfer.models.dsv41.weights` — header verification (dry run) and the streaming loader.

Facts and provenance: ``docs/notes/dsv41-facts.md``.  Stage status: S1 = this package (tree +
loader); S2/S3 add the forward math; S4 adds the host-offload placement policy.
"""

from __future__ import annotations

from .config import FP4_BLOCK, FP8_BLOCK, Dsv41Config
from .modules import (
    Dsv41Aligner,
    Dsv41Attention,
    Dsv41Block,
    Dsv41Compressor,
    Dsv41DSparkBlock,
    Dsv41Engram,
    Dsv41Indexer,
    Dsv41Model,
    Dsv41MoE,
    Dsv41RMSNorm,
    Dsv41Slot,
    Dsv41ViT,
)
from .weights import LoadReport, fold_wo_a, header_map, load_shards, shard_paths, verify_headers

__all__ = [
    "FP4_BLOCK",
    "FP8_BLOCK",
    "Dsv41Config",
    "Dsv41Aligner",
    "Dsv41Attention",
    "Dsv41Block",
    "Dsv41Compressor",
    "Dsv41DSparkBlock",
    "Dsv41Engram",
    "Dsv41Indexer",
    "Dsv41Model",
    "Dsv41MoE",
    "Dsv41RMSNorm",
    "Dsv41Slot",
    "Dsv41ViT",
    "LoadReport",
    "fold_wo_a",
    "header_map",
    "load_shards",
    "shard_paths",
    "verify_headers",
]
