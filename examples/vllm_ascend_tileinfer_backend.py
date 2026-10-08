"""Reference sketch of the vLLM-Ascend attention backend for TileInfer (Phase 2).

This file is *documentation that happens to be Python*: it shows the exact seam where TileInfer
plugs into vLLM-Ascend, mirroring how ``vllm_ascend/attention/fa3_v1.py`` registers the
``FLASH_ATTN`` backend.  It is not imported by TileInfer itself, and it does not import vLLM at
module scope, so it stays readable on a machine without vLLM installed.

Where it goes::

    vllm_ascend/attention/tileinfer_v1.py        <- this file
    vllm_ascend/attention/__init__.py            <- export AscendTileInferBackend
    vllm_ascend/platform.py                      <- register for --attention-backend TILEINFER

The three things that make this integration smaller than a typical custom kernel backend:

1. **Metadata conversion is host-side and happens once per batch shape.**  vLLM hands us
   ``block_tables`` ``[batch, max_pages]`` and ``seq_lens``; :meth:`BatchAttention.plan` compacts
   them into ``kv_indptr`` / ``kv_indices`` / ``kv_last_page_len`` and drops the padding, so a
   kernel block never has to skip over unused table entries.
2. **The plan is captured, not rebuilt.**  ``plan`` is called from
   ``build_attention_metadata`` (or a shape-change hook), so the captured ACLGraph contains only
   the ``run`` call.  Nothing inside the captured region allocates or compiles.
3. **KV layout needs no repacking** for the common Ascend layout
   ``[num_blocks, block_size, num_kv_heads, head_dim]`` (NHD), which is what
   ``get_kv_cache_shape`` returns below — the same layout the FIA path uses, so a model can flip
   between backends without re-allocating its cache.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

try:  # pragma: no cover - only meaningful inside a vLLM-Ascend install
    from vllm.v1.attention.backend import AttentionBackend  # type: ignore
except Exception:  # pragma: no cover
    class AttentionBackend:  # type: ignore
        """Placeholder so this file can be read (and linted) outside vLLM."""


from tileinfer import BatchAttention, PageTable  # noqa: E402


class AscendTileInferBackend(AttentionBackend):
    """``--attention-backend TILEINFER``: TileInfer decode/prefill kernels inside vLLM-Ascend."""

    def __init__(self) -> None:
        super().__init__()
        self._attn: Optional[BatchAttention] = None
        self._plan_cache: dict = {}

    # ------------------------------------------------------------------ identity

    @staticmethod
    def get_name() -> str:
        return "TILEINFER"

    @staticmethod
    def get_impl_cls() -> type:
        return AscendTileInferImpl

    @staticmethod
    def get_builder_cls() -> type:
        return AscendTileInferMetadataBuilder

    # ------------------------------------------------------------------ KV-cache contract

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int, block_size: int, num_kv_heads: int, head_size: int, cache_type: str = ""
    ) -> tuple:
        # NHD — identical to the FIA path, so switching backends does not re-allocate the cache
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_supported_kernel_block_sizes() -> list:
        return [128]

    # ------------------------------------------------------------------ glue

    def attention(self, device: torch.device) -> BatchAttention:
        if self._attn is None:
            self._attn = BatchAttention(backend="auto", device=device, dtype=torch.float16)
        return self._attn


class AscendTileInferMetadataBuilder:
    """Turns vLLM's common metadata into a TileInfer plan.

    vLLM calls this once per step; the expensive part (partitioning + JIT lookup) is keyed on the
    batch *shape*, so steady-state steps hit :meth:`BatchAttention.plan`'s cache and pay only the
    metadata compaction.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.attn: Optional[BatchAttention] = None
        self.plan = None

    def build(self, common_metadata: Any, device: torch.device) -> "AscendTileInferMetadataBuilder":
        seq_lens = common_metadata.seq_lens  # [batch] int32, includes the tokens of this step
        block_table = common_metadata.block_tables  # [batch, max_pages] int32
        query_start_loc = common_metadata.query_start_loc  # [batch + 1] int32
        num_qo_heads = common_metadata.num_qo_heads
        num_kv_heads = common_metadata.num_kv_heads
        head_dim = common_metadata.head_dim
        page_size = common_metadata.block_size

        self.attn = self.attn or BatchAttention(backend="auto", device=device)
        # one call: dense table -> ragged metadata -> schedule -> kernel handle
        self.plan = self.attn.plan_from_page_table(
            block_table,
            # the *cached* prefix plus this step's tokens; for decode this equals seq_lens
            seq_lens,
            qo_lens=query_start_loc[1:] - query_start_loc[:-1],
            page_size=page_size,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )
        return self


class AscendTileInferImpl:
    """The forward pass: one ``run`` call per layer, no host synchronisation."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.attn: Optional[BatchAttention] = None
        self.plan = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        metadata: "AscendTileInferMetadataBuilder",
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # vLLM passes a fused [2, blocks, block_size, kv_heads, dim] cache; TileInfer wants the
        # two halves separately (the kernel reads K once and V once, and V is only needed after
        # the softmax — keeping them apart avoids a slice object per call).
        k_cache, v_cache = kv_cache[0], kv_cache[1]
        assert metadata.plan is not None, "metadata builder must run before forward()"
        self.attn = self.attn or metadata.attn
        return self.attn.run(query, k_cache, v_cache, out=output, plan=metadata.plan)


# ---------------------------------------------------------------------------------------
# What is still missing before this can be merged upstream (tracked in docs/roadmap.md)
# ---------------------------------------------------------------------------------------
#
# * prefill/append kernel: vLLM-Ascend dispatches prefill through the same backend, so Phase 2
#   needs the causal/qo_len>1 kernel or a fallback to the FIA path for prefill chunks;
# * ACLGraph: `plan` must be built before capture; verify that `_refresh_metadata` copies are
#   capture-safe for vLLM's update-i-place metadata buffers;
# * sliding window / attention sink: the plan already carries `window_left`, the kernel does not;
# * one backend instance per (layer, device) to keep the plan cache warm across layers.
