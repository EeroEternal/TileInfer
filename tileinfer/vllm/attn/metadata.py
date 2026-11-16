"""Per-layer attention metadata for the TileInfer backend.

Why a *second* metadata object
------------------------------
vLLM builds one :class:`CommonAttentionMetadata` per batch
(``vllm/v1/attention/backend.py:359-500``) and hands it to every layer's builder
(``AttentionMetadataBuilder.build``, ``:589-607``).  A backend-specific metadata object is then
free to carry whatever its kernels need.  TileInfer needs very little, and that is the point:

* **references, not copies.**  ``query_start_loc`` / ``seq_lens`` / ``block_table_tensor`` /
  ``slot_mapping`` are the engine's live tensors (the engine rewrites ``block_table_tensor`` and
  ``slot_mapping`` in place every step).  Copying them here would double the per-step HBM traffic
  and, worse, would freeze the values at build time — the serving step
  (``tileinfer.serving.step.refresh_step``) reads the *device* table, so a stale CPU-side copy is
  exactly the silent-wrong-answer failure this design avoids.  The contract test asserts
  identity (``meta.seq_lens is common.seq_lens``), not equality.
* **the three integers the serving layer cannot re-derive for free**: ``num_reqs``,
  ``num_actual_tokens`` and ``max_query_len`` (all already plain ints on the common metadata) plus
  the batch **bucket** the kernels are compiled for
  (``tileinfer.serving.buckets.bucket_for``).
* **decode vs prefill, from ``max_query_len`` only.**  Not from ``attn_state`` and not from
  ``is_prefilling``: ``max_query_len > 1`` is the condition the kernels actually care about (one
  query row per request vs several), and it is the discriminator upstream uses to split a batch
  (``backends/utils.py:467``, ``:542-547``).  ``is_prefilling`` (a per-request bool tensor,
  ``:407-410``) would allow a *mixed* batch whose per-request flags disagree with its query
  lengths, which a single kernel launch cannot serve.

Graph capture
-------------
``TileInferMetadataBuilder._cudagraph_support`` declares
``AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE`` (``v1/attention/backend.py:506-520``): the
paged decode kernel is capturable, but only for a batch where *every* request has exactly one
query token — a mixed prefill/decode batch or a spec-decode batch is not.  The level is a promise
about the kernel, not about the plumbing: capture is only safe after the plans and the per-bucket
buffers exist (see ``tileinfer.vllm.attn.impl`` and the W8 assertion in the port plan §6).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

try:  # pragma: no cover - exercised only outside a vLLM env (the CPU suite stubs `vllm`)
    from vllm.v1.attention.backend import (
        AttentionCGSupport,
        AttentionMetadata,
        AttentionMetadataBuilder,
        CommonAttentionMetadata,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "tileinfer.vllm.attn.metadata requires vLLM 0.23's v1 attention API "
        "(vllm.v1.attention.backend.AttentionMetadata / AttentionMetadataBuilder / "
        "AttentionCGSupport); importing it failed with "
        f"{exc!r}. Install vllm>=0.23,<0.24 - this module is not importable without vLLM."
    ) from exc

from tileinfer.serving.buckets import BATCH_BUCKETS, bucket_for, buckets_for
from tileinfer.serving.counters import Counters

__all__ = ["TileInferMetadata", "TileInferMetadataBuilder", "METADATA_COUNTERS"]

#: Build-time counters.  Metadata building happens once per layer per step, so a bug here (e.g. a
#: silently repaired ``max_query_len``) would be invisible without a counter.
METADATA_COUNTERS = Counters(
    {
        "builds": 0,
        "decode_builds": 0,
        "prefill_builds": 0,
        # `max_query_len` declared as 0 (or smaller than the query_start_loc implies) and repaired
        # from the CPU start locations.  Should stay 0 in a correct vLLM env.
        "max_query_len_repaired": 0,
        # Padded requests we invent to reach the compile-time bucket (never read).
        "bucket_padding_rows": 0,
    }
)


@dataclass
class TileInferMetadata(AttentionMetadata):
    """One batch, in the form the TileInfer backend consumes.

    Every tensor field is the engine's own tensor (reference, no copy).  Scalars are plain ints,
    computed once at build time; reading them later costs nothing and cannot synchronise.
    """

    # --- by reference (the engine owns these; it may rewrite them in place every step) ---------
    query_start_loc: torch.Tensor
    """``(num_reqs + 1,)`` device: start of each request inside the query tensor."""

    query_start_loc_cpu: torch.Tensor
    """``(num_reqs + 1,)`` host mirror; the only place per-request query lengths are read without
    a device sync (``v1/attention/backend.py:368-370``)."""

    seq_lens: torch.Tensor
    """``(num_reqs,)`` device: KV length *after* this step's tokens are written
    (``v1/attention/backend.py:372-374``; see ``compute_num_computed_tokens`` at ``:464-469``:
    ``seq_lens - query_lens`` is the cached prefix)."""

    block_table_tensor: torch.Tensor
    """``(num_reqs, max_pages)`` **engine** pages — not kernel pages.  The sub-page expansion is
    done on the device by ``serving.step.refresh_step``."""

    slot_mapping: torch.Tensor
    """``(num_tokens,)`` flat token slots for this step's K/V write."""

    seq_lens_cpu_upper_bound: Optional[torch.Tensor] = None
    """Optional CPU upper bound (``v1/attention/backend.py:412-416``); carried for callers that
    debug ragged reconstruction, never used by the kernels."""

    positions: Optional[torch.Tensor] = None
    """Optional per-token positions (``v1/attention/backend.py:402-405``); unused here."""

    # --- computed scalars ----------------------------------------------------------------------
    num_reqs: int = 0
    """Number of *real* requests (padding rows are not counted)."""

    num_actual_tokens: int = 0
    """Total number of query tokens scheduled this step."""

    max_query_len: int = 1
    """Longest query in the batch; ``> 1`` means prefill/extend (see the module docstring)."""

    max_seq_len: int = 0
    """Longest context length (may be an upper bound), ``v1/attention/backend.py:382-383``."""

    bucket: int = 1
    """Batch bucket this step is padded to (``serving.buckets.bucket_for``)."""

    block_size: int = 0
    """Engine page size in tokens (0 = "ask the KV pool", which the impl does)."""

    def __post_init__(self) -> None:
        if self.num_reqs < 1:
            raise ValueError(f"num_reqs must be >= 1, got {self.num_reqs}")
        if self.num_actual_tokens < 0:
            raise ValueError(f"num_actual_tokens must be >= 0, got {self.num_actual_tokens}")
        if self.bucket < self.num_reqs:
            raise ValueError(
                f"bucket {self.bucket} is smaller than num_reqs {self.num_reqs}; padding can only "
                "grow a batch"
            )
        if self.max_query_len < 1:
            raise ValueError(f"max_query_len must be >= 1, got {self.max_query_len}")
        if self.max_query_len <= 1 and self.num_actual_tokens not in (0, self.num_reqs):
            # A decode batch stages one query row per request, in request order.  That mapping
            # only exists when every request contributes exactly one token; a request with zero
            # scheduled tokens would shift every later row and produce plausible garbage, so it is
            # rejected here (at build time) rather than served by the kernel.
            raise ValueError(
                f"a decode batch (max_query_len={self.max_query_len}) must have one query token "
                f"per request, but num_actual_tokens={self.num_actual_tokens} != "
                f"num_reqs={self.num_reqs}; zero-token requests are not representable"
            )

    # ---------------------------------------------------------------- derived predicates

    @property
    def is_decode(self) -> bool:
        """Every request contributes exactly one query token.

        This is the *only* decode/prefill discriminator used by the backend: the paged kernel has
        no per-request mask, so a batch where some requests have several query tokens and others
        have one is a prefill batch as far as this backend is concerned.
        """
        return self.max_query_len <= 1

    @property
    def is_prefill(self) -> bool:
        """Complement of :attr:`is_decode` (``max_query_len > 1``)."""
        return not self.is_decode

    @property
    def padding_rows(self) -> int:
        """Requests the kernels see beyond the real batch (never read, must stay valid)."""
        return self.bucket - self.num_reqs

    def __repr__(self) -> str:  # pragma: no cover - logging aid
        return (
            f"TileInferMetadata(reqs={self.num_reqs}, tokens={self.num_actual_tokens}, "
            f"bucket={self.bucket}, max_query_len={self.max_query_len}, "
            f"max_seq_len={self.max_seq_len})"
        )


class TileInferMetadataBuilder(AttentionMetadataBuilder):
    """Builds :class:`TileInferMetadata` from the engine's common metadata.

    The constructor signature is the ABC's (`v1/attention/backend.py:535-546`); vLLM passes the
    KV-cache spec, the layer names, the full config and the device.  Only three things are read:
    ``kv_cache_spec.block_size`` (the start-up block-size check), the device, and
    ``vllm_config.scheduler_config.max_num_seqs`` (to bound the bucket list, so a deployment never
    compiles a bucket the engine cannot fill — ``serving.buckets.buckets_for``).
    """

    #: We never reorder the batch.  The ABC documents ``None`` as "this backend does not reorder"
    #: (`v1/attention/backend.py:527-530`); the engine's own reorder (decode -> extend -> prefill)
    #: is left untouched, and the kernels accept whatever order it produces.
    reorder_batch_threshold = None

    #: Capturable only for a batch of single-token requests (see the module docstring).  This is
    #: the ABC's ClassVar (`v1/attention/backend.py:524-526`) read by
    #: ``get_cudagraph_support`` (`:548-555`).
    _cudagraph_support = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

    #: Not advertised: ``AttentionMetadataBuilder.update_block_table`` (`:609-622`) is only called
    #: when this is True, and its call site is not part of the fetched 0.23 subset.  ``build`` is
    #: cheap (integers + references), so we take the boring path instead of an unverified one.
    supports_update_block_table = False

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config,
        kv_cache_spec,
    ) -> AttentionCGSupport:
        """Windowed specs are eager-only; everything else keeps the class-level answer.

        A sliding-window layer takes the staged scratch path on decode (``attn.window``) and the
        masked reference on prefill — neither is verified inside a captured region (the prefill
        path walks the page table on the host), so the support is downgraded here, where vLLM
        can still honour it.  The spec carries the window even in the unified-spec mode (vLLM
        converts ``SlidingWindowSpec`` to ``FullAttentionSpec`` and keeps ``sliding_window``),
        so this sees every windowed configuration.
        """
        if getattr(kv_cache_spec, "sliding_window", None):
            return AttentionCGSupport.NEVER
        return cls._cudagraph_support

    def __init__(
        self,
        kv_cache_spec,
        layer_names: list[str],
        vllm_config,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.device = device
        self.layer_names = list(layer_names)

        block_size = getattr(kv_cache_spec, "block_size", None)
        if block_size is not None:
            from tileinfer.serving.buckets import sub_pages

            # Fail while the model is being configured, not on the first step.  `sub_pages`
            # raises for anything that is not a positive multiple of 128: an engine page that is
            # not a whole number of kernel pages has no sub-page layout at all.
            sub_pages(int(block_size))
            self.block_size = int(block_size)
        else:
            self.block_size = 0

        max_num_seqs = _max_num_seqs(vllm_config)
        self.max_num_seqs = max_num_seqs
        # Bound the bucket set by the engine's own limit: compiling a bucket nobody can reach is
        # ~70 s of pre-warm per shape (serving/buckets.py:93-100).
        self.buckets = buckets_for(max_num_seqs)

    # ------------------------------------------------------------------ build

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> TileInferMetadata:
        """Wrap the engine's metadata; no tensor is copied and no device value is read.

        ``common_prefix_len`` is the cascade-attention prefix length
        (`v1/attention/backend.py:589-607`).  This builder returns ``use_cascade_attention() ==
        False`` (the ABC default, `:658-670`), so upstream never passes a non-zero prefix; if it
        ever does, the value would silently change the result, so it is rejected instead.
        """
        if common_prefix_len:
            raise ValueError(
                f"common_prefix_len={common_prefix_len} requested, but the TileInfer backend does "
                "not implement cascade attention (use_cascade_attention() is False); this "
                "metadata cannot describe a shared prefix"
            )

        cam = common_attn_metadata
        num_reqs = int(cam.num_reqs)
        num_actual_tokens = int(cam.num_actual_tokens)
        max_query_len = self._max_query_len(cam)

        if num_reqs > self.max_num_seqs:
            raise ValueError(
                f"batch of {num_reqs} requests exceeds max_num_seqs={self.max_num_seqs}; the "
                "engine and the backend disagree about the batch bound"
            )
        bucket = bucket_for(num_reqs, self.buckets)

        METADATA_COUNTERS.bump("builds")
        METADATA_COUNTERS.bump("decode_builds" if max_query_len <= 1 else "prefill_builds")
        padding = bucket - num_reqs
        if padding:
            METADATA_COUNTERS.bump("bucket_padding_rows", padding)

        return TileInferMetadata(
            query_start_loc=cam.query_start_loc,
            query_start_loc_cpu=cam.query_start_loc_cpu,
            seq_lens=cam.seq_lens,
            block_table_tensor=cam.block_table_tensor,
            slot_mapping=cam.slot_mapping,
            seq_lens_cpu_upper_bound=getattr(cam, "seq_lens_cpu_upper_bound", None),
            positions=getattr(cam, "positions", None),
            num_reqs=num_reqs,
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            max_seq_len=int(cam.max_seq_len),
            bucket=bucket,
            block_size=self.block_size,
        )

    # ------------------------------------------------------------------ internals

    @staticmethod
    def _max_query_len(cam: CommonAttentionMetadata) -> int:
        """``max_query_len``, cross-checked against the CPU start locations.

        The declared field (`v1/attention/backend.py:380-381`) is authoritative — during graph
        capture the CPU copy may contain padded rows and the declared value is what the buffers
        were sized for.  It is still compared with ``diff(query_start_loc_cpu)`` (a host tensor,
        so no sync) because a *too small* value would under-size the query staging buffer and
        corrupt output silently: that case is repaired and counted.
        """
        declared = int(cam.max_query_len)
        qsl_cpu = cam.query_start_loc_cpu
        derived = 0
        if qsl_cpu is not None and int(qsl_cpu.numel()) >= 2:
            query_lens = qsl_cpu[1:] - qsl_cpu[:-1]
            if int(query_lens.numel()):
                derived = int(query_lens.max().item())
        if derived > declared:
            METADATA_COUNTERS.bump("max_query_len_repaired")
            return derived
        return max(declared, 1)


def _max_num_seqs(vllm_config) -> int:
    """``vllm_config.scheduler_config.max_num_seqs`` with a conservative fallback.

    The builder is also constructed by tests and by tools that pass a partial config, so the
    attribute chain is read defensively; the fallback is the largest bucket, which is the safe
    direction (it can only *offer* buckets, never truncate a real batch).
    """
    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    max_num_seqs = getattr(scheduler_config, "max_num_seqs", None)
    if max_num_seqs is None:
        return BATCH_BUCKETS[-1]
    return max(1, int(max_num_seqs))
