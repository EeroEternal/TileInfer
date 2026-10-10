"""Engine-agnostic attention metadata.

Everything TileInfer needs in order to describe a serving batch lives in this module.  The
vocabulary deliberately mirrors FlashInfer (``qo_indptr`` / ``kv_indptr`` / ``kv_indices`` /
``kv_last_page_len``) so that an engine can be wired up with a few lines of glue code, and so
that people who already know FlashInfer feel at home.

Two representations are supported and can be converted into each other:

``RaggedMetadata``
    the *serving-native* form: a flat list of physical pages plus per-request pointers.  This is
    what the kernels consume and it is independent of any engine's KV-cache bookkeeping.

``PageTable``
    the *dense* form (``block_tables`` in vLLM): ``[batch, max_pages_per_seq]`` with padding.
    Conversión to ragged form removes the padding and is therefore strictly cheaper to execute.

Nothing in this module touches a specific device or backend: every tensor is a plain
``torch.Tensor`` and every scalar a plain Python int.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Optional, Sequence, Tuple, Union

import torch

__all__ = [
    "AttentionMode",
    "ragged_from_page_table_device",
    "pad_batch",
    "RaggedMetadata",
    "PageTable",
    "as_int32",
    "detect_mode",
    "kv_lens_from_indptr",
    "pad_to_multiple",
]

IntLike = Union[int, Sequence[int], torch.Tensor]


class AttentionMode(str, Enum):
    """Which flavour of attention a batch is.

    The kernel selection is driven by this value, not by the engine:

    ``DECODE``
        one query token per request (``qo_len == 1``); the case TPOT cares about.
    ``PREFILL``
        ``qo_len > 1`` and the query covers the whole KV sequence (no cached prefix).
    ``APPEND``
        ``qo_len > 1`` with a non-empty cached prefix — chunked prefill, speculative
        decoding, prefix caching.  Uses causal masking with an offset.
    """

    DECODE = "decode"
    PREFILL = "prefill"
    APPEND = "append"

    @staticmethod
    def from_str(value: Union[str, "AttentionMode"]) -> "AttentionMode":
        if isinstance(value, AttentionMode):
            return value
        try:
            return AttentionMode(str(value).lower())
        except ValueError as exc:  # pragma: no cover - trivial
            raise ValueError(
                f"unknown attention mode {value!r}; expected one of "
                f"{[m.value for m in AttentionMode]}"
            ) from exc


def as_int32(value: IntLike, device: Optional[torch.device] = None) -> torch.Tensor:
    """Normalise ``value`` into a contiguous ``int32`` tensor.

    Accepts python ints / sequences / tensors so that engines can pass whatever they have
    (numpy arrays, lists, device tensors) without writing glue conversions.
    """
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
        if tensor.dtype != torch.int32:
            tensor = tensor.to(torch.int32)
        if device is not None and tensor.device != torch.device(device):
            tensor = tensor.to(device)
        return tensor.contiguous()
    if isinstance(value, int):
        return torch.tensor([value], dtype=torch.int32, device=device)
    return torch.tensor(list(value), dtype=torch.int32, device=device)


def pad_to_multiple(value: int, multiple: int) -> int:
    """Round ``value`` up to the next multiple of ``multiple``."""
    if multiple <= 0:
        raise ValueError("multiple must be positive")
    return ((value + multiple - 1) // multiple) * multiple


def kv_lens_from_indptr(
    kv_indptr: torch.Tensor,
    kv_last_page_len: Optional[torch.Tensor] = None,
    page_size: int = 1,
) -> torch.Tensor:
    """KV length of every request, in tokens.

    With ``page_size == 1`` the ragged metadata degenerates to token granularity and the
    length is simply ``diff(kv_indptr)``.  Otherwise the last page is only partially filled
    and ``kv_last_page_len`` carries the number of valid tokens in it.
    """
    indptr = kv_indptr
    if page_size == 1:
        return (indptr[1:] - indptr[:-1]).to(torch.int32)
    if kv_last_page_len is None:
        raise ValueError("kv_last_page_len is required when page_size > 1")
    pages = indptr[1:] - indptr[:-1]
    return ((pages - 1) * page_size + kv_last_page_len).to(torch.int32)


@dataclass
class PageTable:
    """Dense page table (``block_tables`` in vLLM / MindIE).

    ``table[b, i]`` is the physical page holding request ``b``'s ``i``-th KV page, with
    ``seq_lens`` giving the token length of each request.  Entries beyond
    ``ceil(seq_len / page_size)`` are padding.
    """

    table: torch.Tensor  # [batch, max_pages] int32
    seq_lens: torch.Tensor  # [batch] int32
    page_size: int = 128

    def __post_init__(self) -> None:
        if self.table.dim() != 2:
            raise ValueError(f"page table must be 2-D, got shape {tuple(self.table.shape)}")
        if self.seq_lens.numel() != self.table.shape[0]:
            raise ValueError(
                f"seq_lens has {self.seq_lens.numel()} entries but table has "
                f"{self.table.shape[0]} rows"
            )

    @property
    def batch_size(self) -> int:
        return self.table.shape[0]

    @property
    def pages_per_seq(self) -> torch.Tensor:
        """Number of *used* pages per request."""
        return (self.seq_lens + self.page_size - 1) // self.page_size

    def to_ragged(self) -> "RaggedMetadata":
        """Compact the table into ``indptr`` / ``indices`` / ``last_page_len`` form.

        Padding entries are dropped, so the kernels touch strictly less memory than when
        consuming the dense table directly.
        """
        pages = self.pages_per_seq.to(torch.int64)
        indptr = torch.zeros(self.batch_size + 1, dtype=torch.int32, device=self.table.device)
        indptr[1:] = torch.cumsum(pages, dim=0).to(torch.int32)
        total = int(indptr[-1].item())
        indices = torch.empty(total, dtype=torch.int32, device=self.table.device)
        offset = 0
        for b in range(self.batch_size):
            n = int(pages[b].item())
            if n:
                indices[offset : offset + n] = self.table[b, :n]
            offset += n
        last_page_len = (
            (self.seq_lens - 1) % self.page_size + 1
            if self.page_size > 1
            else torch.ones_like(self.seq_lens)
        )
        return RaggedMetadata(
            kv_indptr=indptr,
            kv_indices=indices,
            kv_last_page_len=last_page_len.to(torch.int32),
            page_size=self.page_size,
        )

    @staticmethod
    def from_seq_lens(seq_lens: IntLike, page_size: int = 128) -> "PageTable":
        """Build a *dense, contiguous* page table — mainly useful in tests and benchmarks."""
        lens = as_int32(seq_lens)
        batch = lens.numel()
        max_pages = int(((lens + page_size - 1) // page_size).max().item()) if batch else 0
        pages = (lens + page_size - 1) // page_size
        table = torch.zeros((batch, max_pages), dtype=torch.int32, device=lens.device)
        cursor = 0
        for b in range(batch):
            n = int(pages[b].item())
            table[b, :n] = torch.arange(cursor, cursor + n, dtype=torch.int32, device=lens.device)
            cursor += n
        return PageTable(table=table, seq_lens=lens, page_size=page_size)


@dataclass
class RaggedMetadata:
    """Serving-native KV-cache metadata.

    Attributes
    ----------
    kv_indptr:
        ``[batch + 1]`` cumulative *page* counts.  ``kv_indptr[b]:kv_indptr[b + 1]`` selects
        the pages of request ``b`` inside ``kv_indices``.
    kv_indices:
        ``[num_pages]`` physical page ids, in logical order.
    kv_last_page_len:
        ``[batch]`` number of valid tokens in the last page of each request.  Required when
        ``page_size > 1``.
    page_size:
        ``1`` means token granularity (no paging); anything else is a real paged cache.
    qo_indptr:
        optional ``[batch + 1]`` cumulative query lengths.  Required for prefill/append.
    """

    kv_indptr: torch.Tensor
    kv_indices: torch.Tensor
    kv_last_page_len: Optional[torch.Tensor] = None
    page_size: int = 1
    qo_indptr: Optional[torch.Tensor] = None
    _num_pages: Optional[int] = field(default=None, repr=False)

    # ------------------------------------------------------------------ properties

    @property
    def batch_size(self) -> int:
        return int(self.kv_indptr.numel() - 1)

    @property
    def num_pages(self) -> int:
        if self._num_pages is not None:
            return self._num_pages
        return int(self.kv_indices.numel())

    @property
    def device(self) -> torch.device:
        return self.kv_indptr.device

    @property
    def kv_lens(self) -> torch.Tensor:
        """Per-request KV length in tokens, computed — never materialised in the metadata."""
        return kv_lens_from_indptr(self.kv_indptr, self.kv_last_page_len, self.page_size)

    @property
    def qo_lens(self) -> Optional[torch.Tensor]:
        if self.qo_indptr is None:
            return None
        return (self.qo_indptr[1:] - self.qo_indptr[:-1]).to(torch.int32)

    @property
    def max_kv_len(self) -> int:
        lens = self.kv_lens
        return int(lens.max().item()) if lens.numel() else 0

    @property
    def total_kv_pages(self) -> int:
        return int(self.kv_indptr[-1].item()) if self.kv_indptr.numel() else 0

    # ------------------------------------------------------------------ construction

    @staticmethod
    def from_page_table(
        block_table: torch.Tensor,
        seq_lens: IntLike,
        page_size: int = 128,
    ) -> "RaggedMetadata":
        """Convenience wrapper: dense engine-side table → ragged metadata."""
        return PageTable(
            table=block_table, seq_lens=as_int32(seq_lens, block_table.device), page_size=page_size
        ).to_ragged()

    @staticmethod
    def contiguous(
        kv_lens: IntLike,
        page_size: int = 1,
        device: Optional[torch.device] = None,
        qo_lens: Optional[IntLike] = None,
    ) -> "RaggedMetadata":
        """Build metadata for a *contiguously laid out* cache (tests, benchmarks, CI)."""
        lens = as_int32(kv_lens, device)
        if page_size <= 1:
            indptr = torch.zeros(lens.numel() + 1, dtype=torch.int32, device=lens.device)
            indptr[1:] = torch.cumsum(lens, dim=0).to(torch.int32)
            indices = torch.arange(int(indptr[-1].item()), dtype=torch.int32, device=lens.device)
            last_page_len = torch.ones_like(lens)
        else:
            pages = (lens + page_size - 1) // page_size
            indptr = torch.zeros(lens.numel() + 1, dtype=torch.int32, device=lens.device)
            indptr[1:] = torch.cumsum(pages, dim=0).to(torch.int32)
            indices = torch.arange(int(indptr[-1].item()), dtype=torch.int32, device=lens.device)
            last_page_len = ((lens - 1) % page_size + 1).to(torch.int32)
        qo_indptr = None
        if qo_lens is not None:
            ql = as_int32(qo_lens, lens.device)
            qo_indptr = torch.zeros(ql.numel() + 1, dtype=torch.int32, device=lens.device)
            qo_indptr[1:] = torch.cumsum(ql, dim=0).to(torch.int32)
        return RaggedMetadata(
            kv_indptr=indptr,
            kv_indices=indices,
            kv_last_page_len=last_page_len,
            page_size=page_size,
            qo_indptr=qo_indptr,
        )

    def to_page_table(self, max_pages: Optional[int] = None) -> PageTable:
        """Ragged metadata → dense page table (round-trip helper, used in tests)."""
        pages_per_seq = (self.kv_indptr[1:] - self.kv_indptr[:-1]).to(torch.int64)
        max_pages = max_pages or (int(pages_per_seq.max().item()) if pages_per_seq.numel() else 0)
        table = torch.zeros(
            (self.batch_size, max_pages), dtype=torch.int32, device=self.kv_indptr.device
        )
        for b in range(self.batch_size):
            start = int(self.kv_indptr[b].item())
            end = int(self.kv_indptr[b + 1].item())
            table[b, : end - start] = self.kv_indices[start:end]
        return PageTable(table=table, seq_lens=self.kv_lens, page_size=max(self.page_size, 1))

    # ------------------------------------------------------------------ checks

    def validate(self) -> "RaggedMetadata":
        """Cheap structural validation; raises ``ValueError`` on inconsistent metadata."""
        if self.kv_indptr.dim() != 1:
            raise ValueError("kv_indptr must be 1-D")
        if self.kv_indices.dim() != 1:
            raise ValueError("kv_indices must be 1-D")
        if self.kv_indptr.numel() < 1:
            raise ValueError("kv_indptr must contain at least one element")
        if not torch.all(self.kv_indptr[1:] >= self.kv_indptr[:-1]):
            raise ValueError("kv_indptr must be non-decreasing")
        if int(self.kv_indptr[-1].item()) > self.kv_indices.numel():
            raise ValueError(
                f"kv_indptr[-1]={int(self.kv_indptr[-1].item())} exceeds the number of "
                f"page indices ({self.kv_indices.numel()})"
            )
        if self.page_size > 1 and self.kv_last_page_len is None:
            raise ValueError("kv_last_page_len is required when page_size > 1")
        if self.kv_last_page_len is not None:
            if self.kv_last_page_len.numel() != self.batch_size:
                raise ValueError("kv_last_page_len must have one entry per request")
            if self.page_size > 1 and bool((self.kv_last_page_len > self.page_size).any()):
                raise ValueError("kv_last_page_len entries must not exceed page_size")
        if self.page_size > 1 and bool((self.kv_lens <= 0).any()) and self.batch_size:
            raise ValueError("every request must have a positive KV length")
        return self

    # ------------------------------------------------------------------ helpers

    def signature(self) -> Tuple[Any, ...]:
        """Hashable description of the *shape* of this batch, used as a plan cache key."""
        return (
            self.batch_size,
            self.page_size,
            self.num_pages,
            int(self.kv_lens.max().item()) if self.batch_size else 0,
            int(self.kv_lens.min().item()) if self.batch_size else 0,
            str(self.kv_indptr.device),
        )


def ragged_from_page_table_device(
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    page_size: int,
    device: Optional[torch.device] = None,
) -> RaggedMetadata:
    """Dense page table → ragged metadata, **on the device and without host syncs**.

    Engines hand over a dense ``block_tables`` whose *contents* change every step (a new token needs
    a new slot, a new page).  :meth:`PageTable.to_ragged` cannot be used there: it walks requests in
    a Python loop and therefore costs a host sync per request, which is exactly what a captured step
    must not do.  This does the same conversion with vectorised tensor ops instead:

    * ``kv_indptr`` = cumulative used pages,
    * ``kv_indices`` = the used entries of the table, in row-major order (i.e. per request, in logical
      page order — which is what the kernels rely on),
    * ``kv_last_page_len`` = ``(seq_len - 1) % page_size + 1``.

    The number of pages covered is the table's *shape*, so the metadata is stable across steps and a
    plan built from it stays valid while the contents move.
    """
    device = device or block_table.device
    table = block_table.to(device)
    lens = as_int32(seq_lens, device)
    pages_per_seq = (lens + page_size - 1) // page_size  # [batch]

    indptr = torch.zeros(lens.numel() + 1, dtype=torch.int32, device=device)
    indptr[1:] = torch.cumsum(pages_per_seq.to(torch.int64), dim=0).to(torch.int32)

    max_pages = table.shape[1]
    cols = torch.arange(max_pages, dtype=torch.int32, device=device).unsqueeze(0)
    used = cols < pages_per_seq.unsqueeze(1)  # [batch, max_pages]
    indices = table.to(torch.int32).masked_select(used)  # row-major: request by request, in order
    last_page_len = ((lens - 1) % page_size + 1).to(torch.int32)

    return RaggedMetadata(
        kv_indptr=indptr,
        kv_indices=indices.contiguous(),
        kv_last_page_len=last_page_len,
        page_size=page_size,
    )


def pad_batch(meta: RaggedMetadata, bucket: int, page_size: int, dummy_page: int = 0) -> RaggedMetadata:
    """Pad a batch up to ``bucket`` requests with a harmless dummy request.

    Serving steps change batch size constantly; a kernel compiled for every size would recompile
    forever, so engines run a handful of buckets instead.  Padded requests must still look *valid* to
    the kernel - one page with one valid token - otherwise the softmax denominator is zero and the
    padded rows come out as ``0/0``.  Their outputs are simply never read.
    """
    if meta.batch_size > bucket:
        raise ValueError(f"batch of {meta.batch_size} does not fit the bucket {bucket}")
    pad = bucket - meta.batch_size
    if pad == 0:
        return meta

    device = meta.device
    indptr = torch.cat(
        [meta.kv_indptr, meta.kv_indptr[-1] + torch.arange(1, pad + 1, dtype=torch.int32, device=device)]
    )
    indices = torch.cat(
        [meta.kv_indices, torch.full((pad,), dummy_page, dtype=torch.int32, device=device)]
    )
    assert meta.kv_last_page_len is not None
    last_page_len = torch.cat(
        [meta.kv_last_page_len, torch.ones(pad, dtype=torch.int32, device=device)]
    )
    qo_indptr = None
    if meta.qo_indptr is not None:
        qo_indptr = torch.cat(
            [
                meta.qo_indptr,
                meta.qo_indptr[-1] + torch.arange(1, pad + 1, dtype=torch.int32, device=device),
            ]
        )
    return RaggedMetadata(
        kv_indptr=indptr,
        kv_indices=indices,
        kv_last_page_len=last_page_len,
        page_size=max(page_size, meta.page_size, 1),
        qo_indptr=qo_indptr,
    )


def detect_mode(
    kv_lens: torch.Tensor,
    qo_lens: Optional[torch.Tensor] = None,
) -> AttentionMode:
    """Infer prefill / append / decode from the ragged metadata itself.

    A batch of one-token queries is a decode; multi-token queries whose length equals the KV
    length are a prefill; anything else is an append (chunked prefill / speculative step).
    """
    if qo_lens is None:
        return AttentionMode.DECODE
    if qo_lens.numel() == 0:
        return AttentionMode.DECODE
    if bool((qo_lens == 1).all()):
        return AttentionMode.DECODE
    if bool((qo_lens == kv_lens[: qo_lens.numel()]).all()):
        return AttentionMode.PREFILL
    return AttentionMode.APPEND


def iter_pages(indptr: torch.Tensor, indices: torch.Tensor) -> Iterable[torch.Tensor]:
    """Yield the page ids of every request (debug helper, host side)."""
    for b in range(indptr.numel() - 1):
        yield indices[int(indptr[b].item()) : int(indptr[b + 1].item())]
