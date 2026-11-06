"""Engine-independent serving arithmetic: batch buckets and 128-token kernel pages.

Why this module exists
----------------------
The TileLang kernels tile a **128-token page** (``KERNEL_PAGE_SIZE``): that is their ``BC`` N
tile, sized to fit the 64 KB L0 buffers.  Serving engines, on the other hand, pick their own page
size, and hybrid models (Qwen3.6: 30 GDN layers + 10 attention layers) force a much larger one -
2048 tokens, aligned with the mamba state page.  A ``block_size = k * 128`` engine page is exactly
``k`` *consecutive* 128-token sub-pages in memory, so the pool is presented as a zero-copy
sub-page view and the engine's page table is expanded to sub-page ids (``page * k + i``).

Everything here was lifted from the vLLM-Ascend adapter with the semantics unchanged - including
two failure modes that are silent when you get them wrong:

* **sub-page mapping** (``expand_page_table``): a column ``j`` of the table maps to ``k``
  consecutive sub-page ids, in order.  Off-by-one ordering reads the wrong token positions and
  produces plausible-looking garbage;
* **bucket padding** (``bucket_for``): the kernel is compiled per batch bucket, so a batch is
  padded *up* to the next bucket.  Padding rows must stay valid (a real page, length >= 1) or the
  softmax denominator becomes NaN;
* **the fused pool's K/V axis** (``fused_halves``): vLLM keeps K and V in one tensor, and which
  dimension holds the ``2`` decides whether each half is a contiguous page array the kernel can
  read.  Reading the wrong axis swaps K and V - again silent, again plausible-looking output.

Dependency rule: this module may import ``torch`` but **never** ``vllm``, ``torch_npu`` or
the legacy adapter under ``legacy/`` - the arithmetic above is the part that must stay testable on
a CPU laptop (``tests/test_serving_buckets.py``).
"""

from __future__ import annotations

import os
from typing import Optional, Sequence, Tuple

import torch

__all__ = [
    "KERNEL_PAGE_SIZE",
    "BATCH_BUCKETS",
    "next_pow2",
    "sub_pages",
    "sub_page_width",
    "buckets_for",
    "bucket_for",
    "expand_page_table",
    "fused_halves",
    "capacity_view",
    "pool_sub_view",
    "last_page_len",
    "last_page_len_",
    "pages_for_lens",
    "padded_rows",
    "declared_pool_cap",
]

#: Tokens per kernel page.  Matches the kernels' ``BC`` N tile; see the module docstring.
KERNEL_PAGE_SIZE = 128

#: Batch sizes the decode kernel is compiled for.  Serving batches cluster well below these
#: ceilings, so the padding waste is far cheaper than recompiling per step.  ``max_num_seqs``
#: truncates the list at start-up (see :func:`buckets_for`).
BATCH_BUCKETS: Tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256)


def next_pow2(n: int) -> int:
    """Smallest power of two ``>= n`` (``1`` stays ``1``).

    Used to derive a *declared* KV-pool capacity, so a pre-warm compiled for one pool size keeps
    matching a pool that drifted by a few pages between runs (the engine computes its pool from a
    memory snapshot, so it is not reproducible).
    """
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def sub_pages(block_size: int) -> int:
    """How many 128-token sub-pages one engine page holds (``1`` means no transformation).

    Raises rather than rounding: a page of 192 tokens has no sub-page layout, and silently
    truncating it would mis-address the second half of every page.
    """
    if block_size < KERNEL_PAGE_SIZE or block_size % KERNEL_PAGE_SIZE != 0:
        raise ValueError(
            f"block_size={block_size} is not a positive multiple of {KERNEL_PAGE_SIZE}; "
            "the TileInfer kernels tile 128-token pages and cannot address this layout"
        )
    return block_size // KERNEL_PAGE_SIZE


def sub_page_width(max_pages: int, block_size: int) -> int:
    """Width, in kernel pages, of a page table whose engine-side width is ``max_pages``."""
    if max_pages <= 0:
        raise ValueError(f"max_pages must be positive, got {max_pages}")
    return max_pages * sub_pages(block_size)


def buckets_for(max_num_reqs: int, buckets: Sequence[int] = BATCH_BUCKETS) -> Tuple[int, ...]:
    """The buckets a deployment will actually use: everything ``<= max_num_reqs``.

    Compiling a bucket nobody can reach is ~70 s of pre-warm per shape, so the declared set is
    derived from the server config instead of being compiled wholesale.
    """
    kept = tuple(b for b in buckets if b <= max(1, int(max_num_reqs)))
    return kept or (1,)


def bucket_for(num_reqs: int, buckets: Sequence[int] = BATCH_BUCKETS) -> int:
    """First bucket ``>= num_reqs``; raises when the batch exceeds every bucket.

    A raise (not a silent clamp) is deliberate: clamping would serve a batch through a kernel
    compiled for fewer rows, and the extra requests would read stale output.
    """
    if num_reqs <= 0:
        raise ValueError(f"num_reqs must be positive, got {num_reqs}")
    ordered = tuple(buckets)
    if list(ordered) != sorted(ordered):
        raise ValueError(f"buckets must be ascending, got {ordered}")
    for bucket in ordered:
        if bucket >= num_reqs:
            return bucket
    raise ValueError(
        f"batch of {num_reqs} exceeds the largest bucket ({ordered[-1]}); "
        "add a larger bucket to BATCH_BUCKETS"
    )


def expand_page_table(block_tables: torch.Tensor, sub: int) -> torch.Tensor:
    """Expand a dense table of ``block_size``-token pages into 128-token sub-page ids.

    Column ``j * sub + i`` of the result holds ``table[j] * sub + i``: sub-page ``i`` of engine
    page ``table[j]``.  Unused columns expand to sub-pages of whatever stale id they hold - the
    ragged rebuild masks them out per request (``j < pages_b``), so their contents never matter.

    ``sub == 1`` returns the input tensor itself (zero-copy); callers must treat the result as
    read-only in that case.
    """
    if sub < 1:
        raise ValueError(f"sub must be >= 1, got {sub}")
    if block_tables.dim() != 2:
        raise ValueError(f"block_tables must be 2-D, got shape {tuple(block_tables.shape)}")
    if sub == 1:
        return block_tables
    scaled = block_tables.to(torch.int32) * sub
    expanded = scaled.repeat_interleave(sub, dim=1)
    offsets = torch.arange(sub, dtype=torch.int32, device=block_tables.device)
    return expanded + offsets.repeat(scaled.shape[1])


def fused_halves(
    kv_cache: torch.Tensor,
    block_dim: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split vLLM's fused NHD pool into its K and V ``[blocks, block_size, heads, dim]`` halves.

    Why the split needs a rule at all
    ---------------------------------
    vLLM's attention pool carries the K/V factor in *one* dimension of a single tensor; which
    dimension that is, is the backend's declaration:

    * ``block_dim == 0`` - ``[blocks, 2, block_size, heads, dim]``.  This is what
      :meth:`tileinfer.vllm.attn.backend.TileInferAttentionBackend.get_kv_cache_shape` returns, so
      that ``get_kv_cache_block_dim`` discovers the *block* in dim 0 and vLLM's hybrid layout
      rewrite (``gpu_model_runner.py:7167-7199``) is skipped: ``block_dim == 0`` makes it
      ``continue`` (``:7190-7191``).  The physical order is declared by
      ``get_kv_cache_stride_order() == (1, 0, 2, 3, 4)``, i.e. ``[2][blocks][block_size][heads]
      [dim]``, which is what keeps each half a *contiguous* ``[blocks, block_size, heads, dim]``
      tensor - the one property ``pool_sub_view`` and the kernel need.
    * ``block_dim == 1`` - ``[2, blocks, block_size, heads, dim]``, the vLLM-native form.  Its
      halves are contiguous only while nothing has rewritten the strides; the hybrid rewrite turns
      this same *shape* into a block-major ``[blocks][2][...]`` physical layout, whose halves are
      strided (that form is still accepted here, and by the write, because a vLLM point release
      could hand it to us - but the kernel cannot read it, see :func:`pool_sub_view`).

    The two orders are indistinguishable from metadata when the pool has exactly two blocks; that
    pool cannot serve a real step (``impl._state_for`` needs ``bucket * width < cap - 1``), so the
    ambiguity is rejected instead of guessed at.  Pass ``block_dim`` explicitly there.
    """
    if kv_cache.dim() != 5:
        raise ValueError(
            "the fused KV pool must be 5-D ([blocks, 2, ...] or [2, blocks, ...]), got shape "
            f"{tuple(kv_cache.shape)}"
        )
    blocks_first, kv_first = int(kv_cache.shape[0]), int(kv_cache.shape[1])
    if block_dim is None:
        if kv_first == 2 and blocks_first != 2:
            block_dim = 0
        elif blocks_first == 2 and kv_first != 2:
            block_dim = 1
        else:
            raise ValueError(
                "cannot tell which axis of the fused KV pool is the K/V factor: shape "
                f"{tuple(kv_cache.shape)} has {blocks_first} in dim 0 and {kv_first} in dim 1. "
                "Pass block_dim=0 for the block-major form (get_kv_cache_shape puts num_blocks "
                "first) or block_dim=1 for the legacy (2, num_blocks, ...) form."
            )
    block_dim = int(block_dim)
    if block_dim == 0:
        if kv_first != 2:
            raise ValueError(
                f"block_dim=0 means [blocks, 2, block_size, heads, dim] but dim 1 is {kv_first} "
                f"(shape {tuple(kv_cache.shape)})"
            )
        if blocks_first < 1:
            raise ValueError("the fused KV pool has no blocks")
        return kv_cache[:, 0], kv_cache[:, 1]
    if block_dim == 1:
        if blocks_first != 2:
            raise ValueError(
                f"block_dim=1 means [2, blocks, block_size, heads, dim] but dim 0 is "
                f"{blocks_first} (shape {tuple(kv_cache.shape)})"
            )
        if kv_first < 1:
            raise ValueError("the fused KV pool has no blocks")
        return kv_cache[0], kv_cache[1]
    raise ValueError(f"block_dim must be 0 or 1, got {block_dim}")


def capacity_view(pool: torch.Tensor, cap: int) -> torch.Tensor:
    """Present ``pool`` at the kernel ABI's *declared* page capacity, without copying.

    The kernel (and its plan key) is compiled for a declared capacity that only has to **cover**
    the real pool: every access is bounded by ``kv_indptr``/``kv_indices``, so pages past the real
    pool are never dereferenced.  ``torch.as_strided`` refuses to build a view larger than the
    storage, but ``Tensor.set_`` does not validate, so the capacity view re-binds the pool's own
    storage at the declared extent.

    **The storage offset must be carried over.**  The V half of a fused pool lives at a non-zero
    offset inside the same allocation (``fused_halves``), and re-binding with offset 0 silently
    aliases it onto the K half - the kernel would then attend K twice, with no error anywhere.
    ``set_(storage, offset, ...)`` keeps the half where it is; only the extent is inflated.

    ``cap < pool.shape[0]`` would hide live pages and is a caller bug (rejected).
    """
    if pool.dim() != 4:
        raise ValueError(
            f"capacity_view needs a [pages, page_size, heads, dim] pool, got shape "
            f"{tuple(pool.shape)}"
        )
    cap = int(cap)
    pages = int(pool.shape[0])
    if cap < pages:
        raise ValueError(
            f"pool grew past the kernel's compiled capacity ({pages} > {cap}); "
            "the declared capacity must cover every live page"
        )
    # Same materialisation the previous implementation used: a strided pool is copied here (the
    # caller's bug - pool_sub_view already refuses one), a contiguous pool is its own view.
    flat = pool if pool.is_contiguous() else pool.contiguous()
    if cap == pages:
        return flat
    view = torch.empty(0, dtype=flat.dtype, device=flat.device)
    view.set_(
        flat.untyped_storage(),
        flat.storage_offset(),
        (cap, *flat.shape[1:]),
        flat.stride(),
    )
    return view


def pool_sub_view(pool: torch.Tensor, sub: int) -> torch.Tensor:
    """View a pool of ``block_size``-token NHD pages as ``sub`` times more 128-token pages.

    ``[blocks, block_size, kv_heads, dim]`` -> ``[blocks * sub, 128, kv_heads, dim]``.  This is a
    pure reshape of contiguous memory: the kernel reads the same bytes the engine writes, so no
    repack and no copy happens at the backend boundary.

    A *half* of the fused pool is what this receives (``fused_halves``), and that half is
    contiguous only because ``get_kv_cache_stride_order`` puts K and V outermost.  In the
    block-interleaved layout vLLM would produce for a ``block_dim == 1`` backend there is **no**
    flat ``[pages, 128, heads, dim]`` view at all: a page is addressed by three indices
    ``(block, kv, sub)``, and flattening them into a page id would require copying the pool.  Such
    a pool is rejected below rather than silently repacked.
    """
    if sub < 1:
        raise ValueError(f"sub must be >= 1, got {sub}")
    if pool.dim() != 4:
        raise ValueError(f"KV pool must be 4-D NHD, got shape {tuple(pool.shape)}")
    if not pool.is_contiguous():
        raise ValueError(
            "KV pool must be contiguous for the zero-copy sub-page view: the kernel reads "
            "[pages, 128, heads, dim] as one contiguous block, and a pool whose K/V halves are "
            "interleaved per block ([blocks][2][page][heads][dim]) cannot be re-viewed without a "
            "copy. Present the block-major fused form instead "
            f"(backend.get_kv_cache_stride_order); got strides {pool.stride()} for shape "
            f"{tuple(pool.shape)}"
        )
    if sub == 1:
        # `sub == 1` claims "this pool already has 128-token pages".  A 256- or 2048-token page
        # passed that way would mis-address every page after the first, so reject it instead of
        # returning a pool the kernel would silently misread.
        if pool.shape[1] != KERNEL_PAGE_SIZE:
            raise ValueError(
                f"pool page is {pool.shape[1]} tokens but sub=1 means {KERNEL_PAGE_SIZE}-token "
                "kernel pages; pass sub_pages(block_size) instead"
            )
        return pool
    blocks, block_size, heads, dim = pool.shape
    if block_size != sub * KERNEL_PAGE_SIZE:
        raise ValueError(
            f"pool page is {block_size} tokens but sub={sub} implies "
            f"{sub * KERNEL_PAGE_SIZE}; the view would be misaligned"
        )
    return pool.view(blocks * sub, KERNEL_PAGE_SIZE, heads, dim)


def last_page_len(lens: torch.Tensor, page_size: int = KERNEL_PAGE_SIZE) -> torch.Tensor:
    """Valid tokens in the last page of each sequence: ``((lens - 1) % page_size) + 1``."""
    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}")
    return (lens - 1).remainder(page_size) + 1


def last_page_len_(lens: torch.Tensor, out: torch.Tensor, page_size: int = KERNEL_PAGE_SIZE) -> None:
    """In-place :func:`last_page_len` into a plan-owned buffer (no allocation, capture-safe).

    Kept separate from the functional form because the serving step must not allocate: every
    buffer it writes is owned by the plan and addressed by a stable pointer.
    """
    if out.shape != lens.shape:
        raise ValueError(
            f"out shape {tuple(out.shape)} must match lens shape {tuple(lens.shape)}"
        )
    torch.sub(lens, 1, out=out)
    out.remainder_(page_size)
    out.add_(1)


def pages_for_lens(lens: torch.Tensor, page_size: int = KERNEL_PAGE_SIZE) -> torch.Tensor:
    """Pages needed per sequence: ``ceil(lens / page_size)`` (allocates; plan-time only)."""
    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}")
    return torch.div(lens + (page_size - 1), page_size, rounding_mode="floor")


def padded_rows(group: int) -> int:
    """The kernel's M tile for a GQA group: the group rounded up to a multiple of 32.

    Why 32: the ND->NZ copy template asserts ``ROWS % 16 == 0`` and ``dual_copy`` splits the M
    dimension over the two AIVs, so ``ROWS = BR // 2``.  A bare group of 8 - a very common serving
    shape - therefore cannot compile, and the caller passes zero-padded ``Q``/``Out`` rows.

    This duplicates ``tileinfer.kernels.attention.paged_decode_ascend950.padded_rows`` on purpose:
    that module imports ``tilelang`` at import time and must stay out of the CPU-testable serving
    layer.  ``tests/test_serving_buckets.py`` pins the two to each other where the toolchain exists.
    """
    if group < 1:
        raise ValueError(f"group must be >= 1, got {group}")
    return max(32, ((group + 31) // 32) * 32)


def declared_pool_cap(pool_pages: int, pinned: Optional[int] = None) -> int:
    """The kernel ABI's *declared* page capacity for a real KV pool (pool-agnostic ABI).

    The kernel never dereferences a page outside the ``kv_indptr``/``kv_indices`` spans, so the
    declared capacity only has to *cover* the real pool - its exact value is free.  Rounding up to
    a power of two (or pinning it with ``TILEINFER_POOL_CAP``) removes the pool size from the
    kernel / plan / pre-warm keys: the engine derives its pool from a memory snapshot, so it drifts
    between runs (9425 pages in one run, 9899 in the next) and used to silently invalidate an
    out-of-process pre-warm.

    The caller presents the pool through a zero-copy capacity view, so ``cap >= pool_pages`` is
    what keeps the kernel's fixed-shape read in bounds.
    """
    pool_pages = int(pool_pages)
    if pool_pages < 0:
        raise ValueError(f"pool_pages must be >= 0, got {pool_pages}")
    if pinned is None:
        raw = os.environ.get("TILEINFER_POOL_CAP", "").strip()
        pinned = int(raw) if raw else None
    if pinned is not None:
        if int(pinned) < pool_pages:
            raise ValueError(
                f"TILEINFER_POOL_CAP={pinned} is below the real pool ({pool_pages} pages); "
                "the declared capacity must cover every live page"
            )
        return int(pinned)
    return next_pow2(pool_pages)
