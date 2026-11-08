# SPDX-License-Identifier: Apache-2.0
"""File-backed, reclaimable storage for vLLM CPU-offloaded weights.

The stock ``PrefetchOffloader`` keeps every offloaded parameter in anonymous
(optionally pinned) host memory.  On a 628 GB single-card host that means the
475 GiB V4.1-Flash checkpoint leaves no room for the Engram host tables: the
engine is OOM-killed during ``process_weights_after_loading`` / host-NZ
preparation.

This module replaces the anonymous allocation with a ``MAP_SHARED`` mapping of
a scratch file.  The pages are clean page cache, so the kernel can evict them
under pressure and fault them back from disk on demand.  Nothing is lost:
weights are written once and only read afterwards.

The store is intentionally dependency free (no numpy) and degrades to ``None``
when ``VLLM_CPU_WEIGHT_MMAP_DIR`` is unset, so the default vLLM path is
unchanged.
"""

from __future__ import annotations

import gc
import itertools
import mmap
import os
import re

import torch

_COUNTER = itertools.count()
_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def mmap_dir() -> str:
    return os.environ.get("VLLM_CPU_WEIGHT_MMAP_DIR", "")


def _contiguous_nbytes(size, stride, itemsize: int) -> int | None:
    """Return the storage byte size iff ``(size, stride)`` is dense row-major."""
    if not size or any(int(s) == 0 for s in size):
        return None
    expected = 1
    reversed_strides = []
    for dim in reversed(size):
        reversed_strides.append(expected)
        expected *= int(dim)
    if tuple(int(s) for s in stride) != tuple(reversed(reversed_strides)):
        return None
    return expected * itemsize


class MmapParamStore:
    """One scratch-file mapping backing a single offloaded CPU parameter."""

    def __init__(self, path: str, size, stride, dtype):
        itemsize = torch.empty((), dtype=dtype).element_size()
        nbytes = _contiguous_nbytes(size, stride, itemsize)
        if nbytes is None:
            raise ValueError("non-contiguous parameter")
        self.path = path
        self.nbytes = nbytes
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.ftruncate(fd, nbytes)
            self._mm = mmap.mmap(fd, nbytes, access=mmap.ACCESS_WRITE)
        finally:
            os.close(fd)
        raw = torch.frombuffer(self._mm, dtype=torch.uint8, count=nbytes)
        tensor = raw.view(dtype).view(tuple(int(s) for s in size))
        self.tensor = tensor

    @classmethod
    def create(cls, module, param_name: str, src: torch.Tensor):
        root = mmap_dir()
        if not root:
            return None
        if src.layout != torch.strided:
            return None
        try:
            itemsize = torch.empty((), dtype=src.dtype).element_size()
            nbytes = _contiguous_nbytes(src.size(), src.stride(), itemsize)
            if nbytes is None:
                return None
            os.makedirs(root, exist_ok=True)
            tag = _SAFE.sub("_", param_name.split(".")[-1])[:40]
            path = os.path.join(root, f"{next(_COUNTER):05d}_{os.getpid():x}_{id(module):x}_{tag}.bin")
            return cls(path, src.size(), src.stride(), src.dtype)
        except (OSError, ValueError, RuntimeError):
            return None

    def flush(self) -> None:
        if os.environ.get("VLLM_CPU_WEIGHT_MMAP_SYNC", "0") not in ("", "0"):
            try:
                self._mm.flush()
            except (OSError, ValueError):
                pass

    def drop_pages(self) -> None:
        """Ask the kernel to evict the resident pages (data stays in the file)."""
        try:
            self._mm.madvise(mmap.MADV_DONTNEED)
        except (AttributeError, OSError, ValueError):
            pass

    def release(self) -> None:
        self.tensor = None
        gc.collect()
        self.drop_pages()
        try:
            self._mm.close()
        except (BufferError, OSError, ValueError):
            pass
        try:
            os.unlink(self.path)
        except OSError:
            pass
