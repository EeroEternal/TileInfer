"""Small device / dtype helpers.

TileInfer must import (and unit-test) fine on a laptop with no NPU, so every device decision
goes through here instead of hard-coding ``"npu"`` anywhere.
"""

from __future__ import annotations

from typing import Optional, Union

import torch

__all__ = [
    "is_npu_available",
    "npu_device_count",
    "resolve_device",
    "device_name",
    "default_dtype",
    "synchronize",
    "have_tilelang",
]

DeviceLike = Union[str, torch.device, None]


def is_npu_available() -> bool:
    """True when a working ``torch_npu`` stack with at least one visible device exists."""
    try:
        import torch_npu  # noqa: F401  (import registers the backend)
    except Exception:
        return False
    try:
        return bool(torch.npu.is_available())
    except Exception:
        return False


def npu_device_count() -> int:
    if not is_npu_available():
        return 0
    try:
        return int(torch.npu.device_count())
    except Exception:
        return 0


def resolve_device(device: DeviceLike = None, prefer_npu: bool = True) -> torch.device:
    """Pick a concrete device.

    ``"auto"`` / ``None`` means "NPU if there is one, otherwise CPU"; an explicit device is
    always honoured, which keeps the reference backend usable on a CPU-only dev box.
    """
    if isinstance(device, torch.device):
        return device
    if device in (None, "auto"):
        if prefer_npu and is_npu_available():
            return torch.device("npu", 0)
        return torch.device("cpu")
    name = str(device)
    if name.startswith("npu"):
        if not is_npu_available():
            raise RuntimeError("requested an NPU device but torch_npu/NPU is unavailable")
        return torch.device(name)
    return torch.device(name)


def device_name(device: torch.device) -> str:
    """Human-readable device name, used in benchmark reports and plan summaries."""
    try:
        if device.type == "npu":
            return str(torch.npu.get_device_name(device))
        if device.type == "cuda":
            return str(torch.cuda.get_device_name(device))
    except Exception:
        pass
    return f"{device.type}:{device.index if device.index is not None else 0}"


def default_dtype(device: torch.device) -> torch.dtype:
    """Preferred compute dtype: fp16 everywhere, bf16 where fp16 is not the fast path."""
    return torch.float16


def synchronize(device: Optional[torch.device] = None) -> None:
    """Block until queued kernels on ``device`` finish (no-op on CPU)."""
    device = resolve_device(device) if device is not None else None
    if device is None:
        return
    try:
        if device.type == "npu":
            torch.npu.synchronize(device)
            return
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    except Exception:
        pass


def have_tilelang() -> bool:
    """Whether a TileLang toolchain (Ascend flavour, containing ``T.Kernel(..., is_npu=True)``)
    is importable.  The plain PyPI ``tilelang`` wheel has no Ascend backend, so presence of the
    module alone is not enough — we also require the NPU pass-config keys."""
    try:
        import tilelang  # noqa: F401
    except Exception:
        return False
    try:
        from tilelang import PassConfigKey

        return hasattr(PassConfigKey, "TL_ASCEND_AUTO_SYNC")
    except Exception:
        return False
