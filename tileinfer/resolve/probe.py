"""What the machine has, without loading a model or touching a device.

The probe is allowed to import packages (it runs at deploy time), but it must not require
the engine to import successfully: a broken vllm-ascend install has to produce a readable
"missing" answer, not a traceback that hides the real 模型 declaration.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import platform
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


def _version(dist: str) -> Optional[str]:
    try:
        return importlib.metadata.version(dist)
    except Exception:
        return None


def _module_present(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _device_type() -> Optional[str]:
    """Read the vendor build's device type without importing the vendor package.

    ``resolve`` is an engine-free layer: importing ``vllm_ascend`` here would pull in torch
    and break the CPU-only property of the decision path (and the layered guard).  The build
    info is a one-line .py file inside the distribution, so read it as data.
    """
    try:
        spec = importlib.util.find_spec("vllm_ascend")
        if spec is None or not spec.origin:
            return None
        build_info = os.path.join(os.path.dirname(spec.origin), "_build_info.py")
        with open(build_info, encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("__device_type__"):
                    return line.split("=", 1)[1].strip().strip("'\"")
    except Exception:
        return None
    return None


@dataclass
class Environment:
    """The facts the decision is allowed to see."""

    vllm: Optional[str] = None
    vllm_ascend: Optional[str] = None
    torch: Optional[str] = None
    python: str = field(default_factory=lambda: platform.python_version())
    device_type: Optional[str] = None
    has_vendor: bool = False
    has_tileinfer: bool = True
    vendor_backend: Optional[str] = None
    notes: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "vllm": self.vllm,
            "vllm_ascend": self.vllm_ascend,
            "torch": self.torch,
            "python": self.python,
            "device_type": self.device_type,
            "has_vendor": self.has_vendor,
            "has_tileinfer": self.has_tileinfer,
            "vendor_backend": self.vendor_backend,
            "notes": dict(self.notes),
        }


def probe() -> Environment:
    """Detect the stack.  Missing pieces are recorded, never raised."""
    env = Environment(
        vllm=_version("vllm"),
        vllm_ascend=_version("vllm_ascend"),
        torch=_version("torch"),
        device_type=_device_type(),
        has_vendor=_module_present("vllm_ascend"),
        has_tileinfer=_module_present("tileinfer"),
    )
    if env.has_vendor:
        # The A5 backend class is stable for the 0.23 line; recording it makes a version
        # drift visible in the plan instead of only in a stack trace.
        if env.device_type in {"A5", "A3", "A2"}:
            env.vendor_backend = "vllm_ascend.attention.attention_v1.AscendAttentionBackend"
        else:
            env.vendor_backend = "vllm_ascend.attention.attention_v1.AscendAttentionBackend"
    env.notes["python_executable"] = sys.executable
    return env
