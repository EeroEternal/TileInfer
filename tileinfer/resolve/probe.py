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
from typing import Any, Dict, Optional, Tuple


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


def _env_int(name: str, default: int = 1) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass
class Environment:
    """The facts the decision is allowed to see.

    The parallel sizes are facts supplied by the deployment, not read from vLLM here: the
    decision path is engine-free (``test_engine_free_layers_import_in_a_clean_subprocess``
    asserts ``vllm`` is never imported), so ``probe`` takes them from ``TILEINFER_*_SIZE`` or
    from the CLI.  A plan records them; the bridge can cross-check the live config later.
    """

    vllm: Optional[str] = None
    vllm_ascend: Optional[str] = None
    torch: Optional[str] = None
    python: str = field(default_factory=lambda: platform.python_version())
    device_type: Optional[str] = None
    has_vendor: bool = False
    has_tileinfer: bool = True
    vendor_backend: Optional[str] = None
    tp_size: int = 1
    ep_size: int = 1
    dp_size: int = 1
    pp_size: int = 1
    notes: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("tp_size", "ep_size", "dp_size", "pp_size"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")

    def parallel_axes(self) -> Tuple[str, ...]:
        """The active sharding axes, or ``("single",)`` when every size is 1."""
        axes = tuple(
            axis
            for axis, size in (
                ("tp", self.tp_size),
                ("ep", self.ep_size),
                ("dp", self.dp_size),
                ("pp", self.pp_size),
            )
            if size > 1
        )
        return axes or ("single",)

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
            "tp_size": self.tp_size,
            "ep_size": self.ep_size,
            "dp_size": self.dp_size,
            "pp_size": self.pp_size,
            "parallel_axes": list(self.parallel_axes()),
            "notes": dict(self.notes),
        }


def probe(
    tp_size: Optional[int] = None,
    ep_size: Optional[int] = None,
    dp_size: Optional[int] = None,
    pp_size: Optional[int] = None,
) -> Environment:
    """Detect the stack.  Missing pieces are recorded, never raised.

    Parallel sizes come from the explicit arguments first, then ``TILEINFER_TP_SIZE`` /
    ``TILEINFER_EP_SIZE`` / ``TILEINFER_DP_SIZE`` / ``TILEINFER_PP_SIZE``, then 1.
    """
    sizes = {
        "tp_size": tp_size if tp_size is not None else _env_int("TILEINFER_TP_SIZE"),
        "ep_size": ep_size if ep_size is not None else _env_int("TILEINFER_EP_SIZE"),
        "dp_size": dp_size if dp_size is not None else _env_int("TILEINFER_DP_SIZE"),
        "pp_size": pp_size if pp_size is not None else _env_int("TILEINFER_PP_SIZE"),
    }
    env = Environment(
        vllm=_version("vllm"),
        vllm_ascend=_version("vllm_ascend"),
        torch=_version("torch"),
        device_type=_device_type(),
        has_vendor=_module_present("vllm_ascend"),
        has_tileinfer=_module_present("tileinfer"),
        **sizes,
    )
    env.notes["parallel_source"] = "cli/env" if any(v > 1 for v in sizes.values()) else "single"
    if env.has_vendor:
        # The A5 backend class is stable for the 0.23 line; recording it makes a version
        # drift visible in the plan instead of only in a stack trace.
        if env.device_type in {"A5", "A3", "A2"}:
            env.vendor_backend = "vllm_ascend.attention.attention_v1.AscendAttentionBackend"
        else:
            env.vendor_backend = "vllm_ascend.attention.attention_v1.AscendAttentionBackend"
    env.notes["python_executable"] = sys.executable
    return env
