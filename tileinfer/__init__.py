"""TileInfer — engine-agnostic, serving-oriented kernels for LLM inference on Ascend.

The public surface is deliberately small: :class:`BatchAttention` plus the metadata types an
engine needs in order to describe a batch.
"""

from ._version import __version__
from .attention import BatchAttention
from .metadata import AttentionMode, PageTable, RaggedMetadata
from .plan import AttentionPlan, TileSchedule, WorkspaceManager
from .utils import have_tilelang, is_npu_available, resolve_device

__all__ = [
    "__version__",
    "BatchAttention",
    "AttentionMode",
    "PageTable",
    "RaggedMetadata",
    "AttentionPlan",
    "TileSchedule",
    "WorkspaceManager",
    "have_tilelang",
    "is_npu_available",
    "resolve_device",
]
