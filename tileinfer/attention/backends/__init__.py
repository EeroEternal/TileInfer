"""Backend registry (importing this package registers the bundled backends)."""

from .base import (
    AttentionBackend,
    build_plan,
    get_backend,
    list_backends,
    register_backend,
)

__all__ = [
    "AttentionBackend",
    "build_plan",
    "get_backend",
    "list_backends",
    "register_backend",
]
