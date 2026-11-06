"""Model profiles: the pluggable home for every model-dependent fact in the vLLM layer.

Importing this package registers the profiles it ships (today only :mod:`qwen3_6`).  Consumers -
``tileinfer.vllm.platform``, ``attn/backend.py``, ``worker.py``, ``ops/install.py`` and the
launchers - call :func:`active_profile` / :func:`profile_for` instead of holding their own copy of
a head width, a page size, a KV layout, an op table or a reject list.

Adding a model family is:

1. a new ``tileinfer/vllm/profiles/<name>.py`` that builds a :class:`~.base.ModelProfile` and calls
   :func:`register_profile`;
2. a new ``tileinfer/vllm/ops/families/<op_family>.py`` that declares ``required_names()`` and
   ``bind()`` (see that package);
3. nothing else: no edit to ``platform.py``, ``attn/backend.py``, ``worker.py`` or
   ``ops/install.py``.

The registry is process-global because vLLM asks the platform/backend hooks for answers without
passing a config (the runner calls ``get_kv_cache_shape(blocks, page, heads, dim)``).  The platform
resolves the architecture once in ``check_and_update_config`` and activates the profile through
:func:`set_active_profile`; before that - and on a hand-built config with no architecture - the
first registered profile is the default.
"""

from __future__ import annotations

from .base import (
    PROFILES,
    BlockSizePolicy,
    HybridKind,
    KvShape,
    ModelProfile,
    UnsupportedFeature,
    active_profile,
    default_profile,
    known_profiles,
    profile_by_name,
    profile_for,
    register_profile,
    reset_active_profile,
    set_active_profile,
    unregister_profile,
)

# Importing the shipped profiles is what registers them.  They are cheap (no vLLM), so this is
# safe at package import; a plugin reload can unregister/register through the public helpers.
from . import qwen3_6 as _qwen3_6  # noqa: E402,F401  (import for the register_profile side effect)

__all__ = [
    "PROFILES",
    "BlockSizePolicy",
    "HybridKind",
    "KvShape",
    "ModelProfile",
    "UnsupportedFeature",
    "active_profile",
    "default_profile",
    "known_profiles",
    "profile_by_name",
    "profile_for",
    "register_profile",
    "reset_active_profile",
    "set_active_profile",
    "unregister_profile",
    "qwen3_6",
    "dsv41",
]

from . import dsv41  # noqa: E402,F401  (registers the DeepSeek-V4.1 profile)
