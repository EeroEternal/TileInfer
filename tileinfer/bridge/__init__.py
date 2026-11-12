"""L5 - the selective interception layer; the only package allowed to import upstream.

It does exactly two jobs: **choose** an implementation from the plan, and **fill** the small
number of gaps the vendor platform does not cover.  It does not own the platform, the worker,
device memory, the KV pool or graph capture - those stay in vllm-ascend, because rewriting
them is the v2 mistake that cost 2.53x TPOT.

Importing this package is cheap and side-effect free.  ``install.install()`` is the single
entry point called by vLLM's general-plugin loader.
"""

from __future__ import annotations

from .install import install, status, uninstall

__all__ = ["install", "status", "uninstall"]
